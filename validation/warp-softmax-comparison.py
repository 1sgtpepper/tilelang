"""Paired CUDA-graph timing for warp softmax or an unchanged Engram consumer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from statistics import median

BASELINE_HEADER_COMMIT = "194c1b897aa5269e9a89d44c88efdf578a3df620"
CONSUMER_COMMIT = "66258df6175d2f630ffecb04c5ab66bff8a2ae6a"
ROWS, WIDTH, THREADS = 32768, 32, 128
ENGRAM_TOKENS, ENGRAM_HC, ENGRAM_HIDDEN = 8001, 4, 4096
ROWS_PER_CTA = THREADS // WIDTH
GRAPH_LAUNCHES = 128
WARMUPS, PAIRS = 10, 21
BOOTSTRAPS, SEED = 10000, 20261010
INPUT_SEED = 20261010
RTOL, ATOL = 1e-5, 1e-6
ROOT = Path("tl_templates/cuda")
HEADER = ROOT / "reduce.h"
STD_HEADER = ROOT / "nvrtc_std.h"
PROTOCOL = "WARP_SOFTMAX "


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha(value: str, label: str, width: int) -> None:
    if re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is None:
        raise ValueError(f"{label} must be {width} lowercase hexadecimal digits")


def root_manifest(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha(path) for path in sorted(root.rglob("*")) if path.is_file()}


def check_roots(args: argparse.Namespace) -> dict[str, str]:
    roots = {name: getattr(args, f"{name}_template").resolve(strict=True) for name in ("baseline", "candidate")}
    manifests = {name: root_manifest(path) for name, path in roots.items()}
    other = {name: {path: digest for path, digest in manifest.items() if path != HEADER.as_posix()} for name, manifest in manifests.items()}
    if other["baseline"] != other["candidate"]:
        raise ValueError("baseline and candidate template trees must differ only at tl_templates/cuda/reduce.h")
    for name in roots:
        if HEADER.as_posix() not in manifests[name] or manifests[name][HEADER.as_posix()] != getattr(args, f"{name}_header_sha"):
            raise ValueError(f"{name} reduce.h SHA-256 does not match the supplied value")
        if STD_HEADER.as_posix() not in manifests[name]:
            raise FileNotFoundError(f"missing NVRTC header: {roots[name] / STD_HEADER}")
    if args.baseline_header_sha == args.candidate_header_sha:
        raise ValueError("baseline and candidate reduce.h hashes are identical")
    tree_json = json.dumps(other["baseline"], sort_keys=True, separators=(",", ":")).encode()
    return {
        "baseline_template": str(roots["baseline"]),
        "candidate_template": str(roots["candidate"]),
        "non_reduce_tree_sha256": hashlib.sha256(tree_json).hexdigest(),
        "non_reduce_tree_files": len(other["baseline"]),
    }


def protocol(message: dict) -> None:
    print(PROTOCOL + json.dumps(message, sort_keys=True), flush=True)


def response(proc: subprocess.Popen, kind: str, variant: str) -> dict:
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith(PROTOCOL):
            value = json.loads(line[len(PROTOCOL) :])
            if value.get("kind") != kind or value.get("variant") != variant:
                raise RuntimeError(f"unexpected {variant} worker response: {value}")
            return value
    raise RuntimeError(f"{variant} worker exited before {kind}; status={proc.poll()}")


def send(proc: subprocess.Popen, command: str, kind: str, variant: str) -> dict:
    if proc.stdin is None:
        raise RuntimeError(f"{variant} worker input is closed")
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    return response(proc, kind, variant)


def softmax_func(T):
    @T.prim_func
    def row_softmax(X: T.Tensor((ROWS, WIDTH), "float32"), O: T.Tensor((ROWS, WIDTH), "float32")):
        with T.Kernel(ROWS // ROWS_PER_CTA, threads=THREADS) as block:
            tid = T.get_thread_binding()
            row = block * ROWS_PER_CTA + tid // WIDTH
            lane = tid % WIDTH
            value = X[row, lane]
            row_max = T.warp_reduce_max(value)
            numerator = T.exp(value - row_max)
            denominator = T.warp_reduce_sum(numerator)
            O[row, lane] = numerator / denominator

    return row_softmax


def worker(config: dict) -> int:
    import importlib.metadata

    import torch
    import tilelang
    import tilelang.language as T
    from cuda.bindings import nvrtc, runtime
    from tilelang.env import CUDA_HOME, TILELANG_TEMPLATE_PATH

    variant = config["variant"]
    template = Path(config["template"]).resolve(strict=True)
    if Path(os.environ.get("TL_TEMPLATE_PATH", "")).resolve(strict=True) != template:
        raise ValueError("worker TL_TEMPLATE_PATH does not select its configured template tree")
    if Path(TILELANG_TEMPLATE_PATH).resolve() != template or os.environ.get("TILELANG_DISABLE_CACHE") != "1":
        raise ValueError("TileLang did not use the requested template root with JIT caching disabled")
    if CUDA_HOME is None or Path(CUDA_HOME).resolve() != Path(os.environ["CUDA_HOME"]).resolve():
        raise ValueError("TileLang CUDA_HOME differs from the pinned runner CUDA_HOME")
    package_path = Path(tilelang.__file__).resolve()
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace and package_path.is_relative_to(Path(workspace).resolve() / "source"):
        raise ValueError(f"TileLang imported from the source checkout: {package_path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("this measurement requires the single-GPU SM120 runner")
    if file_sha(template / HEADER) != config["header_sha"]:
        raise ValueError("worker reduce.h hash mismatch")

    inputs = {}
    for name, spec in config["inputs"].items():
        path = Path(spec["path"]).resolve(strict=True)
        if file_sha(path) != spec["sha256"] or spec["dtype"] not in ("float32", "bfloat16"):
            raise ValueError(f"shared {name} bytes or dtype do not match their provenance")
        dtype = getattr(torch, spec["dtype"])
        count = math.prod(spec["shape"])
        if path.stat().st_size != count * torch.empty((), dtype=dtype).element_size():
            raise ValueError(f"shared {name} has the wrong byte count")
        cpu = torch.from_file(str(path), shared=False, size=count, dtype=dtype).reshape(spec["shape"])
        inputs[name] = cpu.to("cuda")
    del cpu
    x = inputs["x"]
    default_stream = torch.cuda.current_stream()
    if config["workload"] == "engram":
        consumer = Path(config["consumer_root"]).resolve(strict=True)
        for name, digest in config["consumer_files"].items():
            if file_sha(consumer / name) != digest:
                raise ValueError(f"consumer source hash differs: {name}")
        sys.path.insert(0, str(consumer))
        import tile_kernels
        from tile_kernels.config import get_num_sms, get_pdl
        from tile_kernels.engram import engram_gate_fwd
        from tile_kernels.engram.engram_gate_fwd_cuda import get_engram_gate_fwd_kernel_cuda
        from tile_kernels.torch.engram import engram_gate_ref

        if not Path(tile_kernels.__file__).resolve().is_relative_to(consumer / "tile_kernels") or get_pdl():
            raise ValueError("consumer import or default PDL policy differs from the pinned source")
        torch.cuda.reset_peak_memory_stats()
        kv, wh, we = (inputs[name] for name in ("kv", "wh", "we"))
        weight = wh.float() * we.float()
        reference = torch.empty_like(x)
        # Tokens are independent in the upstream reference; bound its temporaries.
        for begin in range(0, ENGRAM_TOKENS, 512):
            end = min(begin + 512, ENGRAM_TOKENS)
            reference[begin:end] = engram_gate_ref(x[begin:end], kv[begin:end], wh, we, 1e-6, 1e-20)
        y, *saved = engram_gate_fwd(x, kv, weight, 1e-20, 1e-6, save_for_backward=False)
        if saved != [None] * 4:
            raise AssertionError("inference wrapper unexpectedly saved backward outputs")
        torch.testing.assert_close(y, reference, rtol=8e-3, atol=8e-3)
        kernel = get_engram_gate_fwd_kernel_cuda(
            ENGRAM_HIDDEN, 1e-20, ENGRAM_HIDDEN**-0.5, get_num_sms(), 1e-6, ENGRAM_HC, False, False, False, use_pdl=False
        )
        if kernel.execution_backend != "tvm_ffi":
            raise ValueError(f"expected the consumer's default FFI backend, got {kernel.execution_backend}")
        kernel_args = (x, kv, weight, None, y, None, None, None, None)
        rtol = atol = 8e-3
        calls = {"tl::warp_reduce_sum(": 3}
        # This is a separate same-source compile inspection, not the timed binary.
        ptx = kernel._get_ptx()
        if ".target sm_120" not in ptx or (config["variant"] == "candidate" and "vote.sync.ballot" not in ptx):
            raise AssertionError("auxiliary PTX does not exercise the expected SM120 fallback")
        Path(config["output_dir"], f"{variant}-auxiliary.ptx").write_text(ptx, encoding="utf-8")
    elif config["workload"] == "softmax":
        y = torch.empty_like(x)
        reference = torch.softmax(x, dim=-1)
        kernel = tilelang.compile(softmax_func(T), out_idx=[], execution_backend="nvrtc")
        kernel_args = (x, y)
        rtol, atol = RTOL, ATOL
        calls = {"tl::warp_reduce_max(": 1, "tl::warp_reduce_sum(": 1}
    else:
        raise ValueError(f"unknown workload: {config['workload']}")
    stream = torch.cuda.Stream()
    stream.wait_stream(default_stream)
    launch_options = {"stream": stream.cuda_stream} if config["workload"] == "softmax" else {}

    source = kernel.get_kernel_source()
    if any(source.count(call) != count for call, count in calls.items()):
        raise ValueError("generated source does not contain the workload's exact direct helper calls")
    source_bytes = source.encode("utf-8")
    source_path = Path(config["output_dir"]) / f"{variant}-kernel.cu"
    source_path.write_bytes(source_bytes)

    for tensor in kernel_args:
        if isinstance(tensor, torch.Tensor):
            tensor.record_stream(stream)
    with torch.cuda.stream(stream):
        kernel(*kernel_args, **launch_options)
    stream.synchronize()
    default_stream.wait_stream(stream)
    if not torch.isfinite(y).all().item():
        raise AssertionError(f"{variant} {config['workload']} returned non-finite values")
    torch.testing.assert_close(y, reference, rtol=rtol, atol=atol)

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(GRAPH_LAUNCHES):
            kernel(*kernel_args, **launch_options)
    stream.synchronize()
    result, _, node_count = runtime.cudaGraphGetNodes(graph.raw_cuda_graph(), numNodes=0)
    if result != runtime.cudaError_t.cudaSuccess or node_count != GRAPH_LAUNCHES:
        raise AssertionError(f"expected {GRAPH_LAUNCHES} captured kernel nodes; got {node_count}, {result}")
    result, nodes, node_count = runtime.cudaGraphGetNodes(graph.raw_cuda_graph(), numNodes=node_count)
    if result != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA graph node query failed: {result}")
    if node_count != GRAPH_LAUNCHES or len(nodes) != GRAPH_LAUNCHES:
        raise AssertionError("CUDA graph node count changed between queries")
    for node in nodes:
        result, kind = runtime.cudaGraphNodeGetType(node)
        if result != runtime.cudaError_t.cudaSuccess or kind != runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            raise AssertionError(f"expected a captured kernel node; got {kind}, {result}")
    graph.instantiate()
    with torch.cuda.stream(stream):
        y.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(y, reference, rtol=rtol, atol=atol)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    nvrtc_result, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
    if nvrtc_result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC version query failed: {nvrtc_result}")
    header_commit = BASELINE_HEADER_COMMIT if variant == "baseline" else config["source_sha"]
    metadata = {
        "variant": variant,
        "template_source_commit": config["source_sha"],
        "wheel_source_commit": config["wheel_source_sha"],
        "reduce_header_commit": header_commit,
        "reduce_header_sha256": config["header_sha"],
        "wheel_sha256": config["wheel_sha"],
        "input_sha256": {name: spec["sha256"] for name, spec in config["inputs"].items()},
        "workload": config["workload"],
        "consumer_commit": CONSUMER_COMMIT if config["workload"] == "engram" else None,
        "execution_backend": kernel.execution_backend,
        "generated_cuda_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "generated_cuda_path": str(source_path),
        "tilelang_path": str(package_path),
        "tilelang_version": importlib.metadata.version("tilelang"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvrtc_version": f"{nvrtc_major}.{nvrtc_minor}",
        "cuda_home": str(Path(CUDA_HOME).resolve()),
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "jit_cache_disabled": True,
        "shape": list(y.shape),
        "output_dtype": "bfloat16" if config["workload"] == "engram" else "float32",
        "threads_per_cta": 32 if config["workload"] == "engram" else THREADS,
        "rtol": rtol,
        "atol": atol,
        "graph_launches_per_sample": GRAPH_LAUNCHES,
        "captured_kernel_nodes": node_count,
        "graph_replay_regenerated_nan_filled_output": True,
        "warmup_graph_replays": WARMUPS,
        "untimed_graph_replays_immediately_before_each_sample": 1,
        "cache_policy": "repeated graph over reused input/output buffers; no flush or cache-residency claim",
        "max_abs_error_vs_upstream_torch_reference": (y.float() - reference.float()).abs().max().item(),
        "output_path": str(Path(config["output_dir"]) / f"{variant}-output.bin"),
    }
    metadata["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
    metadata["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if config["workload"] == "engram":
        # Release reference temporaries' free blocks before the second worker's peak.
        torch.cuda.empty_cache()
    metadata["gpu_free_bytes_at_ready"], metadata["gpu_total_bytes"] = torch.cuda.mem_get_info()
    Path(config["output_dir"], f"{variant}-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol({"kind": "READY", "variant": variant, "metadata": metadata})

    for line in sys.stdin:
        command = line.strip()
        if command == "WARMUP":
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            protocol({"kind": "WARMED", "variant": variant})
        elif command == "RUN":
            with torch.cuda.stream(stream):
                # Warm this worker's graph after the process handoff, before timing.
                graph.replay()
                start.record(stream)
                graph.replay()
                end.record(stream)
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / GRAPH_LAUNCHES
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise RuntimeError(f"invalid CUDA graph replay time: {elapsed_ms}")
            protocol({"kind": "SAMPLE", "variant": variant, "elapsed_ms": elapsed_ms})
        elif command == "STOP":
            stream.synchronize()
            torch.testing.assert_close(y, reference, rtol=rtol, atol=atol)
            raw = y.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            Path(metadata["output_path"]).write_bytes(raw)
            metadata["output_sha256_after_timing"] = hashlib.sha256(raw).hexdigest()
            Path(config["output_dir"], f"{variant}-metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            protocol({"kind": "DONE", "variant": variant, "metadata": metadata})
            return 0
        else:
            raise ValueError(f"unexpected controller command: {command!r}")
    return 0


def numeric(value: str) -> float | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    return float(match.group()) if match else None


def telemetry(query: str, names: tuple[str, ...]) -> dict:
    result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], check=True, capture_output=True, text=True)
    fields = next(csv.reader([result.stdout.strip()], skipinitialspace=True))
    if len(fields) != len(names):
        raise ValueError(f"unexpected nvidia-smi output: {result.stdout!r}")
    return {"raw": result.stdout.strip(), **dict(zip(names, fields)), **{"numeric": dict(zip(names, map(numeric, fields)))}}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def compare(samples: list[float]) -> dict:
    logs = [math.log(value) for value in samples]
    rng = random.Random(SEED)
    boot = [math.exp(median(rng.choice(logs) for _ in logs)) for _ in range(BOOTSTRAPS)]
    ratio = math.exp(median(logs))
    low, high = percentile(boot, 0.025), percentile(boot, 0.975)
    low_pct, high_pct = (low - 1) * 100, (high - 1) * 100
    change = (ratio - 1) * 100
    if low_pct > 0 or high_pct < 0:
        assessment = "MATERIAL_CHANGE" if abs(change) >= 3 else "CHANGE_BELOW_3_PERCENT_THRESHOLD"
    else:
        assessment = "INCONCLUSIVE_FOR_3_PERCENT_EFFECT"
    return {
        "candidate_over_baseline_median": ratio,
        "candidate_over_baseline_bootstrap_95_ci": [low, high],
        "candidate_change_percent_median": change,
        "candidate_change_percent_bootstrap_95_ci": [low_pct, high_pct],
        "bootstrap_samples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "practical_threshold_percent": 3,
        "statistical_assessment": assessment,
    }


def controller(args: argparse.Namespace) -> int:
    import torch

    for value, label, width in (
        (args.source_sha, "source commit", 40),
        (args.wheel_source_sha, "wheel source commit", 40),
        (args.baseline_header_sha, "baseline header SHA-256", 64),
        (args.candidate_header_sha, "candidate header SHA-256", 64),
        (args.wheel_sha, "wheel SHA-256", 64),
    ):
        sha(value, label, width)
    if os.environ.get("TILELANG_DISABLE_CACHE") != "1":
        raise ValueError("set TILELANG_DISABLE_CACHE=1")
    roots = check_roots(args)
    wheel_path = args.wheel_path.resolve(strict=True)
    if file_sha(wheel_path) != args.wheel_sha:
        raise ValueError("installed wheel artifact hash differs from the supplied provenance")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite benchmark evidence in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    consumer_files = {}
    if args.workload == "engram":
        if args.consumer_root is None:
            raise ValueError("Engram requires the pinned consumer checkout")
        consumer = args.consumer_root.resolve(strict=True)
        actual = subprocess.run(
            ["git", "-C", str(consumer), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(["git", "-C", str(consumer), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
        if actual != CONSUMER_COMMIT or status:
            raise ValueError("consumer checkout is not the clean pinned source")
        files = [consumer / "LICENSE", consumer / "pyproject.toml", *sorted((consumer / "tile_kernels").rglob("*.py"))]
        consumer_files = {path.relative_to(consumer).as_posix(): file_sha(path) for path in files}
        shapes = {
            "x": (ENGRAM_TOKENS, ENGRAM_HC, ENGRAM_HIDDEN),
            "kv": (ENGRAM_TOKENS, ENGRAM_HC + 1, ENGRAM_HIDDEN),
            "wh": (ENGRAM_HC, ENGRAM_HIDDEN),
            "we": (ENGRAM_HC, ENGRAM_HIDDEN),
        }
        dtype_name = "bfloat16"
    elif args.consumer_root is not None:
        raise ValueError("softmax does not use a consumer checkout")
    else:
        shapes, dtype_name = {"x": (ROWS, WIDTH)}, "float32"
    generator = torch.Generator(device="cpu").manual_seed(INPUT_SEED)
    inputs = {}
    for name, shape in shapes.items():
        tensor = torch.randn(shape, generator=generator, dtype=getattr(torch, dtype_name))
        input_bytes = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        path = output_dir / f"input-{name}.bin"
        path.write_bytes(input_bytes)
        inputs[name] = {"path": str(path), "sha256": hashlib.sha256(input_bytes).hexdigest(), "shape": list(shape), "dtype": dtype_name}
    del tensor, input_bytes
    device_query = telemetry(
        "name,compute_cap,memory.total,driver_version",
        ("gpu_name", "compute_cap", "memory_total", "driver_version"),
    )
    before_workers = telemetry(
        "clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw",
        ("clocks_sm", "clocks_mem", "utilization_gpu", "temperature_gpu", "power_draw"),
    )
    provenance = {
        **roots,
        "template_source_commit": args.source_sha,
        "wheel_source_commit": args.wheel_source_sha,
        "baseline_reduce_header_commit": BASELINE_HEADER_COMMIT,
        "candidate_reduce_header_commit": args.source_sha,
        "baseline_reduce_header_sha256": args.baseline_header_sha,
        "candidate_reduce_header_sha256": args.candidate_header_sha,
        "wheel_path": str(wheel_path),
        "wheel_sha256": args.wheel_sha,
        "inputs": inputs,
        "input_seed": INPUT_SEED,
        "workload": args.workload,
        "consumer_commit": CONSUMER_COMMIT if args.workload == "engram" else None,
        "consumer_files": consumer_files,
        "gpu": device_query,
        "gpu_state_before_workers": before_workers,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    workers, ready, done = {}, {}, {}
    try:
        for variant in ("baseline", "candidate"):
            config = {
                "variant": variant,
                "template": str(getattr(args, f"{variant}_template").resolve()),
                "header_sha": getattr(args, f"{variant}_header_sha"),
                "source_sha": args.source_sha,
                "wheel_source_sha": args.wheel_source_sha,
                "wheel_sha": args.wheel_sha,
                "inputs": inputs,
                "workload": args.workload,
                "consumer_root": str(args.consumer_root.resolve()) if args.consumer_root else None,
                "consumer_files": consumer_files,
                "output_dir": str(output_dir),
            }
            config_path = output_dir / f"{variant}-worker.json"
            config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
            env = os.environ.copy()
            env["TL_TEMPLATE_PATH"] = config["template"]
            proc = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-config", str(config_path)],
                cwd=output_dir,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            workers[variant] = proc
            ready[variant] = response(proc, "READY", variant)["metadata"]
            print(json.dumps({"event": "worker_ready", "variant": variant, **ready[variant]}, sort_keys=True), flush=True)

        for key in (
            "tilelang_path",
            "tilelang_version",
            "torch_version",
            "torch_cuda_version",
            "nvrtc_version",
            "gpu_name",
            "compute_capability",
            "generated_cuda_sha256",
            "input_sha256",
            "workload",
            "consumer_commit",
            "execution_backend",
            "shape",
            "output_dtype",
        ):
            if ready["baseline"][key] != ready["candidate"][key]:
                raise ValueError(f"baseline and candidate runtime/source differ in {key}")
        if ready["baseline"]["gpu_name"] != device_query["gpu_name"]:
            raise ValueError("Torch and nvidia-smi report different GPUs")

        for i in range(WARMUPS):
            for variant in ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline"):
                send(workers[variant], "WARMUP", "WARMED", variant)

        csv_path = output_dir / "paired-timings.csv"
        times = {"baseline": [], "candidate": []}
        pair_telemetry = []
        columns = (
            "pair",
            "order",
            "baseline_ms_per_kernel",
            "candidate_ms_per_kernel",
            "candidate_over_baseline",
            "gpu_before_json",
            "gpu_after_json",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as raw:
            writer = csv.DictWriter(raw, fieldnames=columns)
            writer.writeheader()
            for i in range(PAIRS):
                order = ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
                before = telemetry(
                    "clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw",
                    ("clocks_sm", "clocks_mem", "utilization_gpu", "temperature_gpu", "power_draw"),
                )
                pair = {}
                for variant in order:
                    pair[variant] = float(send(workers[variant], "RUN", "SAMPLE", variant)["elapsed_ms"])
                    times[variant].append(pair[variant])
                after = telemetry(
                    "clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw",
                    ("clocks_sm", "clocks_mem", "utilization_gpu", "temperature_gpu", "power_draw"),
                )
                ratio = pair["candidate"] / pair["baseline"]
                pair_telemetry.append({"pair": i + 1, "before": before, "after": after})
                writer.writerow(
                    {
                        "pair": i + 1,
                        "order": ">".join(order),
                        "baseline_ms_per_kernel": pair["baseline"],
                        "candidate_ms_per_kernel": pair["candidate"],
                        "candidate_over_baseline": ratio,
                        "gpu_before_json": json.dumps(before, sort_keys=True),
                        "gpu_after_json": json.dumps(after, sort_keys=True),
                    }
                )
                raw.flush()

        for variant in ("baseline", "candidate"):
            done[variant] = send(workers[variant], "STOP", "DONE", variant)["metadata"]
            status = workers[variant].wait(timeout=10)
            if status != 0:
                raise RuntimeError(f"{variant} worker exited with status {status}")
        count = math.prod(ready["baseline"]["shape"])
        dtype = getattr(torch, ready["baseline"]["output_dtype"])
        a = torch.from_file(done["baseline"]["output_path"], shared=False, size=count, dtype=dtype)
        b = torch.from_file(done["candidate"]["output_path"], shared=False, size=count, dtype=dtype)
        torch.testing.assert_close(a, b, rtol=ready["baseline"]["rtol"], atol=ready["baseline"]["atol"])
        output_bytes_equal = done["baseline"]["output_sha256_after_timing"] == done["candidate"]["output_sha256_after_timing"]
        env_flags = []
        start_util = before_workers["numeric"]["utilization_gpu"]
        if start_util is None or start_util >= 20:
            env_flags.append("GPU_BUSY_OR_UTILIZATION_UNAVAILABLE_BEFORE_WORKERS")
        transitions = []
        for item in pair_telemetry:
            for field in ("clocks_sm", "clocks_mem"):
                old, new = item["before"]["numeric"][field], item["after"]["numeric"][field]
                if old is None or new is None or (old and abs(new - old) / old >= 0.20):
                    transitions.append({"pair": item["pair"], "field": field, "before": old, "after": new})
            old, new = item["before"]["numeric"]["temperature_gpu"], item["after"]["numeric"]["temperature_gpu"]
            if old is None or new is None or abs(new - old) >= 5:
                transitions.append({"pair": item["pair"], "field": "temperature_gpu", "before": old, "after": new})
        if transitions:
            env_flags.append("CLOCK_OR_TEMPERATURE_TRANSITION_OR_TELEMETRY_GAP")
        report = compare([c / b for b, c in zip(times["baseline"], times["candidate"])])
        report["assessment"] = "INCONCLUSIVE_ENVIRONMENT" if env_flags else report["statistical_assessment"]
        report.update(
            {
                "fixture": "unchanged upstream TileKernels Engram CUDA kernel; no model-wide claim"
                if args.workload == "engram"
                else "CI-only public-API row-softmax microkernel; no in-tree example or application-wide claim",
                "output_shape": ready["baseline"]["shape"],
                "threads_per_cta": ready["baseline"]["threads_per_cta"],
                "graph_launches_per_sample": GRAPH_LAUNCHES,
                "warmup_graph_replays_per_variant": WARMUPS,
                "untimed_graph_replays_immediately_before_each_sample": 1,
                "paired_samples": PAIRS,
                "cache_mode": "repeated graph over reused input/output buffers; no flush or cache-residency claim",
                "timing": "one immediate untimed replay, then CUDA-event duration of one replay divided by 128 whole kernel launches",
                "times_ms_per_kernel": times,
                "provenance": provenance,
                "variants": done,
                "environment_flags": env_flags,
                "clock_or_temperature_transitions": transitions,
                "telemetry_thresholds": {
                    "initial_gpu_utilization_percent": 20,
                    "within_pair_clock_change_fraction": 0.20,
                    "within_pair_temperature_change_celsius": 5,
                },
                "samples_discarded": 0,
                "clocks_modified": False,
                "output_agreement": {
                    "passed": True,
                    "bytes_equal": output_bytes_equal,
                    "rtol": ready["baseline"]["rtol"],
                    "atol": ready["baseline"]["atol"],
                },
            }
        )
        (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "summary", **report}, sort_keys=True), flush=True)
        return 0
    finally:
        for proc in workers.values():
            if proc.poll() is None:
                try:
                    if proc.stdin is not None:
                        proc.stdin.write("STOP\n")
                        proc.stdin.flush()
                    proc.wait(timeout=10)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    proc.kill()
                    proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=("softmax", "engram"), default="softmax")
    parser.add_argument("--consumer-root", type=Path)
    parser.add_argument("--baseline-template", type=Path)
    parser.add_argument("--candidate-template", type=Path)
    parser.add_argument("--baseline-header-sha")
    parser.add_argument("--candidate-header-sha")
    parser.add_argument("--source-sha")
    parser.add_argument("--wheel-source-sha")
    parser.add_argument("--wheel-path", type=Path)
    parser.add_argument("--wheel-sha")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.worker_config:
        return worker(json.loads(args.worker_config.read_text(encoding="utf-8")))
    required = (
        args.baseline_template,
        args.candidate_template,
        args.baseline_header_sha,
        args.candidate_header_sha,
        args.source_sha,
        args.wheel_source_sha,
        args.wheel_path,
        args.wheel_sha,
        args.output_dir,
    )
    if any(value is None for value in required):
        parser.error("controller mode requires template roots, header SHA-256 values, source/wheel provenance, and output directory")
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
