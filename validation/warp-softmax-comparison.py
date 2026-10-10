"""Paired CUDA-graph timing for a CI-only TileLang warp-softmax microkernel."""

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
ROWS, WIDTH, THREADS = 32768, 32, 128
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

    input_path = Path(config["input_path"]).resolve(strict=True)
    input_sha = file_sha(input_path)
    if input_sha != config["input_sha"]:
        raise ValueError("shared input bytes do not match the recorded SHA-256")
    cpu = torch.from_file(str(input_path), shared=False, size=ROWS * WIDTH, dtype=torch.float32).reshape(ROWS, WIDTH)
    x, y = cpu.to("cuda"), torch.empty((ROWS, WIDTH), dtype=torch.float32, device="cuda")
    default_stream = torch.cuda.current_stream()
    reference = torch.softmax(x, dim=-1)
    stream = torch.cuda.Stream()
    stream.wait_stream(default_stream)

    kernel = tilelang.compile(softmax_func(T), out_idx=[], execution_backend="nvrtc")
    source = kernel.get_kernel_source()
    calls = ("tl::warp_reduce_max(", "tl::warp_reduce_sum(")
    if any(source.count(call) != 1 for call in calls):
        raise ValueError("generated source must contain exactly one max and one sum helper call")
    source_bytes = source.encode("utf-8")
    source_path = Path(config["output_dir"]) / f"{variant}-kernel.cu"
    source_path.write_bytes(source_bytes)

    x.record_stream(stream)
    y.record_stream(stream)
    with torch.cuda.stream(stream):
        kernel(x, y, stream=stream.cuda_stream)
    stream.synchronize()
    default_stream.wait_stream(stream)
    if not torch.isfinite(y).all().item():
        raise AssertionError(f"{variant} softmax returned non-finite values")
    torch.testing.assert_close(y, reference, rtol=RTOL, atol=ATOL)

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(GRAPH_LAUNCHES):
            kernel(x, y, stream=stream.cuda_stream)
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
    torch.testing.assert_close(y, reference, rtol=RTOL, atol=ATOL)
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
        "input_sha256": input_sha,
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
        "shape": [ROWS, WIDTH],
        "threads_per_cta": THREADS,
        "rows_per_cta": ROWS_PER_CTA,
        "graph_launches_per_sample": GRAPH_LAUNCHES,
        "captured_kernel_nodes": node_count,
        "graph_replay_regenerated_nan_filled_output": True,
        "warmup_graph_replays": WARMUPS,
        "cache_policy": "repeated graph over reused input/output buffers; no flush or cache-residency claim",
        "max_abs_error_vs_torch_softmax": (y - reference).abs().max().item(),
        "output_path": str(Path(config["output_dir"]) / f"{variant}-output.f32"),
    }
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
            start.record(stream)
            with torch.cuda.stream(stream):
                graph.replay()
            end.record(stream)
            end.synchronize()
            elapsed_ms = start.elapsed_time(end) / GRAPH_LAUNCHES
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise RuntimeError(f"invalid CUDA graph replay time: {elapsed_ms}")
            protocol({"kind": "SAMPLE", "variant": variant, "elapsed_ms": elapsed_ms})
        elif command == "STOP":
            stream.synchronize()
            torch.testing.assert_close(y, reference, rtol=RTOL, atol=ATOL)
            raw = y.detach().cpu().contiguous().numpy().tobytes()
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

    input = torch.randn((ROWS, WIDTH), generator=torch.Generator(device="cpu").manual_seed(INPUT_SEED))
    input_path = output_dir / "input-f32.bin"
    input_bytes = input.numpy().tobytes()
    input_path.write_bytes(input_bytes)
    input_sha = hashlib.sha256(input_bytes).hexdigest()
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
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "input_seed": INPUT_SEED,
        "input_shape": [ROWS, WIDTH],
        "input_dtype": "float32",
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
                "input_path": str(input_path),
                "input_sha": input_sha,
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
        a = torch.from_file(done["baseline"]["output_path"], shared=False, size=ROWS * WIDTH, dtype=torch.float32)
        b = torch.from_file(done["candidate"]["output_path"], shared=False, size=ROWS * WIDTH, dtype=torch.float32)
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
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
                "fixture": "CI-only public-API row-softmax microkernel; no in-tree example or application-wide claim",
                "rows": ROWS,
                "width": WIDTH,
                "threads_per_cta": THREADS,
                "graph_launches_per_sample": GRAPH_LAUNCHES,
                "warmup_graph_replays_per_variant": WARMUPS,
                "paired_samples": PAIRS,
                "cache_mode": "repeated graph over reused input/output buffers; no flush or cache-residency claim",
                "timing": "CUDA-event duration of one graph replay divided by 128 whole kernel launches",
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
                "output_agreement": {"passed": True, "rtol": RTOL, "atol": ATOL},
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
