"""Compare actual lowered kernels with their correct full/tail references."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import tilelang.language as T
from tilelang.contrib.nvcc import compile_cuda
from tilelang.engine import lower
from tilelang.env import CUDA_HOME, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

SHAPES = [(n, 1, 1) for n in (1, 2, 3, 4, 7, 8, 13, 16, 17, 24, 31, 32, 48, 64, 128, 256, 1024)] + [
    (8, 8, 1),
    (4, 8, 2),
    (7, 7, 1),
    (3, 3, 5),
]


def kernel(dtype, shape):
    size = math.prod(shape)

    @T.prim_func
    def repeated(X: T.Tensor((128 * size,), dtype), O: T.Tensor((128 * size,), dtype)):
        with T.Kernel(128, threads=shape) as block:
            rank = T.get_thread_binding(0) + shape[0] * (T.get_thread_binding(1) + shape[1] * T.get_thread_binding(2))
            index = block * size + rank
            value = T.alloc_local((1,), dtype)
            value[0] = X[index]
            for _ in T.serial(1024):
                value[0] = T.warp_reduce_sum(value[0]) * T.cast(0.03125, dtype) + T.cast((rank % 2 + 1) * 0.015625, dtype)
            O[index] = value[0]

    return repeated


def body(ptx):
    assert len(re.findall(r"\.visible\s+\.entry\s", ptx)) == 1
    match = re.search(r"\.visible\s+\.entry\s+[^\s(]+\s*\([^)]*\)\s*[^\{]*\{(.*)\}\s*$", ptx, re.S)
    if match is None:
        raise ValueError("expected exactly one generated PTX kernel")
    text = re.sub(r"//[^\n]*", "", match.group(1))
    text = re.sub(r"(?m)^\s*\.loc[^\n]*", "", text)
    return " ".join(text.split())


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("baseline", type=Path)
parser.add_argument("ballot", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rows = []
for dtype in ("float32", "float64", "float16", "bfloat16"):
    for shape in SHAPES:
        size = math.prod(shape)
        artifact = lower(kernel(dtype, shape), target="cuda -arch=sm_120", enable_device_compile=False)
        source = artifact.kernel_source
        full = size % 32 == 0
        calls = re.findall(r"tl::warp_reduce_sum(?:<[^<>]+, ([0-9]+)>)?\(", source)
        assert calls == [str(size) if full else ""], (dtype, shape, calls)
        functions = list(artifact.device_mod.functions.values())
        assert len(functions) == 1
        extents = functions[0].attrs["thread_extent"]
        assert math.prod(int(extents.get(f"threadIdx.{axis}", 1)) for axis in "xyz") == size
        reference = re.sub(r"(tl::warp_reduce_sum)<[^<>]+, [0-9]+>\(", r"\1(", source)
        header = (args.baseline if full else args.ballot).resolve()
        reference = reference.replace("#include <tl_templates/cuda/reduce.h>", f'#include "{header}"')
        assert str(header) in reference
        name = dtype + "-" + "x".join(map(str, shape))
        pair = {}
        for label, code in (("candidate", source), ("reference", reference)):
            (args.output / f"{name}-{label}.cu").write_text(code)
            options = ["-std=c++20", f"-I{TILELANG_TEMPLATE_PATH}", f"-I{CUTLASS_INCLUDE_DIR}", f"-I{CUDA_HOME}/include/cccl"]
            ptx = bytes(compile_cuda(code, arch="sm_120", options=options)).decode()
            (args.output / f"{name}-{label}.ptx").write_text(ptx)
            pair[label] = body(ptx)
        assert pair["candidate"] == pair["reference"], (dtype, shape, "PTX differs")
        if full:
            assert "vote.sync.ballot" not in pair["candidate"]
        else:
            assert "vote.sync.ballot" in pair["candidate"]
        rows.append(
            {
                "dtype": dtype,
                "shape": shape,
                "block_threads": size,
                "reference": "original-full" if full else "correct-tail",
                "normalized_ptx_identical": True,
                "body_sha256": hashlib.sha256(pair["candidate"].encode()).hexdigest(),
            }
        )
        print(name, "PTX_IDENTICAL", flush=True)
(args.output / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
assert len(rows) == 84
