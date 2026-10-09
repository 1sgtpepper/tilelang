#!/usr/bin/env python3
"""Compare TileLang warp-reduction PTX for a baseline and candidate commit."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

BASE_SHA = "194c1b897aa5269e9a89d44c88efdf578a3df620"
TARGETS = ("sm_120", "sm_100a", "sm_100f")
KERNELS = (
    "warp_codegen_f32_sum",
    "warp_codegen_f64_sum",
    "warp_codegen_i32_sum",
    "warp_codegen_f32_min",
    "warp_codegen_f32_max",
)
ENTRY = re.compile(r"(?m)^[ \t]*(?:\.visible[ \t]+)?\.entry[ \t]+([^\s(]+)\s*\(")
SHFL = re.compile(r"\bshfl\.sync\.(bfly|down|idx)\.([A-Za-z0-9_]+)\b")
REDUX = re.compile(r"\bredux\.sync\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b")
OPCODE = re.compile(r"^\s*(?:@!?%[\w$]+\s+)?([A-Za-z_][\w.]*)\s*(?:\s|;|$)")


def ptx_entries(ptx: str) -> dict[str, str]:
    entries = {}
    for match in ENTRY.finditer(ptx):
        start = ptx.find("{", match.end())
        if start < 0:
            continue
        depth = 0
        for end in range(start, len(ptx)):
            depth += (ptx[end] == "{") - (ptx[end] == "}")
            if depth == 0:
                entries[match.group(1)] = ptx[start : end + 1]
                break
        else:
            raise ValueError("unclosed PTX entry {}".format(match.group(1)))
    return entries


def kernel_body(entries: dict[str, str], kernel: str) -> tuple[str, str]:
    matches = [(name, body) for name, body in entries.items() if kernel in name]
    if len(matches) != 1:
        raise ValueError(
            "expected one PTX entry for {}; found {}".format(kernel, [m[0] for m in matches])
        )
    return matches[0]


def normalize(body: str) -> list[str]:
    # Ignore only comments and whitespace when comparing hardware fast paths.
    return [
        re.sub(r"\s+", " ", line.split("//", 1)[0].strip())
        for line in body.splitlines()
        if line.split("//", 1)[0].strip()
    ]


def inspect(body: str) -> dict:
    opcodes, shuffles, redux = collections.Counter(), collections.Counter(), collections.Counter()
    operands = {"bfly": [], "down": [], "idx": []}
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith(".") or line.endswith(":"):
            continue
        match = OPCODE.match(line)
        if match:
            opcodes[match.group(1)] += 1
        match = SHFL.search(line)
        if match:
            mode, width = match.groups()
            args = line[match.end() :].split(";", 1)[0].split(",")
            operands[mode].append(
                {"lane_or_delta": args[2].strip() if len(args) >= 3 else "?", "width": width}
            )
            shuffles[mode] += 1
        match = REDUX.search(line)
        if match:
            redux["{}.{}".format(*match.groups())] += 1
    return {
        "instruction_count": sum(opcodes.values()),
        "opcode_counts": dict(sorted(opcodes.items())),
        "shuffle_counts": dict(sorted(shuffles.items())),
        "shuffle_operands": operands,
        "redux_counts": dict(sorted(redux.items())),
    }


def add_check(checks: list[dict], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def shuffle_count(metrics: dict, mode: str) -> int:
    return metrics["shuffle_counts"].get(mode, 0)


def format_shuffles(data: dict, variant: str) -> str:
    return "/".join(
        str(shuffle_count(data[variant], mode)) for mode in ("bfly", "down", "idx")
    )


def hardware_path_unchanged(data: dict, same_body: bool, redux_ops: tuple[str, ...]) -> bool:
    return same_body and all(
        sum(data[variant]["redux_counts"].get(op, 0) for op in redux_ops) == 1
        and shuffle_count(data[variant], "down") == shuffle_count(data[variant], "idx") == 0
        for variant in ("base", "candidate")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptx_dir", type=Path)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--patch-sha", required=True)
    parser.add_argument("--cutlass-sha", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    for value, label in (
        (args.base_sha, "base SHA"),
        (args.patch_sha, "patch SHA"),
        (args.cutlass_sha, "CUTLASS SHA"),
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("{} must be 40 lowercase hexadecimal digits".format(label))
    if args.base_sha != BASE_SHA:
        raise ValueError("expected reviewed base SHA {}".format(BASE_SHA))

    report = {
        "base_sha": args.base_sha,
        "patch_sha": args.patch_sha,
        "cutlass_sha": args.cutlass_sha,
        "ptx_sha256": {},
        "kernels": {},
        "checks": [],
    }
    bodies = {}
    for target in TARGETS:
        for variant in ("base", "candidate"):
            path = args.ptx_dir / "{}-{}.ptx".format(variant, target)
            report["ptx_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            entries = ptx_entries(path.read_text(encoding="utf-8"))
            for kernel in KERNELS:
                _, body = kernel_body(entries, kernel)
                metrics = inspect(body)
                report["kernels"].setdefault(target, {}).setdefault(kernel, {})[variant] = metrics
                is_hardware_path = kernel == "warp_codegen_i32_sum" or (
                    target in ("sm_100a", "sm_100f")
                    and kernel in ("warp_codegen_f32_min", "warp_codegen_f32_max")
                )
                if is_hardware_path:
                    bodies[(target, variant, kernel)] = normalize(body)

    hardware_cases = [
        (target, "warp_codegen_i32_sum", ("add.s32", "add.u32")) for target in TARGETS
    ] + [
        (target, "warp_codegen_f32_{}".format(op), ("{}.f32".format(op),))
        for target in ("sm_100a", "sm_100f")
        for op in ("min", "max")
    ]
    for target, kernel, redux_ops in hardware_cases:
        data = report["kernels"][target][kernel]
        unchanged = bodies[(target, "base", kernel)] == bodies[(target, "candidate", kernel)]
        add_check(
            report["checks"],
            "{}.{}_hardware_path_unchanged".format(target, kernel),
            hardware_path_unchanged(data, unchanged, redux_ops),
        )

    for kernel, words in (("warp_codegen_f32_sum", 1), ("warp_codegen_f64_sum", 2)):
        data = report["kernels"]["sm_120"][kernel]
        old, new = data["base"]["shuffle_counts"], data["candidate"]["shuffle_counts"]
        expected = [str(offset) for offset in (16, 8, 4, 2, 1) for _ in range(words)]
        offsets = {}
        for variant in ("base", "candidate"):
            offsets[variant] = [item["lane_or_delta"] for item in data[variant]["shuffle_operands"]["bfly"]]
        xor_ok = all(seq == expected for seq in offsets.values())
        xor_ok = xor_ok and offsets["base"] == offsets["candidate"]
        add_check(report["checks"], "sm_120.{}.full_xor_preserved".format(kernel), xor_ok)
        idx = [item["lane_or_delta"] for item in data["candidate"]["shuffle_operands"]["idx"]]
        tail_ok = (
            old.get("down", 0) == old.get("idx", 0) == 0
            and new.get("down", 0) == 5 * words
            and new.get("idx", 0) == words
            and idx == ["0"] * words
            and new.get("bfly", 0) == old.get("bfly", 0) == 5 * words
        )
        add_check(report["checks"], "sm_120.{}.tail_tree_added".format(kernel), tail_ok)

    failures = [item["name"] for item in report["checks"] if not item["passed"]]
    report["passed"], report["failed_checks"] = not failures, failures
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    f32 = report["kernels"]["sm_120"]["warp_codegen_f32_sum"]
    f64 = report["kernels"]["sm_120"]["warp_codegen_f64_sum"]
    summary = "PTX checks {}/{}; SM120 f32 shfl {}/{}; f64 {}/{}; no runtime or timing claim.".format(
        len(report["checks"]) - len(failures),
        len(report["checks"]),
        format_shuffles(f32, "base"),
        format_shuffles(f32, "candidate"),
        format_shuffles(f64, "base"),
        format_shuffles(f64, "candidate"),
    )
    print(summary)
    if failures:
        print("Failed checks: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
