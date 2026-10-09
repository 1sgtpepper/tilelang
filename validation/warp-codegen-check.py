#!/usr/bin/env python3
"""Compare exact PTX entry bodies for TileLang warp-reduction codegen."""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import re
import sys
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
ENTRY_RE = re.compile(r"(?m)^[ \t]*(?:\.visible[ \t]+)?\.entry[ \t]+([^\s(]+)\s*\(")
SHFL_RE = re.compile(r"\bshfl\.sync\.(bfly|down|idx)\.([A-Za-z0-9_]+)\b")
REDUX_RE = re.compile(r"\bredux\.sync\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b")
PREDICATED_OPCODE_RE = re.compile(
    r"^\s*(?:@!?%[A-Za-z0-9_$]+\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*(?:\s|;|$)"
)
GEOMETRY_OPCODE_PREFIXES = (
    "add",
    "and",
    "bra",
    "div",
    "mad",
    "mul",
    "rem",
    "selp",
    "setp",
    "shl",
    "shr",
    "sub",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptx_dir", type=Path)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--patch-sha", required=True)
    parser.add_argument("--cutlass-sha", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-notes", type=Path, required=True)
    return parser.parse_args()


def require_sha(value: str, name: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("{} must be exactly 40 lowercase hexadecimal digits".format(name))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_entries(ptx: str) -> list[tuple[str, str]]:
    entries = []
    for match in ENTRY_RE.finditer(ptx):
        body_start = ptx.find("{", match.end())
        if body_start < 0:
            continue
        depth = 0
        body_end = None
        for index in range(body_start, len(ptx)):
            if ptx[index] == "{":
                depth += 1
            elif ptx[index] == "}":
                depth -= 1
                if depth == 0:
                    body_end = index + 1
                    break
        if body_end is None:
            raise ValueError("unclosed PTX entry body for {}".format(match.group(1)))
        entries.append((match.group(1), ptx[body_start:body_end]))
    return entries


def find_kernel_body(entries: list[tuple[str, str]], kernel: str) -> tuple[str, str]:
    matches = [(name, body) for name, body in entries if kernel in name]
    if len(matches) != 1:
        names = [name for name, _ in entries]
        raise ValueError(
            "expected one PTX entry containing {!r}; got {} from {}".format(
                kernel, len(matches), names
            )
        )
    return matches[0]


def clean_line(line: str) -> str:
    return line.split("//", 1)[0].strip()


def canonical_body(body: str) -> list[str]:
    """Normalize only comments/whitespace; retain registers, labels, and op order."""
    result = []
    for line in body.splitlines():
        line = clean_line(line)
        if line:
            result.append(re.sub(r"\s+", " ", line))
    return result


def instruction_data(body: str) -> dict:
    opcodes = collections.Counter()
    shuffles = collections.Counter()
    shuffle_lanes = {"bfly": [], "down": [], "idx": []}
    redux = collections.Counter()

    for raw_line in body.splitlines():
        line = clean_line(raw_line)
        if not line or line.startswith(".") or line.endswith(":"):
            continue

        opcode_match = PREDICATED_OPCODE_RE.match(line)
        if opcode_match:
            opcode = opcode_match.group(1)
            opcodes[opcode] += 1

        shuffle_match = SHFL_RE.search(line)
        if shuffle_match:
            mode, width = shuffle_match.groups()
            shuffles[mode] += 1
            operands = line[shuffle_match.end() :].split(";", 1)[0]
            operands = [part.strip() for part in operands.split(",")]
            # PTX SHFL operands are destination, input, lane/delta, clamp, mask.
            lane = operands[2] if len(operands) >= 3 else "<unparsed>"
            shuffle_lanes[mode].append({"lane_or_delta": lane, "width": width})

        redux_match = REDUX_RE.search(line)
        if redux_match:
            redux["{}.{}".format(*redux_match.groups())] += 1

    geometry_like = collections.Counter()
    for opcode, count in opcodes.items():
        if opcode.split(".", 1)[0] in GEOMETRY_OPCODE_PREFIXES:
            geometry_like[opcode] = count

    return {
        "instruction_count": sum(opcodes.values()),
        "opcode_counts": dict(sorted(opcodes.items())),
        "shuffle_counts": dict(sorted(shuffles.items())),
        "shuffle_lanes": shuffle_lanes,
        "redux_counts": dict(sorted(redux.items())),
        # Includes caller index/address math and reducer arithmetic. It is an
        # opcode inventory, not a dataflow attribution of geometry-only work.
        "geometry_like_opcode_counts": dict(sorted(geometry_like.items())),
    }


def body_diff(old: list[str], new: list[str], context: int = 2) -> list[str]:
    return list(
        difflib.unified_diff(
            old,
            new,
            fromfile="base-entry",
            tofile="candidate-entry",
            lineterm="",
            n=context,
        )
    )


def check(report: dict, name: str, condition: bool, detail: str) -> None:
    report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})


def mode_count(metrics: dict, mode: str) -> int:
    return metrics["shuffle_counts"].get(mode, 0)


def assert_full_warp_offsets(metrics: dict, words_per_value: int) -> bool:
    offsets = [int(item["lane_or_delta"]) for item in metrics["shuffle_lanes"]["bfly"]]
    expected = [offset for offset in (16, 8, 4, 2, 1) for _ in range(words_per_value)]
    return offsets == expected


def analyze(args: argparse.Namespace) -> tuple[dict, str]:
    require_sha(args.base_sha, "base_sha")
    require_sha(args.patch_sha, "patch_sha")
    require_sha(args.cutlass_sha, "cutlass_sha")
    if args.base_sha != BASE_SHA:
        raise ValueError("base SHA must remain the reviewed TileLang base {}".format(BASE_SHA))

    report = {
        "schema_version": 1,
        "scope": {
            "base_sha": args.base_sha,
            "patch_sha": args.patch_sha,
            "cutlass_sha": args.cutlass_sha,
            "targets": list(TARGETS),
            "device_execution": False,
            "performance_claim": None,
            "ptx_probe": "warp-codegen-check.cu; compiler may CSE the same flattened caller index used by the reducer",
        },
        "ptx_files": {},
        "kernels": {},
        "checks": [],
        "status": "pending",
    }
    for filename, key in (
        ("nvcc-version.txt", "nvcc_version"),
        ("gcc-version.txt", "host_compiler_version"),
    ):
        version_file = args.ptx_dir / filename
        if version_file.is_file():
            report.setdefault("toolchain", {})[key] = version_file.read_text(encoding="utf-8").strip()
    entries_by_variant_target = {}
    metrics_by_variant_target_kernel = {}
    bodies_by_variant_target_kernel = {}

    for target in TARGETS:
        entries_by_variant_target[target] = {}
        for variant in ("base", "candidate"):
            path = args.ptx_dir / "{}-{}.ptx".format(variant, target)
            if not path.is_file():
                raise FileNotFoundError("missing PTX input: {}".format(path))
            ptx = path.read_text(encoding="utf-8")
            entries_by_variant_target[target][variant] = extract_entries(ptx)
            report["ptx_files"][path.name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}

            for kernel in KERNELS:
                symbol, body = find_kernel_body(entries_by_variant_target[target][variant], kernel)
                key = (target, variant, kernel)
                bodies_by_variant_target_kernel[key] = canonical_body(body)
                metrics_by_variant_target_kernel[key] = instruction_data(body)
                report["kernels"].setdefault(target, {}).setdefault(kernel, {})[variant] = {
                    "entry_symbol": symbol,
                    **metrics_by_variant_target_kernel[key],
                }

    for target in TARGETS:
        for kernel in KERNELS:
            old_metrics = metrics_by_variant_target_kernel[(target, "base", kernel)]
            new_metrics = metrics_by_variant_target_kernel[(target, "candidate", kernel)]
            same_body = (
                bodies_by_variant_target_kernel[(target, "base", kernel)]
                == bodies_by_variant_target_kernel[(target, "candidate", kernel)]
            )
            old_ops = old_metrics["opcode_counts"]
            new_ops = new_metrics["opcode_counts"]
            delta = {
                opcode: new_ops.get(opcode, 0) - old_ops.get(opcode, 0)
                for opcode in sorted(set(old_ops) | set(new_ops))
                if new_ops.get(opcode, 0) != old_ops.get(opcode, 0)
            }
            report["kernels"][target][kernel]["body_identical_base_candidate"] = same_body
            report["kernels"][target][kernel]["candidate_minus_base_opcode_counts"] = delta
            if not same_body:
                diff = body_diff(
                    bodies_by_variant_target_kernel[(target, "base", kernel)],
                    bodies_by_variant_target_kernel[(target, "candidate", kernel)],
                )
                report["kernels"][target][kernel]["body_diff_excerpt"] = diff[:160]

    for target in TARGETS:
        i32_kernel = "warp_codegen_i32_sum"
        base_i32 = metrics_by_variant_target_kernel[(target, "base", i32_kernel)]
        candidate_i32 = metrics_by_variant_target_kernel[(target, "candidate", i32_kernel)]
        i32_same = bodies_by_variant_target_kernel[(target, "base", i32_kernel)] == bodies_by_variant_target_kernel[
            (target, "candidate", i32_kernel)
        ]
        check(
            report,
            "{}.i32_sum_unchanged".format(target),
            i32_same,
            "Base and candidate normalized PTX entry bodies must match exactly; only comments/whitespace are ignored.",
        )
        check(
            report,
            "{}.i32_sum_uses_native_redux".format(target),
            sum(
                count
                for key, count in base_i32["redux_counts"].items()
                if key in ("add.s32", "add.u32")
            )
            == 1
            and sum(
                count
                for key, count in candidate_i32["redux_counts"].items()
                if key in ("add.s32", "add.u32")
            )
            == 1,
            "Both variants must lower int32 sum to one redux.sync.add.s32 or redux.sync.add.u32.",
        )
        check(
            report,
            "{}.i32_sum_has_no_tail_shuffle".format(target),
            mode_count(candidate_i32, "down") == 0 and mode_count(candidate_i32, "idx") == 0,
            "Native int32 redux must bypass the new tail shuffle path.",
        )

    for target in ("sm_100a", "sm_100f"):
        for kernel, op in (("warp_codegen_f32_min", "min"), ("warp_codegen_f32_max", "max")):
            base_metrics = metrics_by_variant_target_kernel[(target, "base", kernel)]
            candidate_metrics = metrics_by_variant_target_kernel[(target, "candidate", kernel)]
            same = bodies_by_variant_target_kernel[(target, "base", kernel)] == bodies_by_variant_target_kernel[
                (target, "candidate", kernel)
            ]
            check(
                report,
                "{}.{}.feature_path_unchanged".format(target, op),
                same,
                "Base and candidate PTX entry bodies must match exactly for the SM100 float redux path.",
            )
            check(
                report,
                "{}.{}.uses_native_redux".format(target, op),
                candidate_metrics["redux_counts"].get("{}.f32".format(op), 0) == 1,
                "Candidate must contain one redux.sync.{}.f32.".format(op),
            )
            check(
                report,
                "{}.{}.has_no_tail_shuffle".format(target, op),
                mode_count(candidate_metrics, "down") == 0 and mode_count(candidate_metrics, "idx") == 0,
                "Feature-gated min/max return must bypass the new tail shuffle path.",
            )

    for kernel, words_per_value in (("warp_codegen_f32_sum", 1), ("warp_codegen_f64_sum", 2)):
        base_metrics = metrics_by_variant_target_kernel[("sm_120", "base", kernel)]
        candidate_metrics = metrics_by_variant_target_kernel[("sm_120", "candidate", kernel)]
        expected_shuffle_count = 5 * words_per_value
        check(
            report,
            "sm_120.{}.full_warp_xor_tree_preserved".format(kernel),
            mode_count(base_metrics, "bfly") == expected_shuffle_count
            and mode_count(candidate_metrics, "bfly") == expected_shuffle_count
            and assert_full_warp_offsets(base_metrics, words_per_value)
            and assert_full_warp_offsets(candidate_metrics, words_per_value),
            "Base and candidate must retain the XOR offsets 16, 8, 4, 2, 1 ({} PTX instructions for this value width).".format(
                expected_shuffle_count
            ),
        )
        check(
            report,
            "sm_120.{}.tail_tree_added".format(kernel),
            mode_count(base_metrics, "down") == 0
            and mode_count(base_metrics, "idx") == 0
            and mode_count(candidate_metrics, "down") == 5 * words_per_value
            and mode_count(candidate_metrics, "idx") == words_per_value,
            "Candidate must add five logical down steps and one lane-zero broadcast ({} PTX SHFL instructions for this value width).".format(
                6 * words_per_value
            ),
        )

    failed = [item for item in report["checks"] if not item["passed"]]
    report["status"] = "failed" if failed else "passed"
    report["summary"] = {
        "passed_checks": len(report["checks"]) - len(failed),
        "failed_checks": len(failed),
    }
    notes = render_notes(report)
    return report, notes


def render_notes(report: dict) -> str:
    lines = [
        "# Warp reduction PTX comparison",
        "",
        "Status: **{}** ({} checks passed, {} failed).".format(
            report["status"],
            report["summary"]["passed_checks"],
            report["summary"]["failed_checks"],
        ),
        "",
        "This is compile-only evidence for base `{}` and candidate `{}` using the same NVCC invocation. No kernel was launched, and the instruction counts are not throughput or speedup measurements.".format(
            report["scope"]["base_sha"], report["scope"]["patch_sha"]
        ),
        "The probe uses the same flattened caller index as the reducer; NVCC may common-subexpression-eliminate some geometry arithmetic. Opcode inventories are whole-entry counts, not a dataflow attribution of each instruction to geometry.",
        "",
        "| Target | Kernel | BFLY base→candidate | DOWN base→candidate | IDX base→candidate | Redux base→candidate | PTX instructions base→candidate | Body identical |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for target in TARGETS:
        for kernel in KERNELS:
            item = report["kernels"][target][kernel]
            redux_union = sorted(
                set(item["base"]["redux_counts"]) | set(item["candidate"]["redux_counts"])
            )
            redux_text = ", ".join(
                "{} {}/{}".format(
                    key,
                    item["base"]["redux_counts"].get(key, 0),
                    item["candidate"]["redux_counts"].get(key, 0),
                )
                for key in redux_union
            ) or "—"
            base_metrics = item["base"]
            candidate_metrics = item["candidate"]
            lines.append(
                "| {} | {} | {}→{} | {}→{} | {}→{} | {} | {}→{} | {} |".format(
                    target,
                    kernel,
                    mode_count(base_metrics, "bfly"),
                    mode_count(candidate_metrics, "bfly"),
                    mode_count(base_metrics, "down"),
                    mode_count(candidate_metrics, "down"),
                    mode_count(base_metrics, "idx"),
                    mode_count(candidate_metrics, "idx"),
                    redux_text,
                    base_metrics["instruction_count"],
                    candidate_metrics["instruction_count"],
                    "yes" if item["body_identical_base_candidate"] else "no",
                )
            )
    failed = [item for item in report["checks"] if not item["passed"]]
    lines.extend(["", "## Targeted checks", ""])
    lines.append("{} of {} assertions passed.".format(len(report["checks"]) - len(failed), len(report["checks"])))
    for item in failed:
        lines.append("- FAIL: {} — {}".format(item["name"], item["detail"]))
    lines.extend(
        [
            "",
            "## Geometry-like opcode deltas",
            "",
            "These deltas include caller flat-index and reducer arithmetic; inspect the six PTX files for attribution.",
            "",
        ]
    )
    for target in TARGETS:
        for kernel in ("warp_codegen_f32_sum", "warp_codegen_f64_sum"):
            delta = report["kernels"][target][kernel]["candidate_minus_base_opcode_counts"]
            subset = {
                opcode: count
                for opcode, count in delta.items()
                if opcode.split(".", 1)[0] in GEOMETRY_OPCODE_PREFIXES
                or opcode.startswith("shfl.")
            }
            lines.append("- `{}/{}`: `{}`".format(target, kernel, json.dumps(subset, sort_keys=True) if subset else "no opcode-count delta"))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report, notes = analyze(args)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_notes.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_notes.write_text(notes, encoding="utf-8")
    print(notes)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
