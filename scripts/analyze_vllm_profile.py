#!/usr/bin/env python3
"""Summarize matched vLLM PyTorch profiler traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate(events: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != category:
            continue
        duration = float(event.get("dur", 0.0) or 0.0)
        name = str(event.get("name", "<unnamed>"))
        stats[name]["count"] += 1
        stats[name]["total_us"] += duration
    rows = []
    for name, item in stats.items():
        count = int(item["count"])
        total_us = item["total_us"]
        rows.append(
            {
                "name": name,
                "count": count,
                "total_ms": total_us / 1000.0,
                "avg_us": total_us / count if count else 0.0,
            }
        )
    return sorted(rows, key=lambda row: row["total_ms"], reverse=True)


def family(name: str) -> str:
    lower = name.lower()
    if "awq" in lower or "marlin" in lower or "quant" in lower or "int4" in lower:
        return "quantized_linear"
    if "gemm" in lower or "gemv" in lower or "cutlass" in lower or "cublas" in lower:
        return "dense_linear"
    if "flash" in lower or "attention" in lower:
        return "attention"
    if "rms" in lower or "norm" in lower:
        return "normalization"
    if "copy" in lower or "memcpy" in lower:
        return "memory_copy"
    if "elementwise" in lower or "triton_poi" in lower or "triton_red" in lower:
        return "elementwise"
    if "topk" in lower or "sampling" in lower or "multinomial" in lower:
        return "sampling"
    return "other"


def summarize_power(path: Path) -> dict[str, Any]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({key: float(value) for key, value in row.items()})
            except (TypeError, ValueError):
                continue
    if not rows:
        return {"samples": 0}
    power = [row["power_w"] for row in rows]
    util = [row["gpu_utilization_percent"] for row in rows]
    memory = [row["memory_used_mib"] for row in rows]
    return {
        "samples": len(rows),
        "avg_power_w": statistics.fmean(power),
        "peak_power_w": max(power),
        "avg_gpu_utilization_percent": statistics.fmean(util),
        "peak_gpu_utilization_percent": max(util),
        "avg_memory_used_mib": statistics.fmean(memory),
        "peak_memory_used_mib": max(memory),
    }


def analyze_variant(variant_dir: Path) -> dict[str, Any]:
    trace_files = list((variant_dir / "traces").glob("*.json*"))
    if len(trace_files) != 1:
        raise RuntimeError(f"expected one JSON trace under {variant_dir}, found {trace_files}")
    trace = load_trace(trace_files[0])
    events = trace.get("traceEvents", [])
    categories = {}
    details = {}
    for category in ("kernel", "cpu_op", "cuda_runtime", "gpu_memcpy", "gpu_memset"):
        rows = aggregate(events, category)
        details[category] = rows
        categories[category] = {
            "event_count": sum(row["count"] for row in rows),
            "total_ms": sum(row["total_ms"] for row in rows),
        }
    families: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
    for row in details["kernel"]:
        item = families[family(row["name"])]
        item["count"] += row["count"]
        item["total_ms"] += row["total_ms"]
    metadata = json.loads((variant_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "variant": variant_dir.name,
        "trace_file": str(trace_files[0]),
        "profiled_request": metadata.get("profiled_requests", [{}])[0],
        "gpu_after_server_ready": metadata.get("gpu_after_server_ready"),
        "power": summarize_power(variant_dir / "power_samples.csv"),
        "categories": categories,
        "kernel_families": dict(sorted(families.items(), key=lambda pair: pair[1]["total_ms"], reverse=True)),
        "top_kernels": details["kernel"][:30],
        "top_cpu_ops": details["cpu_op"][:20],
        "top_cuda_runtime": details["cuda_runtime"][:20],
    }


def write_csv(path: Path, summaries: list[dict[str, Any]], key: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "rank", "name", "count", "total_ms", "avg_us"])
        writer.writeheader()
        for summary in summaries:
            for rank, row in enumerate(summary[key], 1):
                writer.writerow({"variant": summary["variant"], "rank": rank, **row})


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "profiled_request_elapsed_ms",
        "total_kernel_ms",
        "quantized_linear_ms",
        "dense_linear_ms",
        "attention_ms",
        "memory_copy_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            families = summary["kernel_families"]
            writer.writerow(
                {
                    "variant": summary["variant"],
                    "profiled_request_elapsed_ms": (
                        summary["profiled_request"].get("elapsed_s_with_profiler_overhead", 0.0) * 1000
                    ),
                    "total_kernel_ms": summary["categories"]["kernel"]["total_ms"],
                    "quantized_linear_ms": families.get("quantized_linear", {}).get("total_ms", 0.0),
                    "dense_linear_ms": families.get("dense_linear", {}).get("total_ms", 0.0),
                    "attention_ms": families.get("attention", {}).get("total_ms", 0.0),
                    "memory_copy_ms": families.get("memory_copy", {}).get("total_ms", 0.0),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    preferred_order = ("bf16", "awq_int4", "awq_marlin")
    variant_names = [name for name in preferred_order if (run_dir / name / "metadata.json").exists()]
    variant_names.extend(
        path.name
        for path in sorted(run_dir.iterdir())
        if path.is_dir()
        and path.name not in variant_names
        and (path / "metadata.json").exists()
    )
    summaries = [analyze_variant(run_dir / name) for name in variant_names]
    (run_dir / "analysis.json").write_text(
        json.dumps({"variants": summaries}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(run_dir / "top_kernels.csv", summaries, "top_kernels")
    write_csv(run_dir / "top_cpu_ops.csv", summaries, "top_cpu_ops")
    write_summary_csv(run_dir / "summary.csv", summaries)
    print(json.dumps({"variants": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
