"""Compare BF16 and AWQ real GPU benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


WORKLOAD_KEYS = (
    "prompt_type",
    "prompt_mode",
    "requests_per_level",
    "warmup",
    "runs",
    "max_tokens",
)
METRICS = ("tokens_per_second", "p95_ttft", "p95_tpot", "p95_latency")


def _read_artifact(directory: Path) -> tuple[dict, list[dict]]:
    with (directory / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (directory / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return metadata, rows


def _aggregate(rows: list[dict]) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["concurrency"])].append(row)
    return {
        concurrency: {
            metric: sum(float(row[metric]) for row in group) / len(group)
            for metric in METRICS
        }
        for concurrency, group in grouped.items()
    }


def _change(before: float, after: float) -> float:
    return (after / before - 1) * 100 if before else 0.0


def _memory_after(metadata: dict) -> str:
    environment = metadata.get("environment_after") or metadata.get("environment") or {}
    nvidia_smi = environment.get("nvidia_smi", {})
    if not isinstance(nvidia_smi, dict):
        return "unavailable"
    gpus = nvidia_smi.get("gpus", [])
    if isinstance(gpus, list) and gpus:
        gpu = gpus[0]
        return f"{gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB"
    stdout = nvidia_smi.get("stdout")
    return str(stdout) if stdout else "unavailable"


def _write_report(
    path: Path,
    bf16_meta: dict,
    awq_meta: dict,
    bf16: dict[int, dict[str, float]],
    awq: dict[int, dict[str, float]],
) -> None:
    lines = [
        "# BF16 vs AWQ Real GPU Comparison",
        "",
        "> Positive delta means the AWQ value is higher than BF16. For TTFT, TPOT and E2E latency, a negative delta is an improvement.",
        "",
        "## Controlled Variables",
        "",
    ]
    lines.extend(f"- `{key}`: `{bf16_meta[key]}`" for key in WORKLOAD_KEYS)
    lines.extend(
        [
            f"- BF16 model: `{bf16_meta.get('model')}`",
            f"- AWQ model: `{awq_meta.get('model')}`",
            f"- BF16 server args: `{bf16_meta.get('server_args', '')}`",
            f"- AWQ server args: `{awq_meta.get('server_args', '')}`",
            f"- BF16 memory after run: `{_memory_after(bf16_meta)}`",
            f"- AWQ memory after run: `{_memory_after(awq_meta)}`",
            "",
            "## Results",
            "",
            "| Concurrency | Tokens/s BF16 → AWQ | Δ | P95 TTFT BF16 → AWQ | Δ | P95 TPOT BF16 → AWQ | Δ | P95 E2E BF16 → AWQ | Δ |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for concurrency in sorted(bf16):
        base = bf16[concurrency]
        quantized = awq[concurrency]
        lines.append(
            "| {c} | {b_tokens:.2f} → {a_tokens:.2f} | {d_tokens:+.1f}% | "
            "{b_ttft:.1f} → {a_ttft:.1f} ms | {d_ttft:+.1f}% | "
            "{b_tpot:.2f} → {a_tpot:.2f} ms | {d_tpot:+.1f}% | "
            "{b_e2e:.1f} → {a_e2e:.1f} ms | {d_e2e:+.1f}% |".format(
                c=concurrency,
                b_tokens=base["tokens_per_second"],
                a_tokens=quantized["tokens_per_second"],
                d_tokens=_change(base["tokens_per_second"], quantized["tokens_per_second"]),
                b_ttft=base["p95_ttft"] * 1000,
                a_ttft=quantized["p95_ttft"] * 1000,
                d_ttft=_change(base["p95_ttft"], quantized["p95_ttft"]),
                b_tpot=base["p95_tpot"] * 1000,
                a_tpot=quantized["p95_tpot"] * 1000,
                d_tpot=_change(base["p95_tpot"], quantized["p95_tpot"]),
                b_e2e=base["p95_latency"] * 1000,
                a_e2e=quantized["p95_latency"] * 1000,
                d_e2e=_change(base["p95_latency"], quantized["p95_latency"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "AWQ reduces weight precision, but it does not shrink KV Cache at the same ratio. On this RTX 3060 Laptop GPU run, AWQ saves model weight storage but decode latency is worse because INT4 dequantization and kernel support dominate this small-batch workload. Treat this as a real tradeoff measurement, not as a blanket claim that quantization always improves throughput.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, bf16: dict[int, dict[str, float]], awq: dict[int, dict[str, float]]) -> None:
    concurrencies = sorted(bf16)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    chart_metrics = (
        ("tokens_per_second", "Tokens/s", 1),
        ("p95_ttft", "P95 TTFT (ms)", 1000),
        ("p95_tpot", "P95 TPOT (ms)", 1000),
        ("p95_latency", "P95 E2E (ms)", 1000),
    )
    for axis, (metric, label, scale) in zip(axes.flat, chart_metrics):
        x = list(range(len(concurrencies)))
        width = 0.36
        axis.bar([item - width / 2 for item in x], [bf16[c][metric] * scale for c in concurrencies], width, label="BF16")
        axis.bar([item + width / 2 for item in x], [awq[c][metric] * scale for c in concurrencies], width, label="AWQ INT4")
        axis.set_xticks(x, [str(item) for item in concurrencies])
        axis.set_xlabel("Concurrency")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    figure.suptitle("Real GPU BF16 vs AWQ")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BF16 and AWQ GPU benchmark directories.")
    parser.add_argument("--bf16", default="reports/gpu_benchmark")
    parser.add_argument("--awq", default="reports/gpu_awq")
    parser.add_argument("--output-dir", default="reports/gpu_quantization_comparison")
    args = parser.parse_args()

    bf16_meta, bf16_rows = _read_artifact(Path(args.bf16))
    awq_meta, awq_rows = _read_artifact(Path(args.awq))
    mismatches = [key for key in WORKLOAD_KEYS if bf16_meta.get(key) != awq_meta.get(key)]
    if mismatches:
        parser.error(f"workload mismatch: {', '.join(mismatches)}")
    bf16 = _aggregate(bf16_rows)
    awq = _aggregate(awq_rows)
    if set(bf16) != set(awq):
        parser.error("concurrency levels differ")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_report(output_dir / "REPORT.md", bf16_meta, awq_meta, bf16, awq)
    _write_plot(output_dir / "bf16_vs_awq.png", bf16, awq)
    print(f"saved_report={output_dir / 'REPORT.md'}")
    print(f"saved_plot={output_dir / 'bf16_vs_awq.png'}")


if __name__ == "__main__":
    main()
