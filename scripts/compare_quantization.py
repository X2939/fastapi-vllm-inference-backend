"""Compare BF16 and AWQ real GPU benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


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


def _variant_label(metadata: dict) -> str:
    variant = str(metadata.get("experiment_variant", "awq"))
    return "AWQ-Marlin" if "marlin" in variant.lower() else "AWQ"


def _write_report(
    path: Path,
    bf16_meta: dict,
    awq_meta: dict,
    bf16: dict[int, dict[str, float]],
    awq: dict[int, dict[str, float]],
) -> None:
    awq_label = _variant_label(awq_meta)
    lines = [
        f"# BF16 vs {awq_label} Real GPU Comparison",
        "",
        (
            f"> Positive delta means the {awq_label} value is higher than BF16. "
            "For TTFT, TPOT and E2E latency, a negative delta is an improvement."
        ),
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
            (
                f"| Concurrency | Tokens/s BF16 -> {awq_label} | Delta | "
                f"P95 TTFT BF16 -> {awq_label} | Delta | "
                f"P95 TPOT BF16 -> {awq_label} | Delta | "
                f"P95 E2E BF16 -> {awq_label} | Delta |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for concurrency in sorted(bf16):
        base = bf16[concurrency]
        quantized = awq[concurrency]
        lines.append(
            "| {c} | {b_tokens:.2f} -> {a_tokens:.2f} | {d_tokens:+.1f}% | "
            "{b_ttft:.1f} -> {a_ttft:.1f} ms | {d_ttft:+.1f}% | "
            "{b_tpot:.2f} -> {a_tpot:.2f} ms | {d_tpot:+.1f}% | "
            "{b_e2e:.1f} -> {a_e2e:.1f} ms | {d_e2e:+.1f}% |".format(
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

    concurrencies = sorted(bf16)
    throughput_changes = [
        _change(bf16[c]["tokens_per_second"], awq[c]["tokens_per_second"])
        for c in concurrencies
    ]
    tpot_changes = [_change(bf16[c]["p95_tpot"], awq[c]["p95_tpot"]) for c in concurrencies]
    e2e_changes = [_change(bf16[c]["p95_latency"], awq[c]["p95_latency"]) for c in concurrencies]
    ttft_changes = [_change(bf16[c]["p95_ttft"], awq[c]["p95_ttft"]) for c in concurrencies]
    lines.extend(
        [
            "",
            "## Findings",
            "",
            (
                f"- {awq_label} changed token throughput by {min(throughput_changes):+.1f}% to "
                f"{max(throughput_changes):+.1f}% across the tested concurrency levels."
            ),
            (
                f"- P95 TPOT changed by {min(tpot_changes):+.1f}% to {max(tpot_changes):+.1f}%, "
                f"and P95 E2E changed by {min(e2e_changes):+.1f}% to {max(e2e_changes):+.1f}%."
            ),
            (
                f"- P95 TTFT was mixed ({min(ttft_changes):+.1f}% to {max(ttft_changes):+.1f}%). "
                "A faster decode kernel can improve TPOT without guaranteeing lower prefill or queueing tail latency."
            ),
            "",
            "## Interpretation Boundary",
            "",
            (
                "AWQ is weight-only quantization, so it does not shrink KV Cache at the same ratio. "
                "With a fixed vLLM GPU-memory-utilization budget, lower weight memory can be reassigned "
                "to KV capacity; similar total nvidia-smi memory after startup is therefore expected. "
                "The result is specific to this checkpoint, backend, GPU, software stack, and workload."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(
    path: Path,
    bf16: dict[int, dict[str, float]],
    awq: dict[int, dict[str, float]],
    awq_label: str,
) -> None:
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
        axis.bar(
            [item - width / 2 for item in x],
            [bf16[c][metric] * scale for c in concurrencies],
            width,
            label="BF16",
        )
        axis.bar(
            [item + width / 2 for item in x],
            [awq[c][metric] * scale for c in concurrencies],
            width,
            label=awq_label,
        )
        axis.set_xticks(x, [str(item) for item in concurrencies])
        axis.set_xlabel("Concurrency")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    figure.suptitle(f"Real GPU BF16 vs {awq_label}")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    figure.savefig(path.with_suffix(".svg"))
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
    plot_path = output_dir / "bf16_vs_awq.png"
    _write_plot(plot_path, bf16, awq, _variant_label(awq_meta))
    print(f"saved_report={output_dir / 'REPORT.md'}")
    print(f"saved_plot={plot_path}")
    print(f"saved_vector_plot={plot_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
