"""Compare two real GPU benchmark artifacts with identical workloads."""

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
    "model",
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


def _write_report(
    path: Path,
    before_meta: dict,
    after_meta: dict,
    before: dict[int, dict[str, float]],
    after: dict[int, dict[str, float]],
    *,
    title: str,
    before_label: str,
    after_label: str,
    interpretation: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"> Positive delta means the {after_label} value is higher than {before_label}. For TTFT, TPOT and E2E latency, a negative delta is an improvement.",
        "",
        "## Controlled Variables",
        "",
    ]
    lines.extend(f"- `{key}`: `{before_meta[key]}`" for key in WORKLOAD_KEYS)
    lines.extend(
        [
            f"- {before_label} server args: `{before_meta.get('server_args', '')}`",
            f"- {after_label} server args: `{after_meta.get('server_args', '')}`",
            "",
            "## Results",
            "",
            "| Concurrency | Tokens/s OFF → ON | Δ | P95 TTFT OFF → ON | Δ | P95 TPOT OFF → ON | Δ | P95 E2E OFF → ON | Δ |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for concurrency in sorted(before):
        base = before[concurrency]
        optimized = after[concurrency]
        lines.append(
            "| {c} | {b_tokens:.2f} → {a_tokens:.2f} | {d_tokens:+.1f}% | "
            "{b_ttft:.1f} → {a_ttft:.1f} ms | {d_ttft:+.1f}% | "
            "{b_tpot:.2f} → {a_tpot:.2f} ms | {d_tpot:+.1f}% | "
            "{b_e2e:.1f} → {a_e2e:.1f} ms | {d_e2e:+.1f}% |".format(
                c=concurrency,
                b_tokens=base["tokens_per_second"], a_tokens=optimized["tokens_per_second"],
                d_tokens=_change(base["tokens_per_second"], optimized["tokens_per_second"]),
                b_ttft=base["p95_ttft"] * 1000, a_ttft=optimized["p95_ttft"] * 1000,
                d_ttft=_change(base["p95_ttft"], optimized["p95_ttft"]),
                b_tpot=base["p95_tpot"] * 1000, a_tpot=optimized["p95_tpot"] * 1000,
                d_tpot=_change(base["p95_tpot"], optimized["p95_tpot"]),
                b_e2e=base["p95_latency"] * 1000, a_e2e=optimized["p95_latency"] * 1000,
                d_e2e=_change(base["p95_latency"], optimized["p95_latency"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            interpretation,
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(
    path: Path,
    before: dict[int, dict[str, float]],
    after: dict[int, dict[str, float]],
    *,
    title: str,
    before_label: str,
    after_label: str,
) -> None:
    concurrencies = sorted(before)
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
        axis.bar([item - width / 2 for item in x], [before[c][metric] * scale for c in concurrencies], width, label=before_label)
        axis.bar([item + width / 2 for item in x], [after[c][metric] * scale for c in concurrencies], width, label=after_label)
        axis.set_xticks(x, [str(item) for item in concurrencies])
        axis.set_xlabel("Concurrency")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two GPU benchmark directories.")
    parser.add_argument("--before", required=True, help="Baseline artifact directory")
    parser.add_argument("--after", required=True, help="Optimized artifact directory")
    parser.add_argument("--output-dir", default="reports/gpu_prefix_cache_comparison")
    parser.add_argument("--title", default="Prefix Cache Real GPU A/B")
    parser.add_argument("--before-label", default="Prefix Cache OFF")
    parser.add_argument("--after-label", default="Prefix Cache ON")
    parser.add_argument("--plot-name", default="prefix_cache_ab_comparison.png")
    parser.add_argument(
        "--interpretation",
        default=(
            "Prefix Cache skips repeated prefill work, so TTFT is its primary expected "
            "benefit. TPOT mainly reflects decode work and should not be presented as a "
            "direct Prefix Cache gain. Throughput and E2E latency depend on scheduler "
            "batch composition, cache lookup overhead and the host/GPU runtime; they "
            "must be interpreted from the measured deltas rather than assumed."
        ),
    )
    args = parser.parse_args()

    before_meta, before_rows = _read_artifact(Path(args.before))
    after_meta, after_rows = _read_artifact(Path(args.after))
    mismatches = [key for key in WORKLOAD_KEYS if before_meta.get(key) != after_meta.get(key)]
    if mismatches:
        parser.error(f"workload mismatch: {', '.join(mismatches)}")
    before = _aggregate(before_rows)
    after = _aggregate(after_rows)
    if set(before) != set(after):
        parser.error("concurrency levels differ")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_report(
        output_dir / "REPORT.md",
        before_meta,
        after_meta,
        before,
        after,
        title=args.title,
        before_label=args.before_label,
        after_label=args.after_label,
        interpretation=args.interpretation,
    )
    _write_plot(
        output_dir / args.plot_name,
        before,
        after,
        title=args.title,
        before_label=args.before_label,
        after_label=args.after_label,
    )
    print(f"saved_report={output_dir / 'REPORT.md'}")
    print(f"saved_plot={output_dir / args.plot_name}")


if __name__ == "__main__":
    main()
