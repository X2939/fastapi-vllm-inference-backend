"""Plots for real GPU benchmark artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def write_plots(
    summaries: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Write throughput, streaming-latency and request-latency figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[int(row["concurrency"])].append(row)
    concurrencies = sorted(grouped)

    tokens_per_second = [_mean(grouped[item], "tokens_per_second") for item in concurrencies]
    p95_e2e_ms = [_mean(grouped[item], "p95_latency") * 1000 for item in concurrencies]
    p95_ttft_ms = [_mean(grouped[item], "p95_ttft") * 1000 for item in concurrencies]
    p95_tpot_ms = [_mean(grouped[item], "p95_tpot") * 1000 for item in concurrencies]

    paths: list[Path] = []

    figure, throughput_axis = plt.subplots(figsize=(7, 4.5))
    latency_axis = throughput_axis.twinx()
    throughput_axis.plot(concurrencies, tokens_per_second, "o-", color="#1b9e77", label="Tokens/s")
    latency_axis.plot(concurrencies, p95_e2e_ms, "s--", color="#d95f02", label="P95 E2E")
    throughput_axis.set_xlabel("Concurrency")
    throughput_axis.set_ylabel("Tokens/s", color="#1b9e77")
    latency_axis.set_ylabel("P95 E2E latency (ms)", color="#d95f02")
    throughput_axis.set_title("Real GPU Throughput and P95 E2E Latency")
    throughput_axis.grid(axis="y", alpha=0.25)
    lines = throughput_axis.lines + latency_axis.lines
    throughput_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    path = output_dir / "throughput_vs_concurrency.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(concurrencies, p95_ttft_ms, "o-", label="P95 TTFT")
    axis.plot(concurrencies, p95_tpot_ms, "s-", label="P95 TPOT")
    axis.set_xlabel("Concurrency")
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Real GPU Streaming Latency")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    path = output_dir / "ttft_tpot_vs_concurrency.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    latency_samples: list[list[float]] = []
    for concurrency in concurrencies:
        samples = [
            float(row["latency"]) * 1000
            for row in requests
            if int(row["concurrency"]) == concurrency and row.get("ok") in (True, "True")
        ]
        latency_samples.append(samples)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(latency_samples, tick_labels=[str(item) for item in concurrencies], showfliers=False)
    axis.set_xlabel("Concurrency")
    axis.set_ylabel("E2E latency (ms)")
    axis.set_title("Real GPU Request Latency Distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "request_latency_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    return paths
