"""Visualization for Continuous Batching vs Static Batching experiments.

Provides a single function that renders a compact comparison figure with
throughput, TTFT percentiles, latency percentiles and average batch size.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_continuous_batching_results(
    results: List[Dict],
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Render a comparison figure for continuous vs static batching.

    Args:
        results: List of result dicts (one per mode). Expected keys:
            mode, throughput, ttft_avg, ttft_p50, ttft_p95,
            latency_avg, latency_p50, latency_p95, avg_batch_size.
        output_path: Path to write the PNG file.
        title: Optional figure title.
    """
    if not results:
        return

    modes = [r["mode"].capitalize() for r in results]
    x = np.arange(len(modes))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        title or "Continuous Batching vs Static Batching",
        fontsize=14,
        fontweight="bold",
    )

    # Throughput
    ax = axes[0, 0]
    bars = ax.bar(modes, [r["throughput"] for r in results], color=["#4CAF50", "#F44336"])
    ax.set_ylabel("Tokens / sec")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + height * 0.01,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # TTFT percentiles
    ax = axes[0, 1]
    ttft_avg = [r["ttft_avg"] * 1000 for r in results]
    ttft_p50 = [r["ttft_p50"] * 1000 for r in results]
    ttft_p95 = [r["ttft_p95"] * 1000 for r in results]
    ax.bar(x - width, ttft_avg, width, label="avg", color="#2196F3")
    ax.bar(x, ttft_p50, width, label="p50", color="#03A9F4")
    ax.bar(x + width, ttft_p95, width, label="p95", color="#00BCD4")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("ms")
    ax.set_title("TTFT")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Latency percentiles
    ax = axes[1, 0]
    lat_avg = [r["latency_avg"] * 1000 for r in results]
    lat_p50 = [r["latency_p50"] * 1000 for r in results]
    lat_p95 = [r["latency_p95"] * 1000 for r in results]
    ax.bar(x - width, lat_avg, width, label="avg", color="#FF9800")
    ax.bar(x, lat_p50, width, label="p50", color="#FFC107")
    ax.bar(x + width, lat_p95, width, label="p95", color="#FFEB3B")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("ms")
    ax.set_title("End-to-End Latency")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Average batch size
    ax = axes[1, 1]
    bars = ax.bar(modes, [r["avg_batch_size"] for r in results], color=["#9C27B0", "#673AB7"])
    ax.set_ylabel("Requests")
    ax.set_title("Average Batch Size")
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + height * 0.01,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
