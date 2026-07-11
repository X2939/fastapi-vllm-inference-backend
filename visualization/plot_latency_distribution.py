"""Latency distribution visualization utilities.

Plots histogram and CDF of latency / TTFT / TPOT.
"""
from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_latency_distribution(
    latencies: List[float],
    output_path: str,
    title: str = "Latency Distribution",
    xlabel: str = "Latency (s)",
) -> None:
    """Plot latency distribution (histogram + CDF).

    Args:
        latencies: List of latency values.
        output_path: Path to save the PNG.
        title: Plot title.
        xlabel: X-axis label.
    """
    if not latencies:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(latencies, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Count")
    ax1.set_title(f"{title} - Histogram")
    ax1.grid(True, alpha=0.3)

    sorted_data = sorted(latencies)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, cdf * 100, color="#4CAF50", linewidth=2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("CDF (%)")
    ax2.set_title(f"{title} - CDF")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
