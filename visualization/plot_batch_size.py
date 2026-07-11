"""Batch size visualization utilities.

Plots batch size over time with prefill/decode breakdown.
"""
from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_batch_size(
    steps: List[int],
    prefill_counts: List[int],
    decode_counts: List[int],
    output_path: str,
    title: str = "Batch Size Over Time",
) -> None:
    """Plot batch size over time with stacked prefill/decode bars.

    Args:
        steps: Step indices.
        prefill_counts: Number of prefill requests per step.
        decode_counts: Number of decode requests per step.
        output_path: Path to save the PNG.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(steps, prefill_counts, label="Prefill", color="#FF9800", alpha=0.8)
    ax.bar(steps, decode_counts, bottom=prefill_counts,
           label="Decode", color="#4CAF50", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Batch Size (requests)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
