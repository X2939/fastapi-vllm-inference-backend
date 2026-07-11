"""GPU usage visualization utilities.

Plots GPU occupancy over time and memory usage over time.
"""
from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_gpu_occupancy(
    steps: List[int],
    occupancy: List[float],
    output_path: str,
    title: str = "GPU Occupancy Over Time",
) -> None:
    """Plot GPU occupancy over time steps.

    Args:
        steps: Step indices.
        occupancy: GPU occupancy values (0.0 - 1.0).
        output_path: Path to save the PNG.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, [o * 100 for o in occupancy], color="#2196F3", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("GPU Occupancy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.fill_between(steps, [o * 100 for o in occupancy], alpha=0.1, color="#2196F3")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
