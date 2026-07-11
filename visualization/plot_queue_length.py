"""Queue length visualization utilities.

Plots waiting/running/finished queue sizes over time.
"""
from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_queue_length(
    steps: List[int],
    waiting: List[int],
    running: List[int],
    finished: List[int],
    output_path: str,
    title: str = "Queue Lengths Over Time",
) -> None:
    """Plot queue lengths over time.

    Args:
        steps: Step indices.
        waiting: Waiting queue sizes.
        running: Running queue sizes.
        finished: Finished queue sizes.
        output_path: Path to save the PNG.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, waiting, label="Waiting", color="#F44336", linewidth=2)
    ax.plot(steps, running, label="Running", color="#FF9800", linewidth=2)
    ax.plot(steps, finished, label="Finished", color="#4CAF50", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Requests")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
