"""Memory usage visualization utilities.

Plots KV cache memory usage over time.
"""
from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_memory_usage(
    steps: List[int],
    memory_used: List[int],
    memory_budget: int,
    output_path: str,
    title: str = "Memory Usage Over Time",
) -> None:
    """Plot memory usage over time.

    Args:
        steps: Step indices.
        memory_used: Memory used at each step (in tokens or bytes).
        memory_budget: Total memory budget.
        output_path: Path to save the PNG.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, memory_used, color="#9C27B0", linewidth=2, label="Memory Used")
    ax.axhline(y=memory_budget, color="#F44336", linestyle="--",
               linewidth=1.5, label=f"Budget ({memory_budget})")
    ax.fill_between(steps, memory_used, alpha=0.1, color="#9C27B0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Memory (tokens)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
