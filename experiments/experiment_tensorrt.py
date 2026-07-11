"""Experiment: PyTorch vs ONNX Runtime vs TensorRT comparison.

Compares three inference backends:
    - PyTorch: Uses time.sleep simulation (Executor)
    - ONNX: Uses real ONNX Runtime inference (ONNXExecutor)
    - TensorRT: Uses real TensorRT GPU inference (TensorRTExecutor)

Measures:
    - TTFT (Time To First Token)
    - Latency (end-to-end)
    - Throughput (tokens/sec)
    - Memory usage

Usage:
    python3 -m experiments.experiment_tensorrt
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.inference_engine import InferenceEngine
from engine.policy import MemoryBudgetPolicy

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_backend(
    backend: str,
    onnx_model_path: str = "models/simple_model.onnx",
    trt_engine_path: str = "models/simple_model.engine",
    num_requests: int = 20,
    seed: int = 42,
) -> dict:
    """Run inference with a given backend.

    Args:
        backend: "pytorch", "onnx", or "tensorrt".
        onnx_model_path: Path to ONNX model (for onnx backend).
        trt_engine_path: Path to TensorRT engine (for tensorrt backend).
        num_requests: Number of requests to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary of result metrics.
    """
    random.seed(seed)

    engine = InferenceEngine(
        block_size=16,
        num_blocks=1024,
        prefill_cost_per_token=0.001,
        decode_cost_per_token=0.0005,
        policy=MemoryBudgetPolicy(memory_budget=80000),
        enable_prefix_sharing=True,
        backend=backend,
        onnx_model_path=onnx_model_path,
        trt_engine_path=trt_engine_path,
    )

    # Generate requests
    shared_prefix = list(range(1, 9))
    for i in range(num_requests):
        prompt_length = random.randint(10, 60)
        max_new_tokens = random.randint(5, 25)
        if i % 4 == 0:
            prompt_tokens = shared_prefix + list(
                range(100 + i, 100 + i + prompt_length - len(shared_prefix))
            )
        else:
            prompt_tokens = list(
                range(200 + i * 100, 200 + i * 100 + prompt_length)
            )
        engine.add_request(
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens,
        )

    engine.run(max_steps=100000)
    result = engine.get_results()
    return result


def main():
    print("=" * 60)
    print("  Experiment: PyTorch vs ONNX vs TensorRT")
    print("=" * 60)
    print()

    backends = ["pytorch", "onnx", "tensorrt"]
    onnx_model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "simple_model.onnx"
    )
    trt_engine_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "simple_model.engine"
    )

    all_results = []

    for backend in backends:
        print(f"  Running backend={backend}...")
        result = run_backend(
            backend,
            onnx_model_path=onnx_model_path,
            trt_engine_path=trt_engine_path,
        )
        all_results.append(result)
        print(f"    Throughput:  {result['throughput']:.2f} tok/s")
        print(f"    TTFT avg:    {result['ttft']['avg']:.4f}s")
        print(f"    Latency avg: {result['latency']['avg']:.4f}s")
        print(f"    Total Time:  {result['total_time']:.4f}s")
        print()

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "experiment_tensorrt.csv")
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Saved: {csv_path}")

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "experiment_tensorrt.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Generate plots
    try:
        labels = [r["backend"] for r in all_results]
        colors = ["#2196F3", "#4CAF50", "#FF9800"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].bar(labels, [r["ttft"]["avg"] for r in all_results],
                       color=colors)
        axes[0, 0].set_title("Average TTFT (s)")
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        axes[0, 1].bar(labels, [r["latency"]["avg"] for r in all_results],
                       color=colors)
        axes[0, 1].set_title("Average Latency (s)")
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        axes[1, 0].bar(labels, [r["throughput"] for r in all_results],
                       color=colors)
        axes[1, 0].set_title("Throughput (tok/s)")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        axes[1, 1].bar(labels, [r["total_time"] for r in all_results],
                       color=colors)
        axes[1, 1].set_title("Total Time (s)")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        png_path = os.path.join(RESULTS_DIR, "experiment_tensorrt.png")
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {png_path}")
    except Exception as e:
        print(f"  Warning: Could not generate plot: {e}")

    print()
    print("=" * 60)
    print("  PyTorch vs ONNX vs TensorRT Experiment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
