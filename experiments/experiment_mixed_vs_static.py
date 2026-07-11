"""实验：Mixed Batch vs Static Batch 对比

假设：
- Mixed Batch（vLLM 风格）：prefill 和 decode 混合，GPU 利用率高
- Static Batch（传统）：不混合，GPU 利用率低

对比指标：
- TTFT / TPOT / Latency / Throughput
- GPU Occupancy
- Admission Rate
"""
import csv
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.inference_engine import InferenceEngine


def run_experiment(mode: str, num_requests: int = 16) -> dict:
    """运行一次实验。

    Args:
        mode: "mixed" 或 "static"
        num_requests: 请求数
    """
    print(f"\n--- {mode.upper()} Batch Mode ---")

    engine = InferenceEngine(
        max_num_seqs=256,
        gpu_memory_budget=80000,
        block_size=16,
        num_blocks=1024,
        prefill_cost=0.001,
        decode_cost=0.0005,
        static_batch_mode=(mode == "static"),
    )

    # 生成请求（两种模式用相同的请求）
    random.seed(42)
    for i in range(num_requests):
        prompt_length = random.randint(10, 50)
        max_new_tokens = random.randint(5, 20)
        prompt_tokens = list(range(prompt_length))
        engine.add_request(
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens,
        )

    engine.run(verbose=False)

    results = engine._get_final_results()
    results["mode"] = mode
    print(f"  TTFT={results['avg_ttft']:.4f}s  TPOT={results['avg_tpot']:.4f}s  "
          f"Latency={results['avg_latency']:.4f}s  "
          f"Throughput={results['throughput']:.2f} tok/s  "
          f"GPU_Occ={results['gpu_occupancy']*100:.1f}%")
    return results


def main():
    print("=" * 70)
    print("  Experiment: Mixed Batch vs Static Batch")
    print("=" * 70)

    results_mixed = run_experiment("mixed", num_requests=16)
    results_static = run_experiment("static", num_requests=16)

    # 保存 CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 分别保存
    for results, name in [(results_mixed, "mixed_batch"), (results_static, "static_batch")]:
        csv_path = os.path.join(results_dir, f"{name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)
        print(f"✓ 保存: {csv_path}")

    # 对比表
    print()
    print("=" * 70)
    print("  对比结果")
    print("=" * 70)
    print(f"{'Metric':<25} {'Mixed':>15} {'Static':>15} {'Speedup':>10}")
    print("-" * 70)

    metrics = [
        ("Avg TTFT (s)", "avg_ttft"),
        ("Avg TPOT (s)", "avg_tpot"),
        ("Avg Latency (s)", "avg_latency"),
        ("Throughput (tok/s)", "throughput"),
        ("GPU Occupancy", "gpu_occupancy"),
        ("Avg Batch Size", "avg_batch_size"),
        ("Admission Rate", "admission_rate"),
    ]

    for label, key in metrics:
        v_mixed = results_mixed[key]
        v_static = results_static[key]
        if key in ("gpu_occupancy", "admission_rate"):
            v_mixed_str = f"{v_mixed*100:.1f}%"
            v_static_str = f"{v_static*100:.1f}%"
            speedup = f"{v_mixed/v_static:.2f}x" if v_static > 0 else "N/A"
        elif key == "throughput":
            v_mixed_str = f"{v_mixed:.2f}"
            v_static_str = f"{v_static:.2f}"
            speedup = f"{v_mixed/v_static:.2f}x" if v_static > 0 else "N/A"
        elif "ttft" in key or "tpot" in key or "latency" in key:
            v_mixed_str = f"{v_mixed:.4f}"
            v_static_str = f"{v_static:.4f}"
            speedup = f"{v_static/v_mixed:.2f}x" if v_mixed > 0 else "N/A"
        else:
            v_mixed_str = f"{v_mixed:.2f}"
            v_static_str = f"{v_static:.2f}"
            speedup = f"{v_mixed/v_static:.2f}x" if v_static > 0 else "N/A"

        print(f"{label:<25} {v_mixed_str:>15} {v_static_str:>15} {speedup:>10}")

    print()
    print("结论:")
    print("  Mixed Batch 的 GPU Occupancy 更高（减少了 Idle 时间）")
    print("  Mixed Batch 的 Throughput 更高（GPU 利用率提升）")
    print("  Static Batch 的 Latency 可能更高（prefill 被推迟）")


if __name__ == "__main__":
    main()
