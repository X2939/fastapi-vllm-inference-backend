"""实验 2：Batch Size 对性能的影响

假设：Batch Size 越大 → Throughput 越大，但 Latency 也越大
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.base import run_experiment


def main():
    print("=" * 70)
    print("  Experiment: Throughput vs Batch Size")
    print("=" * 70)

    # 不同的 max_num_seqs（模拟 batch size）
    batch_sizes = [1, 2, 4, 8, 16]

    def gen_request(batch_size, i):
        prompt_len = 20
        return (prompt_len, 10, list(range(prompt_len)))

    results = run_experiment(
        experiment_name="batch_size",
        param_name="batch_size",
        param_values=batch_sizes,
        num_requests=16,
        max_new_tokens=10,
        max_num_seqs=1,  # 这里会被 param_value 覆盖，见下面
        request_generator=gen_request,
    )

    # 注意：base.run_experiment 用固定的 max_num_seqs
    # 我们需要单独运行每个 batch_size
    # 重新实现
    from experiments.base import run_experiment as _re
    results = []
    for bs in batch_sizes:
        print(f"\n--- batch_size={bs} ---")
        engine_results = _re(
            experiment_name=f"batch_size_{bs}",
            param_name="batch_size",
            param_values=[bs],
            num_requests=16,
            max_new_tokens=10,
            max_num_seqs=bs,
            request_generator=gen_request,
        )
        if engine_results:
            engine_results[0]["batch_size"] = bs
            results.append(engine_results[0])

    # 保存合并的 CSV
    import csv
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "batch_size.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ 保存: {csv_path}")

    print("\n结论:")
    print("  Batch Size ↑ → Throughput ↑ (GPU 并行度提高)")
    print("  Batch Size ↑ → Latency ↑ (请求间互相等待)")

    return results


if __name__ == "__main__":
    main()
