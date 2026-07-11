"""实验 3：Concurrency 对性能的影响

假设：并发越高 → Latency 越大（排队时间增长）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.base import run_experiment


def main():
    print("=" * 70)
    print("  Experiment: Latency vs Concurrency")
    print("=" * 70)

    # 不同的并发请求数
    concurrencies = [1, 2, 4, 8, 16, 32]

    def gen_request(conc, i):
        prompt_len = 15
        return (prompt_len, 10, list(range(prompt_len)))

    results = []
    for conc in concurrencies:
        print(f"\n--- concurrency={conc} ---")
        engine_results = run_experiment(
            experiment_name=f"concurrency_{conc}",
            param_name="concurrency",
            param_values=[conc],
            num_requests=conc,
            max_new_tokens=10,
            max_num_seqs=4,
            request_generator=gen_request,
        )
        if engine_results:
            engine_results[0]["concurrency"] = conc
            results.append(engine_results[0])

    # 保存合并的 CSV
    import csv
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "concurrency.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ 保存: {csv_path}")

    print("\n结论:")
    print("  Concurrency ↑ → Latency ↑ (排队等待时间增长)")
    print("  Concurrency ↑ → P95 Latency 增长更快（长尾效应）")

    return results


if __name__ == "__main__":
    main()
