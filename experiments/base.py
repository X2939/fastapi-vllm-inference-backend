"""实验基类：提供统一的实验运行框架。"""
import csv
import os
import sys
import random

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.inference_engine import InferenceEngine


def run_experiment(
    experiment_name: str,
    param_name: str,
    param_values: list,
    num_requests: int = 10,
    max_new_tokens: int = 10,
    max_num_seqs: int = 4,
    prefill_cost: float = 0.001,
    decode_cost: float = 0.0005,
    request_generator=None,
):
    """运行实验并输出 CSV。

    Args:
        experiment_name: 实验名称（用于文件名）
        param_name: 参数名称（CSV 列名）
        param_values: 参数值列表
        num_requests: 每组请求数
        max_new_tokens: 默认 max_new_tokens
        max_num_seqs: 默认 max_num_seqs
        prefill_cost: prefill 计算成本
        decode_cost: decode 计算成本
        request_generator: 自定义请求生成器 fn(param_value, i) -> (prompt_length, max_new_tokens, prompt_tokens)

    Returns:
        results: list of dict
    """
    results = []

    for param_value in param_values:
        print(f"\n--- {experiment_name}: {param_name}={param_value} ---")

        # 创建引擎
        engine = InferenceEngine(
            max_num_seqs=max_num_seqs,
            max_memory_budget=5000,
            block_size=16,
            num_blocks=512,
            prefill_cost=prefill_cost,
            decode_cost=decode_cost,
        )

        # 生成请求
        random.seed(42)
        for i in range(num_requests):
            if request_generator:
                prompt_length, max_new, prompt_tokens = request_generator(param_value, i)
            else:
                prompt_length = param_value if param_name == "prompt_length" else 15
                max_new = max_new_tokens
                prompt_tokens = list(range(prompt_length))

            engine.add_request(
                prompt_length=prompt_length,
                max_new_tokens=max_new,
                prompt_tokens=prompt_tokens,
            )

        # 运行
        engine.run(verbose=False)

        # 收集结果
        stats = engine.benchmark.compute_stats()
        kv_stats = engine.kv_cache.get_stats()

        result = {
            param_name: param_value,
            "num_requests": num_requests,
            "avg_ttft": round(stats.avg_ttft, 6),
            "p50_ttft": round(stats.p50_ttft, 6),
            "p95_ttft": round(stats.p95_ttft, 6),
            "avg_tpot": round(stats.avg_tpot, 6),
            "p50_tpot": round(stats.p50_tpot, 6),
            "p95_tpot": round(stats.p95_tpot, 6),
            "avg_latency": round(stats.avg_latency, 6),
            "p50_latency": round(stats.p50_latency, 6),
            "p95_latency": round(stats.p95_latency, 6),
            "throughput": round(stats.throughput, 2),
            "kv_memory_utilization": round(kv_stats["memory_utilization"], 4),
            "total_tokens": stats.total_tokens,
            "total_steps": engine._step_count,
        }
        results.append(result)

        print(f"  TTFT={stats.avg_ttft:.4f}s  TPOT={stats.avg_tpot:.4f}s  "
              f"Latency={stats.avg_latency:.4f}s  Throughput={stats.throughput:.2f} tok/s")

    # 保存 CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{experiment_name}.csv")

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ 保存: {csv_path}")

    return results
