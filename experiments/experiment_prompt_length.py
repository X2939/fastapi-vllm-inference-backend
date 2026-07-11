"""实验 1：Prompt Length 对性能的影响

假设：Prompt 越长 → TTFT 越大（prefill 计算量大）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.base import run_experiment


def main():
    print("=" * 70)
    print("  Experiment: TTFT vs Prompt Length")
    print("=" * 70)

    # 不同的 prompt 长度
    prompt_lengths = [8, 16, 32, 64, 128, 256, 512]

    def gen_request(prompt_len, i):
        return (prompt_len, 10, list(range(prompt_len)))

    results = run_experiment(
        experiment_name="prompt_length",
        param_name="prompt_length",
        param_values=prompt_lengths,
        num_requests=8,
        max_new_tokens=10,
        max_num_seqs=4,
        request_generator=gen_request,
    )

    print("\n结论:")
    print("  Prompt Length ↑ → TTFT ↑ (prefill 计算量线性增长)")
    print("  Prompt Length 对 TPOT 影响较小（decode 不依赖 prompt 长度）")

    return results


if __name__ == "__main__":
    main()
