"""Thin CLI wrapper around the shared benchmark library."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.common import PromptMode, run_case
from benchmarks.report import write_csv


DEFAULT_URL = "http://127.0.0.1:9000/chat"


def print_row(row: dict) -> None:
    print(
        "concurrency={concurrency:<2} requests={requests:<3} "
        "warmup={warmup:<2} mode={prompt_mode:<14} success={success:<3} failed={failed:<3} "
        "error_rate={error_rate:.2%} avg={avg_latency:.2f}s "
        "p50={p50_latency:.2f}s p95={p95_latency:.2f}s p99={p99_latency:.2f}s "
        "throughput={throughput:.2f} req/s tokens/s={tokens_per_second:.2f} "
        "tpot={tpot:.4f}s/token".format(**row)
    )


def parse_concurrency(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FastAPI + vLLM inference service.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--prompt-type", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--prompt-mode", choices=["unique", "shared_prefix"], default="unique")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default="reports/inference_benchmark.csv")
    args = parser.parse_args()

    concurrency_levels = parse_concurrency(args.concurrency)

    print("=== Inference Benchmark ===")
    print(f"url={args.url}")
    print(f"prompt_type={args.prompt_type}")
    print(f"prompt_mode={args.prompt_mode}")
    print(f"requests_per_level={args.requests}")
    print(f"warmup_per_level={args.warmup}")
    print(f"max_tokens={args.max_tokens}")
    print()

    rows = []
    for concurrency in concurrency_levels:
        row = run_case(
            url=args.url,
            prompt_type=args.prompt_type,
            concurrency=concurrency,
            requests_count=args.requests,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            warmup=args.warmup,
            prompt_mode=PromptMode(args.prompt_mode),
        )
        rows.append(row)
        print_row(row)

    output_path = Path(args.output)
    write_csv(output_path, rows)
    print()
    print(f"saved_csv={output_path}")


if __name__ == "__main__":
    main()
