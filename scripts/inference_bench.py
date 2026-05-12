import argparse
import csv
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


DEFAULT_URL = "http://127.0.0.1:9000/chat"
DEFAULT_PROMPTS = {
    "short": "请用一句话解释什么是 attention。",
    "medium": (
        "请解释大模型推理中的 prefill、decode 和 KV Cache，"
        "并说明它们分别影响哪些性能指标。"
    ),
    "long": (
        "请系统解释 vLLM 在大模型在线推理服务中的作用，包括 "
        "OpenAI-compatible API、continuous batching、PagedAttention、"
        "KV Cache、流式输出、TTFT、吞吐量和显存占用之间的关系。"
    ),
}

#计算「百分位数」
def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

# 发 1 次请求
def one_call(url: str, prompt: str, max_tokens: int, timeout: int, request_id: int) -> dict[str, Any]:
    payload = {
        "message": f"{prompt}\n请求编号：{request_id}",
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}

        return {
            "ok": True,
            "latency": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "error": "",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "latency": elapsed,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "error": str(exc),
        }

#并发压测 vLLM 服务的关键，在于同时发起多个请求并收集它们的响应时间和性能指标
def run_case(
    url: str,
    prompt: str,
    concurrency: int,#一次请求的并发数
    requests_count: int,#请求总数
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(one_call, url, prompt, max_tokens, timeout, request_id)
            for request_id in range(requests_count)
        ]
        # as_completed(futures) 表示哪个请求先完成，就先拿哪个结果
        for future in as_completed(futures):
            results.append(future.result())#future返回的是one_call的返回值

    wall_time = time.perf_counter() - wall_start
    success_results = [result for result in results if result["ok"]]
    failed_results = [result for result in results if not result["ok"]]
    latencies = [result["latency"] for result in success_results]
    completion_tokens = [
        result["completion_tokens"]
        for result in success_results
        if isinstance(result["completion_tokens"], int)
    ]

    total_completion_tokens = sum(completion_tokens)
    tokens_per_second = total_completion_tokens / wall_time if wall_time > 0 else 0.0

    return {
        "concurrency": concurrency,
        "requests": requests_count,
        "success": len(success_results),
        "failed": len(failed_results),
        "wall_time": wall_time,
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "min_latency": min(latencies) if latencies else 0.0,
        "max_latency": max(latencies) if latencies else 0.0,
        "p50_latency": percentile(latencies, 0.50),
        "p95_latency": percentile(latencies, 0.95),
        "throughput": len(success_results) / wall_time if wall_time > 0 else 0.0,
        "tokens_per_second": tokens_per_second,
        "first_error": failed_results[0]["error"] if failed_results else "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "concurrency",
        "requests",
        "success",
        "failed",
        "wall_time",
        "avg_latency",
        "min_latency",
        "max_latency",
        "p50_latency",
        "p95_latency",
        "throughput",
        "tokens_per_second",
        "first_error",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_row(row: dict[str, Any]) -> None:
    print(
        "concurrency={concurrency:<2} requests={requests:<3} "
        "success={success:<3} failed={failed:<3} "
        "avg={avg_latency:.2f}s p50={p50_latency:.2f}s p95={p95_latency:.2f}s "
        "throughput={throughput:.2f} req/s tokens/s={tokens_per_second:.2f}".format(**row)
    )


def parse_concurrency(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FastAPI + vLLM inference service.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--prompt-type", choices=DEFAULT_PROMPTS.keys(), default="medium")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default="reports/inference_benchmark.csv")
    args = parser.parse_args()
#argparse 会自动把横杠 - 转成下划线 _
    prompt = DEFAULT_PROMPTS[args.prompt_type]
    concurrency_levels = parse_concurrency(args.concurrency)

    print("=== Inference Benchmark ===")
    print(f"url={args.url}")
    print(f"prompt_type={args.prompt_type}")
    print(f"requests_per_level={args.requests}")
    print(f"max_tokens={args.max_tokens}")
    print()

    rows = []
    for concurrency in concurrency_levels:
        row = run_case(
            url=args.url,
            prompt=prompt,
            concurrency=concurrency,
            requests_count=args.requests,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        rows.append(row)
        print_row(row)

    output_path = Path(args.output)
    write_csv(output_path, rows)
    print()
    print(f"saved_csv={output_path}")


if __name__ == "__main__":
    main()
