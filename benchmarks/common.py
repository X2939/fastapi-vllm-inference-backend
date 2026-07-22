"""Core benchmark primitives used by scripts and experiment runner."""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any

import requests


class PromptMode(str, Enum):
    """Controls prompt construction for controlled experiments."""

    UNIQUE = "unique"
    SHARED_PREFIX = "shared_prefix"


PROMPT_TEMPLATES = {
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

LONG_PREFILL_BODY = (
    "你是一名 AI Infra 工程师。请阅读下面的推理系统设计材料，然后总结其中的调度、"
    "KV Cache、显存管理和性能指标关系。\n"
    "材料：vLLM 在线推理服务通常把请求分为 prefill 和 decode 两个阶段。Prefill "
    "阶段处理输入 prompt，为每一层生成 Key/Value 并写入 KV Cache；decode 阶段每轮"
    "只生成一个新 token，但需要读取历史 KV。Scheduler 需要在请求并发、token budget、"
    "KV block 容量和服务延迟之间做权衡。PagedAttention 风格的 block table 让逻辑"
    "序列连续而物理 block 不连续，从而减少外部碎片。Prefix Cache 可以复用共享前缀"
    "的 KV block，Chunked Prefill 可以把长 prompt 拆成多个 chunk，让短请求和 decode "
    "请求有机会插入执行。性能分析需要同时观察 TTFT、TPOT、P95、吞吐和显存占用。\n"
)

SHARED_PREFIX_BODY = (
    "以下是一段共享前缀，用于观察相同 prompt 前缀在不同请求中的 prefill 行为。"
    "在大模型推理中，prefill 阶段会为所有输入 token 计算 KV Cache；"
    "decode 阶段则逐步复用已缓存的 Key/Value，只对新 token 做 attention。"
    "当多个请求共享相同前缀且引擎启用了 prefix caching 时，"
    "重复前缀的 prefill 成本可能降低。请基于这段前缀回答："
)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_prompt(
    prompt_type: str,
    request_id: int,
    mode: PromptMode = PromptMode.UNIQUE,
) -> str:
    if prompt_type == "mixed":
        if request_id % 4 == 0:
            repeated = "\n".join(
                f"段落 {index}: {LONG_PREFILL_BODY}" for index in range(18)
            )
            return f"{repeated}\n长请求编号：{request_id}"
        return f"{PROMPT_TEMPLATES['short']}\n短请求编号：{request_id}"

    base = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["medium"])

    if mode == PromptMode.SHARED_PREFIX:
        return f"{SHARED_PREFIX_BODY}{base}\n变体编号：{request_id % 4}。"

    return f"{base}\n请求编号：{request_id}"


def one_call(
    url: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    request_id: int,
) -> dict[str, Any]:
    payload = {
        "message": prompt,
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


def run_warmup(
    url: str,
    prompt_type: str,
    max_tokens: int,
    timeout: int,
    warmup: int,
    prompt_mode: PromptMode,
) -> None:
    for request_id in range(warmup):
        one_call(
            url,
            build_prompt(prompt_type, request_id, prompt_mode),
            max_tokens,
            timeout,
            -(request_id + 1),
        )


def run_case(
    url: str,
    prompt_type: str,
    concurrency: int,
    requests_count: int,
    max_tokens: int,
    timeout: int,
    warmup: int = 0,
    prompt_mode: PromptMode = PromptMode.UNIQUE,
) -> dict[str, Any]:
    run_warmup(url, prompt_type, max_tokens, timeout, warmup, prompt_mode)

    wall_start = time.perf_counter()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                one_call,
                url,
                build_prompt(prompt_type, request_id, prompt_mode),
                max_tokens,
                timeout,
                request_id,
            )
            for request_id in range(requests_count)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    wall_time = time.perf_counter() - wall_start
    success_results = [result for result in results if result["ok"]]
    failed_results = [result for result in results if not result["ok"]]
    latencies = [result["latency"] for result in success_results]
    completion_tokens = [
        result["completion_tokens"]
        for result in success_results
        if isinstance(result["completion_tokens"], int)
    ]
    prompt_tokens = [
        result["prompt_tokens"]
        for result in success_results
        if isinstance(result["prompt_tokens"], int)
    ]

    total_completion_tokens = sum(completion_tokens)
    total_prompt_tokens = sum(prompt_tokens)
    avg_completion_tokens = statistics.mean(completion_tokens) if completion_tokens else 0.0
    avg_prompt_tokens = statistics.mean(prompt_tokens) if prompt_tokens else 0.0
    tokens_per_second = total_completion_tokens / wall_time if wall_time > 0 else 0.0
    tpot = (
        sum(result["latency"] for result in success_results) / total_completion_tokens
        if total_completion_tokens > 0
        else 0.0
    )

    return {
        "concurrency": concurrency,
        "requests": requests_count,
        "warmup": warmup,
        "prompt_type": prompt_type,
        "prompt_mode": prompt_mode.value,
        "max_tokens": max_tokens,
        "success": len(success_results),
        "failed": len(failed_results),
        "error_rate": len(failed_results) / requests_count if requests_count > 0 else 0.0,
        "wall_time": wall_time,
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "min_latency": min(latencies) if latencies else 0.0,
        "max_latency": max(latencies) if latencies else 0.0,
        "p50_latency": percentile(latencies, 0.50),
        "p95_latency": percentile(latencies, 0.95),
        "p99_latency": percentile(latencies, 0.99),
        "throughput": len(success_results) / wall_time if wall_time > 0 else 0.0,
        "total_prompt_tokens": total_prompt_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "tokens_per_second": tokens_per_second,
        "tpot": tpot,
        "first_error": failed_results[0]["error"] if failed_results else "",
    }


def run_stream_once(
    url: str,
    message: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "message": message,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    first_chunk_time = None
    chunks: list[str] = []

    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start
            chunks.append(chunk)

    total_time = time.perf_counter() - start
    output_text = "".join(chunks)

    return {
        "ttft": first_chunk_time or 0.0,
        "total_time": total_time,
        "chars": len(output_text),
        "chars_per_second": len(output_text) / total_time if total_time > 0 else 0.0,
    }
