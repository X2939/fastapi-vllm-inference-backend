"""Streaming benchmark primitives for a real OpenAI-compatible vLLM server.

The default benchmark suite intentionally uses a CPU cost model. This module is
separate: it measures timestamps from real Server-Sent Events (SSE) and is only
meaningful when the target server runs a model on a GPU.
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from benchmarks.common import PromptMode, build_prompt, percentile


def _as_text(line: str | bytes) -> str:
    return line.decode("utf-8") if isinstance(line, bytes) else line


def _sse_payload(line: str | bytes) -> dict[str, Any] | None:
    """Parse one OpenAI-compatible ``data:`` SSE line."""

    text = _as_text(line).strip()
    if not text.startswith("data:"):
        return None
    data = text.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return None
    return json.loads(data)


def stream_one_call(
    *,
    url: str,
    model: str,
    api_key: str | None = None,
    prompt: str,
    max_tokens: int,
    timeout: int,
    request_id: int,
) -> dict[str, Any]:
    """Measure a single streamed request from a vLLM-compatible SSE endpoint.

    TPOT is ``(E2E - TTFT) / (completion_tokens - 1)``. The final vLLM usage
    event is required, therefore token metrics remain ``None`` if a proxy
    strips ``stream_options.include_usage``.
    """

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    start = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    chunks = 0

    try:
        with requests.post(
            url, json=payload, headers=headers, stream=True, timeout=timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = _sse_payload(line)
                if event is None:
                    continue

                usage = event.get("usage") or {}
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")

                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        chunks += 1
                        if first_token_at is None:
                            first_token_at = time.perf_counter()

        finished_at = time.perf_counter()
        ttft = first_token_at - start if first_token_at is not None else None
        latency = finished_at - start
        tpot = (
            (finished_at - first_token_at) / (completion_tokens - 1)
            if first_token_at is not None
            and isinstance(completion_tokens, int)
            and completion_tokens > 1
            else None
        )
        return {
            "request_id": request_id,
            "ok": first_token_at is not None,
            "latency": latency,
            "ttft": ttft,
            "tpot": tpot,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "chunks": chunks,
            "error": "" if first_token_at is not None else "stream finished without content",
        }
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "latency": time.perf_counter() - start,
            "ttft": None,
            "tpot": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "chunks": chunks,
            "error": str(exc),
        }


def run_stream_case(
    *,
    url: str,
    model: str,
    api_key: str | None = None,
    prompt_type: str,
    prompt_mode: PromptMode,
    concurrency: int,
    requests_count: int,
    max_tokens: int,
    timeout: int,
    warmup: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a concurrent streamed workload and return summary plus raw records."""

    for request_id in range(warmup):
        stream_one_call(
            url=url,
            model=model,
            api_key=api_key,
            prompt=build_prompt(prompt_type, -(request_id + 1), prompt_mode),
            max_tokens=max_tokens,
            timeout=timeout,
            request_id=-(request_id + 1),
        )

    wall_start = time.perf_counter()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                stream_one_call,
                url=url,
                model=model,
                api_key=api_key,
                prompt=build_prompt(prompt_type, request_id, prompt_mode),
                max_tokens=max_tokens,
                timeout=timeout,
                request_id=request_id,
            )
            for request_id in range(requests_count)
        ]
        for future in as_completed(futures):
            records.append(future.result())
    wall_time = time.perf_counter() - wall_start

    succeeded = [record for record in records if record["ok"]]
    failed = [record for record in records if not record["ok"]]
    latencies = [record["latency"] for record in succeeded]
    ttfts = [record["ttft"] for record in succeeded if record["ttft"] is not None]
    tpots = [record["tpot"] for record in succeeded if record["tpot"] is not None]
    completion_tokens = [
        record["completion_tokens"]
        for record in succeeded
        if isinstance(record["completion_tokens"], int)
    ]
    prompt_tokens = [
        record["prompt_tokens"]
        for record in succeeded
        if isinstance(record["prompt_tokens"], int)
    ]

    summary = {
        "concurrency": concurrency,
        "requests": requests_count,
        "warmup": warmup,
        "prompt_type": prompt_type,
        "prompt_mode": prompt_mode.value,
        "max_tokens": max_tokens,
        "success": len(succeeded),
        "failed": len(failed),
        "error_rate": len(failed) / requests_count if requests_count else 0.0,
        "wall_time": wall_time,
        "throughput": len(succeeded) / wall_time if wall_time else 0.0,
        "tokens_per_second": sum(completion_tokens) / wall_time if wall_time else 0.0,
        "avg_prompt_tokens": statistics.mean(prompt_tokens) if prompt_tokens else None,
        "avg_completion_tokens": (
            statistics.mean(completion_tokens) if completion_tokens else None
        ),
        "p50_latency": percentile(latencies, 0.50),
        "p95_latency": percentile(latencies, 0.95),
        "p50_ttft": percentile(ttfts, 0.50),
        "p95_ttft": percentile(ttfts, 0.95),
        "p50_tpot": percentile(tpots, 0.50),
        "p95_tpot": percentile(tpots, 0.95),
        "first_error": failed[0]["error"] if failed else "",
        "usage_complete": len(completion_tokens) == len(succeeded),
    }
    return summary, sorted(records, key=lambda record: record["request_id"])
