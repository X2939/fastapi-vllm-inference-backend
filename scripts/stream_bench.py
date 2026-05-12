import argparse
import json
import time
from typing import Any

import requests


DEFAULT_URL = "http://127.0.0.1:9000/chat/stream"


def run_stream_once(url: str, message: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    payload = {
        "message": message,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    first_chunk_time = None
    chunks = []

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
        "ttft": first_chunk_time,
        "total_time": total_time,
        "chars": len(output_text),
        "chars_per_second": len(output_text) / total_time if total_time > 0 else 0.0,
        "preview": output_text[:120]+ "...",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure streaming latency for /chat/stream.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--message",
        default="请解释 vLLM 流式输出、TTFT 和完整生成耗时之间的区别。",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    result = run_stream_once(
        url=args.url,
        message=args.message,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    print("=== Stream Benchmark ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
