"""Streaming TTFT benchmark CLI."""

from __future__ import annotations

import argparse
import json

from benchmarks.common import run_stream_once


DEFAULT_URL = "http://127.0.0.1:9000/chat/stream"


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
