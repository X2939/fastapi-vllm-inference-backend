"""Experiment artifact writers: CSV and Markdown reports."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "concurrency",
    "requests",
    "warmup",
    "prompt_type",
    "prompt_mode",
    "max_tokens",
    "success",
    "failed",
    "error_rate",
    "wall_time",
    "avg_latency",
    "min_latency",
    "max_latency",
    "p50_latency",
    "p95_latency",
    "p99_latency",
    "throughput",
    "total_prompt_tokens",
    "avg_prompt_tokens",
    "total_completion_tokens",
    "avg_completion_tokens",
    "tokens_per_second",
    "tpot",
    "first_error",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Concurrency",
        "P50 (s)",
        "P95 (s)",
        "P99 (s)",
        "Throughput",
        "Tokens/s",
        "TPOT (s)",
        "Error rate",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:" for _ in headers]) + "|",
    ]
    for row in rows:
        lines.append(
            "| {concurrency} | {p50_latency:.2f} | {p95_latency:.2f} | {p99_latency:.2f} | "
            "{throughput:.2f} | {tokens_per_second:.2f} | {tpot:.4f} | {error_rate:.2%} |".format(
                **row
            )
        )
    return "\n".join(lines)


def write_experiment_report(
    path: Path,
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    topic = meta.get("topic", "general")
    analysis_notes = meta.get("analysis_notes", "").strip()
    analysis = meta.get("analysis_guide", "").strip()
    hypothesis = meta.get("hypothesis", "").strip()
    environment = meta.get("environment") or {}

    env_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in environment.items())
    if not env_lines:
        env_lines = "- 未记录"

    content = f"""# {meta.get("name", "experiment")}

> Generated at {generated_at}

## 实验主题

- **Topic**: `{topic}`
- **Hypothesis**: {hypothesis or "未填写"}

## 环境

{env_lines}

## 变量

- URL: `{meta.get("url", "")}`
- Prompt type: `{meta.get("prompt_type", "")}`
- Prompt mode: `{meta.get("prompt_mode", "")}`
- Concurrency levels: `{meta.get("concurrency_levels", "")}`
- Requests per level: `{meta.get("requests", "")}`
- Warmup per level: `{meta.get("warmup", "")}`
- Max tokens: `{meta.get("max_tokens", "")}`

## 结果

{_format_table(rows)}

## 分析指引

{analysis or "对比不同 concurrency 或 prompt_mode 下的 P95 与 throughput 变化。"}

## 结果说明

{analysis_notes or "结合实验数据解释 scheduler batching 或 KV Cache prefill/decode 差异。"}

## 原始数据

- CSV: `{meta.get("csv_path", "")}`
"""
    path.write_text(content, encoding="utf-8")
