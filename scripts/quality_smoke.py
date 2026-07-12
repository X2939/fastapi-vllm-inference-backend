"""Run a small deterministic quality smoke test against a vLLM endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gpu_benchmark import _environment_snapshot


FIELDS = [
    "id",
    "category",
    "ok",
    "json_ok",
    "expected_ok",
    "latency",
    "prompt_tokens",
    "completion_tokens",
    "response",
    "error",
]


def _read_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            case.setdefault("expected_contains", [])
            case.setdefault("require_json", False)
            case.setdefault("category", "general")
            if not case.get("id") or not case.get("prompt"):
                raise ValueError(f"{path}:{line_number} must contain id and prompt")
            cases.append(case)
    if not cases:
        raise ValueError(f"{path} does not contain any cases")
    return cases


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return str(content).strip()


def _parse_json(text: str) -> tuple[bool, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        return True, json.loads(cleaned)
    except json.JSONDecodeError:
        return False, None


def _expected_ok(text: str, expected: list[str]) -> bool:
    lowered = text.lower()
    return all(item.lower() in lowered for item in expected)


def _call_case(
    *,
    url: str,
    model: str,
    api_key: str | None,
    case: dict[str, Any],
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluator target. Return only compact JSON when requested.",
            },
            {"role": "user", "content": case["prompt"]},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        latency = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        text = _extract_text(data)
        json_ok, _parsed = _parse_json(text)
        expected_ok = _expected_ok(text, list(case["expected_contains"]))
        usage = data.get("usage") or {}
        ok = (json_ok or not case["require_json"]) and expected_ok
        return {
            "id": case["id"],
            "category": case["category"],
            "ok": ok,
            "json_ok": json_ok,
            "expected_ok": expected_ok,
            "latency": latency,
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "response": text,
            "error": "",
        }
    except (requests.RequestException, ValueError) as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "ok": False,
            "json_ok": False,
            "expected_ok": False,
            "latency": time.perf_counter() - started,
            "prompt_tokens": "",
            "completion_tokens": "",
            "response": "",
            "error": str(exc),
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    passed = sum(1 for row in rows if row["ok"] is True)
    json_passed = sum(1 for row in rows if row["json_ok"] is True)
    expected_passed = sum(1 for row in rows if row["expected_ok"] is True)
    avg_latency = sum(float(row["latency"]) for row in rows) / total if total else 0.0
    lines = [
        "# Quality Smoke Report",
        "",
        "> This is a small regression smoke test for endpoint sanity, not a formal accuracy benchmark.",
        "",
        "## Run",
        "",
        f"- Label: `{metadata['label']}`",
        f"- Model: `{metadata['model']}`",
        f"- Cases: `{total}`",
        f"- Max tokens: `{metadata['max_tokens']}`",
        "",
        "## Summary",
        "",
        f"- Overall pass: `{passed}/{total}` ({passed / total:.1%})",
        f"- JSON parse pass: `{json_passed}/{total}` ({json_passed / total:.1%})",
        f"- Expected substring pass: `{expected_passed}/{total}` ({expected_passed / total:.1%})",
        f"- Average latency: `{avg_latency:.3f}s`",
        "",
        "## Cases",
        "",
        "| ID | Category | OK | JSON | Expected | Latency (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {category} | {ok} | {json_ok} | {expected_ok} | {latency:.3f} |".format(
                id=row["id"],
                category=row["category"],
                ok="yes" if row["ok"] else "no",
                json_ok="yes" if row["json_ok"] else "no",
                expected_ok="yes" if row["expected_ok"] else "no",
                latency=float(row["latency"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed quality smoke set.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", os.getenv("API_KEY", "")))
    parser.add_argument("--prompt-file", default="benchmarks/quality_prompts.jsonl")
    parser.add_argument("--output-dir", default="reports/quality_smoke")
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required (or set MODEL_NAME)")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    cases = _read_cases(Path(args.prompt_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _call_case(
            url=args.url,
            model=args.model,
            api_key=args.api_key or None,
            case=case,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        for case in cases
    ]
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "model": args.model,
        "url": args.url,
        "prompt_file": args.prompt_file,
        "max_tokens": args.max_tokens,
        "cases": len(cases),
        "environment": _environment_snapshot(),
    }
    _write_csv(output_dir / "results.csv", rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "REPORT.md", metadata, rows)
    passed = sum(1 for row in rows if row["ok"] is True)
    print(f"quality_pass={passed}/{len(rows)}")
    print(f"saved_results={output_dir}")


if __name__ == "__main__":
    main()
