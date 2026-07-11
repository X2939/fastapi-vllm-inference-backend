"""Config-driven experiment runner for AI Infra performance studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from benchmarks.common import PromptMode, run_case
from benchmarks.report import write_csv, write_experiment_report


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid experiment config: {path}")
    return data


def parse_concurrency(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_experiment(config: dict[str, Any], config_path: Path | None = None) -> Path:
    name = config["name"]
    topic = config.get("topic", "general")
    url = config.get("url", "http://127.0.0.1:9000/chat")
    prompt_type = config.get("prompt_type", "medium")
    prompt_mode = PromptMode(config.get("prompt_mode", "unique"))
    concurrency_levels = parse_concurrency(config.get("concurrency", [1, 2, 4]))
    requests_count = int(config.get("requests", 10))
    warmup = int(config.get("warmup", 0))
    max_tokens = int(config.get("max_tokens", 128))
    timeout = int(config.get("timeout", 120))
    output_dir = Path(config.get("output_dir", f"reports/{name}"))

    rows: list[dict[str, Any]] = []
    for concurrency in concurrency_levels:
        row = run_case(
            url=url,
            prompt_type=prompt_type,
            concurrency=concurrency,
            requests_count=requests_count,
            max_tokens=max_tokens,
            timeout=timeout,
            warmup=warmup,
            prompt_mode=prompt_mode,
        )
        rows.append(row)
        print(
            f"[{name}] concurrency={concurrency} p95={row['p95_latency']:.2f}s "
            f"throughput={row['throughput']:.2f} req/s tokens/s={row['tokens_per_second']:.2f}"
        )

    csv_path = output_dir / "results.csv"
    report_path = output_dir / "report.md"
    meta_path = output_dir / "meta.json"

    write_csv(csv_path, rows)
    meta = {
        "name": name,
        "topic": topic,
        "hypothesis": config.get("hypothesis", ""),
        "analysis_guide": config.get("analysis_guide", ""),
        "analysis_notes": config.get("analysis_notes", ""),
        "environment": config.get("environment", {}),
        "url": url,
        "prompt_type": prompt_type,
        "prompt_mode": prompt_mode.value,
        "concurrency_levels": concurrency_levels,
        "requests": requests_count,
        "warmup": warmup,
        "max_tokens": max_tokens,
        "csv_path": str(csv_path),
        "config_path": str(config_path) if config_path else "",
    }
    write_experiment_report(report_path, meta, rows)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_csv={csv_path}")
    print(f"saved_report={report_path}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a YAML-defined AI Infra experiment.")
    parser.add_argument("config", type=Path, help="Path to experiment YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config, config_path=args.config)


if __name__ == "__main__":
    main()
