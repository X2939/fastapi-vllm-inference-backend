"""Compare two quality smoke result directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read(directory: Path) -> tuple[dict, dict[str, dict]]:
    with (directory / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (directory / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    return metadata, rows


def _is_true(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def _rate(rows: dict[str, dict], field: str) -> float:
    return sum(1 for row in rows.values() if _is_true(row[field])) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BF16 and AWQ quality smoke reports.")
    parser.add_argument("--bf16", default="reports/quality_smoke_bf16")
    parser.add_argument("--awq", default="reports/quality_smoke_awq")
    parser.add_argument("--output-dir", default="reports/quality_smoke_comparison")
    args = parser.parse_args()

    bf16_meta, bf16 = _read(Path(args.bf16))
    awq_meta, awq = _read(Path(args.awq))
    if set(bf16) != set(awq):
        parser.error("quality case IDs differ")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BF16 vs AWQ Quality Smoke Comparison",
        "",
        "> This is a fixed prompt regression smoke test, not a formal accuracy benchmark.",
        "",
        "## Runs",
        "",
        f"- BF16: `{bf16_meta.get('model')}`",
        f"- AWQ: `{awq_meta.get('model')}`",
        "",
        "## Summary",
        "",
        "| Metric | BF16 | AWQ |",
        "|---|---:|---:|",
        f"| Overall pass | {_rate(bf16, 'ok'):.1%} | {_rate(awq, 'ok'):.1%} |",
        f"| JSON parse pass | {_rate(bf16, 'json_ok'):.1%} | {_rate(awq, 'json_ok'):.1%} |",
        f"| Expected substring pass | {_rate(bf16, 'expected_ok'):.1%} | {_rate(awq, 'expected_ok'):.1%} |",
        "",
        "## Case Diff",
        "",
        "| ID | BF16 | AWQ |",
        "|---|---:|---:|",
    ]
    for case_id in sorted(bf16):
        lines.append(
            f"| {case_id} | {'pass' if _is_true(bf16[case_id]['ok']) else 'fail'} | "
            f"{'pass' if _is_true(awq[case_id]['ok']) else 'fail'} |"
        )
    report = output_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_report={report}")


if __name__ == "__main__":
    main()
