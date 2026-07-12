"""Regenerate real GPU benchmark plots from existing CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.gpu_report import write_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GPU benchmark PNGs from CSV files.")
    parser.add_argument("--input-dir", default="reports/gpu_benchmark")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    with (input_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    with (input_dir / "requests.csv").open(encoding="utf-8", newline="") as handle:
        requests = list(csv.DictReader(handle))
    paths = write_plots(summaries, requests, input_dir)
    print("saved_plots=" + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
