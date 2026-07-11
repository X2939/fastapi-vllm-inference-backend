"""One-command, reproducible benchmark suite for the inference engine.

The suite exercises the project's real Scheduler, Executor and KV-cache code.
It intentionally uses a fixed-workload simulation backend so it can run on a
CPU-only laptop without downloading a model. Results are not GPU claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.inference_engine import InferenceEngine
from engine.kv_cache import KVCacheManager
from engine.policy import MaxSeqPolicy


PREFILL_COST = 0.00008
DECODE_COST = 0.00012


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _run_engine(
    requests: list[tuple[int, int, list[int]]],
    *,
    max_num_seqs: int,
    prefix_caching: bool = False,
    chunk_size: int = 0,
) -> tuple[dict[str, Any], list[Any]]:
    engine = InferenceEngine(
        block_size=16,
        num_blocks=4096,
        prefill_cost_per_token=PREFILL_COST,
        decode_cost_per_token=DECODE_COST,
        policy=MaxSeqPolicy(max_num_seqs=max_num_seqs),
        enable_prefix_sharing=prefix_caching,
        chunked_prefill=chunk_size > 0,
        prefill_chunk_size=chunk_size or 128,
    )
    for prompt_length, output_tokens, prompt_tokens in requests:
        engine.add_request(
            prompt_length=prompt_length,
            max_new_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
        )

    started = time.perf_counter()
    engine.run()
    wall_time = time.perf_counter() - started
    finished = engine.scheduler.get_finished_requests()
    result = engine.get_results()
    output_tokens = sum(request.generated_tokens for request in finished)
    result["wall_time"] = wall_time
    result["output_throughput"] = output_tokens / wall_time
    result["cached_prompt_tokens"] = sum(
        request.cached_prompt_tokens for request in finished
    )
    result["peak_allocated_blocks"] = engine.kv_cache.get_stats()["peak_allocated"]
    return result, finished


def experiment_scheduler() -> list[dict[str, Any]]:
    requests = []
    for request_id in range(16):
        prompt_length = 80 + (request_id % 4) * 16
        tokens = list(range(request_id * 1000, request_id * 1000 + prompt_length))
        requests.append((prompt_length, 24, tokens))

    rows = []
    for concurrency in (1, 2, 4, 8):
        result, _ = _run_engine(
            requests,
            max_num_seqs=concurrency,
            prefix_caching=False,
        )
        rows.append(
            _metric_row(
                "scheduler_concurrency",
                f"concurrency_{concurrency}",
                concurrency,
                result,
            )
        )
    return rows


def experiment_prefix_cache() -> list[dict[str, Any]]:
    shared_prefix = list(range(128))
    requests = []
    for request_id in range(16):
        suffix = list(range(10_000 + request_id * 32, 10_032 + request_id * 32))
        tokens = shared_prefix + suffix
        requests.append((len(tokens), 16, tokens))

    rows = []
    for enabled in (False, True):
        result, _ = _run_engine(
            requests,
            max_num_seqs=16,
            prefix_caching=enabled,
        )
        row = _metric_row(
            "prefix_cache_optimization",
            "optimized_on" if enabled else "baseline_off",
            int(enabled),
            result,
        )
        row["cached_prompt_tokens"] = result["cached_prompt_tokens"]
        row["cache_hit_rate"] = round(result["kv_prefix_cache_hit_rate"], 4)
        row["peak_shared_blocks"] = result["kv_peak_shared_blocks"]
        rows.append(row)
    return rows


def experiment_chunked_prefill() -> list[dict[str, Any]]:
    # One long prompt competes with short interactive requests. Chunking lets
    # short requests receive a first token before the long prefill completes.
    lengths = [1536, 64, 72, 80, 88, 96, 104, 112]
    requests = [
        (length, 16, list(range(index * 10_000, index * 10_000 + length)))
        for index, length in enumerate(lengths)
    ]

    rows = []
    for chunk_size in (0, 128):
        result, finished = _run_engine(
            requests,
            max_num_seqs=8,
            prefix_caching=False,
            chunk_size=chunk_size,
        )
        short_ttfts = [
            request.prefill_finish_time - request.arrival_time
            for request in finished
            if request.prompt_length < 256 and request.prefill_finish_time is not None
        ]
        row = _metric_row(
            "chunked_prefill",
            "chunk_128" if chunk_size else "standard",
            chunk_size,
            result,
        )
        row["short_ttft_ms"] = round(
            sum(short_ttfts) / len(short_ttfts) * 1000, 3
        )
        row["short_ttft_p95_ms"] = round(
            _percentile(short_ttfts, 95) * 1000, 3
        )
        rows.append(row)
    return rows


def experiment_paged_attention() -> list[dict[str, Any]]:
    prompt_lengths = [31, 47, 63, 78, 95, 111, 127, 143, 159, 175, 191, 207]
    rows = []
    for block_size in (8, 16, 32, 64):
        cache = KVCacheManager(
            block_size=block_size,
            num_blocks=4096,
            enable_prefix_sharing=False,
        )
        for request_id, length in enumerate(prompt_lengths):
            tokens = list(range(request_id * 1000, request_id * 1000 + length))
            table = cache.allocate_blocks(request_id, tokens)
            if table is None:
                raise RuntimeError("Unexpected KV-cache allocation failure")

        stats = cache.get_stats()
        capacity_tokens = stats["allocated_blocks"] * block_size
        requested_tokens = sum(prompt_lengths)
        wasted_tokens = capacity_tokens - requested_tokens
        rows.append(
            {
                "experiment": "paged_attention_block_size",
                "variant": f"block_{block_size}",
                "x_value": block_size,
                "allocated_blocks": stats["allocated_blocks"],
                "requested_tokens": requested_tokens,
                "capacity_tokens": capacity_tokens,
                "wasted_tokens": wasted_tokens,
                "fragmentation_pct": round(wasted_tokens / capacity_tokens * 100, 3),
            }
        )
    return rows


def _metric_row(
    experiment: str,
    variant: str,
    x_value: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "variant": variant,
        "x_value": x_value,
        "throughput_tok_s": round(result["output_throughput"], 3),
        "ttft_ms": round(result["ttft"]["avg"] * 1000, 3),
        "ttft_p95_ms": round(result["ttft"]["p95"] * 1000, 3),
        "tpot_ms": round(result["tpot"]["avg"] * 1000, 3),
        "tpot_p95_ms": round(result["tpot"]["p95"] * 1000, 3),
        "latency_p95_ms": round(result["latency"]["p95"] * 1000, 3),
        "avg_batch_size": round(result["avg_batch_size"], 3),
        "peak_batch_size": result["peak_batch_size"],
    }


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_scheduler(path: Path, rows: list[dict[str, Any]]) -> None:
    x = [row["x_value"] for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    latency_axis = axis.twinx()
    axis.plot(x, [row["throughput_tok_s"] for row in rows], "o-", label="Throughput")
    latency_axis.plot(x, [row["latency_p95_ms"] for row in rows], "s--", color="#d95f02", label="P95 latency")
    axis.set(xlabel="Max concurrent sequences", ylabel="Output throughput (tok/s)")
    latency_axis.set_ylabel("P95 latency (ms)")
    axis.grid(alpha=0.25)
    lines = axis.lines + latency_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_prefix(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = ["Baseline", "Prefix cache"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    metrics = [
        ("throughput_tok_s", "Output throughput", "tok/s"),
        ("ttft_ms", "Average TTFT", "ms"),
        ("latency_p95_ms", "P95 latency", "ms"),
    ]
    for axis, (key, title, unit) in zip(axes, metrics):
        values = [row[key] for row in rows]
        bars = axis.bar(labels, values, color=["#7f7f7f", "#1b9e77"])
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.1f")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_chunked(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = ["Standard", "Chunked 128"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis, key, title, unit in (
        (axes[0], "short_ttft_ms", "Interactive-request TTFT", "ms"),
        (axes[1], "throughput_tok_s", "Output throughput", "tok/s"),
    ):
        bars = axis.bar(labels, [row[key] for row in rows], color=["#7570b3", "#e7298a"])
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.1f")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_paged(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(
        [str(row["x_value"]) for row in rows],
        [row["fragmentation_pct"] for row in rows],
        color="#66a61e",
    )
    axis.set(xlabel="KV block size (tokens)", ylabel="Internal fragmentation (%)")
    axis.set_title("PagedAttention block-size trade-off")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_report(path: Path, groups: dict[str, list[dict[str, Any]]]) -> None:
    scheduler = groups["scheduler"]
    prefix = groups["prefix"]
    chunked = groups["chunked"]
    paged = groups["paged"]
    before, after = prefix
    throughput_gain = (after["throughput_tok_s"] / before["throughput_tok_s"] - 1) * 100
    ttft_drop = (1 - after["ttft_ms"] / before["ttft_ms"]) * 100
    p95_drop = (1 - after["latency_p95_ms"] / before["latency_p95_ms"]) * 100
    short_drop = (1 - chunked[1]["short_ttft_ms"] / chunked[0]["short_ttft_ms"]) * 100

    content = f"""# Benchmark Suite Report

Generated by `python3 -m benchmarks.runner`. Backend: fixed-workload CPU simulation.
These numbers validate project behavior; they are not claims about a production GPU.

## 1. Scheduler concurrency sweep

Concurrency 1 -> 8 changed output throughput from {scheduler[0]['throughput_tok_s']:.1f} to {scheduler[-1]['throughput_tok_s']:.1f} tok/s, while P95 latency changed from {scheduler[0]['latency_p95_ms']:.1f} to {scheduler[-1]['latency_p95_ms']:.1f} ms. In this unsaturated range, larger batches improve GPU-style parallelism and drain the queue faster, so both metrics improve. On a real GPU, P95 normally rises after compute or memory bandwidth saturates.

![Scheduler concurrency](scheduler_concurrency.png)

## 2. Real optimization: prefix-cache compute reuse

The optimized path records block-aligned cache hits in `SchedulerOutput` and lets `ModelRunner` skip those prefill tokens. It reused {after['cached_prompt_tokens']} prompt tokens. Throughput changed by {throughput_gain:+.1f}%, average TTFT by {-ttft_drop:+.1f}%, and P95 latency by {-p95_drop:+.1f}%.

![Prefix cache before and after](prefix_cache_optimization.png)

## 3. Chunked prefill

Splitting the long prompt into 128-token chunks changed short-request TTFT by {-short_drop:+.1f}%. The reason is interleaving: short prefills can finish between long-prompt chunks instead of waiting for one monolithic prefill. Extra scheduling steps may slightly reduce total throughput.

![Chunked prefill](chunked_prefill.png)

## 4. PagedAttention block size

The smallest tested block size had {paged[0]['fragmentation_pct']:.1f}% internal fragmentation; the largest had {paged[-1]['fragmentation_pct']:.1f}%. Smaller blocks waste less KV memory and raise concurrency, but increase block-table and allocator metadata overhead.

![PagedAttention block size](paged_attention_blocks.png)

## Metric interpretation

- **Throughput**: completed output tokens / wall time. Batching typically raises it.
- **TTFT**: arrival to first output token. Prefill compute, queueing, prefix reuse and chunking dominate it.
- **TPOT**: average time between output tokens after the first. Decode batching and memory bandwidth dominate it.
- **P95**: 95% of requests are no slower than this value. It exposes queueing and long-tail prompts hidden by averages.
"""
    path.write_text(content, encoding="utf-8")


def run_suite(output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "scheduler": experiment_scheduler(),
        "prefix": experiment_prefix_cache(),
        "chunked": experiment_chunked_prefill(),
        "paged": experiment_paged_attention(),
    }
    rows = [row for group in groups.values() for row in group]
    _save_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "backend": "fixed-workload CPU simulation",
                    "prefill_cost_per_token": PREFILL_COST,
                    "decode_cost_per_token": DECODE_COST,
                },
                "experiments": groups,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_scheduler(output_dir / "scheduler_concurrency.png", groups["scheduler"])
    _plot_prefix(output_dir / "prefix_cache_optimization.png", groups["prefix"])
    _plot_chunked(output_dir / "chunked_prefill.png", groups["chunked"])
    _plot_paged(output_dir / "paged_attention_blocks.png", groups["paged"])
    _write_report(output_dir / "REPORT.md", groups)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete benchmark suite.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/benchmark_suite"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    rows = run_suite(args.output_dir)
    elapsed = time.perf_counter() - started

    print(f"Completed {len(set(row['experiment'] for row in rows))} experiments in {elapsed:.2f}s")
    print(f"Report: {args.output_dir / 'REPORT.md'}")
    print(f"Data:   {args.output_dir / 'summary.csv'}")
    print(f"Plots:  {len(list(args.output_dir.glob('*.png')))} PNG files")


if __name__ == "__main__":
    main()
