"""Benchmark collector for inference metrics.

Collects per-step timing data and per-request metrics, then computes
comprehensive aggregate statistics. Exports to CSV and JSON.

Metrics:
    - TTFT (Time To First Token): avg, p50, p95, p99, median
    - TPOT (Time Per Output Token): avg, p50, p95, p99, median
    - Latency: avg, p50, p95, p99, median
    - Throughput: total tokens / total time
    - Time breakdown: scheduler, executor, prefill, decode
    - Batch stats: avg, peak
    - Queue stats: avg, peak
    - Admission rate, GPU occupancy
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engine.request import Request

logger = logging.getLogger(__name__)


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Compute the p-th percentile of a sorted list.

    Args:
        sorted_values: Pre-sorted list of values.
        pct: Percentile (0-100).

    Returns:
        The percentile value.
    """
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


@dataclass
class StepTiming:
    """Per-step timing record.

    Attributes:
        step: Step number.
        batch_id: Batch ID.
        sched_time: Time spent in scheduler.
        exec_time: Time spent in executor.
        prefill_time: Time spent in prefill phase.
        decode_time: Time spent in decode phase.
        total_time: Total step duration.
        prefill_count: Prefill requests.
        decode_count: Decode requests.
        batch_size: Total batch size.
        waiting: Waiting queue length.
        running: Running queue length.
        finished: Finished count.
        memory_used: Memory usage.
        gpu_occupancy: GPU occupancy at this step.
        tokens_generated: Total tokens so far.
    """
    step: int
    batch_id: int
    sched_time: float = 0.0
    exec_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0
    prefill_count: int = 0
    decode_count: int = 0
    batch_size: int = 0
    waiting: int = 0
    running: int = 0
    finished: int = 0
    memory_used: int = 0
    gpu_occupancy: float = 0.0
    tokens_generated: int = 0


class BenchmarkCollector:
    """Collects and computes inference benchmark metrics.

    Usage:
        collector = BenchmarkCollector()
        collector.start()
        # ... run engine, call record_step() each step ...
        collector.end()
        collector.process_requests(finished_requests)
        result = collector.compute_stats()
        collector.export_csv("metrics.csv")
        collector.export_summary_json("summary.json")
        collector.export_timeline_csv("timeline.csv")
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._requests: List[Request] = []
        self._step_timings: List[StepTiming] = []

    def start(self) -> None:
        """Mark the start of benchmarking."""
        self._start_time = time.time()

    def end(self) -> None:
        """Mark the end of benchmarking."""
        self._end_time = time.time()

    def record_step(self, step_info: Dict[str, Any]) -> None:
        """Record one step's timing data.

        Args:
            step_info: Step info dictionary from Engine.run_step().
        """
        self._step_timings.append(StepTiming(
            step=step_info.get("step", 0),
            batch_id=step_info.get("batch_id", 0),
            sched_time=step_info.get("sched_time", 0.0),
            exec_time=step_info.get("exec_time", 0.0),
            prefill_time=step_info.get("prefill_time", 0.0),
            decode_time=step_info.get("decode_time", 0.0),
            total_time=step_info.get("step_duration", 0.0),
            prefill_count=step_info.get("prefill", 0),
            decode_count=step_info.get("decode", 0),
            batch_size=step_info.get("batch_size", 0),
            waiting=step_info.get("waiting", 0),
            running=step_info.get("running", 0),
            finished=step_info.get("finished", 0),
            memory_used=step_info.get("memory_used", 0),
            gpu_occupancy=step_info.get("gpu_occupancy", 0.0),
            tokens_generated=step_info.get("total_tokens", 0),
        ))

    def process_requests(self, requests: List[Request]) -> None:
        """Process finished requests for per-request metrics.

        Args:
            requests: List of finished requests.
        """
        self._requests = list(requests)

    def compute_stats(self) -> Dict:
        """Compute all aggregate statistics.

        Returns:
            Dictionary with all computed metrics.
        """
        total_time = (
            (self._end_time - self._start_time)
            if self._start_time and self._end_time else 0.0
        )

        ttfts, tpots, latencies = [], [], []
        total_tokens = 0
        queue_waiting_times = []
        admission_delays = []

        for req in self._requests:
            total_tokens += req.generated_tokens

            if req.prefill_finish_time is not None:
                ttfts.append(req.prefill_finish_time - req.arrival_time)

            if (req.finish_time and req.prefill_finish_time
                    and req.generated_tokens > 1):
                dt = req.finish_time - req.prefill_finish_time
                dk = req.generated_tokens - 1
                if dk > 0:
                    tpots.append(dt / dk)

            if req.finish_time:
                latencies.append(req.finish_time - req.arrival_time)

            if req.admission_time is not None:
                queue_waiting_times.append(req.admission_time - req.arrival_time)
                admission_delays.append(req.admission_time - req.arrival_time)

        ttfts.sort()
        tpots.sort()
        latencies.sort()
        queue_waiting_times.sort()

        # Time breakdowns
        total_sched = sum(s.sched_time for s in self._step_timings)
        total_exec = sum(s.exec_time for s in self._step_timings)
        total_prefill = sum(s.prefill_time for s in self._step_timings)
        total_decode = sum(s.decode_time for s in self._step_timings)

        batch_sizes = [s.batch_size for s in self._step_timings if s.batch_size > 0]
        queue_lengths = [s.waiting for s in self._step_timings]

        return {
            "total_requests": len(self._requests),
            "total_tokens": total_tokens,
            "total_time": total_time,
            "throughput": total_tokens / total_time if total_time > 0 else 0,
            "ttft": {
                "avg": sum(ttfts) / len(ttfts) if ttfts else 0,
                "p50": _percentile(ttfts, 50),
                "p95": _percentile(ttfts, 95),
                "p99": _percentile(ttfts, 99),
                "median": _percentile(ttfts, 50),
            },
            "tpot": {
                "avg": sum(tpots) / len(tpots) if tpots else 0,
                "p50": _percentile(tpots, 50),
                "p95": _percentile(tpots, 95),
                "p99": _percentile(tpots, 99),
                "median": _percentile(tpots, 50),
            },
            "latency": {
                "avg": sum(latencies) / len(latencies) if latencies else 0,
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "p99": _percentile(latencies, 99),
                "median": _percentile(latencies, 50),
            },
            "queue_waiting_time": {
                "avg": sum(queue_waiting_times) / len(queue_waiting_times)
                       if queue_waiting_times else 0,
                "p50": _percentile(queue_waiting_times, 50),
                "p95": _percentile(queue_waiting_times, 95),
                "p99": _percentile(queue_waiting_times, 99),
            },
            "time_breakdown": {
                "scheduler_time": total_sched,
                "executor_time": total_exec,
                "prefill_time": total_prefill,
                "decode_time": total_decode,
                "total_step_time": total_sched + total_exec,
            },
            "batch_stats": {
                "avg": sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0,
                "peak": max(batch_sizes) if batch_sizes else 0,
            },
            "queue_stats": {
                "avg": sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0,
                "peak": max(queue_lengths) if queue_lengths else 0,
            },
        }

    def export_csv(self, filepath: str) -> None:
        """Export per-request metrics to CSV.

        Args:
            filepath: Output CSV path.
        """
        rows = []
        for req in self._requests:
            ttft = (req.prefill_finish_time - req.arrival_time
                    if req.prefill_finish_time else None)
            tpot = None
            if (req.finish_time and req.prefill_finish_time
                    and req.generated_tokens > 1):
                dt = req.finish_time - req.prefill_finish_time
                dk = req.generated_tokens - 1
                if dk > 0:
                    tpot = dt / dk
            latency = (req.finish_time - req.arrival_time
                       if req.finish_time else None)
            queue_wait = (req.admission_time - req.arrival_time
                          if req.admission_time else None)
            rows.append({
                "request_id": req.id,
                "prompt_length": req.prompt_length,
                "max_new_tokens": req.max_new_tokens,
                "generated_tokens": req.generated_tokens,
                "ttft": ttft,
                "tpot": tpot,
                "latency": latency,
                "queue_wait": queue_wait,
                "priority": req.priority,
            })
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        logger.info(f"Exported {len(rows)} request metrics to {filepath}")

    def export_summary_json(self, filepath: str) -> None:
        """Export aggregate statistics to JSON.

        Args:
            filepath: Output JSON path.
        """
        stats = self.compute_stats()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Exported summary to {filepath}")

    def export_timeline_csv(self, filepath: str) -> None:
        """Export per-step timeline data to CSV.

        Args:
            filepath: Output CSV path.
        """
        if not self._step_timings:
            return
        fields = [
            "step", "batch_id", "sched_time", "exec_time",
            "prefill_time", "decode_time", "total_time",
            "prefill_count", "decode_count", "batch_size",
            "waiting", "running", "finished",
            "memory_used", "gpu_occupancy", "tokens_generated",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for t in self._step_timings:
                writer.writerow({k: getattr(t, k) for k in fields})
        logger.info(f"Exported {len(self._step_timings)} step records to {filepath}")

    def print_stats(self, title: str = "Benchmark Results") -> None:
        """Print formatted statistics to logger.

        Args:
            title: Title for the output.
        """
        r = self.compute_stats()
        logger.info(f"=== {title} ===")
        logger.info(f"  Total Requests:  {r['total_requests']}")
        logger.info(f"  Total Tokens:    {r['total_tokens']}")
        logger.info(f"  Total Time:      {r['total_time']:.4f}s")
        logger.info(f"  Throughput:      {r['throughput']:.2f} tok/s")
        logger.info(f"  TTFT  avg/p50/p95/p99: "
                     f"{r['ttft']['avg']:.4f} / {r['ttft']['p50']:.4f} / "
                     f"{r['ttft']['p95']:.4f} / {r['ttft']['p99']:.4f} s")
        logger.info(f"  TPOT  avg/p50/p95/p99: "
                     f"{r['tpot']['avg']:.4f} / {r['tpot']['p50']:.4f} / "
                     f"{r['tpot']['p95']:.4f} / {r['tpot']['p99']:.4f} s")
        logger.info(f"  Lat   avg/p50/p95/p99: "
                     f"{r['latency']['avg']:.4f} / {r['latency']['p50']:.4f} / "
                     f"{r['latency']['p95']:.4f} / {r['latency']['p99']:.4f} s")
        tb = r["time_breakdown"]
        logger.info(f"  Time Breakdown:")
        logger.info(f"    Scheduler: {tb['scheduler_time']:.4f}s")
        logger.info(f"    Executor:  {tb['executor_time']:.4f}s")
        logger.info(f"    Prefill:   {tb['prefill_time']:.4f}s")
        logger.info(f"    Decode:    {tb['decode_time']:.4f}s")
