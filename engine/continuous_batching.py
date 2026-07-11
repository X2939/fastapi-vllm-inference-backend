"""Real Continuous Batching and Static Batching runners.

This module provides two concrete scheduling modes on top of the existing
:class:`InferenceEngine`:

  - :class:`ContinuousBatchingRunner`: requests arrive over time; the engine
    runs one decode step at a time, removes finished requests, and admits new
    requests into the current batch every step. The GPU never waits for an
    entire batch to finish before starting useful work.

  - :class:`StaticBatchingRunner`: requests are grouped into fixed-size batches;
    every request in a batch is prefilled together, then decoded together, and
    the next batch cannot start until all requests in the current batch finish.

Both runners use the same ``Scheduler``, ``Executor`` and ``KVCacheManager``,
so the comparison is fair.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.inference_engine import InferenceEngine
from engine.request import Request, RequestStatus
from engine.policy import BaseAdmissionPolicy, MemoryBudgetPolicy

logger = logging.getLogger(__name__)


@dataclass
class BatchingResult:
    """Result of a batching run.

    Attributes:
        mode: "continuous" or "static".
        total_requests: Number of requests processed.
        total_tokens: Total tokens generated (prompt + output).
        total_time: Simulated wall-clock time of the run (seconds).
        throughput: Tokens per second.
        total_steps: Number of scheduling steps.
        ttft_avg: Average time to first token.
        ttft_p50: P50 TTFT.
        ttft_p95: P95 TTFT.
        latency_avg: Average end-to-end latency.
        latency_p50: P50 latency.
        latency_p95: P95 latency.
        gpu_occupancy: Fraction of steps that processed at least one request.
        avg_batch_size: Average batch size per step.
        peak_batch_size: Largest batch size seen.
        step_records: Per-step records for plotting.
        request_records: Per-request records.
    """

    mode: str
    total_requests: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    throughput: float = 0.0
    total_steps: int = 0
    ttft_avg: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    latency_avg: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    gpu_occupancy: float = 0.0
    avg_batch_size: float = 0.0
    peak_batch_size: int = 0
    step_records: List[Dict] = field(default_factory=list)
    request_records: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding heavy per-step/per-request data)."""
        return {
            "mode": self.mode,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "throughput": self.throughput,
            "total_steps": self.total_steps,
            "ttft_avg": self.ttft_avg,
            "ttft_p50": self.ttft_p50,
            "ttft_p95": self.ttft_p95,
            "latency_avg": self.latency_avg,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "gpu_occupancy": self.gpu_occupancy,
            "avg_batch_size": self.avg_batch_size,
            "peak_batch_size": self.peak_batch_size,
        }


class ContinuousBatchingRunner:
    """Drive :class:`InferenceEngine` in true continuous batching mode.

    Requests are injected into the engine's waiting queue over simulated time.
    Each call to :meth:`InferenceEngine.run_step` performs one scheduling step:
    finished requests are removed, newly arrived requests are admitted, and a
    mixed prefill + decode batch is executed. The loop fast-forwards to the
    next arrival when the GPU would otherwise be idle, so the GPU is never
    waiting for a whole batch to finish.

    This mirrors vLLM's real continuous batching behavior::

        Request Queue -> Scheduler -> Batch -> GPU -> remove done -> repeat
    """

    def __init__(
        self,
        engine: InferenceEngine,
        arrival_times: Optional[Dict[int, float]] = None,
    ):
        """Initialize runner.

        Args:
            engine: The inference engine to drive.
            arrival_times: Optional mapping of request_id -> simulated arrival
                time. If None, all requests are treated as already arrived.
        """
        self.engine = engine
        self.arrival_times = arrival_times or {}
        self._pending_requests: List[Tuple[float, Request]] = []
        self._submitted_ids: set = set()

    def add_request(
        self,
        prompt_length: int,
        max_new_tokens: int,
        arrival_time: float = 0.0,
        prompt_tokens: Optional[List[int]] = None,
        priority: int = 0,
    ) -> Request:
        """Add a request with a simulated arrival time.

        The request is created by the engine but held outside the scheduler
        until ``arrival_time`` is reached.

        Args:
            prompt_length: Prompt length.
            max_new_tokens: Max output tokens.
            arrival_time: Simulation time when the request arrives.
            prompt_tokens: Optional token IDs.
            priority: Scheduling priority (lower = higher).

        Returns:
            The created Request.
        """
        req = self.engine.add_request(
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens,
            priority=priority,
        )
        # Remove from waiting queue; we will inject at arrival_time.
        self.engine.scheduler.waiting_queue.remove(req)
        self._pending_requests.append((arrival_time, req))
        self.arrival_times[req.id] = arrival_time
        return req

    def run(
        self,
        max_steps: int = 100000,
        callback: Optional[callable] = None,
    ) -> BatchingResult:
        """Run continuous batching until all requests finish.

        Args:
            max_steps: Safety step limit.
            callback: Optional per-step callback receiving the step info dict.

        Returns:
            BatchingResult with full statistics.
        """
        start_time = time.time()
        sim_time = 0.0
        step_records: List[Dict] = []

        # Sort pending by arrival time.
        self._pending_requests.sort(key=lambda x: x[0])
        pending_idx = 0

        while (
            pending_idx < len(self._pending_requests)
            or not self.engine.scheduler.is_done
        ) and self.engine.step_count < max_steps:
            # Fast-forward to the next arrival if the GPU is idle.
            # After jumping, fall through to inject requests whose arrival
            # time is now <= sim_time, otherwise we would spin forever.
            if (
                self.engine.scheduler.is_done
                and pending_idx < len(self._pending_requests)
            ):
                sim_time = self._pending_requests[pending_idx][0]

            # Inject newly arrived requests.
            while (
                pending_idx < len(self._pending_requests)
                and self._pending_requests[pending_idx][0] <= sim_time
            ):
                _, req = self._pending_requests[pending_idx]
                self.engine.scheduler.waiting_queue.append(req)
                self._submitted_ids.add(req.id)
                pending_idx += 1

            # Snapshot statuses before this step so we can assign simulated
            # timestamps to prefill/finish events.
            prev_status = {
                r.id: r.status
                for r in (
                    self.engine.scheduler.running_queue
                    + self.engine.scheduler.waiting_queue
                )
            }

            # Run one engine step (schedule + execute + monitor).
            step_info = self.engine.run_step()
            step_end_time = sim_time + step_info["step_duration"]

            # Assign simulated timestamps to first-token and finish events.
            # Override wall-clock timestamps set by the Scheduler because the
            # runner is driving a simulation timeline.
            all_requests = (
                self.engine.scheduler.running_queue
                + self.engine.scheduler.finished_queue
            )
            for req in all_requests:
                prev = prev_status.get(req.id)
                if prev in (RequestStatus.WAITING, RequestStatus.PREFILL) and (
                    req.status in (RequestStatus.DECODE, RequestStatus.FINISHED)
                ):
                    req.prefill_finish_time = step_end_time
                if prev != RequestStatus.FINISHED and req.status == RequestStatus.FINISHED:
                    req.finish_time = step_end_time

            # Eagerly move finished requests out of the running queue so the
            # loop terminates without an extra empty step.
            self.engine.scheduler._free_finished_requests()

            sim_time = step_end_time
            step_records.append(step_info)
            if callback:
                callback(step_info)

        total_time = time.time() - start_time
        logger.info(
            "Continuous batching finished: sim_time=%.3fs wall_time=%.3fs steps=%d",
            sim_time, total_time, self.engine.step_count,
        )
        return self._build_result(sim_time, step_records)

    def _build_result(
        self,
        sim_time: float,
        step_records: List[Dict],
    ) -> BatchingResult:
        """Build BatchingResult from engine state and records."""
        finished = self.engine.scheduler.get_finished_requests()
        result = BatchingResult(mode="continuous")
        result.total_requests = len(finished)
        result.total_time = sim_time
        result.total_steps = self.engine.step_count

        result.total_tokens = sum(
            r.prompt_length + r.generated_tokens for r in finished
        )
        if sim_time > 0:
            result.throughput = result.total_tokens / sim_time

        sizes = [s["batch_size"] for s in step_records]
        if sizes:
            result.avg_batch_size = float(np.mean(sizes))
            result.peak_batch_size = int(max(sizes))

        busy_steps = sum(1 for s in step_records if s["batch_size"] > 0)
        if step_records:
            result.gpu_occupancy = busy_steps / len(step_records)

        ttfts: List[float] = []
        latencies: List[float] = []
        for r in finished:
            arrival = self.arrival_times.get(r.id, 0.0)
            if r.prefill_finish_time is not None:
                ttfts.append(r.prefill_finish_time - arrival)
            if r.finish_time is not None:
                latencies.append(r.finish_time - arrival)
            result.request_records.append({
                "request_id": r.id,
                "arrival_time": arrival,
                "prefill_finish_time": r.prefill_finish_time,
                "finish_time": r.finish_time,
                "prompt_length": r.prompt_length,
                "generated_tokens": r.generated_tokens,
                "ttft": ttfts[-1] if ttfts else None,
                "latency": latencies[-1] if latencies else None,
            })

        result.ttft_avg = float(np.mean(ttfts)) if ttfts else 0.0
        result.ttft_p50 = self._percentile(ttfts, 50)
        result.ttft_p95 = self._percentile(ttfts, 95)
        result.latency_avg = float(np.mean(latencies)) if latencies else 0.0
        result.latency_p50 = self._percentile(latencies, 50)
        result.latency_p95 = self._percentile(latencies, 95)
        result.step_records = step_records

        return result

    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        """Compute percentile using nearest-rank interpolation."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


class StaticBatchingRunner:
    """Run requests in true static batches using the same :class:`InferenceEngine`.

    Static batching characteristics:
      - Requests are grouped into fixed-size batches.
      - All requests in a batch are prefilled and decoded by a dedicated engine.
      - A request that finishes early still occupies a slot until the longest
        request in the batch finishes.
      - The next batch cannot start until the current batch is fully done.

    This models the traditional "batch -> all finish -> batch" behavior that
    continuous batching replaces, while using exactly the same Executor and
    KV Cache implementation as the continuous runner for a fair comparison.
    """

    def __init__(
        self,
        block_size: int = 16,
        num_blocks: int = 1024,
        prefill_cost_per_token: float = 0.001,
        decode_cost_per_token: float = 0.0005,
        policy: Optional[BaseAdmissionPolicy] = None,
        enable_prefix_sharing: bool = True,
    ):
        """Initialize static batching runner.

        Args:
            block_size: KV cache block size.
            num_blocks: Number of physical blocks.
            prefill_cost_per_token: Simulated prefill cost.
            decode_cost_per_token: Simulated decode cost.
            policy: Admission policy.
            enable_prefix_sharing: Enable prefix sharing.
        """
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.prefill_cost = prefill_cost_per_token
        self.decode_cost = decode_cost_per_token
        self.policy = policy or MemoryBudgetPolicy(memory_budget=80000)
        self.enable_prefix_sharing = enable_prefix_sharing

    def run(
        self,
        requests: List[Tuple[int, int, Optional[List[int]], Optional[float]]],
        batch_size: int = 16,
        callback: Optional[callable] = None,
    ) -> BatchingResult:
        """Run static batching on a list of requests.

        Args:
            requests: List of 3-tuples or 4-tuples:
                (prompt_length, max_new_tokens, prompt_tokens)
                (prompt_length, max_new_tokens, prompt_tokens, arrival_time)
                When arrival_time is provided, the runner waits until every
                request in the current batch has arrived before starting it,
                modelling real online static batching.
            batch_size: Fixed batch size.
            callback: Optional per-step callback.

        Returns:
            BatchingResult with statistics.
        """
        start_time = time.time()
        sim_time = 0.0
        step_records: List[Dict] = []
        finished_requests: List[Request] = []
        batch_id = 0

        # Normalize requests to 4-tuples and sort by arrival time if provided.
        has_arrivals = len(requests) > 0 and len(requests[0]) == 4
        normalized: List[Tuple[int, int, Optional[List[int]], float]] = []
        for req in requests:
            if len(req) == 4:
                normalized.append(req)  # type: ignore[arg-type]
            else:
                pl, mt, pt = req  # type: ignore[misc]
                normalized.append((pl, mt, pt, 0.0))
        if has_arrivals:
            normalized.sort(key=lambda x: x[3])

        for batch_start in range(0, len(normalized), batch_size):
            batch_reqs = normalized[batch_start:batch_start + batch_size]
            batch_id += 1

            # Wait until the last request in this batch has arrived.
            batch_arrival = max(at for _, _, _, at in batch_reqs)
            if batch_arrival > sim_time:
                sim_time = batch_arrival

            # Create a fresh engine for each static batch so that no state
            # leaks between batches and the comparison remains fair.
            engine = InferenceEngine(
                block_size=self.block_size,
                num_blocks=self.num_blocks,
                prefill_cost_per_token=self.prefill_cost,
                decode_cost_per_token=self.decode_cost,
                policy=self.policy,
                enable_prefix_sharing=self.enable_prefix_sharing,
                backend="pytorch",
            )

            for prompt_length, max_new_tokens, prompt_tokens, _ in batch_reqs:
                engine.add_request(
                    prompt_length=prompt_length,
                    max_new_tokens=max_new_tokens,
                    prompt_tokens=prompt_tokens,
                )

            # Run the whole batch to completion.
            engine_start = time.time()
            steps = engine.run(callback=callback)
            engine_duration = time.time() - engine_start

            # Convert wall-clock timestamps to the simulated timeline.
            for req in engine.scheduler.get_finished_requests():
                # For requests without explicit arrival times, treat batch start
                # as their arrival time.
                req.arrival_time = batch_arrival if has_arrivals else sim_time
                if req.prefill_finish_time is not None:
                    req.prefill_finish_time = (
                        sim_time + req.prefill_finish_time - engine_start
                    )
                if req.finish_time is not None:
                    req.finish_time = sim_time + req.finish_time - engine_start
                finished_requests.append(req)

            for step in steps:
                step["mode"] = "static"
                step["batch_id"] = batch_id
                step_records.append(step)

            sim_time += engine_duration

        total_time = time.time() - start_time
        logger.info(
            "Static batching finished: sim_time=%.3fs wall_time=%.3fs batches=%d",
            sim_time, total_time, batch_id,
        )
        return self._build_result(sim_time, step_records, finished_requests)

    def _build_result(
        self,
        sim_time: float,
        step_records: List[Dict],
        finished_requests: List[Request],
    ) -> BatchingResult:
        """Build BatchingResult."""
        result = BatchingResult(mode="static")
        result.total_requests = len(finished_requests)
        result.total_time = sim_time
        result.total_tokens = sum(
            r.prompt_length + r.generated_tokens for r in finished_requests
        )
        if sim_time > 0:
            result.throughput = result.total_tokens / sim_time

        sizes = [s["batch_size"] for s in step_records]
        if sizes:
            result.avg_batch_size = float(np.mean(sizes))
            result.peak_batch_size = int(max(sizes))

        busy_steps = sum(1 for s in step_records if s["batch_size"] > 0)
        if step_records:
            result.gpu_occupancy = busy_steps / len(step_records)

        ttfts: List[float] = []
        latencies: List[float] = []
        for r in finished_requests:
            if r.prefill_finish_time is not None:
                ttfts.append(r.prefill_finish_time - r.arrival_time)
            if r.finish_time is not None:
                latencies.append(r.finish_time - r.arrival_time)
            result.request_records.append({
                "request_id": r.id,
                "arrival_time": r.arrival_time,
                "prefill_finish_time": r.prefill_finish_time,
                "finish_time": r.finish_time,
                "prompt_length": r.prompt_length,
                "generated_tokens": r.generated_tokens,
                "ttft": ttfts[-1] if ttfts else None,
                "latency": latencies[-1] if latencies else None,
            })

        result.ttft_avg = float(np.mean(ttfts)) if ttfts else 0.0
        result.ttft_p50 = self._percentile(ttfts, 50)
        result.ttft_p95 = self._percentile(ttfts, 95)
        result.latency_avg = float(np.mean(latencies)) if latencies else 0.0
        result.latency_p50 = self._percentile(latencies, 50)
        result.latency_p95 = self._percentile(latencies, 95)
        result.total_steps = len(step_records)
        result.step_records = step_records

        return result

    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        """Compute percentile using nearest-rank interpolation."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def create_workload(
    num_requests: int,
    seed: int = 42,
    prompt_min: int = 20,
    prompt_max: int = 80,
    output_min: int = 10,
    output_max: int = 30,
    mean_inter_arrival: float = 0.05,
) -> List[Tuple[int, int, Optional[List[int]], float]]:
    """Create a synthetic request workload with random arrival times.

    Args:
        num_requests: Number of requests.
        seed: Random seed.
        prompt_min: Min prompt length.
        prompt_max: Max prompt length.
        output_min: Min output tokens.
        output_max: Max output tokens.
        mean_inter_arrival: Mean time between arrivals (exponential).

    Returns:
        List of (prompt_length, max_new_tokens, prompt_tokens, arrival_time).
    """
    rng = np.random.default_rng(seed)
    workload = []
    arrival_time = 0.0
    for i in range(num_requests):
        prompt_length = int(rng.integers(prompt_min, prompt_max + 1))
        max_new_tokens = int(rng.integers(output_min, output_max + 1))
        prompt_tokens = list(range(1000 + i * 100, 1000 + i * 100 + prompt_length))
        arrival_time += rng.exponential(mean_inter_arrival)
        workload.append(
            (prompt_length, max_new_tokens, prompt_tokens, arrival_time)
        )
    return workload


__all__ = [
    "BatchingResult",
    "ContinuousBatchingRunner",
    "StaticBatchingRunner",
    "create_workload",
]
