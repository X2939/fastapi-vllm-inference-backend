"""GPU utilization and performance monitor.

Tracks comprehensive GPU metrics including busy/idle time, memory usage,
batch sizes, queue lengths, and throughput. The Engine calls record_step()
after each step to feed data to the monitor.

Design:
    - Passive collector: does not influence engine behavior
    - Records per-step data for time-series analysis
    - Computes aggregate statistics on demand
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.batch import Batch

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Per-step GPU record.

    Attributes:
        step: Step number.
        batch_id: Batch ID.
        busy: Whether GPU was busy (non-empty batch).
        duration: Step duration in seconds.
        prefill_count: Prefill requests in this step.
        decode_count: Decode requests in this step.
        batch_size: Total requests in this step.
        memory_used: Memory usage at this step.
        tokens_generated: Tokens generated this step.
    """
    step: int
    batch_id: int
    busy: bool
    duration: float
    prefill_count: int
    decode_count: int
    batch_size: int
    memory_used: int
    tokens_generated: int


@dataclass
class GPUStats:
    """Aggregated GPU statistics.

    Attributes:
        busy_time: Total time GPU was busy.
        idle_time: Total time GPU was idle.
        total_tokens: Total tokens generated.
        peak_memory: Peak memory usage.
        peak_batch_size: Peak batch size.
        memory_samples: List of memory usage samples.
        batch_sizes: List of batch sizes.
        queue_lengths: List of waiting queue lengths.
        running_lengths: List of running queue lengths.
        step_records: Per-step records for time-series.
    """
    busy_time: float = 0.0
    idle_time: float = 0.0
    total_tokens: int = 0
    peak_memory: int = 0
    peak_batch_size: int = 0
    peak_queue: int = 0
    peak_running: int = 0
    memory_samples: List[int] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    queue_lengths: List[int] = field(default_factory=list)
    running_lengths: List[int] = field(default_factory=list)
    step_records: List[StepRecord] = field(default_factory=list)

    @property
    def total_time(self) -> float:
        """Total elapsed time."""
        return self.busy_time + self.idle_time

    @property
    def occupancy(self) -> float:
        """GPU occupancy (busy / total)."""
        if self.total_time == 0:
            return 0.0
        return self.busy_time / self.total_time

    @property
    def avg_memory(self) -> float:
        """Average memory usage."""
        if not self.memory_samples:
            return 0.0
        return sum(self.memory_samples) / len(self.memory_samples)

    @property
    def avg_batch_size(self) -> float:
        """Average batch size."""
        if not self.batch_sizes:
            return 0.0
        return sum(self.batch_sizes) / len(self.batch_sizes)

    @property
    def avg_queue_length(self) -> float:
        """Average waiting queue length."""
        if not self.queue_lengths:
            return 0.0
        return sum(self.queue_lengths) / len(self.queue_lengths)

    @property
    def avg_running_length(self) -> float:
        """Average running queue length."""
        if not self.running_lengths:
            return 0.0
        return sum(self.running_lengths) / len(self.running_lengths)

    @property
    def tokens_per_second(self) -> float:
        """Token throughput."""
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time

    @property
    def requests_per_second(self) -> float:
        """Request throughput (steps with work / total time)."""
        if self.total_time == 0:
            return 0.0
        busy_steps = sum(1 for r in self.step_records if r.busy)
        return busy_steps / self.total_time


class GPUMonitor:
    """Monitors GPU utilization and resource usage.

    The Engine calls record_step() after each step. The monitor
    collects per-step data and computes aggregate statistics.
    """

    def __init__(self):
        self._stats = GPUStats()

    def record_step(
        self,
        step: int,
        batch: Batch,
        duration: float,
        memory_usage: int,
        queue_length: int = 0,
        running_length: int = 0,
    ) -> None:
        """Record one execution step.

        Args:
            step: Step number.
            batch: The batch that was executed.
            duration: Wall-clock duration of this step.
            memory_usage: Current memory usage.
            queue_length: Waiting queue length at this step.
            running_length: Running queue length at this step.
        """
        is_busy = not batch.is_empty
        batch_size = batch.total_requests
        tokens = batch.total_tokens

        if is_busy:
            self._stats.busy_time += duration
            self._stats.total_tokens += tokens
            self._stats.batch_sizes.append(batch_size)
        else:
            self._stats.idle_time += duration

        self._stats.memory_samples.append(memory_usage)
        self._stats.queue_lengths.append(queue_length)
        self._stats.running_lengths.append(running_length)

        if memory_usage > self._stats.peak_memory:
            self._stats.peak_memory = memory_usage
        if batch_size > self._stats.peak_batch_size:
            self._stats.peak_batch_size = batch_size
        if queue_length > self._stats.peak_queue:
            self._stats.peak_queue = queue_length
        if running_length > self._stats.peak_running:
            self._stats.peak_running = running_length

        self._stats.step_records.append(StepRecord(
            step=step,
            batch_id=batch.batch_id,
            busy=is_busy,
            duration=duration,
            prefill_count=len(batch.prefill_requests),
            decode_count=len(batch.decode_requests),
            batch_size=batch_size,
            memory_used=memory_usage,
            tokens_generated=tokens,
        ))

    def get_stats(self) -> GPUStats:
        """Return aggregated statistics."""
        return self._stats

    def get_step_records(self) -> List[StepRecord]:
        """Return per-step records (for time-series plots)."""
        return list(self._stats.step_records)

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats = GPUStats()
