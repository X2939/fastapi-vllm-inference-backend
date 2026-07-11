"""Legacy compute-only executor retained for focused timing experiments.

The EngineCore path uses Worker + ModelRunner. This class deliberately does
not mutate Request or KV state; lifecycle updates belong to Scheduler.
"""

from __future__ import annotations

import math
import time
from typing import List

from engine.batch import Batch
from engine.kv_cache import KVCacheManager
from engine.outputs import ExecutionTiming
from engine.request import Request

__all__ = ["ExecutionTiming", "Executor"]


class Executor:
    """Compute timing primitive with no scheduler-state side effects."""

    def __init__(
        self,
        kv_cache: KVCacheManager,
        prefill_cost_per_token: float = 0.001,
        decode_cost_per_token: float = 0.0005,
        **_: object,
    ) -> None:
        self.kv_cache = kv_cache
        self.prefill_cost_per_token = prefill_cost_per_token
        self.decode_cost_per_token = decode_cost_per_token
        self._total_tokens_generated = 0

    @property
    def total_tokens_generated(self) -> int:
        return self._total_tokens_generated

    def execute(self, batch: Batch) -> ExecutionTiming:
        """Measure batch compute only; Request objects remain unchanged."""
        started = time.perf_counter()
        prefill_time = self.run_prefill(batch.prefill_requests)
        decode_time = self.run_decode(batch.decode_requests)
        return ExecutionTiming(
            prefill_time=prefill_time,
            decode_time=decode_time,
            total_time=time.perf_counter() - started,
            prefill_count=len(batch.prefill_requests),
            decode_count=len(batch.decode_requests),
            tokens_generated=0,
        )

    def build_prefill_batch(self, batch: Batch) -> List[Request]:
        return list(batch.prefill_requests)

    def run_prefill(self, requests: List[Request]) -> float:
        if not requests:
            return 0.0
        token_work = sum(request.remaining_prefill_tokens for request in requests)
        batch_parallelism = math.pow(len(requests), 0.65)
        compute_time = token_work * self.prefill_cost_per_token / batch_parallelism
        time.sleep(compute_time)
        return compute_time

    def build_decode_batch(self, batch: Batch) -> List[Request]:
        return list(batch.decode_requests)

    def run_decode(self, requests: List[Request]) -> float:
        if not requests:
            return 0.0
        batch_parallelism = math.pow(len(requests), 0.80)
        compute_time = (
            len(requests) * self.decode_cost_per_token / batch_parallelism
        )
        time.sleep(compute_time)
        return compute_time

    def reset(self) -> None:
        self._total_tokens_generated = 0
