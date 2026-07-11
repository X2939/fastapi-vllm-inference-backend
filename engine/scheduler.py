"""Token-budget scheduler with vLLM V1-style output/update boundaries."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from engine.batch import Batch
from engine.chunked_prefill import ChunkedPrefillHelper
from engine.outputs import (
    CachedRequestData,
    EngineOutput,
    ModelRunnerOutput,
    NewRequestData,
    SchedulerOutput,
)
from engine.policy import BaseAdmissionPolicy, MemoryBudgetPolicy
from engine.request import Request, RequestStatus

logger = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    """Raw scheduler statistics used by monitoring and benchmarks."""

    total_admission_attempts: int = 0
    successful_admissions: int = 0
    total_batches: int = 0
    batch_size_sum: int = 0
    peak_batch_size: int = 0
    queue_length_sum: int = 0
    queue_length_samples: int = 0
    peak_waiting: int = 0
    peak_running: int = 0
    scheduled_tokens: int = 0
    token_budget_exhaustions: int = 0

    @property
    def admission_rate(self) -> float:
        if self.total_admission_attempts == 0:
            return 0.0
        return self.successful_admissions / self.total_admission_attempts

    @property
    def avg_batch_size(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return self.batch_size_sum / self.total_batches

    @property
    def avg_queue_length(self) -> float:
        if self.queue_length_samples == 0:
            return 0.0
        return self.queue_length_sum / self.queue_length_samples


class Scheduler:
    """Manage request state and emit model-runner work descriptions.

    The scheduler owns request/KV lifecycle. The worker/model runner receives
    immutable metadata and returns sampled token IDs; it never mutates Request.
    """

    def __init__(
        self,
        policy: Optional[BaseAdmissionPolicy] = None,
        kv_alloc_fn: Optional[Callable] = None,
        kv_append_fn: Optional[Callable[[int], object]] = None,
        kv_free_fn: Optional[Callable] = None,
        kv_cached_tokens_fn: Optional[Callable[[int], int]] = None,
        kv_block_table_fn: Optional[Callable[[int], object]] = None,
        chunked_prefill: bool = False,
        prefill_chunk_size: int = 128,
        max_num_scheduled_tokens: int = 4096,
    ) -> None:
        if max_num_scheduled_tokens <= 0:
            raise ValueError("max_num_scheduled_tokens must be positive")
        self._policy = policy or MemoryBudgetPolicy(memory_budget=80000)
        self._kv_alloc_fn = kv_alloc_fn
        self._kv_append_fn = kv_append_fn
        self._kv_free_fn = kv_free_fn
        self._kv_cached_tokens_fn = kv_cached_tokens_fn
        self._kv_block_table_fn = kv_block_table_fn
        self._chunked_prefill = chunked_prefill
        self._prefill_chunk_size = prefill_chunk_size
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self._chunk_helper: Optional[ChunkedPrefillHelper] = (
            ChunkedPrefillHelper(chunk_size=prefill_chunk_size)
            if chunked_prefill
            else None
        )

        self.waiting_queue: List[Request] = []
        self.running_queue: List[Request] = []
        self.finished_queue: List[Request] = []
        self._requests: Dict[int, Request] = {}
        self._finished_since_last_schedule: set[int] = set()
        self._stats = SchedulerStats()
        self._batch_counter = 0
        self._next_request_id = 0

    @property
    def policy(self) -> BaseAdmissionPolicy:
        return self._policy

    @property
    def is_done(self) -> bool:
        return not self.waiting_queue and not self.running_queue

    @property
    def batch_counter(self) -> int:
        return self._batch_counter

    @property
    def chunk_helper(self) -> Optional[ChunkedPrefillHelper]:
        return self._chunk_helper

    def add_request(
        self,
        prompt_length: int,
        max_new_tokens: int,
        prompt_tokens: Optional[List[int]] = None,
        priority: int = 0,
    ) -> Request:
        request = Request(
            id=self._next_request_id,
            arrival_time=time.time(),
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            priority=priority,
        )
        request.prompt_tokens = (
            list(prompt_tokens) if prompt_tokens is not None else list(range(prompt_length))
        )
        self._next_request_id += 1
        self._requests[request.id] = request
        self.waiting_queue.append(request)
        return request

    def schedule(self) -> SchedulerOutput:
        """Schedule tokens, not phases, within a per-step token budget."""
        finished_ids = set(self._finished_since_last_schedule)
        self._finished_since_last_schedule.clear()
        newly_admitted = self._admit_new_requests()
        self._batch_counter += 1

        scheduled, num_scheduled_tokens = self._select_scheduled_requests(
            newly_admitted
        )
        if not scheduled:
            output = SchedulerOutput.empty(self._batch_counter, finished_ids)
            self._update_stats(output.batch, 0)
            return output

        prefill_requests = [
            request for request in scheduled if request.status == RequestStatus.PREFILL
        ]
        decode_requests = [
            request for request in scheduled if request.status == RequestStatus.DECODE
        ]
        batch = Batch.from_requests(
            batch_id=self._batch_counter,
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            num_scheduled_tokens=num_scheduled_tokens,
        )

        new_requests: list[NewRequestData] = []
        cached_requests: list[CachedRequestData] = []
        block_tables: dict[int, list[int]] = {}
        for request in scheduled:
            block_ids = self._get_block_ids(request.id)
            block_tables[request.id] = block_ids
            if not request.has_been_scheduled:
                new_requests.append(
                    NewRequestData(
                        req_id=request.id,
                        prompt_token_ids=list(request.prompt_tokens),
                        block_ids=block_ids,
                        num_computed_tokens=request.num_computed_tokens,
                        num_output_tokens=request.num_output_tokens,
                        max_new_tokens=request.max_new_tokens,
                    )
                )
                request.has_been_scheduled = True
            else:
                cached_requests.append(
                    CachedRequestData(
                        req_id=request.id,
                        block_ids=block_ids,
                        num_computed_tokens=request.num_computed_tokens,
                        num_output_tokens=request.num_output_tokens,
                        prompt_length=request.prompt_length,
                        max_new_tokens=request.max_new_tokens,
                    )
                )

        output = SchedulerOutput(
            scheduled_request_ids=[request.id for request in scheduled],
            num_scheduled_tokens=num_scheduled_tokens,
            new_requests=new_requests,
            cached_requests=cached_requests,
            finished_request_ids=finished_ids,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            block_table_updates=block_tables,
            batch=batch,
        )
        self._update_stats(batch, output.total_num_scheduled_tokens)
        return output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> list[EngineOutput]:
        """Apply model output, update states, and own all KV lifecycle changes."""
        engine_outputs: list[EngineOutput] = []
        now = time.time()

        for request_id in scheduler_output.scheduled_request_ids:
            request = self._requests[request_id]
            computed = model_runner_output.num_computed_tokens.get(request_id, 0)
            if computed:
                prefill_computed = min(request.remaining_prefill_tokens, computed)
                request.num_computed_tokens += computed
                if self._chunk_helper is not None and prefill_computed:
                    self._chunk_helper.mark_chunk_done(request, prefill_computed)

            output_index = model_runner_output.req_id_to_index.get(request_id)
            sampled = (
                model_runner_output.sampled_token_ids[output_index]
                if output_index is not None
                else []
            )

            if sampled and request.prefill_finish_time is None:
                request.prefill_finish_time = now
                request.status = RequestStatus.DECODE

            for token_id in sampled:
                request.output_token_ids.append(token_id)
                request.generated_tokens = request.num_output_tokens
                if self._kv_append_fn is not None:
                    self._kv_append_fn(request.id)

            finished = request.is_finished
            if finished:
                self._finish_request(request, now)

            engine_outputs.append(
                EngineOutput(
                    request_id=request_id,
                    token_ids=list(sampled),
                    finished=finished,
                    finish_reason="length" if finished else None,
                )
            )
        return engine_outputs

    def get_state(self) -> Dict:
        return {
            "waiting_size": len(self.waiting_queue),
            "running_size": len(self.running_queue),
            "finished_size": len(self.finished_queue),
            "batch_counter": self._batch_counter,
        }

    def get_stats(self) -> SchedulerStats:
        return self._stats

    def get_finished_requests(self) -> List[Request]:
        return list(self.finished_queue)

    def get_running_requests(self) -> List[Request]:
        return list(self.running_queue)

    def reset(self) -> None:
        self.waiting_queue.clear()
        self.running_queue.clear()
        self.finished_queue.clear()
        self._requests.clear()
        self._finished_since_last_schedule.clear()
        self._stats = SchedulerStats()
        self._batch_counter = 0
        self._next_request_id = 0
        self._policy.reset()
        if self._chunk_helper is not None:
            self._chunk_helper.reset()

    def _admit_new_requests(self) -> List[Request]:
        admitted: List[Request] = []
        indices_to_remove: List[int] = []
        for index, request in enumerate(self.waiting_queue):
            self._stats.total_admission_attempts += 1
            if not self._policy.can_admit(
                request=request,
                current_running=len(self.running_queue),
                current_memory=0,
            ):
                break
            if self._kv_alloc_fn is not None:
                table = self._kv_alloc_fn(request.id, request.prompt_tokens)
                if table is None:
                    continue
                if self._kv_cached_tokens_fn is not None:
                    request.cached_prompt_tokens = self._kv_cached_tokens_fn(request.id)
                    request.num_computed_tokens = request.cached_prompt_tokens
            self._policy.record_admission(request)
            self._stats.successful_admissions += 1
            request.status = RequestStatus.PREFILL
            request.admission_time = time.time()
            if self._chunk_helper is not None:
                self._chunk_helper.register_request(request)
            self.running_queue.append(request)
            indices_to_remove.append(index)
            admitted.append(request)
        for index in reversed(indices_to_remove):
            del self.waiting_queue[index]
        return admitted

    def _select_scheduled_requests(
        self, newly_admitted: List[Request]
    ) -> tuple[List[Request], dict[int, int]]:
        new_ids = {request.id for request in newly_admitted}
        ordered = [r for r in self.running_queue if r.id not in new_ids]
        ordered.extend(newly_admitted)
        budget = self.max_num_scheduled_tokens
        scheduled: list[Request] = []
        token_counts: dict[int, int] = {}

        for request in ordered:
            if budget <= 0:
                self._stats.token_budget_exhaustions += 1
                break
            if request.status == RequestStatus.PREFILL:
                remaining = request.remaining_prefill_tokens
                desired = remaining if remaining > 0 else 1
                if self._chunked_prefill and remaining > 0:
                    desired = min(desired, self._prefill_chunk_size)
                if not self._chunked_prefill and desired > self.max_num_scheduled_tokens:
                    raise ValueError(
                        f"Request {request.id} needs {desired} prefill tokens but "
                        f"the step budget is {self.max_num_scheduled_tokens}; "
                        "enable chunked_prefill or raise the token budget"
                    )
                if not self._chunked_prefill and desired > budget:
                    continue
            else:
                desired = 1
            num_tokens = min(desired, budget)
            if num_tokens <= 0:
                continue
            scheduled.append(request)
            token_counts[request.id] = num_tokens
            budget -= num_tokens
        return scheduled, token_counts

    def _finish_request(self, request: Request, timestamp: float) -> None:
        request.status = RequestStatus.FINISHED
        request.finish_time = timestamp
        if request in self.running_queue:
            self.running_queue.remove(request)
        if request not in self.finished_queue:
            self.finished_queue.append(request)
        self._policy.record_completion(request)
        if self._kv_free_fn is not None:
            self._kv_free_fn(request.id)
        self._finished_since_last_schedule.add(request.id)
        if self._chunk_helper is not None:
            self._chunk_helper.remove_request(request.id)

    def _free_finished_requests(self) -> int:
        """Compatibility hook; normal completion now happens during update."""
        finished = [request for request in self.running_queue if request.is_finished]
        for request in finished:
            self._finish_request(request, time.time())
        return len(finished)

    def _get_block_ids(self, request_id: int) -> list[int]:
        if self._kv_block_table_fn is None:
            return []
        table = self._kv_block_table_fn(request_id)
        if table is None:
            return []
        return list(table.get_all_physical_blocks())

    def _update_stats(self, batch: Batch, scheduled_tokens: int) -> None:
        batch_size = batch.total_requests
        self._stats.total_batches += 1
        self._stats.batch_size_sum += batch_size
        self._stats.peak_batch_size = max(self._stats.peak_batch_size, batch_size)
        self._stats.scheduled_tokens += scheduled_tokens
        waiting = len(self.waiting_queue)
        running = len(self.running_queue)
        self._stats.queue_length_sum += waiting
        self._stats.queue_length_samples += 1
        self._stats.peak_waiting = max(self._stats.peak_waiting, waiting)
        self._stats.peak_running = max(self._stats.peak_running, running)
