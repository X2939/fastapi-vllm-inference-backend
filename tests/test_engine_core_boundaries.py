"""Tests for the vLLM-style EngineCore interface boundaries."""

from __future__ import annotations

import numpy as np

from engine.attention_backend import AttentionMetadata
from engine.kv_cache import KVCacheManager
from engine.model_runner import ModelRunner
from engine.outputs import SchedulerOutput
from engine.policy import MaxSeqPolicy
from engine.request import RequestStatus
from engine.scheduler import Scheduler
from engine.worker import GPUWorker


def _scheduler(*, chunked: bool = False, budget: int = 32) -> Scheduler:
    cache = KVCacheManager(block_size=4, num_blocks=128)
    return Scheduler(
        policy=MaxSeqPolicy(max_num_seqs=8),
        kv_alloc_fn=cache.allocate_blocks,
        kv_append_fn=cache.append_token,
        kv_free_fn=cache.free_request,
        kv_cached_tokens_fn=cache.get_cached_prompt_tokens,
        kv_block_table_fn=cache.get_block_table,
        chunked_prefill=chunked,
        prefill_chunk_size=4,
        max_num_scheduled_tokens=budget,
    )


def test_schedule_emits_new_then_cached_request_data() -> None:
    scheduler = _scheduler(chunked=True)
    request = scheduler.add_request(10, 2, list(range(10)))

    first = scheduler.schedule()
    assert isinstance(first, SchedulerOutput)
    assert first.scheduled_request_ids == [request.id]
    assert first.num_scheduled_tokens == {request.id: 4}
    assert first.total_num_scheduled_tokens == 4
    assert len(first.new_requests) == 1
    assert first.cached_requests == []
    assert first.block_table_updates[request.id]

    runner = ModelRunner(
        prefill_cost_per_token=0,
        decode_cost_per_token=0,
        attention_backend="naive",
    )
    model_output = runner.execute_model(first)

    # The model side cannot mutate scheduler-owned request state.
    assert request.num_computed_tokens == 0
    assert request.generated_tokens == 0
    scheduler.update_from_output(first, model_output)
    assert request.num_computed_tokens == 4

    second = scheduler.schedule()
    assert second.new_requests == []
    assert len(second.cached_requests) == 1


def test_update_from_output_owns_state_transitions_and_finish() -> None:
    scheduler = _scheduler(chunked=True)
    request = scheduler.add_request(6, 1, list(range(6)))
    worker = GPUWorker(
        ModelRunner(
            prefill_cost_per_token=0,
            decode_cost_per_token=0,
            attention_backend="flash",
        )
    )

    first = scheduler.schedule()
    scheduler.update_from_output(first, worker.execute_model(first))
    assert request.status == RequestStatus.PREFILL
    assert request.generated_tokens == 0

    second = scheduler.schedule()
    engine_outputs = scheduler.update_from_output(
        second, worker.execute_model(second)
    )
    assert request.status == RequestStatus.FINISHED
    assert request.generated_tokens == 1
    assert engine_outputs[0].finished is True
    assert engine_outputs[0].finish_reason == "length"
    assert scheduler.is_done


def test_global_token_budget_caps_scheduled_work() -> None:
    scheduler = _scheduler(budget=12)
    scheduler.add_request(10, 2, list(range(10)))
    scheduler.add_request(10, 2, list(range(100, 110)))

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens <= 12
    assert sum(output.num_scheduled_tokens.values()) == output.total_num_scheduled_tokens


def test_attention_backend_is_on_model_runner_path() -> None:
    class SpyBackend:
        name = "spy"

        def __init__(self) -> None:
            self.metadata: AttentionMetadata | None = None

        def forward(self, query, key_cache, value_cache, metadata):
            self.metadata = metadata
            return np.zeros_like(query)

    scheduler = _scheduler()
    scheduler.add_request(4, 1, list(range(4)))
    scheduler_output = scheduler.schedule()
    runner = ModelRunner(
        prefill_cost_per_token=0,
        decode_cost_per_token=0,
    )
    spy = SpyBackend()
    runner.attention_backend = spy

    output = runner.execute_model(scheduler_output)

    assert output.attention_backend == "spy"
    assert spy.metadata is not None
    assert spy.metadata.block_tables == scheduler_output.block_table_updates
