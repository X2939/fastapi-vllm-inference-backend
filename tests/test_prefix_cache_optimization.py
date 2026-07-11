"""Regression tests for prefix-cache prefill compute reuse."""

import pytest

from engine.executor import Executor
from engine.inference_engine import InferenceEngine
from engine.kv_cache import KVCacheManager
from engine.policy import MaxSeqPolicy
from engine.request import Request


def test_cache_reports_reused_block_aligned_tokens() -> None:
    cache = KVCacheManager(block_size=16, num_blocks=64)
    shared = list(range(32))

    cache.allocate_blocks(0, shared + [100])
    cache.allocate_blocks(1, shared + [200])

    assert cache.get_cached_prompt_tokens(0) == 0
    assert cache.get_cached_prompt_tokens(1) == 32


def test_executor_skips_cached_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = KVCacheManager(block_size=16, num_blocks=64)
    executor = Executor(cache, prefill_cost_per_token=0.01)
    request = Request(
        id=1,
        arrival_time=0.0,
        prompt_length=64,
        max_new_tokens=1,
        cached_prompt_tokens=48,
    )
    sleeps: list[float] = []
    monkeypatch.setattr("engine.executor.time.sleep", sleeps.append)

    elapsed = executor.run_prefill([request])

    assert elapsed == pytest.approx(0.16)
    assert sleeps == [pytest.approx(0.16)]


def test_engine_propagates_cache_hits_to_requests() -> None:
    engine = InferenceEngine(
        block_size=16,
        num_blocks=64,
        prefill_cost_per_token=0,
        decode_cost_per_token=0,
        policy=MaxSeqPolicy(max_num_seqs=2),
        enable_prefix_sharing=True,
    )
    shared = list(range(32))
    engine.add_request(33, 1, shared + [100])
    engine.add_request(33, 1, shared + [200])

    engine.run()

    finished = engine.scheduler.get_finished_requests()
    assert [request.cached_prompt_tokens for request in finished] == [0, 32]
    assert engine.get_results()["cached_prompt_tokens"] == 32
