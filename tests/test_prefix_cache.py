"""Tests for KV-cache prefix sharing."""

from engine.kv_cache import KVCacheManager


def test_requests_share_complete_prefix_blocks() -> None:
    cache = KVCacheManager(block_size=16, num_blocks=64)
    prefix = list(range(1, 21))

    first = cache.allocate_blocks(0, prefix + [100, 101, 102])
    second = cache.allocate_blocks(1, prefix + [200, 201])

    first_blocks = first.get_all_physical_blocks()
    second_blocks = second.get_all_physical_blocks()
    stats = cache.get_stats()

    assert first_blocks[0] == second_blocks[0]
    assert stats["shared_blocks"] >= 1
    assert stats["prefix_cache_size"] >= 1
    assert stats["allocated_blocks"] + stats["free_blocks"] == 64
