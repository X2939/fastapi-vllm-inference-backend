"""KV Cache manager: coordinates block allocation, block tables, and prefix sharing.

This is the public-facing API for KV cache management. It composes
BlockAllocator, BlockManager, and PrefixCache into a unified interface.

The Scheduler and Executor interact with KV cache only through this class.
Internal implementation details (BlockAllocator, BlockManager, PrefixCache)
are hidden from external consumers.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from engine.block_allocator import BlockAllocator
from engine.block_manager import BlockManager, BlockTable
from engine.prefix_cache import PrefixCache

logger = logging.getLogger(__name__)


class KVCacheManager:
    """PagedAttention-style KV cache manager.

    Composes three sub-modules:
        - BlockAllocator: manages the physical block pool
        - BlockManager: manages per-request block tables
        - PrefixCache: manages prefix sharing

    Public API:
        allocate_blocks(request_id, prompt_tokens) -> Optional[BlockTable]
        append_token(request_id) -> Optional[int]
        free_request(request_id) -> int
        get_stats() -> Dict

    Args:
        block_size: Tokens per block.
        num_blocks: Total physical blocks.
        enable_prefix_sharing: Whether to enable prefix sharing.
    """

    def __init__(
        self,
        block_size: int = 16,
        num_blocks: int = 1024,
        enable_prefix_sharing: bool = True,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks

        self._allocator = BlockAllocator(
            block_size=block_size,
            num_blocks=num_blocks,
        )
        self._block_manager = BlockManager(self._allocator)
        self._prefix_cache = PrefixCache(
            block_size=block_size,
            enabled=enable_prefix_sharing,
        )

        self._enable_prefix_sharing = enable_prefix_sharing
        self._peak_shared_blocks: int = 0
        self._cached_tokens_by_request: Dict[int, int] = {}

    @property
    def allocator(self) -> BlockAllocator:
        """Underlying block allocator (for advanced use)."""
        return self._allocator

    @property
    def block_manager(self) -> BlockManager:
        """Underlying block manager (for advanced use)."""
        return self._block_manager

    @property
    def prefix_cache(self) -> PrefixCache:
        """Underlying prefix cache (for advanced use)."""
        return self._prefix_cache

    def allocate_blocks(
        self,
        request_id: int,
        prompt_tokens: List[int],
    ) -> Optional[BlockTable]:
        """Allocate KV cache blocks for a new request.

        If prefix sharing is enabled, first checks for a cached prefix.
        Shared blocks are reused (ref_count++). Remaining tokens get
        fresh blocks.

        Args:
            request_id: Unique request ID.
            prompt_tokens: The prompt token sequence.

        Returns:
            BlockTable for the request, or None if allocation failed.
        """
        # Try prefix sharing first
        shared_blocks: List[int] = []
        shared_token_count = 0

        if self._enable_prefix_sharing:
            shared_blocks, shared_token_count = self._prefix_cache.find_longest_prefix(
                prompt_tokens, self._allocator
            )

        self._cached_tokens_by_request[request_id] = shared_token_count

        # Calculate remaining tokens that need fresh blocks
        remaining_tokens = len(prompt_tokens) - shared_token_count

        if shared_blocks:
            # Create table with shared blocks
            table = self._block_manager.share_blocks(request_id, shared_blocks)
        else:
            # No shared prefix, allocate everything fresh
            table = self._block_manager.allocate_for_request(
                request_id, len(prompt_tokens)
            )
            if table is None:
                self._cached_tokens_by_request.pop(request_id, None)
                return None
            # Cache the prefix
            self._prefix_cache.cache_prefix(
                prompt_tokens, table.get_all_physical_blocks()
            )
            self._update_peak_shared()
            return table

        # Allocate additional blocks for non-prefix tokens
        if remaining_tokens > 0:
            new_ids = self._block_manager.append_additional_blocks(
                request_id, remaining_tokens
            )
            if not new_ids:
                # Failed to allocate, rollback
                self._block_manager.free_request(request_id)
                self._cached_tokens_by_request.pop(request_id, None)
                return None

        # Cache the full prefix
        table = self._block_manager.get_table(request_id)
        if table:
            self._prefix_cache.cache_prefix(
                prompt_tokens, table.get_all_physical_blocks()
            )

        self._update_peak_shared()
        return table

    def get_cached_prompt_tokens(self, request_id: int) -> int:
        """Return the number of prompt tokens reused by prefix caching."""
        return self._cached_tokens_by_request.get(request_id, 0)

    def append_token(self, request_id: int) -> Optional[int]:
        """Append one token to a request's KV cache (decode phase).

        Args:
            request_id: The request ID.

        Returns:
            Physical block ID where token was written, or None on failure.
        """
        return self._block_manager.append_token(request_id)

    def free_request(self, request_id: int) -> int:
        """Free all blocks for a request.

        Args:
            request_id: The request to free.

        Returns:
            Number of blocks fully freed.
        """
        self._cached_tokens_by_request.pop(request_id, None)
        return self._block_manager.free_request(request_id)

    def get_block_table(self, request_id: int) -> Optional[BlockTable]:
        """Get the block table for a request."""
        return self._block_manager.get_table(request_id)

    def _update_peak_shared(self) -> None:
        """Track peak shared blocks count."""
        current = self._allocator.shared_blocks
        if current > self._peak_shared_blocks:
            self._peak_shared_blocks = current

    def get_stats(self) -> Dict:
        """Return comprehensive KV cache statistics."""
        alloc_stats = self._allocator.get_stats()
        cache_stats = self._prefix_cache.get_stats()
        return {
            "total_blocks": alloc_stats["total_blocks"],
            "free_blocks": alloc_stats["free_blocks"],
            "allocated_blocks": alloc_stats["allocated_blocks"],
            "shared_blocks": alloc_stats["shared_blocks"],
            "peak_shared_blocks": self._peak_shared_blocks,
            "peak_allocated": alloc_stats["peak_allocated"],
            "memory_utilization": alloc_stats["utilization"],
            "prefix_cache_size": cache_stats["size"],
            "prefix_cache_hits": cache_stats["hits"],
            "prefix_cache_misses": cache_stats["misses"],
            "prefix_cache_hit_rate": cache_stats["hit_rate"],
        }

    def reset(self) -> None:
        """Reset the entire KV cache."""
        self._allocator.reset()
        self._prefix_cache.clear()
        self._cached_tokens_by_request.clear()
        self._peak_shared_blocks = 0
