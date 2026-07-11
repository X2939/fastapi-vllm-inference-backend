"""Prefix cache for KV cache prefix sharing.

Maintains a mapping from token sequences (prefixes) to physical block
IDs. When a new request's prompt shares a prefix with a cached entry,
the corresponding blocks can be shared instead of recomputed.

Design:
    - Keys are token tuples (hashable).
    - Values are lists of physical block IDs.
    - Only full-block-aligned prefixes are cached (vLLM behavior).
    - Stale entries (freed blocks) are lazily removed.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from engine.block_allocator import BlockAllocator

logger = logging.getLogger(__name__)


class PrefixCache:
    """Cache mapping token prefixes to physical block IDs.

    Responsible for:
        - Storing and looking up prefix -> block mappings
        - Finding the longest matching prefix for a new request
        - Caching new prefixes after allocation

    Not responsible for:
        - Actually sharing blocks (BlockManager does this)
        - Managing block lifecycles (BlockAllocator does this)
    """

    def __init__(self, block_size: int, enabled: bool = True):
        """Initialize the prefix cache.

        Args:
            block_size: Block size (tokens per block). Prefixes must be
                aligned to this boundary.
            enabled: Whether prefix sharing is enabled.
        """
        self.block_size = block_size
        self.enabled = enabled
        self._cache: Dict[Tuple[int, ...], List[int]] = {}
        self._hits: int = 0
        self._misses: int = 0

    @property
    def size(self) -> int:
        """Number of cached prefix entries."""
        return len(self._cache)

    @property
    def hits(self) -> int:
        """Number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def find_longest_prefix(
        self,
        tokens: List[int],
        allocator: BlockAllocator,
    ) -> Tuple[List[int], int]:
        """Find the longest cached prefix for a token sequence.

        Args:
            tokens: The full prompt token sequence.
            allocator: Block allocator to verify block validity.

        Returns:
            Tuple of (shared_block_ids, shared_token_count).
            Empty list and 0 if no prefix found or sharing disabled.
        """
        if not self.enabled:
            self._misses += 1
            return [], 0

        # Search from longest to shortest, aligned to block boundaries
        for length in range(
            len(tokens) // self.block_size * self.block_size,
            0,
            -self.block_size,
        ):
            prefix = tuple(tokens[:length])
            if prefix in self._cache:
                cached_blocks = self._cache[prefix]
                # Verify blocks are still valid
                valid = all(
                    allocator.get_block(bid) is not None
                    and not allocator.get_block(bid).is_free
                    for bid in cached_blocks
                )
                if valid:
                    self._hits += 1
                    return list(cached_blocks), length
                else:
                    # Stale entry, remove it
                    del self._cache[prefix]

        self._misses += 1
        return [], 0

    def cache_prefix(
        self,
        tokens: List[int],
        block_ids: List[int],
    ) -> None:
        """Cache a prefix -> blocks mapping.

        Args:
            tokens: The full token sequence.
            block_ids: Physical block IDs covering the prefix.
        """
        if not self.enabled:
            return

        # Cache at each block-aligned boundary
        for i in range(len(block_ids)):
            end = (i + 1) * self.block_size
            if end > len(tokens):
                end = len(tokens)
            prefix = tuple(tokens[:end])
            if prefix not in self._cache:
                self._cache[prefix] = block_ids[:i + 1]

    def invalidate(self, tokens: List[int]) -> None:
        """Remove a cached prefix entry.

        Args:
            tokens: The prefix token sequence to remove.
        """
        key = tuple(tokens)
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
        }
