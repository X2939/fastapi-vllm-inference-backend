"""Block manager: maps logical blocks to physical blocks per request.

Each request sees its KV cache as a contiguous sequence of logical
blocks (0, 1, 2, ...). The BlockManager maintains the mapping from
logical block indices to physical block IDs, similar to an OS page table.

Design:
    - One BlockTable per request.
    - BlockManager coordinates with BlockAllocator for physical allocation.
    - Supports sharing blocks from another request's table (prefix sharing).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from engine.block_allocator import BlockAllocator, Block

logger = logging.getLogger(__name__)


class BlockTable:
    """Logical-to-physical block mapping for a single request.

    Attributes:
        request_id: The request this table belongs to.
        entries: List where entries[i] = physical block ID for logical block i.
    """

    def __init__(self, request_id: int):
        self.request_id = request_id
        self._entries: List[int] = []

    def append(self, physical_block_id: int) -> int:
        """Append a physical block to the table.

        Args:
            physical_block_id: The physical block to add.

        Returns:
            The logical index of the newly added block.
        """
        idx = len(self._entries)
        self._entries.append(physical_block_id)
        return idx

    def lookup(self, logical_idx: int) -> Optional[int]:
        """Look up the physical block for a logical index.

        Args:
            logical_idx: Logical block index.

        Returns:
            Physical block ID, or None if out of range.
        """
        if 0 <= logical_idx < len(self._entries):
            return self._entries[logical_idx]
        return None

    def get_all_physical_blocks(self) -> List[int]:
        """Return all physical block IDs in this table."""
        return list(self._entries)

    @property
    def num_blocks(self) -> int:
        """Number of logical blocks in this table."""
        return len(self._entries)

    @property
    def last_block_id(self) -> Optional[int]:
        """Physical ID of the last block, or None if empty."""
        return self._entries[-1] if self._entries else None

    def __repr__(self) -> str:
        return f"BlockTable(req={self.request_id}, blocks={self._entries})"


class BlockManager:
    """Manages block tables for all active requests.

    Responsible for:
        - Creating and destroying block tables
        - Allocating physical blocks via BlockAllocator
        - Appending tokens (handling block overflow)
        - Sharing blocks between tables (for prefix sharing)

    Not responsible for:
        - Deciding WHAT to share (PrefixCache does this)
        - Raw block pool management (BlockAllocator does this)
    """

    def __init__(self, block_allocator: BlockAllocator):
        """Initialize with a BlockAllocator.

        Args:
            block_allocator: The physical block pool manager.
        """
        self._allocator = block_allocator
        self._tables: Dict[int, BlockTable] = {}

    @property
    def block_size(self) -> int:
        """Block size (tokens per block)."""
        return self._allocator.block_size

    @property
    def allocator(self) -> BlockAllocator:
        """Underlying block allocator."""
        return self._allocator

    def has_table(self, request_id: int) -> bool:
        """Check if a block table exists for the given request."""
        return request_id in self._tables

    def get_table(self, request_id: int) -> Optional[BlockTable]:
        """Get the block table for a request."""
        return self._tables.get(request_id)

    def allocate_for_request(
        self,
        request_id: int,
        num_tokens: int,
    ) -> Optional[BlockTable]:
        """Allocate blocks for a new request.

        Args:
            request_id: The request ID.
            num_tokens: Number of tokens to allocate for.

        Returns:
            The created BlockTable, or None if allocation failed.
        """
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        if not self._allocator.can_allocate(num_blocks_needed):
            logger.debug(
                f"BlockManager: cannot allocate {num_blocks_needed} blocks "
                f"for request {request_id}"
            )
            return None

        table = BlockTable(request_id)
        for i in range(num_blocks_needed):
            block = self._allocator.allocate()
            if block is None:
                # Rollback
                for bid in table.get_all_physical_blocks():
                    self._allocator.free(bid)
                return None
            start = i * self.block_size
            end = min(start + self.block_size, num_tokens)
            block.set_token_num(end - start)
            table.append(block.block_id)

        self._tables[request_id] = table
        return table

    def share_blocks(
        self,
        request_id: int,
        physical_block_ids: List[int],
    ) -> BlockTable:
        """Share existing physical blocks with a new request.

        Args:
            request_id: The request sharing the blocks.
            physical_block_ids: Physical block IDs to share.

        Returns:
            The created BlockTable with shared blocks.
        """
        table = BlockTable(request_id)
        for bid in physical_block_ids:
            self._allocator.share(bid)
            table.append(bid)
        self._tables[request_id] = table
        return table

    def append_additional_blocks(
        self,
        request_id: int,
        num_tokens: int,
    ) -> List[int]:
        """Allocate additional blocks for an existing request.

        Used when a request needs more blocks after sharing a prefix.

        Args:
            request_id: The request ID.
            num_tokens: Number of additional tokens to allocate.

        Returns:
            List of newly allocated physical block IDs. Empty on failure.
        """
        table = self._tables.get(request_id)
        if table is None:
            return []

        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        if not self._allocator.can_allocate(num_blocks_needed):
            return []

        new_ids = []
        for i in range(num_blocks_needed):
            block = self._allocator.allocate()
            if block is None:
                break
            start = i * self.block_size
            end = min(start + self.block_size, num_tokens)
            block.set_token_num(end - start)
            table.append(block.block_id)
            new_ids.append(block.block_id)
        return new_ids

    def append_token(self, request_id: int) -> Optional[int]:
        """Append one token to a request's KV cache.

        If the last block is full, allocates a new block.

        Args:
            request_id: The request ID.

        Returns:
            Physical block ID where the token was written, or None on failure.
        """
        table = self._tables.get(request_id)
        if table is None:
            return None

        last_id = table.last_block_id
        if last_id is not None:
            block = self._allocator.get_block(last_id)
            if block and block.has_space:
                block.add_token()
                return last_id

        # Need a new block
        block = self._allocator.allocate()
        if block is None:
            return None
        block.add_token()
        table.append(block.block_id)
        return block.block_id

    def free_request(self, request_id: int) -> int:
        """Free all blocks for a request.

        Args:
            request_id: The request to free.

        Returns:
            Number of blocks fully freed.
        """
        table = self._tables.pop(request_id, None)
        if table is None:
            return 0
        freed = 0
        for bid in table.get_all_physical_blocks():
            if self._allocator.free(bid):
                freed += 1
        return freed

    def get_all_table_ids(self) -> List[int]:
        """Return all request IDs that have block tables."""
        return list(self._tables.keys())
