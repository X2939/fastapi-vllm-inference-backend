"""Block allocator for PagedAttention KV cache.

Manages the physical block pool: allocation, recycling, and statistics.
This module knows nothing about requests, prefix sharing, or scheduling.
It only manages raw blocks.

Design:
    - Blocks are identified by integer IDs (0 to num_blocks - 1).
    - Free blocks are tracked in a list for O(1) allocation.
    - Each block has a reference count for safe sharing.
    - When ref_count drops to 0, the block returns to the free list.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BlockState(Enum):
    """Physical block states."""
    FREE = "free"
    ALLOCATED = "allocated"


@dataclass
class Block:
    """A physical KV cache block.

    Attributes:
        block_id: Unique block identifier.
        block_size: Maximum number of tokens this block can hold.
        state: Whether the block is free or allocated.
        ref_count: Reference count (for prefix sharing).
        token_num: Number of tokens currently stored.
    """
    block_id: int
    block_size: int
    state: BlockState = BlockState.FREE
    ref_count: int = 0
    token_num: int = 0

    @property
    def is_free(self) -> bool:
        """Whether this block is available for allocation."""
        return self.state == BlockState.FREE

    @property
    def is_shared(self) -> bool:
        """Whether this block is shared by multiple requests."""
        return self.ref_count > 1

    @property
    def has_space(self) -> bool:
        """Whether this block can accept more tokens."""
        return self.token_num < self.block_size

    @property
    def free_slots(self) -> int:
        """Remaining token slots in this block."""
        return self.block_size - self.token_num

    def allocate(self) -> None:
        """Mark this block as allocated with ref_count=1."""
        if self.state != BlockState.FREE:
            raise RuntimeError(
                f"Block {self.block_id} is not free (state={self.state})"
            )
        self.state = BlockState.ALLOCATED
        self.ref_count = 1
        self.token_num = 0

    def add_ref(self) -> None:
        """Increment reference count (for prefix sharing)."""
        if self.state != BlockState.ALLOCATED:
            raise RuntimeError(
                f"Block {self.block_id} is not allocated (state={self.state})"
            )
        self.ref_count += 1

    def release(self) -> bool:
        """Decrement reference count.

        Returns:
            True if the block was fully freed (ref_count reached 0).
        """
        if self.state != BlockState.ALLOCATED:
            return False
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.state = BlockState.FREE
            self.ref_count = 0
            self.token_num = 0
            return True
        return False

    def add_token(self) -> bool:
        """Add one token to this block.

        Returns:
            True if successful, False if block is full.
        """
        if not self.has_space:
            return False
        self.token_num += 1
        return True

    def set_token_num(self, n: int) -> None:
        """Set the number of tokens (used during prefill allocation)."""
        self.token_num = min(n, self.block_size)

    def __repr__(self) -> str:
        return (
            f"Block(id={self.block_id}, state={self.state.value}, "
            f"ref={self.ref_count}, tokens={self.token_num}/{self.block_size})"
        )


class BlockAllocator:
    """Manages the physical block pool.

    Responsible for:
        - Allocating free blocks
        - Recycling freed blocks
        - Tracking block states
        - Providing statistics

    Not responsible for:
        - Request-to-block mapping (BlockManager does this)
        - Prefix sharing (PrefixCache does this)
    """

    def __init__(self, block_size: int = 16, num_blocks: int = 1024):
        """Initialize the block pool.

        Args:
            block_size: Number of tokens per block.
            num_blocks: Total number of physical blocks.
        """
        self.block_size = block_size
        self.num_blocks = num_blocks

        self._blocks: Dict[int, Block] = {
            i: Block(block_id=i, block_size=block_size)
            for i in range(num_blocks)
        }
        self._free_list: List[int] = list(range(num_blocks))
        self._peak_allocated: int = 0

    @property
    def total_blocks(self) -> int:
        """Total number of physical blocks."""
        return self.num_blocks

    @property
    def free_blocks(self) -> int:
        """Number of free blocks."""
        return len(self._free_list)

    @property
    def allocated_blocks(self) -> int:
        """Number of allocated blocks."""
        return self.num_blocks - len(self._free_list)

    @property
    def peak_allocated(self) -> int:
        """Peak number of allocated blocks during the run."""
        return self._peak_allocated

    @property
    def shared_blocks(self) -> int:
        """Number of blocks currently shared (ref_count > 1)."""
        return sum(1 for b in self._blocks.values() if b.is_shared)

    @property
    def utilization(self) -> float:
        """Fraction of blocks currently allocated."""
        if self.num_blocks == 0:
            return 0.0
        return self.allocated_blocks / self.num_blocks

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we can allocate the requested number of blocks.

        Args:
            num_blocks: Number of blocks needed.

        Returns:
            True if enough free blocks are available.
        """
        return len(self._free_list) >= num_blocks

    def allocate(self) -> Optional[Block]:
        """Allocate a single free block.

        Returns:
            The allocated Block, or None if no free blocks.
        """
        if not self._free_list:
            logger.warning("BlockAllocator: no free blocks available")
            return None
        block_id = self._free_list.pop(0)
        block = self._blocks[block_id]
        block.allocate()
        allocated = self.allocated_blocks
        if allocated > self._peak_allocated:
            self._peak_allocated = allocated
        return block

    def allocate_many(self, count: int) -> List[Block]:
        """Allocate multiple free blocks.

        Args:
            count: Number of blocks to allocate.

        Returns:
            List of allocated Blocks. May be shorter than count if
            not enough free blocks.
        """
        result = []
        for _ in range(count):
            block = self.allocate()
            if block is None:
                break
            result.append(block)
        return result

    def share(self, block_id: int) -> bool:
        """Increment reference count for an existing block.

        Args:
            block_id: The block to share.

        Returns:
            True if successful, False if block doesn't exist or is free.
        """
        block = self._blocks.get(block_id)
        if block is None or block.is_free:
            return False
        block.add_ref()
        return True

    def free(self, block_id: int) -> bool:
        """Release a reference to a block.

        Args:
            block_id: The block to release.

        Returns:
            True if the block was fully freed (ref_count reached 0).
        """
        block = self._blocks.get(block_id)
        if block is None:
            return False
        freed = block.release()
        if freed:
            self._free_list.append(block_id)
        return freed

    def get_block(self, block_id: int) -> Optional[Block]:
        """Get a block by ID."""
        return self._blocks.get(block_id)

    def get_stats(self) -> Dict[str, int]:
        """Return allocation statistics."""
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": self.free_blocks,
            "allocated_blocks": self.allocated_blocks,
            "shared_blocks": self.shared_blocks,
            "peak_allocated": self._peak_allocated,
            "utilization": self.utilization,
        }

    def reset(self) -> None:
        """Reset all blocks to free state."""
        for block in self._blocks.values():
            block.state = BlockState.FREE
            block.ref_count = 0
            block.token_num = 0
        self._free_list = list(range(self.num_blocks))
        self._peak_allocated = 0
