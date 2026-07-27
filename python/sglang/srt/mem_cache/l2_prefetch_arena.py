from __future__ import annotations

import threading
from typing import Protocol

import torch


class _HostPool(Protocol):
    size: int

    def alloc(self, need_size: int) -> torch.Tensor | None: ...

    def free(self, indices: torch.Tensor) -> int: ...


class L2PrefetchArena:
    """A reserved slice of an existing L2 pool used by transient L3 reads.

    The backing KV memory still belongs to ``host_pool``. The arena only owns
    the reserved indices and keeps them out of the normal L2 allocator until
    ``close``. Storage threads may return pages concurrently with the scheduler,
    so all state transitions are synchronized.
    """

    _FREE = 1
    _IN_USE = 2

    def __init__(
        self,
        host_pool: _HostPool,
        capacity_tokens: int,
        page_size: int,
    ) -> None:
        if capacity_tokens <= 0 or capacity_tokens % page_size != 0:
            raise ValueError(
                "L2 prefetch arena capacity must be a positive multiple of page_size"
            )

        self.host_pool = host_pool
        self.capacity_tokens = capacity_tokens
        self.page_size = page_size
        self._lock = threading.RLock()
        self._generation = 0
        self._closed = False
        self._slot_state = torch.zeros(host_pool.size, dtype=torch.uint8)
        self._free_slots = torch.empty(0, dtype=torch.int64)
        self._free_count = 0
        self._reserve_from_host_pool()

    def _reserve_from_host_pool(self) -> None:
        reserved = self.host_pool.alloc(self.capacity_tokens)
        if reserved is None:
            raise RuntimeError(
                "Insufficient free L2 capacity to reserve the prefetch arena "
                f"({self.capacity_tokens} tokens requested)"
            )
        reserved = reserved.to(device="cpu", dtype=torch.int64).flatten()
        if len(reserved) != self.capacity_tokens:
            self.host_pool.free(reserved)
            raise RuntimeError(
                "L2 host allocator returned an unexpected number of indices: "
                f"expected {self.capacity_tokens}, got {len(reserved)}"
            )
        self._slot_state.zero_()
        self._slot_state[reserved] = self._FREE
        self._free_slots = reserved
        self._free_count = self.capacity_tokens
        self._closed = False
        self._generation += 1

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def available_size(self) -> int:
        with self._lock:
            return self._free_count

    def notify_waiters(self) -> None:
        """Make the FIFO head eligible after a queue-only state transition."""
        with self._lock:
            self._generation += 1

    def alloc(self, need_size: int) -> torch.Tensor | None:
        if need_size <= 0 or need_size % self.page_size != 0:
            raise ValueError(
                "Arena allocation must be a positive multiple of page_size"
            )

        with self._lock:
            if self._closed or need_size > self._free_count:
                return None
            start = self._free_count - need_size
            indices = self._free_slots[start : self._free_count].clone()
            if not torch.all(self._slot_state[indices] == self._FREE):
                raise RuntimeError("L2 prefetch arena allocation state is corrupted")
            self._slot_state[indices] = self._IN_USE
            self._free_count = start
            self._generation += 1
            return indices

    def free(self, indices: torch.Tensor) -> int:
        if indices.numel() == 0:
            return 0
        if indices.numel() % self.page_size != 0:
            raise ValueError("Arena release must contain complete pages")

        indices = indices.to(device="cpu", dtype=torch.int64).flatten()
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot free indices to a closed L2 prefetch arena")
            if not torch.all(self._slot_state[indices] == self._IN_USE):
                raise RuntimeError(
                    "L2 prefetch arena detected an invalid or duplicate free"
                )
            end = self._free_count + len(indices)
            if end > self.capacity_tokens:
                raise RuntimeError("L2 prefetch arena free-list size is corrupted")
            self._slot_state[indices] = self._FREE
            self._free_slots[self._free_count : end] = indices
            self._free_count = end
            self._generation += 1
            return len(indices)

    def free_remaining(self, indices: torch.Tensor) -> int:
        """Release the still-owned subset during forced shutdown cleanup."""
        if indices.numel() == 0:
            return 0

        indices = indices.to(device="cpu", dtype=torch.int64).flatten()
        with self._lock:
            if self._closed:
                return 0
            owned = torch.unique(indices[self._slot_state[indices] == self._IN_USE])
            if owned.numel() == 0:
                return 0
            end = self._free_count + len(owned)
            if end > self.capacity_tokens:
                raise RuntimeError("L2 prefetch arena free-list size is corrupted")
            self._slot_state[owned] = self._FREE
            self._free_slots[self._free_count : end] = owned
            self._free_count = end
            self._generation += 1
            return len(owned)

    def rebind_after_host_clear(self) -> None:
        """Reserve a fresh slice after the parent host pool has been cleared."""
        with self._lock:
            self._reserve_from_host_pool()

    def close(self) -> None:
        """Return the complete reserved slice to the normal L2 allocator."""
        with self._lock:
            if self._closed:
                return
            in_use = int((self._slot_state == self._IN_USE).sum().item())
            if in_use:
                raise RuntimeError(
                    "Cannot close L2 prefetch arena with "
                    f"{in_use} tokens still in use"
                )
            if self._free_count != self.capacity_tokens:
                raise RuntimeError("L2 prefetch arena free-list size is corrupted")
            self.host_pool.free(self._free_slots[: self._free_count])
            self._slot_state.zero_()
            self._free_slots = torch.empty(0, dtype=torch.int64)
            self._free_count = 0
            self._closed = True
            self._generation += 1
