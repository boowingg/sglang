from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import List, Optional

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
    StorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageExtraInfo,
)

logger = logging.getLogger(__name__)


class BufferPrefetchOperation(PrefetchOperation):
    """Query-first prefetch state owned only by the buffer-only backend."""

    def __init__(
        self,
        request_id: str,
        host_indices: Optional[torch.Tensor],
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            request_id,
            host_indices,
            token_ids,
            last_hash,
            prefix_keys,
        )
        self.query_first = host_indices is None
        self.query_done = False
        self.io_done = False
        self.storage_hit_count = 0
        self.query_done_time: Optional[float] = None
        self.io_start_time: Optional[float] = None

    def increment(self, num_tokens: int) -> bool:
        with self._lock:
            if self._terminated_flag:
                # Preserve a backend result that was already in flight, then let
                # the transfer loop stop before issuing the next backend call.
                if self.query_first:
                    self.completed_tokens += num_tokens
                return False
            self.completed_tokens += num_tokens
            return True

    def is_terminated(self) -> bool:
        with self._lock:
            return self._terminated_flag

    def set_query_result(self, hash_value: List[str], hit_count: int) -> None:
        with self._lock:
            self.hash_value = hash_value
            self.storage_hit_count = hit_count
            self.query_done = True
            self.query_done_time = time.monotonic()

    def mark_io_started(self) -> None:
        with self._lock:
            if self.io_start_time is None:
                self.io_start_time = time.monotonic()

    def mark_io_done(self) -> None:
        with self._lock:
            self.io_done = True


class BufferOnlyHiCacheController(HiCacheController):
    """HiCache controller whose host pool is an in-flight transfer buffer."""

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        result = super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )
        # Actual free pages and the cache-side write reserve govern admission.
        self.prefetch_capacity_limit = self.mem_pool_host.size
        return result

    def _start_storage_threads(self) -> None:
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()
        self.prefetch_buffer = Queue()

        self.prefetch_revoke_queue: Queue[str] = Queue()
        self.prefetch_hit_queue: Queue[BufferPrefetchOperation] = Queue()
        self.ack_backup_queue: Queue[StorageOperation] = Queue()
        self.host_mem_release_queue: Queue[torch.Tensor] = Queue()

        self.prefetch_thread.start()
        self.backup_thread.start()

    def reset(self) -> None:
        if self.enable_storage:
            self._stop_storage_threads()

        self.write_queue.clear()
        self.load_queue.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        if self.enable_storage:
            self.prefetch_tokens_occupied = 0

        self.storage_stop_event.clear()
        if self.enable_storage:
            self._start_storage_threads()

    def prefetch(
        self,
        request_id: str,
        host_indices: Optional[torch.Tensor],
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> BufferPrefetchOperation:
        operation = BufferPrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys,
        )
        self.prefetch_queue.put(operation)
        return operation

    def _page_transfer(self, operation: BufferPrefetchOperation) -> None:
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            if self.has_draft:
                self._draft_page_get(batch_hashes, batch_host_indices)

            prev_completed_tokens = operation.completed_tokens
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break
            if operation.is_terminated():
                break

            if prefix_keys:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self) -> None:
        while not self.storage_stop_event.is_set():
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                try:
                    self._page_transfer(operation)
                except Exception:
                    operation.mark_terminate()
                    logger.exception(
                        "Prefetch I/O failed for request %s", operation.request_id
                    )
                finally:
                    operation.mark_io_done()
                    # The scheduler owns completed pages; return only the unused
                    # suffix after the I/O thread has stopped touching it.
                    self.append_host_mem_release(
                        operation.host_indices[operation.completed_tokens :]
                    )
            except Empty:
                continue

    def prefetch_thread_func(self) -> None:
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            operation = None
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                try:
                    hash_value, storage_hit_count = self._storage_hit_query(operation)
                except Exception:
                    hash_value, storage_hit_count = [], 0
                    logger.exception(
                        "Storage hit query failed locally for request %s",
                        operation.request_id,
                    )

                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce_prefetch_groups(
                    storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
                )
                storage_hit_count = int(storage_hit_count_tensor.item())
                hash_value = hash_value[: storage_hit_count // self.page_size]
                operation.set_query_result(hash_value, storage_hit_count)

                if storage_hit_count < self.prefetch_threshold:
                    self.prefetch_revoke_queue.put(operation.request_id)
                    if operation.host_indices is not None:
                        self.append_host_mem_release(operation.host_indices)
                    logger.debug(
                        "Revoking prefetch for request %s due to insufficient "
                        "hits (%s).",
                        operation.request_id,
                        storage_hit_count,
                    )
                elif operation.query_first:
                    # The scheduler allocates exactly the confirmed hit length.
                    self.prefetch_hit_queue.put(operation)
                else:
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    self.prefetch_buffer.put(operation)
            except Empty:
                continue
            except Exception:
                if operation is not None:
                    operation.set_query_result([], 0)
                    operation.mark_terminate()
                    self.prefetch_revoke_queue.put(operation.request_id)
                    if operation.host_indices is not None:
                        self.append_host_mem_release(operation.host_indices)
                    logger.exception(
                        "Storage hit query failed for request %s",
                        operation.request_id,
                    )

    def backup_thread_func(self) -> None:
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue
                try:
                    if not self.backup_skip:
                        self._page_backup(operation)
                except Exception:
                    logger.exception("Storage backup operation %s failed", operation.id)
                self.ack_backup_queue.put(operation)
            except Empty:
                continue
