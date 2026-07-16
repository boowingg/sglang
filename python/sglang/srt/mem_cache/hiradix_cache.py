from __future__ import annotations

import atexit
import heapq
import json
import logging
import os
import threading
import time
from array import array
from collections import deque
from dataclasses import dataclass, field
from numbers import Integral
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PrefetchTimeoutConfig,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    attach_hybrid_dsa_pool_to_hiradix_cache,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_BUFFER_WRITE_STALE = 0
_BUFFER_WRITE_DEFER = 1
_BUFFER_WRITE_D2H = 2
_BUFFER_WRITE_INVALID = 3
_BUFFER_WRITE_IN_FLIGHT = 4
_BUFFER_WRITE_L3_RETRY = 5


@dataclass
class WriteThroughContext:
    write_id: int
    lock_node: TreeNode
    backup_len: int
    publish_nodes: List[TreeNode]
    host_indices: Optional[torch.Tensor] = None
    key: Optional[RadixKey] = None
    hash_value: Optional[List[str]] = None
    prefix_keys: Optional[List[str]] = None
    storage_operation_id: Optional[int] = None
    durable_tokens: int = 0
    inflight_offset: int = 0
    inflight_tokens: int = 0
    host_data_ready: bool = False
    source_locked: bool = False
    retry_count: int = 0
    retry_at: float = 0.0


@dataclass
class LoadBackContext:
    end_node: TreeNode
    host_nodes: List[TreeNode] = field(default_factory=list)


@dataclass
class PrefetchContext:
    anchor: TreeNode
    key: RadixKey
    operation: Optional[PrefetchOperation] = None
    host_indices: Optional[torch.Tensor] = None
    allocated_tokens: int = 0
    anchor_device_locked: bool = False
    aborted: bool = False
    query_ready: bool = False
    query_dispatched_tokens: int = 0
    last_hash: Optional[str] = None
    prefix_keys: Optional[List[str]] = None
    device_indices: Optional[torch.Tensor] = None
    device_reservation_tokens: int = 0
    require_full_completion: bool = False


class HiRadixCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        self.host_memory_mode = server_args.hicache_host_memory_mode
        self.buffer_only = self.host_memory_mode == "buffer_only"

        if self.buffer_only and (
            server_args.hicache_storage_backend is None
            or server_args.hicache_write_policy != "write_through"
        ):
            raise ValueError(
                "HiCache buffer_only requires an L3 backend and write_through."
            )
        if self.buffer_only and not isinstance(self.kv_cache, MHATokenToKVPool):
            raise ValueError(
                "HiCache buffer_only currently supports MHA/GQA KV pools only; "
                f"got {type(self.kv_cache).__name__}."
            )

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = get_mha_host_pool_cls(self.kv_cache)(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, DSATokenToKVPool):
            # Filled by attach_hybrid_dsa_pool_to_hiradix_cache after storage extra_config is parsed.
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MiniMaxSparseKVPool):
            # Filled by attach_hybrid_minimax_sparse_pool_to_hiradix_cache.
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError("HiRadixCache only supports MHA, MLA, DSA, and MSA models")

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
            prefetch_stream_window_tokens,
            prefetch_stream_buffer_count,
            prefetch_query_max_ops,
            prefetch_query_max_tokens,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        if isinstance(self.kv_cache, DSATokenToKVPool):
            attach_hybrid_dsa_pool_to_hiradix_cache(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
            )
        elif isinstance(self.kv_cache, MiniMaxSparseKVPool):
            from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
                attach_hybrid_minimax_sparse_pool_to_hiradix_cache,
            )

            attach_hybrid_minimax_sparse_pool_to_hiradix_cache(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
            )
        else:
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
                pp_group=self.pp_group,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                enable_storage_metrics=self.enable_storage_metrics,
                host_memory_mode=self.host_memory_mode,
            )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            prefetch_stream_window_tokens=prefetch_stream_window_tokens,
            prefetch_stream_buffer_count=prefetch_stream_buffer_count,
            prefetch_query_max_ops=prefetch_query_max_ops,
            prefetch_query_max_tokens=prefetch_query_max_tokens,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        self.loading_node_to_ack: Dict[TreeNode, int] = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.pending_write_through = deque()
        self.buffer_write_retry_ids: set[int] = set()
        self.pending_prefetch_query = deque()
        self.aborted_prefetch_req_ids: set[str] = set()
        self.streaming_prefetch_active: set[str] = set()
        self.streaming_prefetch_reservation_tokens = 0
        self.prefetch_query_ops_occupied = 0
        self.prefetch_query_tokens_occupied = 0
        self.transient_host_nodes_by_reqid: Dict[str, List[TreeNode]] = {}
        self.buffer_write_reserve_ratio = 0.2
        self.buffer_write_dispatch_limit = 64
        self.buffer_prefetch_paused_for_write = False
        # track per-request tokens loaded from storage (L3 hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.work_list: List[torch.distributed.Work] = []
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()

        super().__init__(params=params)

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _barrier_attn_groups(self):
        waited = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.barrier(group=group)
                waited = True
        if not waited and self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)

    def _drain_async_work(self):
        """
        Block until all outstanding async sends are consumed, then clear.

        Called at the start of each event round, so work_list holds the sends
        accumulated since the last round. This bounds it and applies
        backpressure when a downstream PP rank lags. Scheduler thread only.
        """
        for work in self.work_list:
            work.wait()
        self.work_list.clear()

    def _all_reduce(self, data: torch.Tensor, tp_reduce_op: torch.distributed.ReduceOp):
        """
        Synchronize data across all TP and PP ranks.

        In particular, "tp_reduce_op" is performed on all TP ranks of the first PP rank,
        and then the result is propagated to all following PP ranks.

        Must be called in the scheduler thread.
        """
        if self.pp_rank == 0:
            self._all_reduce_attn_groups(data, tp_reduce_op)
        self._pp_sync(data)

    def _pp_sync(self, data: torch.Tensor) -> None:
        """
        Synchronize data across the PP pipeline, where PPn (n>0) will receive PP0's data.

        The following diagram illustrates the behavior of _pp_sync.

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts | _pp_sync(data=?) starts | _pp_sync(data=?) starts
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=1) ends   |
        3     |                         |                         | _pp_sync(data=1) ends

        _pp_sync requires no synchronization point among ranks. The following case may also happen.

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts |                         |
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=?) starts |
        3     |                         | _pp_sync(data=1) ends   |
        4     |                         |                         | _pp_sync(data=?) starts
        5     |                         |                         | _pp_sync(data=1) ends
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return
        if self.pp_rank > 0:
            torch.distributed.recv(
                data,
                group_src=self.pp_rank - 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
        if self.pp_rank + 1 < self.pp_size:
            # Make a copy of data, so that the caller is safe to modify `data` after this call.
            # This is cheap, as _pp_sync is not to be used for transmitting large data.
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data,
                group_dst=self.pp_rank + 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
            self.work_list.append(send_work)

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                if self.buffer_only:
                    self._drain_storage_control_queues_local()
                    self.cache_controller.detach_storage_backend()
                    self.cache_controller.write_stream.synchronize()
                    self.cache_controller.load_stream.synchronize()
                    self.cache_controller.prefetch_load_stream.synchronize()
                    self._drain_storage_control_queues_local()
                    self._force_release_pending_storage_ops()
                    self.enable_storage = False
                else:
                    self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_config: PrefetchTimeoutConfig,
        hicache_storage_pass_prefix_keys: bool,
        prefetch_stream_window_tokens: int,
        prefetch_stream_buffer_count: int,
        prefetch_query_max_ops: int,
        prefetch_query_max_tokens: int,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_config = prefetch_timeout_config
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        stream_window = max(self.page_size, prefetch_stream_window_tokens)
        self.prefetch_stream_window_tokens = (
            stream_window // self.page_size * self.page_size
        )
        self.prefetch_stream_buffer_count = prefetch_stream_buffer_count
        self.prefetch_query_max_ops = prefetch_query_max_ops
        self.prefetch_query_max_tokens = prefetch_query_max_tokens
        self.enable_storage_metrics = enable_storage_metrics

        if self.buffer_only and self._buffer_prefetch_statically_impossible():
            logger.warning(
                "HiCache buffer-only storage data restores will be skipped: L2 "
                "cannot hold one minimum stream window or L1 is below the "
                "prefetch threshold."
            )

        if self.enable_storage_metrics:
            attn_cp_rank, attn_cp_size = (
                self.cache_controller.get_attn_cp_rank_and_size()
            )
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
                "attn_cp_rank": attn_cp_rank,
                "attn_cp_size": attn_cp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = getattr(self, "storage_metrics_collector", None)
            if existing_collector is None:
                from sglang.srt.runtime_context import get_server_args

                storage_cls = resolve_collector_class(
                    get_server_args(),
                    STAT_LOGGER_ROLE_STORAGE,
                    StorageMetricsCollector,
                )
                self.storage_metrics_collector = storage_cls(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )
            if self.buffer_only and hicache_write_policy != "write_through":
                return (
                    False,
                    "buffer_only only supports hicache_write_policy='write_through'.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_config,
                hicache_storage_pass_prefix_keys,
                prefetch_stream_window_tokens,
                prefetch_stream_buffer_count,
                prefetch_query_max_ops,
                prefetch_query_max_tokens,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                **self._get_hybrid_storage_attach_kwargs(),
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            prefetch_stream_window_tokens=prefetch_stream_window_tokens,
            prefetch_stream_buffer_count=prefetch_stream_buffer_count,
            prefetch_query_max_ops=prefetch_query_max_ops,
            prefetch_query_max_tokens=prefetch_query_max_tokens,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        if self.buffer_only:
            return (
                False,
                "Cannot detach L3 while --hicache-host-memory-mode buffer_only is active.",
            )

        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    self._release_prefetch_context(
                        info, free_completed=True, free_all_host=True
                    )
                except Exception:
                    logger.exception("Failed to release prefetch context %s", req_id)
                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")
        self.pending_prefetch_query.clear()
        self.aborted_prefetch_req_ids.clear()
        self.streaming_prefetch_active.clear()
        self.streaming_prefetch_reservation_tokens = 0
        self.prefetch_query_ops_occupied = 0
        self.prefetch_query_tokens_occupied = 0
        self.buffer_prefetch_paused_for_write = False

        # Force release leftover backup ops and their owned host pages.
        try:
            for ack_id, entry in list(self.ongoing_backup.items()):
                if isinstance(entry, WriteThroughContext):
                    if entry.host_indices is not None:
                        cc.mem_pool_host.free(entry.host_indices)
                        entry.host_indices = None
                    self._drop_pending_buffer_write(entry)
                else:
                    try:
                        entry.release_host()
                    except Exception:
                        logger.exception(
                            "Failed to release host protection for backup op %s",
                            ack_id,
                        )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

        if self.buffer_only:
            for context in list(self.ongoing_write_through.values()):
                if context.host_indices is not None:
                    cc.mem_pool_host.free(context.host_indices)
                    context.host_indices = None
                self._drop_pending_buffer_write(context)
        self.pending_write_through.clear()
        self.buffer_write_retry_ids.clear()

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_hit=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_hit: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                if info is not None:
                    self._release_prefetch_context(info, free_completed=True)

        def _drain_hits():
            for operation in _drain_queue(cc.prefetch_hit_queue, n_hit):
                context = self.ongoing_prefetch.get(operation.request_id)
                if context is None or context.operation is not operation:
                    continue
                if context.aborted:
                    operation.mark_terminate()
                    self._release_prefetch_context(context, free_completed=True)
                    self.ongoing_prefetch.pop(operation.request_id, None)
                    continue
                if operation.host_indices is not None:
                    continue
                # Queue draining is synchronized across attention ranks. This
                # flag is the scheduler-side readiness gate; polling the
                # background thread's local query_done flag directly can make
                # ranks enter allocation collectives in different rounds.
                context.query_ready = True

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if isinstance(entry, WriteThroughContext):
                    if (
                        entry.host_indices is None
                        or not entry.host_data_ready
                        or entry.inflight_tokens <= 0
                        or entry.inflight_offset != entry.durable_tokens
                    ):
                        raise RuntimeError(
                            f"Buffer-only L3 ACK {ack_id} has invalid write state"
                        )
                    if n_backup is not None:
                        completed_tokens = torch.tensor(
                            operation.completed_tokens, dtype=torch.int
                        )
                        self._all_reduce_attn_groups(
                            completed_tokens, torch.distributed.ReduceOp.MIN
                        )
                        operation.completed_tokens = int(completed_tokens.item())
                    completed = min(
                        max(0, operation.completed_tokens),
                        entry.inflight_tokens,
                    )
                    completed -= completed % self.page_size
                    operation.completed_tokens = completed
                    entry.durable_tokens = entry.inflight_offset + completed
                    self._mark_buffer_write_durable(entry, entry.durable_tokens)
                    self._release_buffer_write_source_if_staged(entry)

                    host_indices = entry.host_indices
                    if completed > 0:
                        cc.mem_pool_host.free(host_indices[:completed])
                    entry.inflight_offset += completed
                    entry.inflight_tokens -= completed
                    if entry.inflight_tokens == 0:
                        entry.host_indices = None
                        entry.host_data_ready = False
                    else:
                        # Keep the failed suffix in L2 and retry L3 directly;
                        # re-reading the same KV from GPU wastes bandwidth and
                        # would require the source to remain pinned.
                        entry.host_indices = host_indices[completed:]
                        entry.host_data_ready = True
                    entry.storage_operation_id = None
                    if entry.durable_tokens >= entry.backup_len:
                        self._drop_pending_buffer_write(entry)
                    else:
                        self._schedule_buffer_write_retry(
                            entry, made_progress=completed > 0
                        )
                elif entry is not None:
                    entry.release_host()
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_hits()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            Backend config plus the prefetch threshold, timeout config,
            prefix-key setting, stream-window size/count, and query limits.
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        defaults = PrefetchTimeoutConfig()
        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop(
            "prefetch_timeout_base", defaults.base
        )  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", defaults.per_ki_token
        )  # seconds per 1024 tokens
        prefetch_timeout_max = extra_config.pop(
            "prefetch_timeout_max", defaults.max
        )  # seconds, upper bound for the linear timeout
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )
        prefetch_stream_window_tokens = extra_config.pop(
            "prefetch_stream_window_tokens", 1024
        )
        prefetch_stream_buffer_count = extra_config.pop(
            "prefetch_stream_buffer_count", 2
        )
        prefetch_query_max_ops = extra_config.pop("prefetch_query_max_ops", 8)
        prefetch_query_max_tokens = extra_config.pop("prefetch_query_max_tokens", 32768)

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(prefetch_timeout_max, (int, float)):
            raise ValueError(
                f"prefetch_timeout_max must be number, got {type(prefetch_timeout_max).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )
        if (
            not isinstance(prefetch_stream_window_tokens, int)
            or isinstance(prefetch_stream_window_tokens, bool)
            or prefetch_stream_window_tokens <= 0
        ):
            raise ValueError("prefetch_stream_window_tokens must be a positive int")
        if (
            not isinstance(prefetch_stream_buffer_count, int)
            or isinstance(prefetch_stream_buffer_count, bool)
            or prefetch_stream_buffer_count not in (1, 2)
        ):
            raise ValueError("prefetch_stream_buffer_count must be 1 or 2")
        if (
            not isinstance(prefetch_query_max_ops, int)
            or isinstance(prefetch_query_max_ops, bool)
            or prefetch_query_max_ops <= 0
        ):
            raise ValueError("prefetch_query_max_ops must be a positive int")
        if (
            not isinstance(prefetch_query_max_tokens, int)
            or isinstance(prefetch_query_max_tokens, bool)
            or prefetch_query_max_tokens <= 0
        ):
            raise ValueError("prefetch_query_max_tokens must be a positive int")

        prefetch_timeout_config = PrefetchTimeoutConfig(
            base=float(prefetch_timeout_base),
            per_ki_token=float(prefetch_timeout_per_ki_token),
            max=float(prefetch_timeout_max),
        )

        return (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
            prefetch_stream_window_tokens,
            prefetch_stream_buffer_count,
            prefetch_query_max_ops,
            prefetch_query_max_tokens,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.write_stream.synchronize()
        self.cache_controller.load_stream.synchronize()
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        self.ongoing_write_through.clear()
        self.ongoing_load_back.clear()
        self.loading_node_to_ack.clear()
        self.ongoing_prefetch.clear()
        self.ongoing_backup.clear()
        self.pending_write_through.clear()
        self.buffer_write_retry_ids.clear()
        self.pending_prefetch_query.clear()
        self.aborted_prefetch_req_ids.clear()
        self.streaming_prefetch_active.clear()
        self.streaming_prefetch_reservation_tokens = 0
        self.prefetch_query_ops_occupied = 0
        self.prefetch_query_tokens_occupied = 0
        self.buffer_prefetch_paused_for_write = False
        self.transient_host_nodes_by_reqid.clear()
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def _get_extra_pools(self) -> dict:
        if not isinstance(self.cache_controller, HybridCacheController):
            return {}
        if isinstance(self.kv_cache, DSATokenToKVPool) or (
            isinstance(self.kv_cache, MiniMaxSparseKVPool)
            and self.kv_cache.index_k_pool is not None
        ):
            pool = PoolTransfer(
                name=PoolName.INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
                indices_from_pool=PoolName.KV,
            )
            return {"extra_pools": [pool]}
        else:
            return {}

    def _get_hybrid_storage_attach_kwargs(self) -> dict:
        """Extra kwargs for attach_storage_backend when controller is HybridCacheController."""
        if isinstance(self.cache_controller, HybridCacheController):
            return {"host_pools": self.cache_controller.mem_pool_host.entries}
        return {}

    def clear_storage_backend(self) -> bool:
        if self.buffer_only:
            logger.warning(
                "Cannot clear L3 while buffer_only is active because L3 durability "
                "metadata would become stale. Restart the engine after clearing."
            )
            return False
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False) -> int:
        if self.buffer_only:
            if write_back:
                raise RuntimeError("buffer_only does not support write_back eviction")
            if node.storage_backed or node.write_through_pending_id is not None:
                return 0

            context = WriteThroughContext(
                write_id=node.id,
                lock_node=node,
                backup_len=len(node.key),
                publish_nodes=[node],
                source_locked=True,
            )
            # The source must remain available until L3 acknowledges every
            # chunk. Otherwise L1 eviction can make a queued write impossible.
            self.inc_lock_ref(node)
            node.write_through_pending_id = context.write_id
            self.ongoing_write_through[context.write_id] = context
            self.pending_write_through.append(context.write_id)
            return len(node.value)

        # Backup invariant (for write-through mode): backed-up nodes must form a
        # contiguous prefix from root — no gaps.  Skip if parent isn't backed
        # up yet;
        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0

        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                **self._get_extra_pools(),
            )
        if host_indices is not None:
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0
            self._track_write_through_node(node, len(node.key))
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def _track_write_through_node(self, node: TreeNode, backup_len: int) -> None:
        node.write_through_pending_id = node.id
        self.ongoing_write_through[node.id] = WriteThroughContext(
            write_id=node.id,
            lock_node=node,
            backup_len=backup_len,
            publish_nodes=[node],
        )

    def _replace_pending_write_through_node(
        self, old_node: TreeNode, new_nodes: List[TreeNode]
    ) -> None:
        ack_id = old_node.write_through_pending_id
        if ack_id is None:
            return

        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        updated_nodes = []
        replaced = False
        for node in pending.publish_nodes:
            if node is old_node:
                updated_nodes.extend(new_nodes)
                replaced = True
            else:
                updated_nodes.append(node)

        if not replaced:
            return

        for node in new_nodes:
            node.write_through_pending_id = ack_id
        pending.publish_nodes = updated_nodes

    def _replace_pending_load_back_node(
        self, old_node: TreeNode, new_nodes: List[TreeNode]
    ) -> None:
        ack_id = self.loading_node_to_ack.get(old_node)
        if ack_id is None:
            return
        entry = self.ongoing_load_back.get(ack_id)
        if not isinstance(entry, LoadBackContext):
            return
        try:
            index = entry.host_nodes.index(old_node)
        except ValueError:
            return

        # load_back() owns one host reference per source node. The split prefix
        # needs its own reference so completion releases both slices once.
        new_nodes[0].protect_host()
        entry.host_nodes[index : index + 1] = new_nodes
        self.loading_node_to_ack.pop(old_node, None)
        for node in new_nodes:
            self.loading_node_to_ack[node] = ack_id

    def _write_context_is_attached(self, context: WriteThroughContext) -> bool:
        nodes = context.publish_nodes
        return bool(nodes) and not any(
            node.value is None
            or node.parent is None
            or node.parent.children.get(node.key.child_key(self.page_size)) is not node
            for node in nodes
        )

    def _snapshot_write_metadata(self, context: WriteThroughContext) -> None:
        if not self._write_context_is_attached(context):
            raise RuntimeError("write-through source was evicted before dispatch")
        nodes = context.publish_nodes
        if (
            sum(len(node.key) for node in nodes) != context.backup_len
            or sum(len(node.value) for node in nodes) != context.backup_len
        ):
            raise RuntimeError("write-through node chain changed unexpectedly")

        if context.key is not None:
            if (
                len(context.key) != context.backup_len
                or context.hash_value is None
                or len(context.hash_value) * self.page_size != context.backup_len
            ):
                raise RuntimeError("cached write-through metadata is invalid")
            return

        top = nodes[0]
        if top.key.is_bigram:
            token_ids = list(top.key.token_ids)
            for node in nodes[1:]:
                token_ids.extend(node.key.token_ids[1:])
        else:
            token_ids = [token for node in nodes for token in node.key.token_ids]
        context.key = RadixKey(token_ids, top.key.extra_key, top.key.is_bigram)
        if len(context.key) != context.backup_len:
            raise RuntimeError("write-through key length does not match source")
        if any(node.hash_value is None for node in nodes):
            raise RuntimeError("write-through source is missing L3 hashes")
        context.hash_value = [value for node in nodes for value in node.hash_value]
        if len(context.hash_value) * self.page_size != context.backup_len:
            raise RuntimeError("write-through hash length does not match source")
        context.prefix_keys = (
            list(top.get_prefix_hash_values(top.parent))
            if self.hicache_storage_pass_prefix_keys
            else None
        )

    def _slice_write_device_indices(
        self, context: WriteThroughContext, offset: int, length: int
    ) -> torch.Tensor:
        pieces = []
        remaining_offset = offset
        remaining_length = length
        for node in context.publish_nodes:
            node_length = len(node.value)
            if remaining_offset >= node_length:
                remaining_offset -= node_length
                continue
            take = min(node_length - remaining_offset, remaining_length)
            pieces.append(node.value[remaining_offset : remaining_offset + take])
            remaining_length -= take
            remaining_offset = 0
            if remaining_length == 0:
                break
        if remaining_length != 0 or not pieces:
            raise RuntimeError("write-through chunk exceeds source node chain")
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def _drop_pending_buffer_write(self, context: WriteThroughContext) -> None:
        self.ongoing_write_through.pop(context.write_id, None)
        retry_ids = getattr(self, "buffer_write_retry_ids", None)
        if retry_ids is not None:
            retry_ids.discard(context.write_id)
        for node in context.publish_nodes:
            if node.write_through_pending_id == context.write_id:
                node.write_through_pending_id = None
        if context.source_locked:
            self.dec_lock_ref(context.lock_node)
            context.source_locked = False
        context.inflight_tokens = 0
        context.host_data_ready = False
        context.storage_operation_id = None

    def _release_buffer_write_source_if_staged(
        self, context: WriteThroughContext
    ) -> None:
        if (
            context.source_locked
            and context.host_data_ready
            and context.inflight_offset + context.inflight_tokens >= context.backup_len
        ):
            # The final suffix is now protected by L2. L3 retries can use the
            # same host pages, so the GPU source no longer needs to stay pinned.
            self.dec_lock_ref(context.lock_node)
            context.source_locked = False

    def _mark_buffer_write_durable(
        self, context: WriteThroughContext, durable_tokens: int
    ) -> None:
        remaining = durable_tokens
        for node in context.publish_nodes:
            completed = min(len(node.key), remaining)
            node.storage_backed_tokens = max(node.storage_backed_tokens, completed)
            remaining -= completed
            if remaining <= 0:
                break

    def _schedule_buffer_write_retry(
        self, context: WriteThroughContext, *, made_progress: bool
    ) -> None:
        if made_progress:
            context.retry_count = 0
            context.retry_at = 0.0
            retry_ids = getattr(self, "buffer_write_retry_ids", None)
            if retry_ids is not None:
                retry_ids.discard(context.write_id)
        else:
            context.retry_count += 1
            delay = min(5.0, 0.1 * (2 ** min(context.retry_count - 1, 6)))
            context.retry_at = time.monotonic() + delay
            retry_ids = getattr(self, "buffer_write_retry_ids", None)
            if retry_ids is not None:
                retry_ids.add(context.write_id)
        self.pending_write_through.append(context.write_id)

    def _dispatch_pending_buffer_writes(self, pending_count: int) -> None:
        if not self.buffer_only:
            return

        dispatch_count = min(
            pending_count,
            len(self.pending_write_through),
            self.buffer_write_dispatch_limit,
        )
        if dispatch_count == 0:
            return

        write_ids = [
            self.pending_write_through.popleft() for _ in range(dispatch_count)
        ]
        records = []
        now = time.monotonic()
        requested_tokens = 0
        for write_id in write_ids:
            context = self.ongoing_write_through.get(write_id)
            local_action = _BUFFER_WRITE_STALE
            valid = 0
            remaining = 0
            if context is not None:
                if context.storage_operation_id is not None:
                    # A duplicate queue entry must not cancel an in-flight write.
                    local_action = _BUFFER_WRITE_IN_FLIGHT
                elif context.host_indices is not None:
                    if not context.host_data_ready:
                        # D2H is still in flight; this is a duplicate entry.
                        local_action = _BUFFER_WRITE_IN_FLIGHT
                    elif context.retry_at > now:
                        local_action = _BUFFER_WRITE_DEFER
                    else:
                        local_action = _BUFFER_WRITE_L3_RETRY
                        remaining = context.inflight_tokens
                        valid = int(
                            remaining > 0
                            and remaining % self.page_size == 0
                            and len(context.host_indices) == remaining
                            and context.inflight_offset == context.durable_tokens
                            and context.key is not None
                            and context.hash_value is not None
                        )
                else:
                    attached = self._write_context_is_attached(context)
                    top = context.publish_nodes[0] if attached else None
                    if not attached:
                        local_action = _BUFFER_WRITE_INVALID
                    elif context.retry_at > now:
                        local_action = _BUFFER_WRITE_DEFER
                    elif top.parent is self.root_node or top.parent.storage_backed:
                        local_action = _BUFFER_WRITE_D2H
                    elif (
                        top.parent is not None
                        and top.parent.write_through_pending_id is not None
                    ):
                        local_action = _BUFFER_WRITE_DEFER
                    else:
                        local_action = _BUFFER_WRITE_INVALID
                    if local_action == _BUFFER_WRITE_D2H:
                        try:
                            self._snapshot_write_metadata(context)
                        except RuntimeError:
                            local_action = _BUFFER_WRITE_INVALID
                        else:
                            remaining = context.backup_len - context.durable_tokens
                            valid = int(
                                remaining > 0 and remaining % self.page_size == 0
                            )
                            requested_tokens += max(0, remaining)
            records.append((write_id, context, local_action, valid, remaining))

        local_available = self.cache_controller.mem_pool_host.available_size()
        reclaim = min(requested_tokens, self.cache_controller.mem_pool_host.size)
        reclaim_needed = max(0, reclaim - local_available)
        if reclaim_needed > 0:
            self.evict_host(reclaim_needed)
        sync_values = [self.cache_controller.mem_pool_host.available_size()]
        for write_id, _context, action, valid, remaining in records:
            sync_values.extend(
                (
                    write_id,
                    -write_id,
                    action,
                    -action,
                    valid,
                    -valid,
                    remaining,
                    -remaining,
                )
            )
        status = torch.tensor(sync_values, dtype=torch.int)
        self._all_reduce_attn_groups(status, torch.distributed.ReduceOp.MIN)

        available = int(status[0].item())
        for index, (
            write_id,
            context,
            _local_action,
            _local_valid,
            _local_remaining,
        ) in enumerate(records):
            base = 1 + index * 8
            (
                min_write_id,
                neg_max_write_id,
                action,
                neg_max_action,
                valid,
                neg_max_valid,
                min_remaining,
                neg_max_remaining,
            ) = map(int, status[base : base + 8].tolist())
            if min_write_id != -neg_max_write_id or min_write_id != write_id:
                raise RuntimeError("Buffer-only write queue diverged across TP ranks")
            max_action = -neg_max_action
            if action != max_action:
                # Retry deadlines are based on each rank's monotonic clock. If
                # some ranks just crossed the deadline, defer on every rank.
                if action == _BUFFER_WRITE_DEFER and max_action in (
                    _BUFFER_WRITE_D2H,
                    _BUFFER_WRITE_L3_RETRY,
                ):
                    action = _BUFFER_WRITE_DEFER
                else:
                    raise RuntimeError(
                        "Buffer-only write state diverged across TP ranks"
                    )
            elif action in (_BUFFER_WRITE_D2H, _BUFFER_WRITE_L3_RETRY) and (
                valid != -neg_max_valid or min_remaining != -neg_max_remaining
            ):
                raise RuntimeError(
                    "Buffer-only write progress diverged across TP ranks"
                )
            if action == _BUFFER_WRITE_STALE:
                continue
            if action == _BUFFER_WRITE_DEFER:
                self.pending_write_through.append(write_id)
                continue
            if action == _BUFFER_WRITE_INVALID:
                raise RuntimeError(
                    f"Buffer-only write {write_id} lost its source or metadata"
                )
            if action == _BUFFER_WRITE_IN_FLIGHT:
                continue
            if valid == 0:
                raise RuntimeError(
                    f"Buffer-only write {write_id} has invalid progress metadata"
                )
            if action == _BUFFER_WRITE_L3_RETRY:
                self._submit_buffer_write_chunk_to_storage(context)
                continue

            chunk_tokens = min(min_remaining, available)
            chunk_tokens -= chunk_tokens % self.page_size
            if chunk_tokens == 0:
                self.pending_write_through.append(write_id)
                continue

            if not context.source_locked:
                self.inc_lock_ref(context.lock_node)
                context.source_locked = True
            device_indices = self._slice_write_device_indices(
                context, context.durable_tokens, chunk_tokens
            )

            host_indices = self.cache_controller.write(
                device_indices=device_indices,
                node_id=context.write_id,
                **self._get_extra_pools(),
            )
            if host_indices is None:
                self._schedule_buffer_write_retry(context, made_progress=False)
                continue

            context.host_indices = host_indices
            context.inflight_offset = context.durable_tokens
            context.inflight_tokens = chunk_tokens
            context.host_data_ready = False
            available -= chunk_tokens

    def _submit_buffer_write_chunk_to_storage(
        self, context: WriteThroughContext
    ) -> bool:
        if (
            context.host_indices is None
            or not context.host_data_ready
            or context.inflight_tokens <= 0
            or len(context.host_indices) != context.inflight_tokens
            or context.inflight_offset != context.durable_tokens
            or context.key is None
            or context.hash_value is None
        ):
            raise RuntimeError(
                f"Buffer-only write {context.write_id} has no retryable L2 chunk"
            )

        chunk_start = context.inflight_offset
        chunk_end = chunk_start + context.inflight_tokens
        chunk_key = context.key[chunk_start:chunk_end]
        chunk_hash_value = context.hash_value[
            chunk_start // self.page_size : chunk_end // self.page_size
        ]
        chunk_prefix_keys = None
        if context.prefix_keys is not None:
            chunk_prefix_keys = list(context.prefix_keys)
            if chunk_start > 0:
                chunk_prefix_keys.extend(
                    context.hash_value[: chunk_start // self.page_size]
                )
        try:
            operation_id = self.cache_controller.write_storage(
                context.host_indices,
                chunk_key,
                chunk_hash_value,
                chunk_prefix_keys,
                **self._get_extra_pools(),
            )
        except Exception:
            self._schedule_buffer_write_retry(context, made_progress=False)
            logger.exception(
                "Failed to submit buffer-only L3 write %s", context.write_id
            )
            return False
        if operation_id is None:
            self._schedule_buffer_write_retry(context, made_progress=False)
            logger.error(
                "Buffer-only L3 write %s returned no operation id",
                context.write_id,
            )
            return False
        context.storage_operation_id = operation_id
        self.ongoing_backup[operation_id] = context
        return True

    def _finish_write_through_ack(self, ack_id: int, *, release_lock: bool) -> None:
        context = self.ongoing_write_through.get(ack_id)
        if context is None:
            logger.warning("Ignoring stale write-through ACK %s", ack_id)
            return
        if self.buffer_only:
            if context.storage_operation_id is not None:
                logger.warning("Ignoring duplicate buffer-only write ACK %s", ack_id)
                return
            if context.host_data_ready:
                logger.warning("Ignoring stale buffer-only D2H ACK %s", ack_id)
                return
            if (
                context.host_indices is None
                or context.inflight_tokens <= 0
                or len(context.host_indices) != context.inflight_tokens
                or context.inflight_offset != context.durable_tokens
                or context.key is None
                or context.hash_value is None
            ):
                raise RuntimeError(
                    f"Buffer-only write {ack_id} has an invalid in-flight chunk"
                )
            context.host_data_ready = True
            self._release_buffer_write_source_if_staged(context)
            self._submit_buffer_write_chunk_to_storage(context)
            return

        self.ongoing_write_through.pop(ack_id)
        for node in context.publish_nodes:
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
            # DMA confirmed -- block is now on host.
            self._record_store_event(node, medium=StorageMedium.CPU)
        if self.enable_storage:
            self.write_backup_storage(context.lock_node, context.backup_len)
        if release_lock:
            self.dec_lock_ref(context.lock_node)

    def write_backup_storage(self, node: TreeNode, backup_len: Optional[int] = None):
        # Recover pre-split data via walk-and-concat if node was split.
        # prefix_keys anchored at chain top to avoid double-counting.
        if backup_len is None or len(node.key) == backup_len:
            top, key, hash_value, host_value = (
                node,
                node.key,
                node.hash_value,
                node.host_value,
            )
        else:
            top, key, hash_value, host_value = self._concat_split_chain(
                node, backup_len
            )

        prefix_keys = (
            top.get_prefix_hash_values(top.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        operation_id = self.cache_controller.write_storage(
            host_value, key, hash_value, prefix_keys, **self._get_extra_pools()
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _concat_split_chain(self, node: TreeNode, backup_len: int):
        """Recover enqueue-time key/hash/host by walking the split chain."""
        chain, accumulated = [], 0
        current = node
        while current is not self.root_node and accumulated < backup_len:
            chain.append(current)
            accumulated += len(current.key)
            current = current.parent
        assert accumulated == backup_len, (
            f"backup chain length mismatch for node {node.id}: "
            f"expected {backup_len}, got {accumulated}"
        )
        chain.reverse()  # parent-first
        top = chain[0]
        if top.key.is_bigram:
            # Bigram segments share boundary tokens; drop overlap after first.
            token_ids = list(chain[0].key.token_ids)
            for n in chain[1:]:
                token_ids.extend(n.key.token_ids[1:])
        else:
            token_ids = []
            for n in chain:
                token_ids.extend(n.key.token_ids)
        key = RadixKey(token_ids, top.key.extra_key, top.key.is_bigram)

        if all(n.hash_value is not None for n in chain):
            hash_value = []
            for n in chain:
                hash_value.extend(n.hash_value)
        else:
            hash_value = None
        host_value = torch.cat([n.host_value for n in chain])
        return top, key, hash_value, host_value

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        needs_backup = (
            not node.storage_backed if self.buffer_only else not node.backuped
        )
        if needs_backup and node.write_through_pending_id is None:
            if node.hit_count >= self.write_through_threshold:
                # write to host if the node is not backuped
                self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        self._finish_write_through_ack(ack_id, release_lock=False)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # Every rank must enter the all_reduce below; ongoing_write_through can
        # diverge across ranks (e.g. write_backup returning 0 on a subset under
        # host memory pressure), so a conditional skip desyncs the NCCL op
        # sequence and deadlocks under TP > 1. (Matches UnifiedRadixCache.)
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                if not finish_event.query():
                    break
                finish_count += 1
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} write back operations")
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self._finish_write_through_ack(ack_id, release_lock=True)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        if self.pp_rank == 0:
            for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
                if not finish_event.query():
                    break
                finish_count += 1
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} load operations")
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_load_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self._finish_load_back_ack(ack_id)
            finish_count -= 1

    def _finish_load_back_ack(self, ack_id: int) -> None:
        entry = self.ongoing_load_back.pop(ack_id, None)
        if entry is None:
            return
        if isinstance(entry, LoadBackContext):
            for node in entry.host_nodes:
                self.loading_node_to_ack.pop(node, None)
                node.device_loading = False
                self._record_store_event(node, medium=StorageMedium.GPU)
                node.release_host()
                if node.host_value_transient and node.host_value is not None:
                    self._forget_transient_host_node(node)
                    self.cache_controller.mem_pool_host.free(node.host_value)
                    node.host_value = None
                    node.host_value_transient = False
                    self._update_host_leaf_status(node)
            self.dec_lock_ref(entry.end_node)
        else:
            self.dec_lock_ref(entry)

    def _synchronize_loading_prefix(
        self, key: RadixKey, *, start_node: Optional[TreeNode] = None
    ) -> bool:
        if not self.loading_node_to_ack:
            return True

        needed_ack_ids = set()
        node = start_node if start_node is not None else self.root_node
        remaining_key = key
        if node.device_loading:
            ack_id = self.loading_node_to_ack.get(node)
            if ack_id is not None:
                needed_ack_ids.add(ack_id)
        while len(remaining_key) > 0:
            child_key = remaining_key.child_key(self.page_size)
            child = node.children.get(child_key)
            if child is None:
                break
            if child.device_loading:
                ack_id = self.loading_node_to_ack.get(child)
                if ack_id is not None:
                    needed_ack_ids.add(ack_id)
            prefix_len = child.key.match(remaining_key, page_size=self.page_size)
            if prefix_len < len(child.key):
                break
            remaining_key = remaining_key[prefix_len:]
            node = child

        if not needed_ack_ids:
            return True

        finish_through = -1
        queued_ack_ids = set()
        for index, (_, _, ack_list) in enumerate(self.cache_controller.ack_load_queue):
            queued_ack_ids.update(ack_list)
            if needed_ack_ids.intersection(ack_list):
                finish_through = index
        missing_ack_ids = needed_ack_ids.difference(queued_ack_ids)
        if missing_ack_ids:
            logger.debug(
                "Load-back completion event is not attached yet for radix nodes: %s",
                sorted(missing_ack_ids),
            )
            return False

        # The load stream is FIFO. Completing through the latest matching event
        # also makes every earlier event safe to retire locally.
        for _ in range(finish_through + 1):
            _, finish_event, ack_list = self.cache_controller.ack_load_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self._finish_load_back_ack(ack_id)
        return True

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        """Return True after the local load-back event is complete."""
        if consumer_index < 0:
            return True

        finish_event = self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event
        if not finish_event.query():
            return False

        self.loading_check()
        return True

    def evictable_size(self):
        return self.evictable_size_

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.backuped:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        if self.cache_controller.write_policy == "write_back":
            num_evicted = self._evict_write_back(num_tokens)
        else:
            num_evicted = self._evict_write_through(num_tokens)
        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _make_eviction_heap(self):
        heap = [
            (self.eviction_strategy.get_priority(node), node)
            for node in self.evictable_leaves
        ]
        heapq.heapify(heap)
        return heap

    def _promote_parent(self, node: TreeNode, heap) -> None:
        # Once all of a node's children are evicted, it becomes a device leaf.
        p = node.parent
        if p is not self.root_node and all(c.evicted for c in p.children.values()):
            heapq.heappush(heap, (self.eviction_strategy.get_priority(p), p))

    def _drop_unprotected_transient_host_descendants(self, node: TreeNode) -> bool:
        """Detach idle L3-prefetched descendants before evicting a GPU node.

        Pure buffer mode permits a GPU node to have host-only descendants, so a
        device leaf is not necessarily a structural tree leaf.
        """
        descendants = []
        stack = list(node.children.values())
        while stack:
            current = stack.pop()
            descendants.append(current)
            stack.extend(current.children.values())

        if any(
            not current.evicted
            or not current.host_value_transient
            or current.host_value is None
            or current.device_loading
            or current.host_ref_counter > 0
            or current.lock_ref > 0
            for current in descendants
        ):
            return False

        for current in reversed(descendants):
            self._forget_transient_host_node(current)
            self.cache_controller.mem_pool_host.free(current.host_value)
            current.host_value = None
            current.host_value_transient = False
            self.evictable_host_leaves.discard(current)
            self.evictable_leaves.discard(current)
        node.children.clear()
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        return True

    def _evict_write_through(self, num_tokens: int) -> int:
        """write_through / write_through_selective: drop non-backuped leaves,
        demote already-backuped ones. Nothing is staged to host during eviction,
        so this is a plain on-the-fly pass.
        """
        heap = self._make_eviction_heap()
        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0 or (self.buffer_only and x.host_ref_counter > 0):
                continue
            if (
                self.buffer_only
                and x.children
                and not x.backuped
                and not self._drop_unprotected_transient_host_descendants(x)
            ):
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            else:
                num_evicted += self._evict_regular(x)
            self._promote_parent(x, heap)
        return num_evicted

    def _evict_write_back(self, num_tokens: int) -> int:
        """eviction for write_back mode: demote already-backuped leaves, stage non-backuped ones to host if possible, otherwise drop them.
        note this path will be deprecated in the future.
        """
        heap = self._make_eviction_heap()
        num_evicted = 0
        staged: List[Tuple[TreeNode, torch.Tensor]] = []

        def flush_staged() -> None:
            if not staged:
                return
            self.writing_check(write_back=True)
            for node, device_indices in staged:
                self.cache_controller.evict_device(device_indices)
                node.release_host()
            staged.clear()

        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0:
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            elif self.write_backup(x, write_back=True) > 0:
                x.protect_host()
                staged.append((x, x.value))
                num_evicted += self._detach_backuped(x)
            else:
                flush_staged()
                num_evicted += self._drop_subtree_no_host(x)
            self._promote_parent(x, heap)
        flush_staged()
        return num_evicted

    def _detach_backuped(self, node: TreeNode) -> int:
        # detach nodes from tree while keeping device slots, for write-back eviction
        self._record_remove_event(node, medium=StorageMedium.GPU)
        num_evicted = len(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_backuped(self, node: TreeNode):
        device_indices = node.value
        num_evicted = self._detach_backuped(node)
        self.cache_controller.evict_device(device_indices)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host -- emit BlockRemoved
        assert len(node.children) == 0, f"non-leaf, {node.id=}"

        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def _drop_subtree_no_host(self, root: TreeNode) -> int:
        nodes = []
        stack = [root]
        while stack:
            n = stack.pop()
            nodes.append(n)
            stack.extend(n.children.values())

        if any(n.host_ref_counter > 0 for n in nodes):
            return 0

        logger.warning(
            "write_back: KV cache on device are dropped without backup due to host memory pressure, subtree root %d, num_nodes %d",
            root.id,
            len(nodes),
        )

        freed_device = 0
        for n in nodes:
            if n.host_value is not None:
                self._record_remove_event(n, medium=StorageMedium.CPU)
                self.cache_controller.evict_host(n.host_value)
                n.host_value = None
            if n.value is not None:
                self._record_remove_event(n, medium=StorageMedium.GPU)
                self.cache_controller.mem_pool_device_allocator.free(n.value)
                freed_device += len(n.value)
                self.evictable_size_ -= len(n.value)
                n.value = None
            self.ongoing_write_through.pop(n.id, None)
            self.evictable_leaves.discard(n)
            self.evictable_host_leaves.discard(n)

        key = root.key.child_key(self.page_size)
        root.parent.children.pop(key, None)
        self._update_leaf_status(root.parent)
        self._update_host_leaf_status(root.parent)
        return freed_device

    def evict_host(self, num_tokens: int):
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            if x.host_ref_counter > 0:
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --
            # emit remove(CPU) so the router drops the host-tier entry.
            if not self.buffer_only:
                self._record_remove_event(x, medium=StorageMedium.CPU)
            host_value = x.host_value
            num_evicted += self.cache_controller.evict_host(host_value)
            if self.buffer_only:
                self._forget_transient_host_node(x)
                x.host_value = None
                x.host_value_transient = False

            key = x.key.child_key(self.page_size)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        # Protect the nodes being loaded from host eviction.
        for n in nodes_to_load:
            n.protect_host()

        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            **self._get_extra_pools(),
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            for n in nodes_to_load:
                n.release_host()
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        if not self.buffer_only:
            for n in nodes_to_load:
                n.release_host()
            self.ongoing_load_back[last_hit_node.id] = last_hit_node
        else:
            context = LoadBackContext(
                end_node=last_hit_node,
                host_nodes=nodes_to_load,
            )
            self.ongoing_load_back[last_hit_node.id] = context
            for host_node in nodes_to_load:
                self.loading_node_to_ack[host_node] = last_hit_node.id
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            if self.buffer_only:
                node.device_loading = True
            offset += len(node.host_value)
            # Block promoted from host to GPU -- emit store(GPU) so downstream
            # indexers see it as device-local again.
            if not self.buffer_only:
                self._record_store_event(node, medium=StorageMedium.GPU)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.best_match_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            self._empty_match_result.device_indices,
            last_node,
        )

    def query_storage_hit(
        self,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> Tuple[List[str], int]:
        if not self.enable_storage:
            return [], 0
        if not self.buffer_only and self.cache_controller.prefetch_rate_limited():
            return [], 0

        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        if len(prefetch_key) < self.prefetch_threshold:
            return [], 0

        if self.buffer_only:
            # Decode-side PD queries run synchronously and bypass the normal
            # query dispatcher. Synchronize the write-first and static-capacity
            # gates before any rank returns or enters the backend query.
            control = torch.tensor(
                [
                    int(self._has_l3_write_query_pressure()),
                    int(self._buffer_prefetch_statically_impossible()),
                ],
                dtype=torch.int,
            )
            self._all_reduce_attn_groups(control, torch.distributed.ReduceOp.MAX)
            if control.any().item():
                return [], 0

        prefetch_op_cls = (
            HybridPrefetchOperation
            if isinstance(self.cache_controller, HybridCacheController)
            else PrefetchOperation
        )
        extra_kwargs = {}
        if prefetch_op_cls is HybridPrefetchOperation:
            extra_kwargs["pool_transfers"] = self._get_extra_pools().get("extra_pools")
        operation = prefetch_op_cls(
            "__storage_hit_query__",
            self.cache_controller.mem_pool_host.get_dummy_flat_data_page()[:0],
            prefetch_key,
            last_hash,
            prefix_keys,
            **extra_kwargs,
        )
        try:
            hash_values, storage_hit_count = self.cache_controller._storage_hit_query(
                operation
            )
        except Exception:
            # Every attention rank must still enter the reduction below. A local
            # backend error therefore turns the synchronized result into a miss.
            hash_values, storage_hit_count = [], 0
            logger.exception("Synchronous HiCache storage hit query failed")
        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_attn_groups(
            storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
        )
        storage_hit_count = storage_hit_count_tensor.item()
        storage_hit_count = storage_hit_count - (storage_hit_count % self.page_size)
        if storage_hit_count < self.prefetch_threshold:
            return [], 0
        hash_values = hash_values[: storage_hit_count // self.page_size]
        return hash_values, storage_hit_count

    def query_storage_hit_length(
        self,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        """Compatibility wrapper for callers that only need the hit length."""
        _, storage_hit_count = self.query_storage_hit(
            last_host_node,
            new_input_tokens,
            last_hash,
            prefix_keys,
        )
        return storage_hit_count

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def flush_write_through_acks(self) -> None:
        self.writing_check()

    def check_hicache_events(self):
        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.prefetch_hit_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
                len(self.pending_write_through),
                len(self.pending_prefetch_query),
                -int(self._has_l3_write_query_pressure()),
                len(self.aborted_prefetch_req_ids),
            ],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)

        (
            n_revoke,
            n_hit,
            n_backup,
            n_release,
            n_pending_write,
            n_pending_query,
            neg_write_query_pressure,
            n_aborted_prefetch,
        ) = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_hit=n_hit,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )
        self._reap_aborted_prefetches(n_aborted_prefetch)
        if self.buffer_only:
            # Writes get first use of released pages. Streaming reads are
            # admitted lazily when the scheduler examines the owning request,
            # so queued requests do not reserve L1 or L2 ahead of admission.
            pause_for_write = neg_write_query_pressure < 0
            self._dispatch_pending_buffer_writes(n_pending_write + n_backup)
            # Dispatch can create the first retry in this round. This set is
            # scheduler-owned and follows the synchronized write state machine;
            # do not re-sample backup_queue.qsize() here because its background
            # consumer can make that value differ transiently across ranks.
            pause_for_write = pause_for_write or bool(self.buffer_write_retry_ids)
            self.buffer_prefetch_paused_for_write = pause_for_write
            self._dispatch_pending_prefetch_queries(
                n_pending_query,
                pause_for_write=pause_for_write,
            )

    def _has_l3_write_query_pressure(self) -> bool:
        backup_queue = getattr(self.cache_controller, "backup_queue", None)
        if (
            backup_queue is not None
            and backup_queue.qsize()
            >= getattr(self, "buffer_write_dispatch_limit", 64)
        ):
            return True
        retry_ids = getattr(self, "buffer_write_retry_ids", None)
        if retry_ids is not None:
            return bool(retry_ids)
        return any(
            context.retry_count > 0
            for context in getattr(self, "ongoing_write_through", {}).values()
            if isinstance(context, WriteThroughContext)
        )

    def _release_prefetch_query_slot(self, context: PrefetchContext) -> None:
        if context.query_dispatched_tokens <= 0:
            return
        self.prefetch_query_ops_occupied = max(0, self.prefetch_query_ops_occupied - 1)
        self.prefetch_query_tokens_occupied = max(
            0,
            self.prefetch_query_tokens_occupied - context.query_dispatched_tokens,
        )
        context.query_dispatched_tokens = 0

    def _release_streaming_device_reservation(self, context: PrefetchContext) -> None:
        if context.device_reservation_tokens <= 0:
            return
        self.streaming_prefetch_reservation_tokens = max(
            0,
            getattr(self, "streaming_prefetch_reservation_tokens", 0)
            - context.device_reservation_tokens,
        )
        context.device_reservation_tokens = 0

    def _release_prefetch_context(
        self,
        context: PrefetchContext,
        *,
        free_completed: bool,
        free_all_host: bool = False,
    ) -> None:
        operation = context.operation
        self._release_prefetch_query_slot(context)
        self._release_streaming_device_reservation(context)
        streaming = bool(operation is not None and operation.stream_to_device)
        if context.host_indices is not None:
            if free_all_host or streaming:
                self.cache_controller.mem_pool_host.free(context.host_indices)
            elif free_completed:
                completed = min(
                    operation.completed_tokens if operation is not None else 0,
                    context.allocated_tokens,
                )
                if completed > 0:
                    self.cache_controller.mem_pool_host.free(
                        context.host_indices[:completed]
                    )
        if context.device_indices is not None:
            self.cache_controller.mem_pool_device_allocator.free(context.device_indices)
            context.device_indices = None
        if context.anchor_device_locked:
            self.dec_lock_ref(context.anchor)
            context.anchor_device_locked = False
        elif not getattr(self, "buffer_only", False):
            context.anchor.release_host()
        self.cache_controller.prefetch_tokens_occupied = max(
            0,
            self.cache_controller.prefetch_tokens_occupied - context.allocated_tokens,
        )
        context.host_indices = None
        context.allocated_tokens = 0
        if operation is not None:
            operation.host_indices = None
            operation.device_indices = None
            operation.device_ready_event = None
            getattr(self, "streaming_prefetch_active", set()).discard(
                operation.request_id
            )

    def _forget_transient_host_node(self, node: TreeNode) -> None:
        req_id = node.transient_owner_req_id
        node.transient_owner_req_id = None
        if req_id is None:
            return
        tracked = self.transient_host_nodes_by_reqid.get(req_id)
        if tracked is None:
            return
        self.transient_host_nodes_by_reqid[req_id] = [
            entry for entry in tracked if entry is not node
        ]
        if not self.transient_host_nodes_by_reqid[req_id]:
            self.transient_host_nodes_by_reqid.pop(req_id, None)

    def _release_recomputed_transient_host_value(self, node: TreeNode) -> None:
        if (
            not self.buffer_only
            or not node.host_value_transient
            or node.host_value is None
            or node.device_loading
        ):
            return
        self._forget_transient_host_node(node)
        self.cache_controller.mem_pool_host.free(node.host_value)
        node.host_value = None
        node.host_value_transient = False

    def _release_transient_host_nodes(self, req_id: str) -> None:
        nodes = self.transient_host_nodes_by_reqid.pop(req_id, [])
        for node in reversed(nodes):
            if node.transient_owner_req_id != req_id:
                continue
            node.transient_owner_req_id = None
            if (
                not node.host_value_transient
                or node.host_value is None
                or node.device_loading
                or node.host_ref_counter > 0
            ):
                continue
            if not node.evicted or node.children:
                continue

            self.cache_controller.mem_pool_host.free(node.host_value)
            node.host_value = None
            node.host_value_transient = False
            self.evictable_host_leaves.discard(node)
            parent = node.parent
            child_key = node.key.child_key(self.page_size)
            if parent is not None and parent.children.get(child_key) is node:
                parent.children.pop(child_key)
                self._update_leaf_status(parent)
                self._update_host_leaf_status(parent)

    def _trim_buffer_prefetch_against_device(self, context: PrefetchContext) -> bool:
        """Drop a newly resident L1 prefix from a queued streaming read.

        Multiple requests can query the same L3 prefix before the first read is
        published. Revalidating at admission prevents every later request from
        reading and copying the same pages again. The context lock is moved to
        the deepest contiguous device node so the remaining suffix stays safe.

        Returns True when enough queried L3 pages remain to justify a read.
        """
        operation = context.operation
        if operation is None or operation.host_indices is not None:
            return True
        if not context.anchor_device_locked or len(context.key) == 0:
            return operation.storage_hit_count >= self.prefetch_threshold

        node = context.anchor
        remaining = context.key
        matched_tokens = 0
        node.last_access_time = time.monotonic()

        while len(remaining) > 0:
            child_key = remaining.child_key(self.page_size)
            child = node.children.get(child_key)
            if child is None or child.evicted or child.device_loading:
                break

            prefix_len = child.key.match(remaining, page_size=self.page_size)
            if prefix_len <= 0:
                break
            if prefix_len < len(child.key):
                child = self._split_node(child.key, child, prefix_len)

            matched_tokens += prefix_len
            remaining = remaining[prefix_len:]
            node = child
            node.last_access_time = time.monotonic()
            if prefix_len < len(child.key):
                break

        matched_tokens -= matched_tokens % self.page_size
        if matched_tokens == 0:
            return operation.storage_hit_count >= self.prefetch_threshold

        matched_pages = min(
            matched_tokens // self.page_size,
            len(operation.hash_value),
        )
        matched_hashes = operation.hash_value[:matched_pages]

        self.inc_lock_ref(node)
        self.dec_lock_ref(context.anchor)
        context.anchor = node
        context.key = context.key[matched_tokens:]

        operation.hash_value = operation.hash_value[matched_pages:]
        operation.storage_hit_count = len(operation.hash_value) * self.page_size
        operation.token_ids = context.key
        if matched_hashes:
            context.last_hash = matched_hashes[-1]
            operation.last_hash = context.last_hash
        if context.prefix_keys is not None:
            context.prefix_keys = list(context.prefix_keys)
            context.prefix_keys.extend(matched_hashes)
            operation.prefix_keys = list(context.prefix_keys)

        return (
            operation.storage_hit_count >= self.prefetch_threshold
            and len(context.key) >= self.prefetch_threshold
        )

    def _lock_or_rebase_buffer_prefetch_anchor(
        self, context: PrefetchContext
    ) -> bool:
        """Lock the deepest still-resident ancestor before dispatching a query.

        Pending metadata queries intentionally do not pin L1. If their original
        anchor is evicted while waiting, prepend the detached radix path to the
        query suffix and continue from the nearest attached device ancestor.
        """
        if context.anchor_device_locked:
            return True

        chain = []
        cursor = context.anchor
        while cursor is not self.root_node:
            if cursor is None or cursor.parent is None:
                return False
            chain.append(cursor)
            cursor = cursor.parent
        chain.reverse()

        anchor = self.root_node
        attached_count = 0
        for candidate in chain:
            child_key = candidate.key.child_key(self.page_size)
            if (
                candidate.parent is not anchor
                or anchor.children.get(child_key) is not candidate
                or candidate.evicted
                or candidate.device_loading
            ):
                break
            anchor = candidate
            attached_count += 1

        detached = chain[attached_count:]
        if detached:
            parts = [node.key for node in detached]
            parts.append(context.key)
            is_bigram = context.key.is_bigram
            raw_tokens = []
            for part in parts:
                part_tokens = list(part.raw_token_ids())
                if not part_tokens:
                    continue
                if is_bigram and raw_tokens:
                    if raw_tokens[-1] != part_tokens[0]:
                        logger.warning(
                            "Cannot rebase buffer-only prefetch across a mismatched "
                            "bigram boundary"
                        )
                        return False
                    raw_tokens.extend(part_tokens[1:])
                else:
                    raw_tokens.extend(part_tokens)
            context.key = RadixKey(
                array("q", raw_tokens),
                extra_key=context.key.extra_key,
                is_bigram=is_bigram,
            ).page_aligned(self.page_size)

        if len(context.key) < self.prefetch_threshold:
            return False

        context.anchor = anchor
        context.last_hash = anchor.get_last_hash_value()
        context.prefix_keys = (
            anchor.get_prefix_hash_values(anchor.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )
        self.inc_lock_ref(anchor)
        context.anchor_device_locked = True
        return True

    def _minimum_buffer_prefetch_window_tokens(self) -> int:
        min_window_tokens = min(
            self.prefetch_threshold,
            getattr(
                self,
                "prefetch_stream_window_tokens",
                self.prefetch_threshold,
            ),
        )
        return max(
            self.page_size,
            (min_window_tokens + self.page_size - 1) // self.page_size * self.page_size,
        )

    def _buffer_prefetch_statically_impossible(self) -> bool:
        """Return whether configured L1/L2 capacity can never admit a stream."""
        controller = getattr(self, "cache_controller", None)
        host_pool = getattr(controller, "mem_pool_host", None)
        host_capacity = getattr(host_pool, "size", None)
        if isinstance(host_capacity, Integral):
            host_capacity = int(host_capacity)
            host_capacity -= host_capacity % self.page_size
            if host_capacity < self._minimum_buffer_prefetch_window_tokens():
                return True

        device_allocator = getattr(controller, "mem_pool_device_allocator", None)
        device_capacity = getattr(device_allocator, "size", None)
        if isinstance(device_capacity, Integral):
            device_capacity = int(device_capacity)
            device_capacity -= device_capacity % self.page_size
            if device_capacity < self.prefetch_threshold:
                return True

        return False

    def _allocate_buffer_prefetch(
        self,
        context: PrefetchContext,
        device_admission_tokens: Optional[int] = None,
        allow_device_allocation: bool = True,
    ) -> bool:
        operation = context.operation
        if operation is None:
            return False
        if operation.host_indices is not None:
            return True
        device_allocator = self.cache_controller.mem_pool_device_allocator
        local_device_available = device_allocator.available_size()
        local_device_budget = local_device_available + self.evictable_size()
        required_target = min(
            operation.storage_hit_count,
            len(context.key),
        )
        required_target -= required_target % self.page_size
        target = min(required_target, local_device_budget)
        target -= target % self.page_size

        available = self.cache_controller.mem_pool_host.available_size()
        if self.ongoing_write_through or self.ongoing_backup:
            reserve = int(
                self.buffer_write_reserve_ratio
                * self.cache_controller.mem_pool_host.size
            )
        else:
            reserve = 0
        allocatable = max(0, available - reserve)
        allocatable -= allocatable % self.page_size

        # MIN(-admission) is -MAX(admission). Including the local admission gate
        # keeps every rank in this collective even when allocator budgets drift.
        admission_tokens = max(0, int(device_admission_tokens or 0))
        allocation_allowed = bool(allow_device_allocation) and not bool(
            self.streaming_prefetch_active
        )
        limits = torch.tensor(
            [
                target,
                -required_target,
                allocatable,
                -admission_tokens,
                local_device_available,
                int(allocation_allowed),
            ],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(limits, torch.distributed.ReduceOp.MIN)
        (
            target,
            neg_required_target,
            allocatable,
            neg_admission_tokens,
            device_available,
            allocation_allowed,
        ) = map(int, limits.tolist())
        required_target = -neg_required_target
        admission_tokens = -neg_admission_tokens
        if allocation_allowed == 0:
            return False
        if context.require_full_completion and target != required_target:
            return False

        # This is the complete request budget checked by PrefillAdder before
        # streaming starts. The L3-hit portion is physically allocated below;
        # reserve only the remaining suffix/decode budget while the owner waits.
        min_window_tokens = self._minimum_buffer_prefetch_window_tokens()
        if target < self.prefetch_threshold or allocatable < min_window_tokens:
            return False

        window_tokens = min(
            target,
            allocatable,
            self.prefetch_stream_window_tokens,
        )
        window_tokens -= window_tokens % self.page_size
        if window_tokens < min_window_tokens:
            return False
        host_tokens = min(
            target,
            allocatable,
            window_tokens * self.prefetch_stream_buffer_count,
        )
        host_tokens -= host_tokens % self.page_size

        host_indices = self.cache_controller.mem_pool_host.alloc(host_tokens)
        allocated = torch.tensor(int(host_indices is not None), dtype=torch.int)
        self._all_reduce_attn_groups(allocated, torch.distributed.ReduceOp.MIN)
        if allocated.item() == 0:
            if host_indices is not None:
                self.cache_controller.mem_pool_host.free(host_indices)
            return False

        if target > device_available:
            # Evict the same logical radix prefix on every attention rank even
            # if their allocator free counts have drifted temporarily.
            self.evict(EvictParams(num_tokens=target - device_available))
        device_indices = device_allocator.alloc(target)
        device_allocated = torch.tensor(
            int(device_indices is not None), dtype=torch.int
        )
        self._all_reduce_attn_groups(device_allocated, torch.distributed.ReduceOp.MIN)
        if device_allocated.item() == 0:
            if device_indices is not None:
                device_allocator.free(device_indices)
            self.cache_controller.mem_pool_host.free(host_indices)
            return False

        setup_failed = False
        try:
            device_ready_event = self.cache_controller.record_device_ready_event()
            operation.configure_device_streaming(
                host_indices,
                device_indices,
                window_tokens,
                device_ready_event,
            )
        except Exception:
            setup_failed = True
            logger.exception(
                "Failed to initialize streaming prefetch for request %s",
                operation.request_id,
            )
        setup_ready = torch.tensor(int(not setup_failed), dtype=torch.int)
        self._all_reduce_attn_groups(setup_ready, torch.distributed.ReduceOp.MIN)
        if setup_ready.item() == 0:
            operation.clear_device_streaming()
            device_allocator.free(device_indices)
            self.cache_controller.mem_pool_host.free(host_indices)
            return False

        context.host_indices = host_indices
        context.device_indices = device_indices
        context.allocated_tokens = host_tokens
        context.device_reservation_tokens = max(0, admission_tokens - target)
        self.streaming_prefetch_reservation_tokens = (
            getattr(self, "streaming_prefetch_reservation_tokens", 0)
            + context.device_reservation_tokens
        )
        context.key = context.key[:target]
        operation.hash_value = operation.hash_value[: target // self.page_size]
        operation.token_ids = operation.token_ids[:target]
        operation.storage_hit_count = target
        operation.mark_io_started()
        # Keep query slots occupied while hit metadata is waiting for the sole
        # data stream. Releasing here admits bounded look-ahead without building
        # an unbounded queue of completed L3 queries.
        self._release_prefetch_query_slot(context)
        self.cache_controller.prefetch_tokens_occupied += host_tokens
        self.streaming_prefetch_active.add(operation.request_id)
        self.cache_controller.prefetch_buffer.put(operation)
        return True

    def _dispatch_pending_prefetch_queries(
        self, pending_count: int, *, pause_for_write: bool = False
    ) -> None:
        if not self.buffer_only or pause_for_write:
            return

        for _ in range(min(pending_count, len(self.pending_prefetch_query))):
            if self.prefetch_query_ops_occupied >= self.prefetch_query_max_ops:
                break
            req_id = self.pending_prefetch_query.popleft()
            context = self.ongoing_prefetch.get(req_id)
            if context is None or context.aborted or context.operation is not None:
                continue

            if not self._lock_or_rebase_buffer_prefetch_anchor(context):
                self.ongoing_prefetch.pop(req_id, None)
                self._release_prefetch_context(context, free_completed=True)
                continue

            # Hash queries carry metadata only. L2 capacity is enforced later
            # by bounded streaming windows, so querying the full suffix does
            # not need to reserve data pages.
            target = len(context.key)
            target -= target % self.page_size
            if target < self.prefetch_threshold:
                self.ongoing_prefetch.pop(req_id, None)
                self._release_prefetch_context(context, free_completed=True)
                continue
            if (
                self.prefetch_query_ops_occupied > 0
                and self.prefetch_query_tokens_occupied + target
                > self.prefetch_query_max_tokens
            ):
                self.dec_lock_ref(context.anchor)
                context.anchor_device_locked = False
                self.pending_prefetch_query.appendleft(req_id)
                break

            context.key = context.key[:target]
            try:
                operation = self.cache_controller.prefetch(
                    req_id,
                    None,
                    context.key,
                    context.last_hash,
                    context.prefix_keys,
                    **self._get_extra_pools(),
                )
                if operation is None:
                    raise RuntimeError("HiCache prefetch dispatcher returned no operation")
            except Exception:
                logger.exception("Failed to dispatch L3 metadata query for %s", req_id)
                self.ongoing_prefetch.pop(req_id, None)
                self._release_prefetch_context(context, free_completed=True)
                continue
            context.operation = operation
            context.query_dispatched_tokens = target
            self.prefetch_query_ops_occupied += 1
            self.prefetch_query_tokens_occupied += target

    def _reap_aborted_prefetches(self, pending_count: Optional[int] = None) -> None:
        pending_aborts = getattr(self, "aborted_prefetch_req_ids", None)
        if pending_aborts is None:
            pending_aborts = {
                req_id
                for req_id, context in self.ongoing_prefetch.items()
                if context.aborted
            }
        req_ids = sorted(pending_aborts)
        if pending_count is not None:
            req_ids = req_ids[:pending_count]
        for req_id in req_ids:
            context = self.ongoing_prefetch.get(req_id)
            if context is None or not context.aborted:
                pending_aborts.discard(req_id)
                continue
            operation = context.operation
            if operation is None:
                self.ongoing_prefetch.pop(req_id, None)
                self._release_prefetch_context(context, free_completed=True)
                pending_aborts.discard(req_id)
                continue
            if operation.host_indices is None:
                if context.query_ready:
                    self.ongoing_prefetch.pop(req_id, None)
                    self._release_prefetch_context(context, free_completed=True)
                    pending_aborts.discard(req_id)
                    continue
                # Query hit/revoke queues are drained with a TP-min count. Let
                # that synchronized handoff release the context and query slot.
                continue
            io_done = torch.tensor(int(operation.io_done), dtype=torch.int)
            self._all_reduce_attn_groups(io_done, torch.distributed.ReduceOp.MIN)
            if io_done.item() == 0:
                continue
            self.ongoing_prefetch.pop(req_id, None)
            self._release_prefetch_context(context, free_completed=True)
            pending_aborts.discard(req_id)

    def _prefetch_timeout_seconds(self, operation: PrefetchOperation) -> float:
        cfg = self.prefetch_timeout_config
        num_tokens = len(operation.hash_value) * self.page_size
        return min(cfg.max, cfg.base + cfg.per_ki_token * num_tokens / 1024)

    def _is_prefetch_allocation_timeout(self, operation: PrefetchOperation) -> bool:
        start_time = operation.query_done_time or operation.start_time
        return time.monotonic() - start_time > self._prefetch_timeout_seconds(operation)

    # Timeout is linearly increasing with the number of pages.
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        start_time = (
            operation.io_start_time
            if getattr(operation, "query_first", False)
            and getattr(operation, "io_start_time", None) is not None
            else operation.start_time
        )
        return time.monotonic() - start_time > self._prefetch_timeout_seconds(operation)

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        if getattr(operation, "query_first", False):
            # Query-first pages are owned by the I/O thread until it reports done.
            # Timeout/best-effort may request termination, but cannot free in-use pages.
            io_done = torch.tensor(
                int(operation.host_indices is not None and operation.io_done),
                dtype=torch.int,
            )
            self._all_reduce_attn_groups(io_done, torch.distributed.ReduceOp.MIN)
            return io_done.item() == 1

        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        if (
            completed
            and getattr(operation, "pool_transfers", None)
            and not getattr(operation, "pool_transfers_done", True)
        ):
            can_terminate = False

        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MAX)
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(
        self,
        req_id: str,
        *,
        allow_device_allocation: bool = True,
        device_admission_tokens: Optional[int] = None,
    ) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        context = self.ongoing_prefetch[req_id]
        operation = context.operation

        if operation is None:
            return False

        if operation.host_indices is None:
            if not self.buffer_only:
                return True
            if not context.query_ready:
                return False
            # Stream setup/completion is TP-synchronized, so this set is a
            # replicated global gate. Check the cheap gates before walking the
            # radix tree: otherwise every ready request rematches its full
            # prefix on every scheduler tick while another stream is active.
            if getattr(self, "streaming_prefetch_active", None):
                return False
            if getattr(self, "buffer_prefetch_paused_for_write", False):
                return False
            if not self._trim_buffer_prefetch_against_device(context):
                operation.mark_terminate()
                self.ongoing_prefetch.pop(req_id, None)
                self._release_prefetch_context(context, free_completed=True)
                return True
            if operation.storage_hit_count < self.prefetch_threshold:
                operation.mark_terminate()
                self.ongoing_prefetch.pop(req_id, None)
                self._release_prefetch_context(context, free_completed=True)
                return True

            if not self._allocate_buffer_prefetch(
                context,
                device_admission_tokens=device_admission_tokens,
                allow_device_allocation=allow_device_allocation,
            ):
                cancel_local = self._buffer_prefetch_statically_impossible() or (
                    not context.require_full_completion
                    and (
                        self.prefetch_stop_policy == "best_effort"
                        or (
                            self.prefetch_stop_policy == "timeout"
                            and self._is_prefetch_allocation_timeout(operation)
                        )
                    )
                )
                cancel = torch.tensor(int(cancel_local), dtype=torch.int)
                self._all_reduce_attn_groups(cancel, torch.distributed.ReduceOp.MAX)
                if cancel.item() == 0:
                    return False

                operation.mark_terminate()
                self.ongoing_prefetch.pop(req_id, None)
                self._release_prefetch_context(context, free_completed=True)
                return True

        if (
            getattr(operation, "query_first", False)
            and not context.require_full_completion
        ):
            terminate_local = (
                self.prefetch_stop_policy == "best_effort"
                and (operation.completed_tokens > 0 or operation.io_done)
            ) or (
                self.prefetch_stop_policy == "timeout"
                and self.is_prefetch_timeout(operation)
            )
            if terminate_local:
                operation.mark_terminate()

        if context.require_full_completion and not getattr(
            operation, "query_first", False
        ):
            # PD prefill has already omitted the promised prefix from its
            # transfer. It cannot be recomputed on decode, so wait until the
            # local I/O worker has either restored the full prefix or stopped
            # after a real backend failure. Do not apply best-effort/timeout.
            io_done = torch.tensor(int(operation.io_done), dtype=torch.int)
            self._all_reduce_attn_groups(io_done, torch.distributed.ReduceOp.MIN)
            if io_done.item() == 0:
                return False
        elif not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        # Synchronize workers before mutating host cache tree state.
        completed_tokens_tensor = torch.tensor(min_completed_tokens, dtype=torch.int)
        self._all_reduce_attn_groups(
            completed_tokens_tensor, torch.distributed.ReduceOp.MIN
        )
        min_completed_tokens = completed_tokens_tensor.item()
        min_completed_tokens = min(
            min_completed_tokens,
            len(context.key),
        )
        min_completed_tokens -= min_completed_tokens % self.page_size

        if operation.stream_to_device:
            device_indices = context.device_indices
            if device_indices is None:
                raise RuntimeError("Streaming prefetch lost its device allocation")

            fetched_key = context.key[:min_completed_tokens]
            has_loading_nodes = bool(getattr(self, "loading_node_to_ack", None))
            if has_loading_nodes and not self._synchronize_loading_prefix(
                fetched_key, start_node=context.anchor
            ):
                # load_back() has published its provisional device indices but
                # ready_to_load_host_cache() has not attached a completion event
                # yet. Keep this stream owned by its context and retry next tick.
                return False

            if min_completed_tokens < len(device_indices):
                self.cache_controller.mem_pool_device_allocator.free(
                    device_indices[min_completed_tokens:]
                )
            matched_length = self._insert_helper_device_prefetch(
                context.anchor,
                fetched_key,
                device_indices[:min_completed_tokens],
                hash_value[: min_completed_tokens // self.page_size],
            )

            self.ongoing_prefetch.pop(req_id, None)
            # Published indices are now owned by the radix tree. Clear the
            # context pointer so common cleanup only frees the unused L2 window.
            context.device_indices = None
            self._release_prefetch_context(context, free_completed=True)

            loaded_from_storage = min_completed_tokens - matched_length
            self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
            if self.enable_storage_metrics:
                self.storage_metrics_collector.log_prefetched_tokens(
                    loaded_from_storage
                )
            return True

        host_indices = context.host_indices
        fetched_key = context.key[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            context.anchor,
            fetched_key,
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
            owner_req_id=req_id if self.buffer_only else None,
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        if context.anchor_device_locked:
            self.dec_lock_ref(context.anchor)
        else:
            context.anchor.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= context.allocated_tokens
        context.host_indices = None
        if getattr(operation, "query_first", False):
            operation.host_indices = None

        # Track tokens actually loaded from storage for this request (L3 hits)
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        operation = self.ongoing_prefetch[req_id].operation
        if operation is None or operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def match_prefix(self, params: MatchPrefixParams):
        if self.disable:
            return self._empty_match_result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            best_match_node=(last_host_node if host_hit_length > 0 else last_node),
            host_hit_length=host_hit_length,
        )

    def cache_unfinished_req(self, req, chunked=False):
        if self.ongoing_load_back:
            key = RadixKey(
                req.get_fill_ids(), req.extra_key, is_bigram=self.is_eagle
            ).page_aligned(self.page_size)
            if not self._synchronize_loading_prefix(key):
                # load_back() publishes provisional device indices before the
                # merged H2D is started. Start any queued transfer here so model
                # output cannot be inserted across an unfinished radix node.
                self.ready_to_load_host_cache()
                if not self._synchronize_loading_prefix(key):
                    raise RuntimeError(
                        "Cannot cache request across an untracked load-back"
                    )
        return super().cache_unfinished_req(req, chunked=chunked)

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        known_hash_values: Optional[List[str]] = None,
        require_full_completion: bool = False,
    ):
        if req_id in self.ongoing_prefetch:
            return

        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        )
        # align the number of fetching tokens to the page size
        prefetch_key = prefetch_key.page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or (not self.buffer_only and self.cache_controller.prefetch_rate_limited())
            or (self.buffer_only and self._buffer_prefetch_statically_impossible())
        ):
            return

        if self.buffer_only:
            if known_hash_values is not None:
                known_tokens = min(
                    prefetch_length,
                    len(known_hash_values) * self.page_size,
                )
                known_tokens -= known_tokens % self.page_size
                if known_tokens < self.prefetch_threshold:
                    return
                prefetch_key = prefetch_key[:known_tokens]
                known_hash_values = known_hash_values[: known_tokens // self.page_size]

            # A direct L3->L1 stream extends the contiguous device radix prefix.
            # An L2-only node cannot be an anchor because its device parent path
            # is not visible to regular prefix matching.
            if last_host_node.evicted:
                return
            anchor_device_locked = known_hash_values is not None
            if anchor_device_locked:
                # PD has already queried L3 synchronously and is about to commit
                # its transfer boundary, so its anchor must remain resident.
                self.inc_lock_ref(last_host_node)
            context = PrefetchContext(
                anchor=last_host_node,
                key=prefetch_key,
                anchor_device_locked=anchor_device_locked,
                last_hash=last_hash,
                prefix_keys=prefix_keys,
                require_full_completion=require_full_completion,
            )
            self.ongoing_prefetch[req_id] = context
            if known_hash_values is None:
                self.pending_prefetch_query.append(req_id)
            else:
                try:
                    context.operation = (
                        self.cache_controller.prefetch_with_query_result(
                            req_id,
                            None,
                            prefetch_key,
                            known_hash_values,
                            last_hash,
                            prefix_keys,
                        )
                    )
                    # query_storage_hit() already synchronized the hit length in
                    # the scheduler thread. Avoid mixing this handoff into the
                    # asynchronously produced hit queue, whose cross-rank FIFO
                    # order could otherwise differ.
                    context.query_ready = True
                except Exception:
                    self.ongoing_prefetch.pop(req_id, None)
                    self._release_prefetch_context(
                        context,
                        free_completed=True,
                    )
                    raise
            return

        last_host_node.protect_host()
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            available_size = self.cache_controller.mem_pool_host.available_size()
            prefetch_length = available_size - (available_size % self.page_size)
            if prefetch_length >= self.prefetch_threshold:
                prefetch_key = prefetch_key[:prefetch_length]
                host_indices = self.cache_controller.mem_pool_host.alloc(
                    prefetch_length
                )
                if host_indices is None:
                    last_host_node.release_host()
                    return
            else:
                last_host_node.release_host()
                # no sufficient host memory for prefetch
                return
        try:
            operation = self.cache_controller.prefetch(
                req_id,
                host_indices,
                prefetch_key,
                last_hash,
                prefix_keys,
                **self._get_extra_pools(),
            )
            if operation is None:
                raise RuntimeError("HiCache prefetch dispatcher returned no operation")
        except Exception:
            self.cache_controller.mem_pool_host.free(host_indices)
            last_host_node.release_host()
            logger.exception("Failed to dispatch L3 prefetch for %s", req_id)
            return
        self.ongoing_prefetch[req_id] = PrefetchContext(
            anchor=last_host_node,
            key=prefetch_key,
            host_indices=host_indices,
            operation=operation,
            allocated_tokens=len(prefetch_key),
            require_full_completion=require_full_completion,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _insert_helper_host(
        self,
        node: TreeNode,
        key: RadixKey,
        host_value,
        hash_value,
        owner_req_id: Optional[str] = None,
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)
            node.storage_backed_tokens = max(
                node.storage_backed_tokens, min(prefix_len, len(node.key))
            )
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.host_value_transient = self.buffer_only
            new_node.transient_owner_req_id = owner_req_id
            new_node.storage_backed_tokens = len(key)
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            # Publish the newly materialized host suffix immediately so downstream
            # cache indexers can resolve descendants that extend this L2-only prefix.
            if not self.buffer_only:
                self._record_store_event(new_node, medium=StorageMedium.CPU)
            elif owner_req_id is not None:
                self.transient_host_nodes_by_reqid.setdefault(owner_req_id, []).append(
                    new_node
                )

        return matched_length

    def _insert_helper_device_prefetch(
        self,
        node: TreeNode,
        key: RadixKey,
        device_value: torch.Tensor,
        hash_value: List[str],
    ) -> int:
        """Publish a completed streamed prefix without retaining an L2 copy.

        Returns the number of streamed tokens that were already resident on
        device by the time this operation was published. Their newly reserved
        indices are freed here instead of being attached to the radix tree.
        """
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0
        if len(device_value) != len(key):
            raise RuntimeError(
                "Streaming prefetch key/device length mismatch: "
                f"key={len(key)}, device={len(device_value)}"
            )
        expected_hashes = len(key) // self.page_size
        if len(hash_value) != expected_hashes:
            raise RuntimeError(
                "Streaming prefetch key/hash length mismatch: "
                f"pages={expected_hashes}, hashes={len(hash_value)}"
            )

        matched_device_tokens = 0
        child_key = key.child_key(self.page_size)
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            if child.device_loading:
                raise RuntimeError(
                    "Streaming prefetch cannot publish across an active load-back"
                )
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len <= 0:
                break

            segment = device_value[:prefix_len]
            segment_hashes = hash_value[: prefix_len // self.page_size]
            if prefix_len < len(child.key):
                child = self._split_node(child.key, child, prefix_len)

            child.storage_backed_tokens = max(child.storage_backed_tokens, prefix_len)
            if child.hash_value is None:
                child.hash_value = list(segment_hashes)

            if child.evicted:
                child.value = segment.clone()
                self._release_recomputed_transient_host_value(child)
                self.evictable_size_ += prefix_len
                self._update_leaf_status(child)
                self._update_host_leaf_status(child)
                self._update_leaf_status(child.parent)
                self._update_host_leaf_status(child.parent)
                self._record_store_event(child, medium=StorageMedium.GPU)
            else:
                self.cache_controller.mem_pool_device_allocator.free(segment)
                matched_device_tokens += prefix_len

            key = key[prefix_len:]
            device_value = device_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            node = child
            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = device_value.clone()
            new_node.storage_backed_tokens = len(key)
            new_node.hash_value = list(hash_value)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(new_node)
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            self._record_store_event(new_node, medium=StorageMedium.GPU)

        return matched_device_tokens

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            if child.device_loading:
                break
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count
        new_node.device_loading = child.device_loading
        new_node.host_value_transient = child.host_value_transient
        new_node.transient_owner_req_id = child.transient_owner_req_id

        backed_prefix = child.storage_backed_tokens
        new_node.storage_backed_tokens = min(backed_prefix, split_len)
        child.storage_backed_tokens = max(0, backed_prefix - split_len)

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        if child.write_through_pending_id is not None:
            self._replace_pending_write_through_node(child, [new_node, child])
        if child.device_loading:
            self._replace_pending_load_back_node(child, [new_node, child])
        if child.transient_owner_req_id is not None:
            tracked = self.transient_host_nodes_by_reqid.get(
                child.transient_owner_req_id, []
            )
            for index, node in enumerate(tracked):
                if node is child:
                    tracked[index : index + 1] = [new_node, child]
                    break

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        node = self.root_node
        child_key = key.child_key(self.page_size)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    self._release_recomputed_transient_host_value(node)
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self._release_recomputed_transient_host_value(new_node)
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage or kv events are enabled
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # Emit BlockStored so the router indexes this block.
            self._record_store_event(new_node)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self._release_transient_host_nodes(rid)

        if rid not in self.ongoing_prefetch:
            return

        context = self.ongoing_prefetch[rid]
        operation = context.operation
        if self.buffer_only:
            context.aborted = True
            self.aborted_prefetch_req_ids.add(rid)
            if operation is not None:
                operation.mark_terminate()
            # Periodic check_hicache_events() reaps this on every attention
            # rank in the same order. Reaping inline could enter collectives
            # from a rank-local abort callback and deadlock TP workers.
            return

        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups()
        context.anchor.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(
            context.host_indices[:completed_tokens]
        )
        self.cache_controller.prefetch_tokens_occupied -= context.allocated_tokens
