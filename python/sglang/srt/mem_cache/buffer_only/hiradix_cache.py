from __future__ import annotations

import heapq
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.managers.cache_controller import PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    InsertResult,
)
from sglang.srt.mem_cache.buffer_only.cache_controller import (
    BufferOnlyHiCacheController,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    split_node_hash_value,
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
    query_tokens_reserved: int = 0
    last_hash: Optional[str] = None
    prefix_keys: Optional[List[str]] = None


class BufferTreeNode(TreeNode):
    """Radix node state used exclusively by the buffer-only backend."""

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        super().__init__(id=id, priority=priority)
        self.children = defaultdict(BufferTreeNode)
        self.host_value_transient = False
        self.transient_owner_req_id: Optional[str] = None
        self.storage_backed_tokens = 0
        self.device_loading = False

    @property
    def storage_backed(self) -> bool:
        return self.parent is None or self.storage_backed_tokens >= len(self.key)


class BufferOnlyHiRadixCache(HiRadixCache):
    """HiCache variant that uses host memory only for in-flight transfers."""

    buffer_only = True
    cache_controller_cls = BufferOnlyHiCacheController
    node_cls = BufferTreeNode

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        if (
            server_args.hicache_storage_backend is None
            or server_args.hicache_write_policy != "write_through"
        ):
            raise ValueError(
                "HiCache buffer_only requires an L3 backend and write_through."
            )
        kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        if not isinstance(kv_cache, MHATokenToKVPool):
            raise ValueError(
                "HiCache buffer_only currently supports MHA/GQA KV pools only; "
                f"got {type(kv_cache).__name__}."
            )

        # RadixCache.__init__ invokes self.reset(), so every field used by this
        # subclass' reset must exist before entering the base constructor.
        self.loading_node_to_ack: Dict[TreeNode, int] = {}
        self.pending_write_through = deque()
        self.pending_prefetch_query = deque()
        self.pending_prefetch_alloc = deque()
        self.transient_host_nodes_by_reqid: Dict[str, List[TreeNode]] = {}
        self.buffer_write_reserve_ratio = 0.2
        self.buffer_write_dispatch_limit = 64
        self.buffer_query_tokens_occupied = 0

        super().__init__(params=params, server_args=server_args)
        self.host_memory_mode = "buffer_only"

    def resolve_storage_prefetch_anchor(self, req) -> Optional[TreeNode]:
        node = req.last_host_node if req.host_hit_length > 0 else req.last_node
        if node.backuped or node is self.root_node or not node.evicted:
            return node
        return None

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
                    self._drain_storage_control_queues_local()
                    self._force_release_pending_storage_ops()
                    self.enable_storage = False
                else:
                    self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        if hicache_write_policy not in (None, "write_through"):
            return False, "buffer_only only supports write_through."
        return super().attach_storage_backend(
            storage_backend=storage_backend,
            storage_backend_extra_config_json=storage_backend_extra_config_json,
            served_model_name=served_model_name,
            hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
            hicache_write_policy=hicache_write_policy,
        )

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
        self.pending_prefetch_alloc.clear()

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
                self.pending_prefetch_alloc.append(operation.request_id)

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
        self.pending_prefetch_query.clear()
        self.pending_prefetch_alloc.clear()
        self.buffer_query_tokens_occupied = 0
        self.transient_host_nodes_by_reqid.clear()
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        # This method replaces HiRadixCache.reset rather than extending it.
        # Calling super() here would reset the controller and restart its
        # storage threads twice.
        RadixCache.reset(self)

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
        else:
            context.retry_count += 1
            delay = min(5.0, 0.1 * (2 ** min(context.retry_count - 1, 6)))
            context.retry_at = time.monotonic() + delay
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

    def _synchronize_loading_prefix(self, key: RadixKey) -> None:
        if not self.loading_node_to_ack:
            return

        needed_ack_ids = set()
        node = self.root_node
        remaining_key = key
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
            return

        finish_through = -1
        for index, (_, _, ack_list) in enumerate(self.cache_controller.ack_load_queue):
            if needed_ack_ids.intersection(ack_list):
                finish_through = index
        if finish_through < 0:
            logger.error(
                "Missing load-back completion event for radix nodes: %s",
                sorted(needed_ack_ids),
            )
            return

        # The load stream is FIFO. Completing through the latest matching event
        # also makes every earlier event safe to retire locally.
        for _ in range(finish_through + 1):
            _, finish_event, ack_list = self.cache_controller.ack_load_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                self._finish_load_back_ack(ack_id)

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

    def check_hicache_events(self):
        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
            self._reap_aborted_prefetches()
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
                len(self.pending_prefetch_alloc),
                max(
                    0,
                    cc.mem_pool_host.size - self.buffer_query_tokens_occupied,
                ),
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
            n_pending_prefetch,
            query_budget,
        ) = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_hit=n_hit,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )
        if self.buffer_only:
            # Writes get first use of pages released by the queues above. Reads
            # then consume the remainder subject to the soft write reserve.
            self._dispatch_pending_buffer_writes(n_pending_write + n_backup)
            self._dispatch_pending_prefetch_alloc(n_pending_prefetch + n_hit)
            self._dispatch_pending_prefetch_queries(n_pending_query, query_budget)

    def _release_prefetch_context(
        self,
        context: PrefetchContext,
        *,
        free_completed: bool,
        free_all_host: bool = False,
    ) -> None:
        operation = context.operation
        if context.host_indices is not None:
            if free_all_host:
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
        if context.anchor_device_locked:
            self.dec_lock_ref(context.anchor)
            context.anchor_device_locked = False
        else:
            context.anchor.release_host()
        self.cache_controller.prefetch_tokens_occupied = max(
            0,
            self.cache_controller.prefetch_tokens_occupied - context.allocated_tokens,
        )
        context.host_indices = None
        context.allocated_tokens = 0
        if operation is not None:
            operation.host_indices = None
        self._release_query_reservation(context)

    def _release_query_reservation(self, context: PrefetchContext) -> None:
        if context.query_tokens_reserved == 0:
            return
        self.buffer_query_tokens_occupied = max(
            0,
            self.buffer_query_tokens_occupied - context.query_tokens_reserved,
        )
        context.query_tokens_reserved = 0

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

    def _allocate_buffer_prefetch(self, context: PrefetchContext) -> bool:
        operation = context.operation
        if operation is None:
            return False
        if operation.host_indices is not None:
            return True

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
        target = min(operation.storage_hit_count, allocatable)
        target_tensor = torch.tensor(target, dtype=torch.int)
        self._all_reduce_attn_groups(target_tensor, torch.distributed.ReduceOp.MIN)
        target = int(target_tensor.item())
        if target < self.prefetch_threshold:
            return False

        host_indices = self.cache_controller.mem_pool_host.alloc(target)
        allocated = torch.tensor(int(host_indices is not None), dtype=torch.int)
        self._all_reduce_attn_groups(allocated, torch.distributed.ReduceOp.MIN)
        if allocated.item() == 0:
            if host_indices is not None:
                self.cache_controller.mem_pool_host.free(host_indices)
            return False

        context.host_indices = host_indices
        context.allocated_tokens = target
        self._release_query_reservation(context)
        operation.host_indices = host_indices
        operation.hash_value = operation.hash_value[: target // self.page_size]
        operation.mark_io_started()
        self.cache_controller.prefetch_tokens_occupied += target
        self.cache_controller.prefetch_buffer.put(operation)
        return True

    def _dispatch_pending_prefetch_queries(
        self, pending_count: int, query_budget: int
    ) -> None:
        if not self.buffer_only or query_budget < self.prefetch_threshold:
            return

        pool_capacity = self.cache_controller.mem_pool_host.size
        for _ in range(min(pending_count, len(self.pending_prefetch_query))):
            req_id = self.pending_prefetch_query.popleft()
            context = self.ongoing_prefetch.get(req_id)
            if context is None or context.aborted or context.operation is not None:
                continue

            target = min(len(context.key), pool_capacity, query_budget)
            target -= target % self.page_size
            if target < self.prefetch_threshold:
                self.pending_prefetch_query.appendleft(req_id)
                break

            context.key = context.key[:target]
            operation = self.cache_controller.prefetch(
                req_id,
                None,
                context.key,
                context.last_hash,
                context.prefix_keys,
                **self._get_extra_pools(),
            )
            context.operation = operation
            context.query_tokens_reserved = target
            self.buffer_query_tokens_occupied += target
            query_budget -= target
            if query_budget < self.prefetch_threshold:
                break

    def _dispatch_pending_prefetch_alloc(self, pending_count: int) -> None:
        if not self.buffer_only:
            return

        for _ in range(min(pending_count, len(self.pending_prefetch_alloc))):
            req_id = self.pending_prefetch_alloc.popleft()
            context = self.ongoing_prefetch.get(req_id)
            if context is None or context.aborted:
                continue
            if not self._allocate_buffer_prefetch(context):
                cancel_local = self.prefetch_stop_policy == "best_effort" or (
                    self.prefetch_stop_policy == "timeout"
                    and self._is_prefetch_allocation_timeout(context.operation)
                )
                cancel = torch.tensor(int(cancel_local), dtype=torch.int)
                self._all_reduce_attn_groups(cancel, torch.distributed.ReduceOp.MAX)
                if cancel.item() == 1:
                    context.operation.mark_terminate()
                    self.ongoing_prefetch.pop(req_id, None)
                    self._release_prefetch_context(context, free_completed=True)
                    continue
                self.pending_prefetch_alloc.append(req_id)
                break

    def _reap_aborted_prefetches(self) -> None:
        for req_id, context in list(self.ongoing_prefetch.items()):
            if not context.aborted:
                continue
            operation = context.operation
            if operation is None:
                self.ongoing_prefetch.pop(req_id, None)
                self._release_prefetch_context(context, free_completed=True)
                continue
            if operation.host_indices is None:
                if not operation.query_done:
                    continue
            elif not operation.io_done:
                continue
            self.ongoing_prefetch.pop(req_id, None)
            self._release_prefetch_context(context, free_completed=True)

    def _prefetch_timeout_seconds(self, operation: PrefetchOperation) -> float:
        cfg = self.prefetch_timeout_config
        num_tokens = len(operation.hash_value) * self.page_size
        return min(cfg.max, cfg.base + cfg.per_ki_token * num_tokens / 1024)

    def _is_prefetch_allocation_timeout(self, operation: PrefetchOperation) -> bool:
        start_time = operation.query_done_time or operation.start_time
        return time.monotonic() - start_time > self._prefetch_timeout_seconds(operation)

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

    def check_prefetch_progress(self, req_id: str) -> bool:
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
            return False if self.buffer_only else True

        if getattr(operation, "query_first", False):
            terminate_local = (
                self.prefetch_stop_policy == "best_effort"
                and (operation.completed_tokens > 0 or operation.io_done)
            ) or (
                self.prefetch_stop_policy == "timeout"
                and self.is_prefetch_timeout(operation)
            )
            if terminate_local:
                operation.mark_terminate()

        if not self.can_terminate_prefetch(operation):
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

    def cache_unfinished_req(self, req, chunked=False):
        if self.ongoing_load_back:
            key = RadixKey(
                req.get_fill_ids(), req.extra_key, is_bigram=self.is_eagle
            ).page_aligned(self.page_size)
            self._synchronize_loading_prefix(key)
        return super().cache_unfinished_req(req, chunked=chunked)

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
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
        ):
            return

        if self.buffer_only:
            anchor_device_locked = not last_host_node.evicted
            if anchor_device_locked:
                self.inc_lock_ref(last_host_node)
            else:
                last_host_node.protect_host()
            self.ongoing_prefetch[req_id] = PrefetchContext(
                anchor=last_host_node,
                key=prefetch_key,
                anchor_device_locked=anchor_device_locked,
                last_hash=last_hash,
                prefix_keys=prefix_keys,
            )
            self.pending_prefetch_query.append(req_id)
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
        operation = self.cache_controller.prefetch(
            req_id,
            host_indices,
            prefetch_key,
            last_hash,
            prefix_keys,
            **self._get_extra_pools(),
        )
        self.ongoing_prefetch[req_id] = PrefetchContext(
            anchor=last_host_node,
            key=prefetch_key,
            host_indices=host_indices,
            operation=operation,
            allocated_tokens=len(prefetch_key),
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
            new_node = self.node_cls(priority=node.priority)
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
        new_node = self.node_cls(priority=child.priority)
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
            new_node = self.node_cls(priority=priority)
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
            if operation is not None:
                operation.mark_terminate()
            self._reap_aborted_prefetches()
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
