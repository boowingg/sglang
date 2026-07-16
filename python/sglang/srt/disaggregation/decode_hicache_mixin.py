"""HiCache integration mixins for the decode side of PD disaggregation"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    l3_hash_values: Optional[List[str]] = None
    prefetch_registered: bool = False
    l2_fallback_hit_length: int = 0

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        """Number of tokens that need L2/L3 load_back to device."""
        return self.decode_prefix_len - self.l1_prefix_len


class HiCacheRestoreResult(Enum):
    """Outcome of one tick of the HiCache local-restore state machine."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DecodeHiCachePreallocMixin:
    """HiCache hooks for ``DecodePreallocQueue``: issue prefetch + reserve tokens."""

    def _build_decode_prefix_match(self, req: Req, result: Any) -> DecodePrefixMatch:
        """Convert a ``match_prefix_for_req`` result into ``DecodePrefixMatch``.

        Performs the optional L3 storage hit length query when decode-side
        HiCache is enabled and the last host node is backed up.
        """
        prefix_indices = result.device_indices
        l1_prefix_len = len(prefix_indices)
        l2_host_hit_length = result.host_hit_length

        l3_storage_hit_length = 0
        l3_hash_values: List[str] = []
        l2_fallback_hit_length = l2_host_hit_length
        last_host_node = None
        if self.scheduler.enable_decode_hicache:
            buffer_only = getattr(self.tree_cache, "buffer_only", False)
            last_host_node = (
                result.last_device_node if buffer_only else result.last_host_node
            )
            can_anchor_prefetch = (
                not last_host_node.evicted
                if buffer_only
                else (
                    last_host_node.backuped
                    or last_host_node is self.tree_cache.root_node
                )
            )
            if can_anchor_prefetch:
                # Streaming must start at a contiguous device prefix. If an old
                # transient L2 entry is present, query it again from L3 together
                # with the following suffix instead of publishing a device child
                # below an evicted radix node.
                matched_len = (
                    l1_prefix_len if buffer_only else l1_prefix_len + l2_host_hit_length
                )
                suffix_tokens = req.origin_input_ids[matched_len:]
                last_hash = last_host_node.get_last_hash_value()
                prefix_keys = (
                    last_host_node.get_prefix_hash_values(last_host_node.parent)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                (
                    l3_hash_values,
                    l3_storage_hit_length,
                ) = self.tree_cache.query_storage_hit(
                    last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
                )
                if buffer_only and l3_storage_hit_length > 0:
                    if l3_storage_hit_length >= l2_host_hit_length:
                        l2_fallback_hit_length = l2_host_hit_length
                        l2_host_hit_length = 0
                    else:
                        # Preserve the longer L2 prefix if L3 no longer covers it.
                        l3_storage_hit_length = 0
                        l3_hash_values = []

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2_host_hit_length,
            l3_storage_hit_length=l3_storage_hit_length,
            last_device_node=result.last_device_node,
            last_host_node=last_host_node if l3_storage_hit_length > 0 else None,
            l3_hash_values=(l3_hash_values if l3_storage_hit_length > 0 else None),
            l2_fallback_hit_length=l2_fallback_hit_length,
        )

    @staticmethod
    def _degrade_hicache_prefetch(prefix_match: DecodePrefixMatch) -> None:
        prefix_match.l3_storage_hit_length = 0
        prefix_match.l3_hash_values = None
        prefix_match.l2_host_hit_length = getattr(
            prefix_match,
            "l2_fallback_hit_length",
            prefix_match.l2_host_hit_length,
        )
        prefix_match.prefetch_registered = False

    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        """Issue L3 storage prefetch after admission succeeds.

        On failure, degrades to L2-only restore by clearing l3 fields.
        """
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        try:
            node = prefix_match.last_host_node
            previous_context = self.tree_cache.ongoing_prefetch.get(req.rid)
            matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
            suffix = req.origin_input_ids[
                matched_len : matched_len + prefix_match.l3_storage_hit_length
            ]
            last_hash = node.get_last_hash_value()
            prefix_keys = (
                node.get_prefix_hash_values(node.parent)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            self.tree_cache.prefetch_from_storage(
                req.rid,
                node,
                suffix,
                last_hash,
                prefix_keys,
                known_hash_values=prefix_match.l3_hash_values,
                require_full_completion=True,
            )
            registered_context = self.tree_cache.ongoing_prefetch.get(req.rid)
            prefix_match.prefetch_registered = (
                registered_context is not None
                and registered_context is not previous_context
            )
            if not prefix_match.prefetch_registered:
                if (
                    previous_context is not None
                    and registered_context is previous_context
                ):
                    # Do not leave an older operation for the same rid holding
                    # L1/L2 ownership while this PD request falls back. The
                    # cache defers the actual free to its TP-synchronized reaper.
                    self.tree_cache.release_aborted_request(req.rid)
                self._degrade_hicache_prefetch(prefix_match)
        except Exception as e:
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s: %s; falling back to L2-only LoadingBack",
                req.rid,
                e,
            )
            self._degrade_hicache_prefetch(prefix_match)

    def _hicache_pending_restore_tokens(self) -> int:
        """Total device tokens reserved for pending HiCache L2/L3 load_back."""
        if not self.scheduler.enable_decode_hicache:
            return 0
        return sum(
            dr.prefix_match.restore_token_count
            for dr in self.transfer_queue.queue
            if dr.prefix_match is not None
            and dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
        )


class HiCacheRestoreGatedKVReceiver:
    """Wraps a kv_receiver so KVPoll.Success is gated on HiCache restore READY."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """HiCache hooks for ``DecodeTransferQueue``: drive restore state machine."""

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        if (
            decode_req.prefix_match is not None
            and decode_req.prefix_match.prefetch_registered
        ):
            self.tree_cache.release_aborted_request(decode_req.req.rid)
        if decode_req.hicache_restored_node is not None:
            self.tree_cache.dec_lock_ref(decode_req.hicache_restored_node)
            decode_req.hicache_restored_node = None

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """Queue one L2->L1 load_back op for ``dr``; True iff a DMA was queued.

        On success, ``dr.hicache_restored_node`` and ``hicache_restored_kv_indices``
        are populated, and an inc_lock_ref is held until commit/abort.
        Trivial cases (all-on-device / no needed coverage) auto-flip to READY.
        Failback paths flip to FAILED.
        """
        pm = dr.prefix_match

        # Wait for the L3 restore to drain (skip when there is no L3 hit).
        if pm.l3_storage_hit_length > 0:
            buffer_only = getattr(self.tree_cache, "buffer_only", False)
            if buffer_only:
                active_streams = self.tree_cache.streaming_prefetch_active
                allow_device_allocation = (
                    not active_streams or dr.req.rid in active_streams
                )
                prefetch_done = self.tree_cache.check_prefetch_progress(
                    dr.req.rid,
                    allow_device_allocation=allow_device_allocation,
                    # DecodePreallocQueue already subtracts restore_token_count
                    # from its admission budget, so no second reservation is needed.
                    device_admission_tokens=0,
                )
            else:
                prefetch_done = self.tree_cache.check_prefetch_progress(dr.req.rid)
            if not prefetch_done:
                return False
            self.tree_cache.pop_prefetch_loaded_tokens(dr.req.rid)

        # Re-match: req.last_node / prefix_indices updated to current device state.
        rematch = match_prefix_for_req(
            self.tree_cache,
            dr.req,
            dr.req.origin_input_ids,
            cow_mamba=False,
            include_req=True,
        )
        new_indices, restored_node = self.tree_cache.init_load_back(
            InitLoadBackParams(
                best_match_node=rematch.best_match_node,
                host_hit_length=rematch.host_hit_length,
                req=dr.req,
            )
        )
        # Failback: total coverage < required prefix means device alloc likely failed.
        if len(rematch.device_indices) + len(new_indices) < pm.decode_prefix_len:
            logger.warning(
                "HiCache load_back failed for rid=%s: device_indices=%d, "
                "new_indices=%d, expected decode_prefix_len=%d (l1=%d, l2=%d, l3=%d)",
                dr.req.rid,
                len(rematch.device_indices),
                len(new_indices),
                pm.decode_prefix_len,
                pm.l1_prefix_len,
                pm.l2_host_hit_length,
                pm.l3_storage_hit_length,
            )
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        dr.hicache_restored_kv_indices = torch.cat(
            [rematch.device_indices[pm.l1_prefix_len :], new_indices]
        )
        dr.hicache_restored_node = restored_node
        self.tree_cache.inc_lock_ref(restored_node)

        if len(new_indices) == 0:
            # Whole prefix already on device; no DMA needed.
            dr.hicache_restore_status = HiCacheRestoreResult.READY
            return False
        return True

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        if not hasattr(self.tree_cache, "is_load_back_event_done"):
            return

        # Filter once: keep only PENDING reqs that still need restore work;
        # trivially-done reqs (no prefix_match / nothing to restore) flip to READY.
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if pm is None or not pm.needs_local_restore:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(dr)

        # Phase A: advance in-flight DMAs to READY.
        for dr in active:
            if (
                dr.hicache_restored_node is not None
                and self.tree_cache.is_load_back_event_done(
                    dr.hicache_load_consumer_index
                )
            ):
                dr.hicache_restore_status = HiCacheRestoreResult.READY

        # Phase B: queue new load_back ops if the next slot is free.
        # The (producer_index + 1) check ensures we never overwrite a still-in-flight slot:
        # if a previous req holds that slot and isn't done, its event won't be signaled.
        counter = self.tree_cache.cache_controller.layer_done_counter
        if not self.tree_cache.is_load_back_event_done(
            (counter.producer_index + 1) % counter.num_counters
        ):
            return
        queued = [
            dr
            for dr in active
            if dr.hicache_restored_node is None
            and self._try_hicache_queue_load_back(dr)
        ]
        if not queued:
            return

        # Phase C: kick off merged DMA, bind consumer_index for Phase A polling next tick.
        consumer_index = self.tree_cache.ready_to_load_host_cache()
        if consumer_index < 0:
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
            return
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            return

        self.tree_cache.dec_lock_ref(prefix_match.last_device_node)

        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.req_pool_idx,
                slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len),
            ),
            decode_req.hicache_restored_kv_indices,
        )
        decode_req.req.prefix_indices = torch.cat(
            [prefix_match.prefix_indices, decode_req.hicache_restored_kv_indices]
        )
        decode_req.req.last_node = decode_req.hicache_restored_node
