from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class L2PrefetchAdmissionConfig:
    enabled: bool = False
    capacity_ratio: float = 0.5
    wait_timeout: float = 30.0

    @classmethod
    def pop_from(
        cls, extra_config: MutableMapping[str, Any]
    ) -> L2PrefetchAdmissionConfig:
        defaults = cls()
        enabled = extra_config.pop("enable_l2_prefetch_queue", defaults.enabled)
        capacity_ratio = extra_config.pop(
            "prefetch_capacity_ratio", defaults.capacity_ratio
        )
        wait_timeout = extra_config.pop(
            "l2_prefetch_wait_timeout", defaults.wait_timeout
        )

        if not isinstance(enabled, bool):
            raise ValueError(
                "enable_l2_prefetch_queue must be bool, got "
                f"{type(enabled).__name__}"
            )
        if (
            isinstance(capacity_ratio, bool)
            or not isinstance(capacity_ratio, (int, float))
            or not math.isfinite(capacity_ratio)
            or not 0 < capacity_ratio <= 1
        ):
            raise ValueError(
                "prefetch_capacity_ratio must be a finite number in (0, 1], "
                f"got {capacity_ratio!r}"
            )
        if (
            isinstance(wait_timeout, bool)
            or not isinstance(wait_timeout, (int, float))
            or not math.isfinite(wait_timeout)
            or wait_timeout <= 0
        ):
            raise ValueError(
                "l2_prefetch_wait_timeout must be a finite number greater than 0, "
                f"got {wait_timeout!r}"
            )

        return cls(
            enabled=enabled,
            capacity_ratio=float(capacity_ratio),
            wait_timeout=float(wait_timeout),
        )


@dataclass
class _WaitingPrefetch:
    # Set only while this request is the FIFO head; followers do not age out.
    head_since: Optional[float]
    # Capacity generation observed by this request's most recent failed attempt.
    failed_capacity_generation: int


class L2PrefetchAdmission:
    """Optional ordered admission wait set for L3 prefetches needing L2 capacity.

    Existing waiters retry in scheduler order after a capacity change, while new
    arrivals cannot skip that waiting cohort. Followers do not age out behind an
    older request: timeout starts only when a request becomes the queue head, so
    one expired head cannot trigger a cascade of immediate fallbacks.
    """

    def __init__(
        self,
        config: L2PrefetchAdmissionConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._waiters: OrderedDict[str, _WaitingPrefetch] = OrderedDict()
        self._capacity_generation = 0
        self._capacity_change_pending = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def waiting_count(self) -> int:
        return len(self._waiters)

    @property
    def oldest_req_id(self) -> Optional[str]:
        return next(iter(self._waiters), None)

    def configure(
        self, config: L2PrefetchAdmissionConfig, cache_controller: Any
    ) -> None:
        self.clear()
        self.config = config
        self.apply_capacity_limit(cache_controller)

    def apply_capacity_limit(self, cache_controller: Any) -> None:
        if not getattr(cache_controller, "enable_storage", False):
            return
        cache_controller.prefetch_capacity_limit = int(
            self.config.capacity_ratio * cache_controller.mem_pool_host.size
        )

    def clear(self) -> None:
        self._waiters.clear()
        self._capacity_generation = 0
        self._capacity_change_pending = False

    def is_waiting(self, req_id: str) -> bool:
        return self.enabled and req_id in self._waiters

    def should_wait_behind_existing(self, req_id: str) -> bool:
        # Existing waiters retry as one scheduler-ordered cohort. Only a new
        # arrival must wait behind that cohort.
        return bool(self._waiters) and req_id not in self._waiters

    def enqueue(self, req_id: str) -> bool:
        if not self.enabled:
            return False

        state = self._waiters.get(req_id)
        if state is None:
            is_first = not self._waiters
            # Starting every follower's timer here would make a long queue expire
            # as a batch before each request gets a full turn as queue head.
            self._waiters[req_id] = _WaitingPrefetch(
                head_since=self._clock() if is_first else None,
                failed_capacity_generation=self._capacity_generation,
            )
        else:
            # Requeue updates retry state only; a head keeps its original timer.
            state.failed_capacity_generation = self._capacity_generation
        return True

    def can_retry(self, req_id: str) -> bool:
        state = self._waiters.get(req_id)
        return (
            self.enabled
            and state is not None
            and state.failed_capacity_generation != self._capacity_generation
        )

    def cancel(self, req_id: str) -> bool:
        was_oldest = self.oldest_req_id == req_id
        removed = self._waiters.pop(req_id, None) is not None
        if removed and self._waiters:
            if was_oldest:
                # Hand a fresh timeout budget to the newly promoted head.
                self._activate_oldest()
            self._advance_capacity_generation()
        return removed

    def capacity_changed(self, *, synchronized: bool = False) -> None:
        if not self._waiters:
            return
        if synchronized:
            self._capacity_change_pending = False
            self._advance_capacity_generation()
        else:
            self._capacity_change_pending = True

    def oldest_expired(self) -> bool:
        oldest_req_id = self.oldest_req_id
        if oldest_req_id is None:
            return False
        state = self._waiters[oldest_req_id]
        head_since = state.head_since
        if head_since is None:
            # Repair the queue invariant defensively instead of expiring a head
            # that has never owned a timeout budget.
            state.head_since = self._clock()
            return False
        return self._clock() - head_since >= self.config.wait_timeout

    def expire_oldest(self) -> Optional[tuple[str, float]]:
        oldest_req_id = self.oldest_req_id
        if oldest_req_id is None:
            return None
        state = self._waiters.pop(oldest_req_id)
        now = self._clock()
        waited = now - (state.head_since if state.head_since is not None else now)
        if self._waiters:
            # Reuse one timestamp for expiry and promotion so no timeout budget
            # is lost between the two state transitions.
            self._activate_oldest(now)
            self._advance_capacity_generation()
        return oldest_req_id, waited

    def drain_storage_control_queues(self, cache: Any) -> Optional[tuple[str, float]]:
        import torch

        cc = cache.cache_controller
        queue_states = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
                int(not self.oldest_expired()),
                int(not self._capacity_change_pending),
            ],
            dtype=torch.int,
        )
        cache._all_reduce_attn_groups(queue_states, torch.distributed.ReduceOp.MIN)
        (
            n_revoke,
            n_backup,
            n_release,
            oldest_wait_not_expired,
            capacity_unchanged,
        ) = map(int, queue_states.tolist())
        self._capacity_change_pending = False
        if not capacity_unchanged:
            self._advance_capacity_generation()
        expired = None if oldest_wait_not_expired else self.expire_oldest()
        cache._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )
        if n_revoke > 0 or n_release > 0:
            self.capacity_changed(synchronized=True)
        return expired

    def _advance_capacity_generation(self) -> None:
        if self._waiters:
            self._capacity_generation += 1

    def _activate_oldest(self, now: Optional[float] = None) -> None:
        """Start the current head's timeout exactly once."""
        oldest_req_id = self.oldest_req_id
        if oldest_req_id is None:
            return
        state = self._waiters[oldest_req_id]
        if state.head_since is None:
            state.head_since = self._clock() if now is None else now

    def prefetch(
        self,
        cache: Any,
        req_id: str,
        last_host_node: Any,
        new_input_tokens: Sequence[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
    ) -> None:
        from sglang.srt.mem_cache.radix_cache import RadixKey

        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=cache.is_eagle,
        ).page_aligned(cache.page_size)
        prefetch_length = len(prefetch_key)
        if not cache.enable_storage or prefetch_length < cache.prefetch_threshold:
            self.cancel(req_id)
            return

        if req_id in cache.ongoing_prefetch:
            self.cancel(req_id)
            return

        # New arrivals cannot bypass the current waiting cohort. A block on any
        # TP rank makes every rank take the same queue path.
        local_blocked = self.should_wait_behind_existing(
            req_id
        ) or self._prefetch_rate_limited(cache.cache_controller)
        if self._blocked_on_any_rank(cache, local_blocked):
            self.enqueue(req_id)
            return

        last_host_node.protect_host()
        try:
            host_indices = self._allocate(cache, prefetch_length)
        except Exception:
            try:
                last_host_node.release_host()
            except Exception:
                logger.exception(
                    "Failed to release host protection for prefetch %s", req_id
                )
            self.cancel(req_id)
            raise

        if host_indices is None:
            last_host_node.release_host()
            # Keep the request schedulable for a later capacity generation
            # instead of silently falling back after one failed L2 allocation.
            self.enqueue(req_id)
            return

        prefetch_key = prefetch_key[: len(host_indices)]
        try:
            operation = cache.cache_controller.prefetch(
                req_id,
                host_indices,
                prefetch_key,
                last_hash,
                prefix_keys,
                **cache._get_extra_pools(),
            )
        except Exception:
            try:
                cache.cache_controller.mem_pool_host.free(host_indices)
            except Exception:
                logger.exception(
                    "Failed to roll back L2 allocation for prefetch %s", req_id
                )
            try:
                last_host_node.release_host()
            except Exception:
                logger.exception(
                    "Failed to release host protection for prefetch %s", req_id
                )
            self.cancel(req_id)
            raise

        cache.ongoing_prefetch[req_id] = (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
        )
        cache.cache_controller.prefetch_tokens_occupied += len(prefetch_key)
        self.cancel(req_id)

    @staticmethod
    def _prefetch_rate_limited(cache_controller: Any) -> bool:
        if not cache_controller.prefetch_rate_limited():
            return False
        return not (
            cache_controller.prefetch_tokens_occupied == 0
            and cache_controller.prefetch_capacity_limit <= 0
        )

    @staticmethod
    def _blocked_on_any_rank(cache: Any, local_blocked: bool) -> bool:
        import torch

        blocked = torch.tensor(int(local_blocked), dtype=torch.int)
        cache._all_reduce_attn_groups(blocked, torch.distributed.ReduceOp.MAX)
        return bool(blocked.item())

    @staticmethod
    def _allocate(cache: Any, requested_tokens: int) -> Optional[Any]:
        import torch

        host_pool = cache.cache_controller.mem_pool_host
        # Preserve the original HiCache policy: full allocation first, eviction
        # second, and a page-aligned partial allocation as the final fallback.
        host_indices = host_pool.alloc(requested_tokens)
        if host_indices is None:
            cache.evict_host(requested_tokens)
            host_indices = host_pool.alloc(requested_tokens)

        if host_indices is None:
            available_tokens = min(requested_tokens, host_pool.available_size())
            available_tokens -= available_tokens % cache.page_size
            if available_tokens >= cache.prefetch_threshold:
                host_indices = host_pool.alloc(available_tokens)

        local_tokens = len(host_indices) if host_indices is not None else 0
        common_tokens = torch.tensor(local_tokens, dtype=torch.int)
        try:
            cache._all_reduce_attn_groups(common_tokens, torch.distributed.ReduceOp.MIN)
        except Exception:
            if host_indices is not None:
                host_pool.free(host_indices)
            raise

        allocated_tokens = int(common_tokens.item())
        allocated_tokens -= allocated_tokens % cache.page_size
        if allocated_tokens < cache.prefetch_threshold:
            if host_indices is not None:
                host_pool.free(host_indices)
            return None

        if allocated_tokens < local_tokens:
            host_pool.free(host_indices[allocated_tokens:])
            host_indices = host_indices[:allocated_tokens]
        return host_indices


def get_l2_prefetch_admission(cache: Any) -> Optional[L2PrefetchAdmission]:
    admission = getattr(cache, "_l2_prefetch_admission", None)
    if isinstance(admission, L2PrefetchAdmission) and admission.enabled:
        return admission
    return None


def log_l2_prefetch_admission_status(cache: Any) -> None:
    cache_dict = getattr(cache, "__dict__", {})
    inner_cache = cache_dict.get("inner", cache)
    cache_impl = type(inner_cache).__name__
    admission = getattr(cache, "_l2_prefetch_admission", None)
    if not isinstance(admission, L2PrefetchAdmission):
        logger.info(
            "HiCache L2 prefetch queue: cache_impl=%s supported=false "
            "enabled=false active=false reason=unsupported_cache_impl",
            cache_impl,
        )
        return

    storage_enabled = bool(getattr(cache, "enable_storage", False))
    controller = getattr(cache, "cache_controller", None)
    capacity_limit = getattr(controller, "prefetch_capacity_limit", 0)
    logger.info(
        "HiCache L2 prefetch queue: cache_impl=%s supported=true enabled=%s "
        "active=%s capacity_ratio=%.3f wait_timeout_s=%.2f "
        "capacity_limit_tokens=%s",
        cache_impl,
        str(admission.enabled).lower(),
        str(admission.enabled and storage_enabled).lower(),
        admission.config.capacity_ratio,
        admission.config.wait_timeout,
        capacity_limit,
    )


def prefetch_from_storage_with_admission(
    cache: Any,
    req_id: str,
    last_host_node: Any,
    new_input_tokens: Sequence[int],
    last_hash: Optional[str] = None,
    prefix_keys: Optional[list[str]] = None,
) -> None:
    admission = get_l2_prefetch_admission(cache)
    if admission is None:
        cache.prefetch_from_storage(
            req_id,
            last_host_node,
            new_input_tokens,
            last_hash,
            prefix_keys,
        )
        return
    admission.prefetch(
        cache,
        req_id,
        last_host_node,
        new_input_tokens,
        last_hash,
        prefix_keys,
    )
