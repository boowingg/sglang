import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodePrefixMatch,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiRadixWaitingL2(unittest.TestCase):
    def _make_cache(self):
        cache = object.__new__(HiRadixCache)
        cache.page_size = 1
        cache.is_eagle = False
        cache.enable_storage = True
        cache.prefetch_threshold = 1
        cache.ongoing_prefetch = {}
        cache.waiting_l2_prefetches = set()
        cache._l2_retry_epoch = 0
        cache._l2_allocation_retry_after_epoch = 0
        cache._l2_allocation_retry_delay_rounds = 0
        cache.ongoing_backup = {}
        cache.evict_host = MagicMock(return_value=0)
        cache._get_extra_pools = MagicMock(return_value={})

        controller = MagicMock()
        controller.prefetch_rate_limited.return_value = False
        controller.prefetch_capacity_limit = 1
        controller.prefetch_tokens_occupied = 0
        cache.cache_controller = controller

        anchor = TreeNode()
        anchor.key = RadixKey(array("q"))
        anchor.host_value = []
        return cache, controller, anchor

    def test_rate_limited_prefetch_waits_without_pinning_anchor(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1

        cache.prefetch_from_storage("req", anchor, [1, 2, 3, 4])

        self.assertTrue(cache.is_waiting_l2_prefetch("req"))
        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.mem_pool_host.alloc.assert_not_called()
        controller.prefetch.assert_not_called()

    def test_waiting_prefetch_can_retry_when_budget_is_available(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("req", anchor, [1, 2])

        controller.prefetch_rate_limited.return_value = False
        controller.prefetch_tokens_occupied = 0
        controller.mem_pool_host.alloc.return_value = torch.arange(
            2, dtype=torch.int64
        )
        controller.prefetch.return_value = MagicMock()
        cache._l2_allocation_retry_delay_rounds = 8
        cache._l2_allocation_retry_after_epoch = 0

        self.assertTrue(cache.can_retry_waiting_l2_prefetch("req"))
        cache.prefetch_from_storage("req", anchor, [1, 2])

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        self.assertIn("req", cache.ongoing_prefetch)
        self.assertEqual(controller.prefetch_tokens_occupied, 2)
        self.assertEqual(cache._l2_allocation_retry_after_epoch, 0)
        self.assertEqual(cache._l2_allocation_retry_delay_rounds, 0)

    def test_new_prefetch_does_not_bypass_existing_waiter(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("first", anchor, [1])

        controller.prefetch_rate_limited.return_value = False
        controller.prefetch_tokens_occupied = 0
        cache.prefetch_from_storage("second", anchor, [2])

        self.assertTrue(cache.is_waiting_l2_prefetch("first"))
        self.assertTrue(cache.is_waiting_l2_prefetch("second"))
        controller.mem_pool_host.alloc.assert_not_called()

    def test_failed_host_allocation_waits_and_releases_anchor(self):
        cache, controller, anchor = self._make_cache()
        controller.mem_pool_host.alloc.side_effect = [None, None]
        controller.mem_pool_host.available_size.return_value = 0

        cache.prefetch_from_storage("req", anchor, [1, 2])

        cache.evict_host.assert_called_once_with(2)
        self.assertTrue(cache.is_waiting_l2_prefetch("req"))
        self.assertGreater(cache._l2_allocation_retry_delay_rounds, 0)
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.prefetch.assert_not_called()

        cache._l2_retry_epoch = cache._l2_allocation_retry_after_epoch
        self.assertTrue(cache.can_retry_waiting_l2_prefetch("req"))

    def test_failed_partial_host_allocation_also_waits(self):
        cache, controller, anchor = self._make_cache()
        controller.mem_pool_host.alloc.side_effect = [None, None, None]
        controller.mem_pool_host.available_size.return_value = 1

        cache.prefetch_from_storage("req", anchor, [1, 2])

        self.assertEqual(controller.mem_pool_host.alloc.call_count, 3)
        self.assertTrue(cache.is_waiting_l2_prefetch("req"))
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.prefetch.assert_not_called()

    def test_allocation_retry_uses_shared_exponential_backoff(self):
        cache, controller, _ = self._make_cache()
        cache._enqueue_waiting_l2_prefetch("first")
        cache._enqueue_waiting_l2_prefetch("second")

        cache._defer_l2_allocation_retry("first")
        first_delay = cache._l2_allocation_retry_delay_rounds
        first_retry_epoch = cache._l2_allocation_retry_after_epoch
        cache._defer_l2_allocation_retry("first")

        self.assertGreater(first_delay, 0)
        self.assertGreater(cache._l2_allocation_retry_delay_rounds, first_delay)
        self.assertGreater(
            cache._l2_allocation_retry_after_epoch, first_retry_epoch
        )
        self.assertFalse(cache.can_retry_waiting_l2_prefetch("second"))
        controller.mem_pool_host.alloc.assert_not_called()

    def test_hicache_event_check_advances_retry_epoch(self):
        cache, _, _ = self._make_cache()
        cache._drain_async_work = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.enable_storage = False
        cache.enable_storage_metrics = False

        cache.check_hicache_events()

        self.assertEqual(cache._l2_retry_epoch, 1)

    def test_abort_removes_waiting_prefetch(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("req", anchor, [1])

        cache.release_aborted_request("req")

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        self.assertNotIn("req", cache.ongoing_prefetch)

    def test_waiting_prefetch_is_incomplete_and_can_be_terminated(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("req", anchor, [1])

        self.assertFalse(cache.check_prefetch_progress("req"))
        cache.terminate_prefetch("req")

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))

    def test_force_cleanup_removes_waiting_prefetches(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("req", anchor, [1])

        cache._force_release_pending_storage_ops()

        self.assertEqual(cache.waiting_l2_prefetches, set())
        self.assertEqual(cache.ongoing_prefetch, {})
        self.assertEqual(cache._l2_allocation_retry_after_epoch, 0)
        self.assertEqual(cache._l2_allocation_retry_delay_rounds, 0)
        self.assertEqual(anchor.host_ref_counter, 0)

    def test_zero_capacity_limit_does_not_deadlock_idle_pool(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_capacity_limit = 0
        controller.prefetch_tokens_occupied = 0
        controller.mem_pool_host.alloc.return_value = torch.arange(
            1, dtype=torch.int64
        )
        controller.prefetch.return_value = MagicMock()

        cache.prefetch_from_storage("req", anchor, [1])

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        controller.prefetch.assert_called_once()

    def test_prefetch_submission_failure_rolls_back_allocation(self):
        cache, controller, anchor = self._make_cache()
        host_indices = torch.arange(2, dtype=torch.int64)
        controller.mem_pool_host.alloc.return_value = host_indices
        controller.prefetch.side_effect = RuntimeError("submit failed")

        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            cache.prefetch_from_storage("req", anchor, [1, 2])

        controller.mem_pool_host.free.assert_called_once()
        self.assertIs(controller.mem_pool_host.free.call_args.args[0], host_indices)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        self.assertNotIn("req", cache.ongoing_prefetch)


class TestDecodeHiCacheWaitingL2Safety(unittest.TestCase):
    def _make_case(self):
        mixin = object.__new__(DecodeHiCachePreallocMixin)
        tree_cache = MagicMock()
        tree_cache.hicache_storage_pass_prefix_keys = False
        tree_cache.ongoing_prefetch = {}
        mixin.tree_cache = tree_cache

        node = MagicMock()
        req = SimpleNamespace(rid="req", origin_input_ids=[1, 2])
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.empty(0, dtype=torch.int64),
            l2_host_hit_length=0,
            l3_storage_hit_length=2,
            last_device_node=MagicMock(),
            last_host_node=node,
        )
        return mixin, tree_cache, req, prefix_match

    def test_decode_cancels_waiting_state_without_shrinking_boundary(self):
        mixin, tree_cache, req, prefix_match = self._make_case()
        tree_cache.is_waiting_l2_prefetch.return_value = True

        mixin._start_hicache_prefetch(req, prefix_match)

        tree_cache.cancel_waiting_l2_prefetch.assert_called_once_with("req")
        self.assertEqual(prefix_match.decode_prefix_len, 2)
        self.assertFalse(prefix_match.prefetch_registered)

    def test_decode_submission_error_keeps_fail_closed_boundary(self):
        mixin, tree_cache, req, prefix_match = self._make_case()
        tree_cache.prefetch_from_storage.side_effect = RuntimeError("submit failed")

        mixin._start_hicache_prefetch(req, prefix_match)

        self.assertEqual(prefix_match.decode_prefix_len, 2)
        self.assertFalse(prefix_match.prefetch_registered)


if __name__ == "__main__":
    unittest.main()
