import unittest
from array import array
from collections import OrderedDict
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
        cache.waiting_l2_prefetch_queue = OrderedDict()
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
        self.assertEqual(list(cache.waiting_l2_prefetch_queue), ["req"])
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.mem_pool_host.alloc.assert_not_called()
        controller.prefetch.assert_not_called()

    def test_waiting_prefetch_retries_in_scheduler_order(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("first", anchor, [1, 2])
        cache.prefetch_from_storage("second", anchor, [3, 4])

        controller.prefetch_rate_limited.return_value = False
        controller.prefetch_tokens_occupied = 0
        self.assertTrue(cache.can_retry_waiting_l2_prefetch("first"))
        self.assertTrue(cache.can_retry_waiting_l2_prefetch("second"))

        controller.mem_pool_host.alloc.return_value = torch.arange(
            2, dtype=torch.int64
        )
        controller.prefetch.return_value = MagicMock()
        cache.prefetch_from_storage("second", anchor, [3, 4])

        self.assertFalse(cache.is_waiting_l2_prefetch("second"))
        self.assertEqual(list(cache.waiting_l2_prefetch_queue), ["first"])
        self.assertTrue(cache.can_retry_waiting_l2_prefetch("first"))
        self.assertEqual(controller.prefetch_tokens_occupied, 2)

    def test_new_prefetch_does_not_bypass_existing_waiters(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("waiting", anchor, [1])

        controller.prefetch_rate_limited.return_value = False
        controller.prefetch_tokens_occupied = 0
        cache.prefetch_from_storage("new", anchor, [2])

        self.assertTrue(cache.is_waiting_l2_prefetch("new"))
        controller.mem_pool_host.alloc.assert_not_called()

    def test_failed_host_allocation_falls_back_and_releases_anchor(self):
        cache, controller, anchor = self._make_cache()
        controller.mem_pool_host.alloc.side_effect = [None, None]
        controller.mem_pool_host.available_size.return_value = 0

        cache.prefetch_from_storage("req", anchor, [1, 2])

        cache.evict_host.assert_called_once_with(2, allow_l1_l2=True)
        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.prefetch.assert_not_called()

    def test_cancel_waiting_prefetch_promotes_next_request(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("first", anchor, [1])
        cache.prefetch_from_storage("second", anchor, [2])

        cache.release_aborted_request("first")
        controller.prefetch_rate_limited.return_value = False
        controller.prefetch_tokens_occupied = 0

        self.assertNotIn("first", cache.ongoing_prefetch)
        self.assertEqual(list(cache.waiting_l2_prefetch_queue), ["second"])
        self.assertTrue(cache.can_retry_waiting_l2_prefetch("second"))

    def test_waiting_prefetch_is_incomplete_and_can_be_terminated(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("req", anchor, [1])

        self.assertFalse(cache.check_prefetch_progress("req"))
        cache.terminate_prefetch("req")

        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(list(cache.waiting_l2_prefetch_queue), [])

    def test_force_cleanup_removes_waiting_prefetch_entries(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1
        cache.prefetch_from_storage("req", anchor, [1])

        cache._force_release_pending_storage_ops()

        self.assertEqual(cache.ongoing_prefetch, {})
        self.assertEqual(list(cache.waiting_l2_prefetch_queue), [])
        self.assertEqual(anchor.host_ref_counter, 0)

    def test_zero_capacity_limit_does_not_deadlock_an_empty_queue(self):
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

    def test_prefetch_submission_failure_rolls_back_host_allocation(self):
        cache, controller, anchor = self._make_cache()
        host_indices = torch.arange(2, dtype=torch.int64)
        controller.mem_pool_host.alloc.return_value = host_indices
        controller.prefetch.side_effect = RuntimeError("submit failed")

        with self.assertRaisesRegex(RuntimeError, "submit failed"):
            cache.prefetch_from_storage("req", anchor, [1, 2])

        controller.mem_pool_host.free.assert_called_once()
        self.assertIs(controller.mem_pool_host.free.call_args.args[0], host_indices)
        self.assertEqual(anchor.host_ref_counter, 0)
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

    def test_decode_cancels_waiting_marker_without_shrinking_boundary(self):
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
