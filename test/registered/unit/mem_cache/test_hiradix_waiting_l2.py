import json
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodePrefixMatch,
)
from sglang.srt.managers.cache_controller import HiCacheController
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
        cache.enable_l2_prefetch_queue = True
        cache.l2_prefetch_wait_timeout = 30.0
        cache.prefetch_threshold = 1
        cache.ongoing_prefetch = {}
        cache.waiting_l2_prefetches = {}
        cache._l2_capacity_generation = 0
        cache.ongoing_backup = {}
        cache.evict_host = MagicMock(return_value=0)
        cache._get_extra_pools = MagicMock(return_value={})
        cache._all_reduce_attn_groups = MagicMock()

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
        controller.mem_pool_host.alloc.return_value = torch.arange(2, dtype=torch.int64)
        controller.prefetch.return_value = MagicMock()
        cache._advance_l2_capacity_generation()

        self.assertTrue(cache.can_retry_waiting_l2_prefetch("req"))
        cache.prefetch_from_storage("req", anchor, [1, 2])

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        self.assertIn("req", cache.ongoing_prefetch)
        self.assertEqual(controller.prefetch_tokens_occupied, 2)

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

    def test_ongoing_prefetch_clears_stale_wait_marker(self):
        cache, controller, anchor = self._make_cache()
        cache._enqueue_waiting_l2_prefetch("req")
        cache.ongoing_prefetch["req"] = MagicMock()

        cache.prefetch_from_storage("req", anchor, [1])

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        controller.mem_pool_host.alloc.assert_not_called()

    def test_peer_admission_limit_makes_all_ranks_wait(self):
        cache, controller, anchor = self._make_cache()

        def report_peer_blocked(tensor, _):
            tensor.fill_(1)

        cache._all_reduce_attn_groups.side_effect = report_peer_blocked

        cache.prefetch_from_storage("req", anchor, [1, 2])

        self.assertTrue(cache.is_waiting_l2_prefetch("req"))
        controller.mem_pool_host.alloc.assert_not_called()
        controller.prefetch.assert_not_called()

    def test_failed_host_allocation_waits_and_releases_anchor(self):
        cache, controller, anchor = self._make_cache()
        controller.mem_pool_host.alloc.side_effect = [None, None]
        controller.mem_pool_host.available_size.return_value = 0

        cache.prefetch_from_storage("req", anchor, [1, 2])

        cache.evict_host.assert_called_once_with(2)
        self.assertTrue(cache.is_waiting_l2_prefetch("req"))
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.prefetch.assert_not_called()

        self.assertFalse(cache.can_retry_waiting_l2_prefetch("req"))
        cache._advance_l2_capacity_generation()
        self.assertTrue(cache.can_retry_waiting_l2_prefetch("req"))

    def test_peer_allocation_failure_rolls_back_local_allocation(self):
        cache, controller, anchor = self._make_cache()
        host_indices = torch.arange(2, dtype=torch.int64)
        controller.mem_pool_host.alloc.return_value = host_indices

        def report_peer_allocation_failure(tensor, op):
            if op == torch.distributed.ReduceOp.MIN:
                tensor.fill_(0)

        cache._all_reduce_attn_groups.side_effect = report_peer_allocation_failure

        cache.prefetch_from_storage("req", anchor, [1, 2])

        controller.mem_pool_host.free.assert_called_once()
        self.assertIs(controller.mem_pool_host.free.call_args.args[0], host_indices)
        self.assertTrue(cache.is_waiting_l2_prefetch("req"))
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.prefetch.assert_not_called()

    def test_prefetch_uses_smallest_allocation_across_ranks(self):
        cache, controller, anchor = self._make_cache()
        controller.mem_pool_host.alloc.return_value = torch.arange(2, dtype=torch.int64)
        controller.prefetch.return_value = MagicMock()

        def report_one_token_peer_allocation(tensor, op):
            if op == torch.distributed.ReduceOp.MIN:
                tensor.fill_(1)

        cache._all_reduce_attn_groups.side_effect = report_one_token_peer_allocation

        cache.prefetch_from_storage("req", anchor, [1, 2])

        submitted_indices = controller.prefetch.call_args.args[1]
        self.assertTrue(
            torch.equal(submitted_indices, torch.tensor([0], dtype=torch.int64))
        )
        released_indices = controller.mem_pool_host.free.call_args.args[0]
        self.assertTrue(
            torch.equal(released_indices, torch.tensor([1], dtype=torch.int64))
        )
        self.assertEqual(controller.prefetch_tokens_occupied, 1)

    def test_failed_partial_host_allocation_also_waits(self):
        cache, controller, anchor = self._make_cache()
        controller.mem_pool_host.alloc.side_effect = [None, None, None]
        controller.mem_pool_host.available_size.return_value = 1

        cache.prefetch_from_storage("req", anchor, [1, 2])

        self.assertEqual(controller.mem_pool_host.alloc.call_count, 3)
        self.assertTrue(cache.is_waiting_l2_prefetch("req"))
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.prefetch.assert_not_called()

    def test_partial_fallback_never_allocates_beyond_requested_tokens(self):
        cache, controller, anchor = self._make_cache()
        controller.mem_pool_host.alloc.side_effect = [
            None,
            None,
            torch.arange(2, dtype=torch.int64),
        ]
        controller.mem_pool_host.available_size.return_value = 10
        controller.prefetch.return_value = MagicMock()

        cache.prefetch_from_storage("req", anchor, [1, 2])

        requested_sizes = [
            call.args[0] for call in controller.mem_pool_host.alloc.call_args_list
        ]
        self.assertEqual(requested_sizes, [2, 2, 2])
        self.assertEqual(controller.prefetch_tokens_occupied, 2)

    def test_allocation_failure_only_blocks_the_failed_request_per_generation(self):
        cache, controller, _ = self._make_cache()
        cache._enqueue_waiting_l2_prefetch("first")
        cache._enqueue_waiting_l2_prefetch("second")
        cache._advance_l2_capacity_generation()

        cache._defer_l2_allocation_retry("first")

        self.assertFalse(cache.can_retry_waiting_l2_prefetch("first"))
        self.assertTrue(cache.can_retry_waiting_l2_prefetch("second"))
        controller.mem_pool_host.alloc.assert_not_called()

    def test_hicache_event_check_does_not_retry_without_capacity_change(self):
        cache, _, _ = self._make_cache()
        cache._enqueue_waiting_l2_prefetch("req")
        cache._drain_async_work = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.enable_storage = False
        cache.enable_storage_metrics = False

        cache.check_hicache_events()

        self.assertEqual(cache._l2_capacity_generation, 0)
        self.assertFalse(cache.can_retry_waiting_l2_prefetch("req"))

    def test_wait_timeout_expires_same_oldest_request_on_all_ranks(self):
        cache, controller, _ = self._make_cache()
        cache._enqueue_waiting_l2_prefetch("first")
        cache._enqueue_waiting_l2_prefetch("second")
        for queue_name in (
            "prefetch_revoke_queue",
            "ack_backup_queue",
            "host_mem_release_queue",
        ):
            getattr(controller, queue_name).qsize.return_value = 0
        cache._drain_storage_control_queues_impl = MagicMock()

        def report_peer_timeout(tensor, op):
            if op == torch.distributed.ReduceOp.MIN:
                tensor[3] = 0

        cache._all_reduce_attn_groups.side_effect = report_peer_timeout

        cache.drain_storage_control_queues()

        self.assertFalse(cache.is_waiting_l2_prefetch("first"))
        self.assertTrue(cache.can_retry_waiting_l2_prefetch("second"))
        self.assertEqual(cache._l2_capacity_generation, 1)

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

        self.assertEqual(cache.waiting_l2_prefetches, {})
        self.assertEqual(cache.ongoing_prefetch, {})
        self.assertEqual(cache._l2_capacity_generation, 0)
        self.assertEqual(anchor.host_ref_counter, 0)

    def test_zero_capacity_limit_does_not_deadlock_idle_pool(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_capacity_limit = 0
        controller.prefetch_tokens_occupied = 0
        controller.mem_pool_host.alloc.return_value = torch.arange(1, dtype=torch.int64)
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

    def test_disabled_queue_drops_rate_limited_prefetch(self):
        cache, controller, anchor = self._make_cache()
        cache.enable_l2_prefetch_queue = False
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1

        cache.prefetch_from_storage("req", anchor, [1, 2])

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.mem_pool_host.alloc.assert_not_called()
        controller.prefetch.assert_not_called()

    def test_disabled_queue_drops_failed_allocation(self):
        cache, controller, anchor = self._make_cache()
        cache.enable_l2_prefetch_queue = False
        controller.mem_pool_host.alloc.side_effect = [None, None]
        controller.mem_pool_host.available_size.return_value = 0

        cache.prefetch_from_storage("req", anchor, [1, 2])

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.prefetch.assert_not_called()

    def test_allow_wait_false_drops_rate_limited_prefetch(self):
        cache, controller, anchor = self._make_cache()
        controller.prefetch_rate_limited.return_value = True
        controller.prefetch_tokens_occupied = 1

        cache.prefetch_from_storage("req", anchor, [1, 2], allow_wait=False)

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        controller.mem_pool_host.alloc.assert_not_called()
        controller.prefetch.assert_not_called()

    def test_allow_wait_false_drops_failed_allocation(self):
        cache, controller, anchor = self._make_cache()
        controller.mem_pool_host.alloc.side_effect = [None, None]
        controller.mem_pool_host.available_size.return_value = 0

        cache.prefetch_from_storage("req", anchor, [1, 2], allow_wait=False)

        self.assertFalse(cache.is_waiting_l2_prefetch("req"))
        self.assertEqual(anchor.host_ref_counter, 0)
        controller.prefetch.assert_not_called()

    def test_parse_prefetch_capacity_and_queue_config(self):
        cache, _, _ = self._make_cache()
        (
            backend_config,
            _,
            _,
            _,
            prefetch_capacity_ratio,
            enable_l2_prefetch_queue,
            l2_prefetch_wait_timeout,
        ) = cache._parse_storage_backend_extra_config(
            json.dumps(
                {
                    "prefetch_capacity_ratio": 0.75,
                    "enable_l2_prefetch_queue": False,
                    "l2_prefetch_wait_timeout": 45,
                    "backend_option": "preserved",
                }
            )
        )

        self.assertEqual(prefetch_capacity_ratio, 0.75)
        self.assertFalse(enable_l2_prefetch_queue)
        self.assertEqual(l2_prefetch_wait_timeout, 45.0)
        self.assertEqual(backend_config, {"backend_option": "preserved"})

    def test_parse_prefetch_capacity_defaults(self):
        cache, _, _ = self._make_cache()

        parsed = cache._parse_storage_backend_extra_config(None)

        self.assertEqual(parsed[4], 0.5)
        self.assertTrue(parsed[5])
        self.assertEqual(parsed[6], 30.0)

    def test_reject_invalid_prefetch_capacity_ratio(self):
        cache, _, _ = self._make_cache()
        invalid_values = [0, -0.1, 1.1, True, "0.5", float("inf"), float("nan")]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "prefetch_capacity_ratio"):
                    cache._parse_storage_backend_extra_config(
                        json.dumps({"prefetch_capacity_ratio": value})
                    )

    def test_reject_non_boolean_queue_config(self):
        cache, _, _ = self._make_cache()

        with self.assertRaisesRegex(ValueError, "enable_l2_prefetch_queue"):
            cache._parse_storage_backend_extra_config(
                json.dumps({"enable_l2_prefetch_queue": 1})
            )

    def test_reject_invalid_l2_wait_timeout(self):
        cache, _, _ = self._make_cache()
        invalid_values = [0, -1, True, "30", float("inf"), float("nan")]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "l2_prefetch_wait_timeout"):
                    cache._parse_storage_backend_extra_config(
                        json.dumps({"l2_prefetch_wait_timeout": value})
                    )

    def test_runtime_config_applies_ratio_and_disables_queue(self):
        cache, controller, _ = self._make_cache()
        cache._enqueue_waiting_l2_prefetch("req")
        controller.prefetch_capacity_limit = 75
        timeout_config = cache._parse_storage_backend_extra_config(None)[2]

        cache._apply_storage_runtime_config(
            storage_backend="test",
            prefetch_threshold=256,
            prefetch_timeout_config=timeout_config,
            hicache_storage_pass_prefix_keys=False,
            prefetch_capacity_ratio=0.75,
            enable_l2_prefetch_queue=False,
            l2_prefetch_wait_timeout=45.0,
            enable_storage=False,
            enable_storage_metrics=False,
            extra_metric_labels=None,
        )

        controller.set_prefetch_capacity_ratio.assert_called_once_with(0.75)
        self.assertFalse(cache.enable_l2_prefetch_queue)
        self.assertEqual(cache.waiting_l2_prefetches, {})
        self.assertEqual(cache.l2_prefetch_wait_timeout, 45.0)


class TestHiCachePrefetchCapacity(unittest.TestCase):
    def test_capacity_ratio_updates_token_limit(self):
        controller = object.__new__(HiCacheController)
        controller.mem_pool_host = SimpleNamespace(size=101)

        controller.set_prefetch_capacity_ratio(0.75)

        self.assertEqual(controller.prefetch_capacity_ratio, 0.75)
        self.assertEqual(controller.prefetch_capacity_limit, 75)


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

    def test_decode_disables_waiting_without_shrinking_boundary(self):
        mixin, tree_cache, req, prefix_match = self._make_case()

        mixin._start_hicache_prefetch(req, prefix_match)

        self.assertFalse(
            tree_cache.prefetch_from_storage.call_args.kwargs["allow_wait"]
        )
        tree_cache.is_waiting_l2_prefetch.assert_not_called()
        tree_cache.cancel_waiting_l2_prefetch.assert_not_called()
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
