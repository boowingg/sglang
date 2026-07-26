import threading
import unittest
from array import array
from queue import Queue
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefetchOperationQueryFirst(unittest.TestCase):
    def test_query_and_read_state_transitions_are_single_use(self):
        with patch(
            "sglang.srt.managers.cache_controller.time.monotonic",
            side_effect=[10.0, 20.0],
        ):
            operation = PrefetchOperation(
                "req",
                None,
                [1, 2, 3, 4],
                query_first=True,
            )
            self.assertIsNone(operation.get_query_result())
            self.assertFalse(operation.is_read_started())

            self.assertTrue(operation.finish_query(["h1", "h2"], 4))
            self.assertEqual(operation.get_query_result(), 4)
            self.assertTrue(
                operation.start_read(torch.arange(2), target_tokens=2, page_size=2)
            )

        self.assertTrue(operation.is_read_started())
        self.assertEqual(operation.hash_value, ["h1"])
        self.assertEqual(operation.start_time, 20.0)
        self.assertFalse(
            operation.start_read(torch.arange(2), target_tokens=2, page_size=2)
        )
        self.assertFalse(operation.finish_query([], 0))

    def test_cancelled_query_cannot_publish_or_start_read(self):
        operation = PrefetchOperation(
            "req",
            None,
            [1, 2],
            query_first=True,
        )
        operation.mark_terminate()

        self.assertFalse(operation.finish_query(["h1"], 2))
        self.assertFalse(
            operation.start_read(torch.arange(2), target_tokens=2, page_size=2)
        )
        self.assertIsNone(operation.host_indices)


class TestHiCacheControllerQueryFirst(unittest.TestCase):
    def _make_controller(self):
        controller = object.__new__(HiCacheController)
        controller.page_size = 2
        controller.prefetch_threshold = 2
        controller.prefetch_queue = Queue()
        controller.prefetch_query_result_queue = Queue()
        controller.prefetch_revoke_queue = Queue()
        controller.host_mem_release_queue = Queue()
        controller.storage_stop_event = threading.Event()
        controller._all_reduce_prefetch_groups = MagicMock()
        controller.append_host_mem_release = MagicMock()
        return controller

    def test_query_thread_publishes_result_without_host_allocation(self):
        controller = self._make_controller()
        operation = PrefetchOperation(
            "req",
            None,
            [1, 2, 3, 4],
            query_first=True,
        )
        controller._storage_hit_query = MagicMock(return_value=(["h1", "h2"], 4))
        controller.prefetch_queue.put(operation)
        controller.storage_stop_event.set()

        controller.prefetch_thread_func()

        self.assertIs(controller.prefetch_query_result_queue.get_nowait(), operation)
        self.assertEqual(operation.get_query_result(), 4)
        self.assertIsNone(operation.host_indices)
        controller.append_host_mem_release.assert_not_called()

    def test_query_failure_is_published_as_zero_hit(self):
        controller = self._make_controller()
        operation = PrefetchOperation(
            "req",
            None,
            [1, 2],
            query_first=True,
        )
        controller._storage_hit_query = MagicMock(side_effect=RuntimeError("failed"))
        controller.prefetch_queue.put(operation)
        controller.storage_stop_event.set()

        with self.assertLogs("sglang.srt.managers.cache_controller", level="ERROR"):
            controller.prefetch_thread_func()

        self.assertIs(controller.prefetch_query_result_queue.get_nowait(), operation)
        self.assertEqual(operation.get_query_result(), 0)


class TestHiRadixQueryFirst(unittest.TestCase):
    def _make_cache(self, *, page_size=2, prefetch_threshold=2):
        cache = object.__new__(HiRadixCache)
        cache.page_size = page_size
        cache.is_eagle = False
        cache.enable_storage = True
        cache.enable_storage_metrics = False
        cache.prefetch_threshold = prefetch_threshold
        cache.prefetch_stop_policy = "timeout"
        cache.ongoing_prefetch = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache._supports_query_first_prefetch = MagicMock(return_value=True)
        cache.evict_host = MagicMock(return_value=0)

        controller = MagicMock()
        controller.prefetch_rate_limited.return_value = False
        controller.prefetch_tokens_occupied = 0
        controller.prefetch_query_result_queue = Queue()
        controller.mem_pool_host.available_size.return_value = 0

        def query_prefetch(
            request_id, new_input_tokens, last_hash=None, prefix_keys=None
        ):
            return PrefetchOperation(
                request_id,
                None,
                new_input_tokens,
                last_hash,
                prefix_keys,
                query_first=True,
            )

        def resume_prefetch(operation, host_indices, target_tokens):
            return operation.start_read(host_indices, target_tokens, page_size)

        controller.query_prefetch.side_effect = query_prefetch
        controller.resume_prefetch.side_effect = resume_prefetch
        cache.cache_controller = controller
        return cache

    def _make_anchor(self):
        anchor = TreeNode()
        anchor.key = RadixKey(array("q"))
        anchor.host_value = []
        return anchor

    def _submit(self, cache, tokens=None):
        tokens = tokens or list(range(1, 9))
        anchor = self._make_anchor()
        cache.prefetch_from_storage("req", anchor, tokens)
        operation = cache.ongoing_prefetch["req"][3]
        return anchor, operation

    def test_submission_queries_without_allocating_l2(self):
        cache = self._make_cache()
        anchor, operation = self._submit(cache, tokens=[1, 2, 3, 4, 5])

        cache.cache_controller.mem_pool_host.alloc.assert_not_called()
        cache.evict_host.assert_not_called()
        self.assertTrue(operation.query_first)
        self.assertEqual(len(operation.token_ids), 4)
        self.assertIsNone(cache.ongoing_prefetch["req"][2])
        self.assertEqual(anchor.host_ref_counter, 1)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 4)

    def test_query_result_allocates_only_actual_hit(self):
        cache = self._make_cache()
        anchor, operation = self._submit(cache)
        host_indices = torch.arange(4)
        cache.cache_controller.mem_pool_host.alloc.return_value = host_indices
        self.assertTrue(operation.finish_query(["h1", "h2"], 4))
        cache.cache_controller.prefetch_query_result_queue.put(operation)

        cache._drain_query_first_results()

        cache.cache_controller.mem_pool_host.alloc.assert_called_once_with(4)
        cache.evict_host.assert_not_called()
        cache.cache_controller.resume_prefetch.assert_called_once_with(
            operation, host_indices, 4
        )
        self.assertTrue(operation.is_read_started())
        self.assertIs(cache.ongoing_prefetch["req"][2], host_indices)
        self.assertEqual(anchor.host_ref_counter, 1)

    def test_low_hit_releases_tracking_without_allocating(self):
        cache = self._make_cache(prefetch_threshold=4)
        anchor, operation = self._submit(cache)
        self.assertTrue(operation.finish_query(["h1"], 2))
        cache.cache_controller.prefetch_query_result_queue.put(operation)

        cache._drain_query_first_results()

        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        self.assertTrue(operation.is_terminated())
        cache.cache_controller.mem_pool_host.alloc.assert_not_called()

    def test_pressure_reclaims_actual_hit_not_candidate_length(self):
        cache = self._make_cache()
        _, operation = self._submit(cache)
        host_indices = torch.arange(4)
        cache.cache_controller.mem_pool_host.alloc.side_effect = [
            None,
            host_indices,
        ]
        self.assertTrue(operation.finish_query(["h1", "h2"], 4))
        cache.cache_controller.prefetch_query_result_queue.put(operation)

        cache._drain_query_first_results()

        self.assertEqual(
            cache.cache_controller.mem_pool_host.alloc.call_args_list,
            [call(4), call(4)],
        )
        cache.evict_host.assert_called_once_with(4, allow_l1_l2=True)
        cache.cache_controller.resume_prefetch.assert_called_once_with(
            operation, host_indices, 4
        )

    def test_partial_allocation_trims_the_read_after_confirmed_hit(self):
        cache = self._make_cache()
        _, operation = self._submit(cache)
        host_indices = torch.arange(4)
        cache.cache_controller.mem_pool_host.alloc.side_effect = [
            None,
            None,
            host_indices,
        ]
        cache.cache_controller.mem_pool_host.available_size.return_value = 5
        self.assertTrue(operation.finish_query(["h1", "h2", "h3", "h4"], 8))
        cache.cache_controller.prefetch_query_result_queue.put(operation)

        cache._drain_query_first_results()

        self.assertEqual(
            cache.cache_controller.mem_pool_host.alloc.call_args_list,
            [call(8), call(8), call(4)],
        )
        cache.evict_host.assert_called_once_with(8, allow_l1_l2=True)
        cache.cache_controller.resume_prefetch.assert_called_once_with(
            operation, host_indices, 4
        )
        self.assertEqual(operation.hash_value, ["h1", "h2"])

    def test_completed_read_rejoins_existing_host_insert_path(self):
        cache = self._make_cache()
        anchor, operation = self._submit(cache)
        host_indices = torch.arange(4)
        cache.cache_controller.mem_pool_host.alloc.return_value = host_indices
        self.assertTrue(operation.finish_query(["h1", "h2"], 4))
        cache.cache_controller.prefetch_query_result_queue.put(operation)
        cache._drain_query_first_results()

        operation.completed_tokens = 4
        cache.can_terminate_prefetch = MagicMock(return_value=True)
        cache.cache_controller.terminate_prefetch.return_value = (
            4,
            operation.hash_value,
        )
        cache._all_reduce_attn_groups = MagicMock()
        cache._insert_helper_host = MagicMock(return_value=0)

        self.assertTrue(cache.check_prefetch_progress("req"))

        inserted_key = cache._insert_helper_host.call_args.args[1]
        inserted_indices = cache._insert_helper_host.call_args.args[2]
        inserted_hashes = cache._insert_helper_host.call_args.args[3]
        self.assertEqual(len(inserted_key), 4)
        self.assertTrue(torch.equal(inserted_indices, host_indices))
        self.assertEqual(inserted_hashes, ["h1", "h2"])
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid["req"], 4)

    def test_abort_during_query_releases_anchor_and_budget_once(self):
        cache = self._make_cache()
        anchor, operation = self._submit(cache)

        cache.release_aborted_request("req")
        cache.release_aborted_request("req")

        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        self.assertTrue(operation.is_terminated())
        cache.cache_controller.mem_pool_host.free.assert_not_called()

    def test_cancelled_resume_frees_new_allocation_and_tracking(self):
        cache = self._make_cache()
        anchor, operation = self._submit(cache)
        host_indices = torch.arange(4)
        cache.cache_controller.mem_pool_host.alloc.return_value = host_indices
        cache.cache_controller.resume_prefetch.side_effect = None
        cache.cache_controller.resume_prefetch.return_value = False
        self.assertTrue(operation.finish_query(["h1", "h2"], 4))
        cache.cache_controller.prefetch_query_result_queue.put(operation)

        cache._drain_query_first_results()

        cache.cache_controller.mem_pool_host.free.assert_called_once_with(host_indices)
        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        self.assertTrue(operation.is_terminated())

    def test_stale_query_result_cannot_resume_reused_request_id(self):
        cache = self._make_cache()
        _, current_operation = self._submit(cache)
        stale_operation = PrefetchOperation(
            "req",
            None,
            [1, 2],
            query_first=True,
        )
        self.assertTrue(stale_operation.finish_query(["stale"], 2))
        cache.cache_controller.prefetch_query_result_queue.put(stale_operation)

        cache._drain_query_first_results()

        self.assertIs(cache.ongoing_prefetch["req"][3], current_operation)
        self.assertFalse(current_operation.is_read_started())
        cache.cache_controller.mem_pool_host.alloc.assert_not_called()

    def test_query_timeout_cleans_up_without_waiting_for_backend_thread(self):
        cache = self._make_cache()
        anchor, operation = self._submit(cache)
        cache.is_prefetch_timeout = MagicMock(return_value=True)

        self.assertTrue(cache.check_prefetch_progress("req"))

        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertTrue(operation.is_terminated())

    def test_best_effort_does_not_wait_for_query(self):
        cache = self._make_cache()
        cache.prefetch_stop_policy = "best_effort"
        anchor, operation = self._submit(cache)

        self.assertTrue(cache.check_prefetch_progress("req"))

        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertTrue(operation.is_terminated())


if __name__ == "__main__":
    unittest.main()
