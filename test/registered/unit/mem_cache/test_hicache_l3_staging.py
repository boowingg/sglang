import threading
import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    HiCachePeriodicStats,
    PrefetchOperation,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode


class _FakeHostPool:
    page_size = 2
    size = 16
    size_per_token = 1

    def __init__(self, available=16):
        self.free_slots = torch.arange(available, dtype=torch.int64)
        self.alloc_requests = []

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size):
        self.alloc_requests.append(need_size)
        if need_size > len(self.free_slots):
            return None
        result = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return result

    def free(self, indices):
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])
        return len(indices)


class _Priority:
    @staticmethod
    def get_priority(node):
        return node.last_access_time


class TestHiCacheL3Staging(unittest.TestCase):
    def _budget_controller(self, available=16):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.prefetch_capacity_limit = 10
        controller.prefetch_tokens_occupied = 0
        controller.prefetch_tokens_reserved = 0
        controller._prefetch_budget_lock = threading.Lock()
        controller._periodic_stats = HiCachePeriodicStats()
        controller.mem_pool_host = _FakeHostPool(available)
        return controller

    def test_periodic_stats_keep_window_and_cumulative_values(self):
        stats = HiCachePeriodicStats()
        stats.add("tokens", 10)
        window, cumulative = stats.snapshot_and_reset()
        self.assertEqual(window["tokens"], 10)
        self.assertEqual(cumulative["tokens"], 10)

        stats.add("tokens", 4)
        window, cumulative = stats.snapshot_and_reset()
        self.assertEqual(window["tokens"], 4)
        self.assertEqual(cumulative["tokens"], 14)

        stats.add_many({"tokens": 3, "requests": 1})
        window, cumulative = stats.snapshot_and_reset()
        self.assertEqual(window, {"tokens": 3, "requests": 1})
        self.assertEqual(cumulative, {"tokens": 17, "requests": 1})

    def test_prefetch_budget_tracks_reserved_and_allocated_tokens(self):
        controller = self._budget_controller(available=8)

        accepted = controller.reserve_prefetch_tokens(12)
        self.assertEqual(accepted, 10)
        self.assertEqual(controller.prefetch_tokens_reserved, 10)
        self.assertEqual(controller.prefetch_tokens_occupied, 10)

        operation = PrefetchOperation("req", None, list(range(12)))
        operation.reserved_tokens = accepted
        controller.commit_prefetch_allocation(operation, 6)
        operation.host_indices = torch.arange(6)

        self.assertEqual(controller.prefetch_tokens_reserved, 0)
        self.assertEqual(controller.prefetch_tokens_occupied, 6)
        self.assertEqual(operation.reserved_tokens, 6)

        controller.release_prefetch_tokens(operation)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)
        controller.release_prefetch_tokens(operation)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)

    def test_resident_allocation_yields_to_discovered_prefetch(self):
        controller = self._budget_controller(available=6)
        controller.prefetch_tokens_reserved = 4

        self.assertTrue(controller.resident_allocation_available(2))
        self.assertFalse(controller.resident_allocation_available(4))

    def test_query_first_allocates_only_confirmed_l3_hit(self):
        host_pool = _FakeHostPool(available=16)
        controller = mock.Mock()
        controller.mem_pool_host = host_pool

        operation = PrefetchOperation("req", None, list(range(8)))
        operation.hash_value = ["page-0", "page-1"]
        operation.storage_hit_tokens = 4
        operation.reserved_tokens = 4
        operation.query_done.set()

        root = TreeNode()
        root.key = RadixKey(array("q"))
        root.value = torch.empty(0, dtype=torch.int64)
        root.host_value = torch.empty(0, dtype=torch.int64)

        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 2
        cache.cache_controller = controller
        cache.evict_host = mock.Mock()
        cache._all_reduce_attn_groups = mock.Mock()
        cache.ongoing_prefetch = {
            "req": (
                root,
                RadixKey(array("q", range(8))),
                None,
                operation,
            )
        }

        self.assertTrue(cache._start_discovered_prefetch("req"))
        self.assertEqual(host_pool.alloc_requests, [4])
        self.assertEqual(len(cache.ongoing_prefetch["req"][1]), 4)
        controller.commit_prefetch_allocation.assert_called_once_with(operation, 4)
        controller.start_prefetch_io.assert_called_once()

    def test_storage_backed_gpu_node_can_release_only_l2_copy(self):
        root = TreeNode()
        root.key = RadixKey(array("q"))
        root.value = torch.empty(0, dtype=torch.int64)
        root.host_value = torch.empty(0, dtype=torch.int64)

        node = TreeNode()
        node.parent = root
        node.key = RadixKey(array("q", [1, 2]))
        node.value = torch.tensor([4, 5])
        node.host_value = torch.tensor([8, 9])
        node.storage_backed = True
        root.children[node.key.child_key(2)] = node

        controller = mock.Mock()
        controller.evict_host.side_effect = lambda indices: len(indices)

        cache = HiRadixCache.__new__(HiRadixCache)
        cache.root_node = root
        cache.page_size = 2
        cache.cache_controller = controller
        cache.eviction_strategy = _Priority()
        cache.evictable_host_leaves = set()
        cache._record_remove_event = mock.Mock()

        cache._update_host_leaf_status(node)
        self.assertIn(node, cache.evictable_host_leaves)

        node.protect_host()
        cache.evict_host(2, allow_storage_backed_gpu_copy=True)
        self.assertIsNotNone(node.host_value)
        node.release_host()
        cache._update_host_leaf_status(node)

        cache.evict_host(2)
        self.assertIsNotNone(node.host_value)

        cache.evict_host(2, allow_storage_backed_gpu_copy=True)
        self.assertIsNotNone(node.value)
        self.assertIsNone(node.host_value)
        self.assertIs(root.children[node.key.child_key(2)], node)
        controller.evict_host.assert_called_once()

    def test_storage_ack_marks_only_fully_completed_nodes(self):
        root = TreeNode()
        root.key = RadixKey(array("q"))
        root.value = torch.empty(0, dtype=torch.int64)
        root.host_value = torch.empty(0, dtype=torch.int64)

        parent = TreeNode()
        parent.parent = root
        parent.key = RadixKey(array("q", [1, 2]))
        parent.value = torch.tensor([1, 2])
        parent.host_value = torch.tensor([3, 4])
        child = TreeNode()
        child.parent = parent
        child.key = RadixKey(array("q", [3, 4]))
        child.value = torch.tensor([5, 6])
        child.host_value = torch.tensor([7, 8])
        parent.children[child.key.child_key(2)] = child
        root.children[parent.key.child_key(2)] = parent

        cache = HiRadixCache.__new__(HiRadixCache)
        cache.root_node = root
        cache.evictable_host_leaves = set()

        cache._mark_storage_backed(child, backup_len=4, completed_tokens=2)
        self.assertTrue(parent.storage_backed)
        self.assertFalse(child.storage_backed)


if __name__ == "__main__":
    unittest.main()
