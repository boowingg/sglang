import sys
import unittest
from array import array
from collections import OrderedDict, defaultdict
from unittest.mock import MagicMock, call

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.evict_policy import LRUStrategy
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeCacheController:
    def __init__(self):
        self.calls = []
        self.mem_pool_host = MagicMock()
        self._host_available = 0
        self.mem_pool_host.available_size.side_effect = lambda: self._host_available

    def evict_device(self, indices):
        self.calls.append(("device", indices.clone()))
        return len(indices)

    def evict_host(self, indices):
        self.calls.append(("host", indices.clone()))
        self._host_available += len(indices)
        return len(indices)


class TestHiRadixL1L2LeafReclaim(unittest.TestCase):
    def _make_cache(self):
        cache = object.__new__(HiRadixCache)
        cache.page_size = 1
        cache.root_node = TreeNode(priority=-sys.maxsize)
        cache.root_node.key = RadixKey(array("q"))
        cache.root_node.value = []
        cache.root_node.host_value = []
        cache.root_node.lock_ref = 1
        cache.root_node.hash_value = []
        cache.evictable_size_ = 0
        cache.protected_size_ = 0
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.eviction_strategy = LRUStrategy()
        cache.cache_controller = _FakeCacheController()
        cache.enable_kv_cache_events = False
        cache._session_leaves = defaultdict(set)
        cache._closed_session_ids = OrderedDict()
        cache._record_remove_event = MagicMock()
        return cache

    def _add_node(
        self,
        cache,
        parent,
        tokens,
        *,
        with_device,
        with_host,
        last_access_time,
    ):
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(array("q", tokens))
        node.value = (
            torch.arange(len(tokens), dtype=torch.int64) if with_device else None
        )
        node.host_value = (
            torch.arange(100, 100 + len(tokens), dtype=torch.int64)
            if with_host
            else None
        )
        node.last_access_time = last_access_time
        parent.children[node.key.child_key(cache.page_size)] = node

        if with_device:
            cache.evictable_size_ += len(tokens)
        cache._update_leaf_status(parent)
        cache._update_leaf_status(node)
        cache._update_host_leaf_status(parent)
        cache._update_host_leaf_status(node)
        return node

    def test_default_host_eviction_keeps_l1_l2_leaf(self):
        cache = self._make_cache()
        leaf = self._add_node(
            cache,
            cache.root_node,
            [1, 2],
            with_device=True,
            with_host=True,
            last_access_time=1.0,
        )

        self.assertEqual(cache.evict_host(2), 0)
        self.assertIs(
            cache.root_node.children[leaf.key.child_key(cache.page_size)], leaf
        )
        self.assertEqual(cache.cache_controller.calls, [])

    def test_l2_only_is_reclaimed_before_l1_l2_leaf(self):
        cache = self._make_cache()
        l2_only = self._add_node(
            cache,
            cache.root_node,
            [1, 2],
            with_device=False,
            with_host=True,
            last_access_time=10.0,
        )
        l1_l2 = self._add_node(
            cache,
            cache.root_node,
            [3, 4, 5],
            with_device=True,
            with_host=True,
            last_access_time=1.0,
        )

        self.assertEqual(cache.evict_host(4, allow_l1_l2=True), 5)
        self.assertEqual(
            [medium for medium, _ in cache.cache_controller.calls],
            ["host", "device", "host"],
        )
        self.assertNotIn(
            l2_only.key.child_key(cache.page_size), cache.root_node.children
        )
        self.assertNotIn(l1_l2.key.child_key(cache.page_size), cache.root_node.children)
        self.assertEqual(cache.evictable_size_, 0)
        self.assertIsNone(l1_l2.parent)
        self.assertIsNone(l1_l2.value)
        self.assertIsNone(l1_l2.host_value)
        self.assertEqual(
            cache._record_remove_event.call_args_list,
            [
                call(l2_only, medium=StorageMedium.CPU),
                call(l1_l2, medium=StorageMedium.GPU),
                call(l1_l2, medium=StorageMedium.CPU),
            ],
        )

    def test_existing_free_space_prevents_unnecessary_l1_reclaim(self):
        cache = self._make_cache()
        cache.cache_controller._host_available = 2
        l2_only = self._add_node(
            cache,
            cache.root_node,
            [1, 2],
            with_device=False,
            with_host=True,
            last_access_time=1.0,
        )
        l1_l2 = self._add_node(
            cache,
            cache.root_node,
            [3, 4],
            with_device=True,
            with_host=True,
            last_access_time=2.0,
        )

        self.assertEqual(cache.evict_host(4, allow_l1_l2=True), 2)
        self.assertNotIn(
            l2_only.key.child_key(cache.page_size), cache.root_node.children
        )
        self.assertIs(
            cache.root_node.children[l1_l2.key.child_key(cache.page_size)], l1_l2
        )
        self.assertEqual(
            [medium for medium, _ in cache.cache_controller.calls],
            ["host"],
        )

    def test_device_leaf_with_topology_child_is_not_reclaimed(self):
        cache = self._make_cache()
        parent = self._add_node(
            cache,
            cache.root_node,
            [1, 2],
            with_device=True,
            with_host=True,
            last_access_time=1.0,
        )
        child = self._add_node(
            cache,
            parent,
            [3, 4],
            with_device=False,
            with_host=True,
            last_access_time=2.0,
        )

        self.assertIn(parent, cache.evictable_leaves)
        self.assertEqual(cache._evict_l1_l2_leaves(2), 0)
        self.assertIs(
            cache.root_node.children[parent.key.child_key(cache.page_size)], parent
        )
        self.assertIs(parent.children[child.key.child_key(cache.page_size)], child)
        self.assertEqual(cache.cache_controller.calls, [])

    def test_referenced_or_pending_leaf_is_not_reclaimed(self):
        cache = self._make_cache()
        host_referenced = self._add_node(
            cache,
            cache.root_node,
            [1],
            with_device=True,
            with_host=True,
            last_access_time=1.0,
        )
        pending = self._add_node(
            cache,
            cache.root_node,
            [2],
            with_device=True,
            with_host=True,
            last_access_time=2.0,
        )
        locked = self._add_node(
            cache,
            cache.root_node,
            [3],
            with_device=True,
            with_host=True,
            last_access_time=3.0,
        )
        host_referenced.protect_host()
        pending.write_through_pending_id = pending.id
        locked.lock_ref = 1
        cache.evictable_leaves.add(locked)

        self.assertEqual(cache._evict_l1_l2_leaves(3), 0)
        self.assertEqual(len(cache.root_node.children), 3)
        self.assertEqual(cache.cache_controller.calls, [])

    def test_parent_is_promoted_only_after_child_is_removed(self):
        cache = self._make_cache()
        parent = self._add_node(
            cache,
            cache.root_node,
            [1, 2],
            with_device=True,
            with_host=True,
            last_access_time=2.0,
        )
        child = self._add_node(
            cache,
            parent,
            [3, 4],
            with_device=True,
            with_host=True,
            last_access_time=1.0,
        )

        self.assertNotIn(parent, cache.evictable_leaves)
        self.assertIn(child, cache.evictable_leaves)
        self.assertEqual(cache._evict_l1_l2_leaves(4), 4)
        self.assertEqual(len(cache.root_node.children), 0)
        self.assertEqual(cache.evictable_size_, 0)
        self.assertEqual(
            [medium for medium, _ in cache.cache_controller.calls],
            ["device", "host", "device", "host"],
        )

    def test_reclaim_removes_session_leaf_index(self):
        cache = self._make_cache()
        leaf = self._add_node(
            cache,
            cache.root_node,
            [1, 2],
            with_device=True,
            with_host=True,
            last_access_time=1.0,
        )
        leaf.session_ids = {"session"}
        cache._session_leaves["session"].add(leaf)

        self.assertEqual(cache._evict_l1_l2_leaves(2), 2)
        self.assertNotIn("session", cache._session_leaves)
        self.assertFalse(hasattr(leaf, "session_ids"))

    def test_stale_detached_candidate_is_skipped(self):
        cache = self._make_cache()
        leaf = self._add_node(
            cache,
            cache.root_node,
            [1, 2],
            with_device=True,
            with_host=True,
            last_access_time=1.0,
        )
        cache.root_node.children.pop(leaf.key.child_key(cache.page_size))

        self.assertEqual(cache._evict_l1_l2_leaves(2), 0)
        self.assertEqual(cache.cache_controller.calls, [])

    def test_distributed_cache_keeps_original_l2_only_policy(self):
        cache = self._make_cache()
        cache.tp_world_size = 2
        leaf = self._add_node(
            cache,
            cache.root_node,
            [1, 2],
            with_device=True,
            with_host=True,
            last_access_time=1.0,
        )

        self.assertEqual(cache.evict_host(2, allow_l1_l2=True), 0)
        self.assertIs(
            cache.root_node.children[leaf.key.child_key(cache.page_size)], leaf
        )
        self.assertTrue(cache._l1_l2_read_reclaim_unsupported_warned)

    def test_prefetch_enables_l1_l2_fallback(self):
        cache = object.__new__(HiRadixCache)
        cache.page_size = 1
        cache.is_eagle = False
        cache.enable_storage = True
        cache.prefetch_threshold = 1
        cache.ongoing_prefetch = {}
        cache.evict_host = MagicMock(return_value=4)
        cache._get_extra_pools = MagicMock(return_value={})

        operation = MagicMock()
        controller = MagicMock()
        controller.prefetch_rate_limited.return_value = False
        controller.mem_pool_host.alloc.side_effect = [
            None,
            torch.arange(4, dtype=torch.int64),
        ]
        controller.prefetch.return_value = operation
        controller.prefetch_tokens_occupied = 0
        cache.cache_controller = controller

        anchor = TreeNode()
        anchor.key = RadixKey(array("q"))
        anchor.host_value = []

        cache.prefetch_from_storage("req", anchor, [1, 2, 3, 4])

        cache.evict_host.assert_called_once_with(4, allow_l1_l2=True)
        self.assertEqual(anchor.host_ref_counter, 1)
        self.assertIn("req", cache.ongoing_prefetch)

    def test_write_retry_uses_original_host_only_policy(self):
        cache = object.__new__(HiRadixCache)
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey(array("q"))
        cache.root_node.value = []
        cache.root_node.host_value = []
        cache.ongoing_write_through = {}
        cache.evict_host = MagicMock(return_value=2)
        cache._get_extra_pools = MagicMock(return_value={})

        controller = MagicMock()
        controller.write.side_effect = [
            None,
            torch.arange(2, dtype=torch.int64),
        ]
        cache.cache_controller = controller

        node = TreeNode()
        node.parent = cache.root_node
        node.key = RadixKey(array("q", [1, 2]))
        node.value = torch.arange(2, dtype=torch.int64)

        self.assertEqual(cache.write_backup(node, write_back=True), 2)
        cache.evict_host.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
