import unittest
from array import array
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.cache_controller import PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.hiradix_cache import (
    HiRadixCache,
    LoadBackContext,
    PrefetchContext,
    WriteThroughContext,
)
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _node(tokens, *, parent=None, values=None):
    node = TreeNode()
    node.parent = parent
    node.key = RadixKey(array("q", tokens))
    node.value = (
        torch.tensor(values, dtype=torch.int64) if values is not None else None
    )
    return node


class TestHiCacheBufferOnly(unittest.TestCase):
    def test_query_first_termination_keeps_current_backend_result(self):
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.mark_terminate()

        self.assertFalse(operation.increment(4))
        self.assertEqual(operation.completed_tokens, 4)

    def test_loading_node_is_hidden_from_regular_prefix_match(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2

        root = _node([], values=[])
        child = _node([1, 2, 3, 4], parent=root, values=[10, 11, 12, 13])
        child.device_loading = True
        root.children[child.key.child_key(cache.page_size)] = child

        values, last_node = cache._match_prefix_helper(root, child.key)
        self.assertEqual(values, [])
        self.assertIs(last_node, root)

    def test_unfinished_request_rematches_indices_loaded_from_l3(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.disable = False
        cache.page_size = 2
        cache.is_eagle = False
        cache.buffer_only = True
        mem_pool_host = MagicMock()
        finish_event = MagicMock()
        cache.cache_controller = SimpleNamespace(
            write_policy="write_back",
            mem_pool_host=mem_pool_host,
            ack_load_queue=[],
        )
        cache.token_to_kv_pool_allocator = SimpleNamespace(free=MagicMock())
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12, 13]], dtype=torch.int64),
            write=MagicMock(),
        )
        cache._empty_match_result = SimpleNamespace(
            device_indices=torch.empty(0, dtype=torch.int64)
        )
        cache.dec_lock_ref = MagicMock()
        cache.inc_lock_ref = MagicMock()
        cache._tag_session_leaf = MagicMock()
        cache._record_store_event = MagicMock()
        cache._update_host_leaf_status = MagicMock()
        cache.transient_host_nodes_by_reqid = {}

        root = _node([], values=[])
        root.host_value = []
        child = _node([1, 2, 3, 4], parent=root, values=[10, 11, 12, 13])
        child.host_value = torch.arange(4)
        child.host_value_transient = True
        child.transient_owner_req_id = "req"
        child.device_loading = True
        child.protect_host()
        root.children[child.key.child_key(cache.page_size)] = child
        cache.root_node = root
        cache.transient_host_nodes_by_reqid["req"] = [child]
        cache.ongoing_load_back = {
            child.id: LoadBackContext(end_node=child, host_nodes=[child])
        }
        cache.cache_controller.ack_load_queue.append(
            (0, finish_event, [child.id])
        )

        req = SimpleNamespace(
            req_pool_idx=0,
            extra_key=None,
            cache_protected_len=4,
            last_node=child,
            prefix_indices=torch.arange(4),
            priority=0,
            get_fill_ids=lambda: [1, 2, 3, 4],
        )

        cache.cache_unfinished_req(req, chunked=True)

        self.assertTrue(torch.equal(req.prefix_indices, child.value))
        self.assertIs(req.last_node, child)
        self.assertFalse(child.device_loading)
        self.assertIsNone(child.host_value)
        self.assertEqual(cache.cache_controller.ack_load_queue, [])
        finish_event.synchronize.assert_called_once()
        mem_pool_host.free.assert_called_once()

    def test_loading_prefix_sync_does_not_wait_for_later_unrelated_load(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache._record_store_event = MagicMock()
        cache.dec_lock_ref = MagicMock()

        root = _node([], values=[])
        target = _node([1, 2], parent=root, values=[10, 11])
        unrelated = _node([9, 10], parent=root, values=[20, 21])
        for node in (target, unrelated):
            node.host_value = torch.arange(2)
            node.device_loading = True
            node.protect_host()
            root.children[node.key.child_key(cache.page_size)] = node
        cache.root_node = root

        target_event = MagicMock()
        unrelated_event = MagicMock()
        cache.ongoing_load_back = {
            target.id: LoadBackContext(end_node=target, host_nodes=[target]),
            unrelated.id: LoadBackContext(
                end_node=unrelated, host_nodes=[unrelated]
            ),
        }
        cache.cache_controller = SimpleNamespace(
            ack_load_queue=[
                (0, target_event, [target.id]),
                (1, unrelated_event, [unrelated.id]),
            ]
        )

        cache._synchronize_loading_prefix(target.key)

        self.assertFalse(target.device_loading)
        self.assertTrue(unrelated.device_loading)
        self.assertNotIn(target.id, cache.ongoing_load_back)
        self.assertIn(unrelated.id, cache.ongoing_load_back)
        self.assertEqual(len(cache.cache_controller.ack_load_queue), 1)
        target_event.synchronize.assert_called_once()
        unrelated_event.synchronize.assert_not_called()

    def test_recomputation_releases_transient_l3_host_pages(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.is_eagle = False
        cache.buffer_only = True
        cache.evictable_size_ = 0
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.transient_host_nodes_by_reqid = {}
        mem_pool_host = MagicMock()
        cache.cache_controller = SimpleNamespace(mem_pool_host=mem_pool_host)

        root = _node([], values=[])
        child = _node([1, 2, 3, 4], parent=root)
        child.host_value = torch.arange(4)
        child.host_value_transient = True
        child.transient_owner_req_id = "req"
        root.children[child.key.child_key(cache.page_size)] = child
        cache.root_node = root
        cache.evictable_host_leaves.add(child)
        cache.transient_host_nodes_by_reqid["req"] = [child]

        cache.insert(
            InsertParams(
                key=child.key,
                value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            )
        )

        mem_pool_host.free.assert_called_once()
        self.assertIsNone(child.host_value)
        self.assertFalse(child.host_value_transient)
        self.assertNotIn("req", cache.transient_host_nodes_by_reqid)

    def test_device_eviction_drops_unprotected_transient_host_subtree(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache.evictable_size_ = 2
        cache.protected_size_ = 0
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.transient_host_nodes_by_reqid = {}
        cache.eviction_strategy = SimpleNamespace(
            get_priority=lambda node: node.id
        )
        cache._record_remove_event = MagicMock()
        cache._discard_session_leaf = MagicMock()
        mem_pool_host = MagicMock()
        mem_pool_device_allocator = MagicMock()
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            mem_pool_device_allocator=mem_pool_device_allocator,
        )

        root = _node([], values=[])
        root.lock_ref = 1
        parent = _node([1, 2], parent=root, values=[10, 11])
        child = _node([3, 4], parent=parent)
        child.host_value = torch.arange(2)
        child.host_value_transient = True
        child.transient_owner_req_id = "req"
        root.children[parent.key.child_key(cache.page_size)] = parent
        parent.children[child.key.child_key(cache.page_size)] = child
        cache.root_node = root
        cache.evictable_host_leaves.add(child)
        cache.transient_host_nodes_by_reqid["req"] = [child]
        cache._update_leaf_status(parent)

        num_evicted = cache._evict_write_through(2)

        self.assertEqual(num_evicted, 2)
        mem_pool_host.free.assert_called_once()
        mem_pool_device_allocator.free.assert_called_once()
        self.assertEqual(parent.children, {})
        self.assertNotIn(parent.key.child_key(cache.page_size), root.children)
        self.assertNotIn("req", cache.transient_host_nodes_by_reqid)

    def test_device_eviction_preserves_referenced_transient_host_subtree(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.transient_host_nodes_by_reqid = {}
        cache.eviction_strategy = SimpleNamespace(
            get_priority=lambda node: node.id
        )
        mem_pool_host = MagicMock()
        mem_pool_device_allocator = MagicMock()
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            mem_pool_device_allocator=mem_pool_device_allocator,
        )

        root = _node([], values=[])
        root.lock_ref = 1
        parent = _node([1, 2], parent=root, values=[10, 11])
        child = _node([3, 4], parent=parent)
        child.host_value = torch.arange(2)
        child.host_value_transient = True
        child.protect_host()
        root.children[parent.key.child_key(cache.page_size)] = parent
        parent.children[child.key.child_key(cache.page_size)] = child
        cache.root_node = root
        cache._update_leaf_status(parent)

        num_evicted = cache._evict_write_through(2)

        self.assertEqual(num_evicted, 0)
        mem_pool_host.free.assert_not_called()
        mem_pool_device_allocator.free.assert_not_called()
        self.assertIs(
            root.children[parent.key.child_key(cache.page_size)], parent
        )
        self.assertIs(
            parent.children[child.key.child_key(cache.page_size)], child
        )

    def test_split_propagates_pending_write_and_storage_state(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.ongoing_write_through = {}

        root = _node([], values=[])
        child = _node([1, 2, 3, 4], parent=root, values=[10, 11, 12, 13])
        child.hash_value = ["h1", "h2"]
        child.storage_backed_tokens = 4
        root.children[child.key.child_key(cache.page_size)] = child

        context = WriteThroughContext(
            write_id=child.id,
            lock_node=child,
            backup_len=4,
            publish_nodes=[child],
        )
        child.write_through_pending_id = context.write_id
        cache.ongoing_write_through[context.write_id] = context

        prefix = cache._split_node(child.key, child, 2)

        self.assertEqual(prefix.storage_backed_tokens, 2)
        self.assertEqual(child.storage_backed_tokens, 2)
        self.assertEqual(prefix.write_through_pending_id, context.write_id)
        self.assertEqual(child.write_through_pending_id, context.write_id)
        self.assertEqual(context.publish_nodes, [prefix, child])

    def test_partial_l3_ack_marks_only_completed_prefix_and_frees_host(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.enable_storage_metrics = False
        cache.ongoing_prefetch = {}
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}
        cache._all_reduce_attn_groups = lambda tensor, op: None

        root = _node([], values=[])
        first = _node([1, 2], parent=root, values=[10, 11])
        second = _node([3, 4], parent=first, values=[12, 13])
        context = WriteThroughContext(
            write_id=first.id,
            lock_node=second,
            backup_len=4,
            publish_nodes=[first, second],
            host_indices=torch.arange(4),
        )
        first.write_through_pending_id = context.write_id
        second.write_through_pending_id = context.write_id
        cache.ongoing_write_through[context.write_id] = context

        operation = SimpleNamespace(id=123, completed_tokens=2)
        mem_pool_host = MagicMock()
        controller = SimpleNamespace(
            prefetch_revoke_queue=Queue(),
            prefetch_hit_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            mem_pool_host=mem_pool_host,
        )
        controller.ack_backup_queue.put(operation)
        cache.cache_controller = controller
        cache.ongoing_backup[operation.id] = context

        cache._drain_storage_control_queues_impl(
            n_revoke=0,
            n_hit=0,
            n_backup=1,
            n_release=0,
            log_metrics=False,
        )

        self.assertTrue(first.storage_backed)
        self.assertFalse(second.storage_backed)
        self.assertIsNone(first.write_through_pending_id)
        self.assertIsNone(second.write_through_pending_id)
        self.assertNotIn(context.write_id, cache.ongoing_write_through)
        mem_pool_host.free.assert_called_once()

    def test_split_propagates_inflight_load_back_ownership(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.ongoing_write_through = {}

        root = _node([], values=[])
        child = _node([1, 2, 3, 4], parent=root, values=[10, 11, 12, 13])
        child.host_value = torch.arange(4)
        child.hash_value = ["h1", "h2"]
        child.host_value_transient = True
        child.device_loading = True
        child.protect_host()
        root.children[child.key.child_key(cache.page_size)] = child
        context = LoadBackContext(end_node=child, host_nodes=[child])
        cache.ongoing_load_back = {child.id: context}

        prefix = cache._split_node(child.key, child, 2)

        self.assertEqual(context.host_nodes, [prefix, child])
        self.assertEqual(prefix.host_ref_counter, 1)
        self.assertEqual(child.host_ref_counter, 1)

    def test_prefetch_soft_reserve_is_borrowable_without_write_pressure(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_write_reserve_ratio = 0.2
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}
        cache._all_reduce_attn_groups = lambda tensor, op: None

        mem_pool_host = MagicMock()
        mem_pool_host.size = 100
        mem_pool_host.available_size.return_value = 100
        mem_pool_host.alloc.side_effect = lambda size: torch.arange(size)
        controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            prefetch_tokens_occupied=0,
            prefetch_buffer=Queue(),
        )
        cache.cache_controller = controller

        operation = PrefetchOperation("req", None, list(range(100)))
        operation.set_query_result([f"h{i}" for i in range(50)], 100)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(100))),
            operation=operation,
        )

        self.assertTrue(cache._allocate_buffer_prefetch(context))
        self.assertEqual(context.allocated_tokens, 100)

        operation2 = PrefetchOperation("req2", None, list(range(100)))
        operation2.set_query_result([f"h{i}" for i in range(50)], 100)
        context2 = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(100))),
            operation=operation2,
        )
        controller.prefetch_tokens_occupied = 0
        mem_pool_host.available_size.return_value = 100
        cache.ongoing_write_through = {1: object()}

        self.assertTrue(cache._allocate_buffer_prefetch(context2))
        self.assertEqual(context2.allocated_tokens, 80)

    def test_aborted_request_releases_unconsumed_transient_host_node(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.evictable_host_leaves = set()
        cache.transient_host_nodes_by_reqid = {}
        cache._update_leaf_status = lambda node: None
        cache._update_host_leaf_status = lambda node: None

        root = _node([], values=[])
        child = _node([1, 2], parent=root)
        child.host_value = torch.arange(2)
        child.host_value_transient = True
        child.transient_owner_req_id = "req"
        root.children[child.key.child_key(cache.page_size)] = child
        cache.evictable_host_leaves.add(child)
        cache.transient_host_nodes_by_reqid["req"] = [child]
        mem_pool_host = MagicMock()
        cache.cache_controller = SimpleNamespace(mem_pool_host=mem_pool_host)

        cache._release_transient_host_nodes("req")

        mem_pool_host.free.assert_called_once()
        self.assertIsNone(child.host_value)
        self.assertNotIn(child.key.child_key(cache.page_size), root.children)
        self.assertNotIn("req", cache.transient_host_nodes_by_reqid)

    def test_prefetch_query_is_capped_by_shared_buffer_budget(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = True
        cache.enable_storage = True
        cache.is_eagle = False
        cache.buffer_query_tokens_occupied = 6
        cache.ongoing_prefetch = {}
        cache._get_extra_pools = lambda: {}
        cache.inc_lock_ref = MagicMock()

        controller = MagicMock()
        controller.mem_pool_host.size = 10
        controller.prefetch_rate_limited.return_value = False
        controller.prefetch.return_value = PrefetchOperation(
            "req", None, list(range(4))
        )
        cache.cache_controller = controller

        anchor = _node([], values=[])
        cache.prefetch_from_storage("req", anchor, list(range(20)))

        controller.prefetch.assert_called_once()
        self.assertEqual(len(controller.prefetch.call_args.args[2]), 4)
        self.assertEqual(cache.buffer_query_tokens_occupied, 10)


if __name__ == "__main__":
    unittest.main()
