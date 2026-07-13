import unittest
from array import array
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.cache_controller import PrefetchOperation
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
