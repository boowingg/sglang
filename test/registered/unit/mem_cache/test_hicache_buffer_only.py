import inspect
import unittest
from array import array
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.buffer_only import hiradix_cache as buffer_cache_module
from sglang.srt.mem_cache.buffer_only.cache_controller import (
    BufferOnlyHiCacheController,
)
from sglang.srt.mem_cache.buffer_only.cache_controller import (
    BufferPrefetchOperation as PrefetchOperation,
)
from sglang.srt.mem_cache.buffer_only.hiradix_cache import (
    BufferOnlyHiRadixCache,
    BufferTreeNode,
    LoadBackContext,
    PrefetchContext,
    WriteThroughContext,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _node(tokens, *, parent=None, values=None):
    node = BufferTreeNode()
    node.parent = parent
    node.key = RadixKey(array("q", tokens))
    node.value = torch.tensor(values, dtype=torch.int64) if values is not None else None
    return node


class TestHiCacheBufferOnly(unittest.TestCase):
    def test_radix_node_factory_is_opt_in(self):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )

        default_cache = RadixCache(params)

        class BufferRadixCache(RadixCache):
            node_cls = BufferTreeNode

        buffer_cache = BufferRadixCache(params)
        buffer_cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2])),
                value=torch.tensor([10, 11]),
            )
        )

        self.assertIs(type(default_cache.root_node), TreeNode)
        self.assertIsInstance(buffer_cache.root_node, BufferTreeNode)
        self.assertTrue(
            all(
                isinstance(node, BufferTreeNode)
                for node in buffer_cache.root_node.children.values()
            )
        )

    def test_overridden_methods_track_upstream_call_signatures(self):
        intentional_extensions = {
            (BufferOnlyHiRadixCache, "_drain_storage_control_queues_impl"),
            (BufferOnlyHiRadixCache, "_insert_helper_host"),
        }

        for subclass, baseclass in (
            (BufferOnlyHiRadixCache, HiRadixCache),
            (BufferOnlyHiCacheController, HiCacheController),
        ):
            for name, method in subclass.__dict__.items():
                if (
                    not inspect.isfunction(method)
                    or not hasattr(baseclass, name)
                    or (subclass, name) in intentional_extensions
                ):
                    continue

                def call_shape(callable_obj):
                    return [
                        (parameter.name, parameter.kind, parameter.default)
                        for parameter in inspect.signature(
                            callable_obj
                        ).parameters.values()
                    ]

                self.assertEqual(
                    call_shape(method),
                    call_shape(getattr(baseclass, name)),
                    f"{subclass.__name__}.{name} must be reviewed after "
                    "the upstream signature change",
                )

    def test_buffer_state_exists_before_base_constructor(self):
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: object())
        )
        server_args = SimpleNamespace(
            hicache_storage_backend="mock",
            hicache_write_policy="write_through",
        )

        def assert_state_initialized(cache, *, params, server_args):
            self.assertEqual(cache.loading_node_to_ack, {})
            self.assertEqual(cache.pending_write_through, deque())
            self.assertEqual(cache.pending_prefetch_query, deque())
            self.assertEqual(cache.pending_prefetch_alloc, deque())
            self.assertEqual(cache.transient_host_nodes_by_reqid, {})

        with (
            patch.object(buffer_cache_module, "MHATokenToKVPool", object),
            patch.object(HiRadixCache, "__init__", new=assert_state_initialized),
        ):
            cache = BufferOnlyHiRadixCache(params, server_args)

        self.assertEqual(cache.host_memory_mode, "buffer_only")

    def test_storage_prefetch_anchor_uses_device_prefix_without_l2_hit(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.root_node = _node([], values=[])
        device_node = _node([1, 2], parent=cache.root_node, values=[10, 11])
        req = SimpleNamespace(
            host_hit_length=0,
            last_host_node=cache.root_node,
            last_node=device_node,
        )

        self.assertIs(cache.resolve_storage_prefetch_anchor(req), device_node)

        device_node.value = None
        self.assertIsNone(cache.resolve_storage_prefetch_anchor(req))

    def test_reset_restarts_storage_threads_once(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.cache_controller = SimpleNamespace(
            write_stream=MagicMock(),
            load_stream=MagicMock(),
            reset=MagicMock(),
        )
        cache.token_to_kv_pool_host = MagicMock()
        cache.ongoing_write_through = {}
        cache.ongoing_load_back = {}
        cache.loading_node_to_ack = {}
        cache.ongoing_prefetch = {}
        cache.ongoing_backup = {}
        cache.pending_write_through = deque()
        cache.pending_prefetch_query = deque()
        cache.pending_prefetch_alloc = deque()
        cache.buffer_query_tokens_occupied = 0
        cache.transient_host_nodes_by_reqid = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.evictable_host_leaves = set()

        with patch.object(RadixCache, "reset") as radix_reset:
            cache.reset()

        cache.cache_controller.reset.assert_called_once_with()
        radix_reset.assert_called_once_with(cache)

    def test_runtime_attach_rejects_non_write_through_policy(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)

        ok, message = cache.attach_storage_backend(
            "mock", hicache_write_policy="write_back"
        )

        self.assertFalse(ok)
        self.assertIn("write_through", message)

    def test_query_first_termination_keeps_current_backend_result(self):
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.mark_terminate()

        self.assertFalse(operation.increment(4))
        self.assertEqual(operation.completed_tokens, 4)

    def test_buffer_write_locks_source_when_it_is_enqueued(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.buffer_only = True
        cache.ongoing_write_through = {}
        cache.pending_write_through = deque()
        cache.inc_lock_ref = MagicMock()

        root = _node([], values=[])
        child = _node([1, 2], parent=root, values=[10, 11])

        self.assertEqual(cache.write_backup(child), 2)

        context = cache.ongoing_write_through[child.id]
        self.assertTrue(context.source_locked)
        cache.inc_lock_ref.assert_called_once_with(child)
        self.assertEqual(list(cache.pending_write_through), [child.id])

    def test_loading_node_is_hidden_from_regular_prefix_match(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2

        root = _node([], values=[])
        child = _node([1, 2, 3, 4], parent=root, values=[10, 11, 12, 13])
        child.device_loading = True
        root.children[child.key.child_key(cache.page_size)] = child

        values, last_node = cache._match_prefix_helper(root, child.key)
        self.assertEqual(values, [])
        self.assertIs(last_node, root)

    def test_unfinished_request_rematches_indices_loaded_from_l3(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
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
        cache.loading_node_to_ack = {child: child.id}
        cache.cache_controller.ack_load_queue.append((0, finish_event, [child.id]))

        req = SimpleNamespace(
            req_pool_idx=0,
            extra_key=None,
            cache_protected_len=4,
            last_node=child,
            prefix_indices=torch.arange(4),
            priority=0,
            get_fill_ids=lambda: array("q", [1, 2, 3, 4]),
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
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
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
            unrelated.id: LoadBackContext(end_node=unrelated, host_nodes=[unrelated]),
        }
        cache.loading_node_to_ack = {
            target: target.id,
            unrelated: unrelated.id,
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
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
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
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache.evictable_size_ = 2
        cache.protected_size_ = 0
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.transient_host_nodes_by_reqid = {}
        cache.eviction_strategy = SimpleNamespace(get_priority=lambda node: node.id)
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
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.transient_host_nodes_by_reqid = {}
        cache.eviction_strategy = SimpleNamespace(get_priority=lambda node: node.id)
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
        self.assertIs(root.children[parent.key.child_key(cache.page_size)], parent)
        self.assertIs(parent.children[child.key.child_key(cache.page_size)], child)

    def test_split_propagates_pending_write_and_storage_state(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
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
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.enable_storage_metrics = False
        cache.ongoing_prefetch = {}
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}
        cache.pending_write_through = deque()
        cache.dec_lock_ref = MagicMock()
        cache._all_reduce_attn_groups = MagicMock()

        root = _node([], values=[])
        first = _node([1, 2], parent=root, values=[10, 11])
        second = _node([3, 4], parent=first, values=[12, 13])
        context = WriteThroughContext(
            write_id=first.id,
            lock_node=second,
            backup_len=4,
            publish_nodes=[first, second],
            host_indices=torch.arange(4),
            storage_operation_id=123,
            inflight_tokens=4,
            host_data_ready=True,
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
        self.assertEqual(first.write_through_pending_id, context.write_id)
        self.assertEqual(second.write_through_pending_id, context.write_id)
        self.assertIn(context.write_id, cache.ongoing_write_through)
        self.assertEqual(context.durable_tokens, 2)
        self.assertIsNone(context.storage_operation_id)
        self.assertTrue(torch.equal(context.host_indices, torch.tensor([2, 3])))
        self.assertTrue(context.host_data_ready)
        self.assertEqual(list(cache.pending_write_through), [context.write_id])
        cache.dec_lock_ref.assert_not_called()
        mem_pool_host.free.assert_called_once()

    def test_split_propagates_inflight_load_back_ownership(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
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
        cache.loading_node_to_ack = {child: child.id}

        prefix = cache._split_node(child.key, child, 2)

        self.assertEqual(context.host_nodes, [prefix, child])
        self.assertEqual(prefix.host_ref_counter, 1)
        self.assertEqual(child.host_ref_counter, 1)
        self.assertEqual(cache.loading_node_to_ack[prefix], child.id)
        self.assertEqual(cache.loading_node_to_ack[child], child.id)

    def test_prefetch_soft_reserve_is_borrowable_without_write_pressure(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
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
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
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

    def test_prefetch_query_waits_for_shared_buffer_budget(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = True
        cache.enable_storage = True
        cache.is_eagle = False
        cache.buffer_query_tokens_occupied = 6
        cache.ongoing_prefetch = {}
        cache.pending_prefetch_query = deque()
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

        controller.prefetch.assert_not_called()
        self.assertEqual(list(cache.pending_prefetch_query), ["req"])
        self.assertIsNone(cache.ongoing_prefetch["req"].operation)

        cache._dispatch_pending_prefetch_queries(1, query_budget=4)

        controller.prefetch.assert_called_once()
        self.assertEqual(len(controller.prefetch.call_args.args[2]), 4)
        self.assertEqual(cache.buffer_query_tokens_occupied, 10)

    def test_prefetch_query_remains_pending_without_budget(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = True
        cache.pending_prefetch_query = deque(["req"])
        cache.ongoing_prefetch = {
            "req": PrefetchContext(
                anchor=_node([], values=[]),
                key=RadixKey(array("q", range(8))),
            )
        }
        cache._get_extra_pools = lambda: {}
        controller = MagicMock()
        controller.mem_pool_host.size = 8
        cache.cache_controller = controller

        cache._dispatch_pending_prefetch_queries(1, query_budget=2)

        controller.prefetch.assert_not_called()
        self.assertEqual(list(cache.pending_prefetch_query), ["req"])

    def test_prefetch_io_timeout_starts_when_io_is_admitted(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.prefetch_timeout_config = SimpleNamespace(
            base=10.0, per_ki_token=0.0, max=10.0
        )
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.set_query_result(["h1", "h2", "h3", "h4"], 8)
        operation.start_time = 0.0
        operation.io_start_time = 95.0

        with patch(
            "sglang.srt.mem_cache.hiradix_cache.time.monotonic",
            return_value=100.0,
        ):
            self.assertFalse(cache._prefetch_timeout_check_linear_func(operation))

        operation.io_start_time = 80.0
        with patch(
            "sglang.srt.mem_cache.hiradix_cache.time.monotonic",
            return_value=100.0,
        ):
            self.assertTrue(cache._prefetch_timeout_check_linear_func(operation))

    def test_prefetch_timeout_keeps_legacy_operation_compatibility(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.prefetch_timeout_config = SimpleNamespace(
            base=10.0, per_ki_token=0.0, max=10.0
        )
        operation = SimpleNamespace(start_time=95.0, hash_value=["h1"])

        with patch(
            "sglang.srt.mem_cache.hiradix_cache.time.monotonic",
            return_value=100.0,
        ):
            self.assertFalse(cache._prefetch_timeout_check_linear_func(operation))

    def test_oversized_buffer_write_dispatches_one_host_sized_chunk(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache.buffer_write_dispatch_limit = 64
        cache.hicache_storage_pass_prefix_keys = False
        cache._all_reduce_attn_groups = MagicMock()
        cache._get_extra_pools = lambda: {}
        cache.evict_host = MagicMock()

        root = _node([], values=[])
        child = _node(list(range(8)), parent=root, values=list(range(10, 18)))
        child.hash_value = ["h1", "h2", "h3", "h4"]
        root.children[child.key.child_key(cache.page_size)] = child
        cache.root_node = root
        context = WriteThroughContext(
            write_id=child.id,
            lock_node=child,
            backup_len=8,
            publish_nodes=[child],
            source_locked=True,
        )
        child.write_through_pending_id = context.write_id
        cache.ongoing_write_through = {context.write_id: context}
        cache.pending_write_through = deque([context.write_id])

        mem_pool_host = MagicMock()
        mem_pool_host.size = 4
        mem_pool_host.available_size.return_value = 4
        controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            write=MagicMock(return_value=torch.arange(4)),
        )
        cache.cache_controller = controller

        cache._dispatch_pending_buffer_writes(1)

        written = controller.write.call_args.kwargs["device_indices"]
        self.assertTrue(torch.equal(written, child.value[:4]))
        self.assertEqual(context.inflight_offset, 0)
        self.assertEqual(context.inflight_tokens, 4)
        self.assertEqual(context.durable_tokens, 0)
        self.assertIn(context.write_id, cache.ongoing_write_through)
        cache._all_reduce_attn_groups.assert_called_once()

    def test_l3_submission_failure_retries_from_retained_host_data(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache.pending_write_through = deque()
        cache.dec_lock_ref = MagicMock()
        cache._get_extra_pools = lambda: {}

        root = _node([], values=[])
        child = _node([1, 2], parent=root, values=[10, 11])
        context = WriteThroughContext(
            write_id=child.id,
            lock_node=child,
            backup_len=2,
            publish_nodes=[child],
            host_indices=torch.arange(2),
            key=child.key,
            hash_value=["h1"],
            inflight_tokens=2,
            source_locked=True,
        )
        child.write_through_pending_id = context.write_id
        cache.ongoing_write_through = {context.write_id: context}

        mem_pool_host = MagicMock()
        controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            write_storage=MagicMock(side_effect=RuntimeError("L3 unavailable")),
        )
        cache.cache_controller = controller

        with patch("sglang.srt.mem_cache.hiradix_cache.logger.exception"):
            cache._finish_write_through_ack(context.write_id, release_lock=True)

        self.assertIn(context.write_id, cache.ongoing_write_through)
        self.assertEqual(list(cache.pending_write_through), [context.write_id])
        self.assertFalse(context.source_locked)
        self.assertTrue(torch.equal(context.host_indices, torch.arange(2)))
        self.assertTrue(context.host_data_ready)
        cache.dec_lock_ref.assert_called_once_with(child)
        mem_pool_host.free.assert_not_called()

    def test_later_write_chunk_carries_full_prefix_metadata(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache._get_extra_pools = lambda: {}
        cache.dec_lock_ref = MagicMock()

        root = _node([], values=[])
        child = _node(list(range(8)), parent=root, values=list(range(10, 18)))
        context = WriteThroughContext(
            write_id=child.id,
            lock_node=child,
            backup_len=8,
            publish_nodes=[child],
            host_indices=torch.arange(4),
            key=child.key,
            hash_value=["h1", "h2", "h3", "h4"],
            prefix_keys=["parent"],
            durable_tokens=4,
            inflight_offset=4,
            inflight_tokens=4,
            source_locked=True,
        )
        cache.ongoing_write_through = {context.write_id: context}
        cache.ongoing_backup = {}
        controller = SimpleNamespace(write_storage=MagicMock(return_value=321))
        cache.cache_controller = controller

        cache._finish_write_through_ack(context.write_id, release_lock=True)

        args = controller.write_storage.call_args.args
        self.assertEqual(list(args[1]), list(range(4, 8)))
        self.assertEqual(args[2], ["h3", "h4"])
        self.assertEqual(args[3], ["parent", "h1", "h2"])
        self.assertIs(cache.ongoing_backup[321], context)
        self.assertFalse(context.source_locked)
        cache.dec_lock_ref.assert_called_once_with(child)

    def test_pending_l3_retry_reuses_host_chunk_without_d2h(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache.buffer_write_dispatch_limit = 64
        cache.pending_write_through = deque()
        cache.ongoing_backup = {}
        cache._all_reduce_attn_groups = MagicMock()
        cache._get_extra_pools = lambda: {}
        cache.evict_host = MagicMock()

        root = _node([], values=[])
        child = _node([1, 2], parent=root, values=[10, 11])
        host_indices = torch.arange(2)
        context = WriteThroughContext(
            write_id=child.id,
            lock_node=child,
            backup_len=2,
            publish_nodes=[child],
            host_indices=host_indices,
            key=child.key,
            hash_value=["h1"],
            inflight_tokens=2,
            host_data_ready=True,
        )
        cache.ongoing_write_through = {context.write_id: context}
        cache.pending_write_through.append(context.write_id)
        mem_pool_host = MagicMock()
        mem_pool_host.size = 2
        mem_pool_host.available_size.return_value = 0
        controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            write=MagicMock(),
            write_storage=MagicMock(return_value=456),
        )
        cache.cache_controller = controller

        cache._dispatch_pending_buffer_writes(1)

        controller.write.assert_not_called()
        controller.write_storage.assert_called_once()
        self.assertIs(controller.write_storage.call_args.args[0], host_indices)
        self.assertEqual(context.storage_operation_id, 456)
        self.assertIs(cache.ongoing_backup[456], context)

    def test_final_l3_ack_clears_pending_write(self):
        cache = BufferOnlyHiRadixCache.__new__(BufferOnlyHiRadixCache)
        cache.page_size = 2
        cache.enable_storage_metrics = False
        cache.ongoing_prefetch = {}
        cache.ongoing_backup = {}
        cache.pending_write_through = deque()
        cache.dec_lock_ref = MagicMock()
        cache._all_reduce_attn_groups = lambda tensor, op: None

        root = _node([], values=[])
        first = _node([1, 2], parent=root, values=[10, 11])
        second = _node([3, 4], parent=first, values=[12, 13])
        first.storage_backed_tokens = 2
        context = WriteThroughContext(
            write_id=first.id,
            lock_node=second,
            backup_len=4,
            publish_nodes=[first, second],
            host_indices=torch.arange(2),
            storage_operation_id=124,
            durable_tokens=2,
            inflight_offset=2,
            inflight_tokens=2,
            host_data_ready=True,
        )
        first.write_through_pending_id = context.write_id
        second.write_through_pending_id = context.write_id
        cache.ongoing_write_through = {context.write_id: context}

        operation = SimpleNamespace(id=124, completed_tokens=2)
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

        self.assertTrue(second.storage_backed)
        self.assertIsNone(first.write_through_pending_id)
        self.assertIsNone(second.write_through_pending_id)
        self.assertNotIn(context.write_id, cache.ongoing_write_through)
        self.assertEqual(list(cache.pending_write_through), [])
        cache.dec_lock_ref.assert_not_called()
        mem_pool_host.free.assert_called_once()


if __name__ == "__main__":
    unittest.main()
