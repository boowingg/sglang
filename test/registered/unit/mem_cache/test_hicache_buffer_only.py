import unittest
from array import array
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
    StorageOperation,
)
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.hiradix_cache import (
    HiRadixCache,
    LoadBackContext,
    PrefetchContext,
    WriteThroughContext,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _node(tokens, *, parent=None, values=None):
    node = TreeNode()
    node.parent = parent
    node.key = RadixKey(array("q", tokens))
    node.value = torch.tensor(values, dtype=torch.int64) if values is not None else None
    return node


class TestHiCacheBufferOnly(unittest.TestCase):
    def test_streaming_config_defaults_and_validation(self):
        cache = HiRadixCache.__new__(HiRadixCache)

        parsed = cache._parse_storage_backend_extra_config(None)

        self.assertEqual(parsed[4:], (1024, 2, 8, 32768))
        invalid_configs = [
            '{"prefetch_stream_window_tokens":0}',
            '{"prefetch_stream_buffer_count":3}',
            '{"prefetch_query_max_ops":false}',
            '{"prefetch_query_max_tokens":0}',
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                cache._parse_storage_backend_extra_config(config)

    def test_query_first_termination_keeps_current_backend_result(self):
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.mark_terminate()

        self.assertFalse(operation.increment(4))
        self.assertEqual(operation.completed_tokens, 4)

    def test_hybrid_prefetch_implements_controller_lifecycle(self):
        operation = HybridPrefetchOperation(
            "req",
            torch.arange(4),
            list(range(4)),
        )

        operation.set_query_result(["h0", "h1"], 4)
        operation.mark_io_started()
        operation.mark_io_done()

        self.assertTrue(operation.query_done)
        self.assertEqual(operation.storage_hit_count, 4)
        self.assertIsNotNone(operation.query_done_time)
        self.assertIsNotNone(operation.io_start_time)
        self.assertTrue(operation.io_done)
        self.assertFalse(operation.stream_to_device)

    def test_hybrid_prefetch_runs_base_page_transfer(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 2
        controller.has_draft = False
        controller.page_get_func = lambda operation, hashes, _indices, _info: (
            operation.increment(len(hashes) * controller.page_size)
        )
        operation = HybridPrefetchOperation(
            "req",
            torch.arange(4),
            list(range(4)),
        )
        operation.set_query_result(["h0", "h1"], 4)

        controller._page_transfer(operation)

        self.assertEqual(operation.completed_tokens, 4)
        self.assertTrue(operation.pool_transfers_done)

    def test_prefetch_reuses_synchronized_query_result(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.prefetch_hit_queue = Queue()
        controller.prefetch_buffer = Queue()

        operation = controller.prefetch_with_query_result(
            "req",
            None,
            list(range(8)),
            ["h0", "h1", "h2"],
            prefix_keys=[],
        )

        self.assertTrue(operation.query_done)
        self.assertEqual(operation.storage_hit_count, 6)
        self.assertEqual(operation.hash_value, ["h0", "h1", "h2"])
        self.assertEqual(list(operation.token_ids), list(range(6)))
        self.assertTrue(controller.prefetch_hit_queue.empty())
        self.assertTrue(controller.prefetch_buffer.empty())

    def test_synchronous_storage_query_applies_prefetch_threshold(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.enable_storage = True
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.is_eagle = False
        cache.buffer_only = False
        cache._get_extra_pools = lambda: {}
        cache._all_reduce_attn_groups = lambda tensor, op: None
        controller = SimpleNamespace(
            prefetch_rate_limited=MagicMock(return_value=False),
            mem_pool_host=SimpleNamespace(
                get_dummy_flat_data_page=MagicMock(return_value=torch.empty(0))
            ),
            _storage_hit_query=MagicMock(return_value=(["h0"], 2)),
        )
        cache.cache_controller = controller

        hashes, hit_tokens = cache.query_storage_hit(
            _node([], values=[]),
            list(range(8)),
        )

        self.assertEqual(hashes, [])
        self.assertEqual(hit_tokens, 0)

    def test_buffer_storage_query_ignores_legacy_l2_capacity_limit(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.enable_storage = True
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.is_eagle = False
        cache.buffer_only = True
        cache._get_extra_pools = lambda: {}
        cache._all_reduce_attn_groups = lambda tensor, op: None
        controller = SimpleNamespace(
            prefetch_rate_limited=MagicMock(return_value=True),
            mem_pool_host=SimpleNamespace(
                get_dummy_flat_data_page=MagicMock(return_value=torch.empty(0))
            ),
            _storage_hit_query=MagicMock(return_value=(["h0", "h1"], 4)),
        )
        cache.cache_controller = controller

        hashes, hit_tokens = cache.query_storage_hit(
            _node([], values=[]),
            list(range(8)),
        )

        self.assertEqual(hashes, ["h0", "h1"])
        self.assertEqual(hit_tokens, 4)
        controller.prefetch_rate_limited.assert_not_called()

    def test_synchronous_storage_query_failure_becomes_tp_safe_miss(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.enable_storage = True
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.is_eagle = False
        cache.buffer_only = True
        cache._get_extra_pools = lambda: {}
        cache._all_reduce_attn_groups = MagicMock()
        controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(
                get_dummy_flat_data_page=MagicMock(return_value=torch.empty(0))
            ),
            _storage_hit_query=MagicMock(side_effect=RuntimeError("L3 unavailable")),
        )
        cache.cache_controller = controller

        with self.assertLogs(level="ERROR"):
            hashes, hit_tokens = cache.query_storage_hit(
                _node([], values=[]),
                list(range(8)),
            )

        self.assertEqual(hashes, [])
        self.assertEqual(hit_tokens, 0)
        self.assertEqual(cache._all_reduce_attn_groups.call_count, 2)

    def test_synchronous_storage_query_yields_to_l3_write_pressure(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.enable_storage = True
        cache.buffer_only = True
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.is_eagle = False
        cache.buffer_write_dispatch_limit = 1
        cache.ongoing_write_through = {}
        backup_queue = Queue()
        backup_queue.put(object())
        controller = SimpleNamespace(
            backup_queue=backup_queue,
            _storage_hit_query=MagicMock(),
        )
        cache.cache_controller = controller
        cache._all_reduce_attn_groups = lambda tensor, op: None

        hashes, hit_tokens = cache.query_storage_hit(
            _node([], values=[]),
            list(range(8)),
        )

        self.assertEqual(hashes, [])
        self.assertEqual(hit_tokens, 0)
        controller._storage_hit_query.assert_not_called()

    def test_synchronous_storage_query_obeys_remote_write_pressure(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.enable_storage = True
        cache.buffer_only = True
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.is_eagle = False
        cache.ongoing_write_through = {}
        cache._get_extra_pools = lambda: {}

        def emulate_remote_pressure(tensor, _op):
            if tensor.numel() == 2:
                tensor[0] = 1

        cache._all_reduce_attn_groups = emulate_remote_pressure
        controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(
                get_dummy_flat_data_page=MagicMock(return_value=torch.empty(0))
            ),
            _storage_hit_query=MagicMock(),
        )
        cache.cache_controller = controller

        hashes, hit_tokens = cache.query_storage_hit(
            _node([], values=[]),
            list(range(8)),
        )

        self.assertEqual(hashes, [])
        self.assertEqual(hit_tokens, 0)
        controller._storage_hit_query.assert_not_called()

    def test_streaming_prefetch_reuses_bounded_host_windows(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.has_draft = False

        def page_get(operation, hashes, host_indices, _extra_info):
            self.assertEqual(len(host_indices), len(hashes) * 2)
            operation.increment(len(hashes) * 2)

        controller.page_get_func = page_get
        events = [MagicMock() for _ in range(3)]
        controller._enqueue_prefetch_chunk_load = MagicMock(
            side_effect=[SimpleNamespace(finish_event=event) for event in events]
        )

        operation = PrefetchOperation("req", None, list(range(10)))
        operation.set_query_result([f"h{i}" for i in range(5)], 10)
        device_ready_event = MagicMock()
        operation.configure_device_streaming(
            torch.arange(8),
            torch.arange(100, 110),
            window_tokens=4,
            device_ready_event=device_ready_event,
        )
        next_storage_operation_id = StorageOperation.counter

        controller._page_transfer_streaming(operation)

        self.assertEqual(operation.completed_tokens, 10)
        self.assertEqual(StorageOperation.counter, next_storage_operation_id)
        calls = controller._enqueue_prefetch_chunk_load.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertTrue(torch.equal(calls[0].args[0], torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.equal(calls[1].args[0], torch.tensor([4, 5, 6, 7])))
        self.assertTrue(torch.equal(calls[2].args[0], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(calls[2].args[1], torch.tensor([108, 109])))
        self.assertIs(calls[0].args[2], device_ready_event)
        self.assertIsNone(calls[1].args[2])
        self.assertIsNone(calls[2].args[2])
        for event in events:
            event.synchronize.assert_called_once()

    def test_direct_stream_waits_before_materializing_device_indices(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.io_backend = "direct"
        controller.layer_num = 0
        controller.has_draft = False
        controller.prefetch_load_stream = MagicMock()
        controller.mem_pool_host = MagicMock()
        controller.mem_pool_device = MagicMock()
        ready_event = MagicMock()
        order = MagicMock()
        ready_event.synchronize.side_effect = order.ready
        host_indices = torch.arange(4)
        device_indices = torch.arange(100, 104)

        def move_indices(host, device):
            order.move()
            return host, device

        controller.move_indices = MagicMock(side_effect=move_indices)
        start_event = MagicMock()
        finish_event = MagicMock()
        stream_context = MagicMock()
        stream_context.__enter__.return_value = None

        with (
            patch(
                "sglang.srt.managers.cache_controller.device_module.Event",
                side_effect=[start_event, finish_event],
            ),
            patch(
                "sglang.srt.managers.cache_controller.device_module.stream",
                return_value=stream_context,
            ),
        ):
            controller._enqueue_prefetch_chunk_load(
                host_indices,
                device_indices,
                ready_event,
            )

        self.assertEqual(order.mock_calls, [call.ready(), call.move()])
        ready_event.wait.assert_not_called()

    def test_streaming_prefetch_fills_window_across_backend_batches(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.has_draft = False
        backend_batches = []

        def page_get(operation, hashes, _host_indices, _extra_info):
            backend_batches.append(list(hashes))
            operation.increment(len(hashes) * 2)

        controller.page_get_func = page_get
        events = [MagicMock(), MagicMock()]
        controller._enqueue_prefetch_chunk_load = MagicMock(
            side_effect=[SimpleNamespace(finish_event=event) for event in events]
        )

        operation = PrefetchOperation("req", None, list(range(12)))
        operation.set_query_result([f"h{i}" for i in range(6)], 12)
        operation.configure_device_streaming(
            torch.arange(16), torch.arange(100, 112), window_tokens=8
        )

        with patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 2):
            controller._page_transfer_streaming(operation)

        self.assertEqual(
            backend_batches,
            [["h0", "h1"], ["h2", "h3"], ["h4", "h5"]],
        )
        calls = controller._enqueue_prefetch_chunk_load.call_args_list
        self.assertEqual([len(call.args[0]) for call in calls], [8, 4])
        self.assertEqual(operation.completed_tokens, 12)

    def test_streaming_prefetch_advances_empty_prefix_metadata(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.has_draft = False
        seen_prefixes = []

        def page_get(operation, hashes, _host_indices, extra_info):
            seen_prefixes.append(list(extra_info.prefix_keys))
            operation.increment(len(hashes) * 2)

        controller.page_get_func = page_get
        finish_event = MagicMock()
        controller._enqueue_prefetch_chunk_load = MagicMock(
            return_value=SimpleNamespace(finish_event=finish_event)
        )

        operation = PrefetchOperation("req", None, list(range(6)), prefix_keys=[])
        operation.set_query_result(["h0", "h1", "h2"], 6)
        operation.configure_device_streaming(
            torch.arange(6), torch.arange(100, 106), window_tokens=6
        )

        with patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1):
            controller._page_transfer_streaming(operation)

        self.assertEqual(seen_prefixes, [[], ["h0"], ["h0", "h1"]])

    def test_streaming_prefetch_keeps_only_contiguous_completed_chunks(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.has_draft = False
        page_get_calls = 0

        def page_get(operation, hashes, _host_indices, _extra_info):
            nonlocal page_get_calls
            page_get_calls += 1
            completed = len(hashes) * 2 if page_get_calls == 1 else 2
            operation.increment(completed)

        controller.page_get_func = page_get
        first_event = MagicMock()
        controller._enqueue_prefetch_chunk_load = MagicMock(
            return_value=SimpleNamespace(finish_event=first_event)
        )

        operation = PrefetchOperation("req", None, list(range(8)))
        operation.set_query_result([f"h{i}" for i in range(4)], 8)
        operation.configure_device_streaming(
            torch.arange(8), torch.arange(100, 108), window_tokens=4
        )

        controller._page_transfer_streaming(operation)

        self.assertEqual(page_get_calls, 2)
        self.assertEqual(operation.completed_tokens, 4)
        self.assertTrue(operation.is_terminated())
        controller._enqueue_prefetch_chunk_load.assert_called_once()
        first_event.synchronize.assert_called_once()

    def test_streaming_prefetch_preserves_batches_before_backend_exception(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.has_draft = False
        controller.prefetch_load_stream = MagicMock()
        page_get_calls = 0

        def page_get(operation, hashes, _host_indices, _extra_info):
            nonlocal page_get_calls
            page_get_calls += 1
            if page_get_calls == 2:
                raise RuntimeError("backend read failed")
            operation.increment(len(hashes) * 2)

        controller.page_get_func = page_get
        finish_event = MagicMock()
        controller._enqueue_prefetch_chunk_load = MagicMock(
            return_value=SimpleNamespace(finish_event=finish_event)
        )

        operation = PrefetchOperation("req", None, list(range(8)))
        operation.set_query_result([f"h{i}" for i in range(4)], 8)
        operation.configure_device_streaming(
            torch.arange(8), torch.arange(100, 108), window_tokens=8
        )

        with (
            patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 2),
            self.assertRaisesRegex(RuntimeError, "backend read failed"),
        ):
            controller._page_transfer_streaming(operation)

        self.assertEqual(operation.completed_tokens, 4)
        controller._enqueue_prefetch_chunk_load.assert_called_once()
        finish_event.synchronize.assert_called_once()
        controller.prefetch_load_stream.synchronize.assert_called_once()

    def test_streaming_prefetch_does_not_publish_past_h2d_failure(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 2
        controller.has_draft = False
        controller.prefetch_load_stream = MagicMock()

        def page_get(operation, hashes, _host_indices, _extra_info):
            operation.increment(len(hashes) * 2)

        controller.page_get_func = page_get
        events = [MagicMock() for _ in range(3)]
        events[1].synchronize.side_effect = RuntimeError("H2D failed")
        controller._enqueue_prefetch_chunk_load = MagicMock(
            side_effect=[SimpleNamespace(finish_event=event) for event in events]
        )

        operation = PrefetchOperation("req", None, list(range(10)))
        operation.set_query_result([f"h{i}" for i in range(5)], 10)
        operation.configure_device_streaming(
            torch.arange(8), torch.arange(100, 110), window_tokens=4
        )

        with self.assertRaisesRegex(RuntimeError, "H2D failed"):
            controller._page_transfer_streaming(operation)

        self.assertEqual(operation.completed_tokens, 4)
        events[0].synchronize.assert_called_once()
        events[1].synchronize.assert_called_once()
        events[2].synchronize.assert_called_once()
        controller.prefetch_load_stream.synchronize.assert_called_once()

    def test_streamed_device_publish_frees_duplicate_prefix_indices(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache.evictable_size_ = 2
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.transient_host_nodes_by_reqid = {}
        cache._record_store_event = MagicMock()
        device_allocator = MagicMock()
        cache.cache_controller = SimpleNamespace(
            mem_pool_device_allocator=device_allocator
        )

        root = _node([], values=[])
        existing = _node([1, 2], parent=root, values=[10, 11])
        existing.storage_backed_tokens = 2
        existing.hash_value = ["h0"]
        root.children[existing.key.child_key(cache.page_size)] = existing
        cache.root_node = root
        cache._update_leaf_status(existing)

        matched = cache._insert_helper_device_prefetch(
            root,
            RadixKey(array("q", [1, 2, 3, 4, 5, 6])),
            torch.arange(100, 106),
            ["h0", "h1", "h2"],
        )

        self.assertEqual(matched, 2)
        device_allocator.free.assert_called_once()
        self.assertTrue(
            torch.equal(
                device_allocator.free.call_args.args[0], torch.tensor([100, 101])
            )
        )
        suffix = existing.children[RadixKey(array("q", [3, 4])).child_key(2)]
        self.assertTrue(torch.equal(suffix.value, torch.tensor([102, 103, 104, 105])))
        self.assertTrue(suffix.storage_backed)
        self.assertEqual(cache.evictable_size_, 6)

    def test_streamed_device_publish_rejects_index_length_mismatch(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        root = _node([], values=[])

        with self.assertRaisesRegex(RuntimeError, "key/device length mismatch"):
            cache._insert_helper_device_prefetch(
                root,
                RadixKey(array("q", [1, 2, 3, 4])),
                torch.arange(3),
                ["h0", "h1"],
            )

    def test_streamed_device_publish_rejects_active_load_back(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        root = _node([], values=[])
        loading = _node([1, 2], parent=root, values=[10, 11])
        loading.device_loading = True
        root.children[loading.key.child_key(cache.page_size)] = loading

        with self.assertRaisesRegex(RuntimeError, "active load-back"):
            cache._insert_helper_device_prefetch(
                root,
                loading.key,
                torch.arange(100, 102),
                ["h0"],
            )

    def test_buffer_write_locks_source_when_it_is_enqueued(self):
        cache = HiRadixCache.__new__(HiRadixCache)
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
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2

        root = _node([], values=[])
        child = _node([1, 2, 3, 4], parent=root, values=[10, 11, 12, 13])
        child.device_loading = True
        root.children[child.key.child_key(cache.page_size)] = child

        values, last_node = cache._match_prefix_helper(root, child.key)
        self.assertEqual(values, [])
        self.assertIs(last_node, root)

    def test_local_query_done_does_not_open_tp_allocation_gate(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.buffer_only = True
        operation = PrefetchOperation("req", None, list(range(4)))
        operation.set_query_result(["h0", "h1"], 4)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(4))),
            operation=operation,
        )
        cache.ongoing_prefetch = {"req": context}
        cache._allocate_buffer_prefetch = MagicMock(return_value=True)

        self.assertFalse(cache.check_prefetch_progress("req"))
        cache._allocate_buffer_prefetch.assert_not_called()

    def test_ready_query_does_not_poll_allocation_while_stream_is_active(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.buffer_only = True
        cache.prefetch_threshold = 4
        cache.streaming_prefetch_active = {"active"}
        operation = PrefetchOperation("req", None, list(range(4)))
        operation.set_query_result(["h0", "h1"], 4)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(4))),
            operation=operation,
            query_ready=True,
        )
        cache.ongoing_prefetch = {"req": context}
        cache._trim_buffer_prefetch_against_device = MagicMock(return_value=True)
        cache._allocate_buffer_prefetch = MagicMock(return_value=True)

        self.assertFalse(cache.check_prefetch_progress("req"))

        cache._trim_buffer_prefetch_against_device.assert_not_called()
        cache._allocate_buffer_prefetch.assert_not_called()

    def test_ready_query_waits_without_cancelling_during_write_pressure(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.buffer_only = True
        cache.prefetch_threshold = 4
        cache.streaming_prefetch_active = set()
        cache.buffer_prefetch_paused_for_write = True
        operation = PrefetchOperation("req", None, list(range(4)))
        operation.set_query_result(["h0", "h1"], 4)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(4))),
            operation=operation,
            query_ready=True,
        )
        cache.ongoing_prefetch = {"req": context}
        cache._trim_buffer_prefetch_against_device = MagicMock(return_value=True)
        cache._allocate_buffer_prefetch = MagicMock(return_value=True)

        self.assertFalse(cache.check_prefetch_progress("req"))

        self.assertFalse(operation.is_terminated())
        cache._trim_buffer_prefetch_against_device.assert_not_called()
        cache._allocate_buffer_prefetch.assert_not_called()

    def test_new_write_pressure_pauses_reads_in_the_same_drain_round(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.buffer_only = True
        cache.buffer_write_dispatch_limit = 64
        cache.buffer_write_retry_ids = set()
        cache.aborted_prefetch_req_ids = set()
        cache.pending_write_through = deque([1])
        cache.pending_prefetch_query = deque(["req"])
        cache.buffer_prefetch_paused_for_write = False
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache._drain_storage_control_queues_impl = MagicMock()
        cache._reap_aborted_prefetches = MagicMock()
        cache._dispatch_pending_prefetch_queries = MagicMock()

        def dispatch_write(_pending_count):
            cache.buffer_write_retry_ids.add(1)

        cache._dispatch_pending_buffer_writes = MagicMock(side_effect=dispatch_write)
        cache.cache_controller = SimpleNamespace(
            prefetch_revoke_queue=Queue(),
            prefetch_hit_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            backup_queue=Queue(),
        )

        cache.drain_storage_control_queues()

        self.assertTrue(cache.buffer_prefetch_paused_for_write)
        cache._dispatch_pending_prefetch_queries.assert_called_once_with(
            1,
            pause_for_write=True,
        )

    def test_synchronized_hit_drain_opens_gate_and_keeps_ready_slot_bounded(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.prefetch_query_ops_occupied = 1
        cache.prefetch_query_tokens_occupied = 8
        cache.ongoing_backup = {}
        cache.ongoing_write_through = {}
        cache.pending_write_through = deque()
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.set_query_result(["h0", "h1", "h2", "h3"], 8)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(8))),
            operation=operation,
            query_dispatched_tokens=8,
        )
        cache.ongoing_prefetch = {"req": context}
        controller = SimpleNamespace(
            prefetch_revoke_queue=Queue(),
            prefetch_hit_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
        )
        controller.prefetch_hit_queue.put(operation)
        cache.cache_controller = controller

        cache._drain_storage_control_queues_impl(
            n_revoke=0,
            n_hit=1,
            n_backup=0,
            n_release=0,
            log_metrics=False,
        )

        self.assertTrue(context.query_ready)
        self.assertEqual(context.query_dispatched_tokens, 8)
        self.assertEqual(cache.prefetch_query_ops_occupied, 1)
        self.assertEqual(cache.prefetch_query_tokens_occupied, 8)

    def test_aborted_query_waits_for_synchronized_control_queue(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        operation = PrefetchOperation("req", None, list(range(4)))
        operation.set_query_result(["h0", "h1"], 4)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(4))),
            operation=operation,
            aborted=True,
        )
        cache.ongoing_prefetch = {"req": context}
        cache._release_prefetch_context = MagicMock()

        cache._reap_aborted_prefetches()

        self.assertIn("req", cache.ongoing_prefetch)
        cache._release_prefetch_context.assert_not_called()

    def test_aborted_prequeried_request_releases_without_async_handoff(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        operation = PrefetchOperation("req", None, list(range(4)))
        operation.set_query_result(["h0", "h1"], 4)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(4))),
            operation=operation,
            aborted=True,
            query_ready=True,
        )
        cache.ongoing_prefetch = {"req": context}
        cache.aborted_prefetch_req_ids = {"req"}
        cache._release_prefetch_context = MagicMock()

        cache._reap_aborted_prefetches()

        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(cache.aborted_prefetch_req_ids, set())
        cache._release_prefetch_context.assert_called_once_with(
            context, free_completed=True
        )

    def test_aborted_stream_releases_only_after_tp_io_completion(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache._release_prefetch_context = MagicMock()
        operation = PrefetchOperation("req", None, list(range(4)))
        operation.host_indices = torch.arange(2)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(4))),
            operation=operation,
            aborted=True,
        )
        cache.ongoing_prefetch = {"req": context}

        cache._reap_aborted_prefetches()
        self.assertIn("req", cache.ongoing_prefetch)

        operation.mark_io_done()
        cache._reap_aborted_prefetches()

        self.assertNotIn("req", cache.ongoing_prefetch)
        cache._release_prefetch_context.assert_called_once_with(
            context, free_completed=True
        )

    def test_prefetch_admission_trims_prefix_loaded_by_another_request(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 2
        cache.inc_lock_ref = MagicMock()
        cache.dec_lock_ref = MagicMock()

        root = _node([], values=[])
        child = _node([1, 2, 3, 4], parent=root, values=[10, 11, 12, 13])
        root.children[child.key.child_key(cache.page_size)] = child
        operation = PrefetchOperation("req", None, list(range(1, 7)))
        operation.set_query_result(["h0", "h1", "h2"], 6)
        context = PrefetchContext(
            anchor=root,
            key=RadixKey(array("q", range(1, 7))),
            operation=operation,
            anchor_device_locked=True,
            prefix_keys=[],
        )

        self.assertTrue(cache._trim_buffer_prefetch_against_device(context))

        self.assertIs(context.anchor, child)
        self.assertEqual(list(context.key), [5, 6])
        self.assertEqual(operation.hash_value, ["h2"])
        self.assertEqual(operation.storage_hit_count, 2)
        self.assertEqual(operation.prefix_keys, ["h0", "h1"])
        cache.inc_lock_ref.assert_called_once_with(child)
        cache.dec_lock_ref.assert_called_once_with(root)

    def test_device_only_match_uses_device_node_as_best_match(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.disable = False
        cache.page_size = 2
        cache.is_eagle = False
        cache._empty_match_result = SimpleNamespace(
            device_indices=torch.empty(0, dtype=torch.int64)
        )

        root = _node([], values=[])
        root.host_value = []
        child = _node([1, 2, 3, 4], parent=root, values=[10, 11, 12, 13])
        root.children[child.key.child_key(cache.page_size)] = child
        cache.root_node = root

        result = cache.match_prefix(MatchPrefixParams(key=child.key))

        self.assertIs(result.last_device_node, child)
        self.assertIs(result.last_host_node, root)
        self.assertIs(result.best_match_node, child)
        self.assertEqual(result.host_hit_length, 0)

    def test_decode_prefetch_anchors_at_device_prefix_in_buffer_mode(self):
        mixin = DecodeHiCachePreallocMixin()
        mixin.scheduler = SimpleNamespace(enable_decode_hicache=True)

        root = _node([], values=[])
        root.host_value = []
        child = _node([1, 2], parent=root, values=[10, 11])
        child.hash_value = ["h0"]
        tree_cache = SimpleNamespace(
            buffer_only=True,
            root_node=root,
            hicache_storage_pass_prefix_keys=False,
            query_storage_hit=MagicMock(return_value=(["h1", "h2"], 4)),
        )
        mixin.tree_cache = tree_cache
        result = SimpleNamespace(
            device_indices=child.value,
            host_hit_length=0,
            last_device_node=child,
            last_host_node=root,
        )
        req = SimpleNamespace(origin_input_ids=array("q", [1, 2, 3, 4, 5, 6]))

        match = mixin._build_decode_prefix_match(req, result)

        self.assertIs(match.last_host_node, child)
        self.assertEqual(match.l3_storage_hit_length, 4)
        self.assertEqual(match.l3_hash_values, ["h1", "h2"])
        tree_cache.query_storage_hit.assert_called_once()
        self.assertIs(tree_cache.query_storage_hit.call_args.args[0], child)

    def test_decode_stream_replaces_transient_l2_gap_with_l3_prefix(self):
        mixin = DecodeHiCachePreallocMixin()
        mixin.scheduler = SimpleNamespace(enable_decode_hicache=True)

        root = _node([], values=[])
        root.host_value = []
        device_node = _node([1, 2], parent=root, values=[10, 11])
        device_node.hash_value = ["h0"]
        host_node = _node([3, 4], parent=device_node)
        host_node.host_value = torch.tensor([20, 21])
        tree_cache = SimpleNamespace(
            buffer_only=True,
            root_node=root,
            hicache_storage_pass_prefix_keys=False,
            query_storage_hit=MagicMock(return_value=(["h1", "h2", "h3"], 6)),
        )
        mixin.tree_cache = tree_cache
        result = SimpleNamespace(
            device_indices=device_node.value,
            host_hit_length=2,
            last_device_node=device_node,
            last_host_node=host_node,
        )
        req = SimpleNamespace(origin_input_ids=array("q", [1, 2, 3, 4, 5, 6, 7, 8]))

        match = mixin._build_decode_prefix_match(req, result)

        self.assertEqual(match.l2_host_hit_length, 0)
        self.assertEqual(match.l3_storage_hit_length, 6)
        self.assertIs(match.last_host_node, device_node)
        queried_suffix = tree_cache.query_storage_hit.call_args.args[1]
        self.assertEqual(list(queried_suffix), [3, 4, 5, 6, 7, 8])

    def test_decode_nonbuffer_degrade_preserves_existing_l2_hit(self):
        mixin = DecodeHiCachePreallocMixin()
        mixin.scheduler = SimpleNamespace(enable_decode_hicache=True)
        root = _node([], values=[])
        device_node = _node([1, 2], parent=root, values=[10, 11])
        host_node = _node([3, 4], parent=device_node)
        host_node.host_value = torch.arange(2)
        host_node.hash_value = ["h1"]
        mixin.tree_cache = SimpleNamespace(
            buffer_only=False,
            root_node=root,
            hicache_storage_pass_prefix_keys=False,
            query_storage_hit=MagicMock(return_value=(["h2", "h3"], 4)),
        )
        result = SimpleNamespace(
            device_indices=device_node.value,
            host_hit_length=2,
            last_device_node=device_node,
            last_host_node=host_node,
        )
        req = SimpleNamespace(origin_input_ids=array("q", range(1, 9)))

        match = mixin._build_decode_prefix_match(req, result)
        mixin._degrade_hicache_prefetch(match)

        self.assertEqual(match.l2_host_hit_length, 2)
        self.assertEqual(match.l3_storage_hit_length, 0)

    def test_decode_prefetch_passes_prequeried_hashes(self):
        mixin = DecodeHiCachePreallocMixin()
        root = _node([], values=[])
        root.host_value = []
        node = _node([1, 2], parent=root, values=[10, 11])
        node.hash_value = ["h0"]
        ongoing_prefetch = {}

        def prefetch(req_id, *_args, **_kwargs):
            ongoing_prefetch[req_id] = object()

        mixin.tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=False,
            ongoing_prefetch=ongoing_prefetch,
            prefetch_from_storage=MagicMock(side_effect=prefetch),
        )
        prefix_match = SimpleNamespace(
            l1_prefix_len=2,
            l2_host_hit_length=0,
            l3_storage_hit_length=4,
            l3_hash_values=["h1", "h2"],
            last_host_node=node,
            prefetch_registered=False,
        )
        req = SimpleNamespace(
            rid="req",
            origin_input_ids=array("q", [1, 2, 3, 4, 5, 6]),
        )

        mixin._start_hicache_prefetch(req, prefix_match)

        call = mixin.tree_cache.prefetch_from_storage.call_args
        self.assertEqual(list(call.args[2]), [3, 4, 5, 6])
        self.assertEqual(call.kwargs["known_hash_values"], ["h1", "h2"])
        self.assertTrue(call.kwargs["require_full_completion"])
        self.assertTrue(prefix_match.prefetch_registered)

    def test_decode_prefetch_registration_failure_degrades_to_recompute(self):
        mixin = DecodeHiCachePreallocMixin()
        node = _node([], values=[])
        mixin.tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=False,
            ongoing_prefetch={},
            prefetch_from_storage=MagicMock(),
        )
        prefix_match = SimpleNamespace(
            l1_prefix_len=2,
            l2_host_hit_length=0,
            l3_storage_hit_length=4,
            l3_hash_values=["h1", "h2"],
            last_host_node=node,
            prefetch_registered=False,
            l2_fallback_hit_length=2,
        )
        req = SimpleNamespace(
            rid="req",
            origin_input_ids=array("q", [1, 2, 3, 4, 5, 6]),
        )

        mixin._start_hicache_prefetch(req, prefix_match)

        self.assertFalse(prefix_match.prefetch_registered)
        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertIsNone(prefix_match.l3_hash_values)
        self.assertEqual(prefix_match.l2_host_hit_length, 2)

    def test_decode_prefetch_does_not_reuse_stale_request_context(self):
        mixin = DecodeHiCachePreallocMixin()
        node = _node([], values=[])
        stale_context = object()
        mixin.tree_cache = SimpleNamespace(
            hicache_storage_pass_prefix_keys=False,
            ongoing_prefetch={"req": stale_context},
            prefetch_from_storage=MagicMock(),
            release_aborted_request=MagicMock(),
        )
        prefix_match = SimpleNamespace(
            l1_prefix_len=2,
            l2_host_hit_length=0,
            l3_storage_hit_length=4,
            l3_hash_values=["h1", "h2"],
            last_host_node=node,
            prefetch_registered=False,
            l2_fallback_hit_length=2,
        )
        req = SimpleNamespace(
            rid="req",
            origin_input_ids=array("q", [1, 2, 3, 4, 5, 6]),
        )

        mixin._start_hicache_prefetch(req, prefix_match)

        self.assertFalse(prefix_match.prefetch_registered)
        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertIs(mixin.tree_cache.ongoing_prefetch["req"], stale_context)
        mixin.tree_cache.release_aborted_request.assert_called_once_with("req")

    def test_decode_stream_waits_behind_another_active_stream(self):
        mixin = DecodeHiCacheTransferMixin()
        tree_cache = SimpleNamespace(
            buffer_only=True,
            streaming_prefetch_active={"active"},
            check_prefetch_progress=MagicMock(return_value=False),
            pop_prefetch_loaded_tokens=MagicMock(),
        )
        mixin.tree_cache = tree_cache
        decode_req = SimpleNamespace(
            prefix_match=SimpleNamespace(l3_storage_hit_length=4),
            req=SimpleNamespace(rid="waiting"),
        )

        self.assertFalse(mixin._try_hicache_queue_load_back(decode_req))
        tree_cache.check_prefetch_progress.assert_called_once_with(
            "waiting",
            allow_device_allocation=False,
            device_admission_tokens=0,
        )
        tree_cache.pop_prefetch_loaded_tokens.assert_not_called()

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

    def test_loading_prefix_sync_requires_every_relevant_ack(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2

        root = _node([], values=[])
        first = _node([1, 2], parent=root, values=[10, 11])
        second = _node([3, 4], parent=first, values=[12, 13])
        for node in (first, second):
            node.host_value = torch.arange(2)
            node.device_loading = True
        root.children[first.key.child_key(cache.page_size)] = first
        first.children[second.key.child_key(cache.page_size)] = second
        cache.root_node = root
        cache.loading_node_to_ack = {first: first.id, second: second.id}
        first_event = MagicMock()
        cache.cache_controller = SimpleNamespace(
            ack_load_queue=[(0, first_event, [first.id])]
        )

        synchronized = cache._synchronize_loading_prefix(
            RadixKey(array("q", [1, 2, 3, 4]))
        )

        self.assertFalse(synchronized)
        self.assertEqual(len(cache.cache_controller.ack_load_queue), 1)
        first_event.synchronize.assert_not_called()

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
        cache = HiRadixCache.__new__(HiRadixCache)
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
        cache.loading_node_to_ack = {child: child.id}

        prefix = cache._split_node(child.key, child, 2)

        self.assertEqual(context.host_nodes, [prefix, child])
        self.assertEqual(prefix.host_ref_counter, 1)
        self.assertEqual(child.host_ref_counter, 1)
        self.assertEqual(cache.loading_node_to_ack[prefix], child.id)
        self.assertEqual(cache.loading_node_to_ack[child], child.id)

    def test_prefetch_soft_reserve_is_borrowable_without_write_pressure(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_write_reserve_ratio = 0.2
        cache.prefetch_stream_window_tokens = 100
        cache.prefetch_stream_buffer_count = 1
        cache.streaming_prefetch_active = set()
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}
        cache.prefetch_query_ops_occupied = 1
        cache.prefetch_query_tokens_occupied = 100
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache.evictable_size = lambda: 0
        cache.evict = MagicMock()

        mem_pool_host = MagicMock()
        mem_pool_host.size = 100
        mem_pool_host.available_size.return_value = 100
        mem_pool_host.alloc.side_effect = lambda size: torch.arange(size)
        device_allocator = MagicMock()
        device_allocator.available_size.return_value = 1000
        device_allocator.alloc.side_effect = lambda size: torch.arange(size) + 1000
        controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            mem_pool_device_allocator=device_allocator,
            prefetch_tokens_occupied=0,
            prefetch_buffer=Queue(),
            record_device_ready_event=MagicMock(return_value=MagicMock()),
        )
        cache.cache_controller = controller

        operation = PrefetchOperation("req", None, list(range(100)))
        operation.set_query_result([f"h{i}" for i in range(50)], 100)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(100))),
            operation=operation,
            query_dispatched_tokens=100,
        )

        self.assertTrue(cache._allocate_buffer_prefetch(context))
        self.assertEqual(context.allocated_tokens, 100)
        self.assertEqual(len(context.device_indices), 100)
        self.assertEqual(cache.prefetch_query_ops_occupied, 0)
        self.assertEqual(cache.prefetch_query_tokens_occupied, 0)

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
        cache.streaming_prefetch_active.clear()

        self.assertTrue(cache._allocate_buffer_prefetch(context2))
        self.assertEqual(context2.allocated_tokens, 80)
        self.assertEqual(len(context2.device_indices), 100)

    def test_stream_window_may_be_smaller_than_prefetch_threshold(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 8
        cache.buffer_write_reserve_ratio = 0.2
        cache.prefetch_stream_window_tokens = 4
        cache.prefetch_stream_buffer_count = 1
        cache.streaming_prefetch_active = set()
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache.evictable_size = lambda: 0

        mem_pool_host = MagicMock()
        mem_pool_host.size = 4
        mem_pool_host.available_size.return_value = 4
        mem_pool_host.alloc.return_value = torch.arange(4)
        device_allocator = MagicMock()
        device_allocator.available_size.return_value = 32
        device_allocator.alloc.return_value = torch.arange(100, 108)
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            mem_pool_device_allocator=device_allocator,
            prefetch_tokens_occupied=0,
            prefetch_buffer=Queue(),
            record_device_ready_event=MagicMock(return_value=MagicMock()),
        )
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.set_query_result(["h0", "h1", "h2", "h3"], 8)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(8))),
            operation=operation,
        )

        self.assertTrue(cache._allocate_buffer_prefetch(context))
        self.assertEqual(context.allocated_tokens, 4)
        self.assertEqual(len(context.device_indices), 8)

    def test_remote_admission_rejection_prevents_local_stream_allocation(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_write_reserve_ratio = 0.2
        cache.prefetch_stream_window_tokens = 4
        cache.prefetch_stream_buffer_count = 1
        cache.streaming_prefetch_active = set()
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}
        cache.evictable_size = lambda: 0

        def emulate_remote_rejection(tensor, _op):
            if tensor.numel() == 6:
                tensor[-1] = 0

        cache._all_reduce_attn_groups = emulate_remote_rejection
        mem_pool_host = MagicMock()
        mem_pool_host.size = 8
        mem_pool_host.available_size.return_value = 8
        device_allocator = MagicMock()
        device_allocator.available_size.return_value = 32
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            mem_pool_device_allocator=device_allocator,
        )
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.set_query_result(["h0", "h1", "h2", "h3"], 8)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(8))),
            operation=operation,
        )

        self.assertFalse(cache._allocate_buffer_prefetch(context))

        mem_pool_host.alloc.assert_not_called()
        device_allocator.alloc.assert_not_called()

    def test_required_stream_waits_for_its_full_l1_destination(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_write_reserve_ratio = 0.2
        cache.prefetch_stream_window_tokens = 4
        cache.prefetch_stream_buffer_count = 1
        cache.streaming_prefetch_active = set()
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache.evictable_size = lambda: 0
        mem_pool_host = MagicMock()
        mem_pool_host.size = 8
        mem_pool_host.available_size.return_value = 8
        device_allocator = MagicMock()
        device_allocator.available_size.return_value = 6
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            mem_pool_device_allocator=device_allocator,
        )
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.set_query_result(["h0", "h1", "h2", "h3"], 8)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(8))),
            operation=operation,
            require_full_completion=True,
        )

        self.assertFalse(cache._allocate_buffer_prefetch(context))

        mem_pool_host.alloc.assert_not_called()
        device_allocator.alloc.assert_not_called()

    def test_impossible_stream_capacity_bypasses_synchronous_storage_query(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.prefetch_stream_window_tokens = 4
        cache.buffer_only = True
        cache.enable_storage = True
        cache.is_eagle = False
        cache.ongoing_write_through = {}
        cache._get_extra_pools = lambda: {}
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(
                size=2,
                get_dummy_flat_data_page=MagicMock(return_value=torch.empty(0)),
            ),
            mem_pool_device_allocator=SimpleNamespace(size=100),
            _storage_hit_query=MagicMock(),
        )

        hashes, hit_tokens = cache.query_storage_hit(
            _node([], values=[]),
            list(range(8)),
        )

        self.assertEqual(hashes, [])
        self.assertEqual(hit_tokens, 0)
        cache.cache_controller._storage_hit_query.assert_not_called()

    def test_impossible_stream_capacity_bypasses_async_query_registration(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.prefetch_stream_window_tokens = 4
        cache.buffer_only = True
        cache.enable_storage = True
        cache.is_eagle = False
        cache.ongoing_prefetch = {}
        cache.pending_prefetch_query = deque()
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(size=2),
            mem_pool_device_allocator=SimpleNamespace(size=100),
            prefetch=MagicMock(),
        )

        cache.prefetch_from_storage(
            "req",
            _node([], values=[]),
            list(range(8)),
        )

        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(list(cache.pending_prefetch_query), [])
        cache.cache_controller.prefetch.assert_not_called()

    def test_wait_complete_cancels_statically_impossible_stream(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.prefetch_stream_window_tokens = 4
        cache.buffer_only = True
        cache.prefetch_stop_policy = "wait_complete"
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache._trim_buffer_prefetch_against_device = MagicMock(return_value=True)
        cache._allocate_buffer_prefetch = MagicMock(return_value=False)
        cache._release_prefetch_context = MagicMock()
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(size=2),
            mem_pool_device_allocator=SimpleNamespace(size=100),
        )
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.set_query_result(["h0", "h1", "h2", "h3"], 8)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(8))),
            operation=operation,
            query_ready=True,
        )
        cache.ongoing_prefetch = {"req": context}

        self.assertTrue(cache.check_prefetch_progress("req"))

        self.assertTrue(operation.is_terminated())
        self.assertNotIn("req", cache.ongoing_prefetch)
        cache._release_prefetch_context.assert_called_once_with(
            context, free_completed=True
        )

    def test_required_stream_is_not_cancelled_while_waiting_for_capacity(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.buffer_only = True
        cache.prefetch_threshold = 4
        cache.prefetch_stop_policy = "best_effort"
        cache.streaming_prefetch_active = set()
        cache._trim_buffer_prefetch_against_device = MagicMock(return_value=True)
        cache._allocate_buffer_prefetch = MagicMock(return_value=False)
        cache._buffer_prefetch_statically_impossible = MagicMock(return_value=False)
        cache._all_reduce_attn_groups = lambda tensor, op: None
        operation = PrefetchOperation("req", None, list(range(4)))
        operation.set_query_result(["h0", "h1"], 4)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(4))),
            operation=operation,
            query_ready=True,
            require_full_completion=True,
        )
        cache.ongoing_prefetch = {"req": context}

        self.assertFalse(cache.check_prefetch_progress("req"))

        self.assertFalse(operation.is_terminated())
        self.assertIn("req", cache.ongoing_prefetch)

    def test_required_stream_ignores_best_effort_partial_completion(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.buffer_only = True
        cache.prefetch_stop_policy = "best_effort"
        cache.can_terminate_prefetch = MagicMock(return_value=False)
        host_indices = torch.arange(4)
        device_indices = torch.arange(100, 104)
        operation = PrefetchOperation("req", None, list(range(4)))
        operation.set_query_result(["h0", "h1"], 4)
        operation.configure_device_streaming(
            host_indices, device_indices, window_tokens=4
        )
        operation.completed_tokens = 2
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(4))),
            operation=operation,
            host_indices=host_indices,
            device_indices=device_indices,
            require_full_completion=True,
        )
        cache.ongoing_prefetch = {"req": context}

        self.assertFalse(cache.check_prefetch_progress("req"))

        self.assertFalse(operation.is_terminated())
        cache.can_terminate_prefetch.assert_called_once_with(operation)

    def test_streaming_prefetch_reserves_only_unallocated_request_budget(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_write_reserve_ratio = 0.2
        cache.prefetch_stream_window_tokens = 4
        cache.prefetch_stream_buffer_count = 1
        cache.streaming_prefetch_active = set()
        cache.streaming_prefetch_reservation_tokens = 0
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache.evictable_size = lambda: 0

        mem_pool_host = MagicMock()
        mem_pool_host.size = 4
        mem_pool_host.available_size.return_value = 4
        mem_pool_host.alloc.return_value = torch.arange(4)
        device_allocator = MagicMock()
        device_allocator.available_size.return_value = 32
        device_allocator.alloc.return_value = torch.arange(100, 108)
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            mem_pool_device_allocator=device_allocator,
            prefetch_tokens_occupied=0,
            prefetch_buffer=Queue(),
            record_device_ready_event=MagicMock(return_value=MagicMock()),
        )
        operation = PrefetchOperation("req", None, list(range(8)))
        operation.set_query_result(["h0", "h1", "h2", "h3"], 8)
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(8))),
            operation=operation,
        )

        self.assertTrue(
            cache._allocate_buffer_prefetch(
                context,
                device_admission_tokens=20,
            )
        )
        self.assertEqual(context.device_reservation_tokens, 12)
        self.assertEqual(cache.streaming_prefetch_reservation_tokens, 12)

    def test_prefill_budget_excludes_active_stream_reservation(self):
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.is_all_swa = False
        adder.is_hybrid_swa = False
        adder.is_hybrid_ssm_cache = False
        adder.rem_total_token_offset = 10
        adder.cur_rem_token_offset = 5
        allocator = MagicMock()
        allocator.available_size.return_value = 100
        adder.token_to_kv_pool_allocator = allocator
        adder.tree_cache = SimpleNamespace(
            evictable_size=lambda: 20,
            streaming_prefetch_reservation_tokens=30,
        )

        self.assertEqual(adder.rem_total_tokens, 80)
        self.assertEqual(adder.cur_rem_tokens, 85)

    def test_streaming_prefetch_release_frees_window_and_device_reservation(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.streaming_prefetch_active = {"req"}
        cache.streaming_prefetch_reservation_tokens = 12
        cache.dec_lock_ref = MagicMock()

        host_pool = MagicMock()
        device_allocator = MagicMock()
        controller = SimpleNamespace(
            mem_pool_host=host_pool,
            mem_pool_device_allocator=device_allocator,
            prefetch_tokens_occupied=4,
        )
        cache.cache_controller = controller

        operation = PrefetchOperation("req", None, list(range(8)))
        host_indices = torch.arange(4)
        device_indices = torch.arange(100, 108)
        operation.configure_device_streaming(
            host_indices, device_indices, window_tokens=4
        )
        context = PrefetchContext(
            anchor=_node([], values=[]),
            operation=operation,
            host_indices=host_indices,
            device_indices=device_indices,
            allocated_tokens=4,
            device_reservation_tokens=12,
            anchor_device_locked=True,
            key=RadixKey(array("q", range(8))),
        )

        cache._release_prefetch_context(context, free_completed=True)

        host_pool.free.assert_called_once()
        device_allocator.free.assert_called_once()
        self.assertIs(host_pool.free.call_args.args[0], host_indices)
        self.assertIs(device_allocator.free.call_args.args[0], device_indices)
        self.assertEqual(controller.prefetch_tokens_occupied, 0)
        self.assertEqual(cache.streaming_prefetch_active, set())
        self.assertEqual(cache.streaming_prefetch_reservation_tokens, 0)
        self.assertIsNone(operation.host_indices)
        self.assertIsNone(operation.device_indices)

    def test_streaming_prefetch_completion_publishes_device_prefix(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.buffer_only = True
        cache.prefetch_stop_policy = "wait_complete"
        cache.enable_storage_metrics = False
        cache.streaming_prefetch_active = {"req"}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache.can_terminate_prefetch = lambda operation: True
        cache.dec_lock_ref = MagicMock()
        cache._insert_helper_device_prefetch = MagicMock(return_value=0)
        cache.loading_node_to_ack = {_node([9, 10]): 1}
        cache._synchronize_loading_prefix = MagicMock(return_value=True)

        host_pool = MagicMock()
        device_allocator = MagicMock()
        controller = SimpleNamespace(
            mem_pool_host=host_pool,
            mem_pool_device_allocator=device_allocator,
            prefetch_tokens_occupied=4,
            terminate_prefetch=MagicMock(return_value=(4, ["h0", "h1"])),
        )
        cache.cache_controller = controller

        operation = PrefetchOperation("req", None, list(range(4)))
        operation.set_query_result(["h0", "h1"], 4)
        host_indices = torch.arange(4)
        device_indices = torch.arange(100, 104)
        operation.configure_device_streaming(
            host_indices, device_indices, window_tokens=4
        )
        operation.completed_tokens = 4
        operation.mark_io_done()
        context = PrefetchContext(
            anchor=_node([], values=[]),
            key=RadixKey(array("q", range(4))),
            operation=operation,
            host_indices=host_indices,
            allocated_tokens=4,
            anchor_device_locked=True,
            device_indices=device_indices,
        )
        cache.ongoing_prefetch = {"req": context}

        self.assertTrue(cache.check_prefetch_progress("req"))

        cache._insert_helper_device_prefetch.assert_called_once()
        cache._synchronize_loading_prefix.assert_called_once()
        sync_key = cache._synchronize_loading_prefix.call_args.args[0]
        sync_anchor = cache._synchronize_loading_prefix.call_args.kwargs["start_node"]
        self.assertEqual(list(sync_key), list(context.key))
        self.assertIs(sync_anchor, context.anchor)
        published = cache._insert_helper_device_prefetch.call_args.args[2]
        self.assertTrue(torch.equal(published, device_indices))
        host_pool.free.assert_called_once()
        device_allocator.free.assert_not_called()
        cache.dec_lock_ref.assert_called_once_with(context.anchor)
        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(cache.streaming_prefetch_active, set())
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid["req"], 4)

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

    def test_streaming_prefetch_query_is_not_capped_by_l2_budget(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = True
        cache.enable_storage = True
        cache.is_eagle = False
        cache.ongoing_prefetch = {}
        cache.pending_prefetch_query = deque()
        cache.prefetch_query_ops_occupied = 0
        cache.prefetch_query_tokens_occupied = 0
        cache.prefetch_query_max_ops = 8
        cache.prefetch_query_max_tokens = 32768
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
        cache.root_node = anchor
        cache.hicache_storage_pass_prefix_keys = False
        cache.prefetch_from_storage("req", anchor, list(range(20)))

        controller.prefetch.assert_not_called()
        cache.inc_lock_ref.assert_not_called()
        self.assertEqual(list(cache.pending_prefetch_query), ["req"])
        self.assertIsNone(cache.ongoing_prefetch["req"].operation)

        cache._dispatch_pending_prefetch_queries(1)

        controller.prefetch.assert_called_once()
        self.assertEqual(len(controller.prefetch.call_args.args[2]), 20)
        cache.inc_lock_ref.assert_called_once_with(anchor)

    def test_streaming_prefetch_rejects_evicted_anchor(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = True
        cache.enable_storage = True
        cache.is_eagle = False
        cache.ongoing_prefetch = {}
        cache.pending_prefetch_query = deque()
        cache.inc_lock_ref = MagicMock()

        anchor = _node([], values=None)
        cache.prefetch_from_storage("req", anchor, list(range(8)))

        self.assertNotIn("req", cache.ongoing_prefetch)
        self.assertEqual(list(cache.pending_prefetch_query), [])
        cache.inc_lock_ref.assert_not_called()

    def test_prequeried_prefetch_is_ready_without_async_queue_handoff(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = True
        cache.enable_storage = True
        cache.is_eagle = False
        cache.ongoing_prefetch = {}
        cache.pending_prefetch_query = deque()
        cache.inc_lock_ref = MagicMock()
        operation = PrefetchOperation("req", None, list(range(4)))
        operation.set_query_result(["h0", "h1"], 4)
        cache.cache_controller = SimpleNamespace(
            prefetch_with_query_result=MagicMock(return_value=operation)
        )

        anchor = _node([], values=[])
        cache.prefetch_from_storage(
            "req",
            anchor,
            list(range(4)),
            known_hash_values=["h0", "h1"],
        )

        context = cache.ongoing_prefetch["req"]
        self.assertIs(context.operation, operation)
        self.assertTrue(context.query_ready)
        self.assertEqual(list(cache.pending_prefetch_query), [])
        cache.inc_lock_ref.assert_called_once_with(anchor)

    def test_streaming_prefetch_query_dispatches_without_l2_budget(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = True
        cache.pending_prefetch_query = deque(["req"])
        cache.prefetch_query_ops_occupied = 0
        cache.prefetch_query_tokens_occupied = 0
        cache.prefetch_query_max_ops = 8
        cache.prefetch_query_max_tokens = 32768
        root = _node([], values=[])
        cache.root_node = root
        cache.hicache_storage_pass_prefix_keys = False
        cache.inc_lock_ref = MagicMock()
        cache.ongoing_prefetch = {
            "req": PrefetchContext(
                anchor=root,
                key=RadixKey(array("q", range(8))),
            )
        }
        cache._get_extra_pools = lambda: {}
        controller = MagicMock()
        controller.mem_pool_host.size = 8
        cache.cache_controller = controller

        cache._dispatch_pending_prefetch_queries(1)

        controller.prefetch.assert_called_once()
        self.assertEqual(len(controller.prefetch.call_args.args[2]), 8)
        self.assertEqual(list(cache.pending_prefetch_query), [])

    def test_streaming_query_dispatch_is_bounded_independently_of_l2(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = True
        cache.pending_prefetch_query = deque(["req1", "req2"])
        cache.prefetch_query_ops_occupied = 0
        cache.prefetch_query_tokens_occupied = 0
        cache.prefetch_query_max_ops = 1
        cache.prefetch_query_max_tokens = 8
        root = _node([], values=[])
        cache.root_node = root
        cache.hicache_storage_pass_prefix_keys = False
        cache.inc_lock_ref = MagicMock()
        cache.dec_lock_ref = MagicMock()
        cache.ongoing_prefetch = {
            req_id: PrefetchContext(
                anchor=root,
                key=RadixKey(array("q", range(8))),
            )
            for req_id in ("req1", "req2")
        }
        cache._get_extra_pools = lambda: {}
        controller = MagicMock()
        controller.prefetch.side_effect = lambda req_id, *_args, **_kwargs: (
            PrefetchOperation(req_id, None, list(range(8)))
        )
        cache.cache_controller = controller

        cache._dispatch_pending_prefetch_queries(2)

        controller.prefetch.assert_called_once()
        self.assertEqual(list(cache.pending_prefetch_query), ["req2"])
        self.assertEqual(cache.prefetch_query_ops_occupied, 1)
        self.assertEqual(cache.prefetch_query_tokens_occupied, 8)

        cache._release_prefetch_query_slot(cache.ongoing_prefetch["req1"])
        cache._dispatch_pending_prefetch_queries(1)

        self.assertEqual(controller.prefetch.call_count, 2)
        self.assertEqual(list(cache.pending_prefetch_query), [])

    def test_pending_query_rebases_after_its_original_anchor_is_evicted(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = True
        cache.prefetch_query_ops_occupied = 0
        cache.prefetch_query_tokens_occupied = 0
        cache.prefetch_query_max_ops = 8
        cache.prefetch_query_max_tokens = 32768
        cache.hicache_storage_pass_prefix_keys = False
        cache._get_extra_pools = lambda: {}
        cache.inc_lock_ref = MagicMock()

        root = _node([], values=[])
        parent = _node([1, 2], parent=root, values=[10, 11])
        parent.hash_value = ["h0"]
        detached = _node([3, 4], parent=parent)
        root.children[parent.key.child_key(cache.page_size)] = parent
        cache.root_node = root
        context = PrefetchContext(
            anchor=detached,
            key=RadixKey(array("q", [5, 6, 7, 8])),
        )
        cache.ongoing_prefetch = {"req": context}
        cache.pending_prefetch_query = deque(["req"])
        operation = PrefetchOperation("req", None, list(range(6)))
        controller = MagicMock()
        controller.prefetch.return_value = operation
        cache.cache_controller = controller

        cache._dispatch_pending_prefetch_queries(1)

        self.assertIs(context.anchor, parent)
        self.assertTrue(context.anchor_device_locked)
        self.assertEqual(list(context.key), [3, 4, 5, 6, 7, 8])
        self.assertEqual(controller.prefetch.call_args.args[3], "h0")
        cache.inc_lock_ref.assert_called_once_with(parent)

    def test_pending_bigram_query_rebases_without_losing_boundary_token(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.hicache_storage_pass_prefix_keys = False
        cache.inc_lock_ref = MagicMock()
        root = _node([], values=[])
        cache.root_node = root
        detached = _node([], parent=root)
        detached.key = RadixKey(array("q", [1, 2, 3]), is_bigram=True)
        context = PrefetchContext(
            anchor=detached,
            key=RadixKey(array("q", [3, 4, 5, 6, 7]), is_bigram=True),
        )

        self.assertTrue(cache._lock_or_rebase_buffer_prefetch_anchor(context))

        self.assertEqual(list(context.key.raw_token_ids()), list(range(1, 8)))
        self.assertEqual(len(context.key), 6)
        self.assertIs(context.anchor, root)

    def test_streaming_query_dispatch_pauses_for_write_pressure(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.buffer_only = True
        cache.pending_prefetch_query = deque(["req"])
        cache.cache_controller = MagicMock()

        cache._dispatch_pending_prefetch_queries(1, pause_for_write=True)

        cache.cache_controller.prefetch.assert_not_called()
        self.assertEqual(list(cache.pending_prefetch_query), ["req"])

    def test_legacy_prefetch_dispatch_failure_releases_host_ownership(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.buffer_only = False
        cache.enable_storage = True
        cache.is_eagle = False
        cache.ongoing_prefetch = {}
        cache._get_extra_pools = lambda: {}
        host_indices = torch.arange(4)
        host_pool = MagicMock()
        host_pool.alloc.return_value = host_indices
        controller = SimpleNamespace(
            prefetch_rate_limited=MagicMock(return_value=False),
            mem_pool_host=host_pool,
            prefetch=MagicMock(side_effect=RuntimeError("queue unavailable")),
        )
        cache.cache_controller = controller
        anchor = _node([], values=[])
        anchor.host_value = torch.empty(0)

        with self.assertLogs(level="ERROR"):
            cache.prefetch_from_storage("req", anchor, list(range(4)))

        host_pool.free.assert_called_once_with(host_indices)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertNotIn("req", cache.ongoing_prefetch)

    def test_prefetch_io_timeout_starts_when_io_is_admitted(self):
        cache = HiRadixCache.__new__(HiRadixCache)
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
        cache = HiRadixCache.__new__(HiRadixCache)
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
        cache = HiRadixCache.__new__(HiRadixCache)
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
        cache = HiRadixCache.__new__(HiRadixCache)
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
        cache = HiRadixCache.__new__(HiRadixCache)
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
        cache = HiRadixCache.__new__(HiRadixCache)
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
        cache = HiRadixCache.__new__(HiRadixCache)
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
