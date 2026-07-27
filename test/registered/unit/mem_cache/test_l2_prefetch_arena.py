import threading
import unittest
from array import array
from queue import Queue
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hiradix_cache import (
    HiRadixCache,
    _ReadyArenaPrefetch,
)
from sglang.srt.mem_cache.l2_prefetch_arena import L2PrefetchArena
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeHostPool:
    def __init__(self, size: int):
        self.size = size
        self._lock = threading.RLock()
        self._used = torch.zeros(size, dtype=torch.bool)
        self._free_slots = torch.arange(size, dtype=torch.int64)

    def clear(self):
        with self._lock:
            self._used.zero_()
            self._free_slots = torch.arange(self.size, dtype=torch.int64)

    def alloc(self, need_size: int):
        with self._lock:
            if need_size > len(self._free_slots):
                return None
            indices = self._free_slots[:need_size]
            self._free_slots = self._free_slots[need_size:]
            assert not self._used[indices].any()
            self._used[indices] = True
            return indices

    def free(self, indices: torch.Tensor):
        with self._lock:
            indices = indices.cpu()
            assert self._used[indices].all()
            self._used[indices] = False
            self._free_slots = torch.cat((self._free_slots, indices))
            return len(indices)

    def available_size(self):
        return len(self._free_slots)


class TestL2PrefetchArena(CustomTestCase):
    def setUp(self):
        self.host_pool = _FakeHostPool(size=32)
        self.arena = L2PrefetchArena(
            host_pool=self.host_pool,
            capacity_tokens=16,
            page_size=4,
        )

    def test_reserved_indices_are_isolated_from_normal_l2(self):
        self.assertEqual(self.host_pool.available_size(), 16)
        normal_indices = self.host_pool.alloc(16)
        self.assertIsNotNone(normal_indices)
        self.assertIsNone(self.host_pool.alloc(4))

        arena_indices = self.arena.alloc(16)
        self.assertIsNotNone(arena_indices)
        self.assertTrue(set(normal_indices.tolist()).isdisjoint(arena_indices.tolist()))

    def test_release_reuses_capacity_and_advances_generation(self):
        initial_generation = self.arena.generation
        first = self.arena.alloc(8)
        self.assertEqual(self.arena.available_size(), 8)
        self.assertGreater(self.arena.generation, initial_generation)

        allocated_generation = self.arena.generation
        self.arena.free(first)
        self.assertEqual(self.arena.available_size(), 16)
        self.assertGreater(self.arena.generation, allocated_generation)
        self.assertIsNotNone(self.arena.alloc(16))

    def test_duplicate_free_is_rejected(self):
        indices = self.arena.alloc(4)
        self.arena.free(indices)
        with self.assertRaisesRegex(RuntimeError, "invalid or duplicate free"):
            self.arena.free(indices)

    def test_close_requires_all_transient_pages_to_be_returned(self):
        indices = self.arena.alloc(4)
        with self.assertRaisesRegex(RuntimeError, "still in use"):
            self.arena.close()

        self.arena.free(indices)
        self.arena.close()
        self.assertEqual(self.host_pool.available_size(), 32)

    def test_rebind_after_parent_pool_clear(self):
        self.arena.alloc(8)
        self.host_pool.clear()
        self.arena.rebind_after_host_clear()

        self.assertEqual(self.host_pool.available_size(), 16)
        self.assertEqual(self.arena.available_size(), 16)
        self.assertIsNotNone(self.arena.alloc(16))


class _FakeCacheController:
    def __init__(self):
        self.prefetch_tokens_occupied = 0
        self.prefetch_calls = []

    def prefetch(
        self,
        request_id,
        host_indices,
        prefetch_key,
        last_hash,
        prefix_keys,
        host_release_callback=None,
        **kwargs,
    ):
        operation = SimpleNamespace(
            request_id=request_id,
            host_indices=host_indices,
            host_release_callback=host_release_callback,
        )
        self.prefetch_calls.append(operation)
        return operation

    def load(self, host_indices, node_id, **kwargs):
        return torch.arange(len(host_indices), dtype=torch.int64) + 100


class TestHiRadixPrefetchArenaFlow(CustomTestCase):
    def test_arena_config_is_removed_from_backend_config(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        (
            backend_config,
            _,
            _,
            _,
            enabled,
            size_gb,
        ) = cache._parse_storage_backend_extra_config(
            '{"enable_l2_prefetch_arena":true,'
            '"l2_prefetch_arena_size":4,"backend_option":"value"}'
        )

        self.assertTrue(enabled)
        self.assertEqual(size_gb, 4.0)
        self.assertEqual(backend_config, {"backend_option": "value"})

    def _make_cache(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.prefetch_threshold = 4
        cache.enable_storage = True
        cache.is_eagle = False
        cache.disable = False
        cache.device = torch.device("cpu")
        cache.cache_controller = _FakeCacheController()
        cache.l2_prefetch_arena = L2PrefetchArena(
            _FakeHostPool(size=32), capacity_tokens=8, page_size=2
        )
        cache.waiting_arena_prefetches = {}
        cache.ready_arena_prefetches = {}
        cache.arena_prefetch_base_len_by_reqid = {}
        cache.arena_release_after_load = {}
        cache.arena_mem_release_queue = Queue()
        cache.ongoing_prefetch = {}
        cache.ongoing_load_back = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.evictable_size_ = 0
        cache.protected_size_ = 0
        cache.metrics_collector = None
        cache._get_extra_pools = lambda: {}
        cache._all_reduce_attn_groups = lambda tensor, op: None
        cache._record_store_event = lambda *args, **kwargs: None

        root = TreeNode(priority=-1)
        root.key = RadixKey(array("q"))
        root.value = []
        root.host_value = []
        root.hash_value = []
        root.lock_ref = 1
        cache.root_node = root
        cache._empty_match_result = MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=root,
            last_host_node=root,
            best_match_node=root,
        )
        return cache

    def test_fifo_waiter_starts_after_arena_capacity_returns(self):
        cache = self._make_cache()
        root = cache.root_node

        cache.prefetch_from_storage("first", root, list(range(8)), prefetch_base_len=0)
        cache.prefetch_from_storage(
            "second", root, list(range(8, 16)), prefetch_base_len=0
        )
        cache.prefetch_from_storage(
            "third", root, list(range(16, 24)), prefetch_base_len=0
        )

        self.assertIn("first", cache.ongoing_prefetch)
        self.assertEqual(list(cache.waiting_arena_prefetches), ["second", "third"])

        first_operation = cache.ongoing_prefetch.pop("first")[3]
        first_operation.host_release_callback(first_operation.host_indices)
        cache.l2_prefetch_arena.free(cache.arena_mem_release_queue.get_nowait())
        cache._dispatch_waiting_arena_prefetches()

        self.assertIn("second", cache.ongoing_prefetch)
        self.assertEqual(list(cache.waiting_arena_prefetches), ["third"])

    def test_full_admission_waits_for_the_complete_decode_suffix(self):
        cache = self._make_cache()
        root = cache.root_node
        held = cache.l2_prefetch_arena.alloc(4)

        cache.prefetch_from_storage(
            "decode",
            root,
            list(range(8)),
            prefetch_base_len=0,
            require_full_prefetch=True,
        )

        self.assertIn("decode", cache.waiting_arena_prefetches)
        self.assertNotIn("decode", cache.ongoing_prefetch)

        cache.l2_prefetch_arena.free(held)
        cache._dispatch_waiting_arena_prefetches()

        self.assertNotIn("decode", cache.waiting_arena_prefetches)
        self.assertEqual(len(cache.ongoing_prefetch["decode"][2]), 8)

    def test_ready_suffix_loads_from_actual_deepest_l1_node(self):
        cache = self._make_cache()
        root = cache.root_node

        l1_node = TreeNode()
        l1_node.parent = root
        l1_node.key = RadixKey(array("q", [1, 2, 3, 4]))
        l1_node.value = torch.arange(4, dtype=torch.int64)
        l1_node.hash_value = ["h1", "h2"]
        root.children[l1_node.key.child_key(cache.page_size)] = l1_node
        cache.evictable_leaves.add(l1_node)
        cache.evictable_size_ = 4

        arena_indices = cache.l2_prefetch_arena.alloc(4)
        l1_node.protect_host()
        cache.ready_arena_prefetches["req"] = _ReadyArenaPrefetch(
            last_host_node=l1_node,
            prefetch_key=RadixKey(array("q", [5, 6, 7, 8])),
            host_indices=arena_indices,
            hash_values=["h3", "h4"],
            base_prefix_len=4,
        )
        req = SimpleNamespace(rid="req")
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3, 4, 5, 6, 7, 8])),
                req=req,
            )
        )

        self.assertIs(match.best_match_node, l1_node)
        self.assertEqual(match.host_hit_length, 4)
        loaded, end_node = cache._load_back_from_arena(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            ),
            cache.ready_arena_prefetches["req"],
        )

        self.assertEqual(len(loaded), 4)
        self.assertIs(end_node.parent, l1_node)
        self.assertEqual(end_node.key.token_ids.tolist(), [5, 6, 7, 8])
        self.assertNotIn("req", cache.ready_arena_prefetches)
        self.assertEqual(cache.l2_prefetch_arena.available_size(), 4)

        cache.l2_prefetch_arena.free(cache.arena_release_after_load.pop(end_node.id))
        self.assertEqual(cache.l2_prefetch_arena.available_size(), 8)

    def test_load_combines_normal_l2_prefix_with_arena_suffix(self):
        cache = self._make_cache()
        root = cache.root_node

        l1_node = TreeNode()
        l1_node.parent = root
        l1_node.key = RadixKey(array("q", [1, 2, 3, 4]))
        l1_node.value = torch.arange(4, dtype=torch.int64)
        l1_node.hash_value = ["h1", "h2"]
        root.children[l1_node.key.child_key(cache.page_size)] = l1_node
        cache.evictable_leaves.add(l1_node)
        cache.evictable_size_ = 4

        normal_host_indices = cache.l2_prefetch_arena.host_pool.alloc(4)
        l2_node = TreeNode()
        l2_node.parent = l1_node
        l2_node.key = RadixKey(array("q", [5, 6, 7, 8]))
        l2_node.value = None
        l2_node.host_value = normal_host_indices
        l2_node.hash_value = ["h3", "h4"]
        l1_node.children[l2_node.key.child_key(cache.page_size)] = l2_node

        arena_indices = cache.l2_prefetch_arena.alloc(4)
        l2_node.protect_host()
        cache.ready_arena_prefetches["req"] = _ReadyArenaPrefetch(
            last_host_node=l2_node,
            prefetch_key=RadixKey(array("q", [9, 10, 11, 12])),
            host_indices=arena_indices,
            hash_values=["h5", "h6"],
            base_prefix_len=8,
        )
        req = SimpleNamespace(rid="req")
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", list(range(1, 13)))),
                req=req,
            )
        )

        self.assertIs(match.best_match_node, l2_node)
        self.assertEqual(match.host_hit_length, 8)
        loaded, end_node = cache._load_back_from_arena(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            ),
            cache.ready_arena_prefetches["req"],
        )

        self.assertEqual(len(loaded), 8)
        self.assertIsNotNone(l2_node.value)
        self.assertEqual(len(l2_node.value), 4)
        self.assertIs(end_node.parent, l2_node)
        self.assertEqual(end_node.key.token_ids.tolist(), [9, 10, 11, 12])
        self.assertTrue(
            cache.l2_prefetch_arena.host_pool._used[normal_host_indices].all()
        )

        cache.l2_prefetch_arena.free(cache.arena_release_after_load.pop(end_node.id))
        self.assertEqual(cache.l2_prefetch_arena.available_size(), 8)

    def test_decode_prefetch_uses_deepest_l1_hash_anchor(self):
        cache = self._make_cache()
        root = cache.root_node

        l1_node = TreeNode()
        l1_node.parent = root
        l1_node.key = RadixKey(array("q", [1, 2, 3, 4]))
        l1_node.value = torch.arange(4, dtype=torch.int64)
        l1_node.hash_value = ["h1", "h2"]
        observed = {}

        def query_storage_hit_length(node, tokens, last_hash, prefix_keys):
            observed.update(
                node=node,
                tokens=tokens,
                last_hash=last_hash,
                prefix_keys=prefix_keys,
            )
            return 12

        cache.hicache_storage_pass_prefix_keys = False
        cache.query_storage_hit_length = query_storage_hit_length

        mixin = DecodeHiCachePreallocMixin()
        mixin.scheduler = SimpleNamespace(enable_decode_hicache=True)
        mixin.tree_cache = cache
        result = SimpleNamespace(
            device_indices=l1_node.value,
            host_hit_length=0,
            last_device_node=l1_node,
            last_host_node=root,
            best_match_node=l1_node,
        )
        prefix_match = mixin._build_decode_prefix_match(
            SimpleNamespace(origin_input_ids=list(range(16))),
            result,
        )

        self.assertIs(observed["node"], l1_node)
        self.assertEqual(observed["tokens"], list(range(4, 16)))
        self.assertEqual(observed["last_hash"], "h2")
        self.assertEqual(prefix_match.l3_storage_hit_length, 8)
        self.assertIs(prefix_match.last_host_node, l1_node)


if __name__ == "__main__":
    unittest.main()
