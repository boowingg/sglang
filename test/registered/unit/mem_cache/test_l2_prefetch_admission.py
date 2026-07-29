import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.l2_prefetch_admission import (
    L2PrefetchAdmission,
    L2PrefetchAdmissionConfig,
    get_l2_prefetch_admission,
    log_l2_prefetch_admission_status,
    prefetch_from_storage_with_admission,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _FakeRadixKey:
    def __init__(self, tokens, extra_key=None, is_bigram=False):
        self.tokens = list(tokens)
        self.extra_key = extra_key
        self.is_bigram = is_bigram

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        result = self.tokens[item]
        if isinstance(item, slice):
            return _FakeRadixKey(result, self.extra_key, self.is_bigram)
        return result

    def page_aligned(self, page_size):
        length = len(self.tokens) - len(self.tokens) % page_size
        return self[:length]


class _FakeNode:
    def __init__(self):
        self.key = SimpleNamespace(extra_key=None)
        self.protection = 0

    def protect_host(self):
        self.protection += 1

    def release_host(self):
        self.protection -= 1


class _FakeHostPool:
    def __init__(self, alloc_results, available_size=0, size=100):
        self.alloc_results = list(alloc_results)
        self._available_size = available_size
        self.size = size
        self.alloc_calls = []
        self.freed_lengths = []

    def alloc(self, num_tokens):
        self.alloc_calls.append(num_tokens)
        if not self.alloc_results:
            return None
        result = self.alloc_results.pop(0)
        if isinstance(result, int):
            return torch.arange(result)
        return result

    def available_size(self):
        return self._available_size

    def free(self, indices):
        self.freed_lengths.append(len(indices))


class _FakeController:
    def __init__(self, host_pool, *, rate_limited=False):
        self.mem_pool_host = host_pool
        self.enable_storage = True
        self.prefetch_capacity_limit = 50
        self.prefetch_tokens_occupied = 0
        self.rate_limited = rate_limited
        self.prefetch_calls = []
        self.prefetch_error = None

        self.prefetch_revoke_queue = Queue()
        self.ack_backup_queue = Queue()
        self.host_mem_release_queue = Queue()

    def prefetch_rate_limited(self):
        return self.rate_limited

    def prefetch(self, *args, **kwargs):
        self.prefetch_calls.append((args, kwargs))
        if self.prefetch_error is not None:
            raise self.prefetch_error
        return object()


class _FakeCache:
    def __init__(self, controller):
        self.cache_controller = controller
        self.enable_storage = True
        self.prefetch_threshold = 4
        self.page_size = 2
        self.is_eagle = False
        self.ongoing_prefetch = {}
        self.peer_min_tokens = None
        self.peer_blocked = False
        self.peer_capacity_changed = False
        self.eviction_calls = []
        self.drain_calls = []

    def _all_reduce_attn_groups(self, tensor, op):
        if op == torch.distributed.ReduceOp.MIN and self.peer_min_tokens is not None:
            tensor.fill_(min(int(tensor.min().item()), self.peer_min_tokens))
        elif (
            op == torch.distributed.ReduceOp.MIN
            and self.peer_capacity_changed
            and len(tensor) == 5
        ):
            tensor[-1] = 0
        elif op == torch.distributed.ReduceOp.MAX and self.peer_blocked:
            tensor.fill_(1)

    def evict_host(self, num_tokens):
        self.eviction_calls.append(num_tokens)

    def _get_extra_pools(self):
        return {}

    def _drain_storage_control_queues_impl(self, **kwargs):
        self.drain_calls.append(kwargs)


def _enabled_config(**overrides):
    values = {
        "enabled": True,
        "capacity_ratio": 0.5,
        "wait_timeout": 30.0,
    }
    values.update(overrides)
    return L2PrefetchAdmissionConfig(**values)


class TestL2PrefetchAdmissionState(unittest.TestCase):
    def test_defaults_are_disabled_and_preserve_backend_config(self):
        extra_config = {"backend_option": "value"}

        config = L2PrefetchAdmissionConfig.pop_from(extra_config)

        self.assertEqual(config, L2PrefetchAdmissionConfig())
        self.assertEqual(extra_config, {"backend_option": "value"})

    def test_config_is_removed_from_backend_options(self):
        extra_config = {
            "enable_l2_prefetch_queue": True,
            "prefetch_capacity_ratio": 0.75,
            "l2_prefetch_wait_timeout": 12,
            "backend_option": "value",
        }

        config = L2PrefetchAdmissionConfig.pop_from(extra_config)

        self.assertEqual(
            config,
            L2PrefetchAdmissionConfig(
                enabled=True, capacity_ratio=0.75, wait_timeout=12.0
            ),
        )
        self.assertEqual(extra_config, {"backend_option": "value"})

    def test_invalid_config_is_rejected(self):
        invalid_configs = [
            {"enable_l2_prefetch_queue": 1},
            {"prefetch_capacity_ratio": True},
            {"prefetch_capacity_ratio": 0},
            {"prefetch_capacity_ratio": 1.1},
            {"prefetch_capacity_ratio": float("nan")},
            {"l2_prefetch_wait_timeout": True},
            {"l2_prefetch_wait_timeout": 0},
            {"l2_prefetch_wait_timeout": float("inf")},
        ]
        for extra_config in invalid_configs:
            with self.subTest(extra_config=extra_config):
                with self.assertRaises(ValueError):
                    L2PrefetchAdmissionConfig.pop_from(extra_config)

    def test_status_log_reports_enabled_supported_cache(self):
        cache = _FakeCache(_FakeController(_FakeHostPool([], size=200)))
        cache._l2_prefetch_admission = L2PrefetchAdmission(
            _enabled_config(capacity_ratio=0.75, wait_timeout=12)
        )
        cache.cache_controller.prefetch_capacity_limit = 150

        with self.assertLogs(
            "sglang.srt.mem_cache.l2_prefetch_admission", level="INFO"
        ) as logs:
            log_l2_prefetch_admission_status(cache)

        message = logs.output[0]
        self.assertIn("cache_impl=_FakeCache", message)
        self.assertIn("supported=true enabled=true active=true", message)
        self.assertIn("capacity_ratio=0.750", message)
        self.assertIn("wait_timeout_s=12.00", message)
        self.assertIn("capacity_limit_tokens=150", message)

    def test_status_log_reports_unsupported_cache(self):
        cache = SimpleNamespace()

        with self.assertLogs(
            "sglang.srt.mem_cache.l2_prefetch_admission", level="INFO"
        ) as logs:
            log_l2_prefetch_admission_status(cache)

        message = logs.output[0]
        self.assertIn("cache_impl=SimpleNamespace", message)
        self.assertIn("supported=false enabled=false active=false", message)
        self.assertIn("reason=unsupported_cache_impl", message)

    def test_disabled_queue_does_not_hold_requests(self):
        admission = L2PrefetchAdmission(L2PrefetchAdmissionConfig())

        self.assertFalse(admission.enqueue("req"))
        self.assertFalse(admission.is_waiting("req"))
        self.assertFalse(admission.can_retry("req"))

    def test_waiting_cohort_retries_in_scheduler_order(self):
        admission = L2PrefetchAdmission(_enabled_config())
        admission.enqueue("first")
        admission.enqueue("second")
        admission.capacity_changed(synchronized=True)

        self.assertTrue(admission.can_retry("first"))
        self.assertTrue(admission.can_retry("second"))
        self.assertFalse(admission.should_wait_behind_existing("second"))
        self.assertTrue(admission.should_wait_behind_existing("new"))

    def test_failed_retry_waits_for_next_capacity_generation(self):
        admission = L2PrefetchAdmission(_enabled_config())
        admission.enqueue("first")
        admission.capacity_changed(synchronized=True)
        self.assertTrue(admission.can_retry("first"))

        admission.enqueue("first")

        self.assertFalse(admission.can_retry("first"))
        admission.capacity_changed(synchronized=True)
        self.assertTrue(admission.can_retry("first"))

    def test_removing_oldest_wakes_next_waiter(self):
        admission = L2PrefetchAdmission(_enabled_config())
        admission.enqueue("first")
        admission.enqueue("second")

        admission.cancel("first")

        self.assertTrue(admission.can_retry("second"))

    def test_requeue_preserves_original_timeout(self):
        clock = _Clock()
        admission = L2PrefetchAdmission(_enabled_config(wait_timeout=10), clock=clock)
        admission.enqueue("first")
        clock.advance(6)
        admission.enqueue("first")
        clock.advance(5)

        self.assertTrue(admission.oldest_expired())
        self.assertEqual(admission.expire_oldest(), ("first", 11.0))

    def test_expiring_oldest_wakes_next_waiter(self):
        clock = _Clock()
        admission = L2PrefetchAdmission(_enabled_config(wait_timeout=5), clock=clock)
        admission.enqueue("first")
        admission.enqueue("second")
        clock.advance(6)

        self.assertEqual(admission.expire_oldest(), ("first", 6.0))
        self.assertTrue(admission.can_retry("second"))

    def test_follower_timeout_starts_only_after_head_promotion(self):
        clock = _Clock()
        admission = L2PrefetchAdmission(_enabled_config(wait_timeout=10), clock=clock)
        admission.enqueue("first")
        admission.enqueue("second")
        clock.advance(11)

        self.assertEqual(admission.expire_oldest(), ("first", 11.0))
        self.assertFalse(admission.oldest_expired())

        clock.advance(9)
        self.assertFalse(admission.oldest_expired())
        clock.advance(1)
        self.assertTrue(admission.oldest_expired())
        self.assertEqual(admission.expire_oldest(), ("second", 10.0))

    def test_canceling_head_starts_next_waiter_timeout(self):
        clock = _Clock()
        admission = L2PrefetchAdmission(_enabled_config(wait_timeout=10), clock=clock)
        admission.enqueue("first")
        admission.enqueue("second")
        clock.advance(20)

        admission.cancel("first")

        self.assertFalse(admission.oldest_expired())
        clock.advance(10)
        self.assertTrue(admission.oldest_expired())

    def test_canceling_follower_does_not_reset_head_timeout(self):
        clock = _Clock()
        admission = L2PrefetchAdmission(_enabled_config(wait_timeout=10), clock=clock)
        admission.enqueue("first")
        admission.enqueue("second")
        clock.advance(6)

        admission.cancel("second")
        clock.advance(4)

        self.assertTrue(admission.oldest_expired())

    def test_configure_clears_waiters_and_applies_capacity_ratio(self):
        controller = _FakeController(_FakeHostPool([], size=200))
        admission = L2PrefetchAdmission(_enabled_config())
        admission.enqueue("first")

        admission.configure(
            _enabled_config(capacity_ratio=0.75),
            controller,
        )

        self.assertEqual(admission.waiting_count, 0)
        self.assertEqual(controller.prefetch_capacity_limit, 150)

    def test_disabled_adapter_calls_original_prefetch(self):
        cache = SimpleNamespace(
            _l2_prefetch_admission=L2PrefetchAdmission(L2PrefetchAdmissionConfig()),
            prefetch_from_storage=Mock(),
        )
        node = object()

        prefetch_from_storage_with_admission(
            cache, "req", node, [1, 2], "hash", ["prefix"]
        )

        cache.prefetch_from_storage.assert_called_once_with(
            "req", node, [1, 2], "hash", ["prefix"]
        )
        self.assertIsNone(get_l2_prefetch_admission(cache))

    def test_enabled_adapter_uses_admission_path(self):
        admission = L2PrefetchAdmission(_enabled_config())
        admission.prefetch = Mock()
        cache = SimpleNamespace(
            _l2_prefetch_admission=admission,
            prefetch_from_storage=Mock(),
        )
        node = object()

        prefetch_from_storage_with_admission(cache, "req", node, [1, 2])

        admission.prefetch.assert_called_once_with(
            cache, "req", node, [1, 2], None, None
        )
        cache.prefetch_from_storage.assert_not_called()


@patch("sglang.srt.mem_cache.radix_cache.RadixKey", _FakeRadixKey)
class TestL2PrefetchAdmissionIO(unittest.TestCase):
    def test_rate_limited_prefetch_waits_without_pinning_node(self):
        controller = _FakeController(_FakeHostPool([]), rate_limited=True)
        controller.prefetch_tokens_occupied = controller.prefetch_capacity_limit
        cache = _FakeCache(controller)
        node = _FakeNode()
        admission = L2PrefetchAdmission(_enabled_config())

        admission.prefetch(cache, "req", node, range(8))

        self.assertTrue(admission.is_waiting("req"))
        self.assertEqual(node.protection, 0)
        self.assertEqual(controller.prefetch_calls, [])

    def test_peer_admission_limit_makes_local_rank_wait(self):
        controller = _FakeController(_FakeHostPool([]))
        cache = _FakeCache(controller)
        cache.peer_blocked = True
        node = _FakeNode()
        admission = L2PrefetchAdmission(_enabled_config())

        admission.prefetch(cache, "req", node, range(8))

        self.assertTrue(admission.is_waiting("req"))
        self.assertEqual(node.protection, 0)

    def test_allocation_failure_waits_and_releases_node(self):
        controller = _FakeController(_FakeHostPool([None, None], available_size=0))
        cache = _FakeCache(controller)
        node = _FakeNode()
        admission = L2PrefetchAdmission(_enabled_config())

        admission.prefetch(cache, "req", node, range(8))

        self.assertTrue(admission.is_waiting("req"))
        self.assertEqual(node.protection, 0)
        self.assertEqual(cache.eviction_calls, [8])

    def test_partial_allocation_is_page_aligned_and_bounded(self):
        host_pool = _FakeHostPool([None, None, 6], available_size=7)
        controller = _FakeController(host_pool)
        cache = _FakeCache(controller)
        node = _FakeNode()
        admission = L2PrefetchAdmission(_enabled_config())

        admission.prefetch(cache, "req", node, range(8))

        self.assertEqual(host_pool.alloc_calls, [8, 8, 6])
        self.assertEqual(controller.prefetch_tokens_occupied, 6)
        self.assertEqual(len(controller.prefetch_calls[0][0][2]), 6)

    def test_partial_fallback_never_allocates_more_than_requested(self):
        host_pool = _FakeHostPool([None, None, 8], available_size=10)
        controller = _FakeController(host_pool)
        cache = _FakeCache(controller)
        node = _FakeNode()
        admission = L2PrefetchAdmission(_enabled_config())

        admission.prefetch(cache, "req", node, range(8))

        self.assertEqual(host_pool.alloc_calls, [8, 8, 8])
        self.assertEqual(controller.prefetch_tokens_occupied, 8)
        self.assertEqual(len(controller.prefetch_calls[0][0][2]), 8)

    def test_smallest_tp_allocation_is_used_on_every_rank(self):
        host_pool = _FakeHostPool([8])
        controller = _FakeController(host_pool)
        cache = _FakeCache(controller)
        cache.peer_min_tokens = 6
        node = _FakeNode()
        admission = L2PrefetchAdmission(_enabled_config())

        admission.prefetch(cache, "req", node, range(8))

        self.assertEqual(host_pool.freed_lengths, [2])
        self.assertEqual(controller.prefetch_tokens_occupied, 6)
        self.assertEqual(len(controller.prefetch_calls[0][0][2]), 6)

    def test_submission_failure_rolls_back_allocation_and_protection(self):
        host_pool = _FakeHostPool([8])
        controller = _FakeController(host_pool)
        controller.prefetch_error = RuntimeError("submission failed")
        cache = _FakeCache(controller)
        node = _FakeNode()
        admission = L2PrefetchAdmission(_enabled_config())

        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            admission.prefetch(cache, "req", node, range(8))

        self.assertEqual(host_pool.freed_lengths, [8])
        self.assertEqual(node.protection, 0)
        self.assertFalse(admission.is_waiting("req"))
        self.assertNotIn("req", cache.ongoing_prefetch)

    def test_successful_retry_removes_wait_state(self):
        host_pool = _FakeHostPool([8])
        controller = _FakeController(host_pool, rate_limited=True)
        controller.prefetch_tokens_occupied = controller.prefetch_capacity_limit
        cache = _FakeCache(controller)
        node = _FakeNode()
        admission = L2PrefetchAdmission(_enabled_config())
        admission.prefetch(cache, "req", node, range(8))

        controller.rate_limited = False
        controller.prefetch_tokens_occupied = 0
        admission.capacity_changed(synchronized=True)
        admission.prefetch(cache, "req", node, range(8))

        self.assertFalse(admission.is_waiting("req"))
        self.assertIn("req", cache.ongoing_prefetch)
        self.assertEqual(controller.prefetch_tokens_occupied, 8)

    def test_timeout_is_synchronized_with_control_queue_drain(self):
        clock = _Clock()
        controller = _FakeController(_FakeHostPool([]))
        cache = _FakeCache(controller)
        admission = L2PrefetchAdmission(_enabled_config(wait_timeout=5), clock=clock)
        admission.enqueue("req")
        clock.advance(6)

        expired = admission.drain_storage_control_queues(cache)

        self.assertEqual(expired, ("req", 6.0))
        self.assertEqual(
            cache.drain_calls,
            [
                {
                    "n_revoke": 0,
                    "n_backup": 0,
                    "n_release": 0,
                    "log_metrics": True,
                }
            ],
        )

    def test_control_queue_drain_does_not_cascade_expired_followers(self):
        clock = _Clock()
        controller = _FakeController(_FakeHostPool([]))
        cache = _FakeCache(controller)
        admission = L2PrefetchAdmission(_enabled_config(wait_timeout=5), clock=clock)
        admission.enqueue("first")
        admission.enqueue("second")
        clock.advance(6)

        first_expired = admission.drain_storage_control_queues(cache)
        second_expired = admission.drain_storage_control_queues(cache)

        self.assertEqual(first_expired, ("first", 6.0))
        self.assertIsNone(second_expired)
        clock.advance(5)
        self.assertEqual(
            admission.drain_storage_control_queues(cache), ("second", 5.0)
        )

    def test_local_capacity_change_is_published_by_control_queue_drain(self):
        controller = _FakeController(_FakeHostPool([]))
        cache = _FakeCache(controller)
        admission = L2PrefetchAdmission(_enabled_config())
        admission.enqueue("req")
        admission.capacity_changed()
        self.assertFalse(admission.can_retry("req"))

        admission.drain_storage_control_queues(cache)

        self.assertTrue(admission.can_retry("req"))

    def test_peer_capacity_change_wakes_local_waiter(self):
        controller = _FakeController(_FakeHostPool([]))
        cache = _FakeCache(controller)
        cache.peer_capacity_changed = True
        admission = L2PrefetchAdmission(_enabled_config())
        admission.enqueue("req")

        admission.drain_storage_control_queues(cache)

        self.assertTrue(admission.can_retry("req"))

    def test_host_release_queue_wakes_waiters_in_same_drain(self):
        controller = _FakeController(_FakeHostPool([]))
        controller.host_mem_release_queue.put(torch.arange(4))
        cache = _FakeCache(controller)
        admission = L2PrefetchAdmission(_enabled_config())
        admission.enqueue("req")

        admission.drain_storage_control_queues(cache)

        self.assertTrue(admission.can_retry("req"))
        self.assertEqual(cache.drain_calls[0]["n_release"], 1)


if __name__ == "__main__":
    unittest.main()
