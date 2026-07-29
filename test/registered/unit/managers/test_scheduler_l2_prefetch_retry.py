import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.l2_prefetch_admission import (
    L2PrefetchAdmission,
    L2PrefetchAdmissionConfig,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_scheduler(admission, req_ids):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tree_cache = SimpleNamespace(_l2_prefetch_admission=admission)
    scheduler.waiting_queue = [SimpleNamespace(rid=req_id) for req_id in req_ids]
    scheduler._prefetch_kvcache = Mock(return_value=True)
    return scheduler


class TestSchedulerL2PrefetchRetry(unittest.TestCase):
    def test_new_waiter_retries_only_oldest_once(self):
        admission = L2PrefetchAdmission(L2PrefetchAdmissionConfig(enabled=True))
        admission.enqueue("first")
        admission.enqueue("second")
        scheduler = _make_scheduler(admission, ["first", "second"])

        Scheduler._retry_oldest_l2_prefetch(scheduler)
        Scheduler._retry_oldest_l2_prefetch(scheduler)

        scheduler._prefetch_kvcache.assert_called_once_with(scheduler.waiting_queue[0])

    def test_successful_oldest_retry_wakes_next_for_following_round(self):
        admission = L2PrefetchAdmission(L2PrefetchAdmissionConfig(enabled=True))
        admission.enqueue("first")
        admission.enqueue("second")
        scheduler = _make_scheduler(admission, ["first", "second"])
        scheduler._prefetch_kvcache.side_effect = lambda req: (
            admission.cancel(req.rid) or True
        )

        Scheduler._retry_oldest_l2_prefetch(scheduler)
        Scheduler._retry_oldest_l2_prefetch(scheduler)

        self.assertEqual(
            [call.args[0].rid for call in scheduler._prefetch_kvcache.call_args_list],
            ["first", "second"],
        )

    def test_retry_hook_precedes_batch_full_return(self):
        source = inspect.getsource(Scheduler._get_new_batch_prefill_raw)

        self.assertLess(
            source.index("self._retry_oldest_l2_prefetch()"),
            source.index("running_batch.batch_is_full or"),
        )


if __name__ == "__main__":
    unittest.main()
