import threading
import time
import unittest

from gozero.job_queue import (
    InferenceJobQueue,
    PublicJobError,
    QueueFull,
)


class InferenceJobQueueTest(unittest.TestCase):
    def setUp(self):
        self.queues = []

    def tearDown(self):
        for queue in self.queues:
            queue.stop()

    def make_queue(self, **kwargs):
        queue = InferenceJobQueue(**kwargs)
        self.queues.append(queue)
        return queue

    def wait_for(self, queue, job_id, status, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            view = queue.view(job_id)
            if view is not None and view["status"] == status:
                return view
            time.sleep(0.01)
        self.fail(f"job {job_id} did not reach {status}")

    def test_reports_queue_position_then_result(self):
        gate = threading.Event()
        queue = self.make_queue(workers=1, max_pending=4)
        first, _ = queue.submit("move", lambda: (gate.wait(), {"first": True})[1])
        self.wait_for(queue, first.id, "running")

        second, _ = queue.submit("move", lambda: {"answer": 42})
        queued = queue.view(second.id)
        self.assertEqual(queued["status"], "queued")
        self.assertEqual(queued["queue_position"], 1)
        self.assertGreaterEqual(queued["estimated_wait_seconds"], 1)

        gate.set()
        completed = self.wait_for(queue, second.id, "completed")
        self.assertEqual(completed["result"], {"answer": 42})

    def test_idempotency_key_returns_the_original_job(self):
        gate = threading.Event()
        calls = []
        queue = self.make_queue(workers=1, max_pending=4)

        def work():
            calls.append(1)
            gate.wait()
            return {"ok": True}

        first, created = queue.submit("new", work, request_id="request-123")
        duplicate, duplicate_created = queue.submit(
            "new", work, request_id="request-123")
        self.assertTrue(created)
        self.assertFalse(duplicate_created)
        self.assertEqual(first.id, duplicate.id)
        gate.set()
        self.wait_for(queue, first.id, "completed")
        self.assertEqual(len(calls), 1)

    def test_pending_queue_is_bounded(self):
        gate = threading.Event()
        queue = self.make_queue(workers=1, max_pending=1)
        running, _ = queue.submit("move", lambda: (gate.wait(), {})[1])
        self.wait_for(queue, running.id, "running")
        queue.submit("move", lambda: {})
        with self.assertRaises(QueueFull):
            queue.submit("move", lambda: {})
        gate.set()

    def test_public_failure_is_returned_without_internal_details(self):
        queue = self.make_queue(workers=1, max_pending=2)

        def fail():
            raise PublicJobError("game state changed", 409)

        job, _ = queue.submit("move", fail)
        failed = self.wait_for(queue, job.id, "failed")
        self.assertEqual(failed["error"], "game state changed")
        self.assertEqual(failed["http_status"], 409)

    def test_legacy_wait_returns_completion_or_current_queue_state(self):
        gate = threading.Event()
        queue = self.make_queue(workers=1, max_pending=2)
        running, _ = queue.submit("new", lambda: (gate.wait(), {"ok": True})[1])
        self.wait_for(queue, running.id, "running")
        still_running = queue.wait(running.id, timeout=0.01)
        self.assertEqual(still_running["status"], "running")
        gate.set()
        completed = queue.wait(running.id, timeout=1.0)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["result"], {"ok": True})

    def test_premium_weight_does_not_starve_standard_jobs(self):
        gate = threading.Event()
        order = []
        queue = self.make_queue(
            workers=1,
            max_pending=10,
            premium_weight=3,
        )
        blocker, _ = queue.submit("move", lambda: (gate.wait(), {})[1])
        self.wait_for(queue, blocker.id, "running")

        jobs = []
        for name in ("p1", "p2", "p3", "p4"):
            job, _ = queue.submit(
                "move",
                lambda name=name: (order.append(name), {})[1],
                lane="premium",
            )
            jobs.append(job)
        for name in ("s1", "s2"):
            job, _ = queue.submit(
                "move",
                lambda name=name: (order.append(name), {})[1],
                lane="standard",
            )
            jobs.append(job)

        gate.set()
        for job in jobs:
            self.wait_for(queue, job.id, "completed")
        self.assertEqual(order, ["p1", "p2", "p3", "s1", "p4", "s2"])


if __name__ == "__main__":
    unittest.main()
