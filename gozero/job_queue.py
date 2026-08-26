"""Bounded background job queue for inference requests.

The HTTP layer submits work and returns immediately.  A small, fixed worker
pool is the only code allowed to start inference, so overload becomes visible
queueing instead of a pile of blocked request threads.
"""
from __future__ import annotations

import math
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


class QueueFull(Exception):
    """The bounded pending queue has no room for another job."""


class PublicJobError(Exception):
    """An expected job failure that is safe to return to the client."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class Job:
    id: str
    operation: str
    work: Callable[[], dict] = field(repr=False)
    lane: str = "standard"
    request_id: str | None = None
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None
    status_code: int = 200


class InferenceJobQueue:
    """FIFO inference queue with a future-ready, starvation-safe premium lane.

    Only trusted server code may select ``lane="premium"``.  The public HTTP
    API deliberately has no client-controlled priority flag until verified
    purchase entitlements exist.
    """

    LANES = ("standard", "premium")

    def __init__(
        self,
        workers: int,
        max_pending: int,
        *,
        result_ttl: float = 15 * 60,
        initial_duration: float = 2.0,
        premium_weight: int = 3,
        start_workers: bool = True,
    ):
        if workers < 1:
            raise ValueError("workers must be positive")
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        if premium_weight < 1:
            raise ValueError("premium_weight must be positive")
        self.workers = workers
        self.max_pending = max_pending
        self.result_ttl = result_ttl
        self.premium_weight = premium_weight
        self._duration_ema = max(0.05, initial_duration)
        self._pending: dict[str, deque[str]] = {
            lane: deque() for lane in self.LANES
        }
        self._jobs: dict[str, Job] = {}
        self._request_jobs: dict[str, str] = {}
        self._active = 0
        self._premium_streak = 0
        self._stopping = False
        self._condition = threading.Condition()
        self._threads: list[threading.Thread] = []
        if start_workers:
            for index in range(workers):
                thread = threading.Thread(
                    target=self._worker,
                    daemon=True,
                    name=f"inference-{index + 1}",
                )
                thread.start()
                self._threads.append(thread)

    def submit(
        self,
        operation: str,
        work: Callable[[], dict],
        *,
        request_id: str | None = None,
        lane: str = "standard",
    ) -> tuple[Job, bool]:
        if lane not in self._pending:
            raise ValueError(f"unknown queue lane: {lane}")
        now = time.time()
        with self._condition:
            self._cleanup_locked(now)
            if request_id:
                key = f"{operation}:{request_id}"
                existing_id = self._request_jobs.get(key)
                if existing_id is not None:
                    existing = self._jobs.get(existing_id)
                    if existing is not None:
                        return existing, False
                    self._request_jobs.pop(key, None)
            if self._pending_count_locked() >= self.max_pending:
                raise QueueFull
            job = Job(
                id=uuid.uuid4().hex,
                operation=operation,
                work=work,
                lane=lane,
                request_id=request_id,
            )
            self._jobs[job.id] = job
            self._pending[lane].append(job.id)
            if request_id:
                self._request_jobs[f"{operation}:{request_id}"] = job.id
            self._condition.notify()
            return job, True

    def view(self, job_id: str) -> dict | None:
        with self._condition:
            self._cleanup_locked(time.time())
            job = self._jobs.get(job_id)
            if job is None:
                return None
            position = self._position_locked(job_id) if job.status == "queued" else None
            return self._view_locked(job, position)

    def wait(self, job_id: str, timeout: float) -> dict | None:
        """Wait for a job to finish, used only by pre-queue legacy clients."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    return None
                if job.status not in ("queued", "running"):
                    return self._view_locked(job, None)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    position = (
                        self._position_locked(job_id) if job.status == "queued" else None
                    )
                    return self._view_locked(job, position)
                self._condition.wait(remaining)

    def status(self) -> dict:
        with self._condition:
            self._cleanup_locked(time.time())
            queued = self._pending_count_locked()
            return {
                "workers": self.workers,
                "active": self._active,
                "queued": queued,
                "queued_standard": len(self._pending["standard"]),
                "queued_premium": len(self._pending["premium"]),
                "queue_capacity": self.max_pending,
                "average_job_seconds": round(self._duration_ema, 2),
                "estimated_tail_wait_seconds": self._estimate_wait_locked(queued),
            }

    def stop(self, timeout: float = 2.0):
        """Stop workers after their current job.  Intended for tests."""
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        deadline = time.time() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.time()))

    def _worker(self):
        while True:
            with self._condition:
                while not self._stopping and self._pending_count_locked() == 0:
                    self._condition.wait()
                if self._stopping:
                    return
                job_id = self._pop_next_locked()
                job = self._jobs[job_id]
                job.status = "running"
                job.started_at = time.time()
                self._active += 1
            try:
                result = job.work()
            except PublicJobError as err:
                with self._condition:
                    job.status = "failed"
                    job.error = err.message
                    job.status_code = err.status_code
            except Exception:
                traceback.print_exc()
                with self._condition:
                    job.status = "failed"
                    job.error = "internal error"
                    job.status_code = 500
            else:
                with self._condition:
                    job.status = "completed"
                    job.result = result
                    job.status_code = 200
            finally:
                with self._condition:
                    job.finished_at = time.time()
                    self._active -= 1
                    duration = max(0.0, job.finished_at - (job.started_at or job.finished_at))
                    self._duration_ema = 0.8 * self._duration_ema + 0.2 * duration
                    self._condition.notify_all()

    def _pending_count_locked(self) -> int:
        return sum(len(items) for items in self._pending.values())

    def _pop_next_locked(self) -> str:
        premium = self._pending["premium"]
        standard = self._pending["standard"]
        if premium and (not standard or self._premium_streak < self.premium_weight):
            self._premium_streak += 1
            return premium.popleft()
        if standard:
            self._premium_streak = 0
            return standard.popleft()
        self._premium_streak += 1
        return premium.popleft()

    def _dispatch_order_locked(self) -> list[str]:
        """Return a simulated dispatch order without mutating the real queues."""
        premium = deque(self._pending["premium"])
        standard = deque(self._pending["standard"])
        streak = self._premium_streak
        order: list[str] = []
        while premium or standard:
            if premium and (not standard or streak < self.premium_weight):
                order.append(premium.popleft())
                streak += 1
            elif standard:
                order.append(standard.popleft())
                streak = 0
            else:
                order.append(premium.popleft())
                streak += 1
        return order

    def _position_locked(self, job_id: str) -> int | None:
        try:
            return self._dispatch_order_locked().index(job_id) + 1
        except ValueError:
            return None

    def _estimate_wait_locked(self, position: int) -> int:
        if position <= 0:
            return 0
        return max(1, math.ceil(position / self.workers * self._duration_ema))

    def _view_locked(self, job: Job, position: int | None) -> dict:
        view: dict = {
            "job_id": job.id,
            "operation": job.operation,
            "status": job.status,
        }
        if job.status == "queued":
            view["queue_position"] = position
            view["estimated_wait_seconds"] = self._estimate_wait_locked(position or 1)
        elif job.status == "running":
            view["queue_position"] = 0
            view["estimated_wait_seconds"] = 0
        elif job.status == "completed":
            view["result"] = job.result
        elif job.status == "failed":
            view["error"] = job.error
            view["http_status"] = job.status_code
        return view

    def _cleanup_locked(self, now: float):
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None and now - job.finished_at > self.result_ttl
        ]
        for job_id in expired:
            job = self._jobs.pop(job_id)
            if job.request_id:
                self._request_jobs.pop(f"{job.operation}:{job.request_id}", None)
