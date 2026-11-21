"""Dynamic micro-batching for in-process GPU encoders.

Concurrent requests each submit a small list of items (a query to embed, query-passage pairs to
rerank, premise-hypothesis pairs to verify). A single worker thread drains the queue, waits at
most `max_wait_ms` to fill up to `max_items`, runs one batched forward pass, and resolves every
caller's future with its slice. This is what dedicated inference servers (TEI, Triton) do; doing
it in-process avoids a network hop per stage.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _Job:
    items: Sequence[Any]
    future: asyncio.Future[np.ndarray]
    loop: asyncio.AbstractEventLoop
    enqueued: float


class MicroBatcher:
    def __init__(
        self,
        fn: Callable[[list[Any]], np.ndarray],
        name: str,
        max_items: int = 256,
        max_wait_ms: float = 4.0,
    ) -> None:
        self.fn = fn
        self.name = name
        self.max_items = max_items
        self.max_wait = max_wait_ms / 1e3
        self._q: queue.Queue[_Job | None] = queue.Queue()
        self.batches = 0
        self.items = 0
        self.busy_s = 0.0
        self._thread = threading.Thread(target=self._worker, name=f"batcher-{name}", daemon=True)
        self._thread.start()

    async def submit(self, items: Sequence[Any]) -> np.ndarray:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[np.ndarray] = loop.create_future()
        if not items:
            fut.set_result(np.zeros((0,), dtype=np.float32))
            return await fut
        self._q.put(_Job(items, fut, loop, time.perf_counter()))
        return await fut

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            jobs = [job]
            n = len(job.items)
            deadline = time.perf_counter() + self.max_wait
            while n < self.max_items:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self._q.put(None)
                    break
                jobs.append(nxt)
                n += len(nxt.items)
            flat = [x for j in jobs for x in j.items]
            t0 = time.perf_counter()
            try:
                out = self.fn(flat)
                err: BaseException | None = None
            except BaseException as e:
                out, err = None, e
            self.busy_s += time.perf_counter() - t0
            self.batches += 1
            self.items += len(flat)
            start = 0
            for j in jobs:
                k = len(j.items)
                if err is not None:
                    j.loop.call_soon_threadsafe(_set_exc, j.future, err)
                else:
                    assert out is not None
                    j.loop.call_soon_threadsafe(_set_result, j.future, out[start : start + k])
                start += k


def _set_result(f: asyncio.Future[np.ndarray], v: np.ndarray) -> None:
    if not f.done():
        f.set_result(v)


def _set_exc(f: asyncio.Future[np.ndarray], e: BaseException) -> None:
    if not f.done():
        f.set_exception(e)
