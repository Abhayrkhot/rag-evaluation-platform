"""Open-loop load generator.

Requests are released on a precomputed Poisson schedule (exponential inter-arrival times at the
target rate), independent of how fast earlier requests complete, so a slow server cannot slow
the offered load down (no coordinated omission). Latency is measured from each request's
scheduled release time.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np


def poisson_schedule(qps: float, duration_s: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    t, out = 0.0, list[float]()
    while True:
        t += rng.expovariate(qps)
        if t >= duration_s:
            return out
        out.append(t)


async def run_load(
    url: str,
    questions: Sequence[tuple[str, str]],  # (qid, question)
    qps: float,
    duration_s: float,
    warmup_s: float = 30.0,
    seed: int = 0,
    budget_ms: float | None = None,
    timeout_s: float = 120.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sched = poisson_schedule(qps, duration_s, seed)
    rng = random.Random(seed + 1)
    picks = [questions[rng.randrange(len(questions))] for _ in sched]
    limits = httpx.Limits(max_connections=4096, max_keepalive_connections=4096)
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        start = time.perf_counter() + 0.5

        async def one(i: int, at: float, qid: str, q: str) -> None:
            delay = start + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            sent = time.perf_counter()
            row: dict[str, Any] = {
                "i": i,
                "qid": qid,
                "at": at,
                "send_lag_ms": (sent - start - at) * 1e3,
            }
            body: dict[str, Any] = {"question": q, "request_id": f"lt{seed}-{i}"}
            if budget_ms is not None:
                body["budget_ms"] = budget_ms
            try:
                r = await client.post(url, json=body)
                row["status"] = r.status_code
                if r.status_code == 200:
                    d = r.json()
                    row.update(
                        route=d.get("route"),
                        route_reason=d.get("route_reason"),
                        degraded=d.get("degraded"),
                        server_ms=d.get("latency_ms"),
                        timings_ms=d.get("timings_ms"),
                        abstained=d.get("abstained"),
                        answer=d.get("answer"),
                        statements=d.get("statements"),
                        context=d.get("context"),
                    )
            except httpx.HTTPError as e:
                row["status"] = -1
                row["error"] = type(e).__name__
            row["latency_ms"] = (time.perf_counter() - (start + at)) * 1e3
            rows.append(row)

        t0 = time.perf_counter()
        await asyncio.gather(
            *(one(i, at, qid, q) for i, (at, (qid, q)) in enumerate(zip(sched, picks, strict=True)))
        )
        wall = time.perf_counter() - t0
    rows.sort(key=lambda r: r["i"])
    return rows, summarize(rows, qps, duration_s, warmup_s, wall)


def summarize(
    rows: Sequence[dict[str, Any]],
    qps: float,
    duration_s: float,
    warmup_s: float,
    wall_s: float | None = None,
) -> dict[str, Any]:
    steady = [r for r in rows if r["at"] >= warmup_s]
    ok = [r for r in steady if r.get("status") == 200]
    lat = np.array([r["latency_ms"] for r in ok]) if ok else np.array([np.nan])
    routes = Counter(r.get("route") for r in ok)
    reasons = Counter(f"{r.get('route')}:{r.get('route_reason')}" for r in ok)
    degraded = Counter(d for r in ok for d in (r.get("degraded") or []))
    window = duration_s - warmup_s
    return {
        "target_qps": qps,
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "offered": len(steady),
        "offered_qps": len(steady) / window if window > 0 else None,
        "completed_ok": len(ok),
        "errors": len(steady) - len(ok),
        "error_rate": (len(steady) - len(ok)) / max(len(steady), 1),
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_ms": float(np.mean(lat)),
        "max_send_lag_ms": float(max((r["send_lag_ms"] for r in steady), default=0.0)),
        "route_share": {k: v / max(len(ok), 1) for k, v in routes.items()},
        "route_reasons": dict(reasons),
        "degraded_share": {k: v / max(len(ok), 1) for k, v in degraded.items()},
        "wall_s": wall_s,
    }
