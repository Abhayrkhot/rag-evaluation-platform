"""Compare load-test runs (e.g. static vs adaptive routing at 50 QPS across seeds).

Latencies under load are autocorrelated (a queue that builds up slows many consecutive requests),
so confidence intervals use a block bootstrap: each run's steady-state window is cut into
fixed-length time blocks, blocks are resampled with replacement, and the percentile is recomputed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rageval.io import read_json, read_jsonl


def load_run(run_dir: Path, warmup_s: float = 30.0) -> dict[str, Any]:
    rows = [r for r in read_jsonl(run_dir / "load.jsonl") if r["at"] >= warmup_s]
    summary = read_json(run_dir / "load.summary.json")
    return {"rows": rows, "summary": summary, "name": run_dir.name}


def _blocks(rows: Sequence[dict[str, Any]], block_s: float) -> list[np.ndarray]:
    by: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        # failed requests count as infinitely slow so they cannot improve a percentile
        lat = r["latency_ms"] if r.get("status") == 200 else np.inf
        by[int(r["at"] // block_s)].append(lat)
    return [np.array(v) for _, v in sorted(by.items())]


def block_bootstrap_quantile(
    runs: Sequence[Sequence[dict[str, Any]]],
    q: float,
    block_s: float = 10.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap distribution of the pooled q-quantile over several runs (blocks resampled within each run)."""
    rng = np.random.default_rng(seed)
    blocks = [_blocks(r, block_s) for r in runs]
    out = np.empty(n_boot)
    for b in range(n_boot):
        parts: list[np.ndarray] = []
        for bl in blocks:
            idx = rng.integers(0, len(bl), size=len(bl))
            parts.extend(bl[i] for i in idx)
        out[b] = np.quantile(np.concatenate(parts), q)
    return out


def pooled_quantile(runs: Sequence[Sequence[dict[str, Any]]], q: float) -> float:
    lat = np.concatenate(
        [[r["latency_ms"] if r.get("status") == 200 else np.inf for r in run] for run in runs]
    )
    return float(np.quantile(lat, q))


def compare_modes(
    base: Sequence[dict[str, Any]], treat: Sequence[dict[str, Any]], block_s: float = 10.0
) -> dict[str, Any]:
    """base / treat: runs loaded with load_run (one per seed)."""
    b_rows = [r["rows"] for r in base]
    t_rows = [r["rows"] for r in treat]
    out: dict[str, Any] = {"seeds": len(base), "block_s": block_s}
    for name, q in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
        bq = pooled_quantile(b_rows, q)
        tq = pooled_quantile(t_rows, q)
        bb = block_bootstrap_quantile(b_rows, q, block_s, seed=1)
        tb = block_bootstrap_quantile(t_rows, q, block_s, seed=2)
        red = 1 - tb / bb
        out[name] = {
            "base_ms": bq,
            "treat_ms": tq,
            "reduction": 1 - tq / bq,
            "reduction_lo": float(np.quantile(red, 0.025)),
            "reduction_hi": float(np.quantile(red, 0.975)),
            "base_per_seed_ms": [
                float(np.quantile([x["latency_ms"] for x in r if x.get("status") == 200], q))
                for r in b_rows
            ],
            "treat_per_seed_ms": [
                float(np.quantile([x["latency_ms"] for x in r if x.get("status") == 200], q))
                for r in t_rows
            ],
        }
    for label, runs in (("base", base), ("treat", treat)):
        n = sum(len(r["rows"]) for r in runs)
        ok = sum(1 for r in runs for x in r["rows"] if x.get("status") == 200)
        routes: dict[str, int] = defaultdict(int)
        reasons: dict[str, int] = defaultdict(int)
        degraded: dict[str, int] = defaultdict(int)
        for r in runs:
            for x in r["rows"]:
                if x.get("status") == 200:
                    routes[str(x.get("route"))] += 1
                    reasons[f"{x.get('route')}:{x.get('route_reason')}"] += 1
                    for d in x.get("degraded") or []:
                        degraded[d] += 1
        out[label] = {
            "requests": n,
            "ok": ok,
            "error_rate": 1 - ok / max(n, 1),
            "offered_qps": float(np.mean([r["summary"]["offered_qps"] for r in runs])),
            "route_share": {k: v / max(ok, 1) for k, v in routes.items()},
            "route_reasons": dict(reasons),
            "degraded_share": {k: v / max(ok, 1) for k, v in degraded.items()},
        }
    return out


def node_breakdown(runs: Sequence[dict[str, Any]], q: float = 0.95) -> dict[str, float]:
    """Per-node latency quantiles from the server-reported timings (from the trace log fields)."""
    by: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for x in run["rows"]:
            for k, v in (x.get("timings_ms") or {}).items():
                by[k].append(v)
    return {k: float(np.quantile(v, q)) for k, v in sorted(by.items())}
