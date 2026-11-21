"""Rank fusion of sparse and dense result lists."""

from __future__ import annotations

from collections.abc import Sequence

from rageval.types import Hit


def rrf(
    lists: Sequence[Sequence[Hit]], k: int = 60, weights: Sequence[float] | None = None
) -> list[Hit]:
    """Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum_i w_i / (k + rank_i(d))."""
    weights = weights or [1.0] * len(lists)
    scores: dict[str, float] = {}
    for w, hits in zip(weights, lists, strict=True):
        for rank, h in enumerate(hits, start=1):
            scores[h.doc_id] = scores.get(h.doc_id, 0.0) + w / (k + rank)
    return [Hit(d, s, "fused") for d, s in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def _minmax(hits: Sequence[Hit]) -> dict[str, float]:
    if not hits:
        return {}
    vals = [h.score for h in hits]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return {h.doc_id: 1.0 for h in hits}
    return {h.doc_id: (h.score - lo) / span for h in hits}


def weighted(sparse: Sequence[Hit], dense: Sequence[Hit], alpha: float = 0.5) -> list[Hit]:
    """Convex combination of min-max normalised scores: alpha * dense + (1 - alpha) * sparse.

    A document missing from one list gets 0 for that list (it scored below that list's cutoff).
    """
    s, d = _minmax(sparse), _minmax(dense)
    keys = set(s) | set(d)
    scored = {k: alpha * d.get(k, 0.0) + (1 - alpha) * s.get(k, 0.0) for k in keys}
    return [Hit(k, v, "fused") for k, v in sorted(scored.items(), key=lambda x: (-x[1], x[0]))]


def fuse(
    sparse: Sequence[Hit],
    dense: Sequence[Hit],
    method: str = "rrf",
    rrf_k: int = 60,
    alpha: float = 0.5,
) -> list[Hit]:
    if method == "rrf":
        return rrf([sparse, dense], k=rrf_k)
    if method == "wrrf":
        return rrf([sparse, dense], k=rrf_k, weights=[1 - alpha, alpha])
    if method == "weighted":
        return weighted(sparse, dense, alpha=alpha)
    if method == "bm25":
        return list(sparse)
    if method == "dense":
        return list(dense)
    raise ValueError(f"unknown fusion method {method}")
