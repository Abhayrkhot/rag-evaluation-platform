"""Retrieval metrics over ranked doc-id lists (binary relevance)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold documents in the top k (the BEIR / TREC definition)."""
    g = set(gold)
    if not g:
        return float("nan")
    return len(g.intersection(ranked[:k])) / len(g)


def all_recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1 if every gold document is in the top k (both hops of a multi-hop question)."""
    g = set(gold)
    if not g:
        return float("nan")
    return float(g.issubset(ranked[:k]))


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    if not g:
        return float("nan")
    return float(bool(g.intersection(ranked[:k])))


def mrr_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    if not g:
        return float("nan")
    for i, d in enumerate(ranked[:k], start=1):
        if d in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    g = set(gold)
    if not g:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in g)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(g), k) + 1))
    return dcg / idcg


METRICS = {
    "recall": recall_at_k,
    "all_recall": all_recall_at_k,
    "hit": hit_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def per_question(
    ranked: Sequence[str], gold: Sequence[str], ks: Sequence[int] = (1, 5, 10, 20, 50, 100)
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"all_recall@{k}"] = all_recall_at_k(ranked, gold, k)
    out["mrr@10"] = mrr_at_k(ranked, gold, 10)
    out["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
    out["hit@10"] = hit_at_k(ranked, gold, 10)
    return out
