"""Hybrid retriever: BM25 + FAISS candidates, rank fusion, cross-encoder reranking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel

from rageval.corpus import Corpus
from rageval.retrieval.bm25 import BM25Index
from rageval.retrieval.dense import DenseIndex, Embedder, format_query
from rageval.retrieval.fusion import fuse
from rageval.retrieval.rerank import Reranker
from rageval.types import Hit


class RetrievalConfig(BaseModel):
    bm25_k: int = 100
    dense_k: int = 100
    fusion: str = "rrf"  # rrf | wrrf | weighted | bm25 | dense
    rrf_k: int = 60
    alpha: float = 0.5
    rerank: bool = True
    rerank_depth: int = 50
    final_k: int = 10
    ef_search: int = 128
    # Final score = reranker score, optionally blended with fused rank (0 = pure reranker).
    rerank_blend: float = 0.0
    # Two-hop retrieval (rageval.retrieval.multihop): 1 = single shot.
    hops: int = 1
    plan_k: int = 5
    hop2_depth: int = 30
    hop2_keep: int = 3
    hop2_take: int = 3


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    candidates: list[Hit] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def doc_ids(self) -> list[str]:
        return [h.doc_id for h in self.hits]


def blend_rerank(fused: list[Hit], scores: np.ndarray, blend: float) -> list[Hit]:
    """Order candidates by reranker score; with blend > 0, mix in the fused rank as 1/(1+rank)."""
    order = np.argsort(-scores, kind="stable")
    if blend <= 0:
        return [Hit(fused[i].doc_id, float(scores[i]), "rerank") for i in order]
    r_rank = np.empty(len(scores))
    r_rank[order] = np.arange(len(scores))
    combined = (1 - blend) / (1 + r_rank) + blend / (1 + np.arange(len(scores)))
    order2 = np.argsort(-combined, kind="stable")
    return [Hit(fused[i].doc_id, float(combined[i]), "rerank") for i in order2]


class HybridRetriever:
    def __init__(
        self,
        corpus: Corpus,
        bm25: BM25Index,
        dense: DenseIndex | None,
        embedder: Embedder | None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def retrieve_batch(self, queries: list[str], **overrides: Any) -> list[RetrievalResult]:
        cfg = self.config.model_copy(update=overrides) if overrides else self.config
        t0 = time.perf_counter()
        sparse = (
            self.bm25.search(queries, k=cfg.bm25_k)
            if cfg.fusion != "dense"
            else [[] for _ in queries]
        )
        t1 = time.perf_counter()
        if cfg.fusion != "bm25" and self.dense is not None and self.embedder is not None:
            qv = self.embedder.embed([format_query(self.embedder.model, q) for q in queries])
            t2 = time.perf_counter()
            dense = self.dense.search(qv, k=cfg.dense_k, ef_search=cfg.ef_search)
        else:
            t2 = time.perf_counter()
            dense = [[] for _ in queries]
        t3 = time.perf_counter()
        n = len(queries)
        results = []
        for q, s, d in zip(queries, sparse, dense, strict=True):
            fused = fuse(s, d, cfg.fusion, cfg.rrf_k, cfg.alpha)
            results.append(RetrievalResult(q, fused[: cfg.final_k], fused))
        t4 = time.perf_counter()
        timing = {
            "bm25": (t1 - t0) * 1e3 / n,
            "embed": (t2 - t1) * 1e3 / n,
            "faiss": (t3 - t2) * 1e3 / n,
            "fusion": (t4 - t3) * 1e3 / n,
        }
        if cfg.rerank and self.reranker is not None:
            pairs: list[tuple[str, str]] = []
            spans = []
            for r in results:
                cand = r.candidates[: cfg.rerank_depth]
                spans.append((len(pairs), len(cand)))
                pairs.extend((r.query, self.corpus.index_text(h.doc_id)) for h in cand)
            t5 = time.perf_counter()
            scores = self.reranker.score(pairs)
            t6 = time.perf_counter()
            for r, (start, m) in zip(results, spans, strict=True):
                ranked = blend_rerank(r.candidates[:m], scores[start : start + m], cfg.rerank_blend)
                r.hits = ranked[: cfg.final_k]
            timing["rerank"] = (t6 - t5) * 1e3 / n
        for r in results:
            r.timings_ms = dict(timing)
        return results

    def retrieve(self, query: str, **overrides: Any) -> RetrievalResult:
        return self.retrieve_batch([query], **overrides)[0]
