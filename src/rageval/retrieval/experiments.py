"""Offline retrieval experiments.

Expensive work is cached so a grid of fusion / reranking configurations can be scored cheaply:
  1. index:      BM25 + document embeddings + FAISS (flat and HNSW) per embedder
  2. candidates: per question, BM25 top-N and dense top-N (exact and HNSW) with scores
  3. rerank:     cross-encoder scores for every (question, doc) pair any grid config needs
  4. sweep:      fuse + rerank from the caches, score with retrieval metrics
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np

from rageval.corpus import Corpus
from rageval.eval.retrieval_metrics import per_question
from rageval.eval.stats import bootstrap_ci
from rageval.io import read_json, read_jsonl, write_json, write_jsonl
from rageval.retrieval.bm25 import BM25Index
from rageval.retrieval.dense import DenseIndex, Embedder, format_doc, format_query
from rageval.retrieval.fusion import fuse
from rageval.retrieval.hybrid import blend_rerank
from rageval.retrieval.rerank import Reranker
from rageval.types import Hit, Question


def slug(model: str) -> str:
    return model.split("/")[-1].lower()


# ---------------------------------------------------------------- 1. indexes


def build_bm25(corpus: Corpus, out: Path, k1: float = 0.9, b: float = 0.4) -> dict[str, Any]:
    t0 = time.perf_counter()
    idx = BM25Index(k1=k1, b=b).build(corpus.ids, [corpus.index_text(i) for i in corpus.ids])
    idx.save(out)
    return {"bm25_build_seconds": round(time.perf_counter() - t0, 2), "k1": k1, "b": b}


def build_dense(
    corpus: Corpus, embedder: Embedder, out: Path, batch: int = 4096, hnsw_m: int = 32
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    texts = [format_doc(embedder.model, d.title, d.text) for d in corpus]
    t0 = time.perf_counter()
    parts = [embedder.embed(texts[i : i + batch]) for i in range(0, len(texts), batch)]
    vecs = np.concatenate(parts).astype(np.float32)
    embed_s = time.perf_counter() - t0
    np.save(out / "embeddings.npy", vecs.astype(np.float16))
    flat = DenseIndex("flat").build(corpus.ids, vecs, embedder=embedder.model)
    flat.save(out / "flat")
    hnsw = DenseIndex("hnsw", m=hnsw_m).build(corpus.ids, vecs, embedder=embedder.model)
    hnsw.save(out / "hnsw")
    meta = {
        "embedder": embedder.model,
        "n_docs": len(texts),
        "dim": int(vecs.shape[1]),
        "embed_seconds": round(embed_s, 2),
        "docs_per_second": round(len(texts) / embed_s, 1),
        "hnsw_build_seconds": hnsw.meta["build_seconds"],
        "hnsw_m": hnsw_m,
    }
    write_json(out / "build.json", meta)
    return meta


# ---------------------------------------------------------------- 2. candidates


def compute_candidates(
    questions: list[Question],
    index_dir: Path,
    embedder: Embedder,
    k: int = 200,
    ef_search: int = 256,
) -> list[dict[str, Any]]:
    """BM25 and dense (exact + HNSW) top-k for each question, plus per-stage latency."""
    bm25 = BM25Index.load(index_dir / "bm25")
    dense_dir = index_dir / "dense" / slug(embedder.model)
    flat = DenseIndex.load(dense_dir / "flat")
    hnsw = DenseIndex.load(dense_dir / "hnsw")
    qs = [q.question for q in questions]
    t0 = time.perf_counter()
    sparse = bm25.search(qs, k=k)
    t1 = time.perf_counter()
    qv = embedder.embed([format_query(embedder.model, q) for q in qs])
    t2 = time.perf_counter()
    ex = flat.search(qv, k=k)
    t3 = time.perf_counter()
    ann = hnsw.search(qv, k=k, ef_search=ef_search)
    t4 = time.perf_counter()
    n = max(len(qs), 1)
    timing = {
        "bm25_ms_per_q_batched": (t1 - t0) * 1e3 / n,
        "embed_ms_per_q_batched": (t2 - t1) * 1e3 / n,
        "flat_ms_per_q_batched": (t3 - t2) * 1e3 / n,
        "hnsw_ms_per_q_batched": (t4 - t3) * 1e3 / n,
    }
    rows: list[dict[str, Any]] = []
    for q, s, e, a in zip(questions, sparse, ex, ann, strict=True):
        rows.append(
            {
                "qid": q.qid,
                "bm25": [[h.doc_id, h.score] for h in s],
                "flat": [[h.doc_id, h.score] for h in e],
                "hnsw": [[h.doc_id, h.score] for h in a],
            }
        )
    if rows:
        rows[0]["_timing"] = timing
    return rows


def load_candidates(path: Path) -> dict[str, dict[str, list[Hit]]]:
    out: dict[str, dict[str, list[Hit]]] = {}
    for r in read_jsonl(path):
        out[r["qid"]] = {
            name: [Hit(d, float(s), name) for d, s in r[name]] for name in ("bm25", "flat", "hnsw")
        }
    return out


# ---------------------------------------------------------------- 3. rerank cache


class RerankCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scores: dict[tuple[str, str], float] = {}
        if path.exists():
            for r in read_jsonl(path):
                self.scores[(r["qid"], r["doc_id"])] = float(r["score"])

    def missing(self, pairs: set[tuple[str, str]]) -> list[tuple[str, str]]:
        return sorted(p for p in pairs if p not in self.scores)

    def fill(
        self,
        pairs: list[tuple[str, str]],
        reranker: Reranker,
        qtext: dict[str, str],
        corpus: Corpus,
        chunk: int = 20_000,
    ) -> float:
        t0 = time.perf_counter()
        for i in range(0, len(pairs), chunk):
            part = pairs[i : i + chunk]
            s = reranker.score([(qtext[q], corpus.index_text(d)) for q, d in part])
            for (q, d), v in zip(part, s, strict=True):
                self.scores[(q, d)] = float(v)
        dt = time.perf_counter() - t0
        write_jsonl(
            self.path,
            ({"qid": q, "doc_id": d, "score": v} for (q, d), v in sorted(self.scores.items())),
        )
        return dt


# ---------------------------------------------------------------- 4. sweep


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    return [
        dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))
    ]


def rank_one(
    cands: dict[str, list[Hit]],
    cfg: dict[str, Any],
    rerank: dict[tuple[str, str], float] | None,
    qid: str,
    final_k: int = 100,
) -> list[str]:
    dense = cands[cfg.get("dense_index", "flat")]
    fused = fuse(
        cands["bm25"][: cfg.get("bm25_k", 100)],
        dense[: cfg.get("dense_k", 100)],
        cfg.get("fusion", "rrf"),
        cfg.get("rrf_k", 60),
        cfg.get("alpha", 0.5),
    )
    depth = int(cfg.get("rerank_depth", 0))
    if depth > 0 and rerank is not None:
        head = fused[:depth]
        scores = np.array([rerank[(qid, h.doc_id)] for h in head], dtype=np.float32)
        ranked = blend_rerank(head, scores, float(cfg.get("rerank_blend", 0.0)))
        rest = fused[depth:]
        return [h.doc_id for h in ranked + rest][:final_k]
    return [h.doc_id for h in fused][:final_k]


def needed_pairs(
    questions: list[Question], cands: dict[str, dict[str, list[Hit]]], configs: list[dict[str, Any]]
) -> set[tuple[str, str]]:
    need: set[tuple[str, str]] = set()
    for cfg in configs:
        depth = int(cfg.get("rerank_depth", 0))
        if depth <= 0:
            continue
        for q in questions:
            c = cands[q.qid]
            fused = fuse(
                c["bm25"][: cfg.get("bm25_k", 100)],
                c[cfg.get("dense_index", "flat")][: cfg.get("dense_k", 100)],
                cfg.get("fusion", "rrf"),
                cfg.get("rrf_k", 60),
                cfg.get("alpha", 0.5),
            )
            need.update((q.qid, h.doc_id) for h in fused[:depth])
    return need


def score_config(
    questions: list[Question],
    cands: dict[str, dict[str, list[Hit]]],
    cfg: dict[str, Any],
    rerank: dict[tuple[str, str], float] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for q in questions:
        if not q.answerable:
            continue
        ranked = rank_one(cands[q.qid], cfg, rerank, q.qid)
        m = per_question(ranked, q.gold_doc_ids)
        rows.append({"qid": q.qid, "qtype": q.qtype, **m, "top10": ranked[:10]})
    summary: dict[str, Any] = {"config": cfg, "n": len(rows)}
    for key in (
        "recall@10",
        "all_recall@10",
        "ndcg@10",
        "mrr@10",
        "recall@5",
        "recall@20",
        "recall@100",
    ):
        summary[key] = bootstrap_ci([r[key] for r in rows]).to_dict()
    for t in ("bridge", "comparison"):
        sub = [r["recall@10"] for r in rows if r["qtype"] == t]
        if sub:
            summary[f"recall@10/{t}"] = bootstrap_ci(sub).to_dict()
    return summary, rows


def ann_recall(cands: dict[str, dict[str, list[Hit]]], k: int = 10) -> float:
    """Overlap of HNSW top-k with exact top-k, averaged over questions."""
    vals = []
    for c in cands.values():
        ex = {h.doc_id for h in c["flat"][:k]}
        ap = {h.doc_id for h in c["hnsw"][:k]}
        if ex:
            vals.append(len(ex & ap) / len(ex))
    return float(np.mean(vals)) if vals else float("nan")


def candidate_timing(path: Path) -> dict[str, float]:
    for r in read_jsonl(path):
        return dict(r.get("_timing", {}))
    return {}


def load_build_meta(index_dir: Path, model: str) -> dict[str, Any]:
    p = index_dir / "dense" / slug(model) / "build.json"
    return dict(read_json(p)) if p.exists() else {}
