"""Pipeline configuration and component loading shared by offline evaluation and serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from rageval.retrieval.hybrid import RetrievalConfig


class PipelineConfig(BaseModel):
    name: str
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    context_k: int = 10  # documents shown to the generator
    prompt: str = "zero_shot"  # naive | zero_shot | few_shot | cot
    verify: str = "none"  # none | repair | prune
    verify_tau: float = 0.5
    temperature: float = 0.0
    max_tokens: int | None = None


class ModelsConfig(BaseModel):
    corpus: str = "data/hotpot100k/corpus.jsonl"
    index_dir: str = "artifacts/index/hotpot100k"
    embedder: str = "Qwen/Qwen3-Embedding-0.6B"
    embedder_url: str | None = None
    dense_index: str = "hnsw"  # flat | hnsw
    reranker: str | None = "BAAI/bge-reranker-v2-m3"
    reranker_url: str | None = None
    verifier: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    generator: str = "Qwen/Qwen3-8B"
    generator_url: str = "http://127.0.0.1:8000"
    fallback: str | None = "Qwen/Qwen3-1.7B"
    fallback_url: str | None = "http://127.0.0.1:8001"


def load_pipelines(path: str | Path) -> list[PipelineConfig]:
    spec = yaml.safe_load(Path(path).read_text())
    base: dict[str, Any] = spec.get("base", {})
    out = []
    for name, over in spec["pipelines"].items():
        cfg = _deep_merge(base, over or {})
        cfg["name"] = name
        out.append(PipelineConfig.model_validate(cfg))
    return out


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_retriever(
    models: ModelsConfig, config: RetrievalConfig | None = None, device: str | None = None
) -> Any:
    from rageval.corpus import Corpus
    from rageval.retrieval.bm25 import BM25Index
    from rageval.retrieval.dense import DenseIndex, make_embedder
    from rageval.retrieval.experiments import slug
    from rageval.retrieval.hybrid import HybridRetriever
    from rageval.retrieval.rerank import make_reranker

    corpus = Corpus.load(models.corpus)
    idx = Path(models.index_dir)
    bm25 = BM25Index.load(idx / "bm25")
    dense = DenseIndex.load(idx / "dense" / slug(models.embedder) / models.dense_index)
    kw: dict[str, Any] = {} if models.embedder_url else {"device": device}
    emb = make_embedder(models.embedder, models.embedder_url, **kw)
    rr = None
    if models.reranker:
        kw = {} if models.reranker_url else {"device": device}
        rr = make_reranker(models.reranker, models.reranker_url, **kw)
    return HybridRetriever(corpus, bm25, dense, emb, rr, config)
