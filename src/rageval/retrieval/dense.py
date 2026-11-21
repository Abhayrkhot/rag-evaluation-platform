"""Dense retrieval: embedders (in-process or OpenAI-compatible server) and FAISS indexes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import httpx
import numpy as np

from rageval.types import Hit

RETRIEVAL_TASK = "Given a multi-hop question, retrieve Wikipedia paragraphs that help answer it"


@dataclass(frozen=True)
class EmbedFormat:
    query_prefix: str = ""
    doc_prefix: str = ""
    instruct: bool = False  # Qwen3-Embedding style "Instruct: ...\nQuery:..."


FORMATS: dict[str, EmbedFormat] = {
    "Qwen/Qwen3-Embedding-0.6B": EmbedFormat(instruct=True),
    "Qwen/Qwen3-Embedding-4B": EmbedFormat(instruct=True),
    "BAAI/bge-small-en-v1.5": EmbedFormat(
        "Represent this sentence for searching relevant passages: "
    ),
    "BAAI/bge-base-en-v1.5": EmbedFormat(
        "Represent this sentence for searching relevant passages: "
    ),
    "BAAI/bge-large-en-v1.5": EmbedFormat(
        "Represent this sentence for searching relevant passages: "
    ),
    "Snowflake/snowflake-arctic-embed-m-v1.5": EmbedFormat(
        "Represent this sentence for searching relevant passages: "
    ),
    "intfloat/e5-base-v2": EmbedFormat("query: ", "passage: "),
    "intfloat/e5-large-v2": EmbedFormat("query: ", "passage: "),
}


def format_query(model: str, q: str) -> str:
    f = FORMATS.get(model, EmbedFormat())
    if f.instruct:
        return f"Instruct: {RETRIEVAL_TASK}\nQuery:{q}"
    return f.query_prefix + q


def format_doc(model: str, title: str, text: str) -> str:
    f = FORMATS.get(model, EmbedFormat())
    return f"{f.doc_prefix}{title}\n{text}"


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return np.ascontiguousarray(x / n)


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class LocalEmbedder:
    """Transformer bi-encoder in this process (CPU or GPU); see rageval.models.HFEmbedder."""

    def __init__(self, model: str, **kw: Any) -> None:
        from rageval.models import HFEmbedder

        self.model = model
        self._enc = HFEmbedder(model, **kw)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._enc.embed(texts)


class RemoteEmbedder:
    """OpenAI-compatible /v1/embeddings (vLLM pooling runner)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        batch_size: int = 128,
        timeout: float = 60.0,
        served_name: str | None = None,
    ) -> None:
        self.model = model
        self.served_name = served_name or model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=timeout)
        self._aclient: httpx.AsyncClient | None = None
        self._timeout = timeout

    def _payload(self, texts: list[str]) -> dict[str, Any]:
        return {"model": self.served_name, "input": texts, "encoding_format": "float"}

    @staticmethod
    def _parse(data: dict[str, Any]) -> np.ndarray:
        rows = sorted(data["data"], key=lambda r: r["index"])
        return _normalize(np.array([r["embedding"] for r in rows], dtype=np.float32))

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            r = self._client.post(
                f"{self.base_url}/v1/embeddings", json=self._payload(texts[i : i + self.batch_size])
            )
            r.raise_for_status()
            out.append(self._parse(r.json()))
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)

    async def aembed(self, texts: list[str]) -> np.ndarray:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(
                timeout=self._timeout, limits=httpx.Limits(max_connections=256)
            )
        r = await self._aclient.post(f"{self.base_url}/v1/embeddings", json=self._payload(texts))
        r.raise_for_status()
        return self._parse(r.json())


def make_embedder(model: str, base_url: str | None = None, **kw: Any) -> Embedder:
    if base_url:
        return RemoteEmbedder(model, base_url, **kw)
    return LocalEmbedder(model, **kw)


class DenseIndex:
    """FAISS inner-product index over L2-normalised embeddings (cosine similarity).

    kind = "flat" (exact) or "hnsw" (graph ANN; recall vs exact is reported in M1).
    """

    def __init__(self, kind: str = "hnsw", m: int = 32, ef_construction: int = 200) -> None:
        self.kind = kind
        self.m = m
        self.ef_construction = ef_construction
        self.index: Any = None
        self.doc_ids: list[str] = []
        self.meta: dict[str, Any] = {}

    def build(self, doc_ids: list[str], vecs: np.ndarray, **meta: Any) -> DenseIndex:
        vecs = _normalize(vecs)
        d = vecs.shape[1]
        index: Any
        if self.kind == "flat":
            index = faiss.IndexFlatIP(d)
        elif self.kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, self.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ef_construction
        else:
            raise ValueError(f"unknown index kind {self.kind}")
        t0 = time.perf_counter()
        index.add(vecs)
        self.index = index
        self.doc_ids = list(doc_ids)
        self.meta = {
            "kind": self.kind,
            "dim": d,
            "n": len(doc_ids),
            "m": self.m,
            "ef_construction": self.ef_construction,
            "build_seconds": round(time.perf_counter() - t0, 3),
            **meta,
        }
        return self

    def search(self, qvecs: np.ndarray, k: int = 100, ef_search: int = 128) -> list[list[Hit]]:
        if self.kind == "hnsw":
            self.index.hnsw.efSearch = max(ef_search, k)
        scores, idx = self.index.search(_normalize(qvecs), k)
        out: list[list[Hit]] = []
        for row_i, row_s in zip(idx, scores, strict=True):
            out.append(
                [
                    Hit(self.doc_ids[int(i)], float(s), "dense")
                    for i, s in zip(row_i, row_s, strict=True)
                    if i >= 0
                ]
            )
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "doc_ids.json").write_text(json.dumps(self.doc_ids))
        (path / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DenseIndex:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        self = cls(
            kind=meta["kind"], m=meta.get("m", 32), ef_construction=meta.get("ef_construction", 200)
        )
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.doc_ids = json.loads((path / "doc_ids.json").read_text())
        self.meta = meta
        return self
