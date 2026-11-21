"""Cross-encoder reranking (in-process or a vLLM /rerank server)."""

from __future__ import annotations

from typing import Any, Protocol

import httpx
import numpy as np


class Reranker(Protocol):
    model: str

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


class LocalReranker:
    """Cross-encoder in this process; see rageval.models.HFCrossEncoder."""

    def __init__(self, model: str, **kw: Any) -> None:
        from rageval.models import HFCrossEncoder

        self.model = model
        self._ce = HFCrossEncoder(model, **kw)

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return self._ce.score(pairs)


class RemoteReranker:
    """vLLM's Jina/Cohere-compatible POST /rerank {model, query, documents}."""

    def __init__(
        self, model: str, base_url: str, timeout: float = 60.0, served_name: str | None = None
    ) -> None:
        self.model = model
        self.served_name = served_name or model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._aclient: httpx.AsyncClient | None = None
        self._timeout = timeout

    def _payload(self, query: str, docs: list[str]) -> dict[str, Any]:
        return {"model": self.served_name, "query": query, "documents": docs, "top_n": len(docs)}

    @staticmethod
    def _parse(data: dict[str, Any], n: int) -> np.ndarray:
        out = np.full(n, -np.inf, dtype=np.float32)
        for r in data["results"]:
            out[int(r["index"])] = float(r["relevance_score"])
        return out

    def rerank(self, query: str, docs: list[str]) -> np.ndarray:
        r = self._client.post(f"{self.base_url}/rerank", json=self._payload(query, docs))
        r.raise_for_status()
        return self._parse(r.json(), len(docs))

    async def arerank(self, query: str, docs: list[str]) -> np.ndarray:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(
                timeout=self._timeout, limits=httpx.Limits(max_connections=256)
            )
        r = await self._aclient.post(f"{self.base_url}/rerank", json=self._payload(query, docs))
        r.raise_for_status()
        return self._parse(r.json(), len(docs))

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        # group consecutive pairs by query so each group is one /rerank call
        out = np.zeros(len(pairs), dtype=np.float32)
        i = 0
        while i < len(pairs):
            j = i
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1
            out[i:j] = self.rerank(pairs[i][0], [p[1] for p in pairs[i:j]])
            i = j
        return out


def make_reranker(model: str, base_url: str | None = None, **kw: Any) -> Reranker:
    if base_url:
        return RemoteReranker(model, base_url, **kw)
    return LocalReranker(model, **kw)
