"""Stand-alone encoder servers (one model per process, own GIL, dynamic micro-batching).

    rageval encoders serve --kind embed  --model BAAI/bge-large-en-v1.5 --port 8101
    rageval encoders serve --kind rerank --model Alibaba-NLP/gte-reranker-modernbert-base --port 8102
    rageval encoders serve --kind nli    --model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli --port 8103

Wire formats follow the common servers so the agent can also point at vLLM or TEI:
  POST /v1/embeddings {"input": [str]}                      -> {"data": [{"index", "embedding"}]}
  POST /rerank        {"query": str, "documents": [str]}    -> {"results": [{"index", "relevance_score"}]}
  POST /nli           {"premises": [str], "hypotheses": [str]} -> {"probs": [float]}

Clients (`Remote*Client`) expose the same `submit(items)` coroutine as the in-process
MicroBatcher, so the agent does not care where a model runs.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from pydantic import BaseModel

from rageval.serve.batcher import MicroBatcher

ENC_SECONDS = Histogram(
    "rag_encoder_request_seconds",
    "Encoder request latency",
    ["kind"],
    buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)


class EmbedRequest(BaseModel):
    input: list[str]
    model: str | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model: str | None = None
    top_n: int | None = None


class NLIRequest(BaseModel):
    premises: list[str]
    hypotheses: list[str]


def create_encoder_app(
    kind: str,
    model: str,
    device: str | None = None,
    max_items: int = 256,
    max_wait_ms: float = 4.0,
    batch_size: int = 128,
) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        fn: Any
        if kind == "embed":
            from rageval.models import HFEmbedder

            enc = HFEmbedder(model, device=device, batch_size=batch_size)
            fn = enc.embed
        elif kind == "rerank":
            from rageval.models import HFCrossEncoder

            ce = HFCrossEncoder(model, device=device, batch_size=batch_size)
            fn = ce.score
        elif kind == "nli":
            from rageval.judges.nli import ClassNLI

            nli = ClassNLI(model, device=device, batch_size=batch_size)

            def nli_fn(pairs: list[tuple[str, str]]) -> np.ndarray:
                return nli.prob([p for p, _ in pairs], [h for _, h in pairs])

            fn = nli_fn

        else:
            raise ValueError(kind)
        state["batcher"] = MicroBatcher(fn, kind, max_items, max_wait_ms)
        yield
        state["batcher"].close()

    app = FastAPI(title=f"{kind} encoder", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        if "batcher" not in state:
            raise HTTPException(503, "loading")
        return {"status": "ok", "kind": kind, "model": model}

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbedRequest) -> dict[str, Any]:
        if kind != "embed":
            raise HTTPException(404)
        t0 = time.perf_counter()
        v = await state["batcher"].submit(req.input)
        ENC_SECONDS.labels(kind).observe(time.perf_counter() - t0)
        return {
            "data": [{"index": i, "embedding": row.tolist()} for i, row in enumerate(v)],
            "model": model,
        }

    @app.post("/rerank")
    async def rerank(req: RerankRequest) -> dict[str, Any]:
        if kind != "rerank":
            raise HTTPException(404)
        t0 = time.perf_counter()
        s = await state["batcher"].submit([(req.query, d) for d in req.documents])
        ENC_SECONDS.labels(kind).observe(time.perf_counter() - t0)
        return {
            "results": [{"index": i, "relevance_score": float(x)} for i, x in enumerate(s)],
            "model": model,
        }

    @app.post("/nli")
    async def nli_ep(req: NLIRequest) -> dict[str, Any]:
        if kind != "nli":
            raise HTTPException(404)
        if len(req.premises) != len(req.hypotheses):
            raise HTTPException(422, "premises and hypotheses differ in length")
        t0 = time.perf_counter()
        p = await state["batcher"].submit(list(zip(req.premises, req.hypotheses, strict=True)))
        ENC_SECONDS.labels(kind).observe(time.perf_counter() - t0)
        return {"probs": [float(x) for x in p]}

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        b: MicroBatcher = state["batcher"]
        return {
            "kind": kind,
            "model": model,
            "batches": b.batches,
            "items": b.items,
            "busy_s": round(b.busy_s, 3),
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


class _Remote:
    def __init__(self, base_url: str, timeout: float = 30.0, model: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model  # sent as "model" (required by vLLM's OpenAI-compatible endpoints)
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self.batches = self.items = 0
        self.busy_s = 0.0

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=512, max_keepalive_connections=512),
            )
        return self._client

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        t0 = time.perf_counter()
        r = await self.client().post(f"{self.base_url}{path}", json=body)
        r.raise_for_status()
        self.batches += 1
        self.busy_s += time.perf_counter() - t0
        return r.json()

    def close(self) -> None:
        pass


class RemoteEmbedClient(_Remote):
    async def submit(self, texts: Sequence[str]) -> np.ndarray:
        self.items += len(texts)
        body: dict[str, Any] = {"input": list(texts)}
        if self.model:
            body["model"] = self.model
        d = await self._post("/v1/embeddings", body)
        rows = sorted(d["data"], key=lambda r: r["index"])
        return np.asarray([r["embedding"] for r in rows], dtype=np.float32)


class RemoteRerankClient(_Remote):
    async def submit(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Pairs from the agent share one query; mixed queries are grouped into one call each."""
        self.items += len(pairs)
        out = np.zeros(len(pairs), dtype=np.float32)
        groups: dict[str, list[int]] = {}
        for i, (q, _) in enumerate(pairs):
            groups.setdefault(q, []).append(i)
        for q, idx in groups.items():
            body: dict[str, Any] = {"query": q, "documents": [pairs[i][1] for i in idx]}
            if self.model:
                body["model"] = self.model
            d = await self._post("/rerank", body)
            for r in d["results"]:
                out[idx[int(r["index"])]] = float(r["relevance_score"])
        return out


class RemoteNLIClient(_Remote):
    async def submit(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        self.items += len(pairs)
        d = await self._post(
            "/nli", {"premises": [p for p, _ in pairs], "hypotheses": [h for _, h in pairs]}
        )
        return np.asarray(d["probs"], dtype=np.float32)
