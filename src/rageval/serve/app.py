"""FastAPI service exposing the LangGraph RAG agent.

POST /v1/answer    {"question": str, "budget_ms"?: float, "request_id"?: str}
POST /v1/retrieve  {"question": str, "k"?: int}
GET  /health       liveness + readiness (models loaded)
GET  /metrics      Prometheus
GET  /v1/stats     router / batcher counters
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from rageval.agent.graph import AgentConfig, RAGAgent
from rageval.llm.client import QWEN3_NO_THINK, ChatClient, LLMError
from rageval.llm.router import FallbackRouter, RouterPolicy
from rageval.pipeline import ModelsConfig
from rageval.serve import telemetry as tm
from rageval.serve.batcher import MicroBatcher


class BatchCfg(BaseModel):
    max_items: int = 256
    max_wait_ms: float = 4.0


class TelemetryCfg(BaseModel):
    span_log: str | None = None
    trace_log: str | None = None
    otlp_endpoint: str | None = None
    service: str = "rag-agent"


class ServeConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    router: RouterPolicy | str = "adaptive"  # "static" | "adaptive" | explicit fields
    telemetry: TelemetryCfg = Field(default_factory=TelemetryCfg)
    embed_batch: BatchCfg = Field(default_factory=lambda: BatchCfg(max_items=128, max_wait_ms=2.0))
    rerank_batch: BatchCfg = Field(default_factory=lambda: BatchCfg(max_items=512, max_wait_ms=4.0))
    verify_batch: BatchCfg = Field(default_factory=lambda: BatchCfg(max_items=512, max_wait_ms=4.0))
    device: str | None = None
    cpu_workers: int = 8

    model_config = {"arbitrary_types_allowed": True}

    def policy(self) -> RouterPolicy:
        if isinstance(self.router, RouterPolicy):
            return self.router
        if self.router == "static":
            return RouterPolicy.static()
        if self.router == "adaptive":
            return RouterPolicy()
        raise ValueError(f"unknown router policy {self.router}")


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Dotted-key overrides, values parsed as YAML: ["router=static", "agent.budget_ms=4000"]."""
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = yaml.safe_load(val)
    return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> ServeConfig:
    raw = apply_overrides(yaml.safe_load(Path(path).read_text()) or {}, overrides)
    if isinstance(raw.get("router"), dict):
        raw["router"] = RouterPolicy(**raw["router"])
    return ServeConfig.model_validate(raw)


class AnswerRequest(BaseModel):
    question: str
    budget_ms: float | None = None
    request_id: str | None = None


class RetrieveRequest(BaseModel):
    question: str
    k: int = 10


def build_agent(cfg: ServeConfig) -> RAGAgent:
    from rageval.corpus import Corpus
    from rageval.models import HFCrossEncoder, HFEmbedder
    from rageval.retrieval.bm25 import BM25Index
    from rageval.retrieval.dense import DenseIndex
    from rageval.retrieval.experiments import slug

    m = cfg.models
    corpus = Corpus.load(m.corpus)
    idx = Path(m.index_dir)
    bm25 = BM25Index.load(idx / "bm25")
    dense = DenseIndex.load(idx / "dense" / slug(m.embedder) / m.dense_index)
    from rageval.serve.encoders import RemoteEmbedClient, RemoteNLIClient, RemoteRerankClient

    embed_b: Any
    if m.embedder_url:
        embed_b = RemoteEmbedClient(m.embedder_url)
    else:
        emb = HFEmbedder(m.embedder, device=cfg.device, batch_size=cfg.embed_batch.max_items)
        embed_b = MicroBatcher(
            emb.embed, "embed", cfg.embed_batch.max_items, cfg.embed_batch.max_wait_ms
        )
    rerank_b: Any = None
    if m.reranker and cfg.agent.retrieval.rerank:
        if m.reranker_url:
            rerank_b = RemoteRerankClient(m.reranker_url)
        else:
            ce = HFCrossEncoder(m.reranker, device=cfg.device, batch_size=128)
            rerank_b = MicroBatcher(
                ce.score, "rerank", cfg.rerank_batch.max_items, cfg.rerank_batch.max_wait_ms
            )
    verify_b: Any = None
    if cfg.agent.verify != "none":
        if m.verifier_url:
            verify_b = RemoteNLIClient(m.verifier_url)
        else:
            from rageval.judges.nli import ClassNLI

            nli = ClassNLI(m.verifier, device=cfg.device, batch_size=128)
            verify_b = MicroBatcher(
                lambda pairs: nli.prob([p for p, _ in pairs], [h for _, h in pairs]),
                "verify",
                cfg.verify_batch.max_items,
                cfg.verify_batch.max_wait_ms,
            )
    gen_params = dict(cfg.agent.gen_params)
    if "qwen3" in m.generator.lower() and "chat_template_kwargs" not in gen_params:
        gen_params.update(QWEN3_NO_THINK)
    cfg.agent.gen_params = gen_params
    pol = cfg.policy()
    primary = ChatClient(
        m.generator_url, m.generator, name="primary", timeout=pol.primary_timeout_s
    )
    fallback = (
        ChatClient(m.fallback_url, m.fallback, name="fallback", timeout=pol.fallback_timeout_s)
        if m.fallback and m.fallback_url
        else None
    )
    router = FallbackRouter(primary, fallback, pol)
    return RAGAgent(
        corpus,
        bm25,
        dense,
        m.embedder,
        embed_b,
        rerank_b,
        verify_b,
        router,
        cfg.agent,
        cfg.cpu_workers,
    )


def create_app(cfg: ServeConfig, agent_factory: Any = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        tm.setup_tracing(cfg.telemetry.service, cfg.telemetry.span_log, cfg.telemetry.otlp_endpoint)
        state["tracelog"] = tm.TraceLog(cfg.telemetry.trace_log)
        state["agent"] = (agent_factory or build_agent)(cfg)
        state["ready"] = True
        yield
        state["ready"] = False
        state["tracelog"].close()

    app = FastAPI(title="RAG agent", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        if not state.get("ready"):
            raise HTTPException(503, "loading")
        return {"status": "ok"}

    @app.post("/v1/answer")
    async def answer(req: AnswerRequest) -> dict[str, Any]:
        agent: RAGAgent = state["agent"]
        t0 = time.perf_counter()
        try:
            out = await agent.answer(req.question, req.request_id, req.budget_ms)
        except LLMError as e:
            tm.ERRORS_TOTAL.labels("llm").inc()
            raise HTTPException(503, f"generation failed: {e}") from e
        ms = (time.perf_counter() - t0) * 1e3
        resp = dict(out["response"])
        resp["latency_ms"] = ms
        resp["timings_ms"] = out.get("timings", {})
        resp["verify"] = out.get("verify")
        degraded = ",".join(out.get("degraded", [])) or "none"
        tm.REQUEST_SECONDS.labels(out.get("route", "none"), degraded).observe(ms / 1e3)
        state["tracelog"].write(
            {
                "ts": time.time(),
                "request_id": resp["request_id"],
                "latency_ms": ms,
                "route": out.get("route"),
                "route_reason": out.get("route_reason"),
                "model": out.get("model"),
                "degraded": out.get("degraded", []),
                "timings_ms": out.get("timings", {}),
                "prompt_tokens": out.get("prompt_tokens"),
                "completion_tokens": out.get("completion_tokens"),
                "abstained": resp["abstained"],
                "n_citations": len(resp["citations"]),
                "verify": out.get("verify"),
            }
        )
        return resp

    @app.post("/v1/retrieve")
    async def retrieve(req: RetrieveRequest) -> dict[str, Any]:
        agent: RAGAgent = state["agent"]
        st: Any = {"question": req.question, "t0": time.perf_counter(), "budget_ms": None}
        r = await agent.n_retrieve(st)
        st.update(r)
        if agent.rerank_b is not None and agent.cfg.retrieval.rerank:
            st.update(await agent.n_rerank(st))
        else:
            st["hits"] = st["fused"]
        ids = st["hits"][: req.k]
        return {"doc_ids": ids, "titles": [agent.corpus[d].title for d in ids]}

    @app.get("/metrics")
    async def metrics() -> Response:
        agent: RAGAgent | None = state.get("agent")
        if agent is not None:
            for name, s in agent.router.stats.items():
                tm.LLM_INFLIGHT.labels(name).set(s.inflight)
            tm.BREAKER_OPEN.set(1.0 if agent.router.breaker.state == "open" else 0.0)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/stats")
    async def stats() -> dict[str, Any]:
        agent: RAGAgent = state["agent"]
        b = {
            name: {"batches": x.batches, "items": x.items, "busy_s": round(x.busy_s, 3)}
            for name, x in [
                ("embed", agent.embed),
                ("rerank", agent.rerank_b),
                ("verify", agent.verifier),
            ]
            if x is not None
        }
        return {
            "router": agent.router.counters,
            "router_stats": {k: v.__dict__ for k, v in agent.router.stats.items()},
            "breaker": agent.router.breaker.state,
            "batchers": b,
            "tracelog_dropped": state["tracelog"].dropped,
        }

    return app


def app_from_env() -> FastAPI:
    """uvicorn factory (one per worker process):
    RAG_SERVE_CONFIG=path [RAG_SERVE_SET='["k=v", ...]'] uvicorn --factory rageval.serve.app:app_from_env"""
    import json

    overrides = json.loads(os.environ.get("RAG_SERVE_SET", "[]"))
    cfg = load_config(os.environ["RAG_SERVE_CONFIG"], overrides)
    wid = os.getpid()
    for key in ("trace_log", "span_log"):  # one file per worker process
        path = getattr(cfg.telemetry, key)
        if path and int(os.environ.get("RAG_SERVE_WORKERS", "1")) > 1:
            setattr(cfg.telemetry, key, path.replace(".jsonl", f".{wid}.jsonl"))
    return create_app(cfg)
