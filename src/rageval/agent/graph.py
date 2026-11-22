"""The serving agent: a LangGraph state machine over the RAG pipeline.

    START -> retrieve -> (rerank | fuse_only) -> [plan -> retrieve2] -> generate
          -> (verify [-> revise] | skip_verify) -> finalize -> END

retrieve   BM25 (thread pool) and dense search (micro-batched query embedding + FAISS) in
           parallel, fused with RRF or weighted fusion
rerank     cross-encoder over the fused head (micro-batched across concurrent requests)
fuse_only  taken when reranking is disabled, or skipped because the request is behind its deadline
plan       (two-hop) the LLM names the article still missing for a bridge question
retrieve2  (two-hop) hybrid + rerank for that query, spliced into the first-hop ranking
generate   grounded, cited answer through the FallbackRouter (primary / fallback model)
verify     NLI citation verification and repair (micro-batched)
revise     (policy "revise") one rewrite of the flagged sentences, re-verified with pruning
finalize   response assembly

Every node runs inside an OpenTelemetry span and records its latency in the state's `timings`,
which becomes the request's trace-log line.
"""

from __future__ import annotations

import asyncio
import operator
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, TypedDict, cast

import numpy as np
from langgraph.graph import END, START, StateGraph
from opentelemetry import trace
from pydantic import BaseModel, Field

from rageval.corpus import Corpus
from rageval.generation.citations import ParsedAnswer, parse_answer
from rageval.generation.prompts import build_messages, max_tokens_for, revise_messages
from rageval.generation.verify import averify_answer, prune_flagged
from rageval.llm.router import FallbackRouter
from rageval.retrieval.bm25 import BM25Index
from rageval.retrieval.dense import DenseIndex, format_query
from rageval.retrieval.fusion import fuse
from rageval.retrieval.hybrid import RetrievalConfig
from rageval.retrieval.multihop import merge_hops, parse_followup, plan_messages
from rageval.serve import telemetry as tm


class AgentConfig(BaseModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    context_k: int = 5
    prompt: str = "zero_shot"
    verify: str = "repair"  # none | repair | prune | revise
    verify_tau: float = 0.5
    budget_ms: float | None = None  # per-request deadline; None = no deadline
    adaptive: bool = False  # skip rerank / verify when the deadline is close
    rerank_min_remaining_ms: float = 3000.0
    plan_min_remaining_ms: float = 3500.0  # second hop only when this much budget is left
    # adaptive load shedding of optional stages (0 = off): skip reranking / verification when this
    # worker already has that many requests in that stage
    rerank_max_inflight: int = 0
    verify_max_inflight: int = 0
    # stage-level fallbacks (adaptive only): rerank fewer candidates, or verify on the fast path
    # (cited documents only, prune what fails) when the stage is backed up or the deadline is close
    rerank_fast_depth: int = 0  # 0 = skip reranking instead
    verify_fast_inflight: int = 0  # 0 = off
    verify_fast_remaining_ms: float = 0.0
    verify_min_remaining_ms: float = 400.0
    revise_min_remaining_ms: float = 1500.0  # one revision round only with this much budget left
    verify_reserve_ms: float = 150.0  # budget held back from generation for verification
    gen_params: dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict, total=False):
    request_id: str
    question: str
    t0: float
    budget_ms: float | None
    fused: list[str]
    hits: list[str]
    followup: str | None
    context: list[str]
    raw: str
    parsed: ParsedAnswer
    route: str
    route_reason: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    verify: dict[str, Any]
    unsupported: list[str]
    revise: dict[str, Any]
    timings: Annotated[dict[str, float], operator.or_]
    degraded: Annotated[list[str], operator.add]
    response: dict[str, Any]


NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]


class RAGAgent:
    def __init__(
        self,
        corpus: Corpus,
        bm25: BM25Index,
        dense: DenseIndex,
        embed_model: str,
        embed: Any,
        rerank: Any | None,
        verifier: Any | None,
        router: FallbackRouter,
        config: AgentConfig,
        cpu_workers: int = 8,
    ) -> None:
        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.embed_model = embed_model
        self.embed = embed
        self.rerank_b = rerank
        self.verifier = verifier
        self.router = router
        self.cfg = config
        self.pool = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix="cpu")
        self.inflight = {"rerank": 0, "verify": 0}
        self.tracer = trace.get_tracer("rageval.agent")
        self.graph = self._build()

    # ------------------------------------------------------------------ helpers

    def remaining_ms(self, state: AgentState) -> float | None:
        b = state.get("budget_ms")
        if b is None:
            return None
        return b - (time.perf_counter() - state["t0"]) * 1e3

    def _traced(self, name: str, fn: NodeFn) -> NodeFn:
        async def wrapper(state: AgentState) -> dict[str, Any]:
            with self.tracer.start_as_current_span(f"node.{name}") as span:
                t0 = time.perf_counter()
                try:
                    upd = await fn(state)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)[:200]))
                    tm.ERRORS_TOTAL.labels(name).inc()
                    raise
                dt = time.perf_counter() - t0
                tm.NODE_SECONDS.labels(name).observe(dt)
                rem = self.remaining_ms(state)
                if rem is not None:
                    span.set_attribute("rag.remaining_ms", rem)
                for k in ("route", "route_reason", "model", "completion_tokens"):
                    if k in upd:
                        span.set_attribute(f"rag.{k}", upd[k])
                upd["timings"] = {name: dt * 1e3}
                return upd

        return wrapper

    async def _aprob(self, premises: Any, hyps: Any) -> np.ndarray:
        assert self.verifier is not None
        return await self.verifier.submit(list(zip(premises, hyps, strict=True)))

    # ------------------------------------------------------------------ nodes

    async def _search(self, q: str, keep: int) -> list[str]:
        rc = self.cfg.retrieval
        loop = asyncio.get_running_loop()
        sparse_f = loop.run_in_executor(self.pool, self.bm25.search, [q], rc.bm25_k)
        qv = await self.embed.submit([format_query(self.embed_model, q)])
        dense = await loop.run_in_executor(
            self.pool, self.dense.search, qv, rc.dense_k, rc.ef_search
        )
        sparse = await sparse_f
        fused = fuse(sparse[0], dense[0], rc.fusion, rc.rrf_k, rc.alpha)
        return [h.doc_id for h in fused[:keep]]

    async def _rerank(self, q: str, cand: list[str]) -> list[str]:
        assert self.rerank_b is not None
        scores = await self.rerank_b.submit([(q, self.corpus.index_text(d)) for d in cand])
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [cand[i] for i in order]

    async def n_retrieve(self, state: AgentState) -> dict[str, Any]:
        keep = max(self.cfg.retrieval.rerank_depth, self.cfg.context_k)
        return {"fused": await self._search(state["question"], keep)}

    async def n_rerank(self, state: AgentState) -> dict[str, Any]:
        depth = self.cfg.retrieval.rerank_depth
        upd: dict[str, Any] = {}
        cap = self.cfg.rerank_max_inflight
        if (
            self.cfg.adaptive
            and cap
            and self.cfg.rerank_fast_depth
            and self.inflight["rerank"] >= cap
        ):
            depth = self.cfg.rerank_fast_depth
            upd["degraded"] = ["rerank_shallow"]
            tm.DEGRADED_TOTAL.labels("rerank_shallow").inc()
        cand = state["fused"][:depth]
        self.inflight["rerank"] += 1
        try:
            ranked = await self._rerank(state["question"], cand)
        finally:
            self.inflight["rerank"] -= 1
        upd["hits"] = ranked + [d for d in state["fused"] if d not in set(cand)]
        return upd

    async def n_plan(self, state: AgentState) -> dict[str, Any]:
        rc = self.cfg.retrieval
        docs = [(self.corpus[d].title, self.corpus[d].text) for d in state["hits"][: rc.plan_k]]
        rr = await self.router.chat(
            plan_messages(state["question"], docs),
            remaining_ms=self.remaining_ms(state),
            max_tokens=32,
            temperature=0.0,
            **self.cfg.gen_params,
        )
        tm.ROUTE_TOTAL.labels(f"plan:{rr.route}", rr.reason).inc()
        return {"followup": parse_followup(rr.result.text)}

    async def n_retrieve2(self, state: AgentState) -> dict[str, Any]:
        rc = self.cfg.retrieval
        q2 = state.get("followup") or ""
        cand = await self._search(q2, max(rc.hop2_depth, rc.hop2_take))
        hop2 = await self._rerank(q2, cand[: rc.hop2_depth]) if self.rerank_b is not None else cand
        return {"hits": merge_hops(state["hits"], hop2, rc.hop2_keep, rc.hop2_take)}

    async def n_fuse_only(self, state: AgentState) -> dict[str, Any]:
        upd: dict[str, Any] = {"hits": state["fused"]}
        if self.cfg.retrieval.rerank and self.rerank_b is not None:
            upd["degraded"] = ["rerank"]
            tm.DEGRADED_TOTAL.labels("rerank").inc()
        return upd

    async def n_generate(self, state: AgentState) -> dict[str, Any]:
        ctx = state["hits"][: self.cfg.context_k]
        docs = [self.corpus[d] for d in ctx]
        msgs = build_messages(self.cfg.prompt, state["question"], docs)
        rem = self.remaining_ms(state)
        if rem is not None and self.cfg.verify != "none":
            rem -= self.cfg.verify_reserve_ms
        rr = await self.router.chat(
            msgs,
            remaining_ms=rem,
            max_tokens=max_tokens_for(self.cfg.prompt),
            temperature=0.0,
            **self.cfg.gen_params,
        )
        tm.ROUTE_TOTAL.labels(rr.route, rr.reason).inc()
        tm.TOKENS_TOTAL.labels(rr.route, "prompt").inc(rr.result.prompt_tokens)
        tm.TOKENS_TOTAL.labels(rr.route, "completion").inc(rr.result.completion_tokens)
        return {
            "context": ctx,
            "raw": rr.result.text,
            "parsed": parse_answer(rr.result.text),
            "route": rr.route,
            "route_reason": rr.reason,
            "model": rr.result.model,
            "prompt_tokens": rr.result.prompt_tokens,
            "completion_tokens": rr.result.completion_tokens,
        }

    def _docs(self, state: AgentState) -> list[tuple[str, str]]:
        return [(self.corpus[d].title, self.corpus[d].text) for d in state["context"]]

    async def n_verify(self, state: AgentState) -> dict[str, Any]:
        c = self.cfg
        rem = self.remaining_ms(state)
        fast = c.adaptive and (
            (c.verify_fast_inflight and self.inflight["verify"] >= c.verify_fast_inflight)
            or (c.verify_fast_remaining_ms and rem is not None and rem < c.verify_fast_remaining_ms)
        )
        self.inflight["verify"] += 1
        try:
            parsed, st = await averify_answer(
                state["parsed"],
                self._docs(state),
                self._aprob,
                tau=c.verify_tau,
                policy="prune" if fast else c.verify,
                fast=bool(fast),
            )
        finally:
            self.inflight["verify"] -= 1
        if fast:
            tm.DEGRADED_TOTAL.labels("verify_fast").inc()
            return {
                "parsed": parsed,
                "unsupported": [],
                "verify": {k: v for k, v in st.__dict__.items() if k != "detail"},
                "degraded": ["verify_fast"],
            }
        bad = [d["text"] for d in st.detail if d.get("unsupported")]
        upd: dict[str, Any] = {
            "parsed": parsed,
            "unsupported": bad,
            "verify": {k: v for k, v in st.__dict__.items() if k != "detail"},
        }
        if self.cfg.verify == "revise" and bad and not self._can_revise(state):
            # no budget for a revision round: drop what failed verification instead
            upd["parsed"] = prune_flagged(parsed, st)
            upd["degraded"] = ["revise"]
            tm.DEGRADED_TOTAL.labels("revise").inc()
        return upd

    def _can_revise(self, state: AgentState) -> bool:
        rem = self.remaining_ms(state)
        return not (
            self.cfg.adaptive and rem is not None and rem < self.cfg.revise_min_remaining_ms
        )

    async def n_revise(self, state: AgentState) -> dict[str, Any]:
        docs = [self.corpus[d] for d in state["context"]]
        msgs = revise_messages(
            state["question"], docs, state["parsed"].answer, state.get("unsupported", [])
        )
        rr = await self.router.chat(
            msgs,
            remaining_ms=self.remaining_ms(state),
            max_tokens=320,
            temperature=0.0,
            **self.cfg.gen_params,
        )
        tm.ROUTE_TOTAL.labels(f"revise:{rr.route}", rr.reason).inc()
        parsed, st = await averify_answer(
            parse_answer(rr.result.text),
            self._docs(state),
            self._aprob,
            tau=self.cfg.verify_tau,
            policy="prune",
        )
        return {
            "parsed": parsed,
            "revise": {
                "route": rr.route,
                **{k: v for k, v in st.__dict__.items() if k != "detail"},
            },
        }

    async def n_skip_verify(self, state: AgentState) -> dict[str, Any]:
        tm.DEGRADED_TOTAL.labels("verify").inc()
        return {"degraded": ["verify"]}

    async def n_finalize(self, state: AgentState) -> dict[str, Any]:
        p = state["parsed"]
        ctx = state["context"]
        cited = sorted({c for s in p.statements for c in s.citations if 1 <= c <= len(ctx)})
        return {
            "response": {
                "request_id": state["request_id"],
                "answer": p.answer,
                "abstained": p.abstained,
                "citations": [
                    {"n": c, "doc_id": ctx[c - 1], "title": self.corpus[ctx[c - 1]].title}
                    for c in cited
                ],
                "statements": [
                    {"text": s.text, "citations": s.citations}
                    for s in p.statements
                    if not s.abstain
                ],
                "context": ctx,
                "route": state.get("route"),
                "route_reason": state.get("route_reason"),
                "model": state.get("model"),
                "degraded": state.get("degraded", []),
            }
        }

    # ------------------------------------------------------------------ edges

    def e_after_retrieve(self, state: AgentState) -> str:
        if not self.cfg.retrieval.rerank or self.rerank_b is None:
            return "fuse_only"
        cap = self.cfg.rerank_max_inflight
        if self.cfg.adaptive and cap and self.inflight["rerank"] >= cap:
            return "fuse_only"
        rem = self.remaining_ms(state)
        if self.cfg.adaptive and rem is not None and rem < self.cfg.rerank_min_remaining_ms:
            return "fuse_only"
        return "rerank"

    def e_after_hop1(self, state: AgentState) -> str:
        if self.cfg.retrieval.hops < 2:
            return "generate"
        rem = self.remaining_ms(state)
        if self.cfg.adaptive and rem is not None and rem < self.cfg.plan_min_remaining_ms:
            tm.DEGRADED_TOTAL.labels("hop2").inc()
            return "generate"
        return "plan"

    def e_after_plan(self, state: AgentState) -> str:
        return "retrieve2" if state.get("followup") else "generate"

    def e_after_verify(self, state: AgentState) -> str:
        if self.cfg.verify == "revise" and state.get("unsupported") and self._can_revise(state):
            return "revise"
        return "finalize"

    def e_after_generate(self, state: AgentState) -> str:
        if self.cfg.verify == "none" or self.verifier is None or state["parsed"].abstained:
            return "finalize"
        rem = self.remaining_ms(state)
        if self.cfg.adaptive and rem is not None and rem < self.cfg.verify_min_remaining_ms:
            return "skip_verify"
        cap = self.cfg.verify_max_inflight
        if self.cfg.adaptive and cap and self.inflight["verify"] >= cap:
            return "skip_verify"
        return "verify"

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        for name, fn in [
            ("retrieve", self.n_retrieve),
            ("rerank", self.n_rerank),
            ("fuse_only", self.n_fuse_only),
            ("plan", self.n_plan),
            ("retrieve2", self.n_retrieve2),
            ("generate", self.n_generate),
            ("verify", self.n_verify),
            ("skip_verify", self.n_skip_verify),
            ("revise", self.n_revise),
            ("finalize", self.n_finalize),
        ]:
            g.add_node(name, cast(Any, self._traced(name, fn)))
        g.add_edge(START, "retrieve")
        g.add_conditional_edges("retrieve", self.e_after_retrieve, ["rerank", "fuse_only"])
        g.add_conditional_edges("rerank", self.e_after_hop1, ["plan", "generate"])
        g.add_conditional_edges("fuse_only", self.e_after_hop1, ["plan", "generate"])
        g.add_conditional_edges("plan", self.e_after_plan, ["retrieve2", "generate"])
        g.add_edge("retrieve2", "generate")
        g.add_conditional_edges(
            "generate", self.e_after_generate, ["verify", "skip_verify", "finalize"]
        )
        g.add_conditional_edges("verify", self.e_after_verify, ["revise", "finalize"])
        g.add_edge("revise", "finalize")
        g.add_edge("skip_verify", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------ entry point

    async def answer(
        self, question: str, request_id: str | None = None, budget_ms: float | None = None
    ) -> AgentState:
        rid = request_id or uuid.uuid4().hex[:16]
        state: AgentState = {
            "request_id": rid,
            "question": question,
            "t0": time.perf_counter(),
            "budget_ms": budget_ms if budget_ms is not None else self.cfg.budget_ms,
            "timings": {},
            "degraded": [],
        }
        with self.tracer.start_as_current_span("rag.request") as span:
            span.set_attribute("rag.request_id", rid)
            out: AgentState = await self.graph.ainvoke(state)
            span.set_attribute("rag.route", out.get("route", ""))
            span.set_attribute("rag.degraded", ",".join(out.get("degraded", [])))
        return out
