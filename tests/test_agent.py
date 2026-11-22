import hashlib

import numpy as np
from fastapi.testclient import TestClient

from rageval.agent.graph import AgentConfig, RAGAgent
from rageval.corpus import Corpus
from rageval.llm.client import ChatResult
from rageval.llm.router import FallbackRouter, RouterPolicy
from rageval.retrieval.bm25 import BM25Index
from rageval.retrieval.dense import DenseIndex
from rageval.retrieval.hybrid import RetrievalConfig
from rageval.serve.app import ServeConfig, create_app
from rageval.serve.batcher import MicroBatcher
from rageval.types import Doc

DOCS = [
    Doc("d1", "Scott Derrickson", ["Scott Derrickson is an American director."]),
    Doc("d2", "Ed Wood", ["Ed Wood was an American filmmaker."]),
    Doc("d3", "Paris", ["Paris is the capital of France."]),
    Doc("d4", "Tyler Bates", ["Tyler Bates is an American musician."]),
]


def bow(texts, dim=64):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out


class FakeLLM:
    def __init__(self, name, text, plan="Tyler Bates"):
        self.name, self.text, self.plan, self.calls = name, text, plan, 0

    async def chat(self, messages, timeout=None, **kw):
        self.calls += 1
        if "plan searches" in messages[0]["content"]:
            return ChatResult(text=self.plan, model=self.name, completion_tokens=3)
        return ChatResult(text=self.text, model=self.name, prompt_tokens=10, completion_tokens=5)


def nli(pairs):
    # entailed iff the hypothesis' key word appears in the premise
    return np.array([1.0 if h.split()[0] in p else 0.0 for p, h in pairs], dtype=np.float32)


def make_agent(answer, rc_over=None, **cfg):
    corpus = Corpus(DOCS)
    texts = [corpus.index_text(d.doc_id) for d in DOCS]
    bm25 = BM25Index().build(corpus.ids, texts)
    dense = DenseIndex("flat").build(corpus.ids, bow(texts))
    embed = MicroBatcher(lambda qs: bow(qs), "embed")
    rerank = MicroBatcher(
        lambda pairs: bow([q for q, _ in pairs]).__mul__(bow([d for _, d in pairs])).sum(1),
        "rerank",
    )
    verify = MicroBatcher(nli, "verify")
    primary = FakeLLM("primary", answer)
    router = FallbackRouter(primary, FakeLLM("fallback", answer), RouterPolicy(hedge=False))
    rc = RetrievalConfig(bm25_k=4, dense_k=4, rerank_depth=4, final_k=4, **(rc_over or {}))
    agent = RAGAgent(
        corpus,
        bm25,
        dense,
        "fake",
        embed,
        rerank,
        verify,
        router,
        AgentConfig(retrieval=rc, **({"context_k": 3} | cfg)),
    )
    return agent


async def test_agent_end_to_end_with_verification():
    # cites [3] wrongly; the verifier should re-cite to the document that says "Scott"
    agent = make_agent("Scott Derrickson is an American director [3].")
    out = await agent.answer("Who is Scott Derrickson?")
    r = out["response"]
    assert r["route"] == "primary" and not r["abstained"]
    ctx = r["context"]
    assert "d1" in ctx
    assert r["citations"][0]["doc_id"] == "d1"
    assert set(out["timings"]) == {"retrieve", "rerank", "generate", "verify", "finalize"}
    assert out["verify"]["recited"] == 1


async def test_adaptive_skips_rerank_and_verify_when_behind_deadline():
    agent = make_agent("Ed Wood was an American filmmaker [1].", adaptive=True, budget_ms=1.0)
    out = await agent.answer("Who was Ed Wood?")
    assert out["degraded"] == ["rerank", "verify"]
    assert "fuse_only" in out["timings"] and "verify" not in out["timings"]


async def test_abstention_skips_verification():
    agent = make_agent("I cannot answer this question based on the provided documents.")
    out = await agent.answer("What is the capital of Mars?")
    assert out["response"]["abstained"] and "verify" not in out["timings"]


def test_http_api_and_metrics(tmp_path):
    cfg = ServeConfig()
    cfg.telemetry.trace_log = str(tmp_path / "trace.jsonl")
    cfg.telemetry.span_log = str(tmp_path / "spans.jsonl")
    app = create_app(cfg, agent_factory=lambda c: make_agent("Paris is the capital of France [1]."))
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}
        r = c.post("/v1/answer", json={"question": "What is the capital of France?"})
        assert r.status_code == 200
        body = r.json()
        assert body["citations"][0]["title"] == "Paris"
        assert body["latency_ms"] > 0
        m = c.get("/metrics").text
        assert "rag_request_seconds_bucket" in m and "rag_node_seconds_bucket" in m
        assert c.get("/v1/stats").json()["router"]
        ret = c.post("/v1/retrieve", json={"question": "capital of France", "k": 2}).json()
        assert ret["titles"][0] == "Paris"
    lines = (tmp_path / "trace.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and '"route":"primary"' in lines[0]


async def test_two_hop_plans_and_splices_second_hop():
    agent = make_agent(
        "Paris is the capital of France [1].",
        rc_over={"hops": 2, "plan_k": 2, "hop2_depth": 4, "hop2_keep": 1, "hop2_take": 1},
        context_k=3,
        verify="none",
    )
    out = await agent.answer("What is the capital of France?")
    assert out["followup"] == "Tyler Bates"
    assert "plan" in out["timings"] and "retrieve2" in out["timings"]
    assert out["response"]["context"][1] == "d4"  # second-hop hit spliced after the first-hop head


async def test_revise_policy_rewrites_flagged_sentences():
    agent = make_agent(
        "Paris is the capital of France [1]. Zzz is unsupported [2].", verify="revise"
    )
    # second call (the revision) returns a clean answer
    texts = iter(
        [
            "Paris is the capital of France [1]. Zzz is unsupported [2].",
            "Paris is the capital of France [1].",
        ]
    )

    async def chat(messages, timeout=None, **kw):
        from rageval.llm.client import ChatResult

        return ChatResult(text=next(texts), model="primary")

    agent.router.primary.chat = chat
    out = await agent.answer("What is the capital of France?")
    assert "revise" in out["timings"]
    assert [s["text"] for s in out["response"]["statements"]] == ["Paris is the capital of France."]


async def test_verify_fast_path_under_deadline():
    agent = make_agent(
        "Scott Derrickson is an American director [3].",
        adaptive=True,
        budget_ms=10_000,
        verify_fast_remaining_ms=20_000,  # always "late": take the fast path
    )
    out = await agent.answer("Who is Scott Derrickson?")
    assert "verify_fast" in out["degraded"]
    # fast path does not search for a better citation: the wrong one is pruned instead
    assert out["response"]["abstained"]
