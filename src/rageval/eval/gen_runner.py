"""Offline generation over a suite for one or more pipeline configurations.

Retrieval is computed once per distinct retrieval config and shared by the prompt variants that
use it, so prompt comparisons see identical contexts (a requirement for paired tests).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from rageval.corpus import Corpus
from rageval.eval.retrieval_metrics import recall_at_k
from rageval.generation.citations import parse_answer
from rageval.generation.prompts import build_messages, max_tokens_for
from rageval.generation.verify import averify_answer
from rageval.io import stable_hash
from rageval.llm.client import ChatClient, ChatResult, gather_limited
from rageval.pipeline import PipelineConfig
from rageval.retrieval.hybrid import HybridRetriever, RetrievalResult
from rageval.types import Hit, Question


def retrieve_all(
    retriever: HybridRetriever,
    questions: Sequence[Question],
    cfgs: Sequence[PipelineConfig],
    batch: int = 64,
    client: ChatClient | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, dict[str, RetrievalResult]]:
    """{retrieval-config hash: {qid: result}} for every distinct retrieval config. Two-hop
    configs reuse the matching single-hop ranking as their first hop."""
    from rageval.retrieval.multihop import merge_hops, second_hop

    out: dict[str, dict[str, RetrievalResult]] = {}
    for cfg in sorted(cfgs, key=lambda c: c.retrieval.hops):
        key = retrieval_key(cfg)
        if key in out:
            continue
        rc = cfg.retrieval.model_copy(update={"final_k": 100})
        one = rc.model_copy(update={"hops": 1})
        one_key = stable_hash(one.model_dump())
        if one_key not in out:
            res: dict[str, RetrievalResult] = {}
            for i in range(0, len(questions), batch):
                chunk = questions[i : i + batch]
                for q, r in zip(
                    chunk,
                    retriever.retrieve_batch([q.question for q in chunk], **one.model_dump()),
                    strict=True,
                ):
                    res[q.qid] = r
            out[one_key] = res
        if rc.hops == 1:
            continue
        assert client is not None, "two-hop retrieval needs the planner LLM"
        first = out[one_key]
        # a fresh client: httpx connection pools are bound to the event loop that created them
        planner = ChatClient(client.base_url, client.model, timeout=client.timeout)
        mh = asyncio.run(
            second_hop(
                retriever,
                planner,
                [q.question for q in questions],
                [first[q.qid].doc_ids for q in questions],
                one,
                rc.plan_k,
                rc.hop2_depth,
                128,
                params,
            )
        )
        two: dict[str, RetrievalResult] = {}
        for q, m in zip(questions, mh, strict=True):
            merged = merge_hops(m.hop1, m.hop2, rc.hop2_keep, rc.hop2_take) if m.hop2 else m.hop1
            r = RetrievalResult(q.question, [Hit(d, 0.0, "multihop") for d in merged])
            r.timings_ms = {"planner": m.planner_ms}
            r.followup = m.followup  # type: ignore[attr-defined]
            two[q.qid] = r
        out[key] = two
    return out


def retrieval_key(cfg: PipelineConfig) -> str:
    rc = cfg.retrieval.model_copy(update={"final_k": 100})
    return stable_hash(rc.model_dump())


async def generate(
    questions: Sequence[Question],
    corpus: Corpus,
    retrieved: dict[str, dict[str, RetrievalResult]],
    cfgs: Sequence[PipelineConfig],
    client: ChatClient,
    concurrency: int = 128,
    extra_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    jobs: list[tuple[PipelineConfig, Question, list[str]]] = []
    coros = []
    for cfg in cfgs:
        res = retrieved[retrieval_key(cfg)]
        for q in questions:
            ctx = res[q.qid].doc_ids[: cfg.context_k]
            docs = [corpus[d] for d in ctx]
            msgs = build_messages(cfg.prompt, q.question, docs)
            jobs.append((cfg, q, ctx))
            coros.append(
                client.chat(
                    msgs,
                    max_tokens=cfg.max_tokens or max_tokens_for(cfg.prompt),
                    temperature=cfg.temperature,
                    **(extra_params or {}),
                )
            )
    t0 = time.perf_counter()
    outs = await gather_limited(coros, concurrency)
    wall = time.perf_counter() - t0
    records = []
    for (cfg, q, ctx), r in zip(jobs, outs, strict=True):
        rec: dict[str, Any] = {
            "pipeline": cfg.name,
            "config": cfg.model_dump(),
            "qid": q.qid,
            "qtype": q.qtype,
            "answerable": q.answerable,
            "question": q.question,
            "gold_answer": q.answer,
            "gold_doc_ids": q.gold_doc_ids,
            "context": ctx,
            "context_recall": recall_at_k(ctx, q.gold_doc_ids, len(ctx)) if q.answerable else None,
            "followup": getattr(retrieved[retrieval_key(cfg)][q.qid], "followup", None),
        }
        if isinstance(r, ChatResult):
            parsed = parse_answer(r.text)
            rec.update(
                raw=r.text,
                model=r.model,
                latency_ms=r.latency_ms,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                finish_reason=r.finish_reason,
                error=None,
            )
        else:
            parsed = parse_answer("")
            rec.update(raw="", error=str(r))
        rec["parsed_raw"] = _parsed_dict(parsed)
        rec["_parsed"] = parsed
        records.append(rec)
    if records:
        records[0]["_wall_seconds"] = wall
    return records


def apply_verifier(
    records: list[dict[str, Any]],
    corpus: Corpus,
    prob: Any,
    client: ChatClient | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Verify every record whose pipeline asks for it, batching NLI calls across answers.
    Policy "revise" sends flagged sentences back to the generator once (needs `client`) and
    verifies the rewrite with "prune"."""
    from rageval.generation.prompts import revise_messages
    from rageval.serve.batcher import MicroBatcher

    batcher = None
    if prob is not None:
        batcher = MicroBatcher(
            lambda pairs: prob([p for p, _ in pairs], [h for _, h in pairs]),
            "verify-offline",
            max_items=512,
            max_wait_ms=20,
        )

    async def aprob(ps: Sequence[str], hs: Sequence[str]) -> Any:
        assert batcher is not None
        return await batcher.submit(list(zip(ps, hs, strict=True)))

    reviser = ChatClient(client.base_url, client.model, timeout=client.timeout) if client else None

    async def one(rec: dict[str, Any]) -> None:
        cfg = PipelineConfig.model_validate(rec["config"])
        parsed = rec.pop("_parsed")
        if cfg.verify != "none" and not rec.get("error"):
            docs = [(corpus[d].title, corpus[d].text) for d in rec["context"]]
            t0 = time.perf_counter()
            parsed, st = await averify_answer(
                parsed, docs, aprob, tau=cfg.verify_tau, policy=cfg.verify
            )
            rec["verify"] = {**st.__dict__, "ms": (time.perf_counter() - t0) * 1e3}
            bad = [d["text"] for d in st.detail if d.get("unsupported")]
            if cfg.verify == "revise" and bad:
                assert reviser is not None, "policy revise needs the generator client"
                msgs = revise_messages(
                    rec["question"], [corpus[d] for d in rec["context"]], parsed.answer, bad
                )
                try:
                    r2 = await reviser.chat(msgs, max_tokens=320, temperature=0.0, **(params or {}))
                    revised = parse_answer(r2.text)
                    parsed, st2 = await averify_answer(
                        revised, docs, aprob, tau=cfg.verify_tau, policy="prune"
                    )
                    rec["revise"] = {
                        "unsupported_before": bad,
                        "raw": r2.text,
                        "completion_tokens": r2.completion_tokens,
                        "verify2": {k: v for k, v in st2.__dict__.items() if k != "detail"},
                    }
                except Exception as e:  # keep the repaired first-pass answer, but prune what failed
                    rec["revise"] = {"error": str(e)[:200]}
                    parsed, _ = await averify_answer(
                        parsed, docs, aprob, tau=cfg.verify_tau, policy="prune"
                    )
        rec["final"] = _parsed_dict(parsed)

    async def all_() -> None:
        sem = asyncio.Semaphore(256)

        async def guarded(r: dict[str, Any]) -> None:
            async with sem:
                await one(r)

        await asyncio.gather(*(guarded(r) for r in records))

    asyncio.run(all_())
    if batcher is not None:
        batcher.close()


def _parsed_dict(p: Any) -> dict[str, Any]:
    return {
        "answer": p.answer,
        "abstained": p.abstained,
        "reasoning": p.reasoning,
        "statements": [
            {"text": s.text, "citations": s.citations, "abstain": s.abstain} for s in p.statements
        ],
    }


def run(coro: Any) -> Any:
    return asyncio.run(coro)
