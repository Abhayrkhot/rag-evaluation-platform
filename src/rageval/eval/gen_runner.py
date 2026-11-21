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
from rageval.types import Question


def retrieve_all(
    retriever: HybridRetriever,
    questions: Sequence[Question],
    cfgs: Sequence[PipelineConfig],
    batch: int = 64,
) -> dict[str, dict[str, RetrievalResult]]:
    """{retrieval-config hash: {qid: result}} for every distinct retrieval config."""
    out: dict[str, dict[str, RetrievalResult]] = {}
    for cfg in cfgs:
        rc = cfg.retrieval.model_copy(update={"final_k": max(cfg.context_k, cfg.retrieval.final_k)})
        key = stable_hash(rc.model_dump())
        if key in out:
            continue
        res: dict[str, RetrievalResult] = {}
        for i in range(0, len(questions), batch):
            chunk = questions[i : i + batch]
            for q, r in zip(
                chunk,
                retriever.retrieve_batch([q.question for q in chunk], **rc.model_dump()),
                strict=True,
            ):
                res[q.qid] = r
        out[key] = res
    return out


def retrieval_key(cfg: PipelineConfig) -> str:
    rc = cfg.retrieval.model_copy(update={"final_k": max(cfg.context_k, cfg.retrieval.final_k)})
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


def apply_verifier(records: list[dict[str, Any]], corpus: Corpus, prob: Any) -> None:
    """Verify every record whose pipeline asks for it, batching NLI calls across answers."""
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
