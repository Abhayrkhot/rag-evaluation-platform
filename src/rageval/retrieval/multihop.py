"""Two-hop retrieval for bridge questions.

HotpotQA bridge questions name only the first entity ("the woman who portrayed Corliss Archer in
Kiss and Tell"); the paragraph that holds the answer is about a second entity that only appears
inside the first paragraph. Single-shot retrieval misses that paragraph whenever the question and
it share few words. After the first hybrid + rerank pass, the planner LLM reads the top passages
and names the missing article (or says NONE); a second hybrid + rerank pass runs on that query, and
its top hits are spliced in right after the first-hop head. The planner always names an article
(an earlier version could answer NONE and declined far too often; see docs/RESULTS_LOG.md):

    merged = hop1[:keep] + hop2[:take] + hop1[keep:]   (deduplicated)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rageval.corpus import Corpus
from rageval.llm.client import ChatClient, ChatResult, gather_limited
from rageval.retrieval.hybrid import HybridRetriever, RetrievalConfig

PLAN_SYSTEM = """You plan searches over Wikipedia to answer multi-hop questions.
Most questions need two articles: one about an entity named in the question, and one about a second entity that is only identified through the first (for example, the inventor of a product named in the question, or the director of a film named in the question).
Read the question and the passages already retrieved, then reply with the title of the Wikipedia article that is still missing.
If the passages already cover every entity the question depends on, reply with the title of the article about the entity whose fact the question asks for.
Use what the passages say to identify the second entity. Reply with the title only, on one line."""


def plan_messages(question: str, docs: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    ctx = "\n\n".join(f"[{i}] Title: {t}\n{x}" for i, (t, x) in enumerate(docs, start=1))
    return [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{ctx}"},
    ]


def parse_followup(text: str, max_words: int = 16) -> str | None:
    line = next((x.strip() for x in text.strip().splitlines() if x.strip()), "")
    line = (
        re.sub(r"^(search\s+query|query)\s*:\s*", "", line, flags=re.I)
        .strip()
        .strip("\"'`*")
        .strip()
    )
    if not line or line.upper().startswith("NONE") or len(line.split()) > max_words:
        return None
    return line


def merge_hops(hop1: Sequence[str], hop2: Sequence[str], keep: int, take: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(d: str) -> None:
        if d not in seen:
            seen.add(d)
            out.append(d)

    for d in hop1[:keep]:
        add(d)
    n = 0
    for d in hop2:
        if n >= take:
            break
        if d not in seen:
            add(d)
            n += 1
    for d in hop1[keep:]:
        add(d)
    return out


@dataclass
class MultiHopResult:
    question: str
    hop1: list[str]
    followup: str | None
    hop2: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    planner_ms: float = 0.0
    planner_tokens: int = 0


async def plan_batch(
    client: ChatClient,
    corpus: Corpus,
    questions: Sequence[str],
    hop1: Sequence[Sequence[str]],
    plan_k: int,
    concurrency: int = 128,
    params: dict[str, Any] | None = None,
) -> list[ChatResult | BaseException]:
    coros = [
        client.chat(
            plan_messages(q, [(corpus[d].title, corpus[d].text) for d in h[:plan_k]]),
            max_tokens=32,
            temperature=0.0,
            **(params or {}),
        )
        for q, h in zip(questions, hop1, strict=True)
    ]
    return await gather_limited(coros, concurrency)


async def second_hop(
    retriever: HybridRetriever,
    client: ChatClient,
    questions: Sequence[str],
    hop1: Sequence[Sequence[str]],
    cfg: RetrievalConfig,
    plan_k: int = 5,
    hop2_depth: int = 30,
    concurrency: int = 128,
    params: dict[str, Any] | None = None,
) -> list[MultiHopResult]:
    """Plan a follow-up query from each question's first-hop ranking and retrieve for it.
    `merged` is left empty: the splice (keep, take) is applied by the caller."""
    plans = await plan_batch(client, retriever.corpus, questions, hop1, plan_k, concurrency, params)
    out: list[MultiHopResult] = []
    for q, h, p in zip(questions, hop1, plans, strict=True):
        if isinstance(p, ChatResult):
            out.append(
                MultiHopResult(
                    q,
                    list(h),
                    parse_followup(p.text),
                    planner_ms=p.latency_ms,
                    planner_tokens=p.completion_tokens,
                )
            )
        else:
            out.append(MultiHopResult(q, list(h), None))
    todo = [i for i, m in enumerate(out) if m.followup]
    if todo:
        hop2_cfg = cfg.model_copy(update={"final_k": 10, "rerank_depth": hop2_depth}).model_dump()
        for s in range(0, len(todo), 256):
            chunk = todo[s : s + 256]
            r2 = retriever.retrieve_batch([out[i].followup or "" for i in chunk], **hop2_cfg)
            for i, r in zip(chunk, r2, strict=True):
                out[i].hop2 = r.doc_ids
    return out


async def multihop_batch(
    retriever: HybridRetriever,
    client: ChatClient,
    questions: Sequence[str],
    cfg: RetrievalConfig,
    plan_k: int = 5,
    hop2_depth: int = 30,
    keep: int = 3,
    take: int = 3,
    concurrency: int = 128,
    params: dict[str, Any] | None = None,
    depth_out: int = 100,
) -> list[MultiHopResult]:
    """Full two-hop retrieval: first hop online, then plan, second hop and splice."""
    base = cfg.model_copy(update={"final_k": depth_out}).model_dump()
    hop1: list[list[str]] = []
    for s in range(0, len(questions), 256):
        hop1 += [r.doc_ids for r in retriever.retrieve_batch(list(questions[s : s + 256]), **base)]
    out = await second_hop(
        retriever, client, questions, hop1, cfg, plan_k, hop2_depth, concurrency, params
    )
    for m in out:
        m.merged = merge_hops(m.hop1, m.hop2, keep, take) if m.hop2 else list(m.hop1)
    return out
