"""Judge generation records: ALCE citation metrics (NLI) + claims and correctness (LLM)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from rageval.corpus import Corpus
from rageval.eval.answer_metrics import answer_match
from rageval.eval.grounding import AnswerInput, EntailFn, StatementInput, score_answers


def rec_key(r: dict[str, Any]) -> str:
    return f"{r['pipeline']}::{r['qid']}"


def claims_of(r: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in r["final"]["statements"] if not s["abstain"] and s["text"]]


def answered(r: dict[str, Any]) -> bool:
    return not r.get("error") and not r["final"]["abstained"] and bool(claims_of(r))


def context_docs(r: dict[str, Any], corpus: Corpus) -> list[tuple[str, str]]:
    return [(corpus[d].title, corpus[d].text) for d in r["context"]]


def nli_citations(
    records: Sequence[dict[str, Any]], corpus: Corpus, entail: EntailFn
) -> dict[str, dict[str, Any]]:
    inputs = []
    for r in records:
        if not answered(r):
            continue
        stmts = [StatementInput(s["text"], [c - 1 for c in s["citations"]]) for s in claims_of(r)]
        inputs.append(AnswerInput(rec_key(r), context_docs(r, corpus), stmts))
    t0 = time.perf_counter()
    scores = score_answers(inputs, entail)
    dt = time.perf_counter() - t0
    out = {s.key: s.to_dict() for s in scores}
    out["_meta"] = {"seconds": dt, "answers": len(inputs)}
    return out


async def llm_judgments(
    records: Sequence[dict[str, Any]], corpus: Corpus, judge: Any
) -> dict[str, dict[str, Any]]:
    claim_items = []
    claim_keys = []
    for r in records:
        if answered(r):
            claim_items.append((context_docs(r, corpus), [s["text"] for s in claims_of(r)]))
            claim_keys.append(rec_key(r))
    corr_items = []
    corr_keys = []
    for r in records:
        if not r.get("error") and not r["final"]["abstained"]:
            corr_items.append((r["question"], r["gold_answer"], r["final"]["answer"]))
            corr_keys.append(rec_key(r))
    t0 = time.perf_counter()
    claims, correct = await asyncio.gather(judge.claims(claim_items), judge.correctness(corr_items))
    dt = time.perf_counter() - t0
    out: dict[str, dict[str, Any]] = {}
    for k, v in zip(claim_keys, claims, strict=True):
        out.setdefault(k, {})["claims_supported"] = v.supported
        out[k]["claims_raw"] = v.raw[-2000:]
    for k, c in zip(corr_keys, correct, strict=True):
        out.setdefault(k, {})["judge_correct"] = c
    out["_meta"] = {
        "seconds": dt,
        "claim_calls": len(claim_items),
        "correct_calls": len(corr_items),
    }
    return out


def merge(
    records: Sequence[dict[str, Any]], nli: dict[str, Any] | None, llm: dict[str, Any] | None
) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        k = rec_key(r)
        a = answered(r)
        row: dict[str, Any] = {
            "pipeline": r["pipeline"],
            "qid": r["qid"],
            "qtype": r["qtype"],
            "answerable": r["answerable"],
            "error": r.get("error"),
            "abstained": r["final"]["abstained"] or not claims_of(r),
            "n_claims": len(claims_of(r)) if a else 0,
            "context_recall": r.get("context_recall"),
            "str_match": answer_match(r["final"]["answer"], r["gold_answer"]) if a else False,
            "latency_ms": r.get("latency_ms"),
            "completion_tokens": r.get("completion_tokens"),
            "verify": r.get("verify"),
        }
        if nli is not None and k in nli:
            s = nli[k]
            row.update(
                cit_recall=s["citation_recall"],
                cit_precision=s["citation_precision"],
                n_citations=s["n_citations"],
                nli_supported=s["statement_supported"],
            )
        if llm is not None:
            j = llm.get(k, {})
            sup = j.get("claims_supported")
            if a and sup is not None:
                row["claims_supported"] = sup
                row["n_unsupported"] = sum(1 for x in sup if x is False)
                row["n_unparsed"] = sum(1 for x in sup if x is None)
            row["judge_correct"] = j.get("judge_correct", False if row["abstained"] else None)
        rows.append(row)
    return rows
