"""In-pipeline citation verification and repair with a small NLI model.

For each factual statement of an answer:
  1. If its citations jointly entail it, drop citations that are unnecessary (each alone does
     not entail it while the others still do), mirroring what ALCE penalises.
  2. If they do not, look for support among all context documents: the best single document,
     then the best pair among the top single-document candidates; re-cite on success.
  3. A statement nothing supports is flagged. Policy "repair" keeps it (with its original
     citations); policy "prune" removes it, and an answer left with no claims becomes an
     abstention. Policy "revise" (handled by the caller: gen_runner / the agent) sends the
     flagged sentences back to the generator once, then verifies the rewrite with "prune".
The judge used for evaluation (TRUE T5-11B, gpt-oss-120b) is a different model, so the pipeline is
not graded by the same model that repaired it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rageval.generation.citations import ParsedAnswer, Statement, render
from rageval.generation.prompts import ABSTAIN

ProbFn = Callable[[Sequence[str], Sequence[str]], np.ndarray]
AsyncProbFn = Callable[[Sequence[str], Sequence[str]], Awaitable[np.ndarray]]


def _premise(docs: Sequence[tuple[str, str]], idx: Sequence[int]) -> str:
    return "\n".join(f"Title: {docs[i][0]}\n{docs[i][1]}" for i in idx)


@dataclass
class VerifyStats:
    statements: int = 0
    kept: int = 0
    dropped_citations: int = 0
    recited: int = 0
    unsupported: int = 0
    pruned: int = 0
    nli_pairs: int = 0
    became_abstention: bool = False
    detail: list[dict[str, Any]] = field(default_factory=list)


class _Pairs:
    def __init__(self, aprob: AsyncProbFn) -> None:
        self.aprob = aprob
        self.cache: dict[tuple[str, str], float] = {}
        self.calls = 0

    async def get(self, reqs: list[tuple[str, str]]) -> list[float]:
        missing = list(dict.fromkeys(r for r in reqs if r not in self.cache))
        if missing:
            vals = await self.aprob([m[0] for m in missing], [m[1] for m in missing])
            self.calls += len(missing)
            for m, v in zip(missing, np.asarray(vals).tolist(), strict=True):
                self.cache[m] = float(v)
        return [self.cache[r] for r in reqs]


async def _minimize(
    cites: list[int], s: Statement, docs: Sequence[tuple[str, str]], P: _Pairs, tau: float
) -> list[int]:
    """Drop citations that are unnecessary: alone they do not entail the statement while the
    remaining citations still do (what ALCE penalises)."""
    if len(cites) < 2:
        return cites
    singles = await P.get([(_premise(docs, [c]), s.text) for c in cites])
    if max(singles) >= tau:  # one document is enough: keep the strongest single source
        return [cites[int(np.argmax(singles))]]
    keep = list(cites)
    for c in cites:
        if len(keep) == 1:
            break
        rest = [x for x in keep if x != c]
        if (await P.get([(_premise(docs, rest), s.text)]))[0] >= tau:
            keep = rest
    return keep


async def _check(
    s: Statement,
    docs: Sequence[tuple[str, str]],
    P: _Pairs,
    tau: float,
    pair_candidates: int,
) -> tuple[list[int] | None, dict[str, Any], str]:
    """Returns (new 0-based citations or None if unsupported, record, outcome).

    Search order when the cited documents do not entail the statement:
      1. complete the citation: cited + one more context document (two-hop sentences that cite
         only one of their two sources)
      2. the best single context document
      3. the best pair among the top context documents
    Every multi-document result is then minimised (unnecessary citations dropped).
    """
    n = len(docs)
    cites0 = [c - 1 for c in s.citations if 1 <= c <= n]
    record: dict[str, Any] = {"text": s.text, "before": [c + 1 for c in cites0]}
    joint_ok = bool(cites0) and (await P.get([(_premise(docs, cites0), s.text)]))[0] >= tau
    if joint_ok:
        return await _minimize(cites0, s, docs, P, tau), record, "kept"
    if 0 < len(cites0) < 3:
        extra = [d for d in range(n) if d not in cites0]
        probs = await P.get([(_premise(docs, sorted([*cites0, d])), s.text) for d in extra])
        j = int(np.argmax(probs)) if probs else -1
        if j >= 0 and probs[j] >= tau:
            return (
                await _minimize(sorted([*cites0, extra[j]]), s, docs, P, tau),
                record,
                "completed",
            )
    singles = await P.get([(_premise(docs, [i]), s.text) for i in range(n)])
    best = int(np.argmax(singles)) if singles else -1
    if best >= 0 and singles[best] >= tau:
        return [best], record, "recited"
    top = list(np.argsort(singles)[::-1][: max(pair_candidates, 2)])
    pairs = [(int(a), int(b)) for i, a in enumerate(top) for b in top[i + 1 :]]
    probs = await P.get([(_premise(docs, sorted(p)), s.text) for p in pairs]) if pairs else []
    j = int(np.argmax(probs)) if probs else -1
    if j >= 0 and probs[j] >= tau:
        return sorted(pairs[j]), record, "recited"
    return None, record, "unsupported"


async def averify_answer(
    parsed: ParsedAnswer,
    docs: Sequence[tuple[str, str]],
    aprob: AsyncProbFn,
    tau: float = 0.5,
    policy: str = "repair",
    pair_candidates: int = 6,
) -> tuple[ParsedAnswer, VerifyStats]:
    st = VerifyStats()
    if parsed.abstained:
        return parsed, st
    P = _Pairs(aprob)
    claims = [s for s in parsed.statements if not s.abstain and s.text]
    checks = await asyncio.gather(*(_check(s, docs, P, tau, pair_candidates) for s in claims))
    by_id = {id(s): c for s, c in zip(claims, checks, strict=True)}
    new_statements: list[Statement] = []
    for s in parsed.statements:
        if id(s) not in by_id:
            new_statements.append(s)
            continue
        cites, record, outcome = by_id[id(s)]
        st.statements += 1
        if outcome in ("recited", "completed"):
            st.recited += 1
        if cites is None:
            st.unsupported += 1
            record["unsupported"] = True
            if policy == "prune":
                st.pruned += 1
                record["after"] = None
                st.detail.append(record)
                continue
            cites = [c - 1 for c in s.citations if 1 <= c <= len(docs)]
        elif outcome == "kept":
            st.dropped_citations += len(record["before"]) - len(cites)
        st.kept += 1
        record["after"] = [c + 1 for c in cites]
        st.detail.append(record)
        new_statements.append(Statement(s.text, s.raw, [c + 1 for c in cites], s.abstain))
    st.nli_pairs = P.calls
    if not [s for s in new_statements if not s.abstain and s.text]:
        st.became_abstention = True
        abst = Statement(ABSTAIN, ABSTAIN, [], True)
        return ParsedAnswer(ABSTAIN, [abst], True, parsed.reasoning), st
    return ParsedAnswer(render(new_statements), new_statements, False, parsed.reasoning), st


def verify_answer(
    parsed: ParsedAnswer,
    docs: Sequence[tuple[str, str]],
    prob: ProbFn,
    tau: float = 0.5,
    policy: str = "repair",
    pair_candidates: int = 6,
) -> tuple[ParsedAnswer, VerifyStats]:
    """Synchronous wrapper (tests, scripts)."""

    async def aprob(p: Sequence[str], h: Sequence[str]) -> np.ndarray:
        return np.asarray(prob(p, h))

    return asyncio.run(averify_answer(parsed, docs, aprob, tau, policy, pair_candidates))


def prune_flagged(parsed: ParsedAnswer, st: VerifyStats) -> ParsedAnswer:
    """Drop the statements a verification pass flagged as unsupported (no extra NLI calls)."""
    bad = {d["text"] for d in st.detail if d.get("unsupported")}
    if not bad:
        return parsed
    kept = [s for s in parsed.statements if s.abstain or s.text not in bad]
    if not [s for s in kept if not s.abstain and s.text]:
        abst = Statement(ABSTAIN, ABSTAIN, [], True)
        return ParsedAnswer(ABSTAIN, [abst], True, parsed.reasoning)
    return ParsedAnswer(render(kept), kept, False, parsed.reasoning)
