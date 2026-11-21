"""ALCE citation metrics (Gao et al., 2023) over parsed answers, with any NLI judge.

Citation recall (per statement): the concatenation of its cited documents entails it.
Citation precision (per citation): following ALCE's reference implementation, a citation is
imprecise if the statement's citations jointly fail to entail it, or if the statement has
several citations, this one alone does not entail the statement, and the remaining citations
still entail it without it (the citation is unnecessary).
Scores are averaged per answer, then over answers (macro), as in ALCE.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

EntailFn = Callable[[Sequence[str], Sequence[str]], list[bool]]


def format_doc(title: str, text: str) -> str:
    return f"Title: {title}\n{text}"


@dataclass
class StatementInput:
    text: str
    citations: list[int]  # 0-based indexes into the answer's document list


@dataclass
class AnswerInput:
    key: str
    docs: list[tuple[str, str]]  # (title, text) in the order shown to the generator
    statements: list[StatementInput]


@dataclass
class AnswerScore:
    key: str
    citation_recall: float
    citation_precision: float
    n_statements: int
    n_citations: int
    statement_supported: list[bool] = field(default_factory=list)
    citation_precise: list[list[bool]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class _Batcher:
    """Collects (premise, hypothesis) pairs, deduplicates, and resolves them in one judge call."""

    def __init__(self, entail: EntailFn) -> None:
        self.entail = entail
        self.pending: dict[tuple[str, str], None] = {}
        self.done: dict[tuple[str, str], bool] = {}

    def ask(self, premise: str, hypothesis: str) -> tuple[str, str]:
        k = (premise, hypothesis)
        if k not in self.done:
            self.pending.setdefault(k, None)
        return k

    def resolve(self) -> None:
        if not self.pending:
            return
        keys = list(self.pending)
        res = self.entail([k[0] for k in keys], [k[1] for k in keys])
        for k, v in zip(keys, res, strict=True):
            self.done[k] = bool(v)
        self.pending.clear()

    def __getitem__(self, k: tuple[str, str]) -> bool:
        return self.done[k]


def _premise(docs: list[tuple[str, str]], idx: Sequence[int]) -> str:
    return "\n".join(format_doc(*docs[i]) for i in idx)


def score_answers(answers: Sequence[AnswerInput], entail: EntailFn) -> list[AnswerScore]:
    b = _Batcher(entail)
    # phase 1: joint entailment of each statement by all of its citations
    joint: dict[tuple[int, int], tuple[str, str]] = {}
    for ai, a in enumerate(answers):
        for si, s in enumerate(a.statements):
            cites = [c for c in s.citations if 0 <= c < len(a.docs)]
            if cites:
                joint[(ai, si)] = b.ask(_premise(a.docs, cites), s.text)
    b.resolve()
    # phase 2: for jointly supported multi-citation statements, single-doc and leave-one-out checks
    single: dict[tuple[int, int, int], tuple[str, str]] = {}
    for (ai, si), k in joint.items():
        s = answers[ai].statements[si]
        cites = [c for c in s.citations if 0 <= c < len(answers[ai].docs)]
        if b[k] and len(cites) > 1:
            for c in cites:
                single[(ai, si, c)] = b.ask(_premise(answers[ai].docs, [c]), s.text)
    b.resolve()
    loo: dict[tuple[int, int, int], tuple[str, str]] = {}
    for (ai, si, c), k in single.items():
        if not b[k]:
            s = answers[ai].statements[si]
            rest = [x for x in s.citations if x != c and 0 <= x < len(answers[ai].docs)]
            loo[(ai, si, c)] = b.ask(_premise(answers[ai].docs, rest), s.text)
    b.resolve()

    out: list[AnswerScore] = []
    for ai, a in enumerate(answers):
        supported: list[bool] = []
        precise: list[list[bool]] = []
        n_cit = 0
        for si, s in enumerate(a.statements):
            cites = [c for c in s.citations if 0 <= c < len(a.docs)]
            # out-of-range citation numbers count as citations that cannot be precise
            invalid = len(s.citations) - len(cites)
            ok = bool(cites) and b[joint[(ai, si)]]
            supported.append(ok)
            flags: list[bool] = []
            if ok and len(cites) > 1:
                for c in cites:
                    if b[single[(ai, si, c)]]:
                        flags.append(True)
                    else:
                        flags.append(not b[loo[(ai, si, c)]])
            else:
                flags = [ok] * len(cites)
            flags += [False] * invalid
            precise.append(flags)
            n_cit += len(flags)
        n_st = len(a.statements)
        out.append(
            AnswerScore(
                key=a.key,
                citation_recall=sum(supported) / n_st if n_st else float("nan"),
                citation_precision=sum(sum(f) for f in precise) / n_cit if n_cit else 0.0,
                n_statements=n_st,
                n_citations=n_cit,
                statement_supported=supported,
                citation_precise=precise,
            )
        )
    return out
