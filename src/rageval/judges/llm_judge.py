"""LLM-as-judge (gpt-oss-120b by default, a different model family from the Qwen generator).

claims      : for each factual statement of an answer, is it fully supported by the documents the
              generator was shown? Unsupported statements are the "unsupported claims" metric.
correctness : does the response give the gold answer to the question?
citation    : does one cited document (alone) support a statement? Cross-checks the NLI judge's
              citation precision.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from rageval.llm.client import ChatClient, gather_limited

CLAIMS_SYSTEM = """You verify whether statements are supported by source documents.
A statement is SUPPORTED only if the documents state or directly imply everything it asserts (names, dates, numbers, relations). Background knowledge does not count. A statement that is partly supported is UNSUPPORTED.
Reply with one line per statement, in order, formatted exactly as:
1: SUPPORTED
2: UNSUPPORTED"""

CORRECT_SYSTEM = """You grade answers to questions against a reference answer.
The response is CORRECT if it clearly gives an answer equivalent to the reference (aliases, paraphrases and extra correct detail are fine). It is INCORRECT if it gives a different answer, hedges between answers, or declines to answer.
Reply with exactly one word: CORRECT or INCORRECT."""

CITE_SYSTEM = """You check whether a single document supports a statement.
Reply FULL if the document alone states or directly implies everything in the statement, PARTIAL if it supports some but not all of it, NONE otherwise. Reply with exactly one word."""


def _docs_block(docs: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{i}] Title: {t}\n{x}" for i, (t, x) in enumerate(docs, start=1))


@dataclass
class ClaimVerdicts:
    supported: list[bool | None]  # None = judge output unparseable
    raw: str


class LLMJudge:
    def __init__(
        self, client: ChatClient, concurrency: int = 256, reasoning_effort: str = "low"
    ) -> None:
        self.client = client
        self.concurrency = concurrency
        self.params: dict[str, Any] = (
            {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        )

    async def _ask(self, system: str, user: str, max_tokens: int) -> str:
        r = await self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0,
            **self.params,
        )
        return r.text

    async def claims(
        self, items: Sequence[tuple[Sequence[tuple[str, str]], Sequence[str]]]
    ) -> list[ClaimVerdicts]:
        """items: (documents, statements) per answer."""

        def prompt(docs: Sequence[tuple[str, str]], stmts: Sequence[str]) -> str:
            lines = "\n".join(f"{i}: {s}" for i, s in enumerate(stmts, start=1))
            return f"Documents:\n{_docs_block(docs)}\n\nStatements:\n{lines}"

        coros = [self._ask(CLAIMS_SYSTEM, prompt(d, s), 2048) for d, s in items if s]
        raw = await gather_limited(coros, self.concurrency)
        out: list[ClaimVerdicts] = []
        it = iter(raw)
        for _, stmts in items:
            if not stmts:
                out.append(ClaimVerdicts([], ""))
                continue
            r = next(it)
            text = r if isinstance(r, str) else f"ERROR {r}"
            out.append(ClaimVerdicts(parse_claim_lines(text, len(stmts)), text))
        return out

    async def correctness(self, items: Sequence[tuple[str, str, str]]) -> list[bool | None]:
        """items: (question, reference answer, response)."""
        coros = [
            self._ask(CORRECT_SYSTEM, f"Question: {q}\nReference answer: {g}\nResponse: {r}", 1024)
            for q, g, r in items
        ]
        raw = await gather_limited(coros, self.concurrency)
        return [
            cast(bool | None, parse_word(x, {"CORRECT": True, "INCORRECT": False}))
            if isinstance(x, str)
            else None
            for x in raw
        ]

    async def citation(self, items: Sequence[tuple[tuple[str, str], str]]) -> list[str | None]:
        """items: ((title, text), statement) -> FULL | PARTIAL | NONE."""
        coros = [
            self._ask(CITE_SYSTEM, f"Document:\nTitle: {d[0]}\n{d[1]}\n\nStatement: {s}", 1024)
            for d, s in items
        ]
        raw = await gather_limited(coros, self.concurrency)
        return [
            cast(str | None, parse_word(x, {"FULL": "FULL", "PARTIAL": "PARTIAL", "NONE": "NONE"}))
            if isinstance(x, str)
            else None
            for x in raw
        ]


def parse_claim_lines(text: str, n: int) -> list[bool | None]:
    out: list[bool | None] = [None] * n
    for m in re.finditer(
        r"(?m)^\s*\**\s*(\d+)\s*[:.)\-]\s*\**\s*(UNSUPPORTED|SUPPORTED|NOT SUPPORTED)", text, re.I
    ):
        i = int(m.group(1)) - 1
        if 0 <= i < n:
            out[i] = m.group(2).upper() == "SUPPORTED"
    return out


def parse_word(text: str, mapping: dict[str, object]) -> object | None:
    # longest keys first so INCORRECT is not read as CORRECT
    t = text.strip().upper()
    for k in sorted(mapping, key=len, reverse=True):
        if re.search(rf"\b{k}\b", t):
            return mapping[k]
    return None
