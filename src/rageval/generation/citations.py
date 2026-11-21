"""Parse model answers into statements with citations, and detect abstention."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
# citations written after the sentence terminator: "... in 1950. [1][2]" -> "... in 1950 [1][2]."
TRAILING = re.compile(r"([.!?])((?:\s*\[\d+(?:\s*[,;]\s*\d+)*\])+)")
ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Ft|Gen|Col|Lt|Sgt|Capt|Gov|Sen|Rep|Rev|Inc|Ltd|Co|Corp|No|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.",
)
INITIALS = re.compile(r"\b(?:[A-Z]\.){1,}(?=\s*[A-Z]?)")
SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(]?[A-Z0-9])")
ABSTAIN_RE = re.compile(
    r"(cannot|can't|can ?not|unable to|not able to)\s+(be\s+)?(answer|determine|find|tell|say|identify|confirm)"
    r"|(do|does|did)\s?(n't| not)\s+(contain|provide|mention|include|state|specify|say|give)"
    r"|(not|no)\s+(enough|sufficient)\s+(information|evidence|detail)"
    r"|insufficient\s+(information|evidence)"
    r"|no\s+(relevant\s+)?information"
    r"|(is|are)\s+not\s+(mentioned|stated|specified|provided)",
    re.IGNORECASE,
)
_DOT = "․"  # one-dot leader, a placeholder that never ends a sentence


@dataclass
class Statement:
    text: str  # sentence without citation markers
    raw: str
    citations: list[int] = field(
        default_factory=list
    )  # 1-based document numbers, deduped, in order
    abstain: bool = False


@dataclass
class ParsedAnswer:
    answer: str  # final answer text (CoT reasoning removed)
    statements: list[Statement]
    abstained: bool
    reasoning: str = ""

    @property
    def claims(self) -> list[Statement]:
        """Factual statements to verify (abstention sentences are not claims)."""
        return [s for s in self.statements if not s.abstain and s.text]


def extract_answer(text: str) -> tuple[str, str]:
    """Split a CoT response into (reasoning, answer). Non-CoT text is all answer."""
    reasoning = ""
    m = re.search(r"<reasoning>(.*?)(</reasoning>|<answer>|$)", text, re.S | re.I)
    if m:
        reasoning = m.group(1).strip()
    a = re.search(r"<answer>(.*?)(</answer>|$)", text, re.S | re.I)
    if a:
        return reasoning, a.group(1).strip()
    if m:  # reasoning but no answer tags: take whatever follows the reasoning block
        tail = text[m.end() :].strip()
        tail = re.sub(r"^(final\s+)?answer\s*:\s*", "", tail, flags=re.I)
        return reasoning, tail
    text = re.sub(r"^\s*(final\s+)?answer\s*:\s*", "", text.strip(), flags=re.I)
    return reasoning, text


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    text = TRAILING.sub(lambda m: f"{m.group(2)}{m.group(1)}", text)
    protected = ABBREV.sub(lambda m: m.group(0)[:-1] + _DOT, text)
    protected = INITIALS.sub(lambda m: m.group(0).replace(".", _DOT), protected)
    parts = [p.strip() for p in SPLIT.split(protected) if p.strip()]
    out = [p.replace(_DOT, ".") for p in parts]
    # a fragment that is only citations belongs to the previous sentence
    merged: list[str] = []
    for p in out:
        if merged and not CITE.sub("", p).strip(" .!?"):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def parse_citations(sentence: str) -> list[int]:
    seen: dict[int, None] = {}
    for m in CITE.finditer(sentence):
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            if n.isdigit():
                seen.setdefault(int(n), None)
    return list(seen)


def strip_citations(sentence: str) -> str:
    s = CITE.sub("", sentence)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def is_abstention(text: str) -> bool:
    return bool(ABSTAIN_RE.search(text))


def parse_answer(text: str, n_docs: int | None = None) -> ParsedAnswer:
    reasoning, answer = extract_answer(text)
    statements = []
    for sent in split_sentences(answer):
        cites = parse_citations(sent)
        if n_docs is not None:
            cites = [c for c in cites if 1 <= c <= n_docs]
        plain = strip_citations(sent)
        statements.append(Statement(plain, sent, cites, abstain=is_abstention(plain)))
    claims = [s for s in statements if not s.abstain and s.text]
    abstained = not answer.strip() or (not claims and any(s.abstain for s in statements))
    return ParsedAnswer(
        answer=answer, statements=statements, abstained=abstained, reasoning=reasoning
    )


def render(statements: list[Statement]) -> str:
    """Re-render statements with (possibly repaired) citations."""
    out = []
    for s in statements:
        body = s.text.rstrip()
        end = ""
        if body and body[-1] in ".!?":
            body, end = body[:-1], body[-1]
        cites = "".join(f"[{c}]" for c in s.citations)
        out.append(f"{body} {cites}{end}".replace("  ", " ") if cites else f"{body}{end}")
    return " ".join(out)
