"""Core records shared by ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Doc:
    """One retrievable unit (a Wikipedia intro paragraph in the HotpotQA corpus).

    ``sentences`` keeps the source sentence split so supporting-fact labels and sentence-level
    citations can be checked; ``text`` is their concatenation.
    """

    doc_id: str
    title: str
    sentences: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(self.sentences).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Doc:
        return cls(
            doc_id=d["doc_id"],
            title=d["title"],
            sentences=list(d["sentences"]),
            meta=dict(d.get("meta") or {}),
        )


@dataclass(slots=True)
class Question:
    qid: str
    question: str
    answer: str
    qtype: str  # bridge | comparison
    answerable: bool
    gold_doc_ids: list[str]
    # (doc_id, sentence index) pairs from HotpotQA supporting_facts
    gold_sentences: list[tuple[str, int]] = field(default_factory=list)
    split: str = "test"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gold_sentences"] = [list(x) for x in self.gold_sentences]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Question:
        return cls(
            qid=d["qid"],
            question=d["question"],
            answer=d["answer"],
            qtype=d["qtype"],
            answerable=bool(d["answerable"]),
            gold_doc_ids=list(d["gold_doc_ids"]),
            gold_sentences=[(str(a), int(b)) for a, b in d.get("gold_sentences", [])],
            split=d.get("split", "test"),
        )


@dataclass(slots=True)
class Hit:
    """A scored retrieval result."""

    doc_id: str
    score: float
    source: str = ""  # bm25 | dense | fused | rerank

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "score": float(self.score), "source": self.source}
