"""In-memory document store and suite loading."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rageval.io import read_jsonl
from rageval.types import Doc, Question


class Corpus:
    def __init__(self, docs: list[Doc]) -> None:
        self.docs = docs
        self.by_id = {d.doc_id: d for d in docs}
        if len(self.by_id) != len(docs):
            raise ValueError("duplicate doc_id in corpus")

    @classmethod
    def load(cls, path: str | Path) -> Corpus:
        return cls([Doc.from_dict(r) for r in read_jsonl(path)])

    def __len__(self) -> int:
        return len(self.docs)

    def __iter__(self) -> Iterator[Doc]:
        return iter(self.docs)

    def __getitem__(self, doc_id: str) -> Doc:
        return self.by_id[doc_id]

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self.by_id

    @property
    def ids(self) -> list[str]:
        return [d.doc_id for d in self.docs]

    def index_text(self, doc_id: str) -> str:
        """Text used for sparse indexing and reranking: title line + paragraph."""
        d = self.by_id[doc_id]
        return f"{d.title}\n{d.text}"


def load_suite(path: str | Path) -> list[Question]:
    return [Question.from_dict(r) for r in read_jsonl(path)]
