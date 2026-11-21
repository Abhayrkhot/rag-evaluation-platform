"""Sparse retrieval: Okapi BM25 (Lucene variant) via bm25s with Snowball stemming."""

from __future__ import annotations

import json
from pathlib import Path

import bm25s
import Stemmer
from bm25s.tokenization import Tokenizer

from rageval.types import Hit


class BM25Index:
    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:
        # k1/b defaults follow Anserini's BEIR "flat" baselines, which beat the 1.2/0.75 textbook
        # values on Wikipedia paragraphs; they are re-tuned on dev in M1.
        self.k1 = k1
        self.b = b
        self._stemmer = Stemmer.Stemmer("english")
        self.tokenizer = Tokenizer(stemmer=self._stemmer, stopwords="en")
        self.model = bm25s.BM25(k1=k1, b=b)
        self.doc_ids: list[str] = []

    def build(self, doc_ids: list[str], texts: list[str]) -> BM25Index:
        tokens = self.tokenizer.tokenize(texts, return_as="tuple", show_progress=False)
        self.model.index(tokens, show_progress=False)
        self.doc_ids = list(doc_ids)
        return self

    def search(self, queries: list[str], k: int = 100, n_threads: int = 0) -> list[list[Hit]]:
        if not queries:
            return []
        q = self.tokenizer.tokenize(
            queries, update_vocab=False, return_as="tuple", show_progress=False
        )
        k = min(k, len(self.doc_ids))
        idx, scores = self.model.retrieve(q, k=k, show_progress=False, n_threads=n_threads)
        out: list[list[Hit]] = []
        for row_i, row_s in zip(idx, scores, strict=True):
            out.append(
                [
                    Hit(self.doc_ids[int(i)], float(s), "bm25")
                    for i, s in zip(row_i, row_s, strict=True)
                    if s > 0
                ]
            )
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / "bm25s"))
        self.tokenizer.save_vocab(str(path / "bm25s"))
        self.tokenizer.save_stopwords(str(path / "bm25s"))
        (path / "doc_ids.json").write_text(json.dumps(self.doc_ids))
        (path / "params.json").write_text(json.dumps({"k1": self.k1, "b": self.b}))

    @classmethod
    def load(cls, path: str | Path, mmap: bool = True) -> BM25Index:
        path = Path(path)
        params = json.loads((path / "params.json").read_text())
        self = cls(k1=params["k1"], b=params["b"])
        self.model = bm25s.BM25.load(str(path / "bm25s"), mmap=mmap)
        self.tokenizer.load_vocab(str(path / "bm25s"))
        self.tokenizer.load_stopwords(str(path / "bm25s"))
        self.doc_ids = json.loads((path / "doc_ids.json").read_text())
        return self
