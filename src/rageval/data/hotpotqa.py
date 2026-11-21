"""Build the 100K-document HotpotQA corpus and the evaluation suites.

Corpus: every paragraph that appears in the HotpotQA validation (distractor) contexts, which
includes the eight TF-IDF hard negatives HotpotQA mined from full Wikipedia for each question,
topped up with paragraphs sampled from the train contexts, minus the supporting paragraphs of the
questions chosen to be unanswerable. Paragraphs are keyed by Wikipedia title.

Suites (all from the validation split, disjoint):
  test  : the 500-question evaluation suite (answerable in the natural bridge/comparison ratio,
          plus unanswerable questions whose evidence was removed from the corpus)
  dev   : used for every tuning decision (fusion weights, depths, prompts, thresholds)
  full  : every other answerable validation question, for retrieval-only metrics with tight CIs
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rageval.io import sha256_file, write_json, write_jsonl
from rageval.types import Doc, Question

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"


def doc_id_for(title: str) -> str:
    return "h" + hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]


@dataclass
class BuildConfig:
    n_docs: int = 100_000
    n_test: int = 500
    n_test_unanswerable: int = 100
    n_dev: int = 250
    n_dev_unanswerable: int = 50
    seed: int = 13
    revision: str | None = None


def _gold_titles(ex: dict[str, Any]) -> list[str]:
    seen: dict[str, None] = {}
    for t in ex["supporting_facts"]["title"]:
        seen.setdefault(t, None)
    return list(seen)


def _stratified(pool: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Sample n examples keeping the pool's qtype proportions (largest-remainder rounding)."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for ex in pool:
        by_type.setdefault(ex["type"], []).append(ex)
    total = len(pool)
    quotas = {t: n * len(v) / total for t, v in by_type.items()}
    alloc = {t: int(q) for t, q in quotas.items()}
    rest = n - sum(alloc.values())
    for t in sorted(quotas, key=lambda t: quotas[t] - alloc[t], reverse=True)[:rest]:
        alloc[t] += 1
    out: list[dict[str, Any]] = []
    for t in sorted(by_type):
        items = by_type[t][:]
        rng.shuffle(items)
        out.extend(items[: alloc[t]])
    rng.shuffle(out)
    return out


def _paragraphs(examples: Iterable[dict[str, Any]]) -> tuple[dict[str, list[str]], int]:
    store: dict[str, list[str]] = {}
    conflicts = 0
    for ex in examples:
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"], strict=True):
            prev = store.get(title)
            if prev is None:
                store[title] = list(sents)
            elif prev != list(sents):
                conflicts += 1
    return store, conflicts


def _question(ex: dict[str, Any], answerable: bool, split: str) -> Question:
    golds = _gold_titles(ex)
    return Question(
        qid=ex["id"],
        question=ex["question"].strip(),
        answer=ex["answer"].strip(),
        qtype=ex["type"],
        answerable=answerable,
        gold_doc_ids=[doc_id_for(t) for t in golds],
        gold_sentences=[
            (doc_id_for(t), int(s))
            for t, s in zip(
                ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"], strict=True
            )
        ],
        split=split,
    )


def build(out_dir: str | Path, cfg: BuildConfig | None = None) -> dict[str, Any]:
    from datasets import load_dataset

    cfg = cfg or BuildConfig()
    out = Path(out_dir)
    rng = random.Random(cfg.seed)

    val = [
        dict(x) for x in load_dataset(DATASET, CONFIG, split="validation", revision=cfg.revision)
    ]
    train = load_dataset(DATASET, CONFIG, split="train", revision=cfg.revision)

    val_paras, val_conflicts = _paragraphs(val)
    for ex in val:
        for t in _gold_titles(ex):
            assert t in val_paras, f"gold paragraph missing from context: {t}"

    order = val[:]
    rng.shuffle(order)

    # 1) Unanswerable questions: their supporting paragraphs are removed from the corpus.
    n_unans = cfg.n_test_unanswerable + cfg.n_dev_unanswerable
    unans = _stratified(order, n_unans, rng)
    removed: set[str] = set()
    for ex in unans:
        removed.update(_gold_titles(ex))
    unans_ids = {ex["id"] for ex in unans}

    # 2) Answerable pool: questions none of whose supporting paragraphs were removed.
    pool = [
        ex for ex in order if ex["id"] not in unans_ids and not (set(_gold_titles(ex)) & removed)
    ]
    n_test_ans = cfg.n_test - cfg.n_test_unanswerable
    n_dev_ans = cfg.n_dev - cfg.n_dev_unanswerable
    test_ans = _stratified(pool, n_test_ans, rng)
    taken = {ex["id"] for ex in test_ans}
    dev_ans = _stratified([ex for ex in pool if ex["id"] not in taken], n_dev_ans, rng)
    dev_ids = {ex["id"] for ex in dev_ans}

    unans_test = unans[: cfg.n_test_unanswerable]
    unans_dev = unans[cfg.n_test_unanswerable :]

    test = [_question(ex, True, "test") for ex in test_ans] + [
        _question(ex, False, "test") for ex in unans_test
    ]
    dev = [_question(ex, True, "dev") for ex in dev_ans] + [
        _question(ex, False, "dev") for ex in unans_dev
    ]
    rng.shuffle(test)
    rng.shuffle(dev)
    full = [_question(ex, True, "full") for ex in pool if ex["id"] not in dev_ids]

    # 3) Corpus: validation paragraphs minus removed, topped up from train paragraphs.
    corpus_titles = [t for t in val_paras if t not in removed]
    need = cfg.n_docs - len(corpus_titles)
    assert need >= 0, "validation paragraphs alone exceed n_docs"
    train_paras, train_conflicts = _paragraphs(dict(x) for x in train)
    extra_pool = sorted(t for t in train_paras if t not in val_paras and t not in removed)
    rng.shuffle(extra_pool)
    extra = extra_pool[:need]
    docs: list[Doc] = []
    for t in corpus_titles:
        docs.append(Doc(doc_id_for(t), t, val_paras[t], {"source": "hotpotqa-validation"}))
    for t in extra:
        docs.append(Doc(doc_id_for(t), t, train_paras[t], {"source": "hotpotqa-train"}))
    docs.sort(key=lambda d: d.doc_id)
    ids = [d.doc_id for d in docs]
    assert len(set(ids)) == len(ids) == cfg.n_docs, "doc id collision or wrong corpus size"
    present = set(ids)
    for q in test + dev + full:
        if q.answerable:
            assert all(g in present for g in q.gold_doc_ids), q.qid
        else:
            assert not any(g in present for g in q.gold_doc_ids), q.qid

    write_jsonl(out / "corpus.jsonl", (d.to_dict() for d in docs))
    write_jsonl(out / "test.jsonl", (q.to_dict() for q in test))
    write_jsonl(out / "dev.jsonl", (q.to_dict() for q in dev))
    write_jsonl(out / "full.jsonl", (q.to_dict() for q in full))

    def mix(qs: list[Question]) -> dict[str, int]:
        c = Counter(f"{q.qtype}/{'answerable' if q.answerable else 'unanswerable'}" for q in qs)
        return dict(sorted(c.items()))

    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "revision": cfg.revision,
        "build": cfg.__dict__,
        "n_docs": len(docs),
        "docs_from_validation": len(corpus_titles),
        "docs_from_train": len(extra),
        "removed_paragraphs": len(removed),
        "paragraph_text_conflicts": {"validation": val_conflicts, "train": train_conflicts},
        "suites": {"test": mix(test), "dev": mix(dev), "full": mix(full)},
        "sha256": {
            name: sha256_file(out / name)
            for name in ("corpus.jsonl", "test.jsonl", "dev.jsonl", "full.jsonl")
        },
    }
    write_json(out / "manifest.json", manifest)
    return manifest
