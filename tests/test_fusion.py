import numpy as np
import pytest

from rageval.retrieval.fusion import fuse, rrf, weighted
from rageval.retrieval.hybrid import blend_rerank
from rageval.types import Hit


def H(*ids):
    return [Hit(d, 10.0 - i) for i, d in enumerate(ids)]


def test_rrf_rewards_agreement():
    out = rrf([H("a", "b", "c"), H("b", "a", "d")], k=60)
    assert [h.doc_id for h in out[:2]] == ["a", "b"] or [h.doc_id for h in out[:2]] == ["b", "a"]
    assert out[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert {h.doc_id for h in out} == {"a", "b", "c", "d"}


def test_weighted_alpha_extremes():
    sparse = [Hit("s1", 5.0), Hit("s2", 1.0)]
    dense = [Hit("d1", 0.9), Hit("d2", 0.1)]
    assert weighted(sparse, dense, alpha=1.0)[0].doc_id == "d1"
    assert weighted(sparse, dense, alpha=0.0)[0].doc_id == "s1"


def test_fuse_passthrough():
    s, d = H("a"), H("b")
    assert [h.doc_id for h in fuse(s, d, "bm25")] == ["a"]
    assert [h.doc_id for h in fuse(s, d, "dense")] == ["b"]
    with pytest.raises(ValueError):
        fuse(s, d, "nope")


def test_blend_rerank_pure_and_blended():
    fused = H("a", "b", "c")
    scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    assert [h.doc_id for h in blend_rerank(fused, scores, 0.0)] == ["b", "c", "a"]
    # with full blend the fused order wins
    assert [h.doc_id for h in blend_rerank(fused, scores, 1.0)] == ["a", "b", "c"]
