import math

import pytest

from rageval.eval.retrieval_metrics import (
    all_recall_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_counts_fraction_of_gold():
    ranked = ["a", "x", "b", "y"]
    assert recall_at_k(ranked, ["a", "b"], 2) == 0.5
    assert recall_at_k(ranked, ["a", "b"], 3) == 1.0
    assert all_recall_at_k(ranked, ["a", "b"], 2) == 0.0
    assert all_recall_at_k(ranked, ["a", "b"], 3) == 1.0
    assert hit_at_k(ranked, ["b"], 2) == 0.0


def test_empty_gold_is_nan():
    assert math.isnan(recall_at_k(["a"], [], 10))


def test_mrr_and_ndcg():
    ranked = ["x", "a", "y", "b"]
    assert mrr_at_k(ranked, ["a", "b"], 10) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(ranked, ["a", "b"], 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b"], ["a", "b"], 10) == pytest.approx(1.0)
