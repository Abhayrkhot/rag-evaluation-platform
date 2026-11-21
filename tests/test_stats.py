import numpy as np
import pytest

from rageval.eval.stats import (
    bootstrap_ci,
    compare,
    holm,
    mcnemar_p,
    paired_bootstrap,
    ratio_bootstrap_ci,
)


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.7, 0.1, size=400)
    ci = bootstrap_ci(x)
    assert ci.lo < x.mean() < ci.hi
    assert ci.hi - ci.lo < 0.03


def test_paired_bootstrap_detects_shift():
    rng = np.random.default_rng(2)
    a = rng.random(300)
    b = a + 0.05 + rng.normal(0, 0.01, 300)
    d, lo, _hi, p = paired_bootstrap(a, b)
    assert d == pytest.approx(0.05, abs=0.005)
    assert lo > 0 and p < 0.01


def test_paired_bootstrap_null():
    rng = np.random.default_rng(3)
    a = rng.random(300)
    b = a + rng.normal(0, 0.05, 300)
    _, lo, hi, p = paired_bootstrap(a, b)
    assert lo < 0 < hi and p > 0.05


def test_mcnemar_and_holm():
    a = [False] * 50 + [True] * 50
    b = [True] * 50 + [True] * 50
    assert mcnemar_p(a, b) < 1e-10
    assert mcnemar_p(a, a) == 1.0
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_ratio_ci_and_compare():
    ci = ratio_bootstrap_ci([1, 2, 3], [2, 2, 4])
    assert ci.mean == pytest.approx(6 / 8)
    r = compare("m", "a", [0, 0, 1, 1], "b", [1, 1, 1, 1], binary=True)
    assert r.delta == pytest.approx(0.5)
    assert r.p_mcnemar is not None
