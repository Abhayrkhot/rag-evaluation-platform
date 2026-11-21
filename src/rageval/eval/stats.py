"""Confidence intervals and paired significance tests for per-question metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats

Vec = Sequence[float] | Sequence[bool] | np.ndarray[Any, Any]


@dataclass
class CI:
    mean: float
    lo: float
    hi: float
    n: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def fmt(self, pct: bool = True, digits: int = 1) -> str:
        m = 100.0 if pct else 1.0
        return f"{self.mean * m:.{digits}f} [{self.lo * m:.{digits}f}, {self.hi * m:.{digits}f}]"


def bootstrap_ci(x: Vec, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> CI:
    a = np.asarray(x, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(a.mean()), float(lo), float(hi), int(a.size))


def ratio_bootstrap_ci(
    num: Vec,
    den: Vec,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """CI for sum(num)/sum(den) (micro-average), resampling questions."""
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    if d.sum() == 0:
        return CI(float("nan"), float("nan"), float("nan"), int(n.size))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n.size, size=(n_boot, n.size))
    ds = d[idx].sum(axis=1)
    ratios = np.where(ds > 0, n[idx].sum(axis=1) / np.maximum(ds, 1e-12), np.nan)
    lo, hi = np.nanquantile(ratios, [alpha / 2, 1 - alpha / 2])
    return CI(float(n.sum() / d.sum()), float(lo), float(hi), int(n.size))


@dataclass
class PairedResult:
    metric: str
    a: str
    b: str
    n: int
    mean_a: float
    mean_b: float
    delta: float  # mean_b - mean_a
    delta_lo: float
    delta_hi: float
    p_bootstrap: float
    p_wilcoxon: float
    p_mcnemar: float | None = None
    p_holm: float | None = None

    def to_dict(self) -> dict[str, float | str | int | None]:
        return asdict(self)


def paired_bootstrap(
    a: Vec, b: Vec, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float, float]:
    """Paired bootstrap on per-question differences. Returns (delta, lo, hi, two-sided p)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    diff = y[mask] - x[mask]
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    # p-value: fraction of the null-centred bootstrap distribution at least as extreme
    centred = means - diff.mean()
    p = float((np.abs(centred) >= abs(diff.mean())).mean())
    return float(diff.mean()), float(lo), float(hi), max(p, 1.0 / n_boot)


def wilcoxon_p(a: Vec, b: Vec) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    d = y[mask] - x[mask]
    if d.size == 0 or np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def mcnemar_p(a: Vec, b: Vec) -> float:
    """Exact McNemar test on paired binary outcomes."""
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    n01 = int((~x & y).sum())
    n10 = int((x & ~y).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def compare(
    metric: str,
    name_a: str,
    a: Vec,
    name_b: str,
    b: Vec,
    binary: bool = False,
    seed: int = 0,
) -> PairedResult:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    delta, lo, hi, p = paired_bootstrap(x, y, seed=seed)
    mask = ~(np.isnan(x) | np.isnan(y))
    res = PairedResult(
        metric=metric,
        a=name_a,
        b=name_b,
        n=int(mask.sum()),
        mean_a=float(x[mask].mean()) if mask.any() else float("nan"),
        mean_b=float(y[mask].mean()) if mask.any() else float("nan"),
        delta=delta,
        delta_lo=lo,
        delta_hi=hi,
        p_bootstrap=p,
        p_wilcoxon=wilcoxon_p(x, y),
    )
    if binary:
        res.p_mcnemar = mcnemar_p(x[mask] > 0.5, y[mask] > 0.5)
    return res
