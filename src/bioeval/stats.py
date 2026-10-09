"""Uncertainty for benchmark numbers.

With 30 questions, a 0.05 gap in mean score can be noise. Every headline
number gets a bootstrap confidence interval, and model comparisons use a
*paired* bootstrap over questions, since both models answered the same items.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_mean(values, n_resamples: int = 2000, alpha: float = 0.05,
                   seed: int = 0) -> Interval:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(x.mean()), float(lo), float(hi))


@dataclass(frozen=True)
class PairedComparison:
    mean_diff: float
    low: float
    high: float
    p_value: float  # two-sided bootstrap p-value for "no difference"
    n: int
    n_resamples: int = 5000

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0

    @property
    def p_value_text(self) -> str:
        # B resamples cannot resolve a p-value below 1/B; reporting 0 would overstate it.
        floor = 1 / self.n_resamples
        return f"< {floor:.4f}" if self.p_value < floor else f"{self.p_value:.4f}"


def paired_bootstrap(a, b, n_resamples: int = 5000, alpha: float = 0.05,
                     seed: int = 0) -> PairedComparison:
    """Compare two models scored on the same questions (``a[i]`` and ``b[i]`` pair up)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    if d.size == 0:
        nan = float("nan")
        return PairedComparison(nan, nan, nan, nan, 0, n_resamples)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # Shift the bootstrap distribution to the null (mean 0) and ask how often it
    # is at least as extreme as what was observed.
    centred = boots - d.mean()
    p = float((np.abs(centred) >= abs(d.mean())).mean())
    return PairedComparison(float(d.mean()), float(lo), float(hi), p, int(d.size), n_resamples)


def spearman(a, b) -> float:
    """Rank correlation without SciPy: Pearson correlation of average ranks."""
    import pandas as pd

    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 3 or s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(s["a"].rank(), s["b"].rank())[0, 1])
