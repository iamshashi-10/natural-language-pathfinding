"""Statistics for the evaluation (plan §4.3–4.4), using only numpy and math.

* ``wilson`` — Wilson 95% confidence interval for a rate.
* ``mcnemar_exact`` — exact two-sided McNemar test on paired successes.
* ``bootstrap_mean_ci`` — percentile bootstrap CI of a mean (fixed seed).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Z95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (nan, nan) if n = 0."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar test for paired outcomes ``a[i]`` vs ``b[i]``.

    Only discordant pairs matter: n01 = a wrong & b right, n10 = a right & b wrong.
    Under H0 each discordant pair is a fair coin, so the p-value is the
    two-sided binomial tail. Returns counts and p-value.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        p = 1.0
    else:
        k = min(n10, n01)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {"n": len(a), "a_only": n10, "b_only": n01, "p_value": p}


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10_000,
                      seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high): 95% percentile bootstrap interval of the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(low), float(high)


def fmt_rate(successes: int, n: int) -> str:
    """'87% [81–91] (n=150)' style cell."""
    if n == 0:
        return "–"
    low, high = wilson(successes, n)
    return f"{successes / n:.0%} [{low:.0%}–{high:.0%}] (n={n})"
