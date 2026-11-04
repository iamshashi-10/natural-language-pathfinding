"""Tests for nlpath.metrics against values worked out by hand."""

import math

import pytest

from nlpath.metrics import bootstrap_mean_ci, fmt_rate, mcnemar_exact, wilson


def test_wilson_known_values():
    # Textbook example: 8 of 10 -> [0.490, 0.943] (Wilson 1927; Newcombe 1998).
    low, high = wilson(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3) and high == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == pytest.approx(1.0)
    assert all(math.isnan(v) for v in wilson(0, 0))


def test_mcnemar_exact():
    # 6 discordant pairs, all favouring a: p = 2 * (1/2)^6 = 0.03125.
    a = [True] * 6 + [True, False] * 5
    b = [False] * 6 + [True, False] * 5
    res = mcnemar_exact(a, b)
    assert (res["a_only"], res["b_only"]) == (6, 0)
    assert res["p_value"] == pytest.approx(0.03125)
    # Balanced discordance -> p = 1.
    assert mcnemar_exact([True, False], [False, True])["p_value"] == pytest.approx(1.0)
    assert mcnemar_exact([True], [True])["p_value"] == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact([True], [])


def test_bootstrap_ci():
    mean, low, high = bootstrap_mean_ci([1.0] * 20)
    assert mean == low == high == 1.0
    mean, low, high = bootstrap_mean_ci(list(range(100)))
    assert mean == pytest.approx(49.5) and low < mean < high
    assert bootstrap_mean_ci(list(range(100))) == bootstrap_mean_ci(list(range(100)))


def test_fmt_rate():
    assert fmt_rate(8, 10) == "80% [49%–94%] (n=10)"
    assert fmt_rate(0, 0) == "–"
