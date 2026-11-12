"""Synthetic checks of the preregistered decision rule; no data or model needed."""

import numpy as np

from confirm_stats import tost


def pair(n: int, only_sys: int, only_ref: int, both: int) -> tuple[np.ndarray, np.ndarray]:
    ref = np.zeros(n, bool)
    other = np.zeros(n, bool)
    ref[:both] = other[:both] = True
    other[both:both + only_sys] = True
    ref[both + only_sys:both + only_sys + only_ref] = True
    return ref, other


def test_balanced_discordance_is_equivalent():
    ref, other = pair(572, 30, 30, 450)
    r = tost(ref, other)
    assert r["diff"] == 0 and r["verdict"] == "equivalent"
    assert -0.05 < r["ci90_low"] < 0 < r["ci90_high"] < 0.05


def test_large_gap_is_different_not_equivalent():
    ref, other = pair(572, 10, 80, 400)
    r = tost(ref, other)
    assert r["diff"] < -0.10 and r["verdict"] == "different"


def test_small_sample_is_inconclusive():
    ref, other = pair(40, 4, 6, 25)
    assert tost(ref, other)["verdict"] == "inconclusive"


def test_mcnemar_counts_and_identical_arms():
    ref, other = pair(100, 7, 3, 60)
    r = tost(ref, other)
    assert (r["only_sys"], r["only_ref"]) == (7, 3)
    same = tost(ref, ref)
    assert same["p_mcnemar"] == 1.0 and same["verdict"] == "equivalent"
