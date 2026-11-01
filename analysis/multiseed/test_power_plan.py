"""Exact power checks independent of empirical evaluation results."""

import numpy as np
import pytest
from scipy.stats import multinomial

from power_plan import exact_power, required_n
from stats import exact_mcnemar


def test_power_hand_values_at_all_discordant_boundary():
    assert exact_power(5, 1.0, 1.0) == 0
    assert exact_power(6, 1.0, 1.0) == 1
    assert exact_power(6, 0.0, 1.0) == pytest.approx(0.03125)


def test_power_matches_direct_multinomial_enumeration():
    n, delta, discordance = 8, 0.1, 0.3
    direct = 0.0
    for b in range(n + 1):
        for c in range(n - b + 1):
            if exact_mcnemar(b, c) < 0.05:
                direct += multinomial.pmf(
                    [b, c, n - b - c],
                    n,
                    [
                        (discordance + delta) / 2,
                        (discordance - delta) / 2,
                        1 - discordance,
                    ],
                )
    assert exact_power(n, delta, discordance) == pytest.approx(direct)
    assert exact_power(n, -delta, discordance) == pytest.approx(direct)


def test_required_n_is_first_threshold_crossing():
    n, power = required_n(1.0, 1.0, target=0.8, max_n=10)
    assert n == 6 and power == 1
    assert required_n(0.0, 0.2, max_n=20) == (None, None)
    assert np.isfinite(exact_power(100, 0.05, 0.2))
