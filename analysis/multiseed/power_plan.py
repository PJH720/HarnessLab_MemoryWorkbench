"""Prospective exact-McNemar sensitivity scenarios, independent of run outcomes."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


def conditional_power(
    max_n: int, delta: float, discordance: float, alpha: float = 0.05
):
    """Power conditional on each possible number of discordant pairs.

    delta=P(reference-only correct)-P(other-only correct); discordance is their
    sum. This integrates the same two-sided exact rejection rule used by the
    engine; neither parameter is estimated from the observed experiment.
    """
    if not 0 < discordance <= 1 or not abs(delta) <= discordance or not 0 < alpha < 1:
        raise ValueError("Invalid paired-outcome probabilities or alpha")
    d = np.arange(max_n + 1)
    cutoff = binom.ppf(alpha / 2, d, 0.5).astype(int) - 1
    q = 0.5 + delta / (2 * discordance)
    return binom.cdf(cutoff, d, q) + binom.sf(d - cutoff - 1, d, q)


def exact_power(n: int, delta: float, discordance: float, alpha: float = 0.05) -> float:
    if n < 0 or type(n) is not int:
        raise ValueError("n must be a nonnegative integer")
    power = conditional_power(n, delta, discordance, alpha)
    return float(np.dot(binom.pmf(np.arange(n + 1), n, discordance), power))


def required_n(
    delta: float, discordance: float, target: float = 0.8, max_n: int = 5000
):
    """Search every n; avoid assuming an exact discrete test has monotone power."""
    if not 0 < target < 1:
        raise ValueError("target power must lie strictly between zero and one")
    power = conditional_power(max_n, delta, discordance)
    for n in range(1, max_n + 1):
        achieved = float(
            np.dot(binom.pmf(np.arange(n + 1), n, discordance), power[: n + 1])
        )
        if achieved >= target:
            return n, achieved
    return None, None


def main():
    out = Path(__file__).resolve().parent / "out"
    records = []
    # Fixed sensitivity grid declared without fitting to the running experiment.
    for delta in [0.03, 0.05, 0.08]:
        for discordance in [0.1, 0.2, 0.3]:
            n, power = required_n(delta, discordance)
            records.append(
                dict(
                    minimum_effect_pp=delta * 100,
                    assumed_discordance=discordance,
                    alpha=0.05,
                    target_power=0.8,
                    independent_pairs_required=n,
                    power_at_n=power,
                )
            )
    pd.DataFrame(records).to_csv(out / "power_plan.csv", index=False)
    lines = [
        "# Prospective power sensitivity plan",
        "",
        "This is planning, not a new experiment or an amendment to the existing preregistration. No observed accuracy, effect estimate or p-value is used to choose the grid or sample size.",
        "",
        "All scenarios use a two-sided exact McNemar test, alpha=.05 and target power=.80. Delta is the desired reference-minus-other accuracy difference; discordance is the probability that exactly one system is correct. Sample size counts independent paired questions, not repeated answers to the same question.",
        "",
        "| Minimum effect (pp) | Assumed discordance | Required independent pairs | Power at n |",
        "|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['minimum_effect_pp']:g} | {r['assumed_discordance']:.2f} | {r['independent_pairs_required']} | {r['power_at_n']:.6f} |"
        )
    lines += [
        "",
        "For a new confirmatory experiment, choose one scientifically meaningful effect threshold and a defensible discordance assumption, fix the sample size and analysis before collecting fresh data, and report all results regardless of significance. Do not keep sampling until p<.05. If the real effect is absent, more data will not establish the desired direction.",
        "",
        "The existing three seeds evaluate the same fixed symptom texts; 3 x 212 evaluations are not 636 independent new questions. More seeds may characterize training-memory variability, but do not substitute for independent test data.",
        "",
        "Exact calculation: D~Binomial(n,discordance); B|D~Binomial(D,.5+delta/(2*discordance)); sum the alternative probability of the exact two-sided p<.05 rejection region. Search n sequentially because discrete-test power can oscillate.",
    ]
    (out / "POWER_PLAN.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
