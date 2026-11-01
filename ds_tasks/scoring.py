"""Score an agent's answer against a task card."""

from __future__ import annotations

import math


def score(card: dict, answer) -> bool:
    kind, expected, tol = card["answer_type"], card["answer"], card.get("tolerance", {})
    try:
        if kind == "number":
            got = float(answer)
            if "log10" in tol:  # p-values: compare orders of magnitude
                return abs(math.log10(max(got, 1e-300)) - math.log10(max(expected, 1e-300))) <= tol["log10"]
            if "rel" in tol:
                return abs(got - expected) <= tol["rel"] * abs(expected)
            return abs(got - expected) <= tol.get("abs", 0.0)
        if kind == "category":
            return str(answer).strip().lower() == str(expected).strip().lower()
        if kind == "set":
            norm = lambda xs: {str(x).strip().lower() for x in xs}
            return norm(answer) == norm(expected)
    except (TypeError, ValueError):
        return False
    raise ValueError(f"unknown answer_type {kind}")
