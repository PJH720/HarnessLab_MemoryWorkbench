"""Cards are generated from reference.py and score their own answers correctly."""

import json
from pathlib import Path

import pytest

from scoring import score

CARDS = sorted((Path(__file__).parent.parent / "cards").glob("*.json"))


def perturb(card):
    a = card["answer"]
    if card["answer_type"] == "number":
        return a * 10 + 1 if a else 1.0
    if card["answer_type"] == "set":
        return list(a)[:-1] + ["not_a_feature"]
    return "definitely-wrong"


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.stem)
def test_card_scores_own_answer_and_rejects_wrong(path):
    card = json.loads(path.read_text())
    assert score(card, card["answer"])
    assert not score(card, perturb(card))


def test_enough_cards_and_tags():
    cards = [json.loads(p.read_text()) for p in CARDS]
    assert 30 <= len(cards) <= 40
    tags = {t for c in cards for t in c["tags"]}
    assert {"join", "segment", "test", "model", "explain"} <= tags
