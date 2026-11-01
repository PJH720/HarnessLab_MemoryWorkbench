"""Offline tests: routing, prompt parsing, and an end-to-end predict with a stub LLM."""

import json

import pytest

from harness_cli.commands import Command, CommandRegistry, route_input
from harness_cli.config import DEFAULT_MEMORY, Settings
from harness_cli.engine import HarnessEngine, parse_prompt

REGISTRY = CommandRegistry([Command("help", "h", lambda: True, ("h",)), Command("exit", "x", lambda: False)])


@pytest.mark.parametrize(
    ("line", "kind"),
    [("", "empty"), ("   ", "empty"), ("/help", "command"), ("/H", "command"), ("/nope", "unknown"), ("/", "unknown"), ("fever and chills", "query")],
)
def test_route_input(line, kind):
    assert route_input(line, REGISTRY).kind == kind


def test_parse_prompt_reads_examples_and_warnings():
    prompt = (
        "Classify the problem below using the solved examples.\n\n"
        "**Similar solved examples:**\nQ: itchy rash\nA: fungal infection\n\nQ: high fever\nA: malaria\n\n"
        "**Known confusions (a reader often mislabels these):**\n- 'dengue' was mistaken for 'malaria' (1x)\n\n"
        "**Problem:**\nX"
    )
    evidence, warnings = parse_prompt(prompt, {"high fever"})
    assert [(e.label, e.past_mistake) for e in evidence] == [("fungal infection", False), ("malaria", True)]
    assert warnings == ("'dengue' was mistaken for 'malaria' (1x)",)


needs_memory = pytest.mark.skipif(not DEFAULT_MEMORY.exists(), reason="midterm memory.json not present")


@needs_memory
def test_predict_end_to_end_with_stub_llm():
    stub = lambda prompt: json.dumps({"reasoning": "periodic fever", "final_answer": "malaria"})
    engine = HarnessEngine(Settings.from_env(), llm=stub)
    pred = engine.predict("High fever with chills every other day after travelling.")
    assert pred.diagnosis == "malaria"
    assert pred.evidence, "confusion_contrast should select at least one example"
    assert "## Patient Symptoms" in engine.wrap("x")


@needs_memory
def test_stats_reports_memory():
    engine = HarnessEngine(Settings.from_env(), llm=lambda p: "{}")
    stats = engine.stats()
    assert stats.examples == 60
    assert sum(c for _, c in stats.labels) == 60
