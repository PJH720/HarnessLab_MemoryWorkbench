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


# --- demo additions: presets, routing with arguments, fail-safe replay, compare view ---

import time

from rich.console import Console

from harness_cli import ui
from harness_cli.config import NO_MEMORY_PATH
from harness_cli.engine import OFFLINE_PREVIEW, ReplayCache, SafeEngine, run_with_deadline, with_system
from harness_cli.presets import PRESETS, find_preset, token_reduction_vs_fewshot_all

ARG_REGISTRY = CommandRegistry([Command("demo", "d", lambda args: args == "2", takes_args=True)])


def test_bare_preset_number_routes_to_preset_but_sentences_do_not():
    assert route_input("2", REGISTRY, ("1", "2", "3")).kind == "preset"
    assert route_input("4", REGISTRY, ("1", "2", "3")).kind == "query"
    assert route_input("2 days of fever", REGISTRY, ("1", "2", "3")).kind == "query"


def test_slash_command_passes_arguments():
    route = route_input("/demo 2", ARG_REGISTRY)
    assert (route.kind, route.args) == ("command", "2")
    assert route.command.run(route.args) is True
    assert REGISTRY.find("help").run("ignored") is True  # zero-arg handlers still work


def test_presets_cover_three_cases_with_known_labels():
    assert [p.key for p in PRESETS] == ["1", "2", "3"]
    assert find_preset(" 3 ").contested[0] == "typhoid"
    assert find_preset("9") is None


def test_ascii_title_spells_harness_with_two_s_and_aligned_rows():
    rows = ui.ascii_title()
    assert len(rows) == 6 and len({len(r) for r in rows}) == 1
    assert ui.ascii_title("SS")[0] == ui._GLYPHS["S"][0] * 2
    assert ui.TITLE == "HARNESS-CLI"


def test_token_reduction_badge_is_against_fewshot_all():
    assert 0.79 < token_reduction_vs_fewshot_all() < 0.81


def test_run_with_deadline_gives_up_on_a_hung_call():
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        run_with_deadline(lambda: time.sleep(5), 0.2)
    assert time.perf_counter() - started < 1


def _offline_engine(system="confusion_contrast", path=DEFAULT_MEMORY):
    return HarnessEngine(with_system(Settings.from_env(), system, path), offline=True)


@needs_memory
def test_failed_live_call_falls_back_to_recorded_response():
    engine = _offline_engine()  # its LLM raises ConnectionError, like a dropped network
    safe = SafeEngine(engine, ReplayCache(), live=True, deadline=5)
    pred, note = safe.predict(PRESETS[1].symptoms)
    assert pred.source == "replay" and pred.diagnosis
    assert pred.evidence, "retrieval is local, so evidence is rebuilt even offline"
    assert "녹화" in note and safe.live is False


@needs_memory
def test_unrecorded_symptoms_offline_return_prompt_preview_not_a_refusal():
    safe = SafeEngine(_offline_engine(), ReplayCache(), live=False)
    pred, note = safe.predict("sharp pain in my lower right belly with nausea and a mild fever since last night")
    assert pred is not None and pred.source == "preview" and pred.diagnosis == ""
    assert pred.evidence, "retrieval runs offline, so the harness still picks examples"
    assert "sharp pain in my lower right belly" in pred.prompt and pred.prompt_chars == len(pred.prompt)
    assert OFFLINE_PREVIEW in note and "녹화된 응답이 없습니다" not in note


@needs_memory
def test_compare_view_renders_both_systems_and_replay_badge():
    cache = ReplayCache()
    raw = SafeEngine(_offline_engine("no_memory", NO_MEMORY_PATH), cache, live=False)
    harness = SafeEngine(_offline_engine(), cache, live=False)
    symptoms = PRESETS[0].symptoms
    console = Console(record=True, width=140)
    ui.comparison(console, ("no_memory", raw.predict(symptoms)), ("confusion_contrast", harness.predict(symptoms)))
    out = console.export_text()
    assert "no_memory" in out and "confusion_contrast" in out
    assert "DEMO MODE: CACHED REPLAY" in out and "test 정확도" in out
