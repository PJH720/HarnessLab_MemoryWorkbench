"""REPL commands of v0.2: /harness, /switch, /memory, /record, /compare, /study. No LLM calls (demo/replay only)."""

import csv
import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from harness_cli import cli, config, lego, study, ui
from harness_cli.commands import (
    COMPARE_USAGE,
    Command,
    CommandRegistry,
    UsageError,
    parse_compare,
    parse_record,
    route_input,
)
from harness_cli.config import Settings
from harness_cli.engine import EnginePool, ReplayCache
from harness_cli.presets import PRESETS

needs_memories = pytest.mark.skipif(
    not all(b.memory_path.exists() for b in lego.BLOCKS.values()), reason="frozen memories not present"
)


# --- routing ----------------------------------------------------------------------------------

def _registry() -> CommandRegistry:
    noop = lambda args: True  # noqa: E731
    return CommandRegistry([
        Command("harness", "", noop, ("switch",), True),
        Command("memory", "", noop, takes_args=True),
        Command("record", "", noop, takes_args=True),
        Command("compare", "", noop, ("vs",), True),
        Command("study", "", lambda: True),
    ])


@pytest.mark.parametrize(
    ("line", "name", "args"),
    [
        ("/harness knn_top6", "harness", "knn_top6"),
        ("/switch cue", "harness", "cue"),
        ("/harness", "harness", ""),
        ("/memory status", "memory", "status"),
        ("/memory reset", "memory", "reset"),
        ('/record "fever" -> "malaria"', "record", '"fever" -> "malaria"'),
        ("/compare no_memory knn_top6 1", "compare", "no_memory knn_top6 1"),
        ("/vs 2", "compare", "2"),
        ("/study", "study", ""),
    ],
)
def test_new_commands_route_with_their_arguments(line, name, args):
    route = route_input(line, _registry(), ("1", "2", "3"))
    assert route.kind == "command" and route.command.name == name and route.args == args


def test_presets_still_route_as_presets():
    assert route_input("3", _registry(), ("1", "2", "3")).kind == "preset"


# --- /record parsing ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "args",
    [
        '"Itchy red patches on my elbows" -> "psoriasis"',
        "'Itchy red patches on my elbows' -> 'psoriasis'",
        '"Itchy red patches on my elbows"->"psoriasis"',
        "Itchy red patches on my elbows -> psoriasis",
        '“Itchy red patches on my elbows” → psoriasis',
    ],
)
def test_record_accepts_quoted_and_bare_forms(args):
    assert parse_record(args) == ("Itchy red patches on my elbows", "psoriasis")


def test_record_keeps_an_arrow_inside_quoted_symptoms():
    assert parse_record('"pain -> then fever for days" -> "dengue"') == ("pain -> then fever for days", "dengue")


@pytest.mark.parametrize("args", ["", "fever and chills", '"fever" ->', '-> "malaria"', "a -> b -> c", '"" -> ""'])
def test_record_rejects_malformed_input_with_usage(args):
    with pytest.raises(UsageError, match="사용법"):
        parse_record(args)


# --- /compare parsing --------------------------------------------------------------------------

def test_compare_two_blocks_and_preset():
    parsed = parse_compare("no_memory knn_top6 1", current="confusion_contrast")
    assert (parsed.left, parsed.right, parsed.preset.key) == ("no_memory", "knn_top6", "1")
    assert parsed.symptoms == PRESETS[0].symptoms


def test_compare_two_aliases_and_text():
    parsed = parse_compare("knn cue sudden fever with joint pain", current="no_memory")
    assert (parsed.left, parsed.right, parsed.preset) == ("knn_top6", "cue_bank", None)
    assert parsed.symptoms == "sudden fever with joint pain"


def test_compare_two_blocks_without_a_case_uses_preset_1():
    assert parse_compare("fewshot_all knn_top6", current="knn_top6").preset.key == "1"


def test_compare_legacy_preset_is_no_memory_vs_current():
    parsed = parse_compare("2", current="cue_bank")
    assert (parsed.left, parsed.right, parsed.preset.key) == ("no_memory", "cue_bank", "2")


def test_compare_legacy_text_even_when_it_starts_with_an_alias():
    parsed = parse_compare("all my joints ache after a fever", current="knn_top6")
    assert (parsed.left, parsed.right, parsed.preset) == ("no_memory", "knn_top6", None)
    assert parsed.symptoms == "all my joints ache after a fever"


@pytest.mark.parametrize("args", ["", "   ", "knn_top6", "knn_top6 2", "knn 2", "cue"])
def test_compare_rejects_empty_or_one_block(args):
    with pytest.raises(UsageError):
        parse_compare(args, current="knn_top6")
    assert "사용법" in COMPARE_USAGE


# --- badges and tables -------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(lego.BLOCKS))
def test_badge_renders_for_every_block(name):
    console = Console(record=True, width=120)
    block = lego.BLOCKS[name]
    ui.switched(console, block)
    out = console.export_text()
    assert name in out and "n=212" in out and "seed 42 memory.json" in out
    assert f"{block.scores.mean_accuracy('test'):.1%}" in out
    if not block.multi_seed:
        assert "단일 실행" in out and "llm_input_tokens" in out


def test_blocks_table_lists_all_eight():
    console = Console(record=True, width=200)
    console.print(ui.blocks_table("knn_top6"))
    out = console.export_text()
    assert all(name in out for name in lego.BLOCKS)
    assert "83.0%" in out and "67.0%" in out and "84.6%" in out


def test_cue_bank_badge_is_neutral():
    console = Console(record=True, width=120)
    ui.switched(console, lego.BLOCKS["cue_bank"])
    out = console.export_text()
    assert "닻" not in out and "anchor" not in out.lower()


# --- /study ------------------------------------------------------------------------------------

def _scripted(*answers):
    it = iter(answers)

    def ask(prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None
    return ask


def test_study_quits_at_once_without_running_a_compare():
    calls = []
    console = Console(record=True, width=120)
    shown = study.run(console, lambda *a: calls.append(a), _scripted("q"), live=False)
    assert shown == 0 and calls == []
    assert study.TALK_URL in console.export_text()


def test_study_walks_all_four_steps_and_states_the_facts():
    calls = []
    console = Console(record=True, width=160)
    shown = study.run(console, lambda *a: calls.append(a), _scripted("", "", "", ""), live=False)
    out = console.export_text()
    assert shown == 4 and calls == list(study.STEP_COMPARES)
    for fact in ("-12.0%p", "7:1", "p = 0.070", "-1.4%p", "13:10", "p = 0.68", "NOT SUPPORTED", "구분되지 않습니다", "80.8%"):
        assert fact in out, fact
    assert "같다" not in out.replace("같다는 뜻은 아님", "")


def test_study_treats_eof_as_quit_mid_way():
    calls = []
    shown = study.run(Console(record=True), lambda *a: calls.append(a), _scripted(""), live=False)
    assert shown == 1 and len(calls) == 1


def test_study_warns_when_session_memory_changes_offline_replays():
    console = Console(record=True, width=160)
    study.run(console, lambda *a: None, _scripted("q"), live=False, session_records=2)
    assert "/memory reset" in console.export_text()


ROOT = config.PROJECT_ROOT
needs_analysis = pytest.mark.skipif(not (ROOT / study.MCNEMAR_CSV).exists(), reason="analysis outputs not present")


def _rows(rel):
    with open(ROOT / rel, newline="") as f:
        return list(csv.DictReader(f))


@needs_analysis
@pytest.mark.parametrize("fact", [study.CUE_BANK_VAL, study.CUE_BANK_TEST])
def test_cue_bank_mcnemar_facts_match_the_csv(fact):
    row = next(r for r in _rows(fact.source)
               if r["sys"] == fact.system and r["seed"] == str(fact.seed) and r["split"] == fact.split)
    assert (int(row["only_ref"]), int(row["only_sys"])) == (fact.only_ref, fact.only_sys)
    assert float(row["p_exact"]) == pytest.approx(fact.p)


@needs_analysis
@pytest.mark.parametrize("fact", [study.POOLED_NO_MEMORY, study.POOLED_FEWSHOT_ALL])
def test_pooled_facts_match_the_csv(fact):
    row = next(r for r in _rows(fact.source) if r["sys"] == fact.system and r["split"] == fact.split)
    assert (int(row["only_ref"]), int(row["only_sys"])) == (fact.only_ref, fact.only_sys)
    assert float(row["mh_p"]) == pytest.approx(fact.p)


@needs_analysis
def test_h1_decision_and_reference_match_the_analysis():
    coverage = json.loads((ROOT / study.H1_SOURCES[0]).read_text())
    assert coverage["decision"] == study.H1_DECISION and coverage["complete"] is True
    assert f'REFERENCE = "{study.REFERENCE}"' in (ROOT / "analysis/multiseed/stats.py").read_text()


# --- REPL handlers on a real offline session ------------------------------------------------------

NEW_SYMPTOMS = "Bright violet freckles spread across both kneecaps after swimming in a cold quarry lake."


def _session(tmp_path) -> cli.Session:
    pool = EnginePool(Settings.from_env(), ReplayCache(), live=False)
    return cli.Session(pool, "confusion_contrast")


@needs_memories
def test_switch_record_memory_and_compare_handlers(tmp_path, monkeypatch):
    console = Console(record=True, width=160)
    monkeypatch.setattr(cli, "console", console)
    session = _session(tmp_path)

    cli._switch(session, "knn")
    assert session.current == "knn_top6"
    cli._switch(session, "no_such_block")
    assert session.current == "knn_top6"

    cli._record(session, "fever -> malaria")  # too short: refused, nothing added
    assert session.scratchpad.records == ()
    cli._record(session, f'"{NEW_SYMPTOMS}" -> "Psoriasis"')
    assert [r.disease for r in session.scratchpad.records] == ["psoriasis"]

    cli._compare(session, f"no_memory knn_top6 {NEW_SYMPTOMS}")
    cli._memory(session, "status")
    out = console.export_text()
    assert "알 수 없는 하네스" in out and "세션 기억에 추가" in out and "★" in out
    assert "PREVIEW" in out and "원본 예시 (동결)" in out
    assert all(name in out for name in lego.BLOCKS)

    cli._memory(session, "reset")
    assert session.scratchpad.records == ()


@needs_memories
def test_study_runs_headless_on_an_offline_session(tmp_path, monkeypatch):
    console = Console(record=True, width=160)
    monkeypatch.setattr(cli, "console", console)
    shown = cli._study(_session(tmp_path), ask=_scripted("", "", "", ""))
    out = console.export_text()
    assert shown == 4
    assert "DEMO MODE: CACHED REPLAY" in out and "cue_bank" in out and "fewshot_all" in out


@needs_memories
def test_typer_compare_accepts_two_blocks_and_a_preset_in_demo_mode():
    result = CliRunner().invoke(cli.app, ["--demo", "compare", "no_memory", "knn_top6", "1"])
    assert result.exit_code == 0, result.output
    assert "knn_top6" in result.output and "기록된 평가 결과" in result.output
