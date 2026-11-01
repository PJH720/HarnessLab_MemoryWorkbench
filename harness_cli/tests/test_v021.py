"""v0.2.1: live provider cascade, offline prompt preview, study task, ~/.harness persistence, /inspect, /explain."""

import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from harness_cli import cli, config, lego, panels
from harness_cli.commands import UsageError, parse_study_compare
from harness_cli.config import DEFAULT_MEMORY, Settings
from harness_cli.docqa import NotesError, StudyBuddy, chunk_text
from harness_cli.engine import HarnessEngine, discover, with_system
from harness_cli.home import HarnessHome, HomeError
from harness_cli.scratchpad import Scratchpad

needs_memory = pytest.mark.skipif(not DEFAULT_MEMORY.exists(), reason="midterm memory.json not present")
CUSTOM = "sharp pain in my lower right belly with nausea and a mild fever since last night"
NOTES = """# Hypothesis testing

A p-value is the probability of data at least as extreme as observed, assuming the null hypothesis is true.

## Welch t-test

Welch's t-test compares two means without assuming equal variances. Use it when group sizes or variances differ.

## McNemar test

McNemar's test compares paired binary outcomes, such as two classifiers graded on the same questions.
Only the discordant pairs carry information.

## Bootstrap

A paired bootstrap resamples questions with replacement to build a confidence interval for an accuracy difference.
"""


# --- provider cascade ----------------------------------------------------------------------------

@pytest.fixture
def clean_env(monkeypatch):
    for key in ("HARNESS_API_BASE", "OPENAI_BASE_URL", "HARNESS_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_candidates_follow_env_priority_then_onprem_then_ollama(clean_env):
    assert config.api_base_candidates() == (config.ONPREM_API_BASE, config.OLLAMA_API_BASE)
    clean_env.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
    clean_env.setenv("HARNESS_API_BASE", "http://gpu-box:30000/v1")
    assert config.api_base_candidates() == (
        "http://gpu-box:30000/v1", "https://openrouter.ai/api/v1", config.ONPREM_API_BASE, config.OLLAMA_API_BASE)
    assert Settings.from_env().api_base == "http://gpu-box:30000/v1"


def test_discover_skips_dead_endpoints_and_picks_a_served_model(clean_env):
    served = {config.OLLAMA_API_BASE: ["llama3.2:3b", "qwen2.5:7b"]}
    found = discover(Settings.from_env(), lister=lambda base, timeout: served.get(base))
    assert found.api_base == config.OLLAMA_API_BASE and found.model == "openai/llama3.2:3b"
    assert not found.scored_model, "badge numbers belong to the on-prem Qwen, not this model"


def test_discover_keeps_scored_model_on_prem_and_respects_harness_model(clean_env):
    onprem = discover(Settings.from_env(), lister=lambda base, t: ["Inferact/Qwen3.8-Flash-Next-NVFP4"])
    assert onprem.api_base == config.ONPREM_API_BASE and onprem.scored_model
    clean_env.setenv("HARNESS_MODEL", "openai/my-model")
    ollama_only = lambda base, t: ["x"] if base == config.OLLAMA_API_BASE else None  # noqa: E731
    assert discover(Settings.from_env(), lister=ollama_only).model == "openai/my-model"


def test_discover_returns_none_when_every_endpoint_is_down(clean_env):
    assert discover(Settings.from_env(), lister=lambda base, t: None) is None


def test_reasoning_off_flag_is_not_sent_to_hosted_apis(monkeypatch):
    monkeypatch.delenv("TC_DISABLE_THINKING", raising=False)
    config.prepare_runtime_env(api_base="https://api.openai.com/v1")
    assert "TC_DISABLE_THINKING" not in __import__("os").environ
    config.prepare_runtime_env(api_base=config.OLLAMA_API_BASE)
    assert __import__("os").environ["TC_DISABLE_THINKING"] == "1"


# --- offline fallback for arbitrary input (the v0.2.0 'no recorded response' refusal) -------------

@needs_memory
def test_predict_command_offline_shows_prompt_preview_for_custom_symptoms():
    result = CliRunner().invoke(cli.app, ["--demo", "predict", CUSTOM])
    assert result.exit_code == 2, result.output  # 2 = preview only, no diagnosis
    assert "OFFLINE RETRIEVAL PREVIEW" in result.output
    assert "녹화된 응답이 없습니다" not in result.output
    assert "lower right belly" in result.output, "the exact prompt is shown"


@needs_memory
def test_predict_json_offline_carries_the_exact_prompt():
    result = CliRunner().invoke(cli.app, ["--demo", "predict", "--json", CUSTOM])
    data = json.loads(result.output[result.output.index("{"):])
    assert data["source"] == "preview" and data["diagnosis"] == "" and CUSTOM in data["prompt"]


@needs_memory
def test_korean_symptoms_are_refused_before_any_prompt_is_built():
    result = CliRunner().invoke(cli.app, ["--demo", "predict", "오른쪽 아랫배가 아파요"])
    assert result.exit_code == 1 and "영어로" in result.output


# --- /inspect: why each example was chosen --------------------------------------------------------

def _engine(name):
    return HarnessEngine(with_system(Settings.from_env(), name, lego.resolve(name).memory_path), offline=True)


@needs_memory
def test_knn_top6_picks_are_the_six_nearest_with_descending_scores():
    picks = _engine("knn_top6").picks(CUSTOM)
    assert len(picks) == 6 and all(p.reason == "top-6 similar" for p in picks)
    scores = [p.score for p in picks]
    assert scores == sorted(scores), "nearest example is placed last, right before the problem"


@needs_memory
def test_coverage_and_mistake_picks_are_not_labelled_as_similar():
    picks = _engine("confusion_contrast").picks(CUSTOM)
    reasons = {p.reason for p in picks}
    assert "top-6 similar" in reasons and reasons <= {"top-6 similar", "past mistake", "label coverage"}
    assert all(p.score is not None for p in picks)


@needs_memory
def test_fewshot_all_has_no_ranking_and_no_memory_has_no_picks():
    assert {p.reason for p in _engine("fewshot_all").picks(CUSTOM)} == {"all"}
    assert _engine("no_memory").picks(CUSTOM) == ()


# --- /explain fewshot ------------------------------------------------------------------------------

def test_pareto_frontier_is_computed_from_the_logged_constants():
    frontier = lego.pareto_frontier("test")
    assert frontier[0] == "no_memory" and frontier[-1] == "fewshot_all"
    assert "knn_top6" in frontier and "confusion_contrast" not in frontier  # different token metric
    for name in frontier:  # nothing on the frontier is dominated
        a = lego.resolve(name).scores
        for other in frontier:
            b = lego.resolve(other).scores
            assert not (b.mean_accuracy() > a.mean_accuracy() and b.mean_tokens() < a.mean_tokens())


def test_explain_fewshot_quotes_logged_numbers():
    console = Console(record=True, width=200)
    panels.explain_fewshot(console)
    out = console.export_text()
    knn, few = lego.resolve("knn_top6").scores, lego.resolve("fewshot_all").scores
    assert f"{knn.mean_accuracy():.1%}" in out and f"{few.mean_accuracy():.1%}" in out
    assert f"{lego.token_reduction('knn_top6'):.1%}" in out and "30,000" in out


# --- ~/.harness persistence -----------------------------------------------------------------------

def test_home_defaults_to_harness_home_env(isolated_harness_home):
    assert HarnessHome().root == isolated_harness_home.resolve()


def test_profile_roundtrip_and_validation():
    home = HarnessHome()
    home.set_profile("target_exam", "Big Data capstone final")
    home.set_profile("style", "short bullet points with one example")
    assert HarnessHome().profile() == {"target_exam": "Big Data capstone final",
                                       "style": "short bullet points with one example"}
    with pytest.raises(HomeError):
        home.set_profile("password", "x")
    with pytest.raises(HomeError, match="영어"):
        home.set_profile("topics", "가설 검정")
    home.set_profile("style", "")
    assert "style" not in home.profile()


def test_home_refuses_the_frozen_logs():
    with pytest.raises(HomeError):
        HarnessHome(config.LOGS / "anything")


def test_medical_records_persist_across_sessions():
    labels = ("psoriasis", "dengue")
    pad = Scratchpad(HarnessHome())
    pad.add("itchy red rash on both arms and elbows", "psoriasis", labels)
    again = Scratchpad(HarnessHome())
    assert [r.disease for r in again.records] == ["psoriasis"] and again.persistent
    again.reset()
    assert Scratchpad(HarnessHome()).records == ()


def test_scratchpad_without_home_writes_nothing(isolated_harness_home):
    Scratchpad().add("itchy red rash on both arms and elbows", "psoriasis", ("psoriasis",))
    assert not isolated_harness_home.exists()


# --- study task ------------------------------------------------------------------------------------

@pytest.fixture
def notes_file(tmp_path):
    path = tmp_path / "stats_notes.md"
    path.write_text(NOTES)
    return path


def test_chunking_follows_headings_and_size_limit():
    chunks = chunk_text(NOTES, "stats_notes.md")
    assert [c.title for c in chunks] == ["Hypothesis testing", "Welch t-test", "McNemar test", "Bootstrap"]
    long = chunk_text("# T\n\n" + "A short sentence here. " * 200, "x.md", limit=300)
    assert len(long) > 1 and all(len(c.text) <= 300 for c in long)


def test_knn_retrieval_finds_the_relevant_section(notes_file):
    buddy = StudyBuddy()
    assert buddy.load(notes_file) == 4
    ans = buddy.prepare("When should I use McNemar's test for paired classifiers?", "knn_retrieval", {})
    assert ans.notes[0].chunk.title == "McNemar test" and ans.source == "preview"
    assert "**Course notes:**" in ans.prompt and "McNemar's test compares paired" in ans.prompt
    assert 0 < ans.context_saved < 1


def test_no_memory_prompt_has_no_notes_rules_or_profile(notes_file):
    buddy = StudyBuddy()
    buddy.load(notes_file)
    ans = buddy.prepare("What is a p-value?", "no_memory", {"style": "short bullet points"})
    assert ans.notes == () and "Course notes" not in ans.prompt and "Learner profile" not in ans.prompt


def test_recorded_lesson_is_retrieved_on_the_next_question_and_persists(notes_file):
    buddy = StudyBuddy(HarnessHome())
    buddy.load(notes_file)
    question = "Which correction applies to many simultaneous hypothesis tests?"
    before = buddy.prepare(question, "lesson_cues", {})
    assert before.cues == () and "- none recorded" in before.prompt
    buddy.add_lesson("Bonferroni correction for multiple hypothesis tests",
                     "divide alpha by the number of tests to control the family-wise error rate")
    after = buddy.prepare(question, "lesson_cues", {})
    assert after.cues and "divide alpha by the number of tests" in after.prompt
    knn = buddy.prepare(question, "knn_retrieval", {})
    assert any(h.chunk.lesson for h in knn.notes), "taught lessons are indexed with the notes"
    assert StudyBuddy(HarnessHome()).lessons == buddy.lessons
    assert buddy.reset_lessons() == 1 and StudyBuddy(HarnessHome()).lessons == ()


def test_study_inputs_must_be_english(notes_file, tmp_path):
    buddy = StudyBuddy()
    korean = tmp_path / "ko.md"
    korean.write_text("# 가설 검정\n\n유의확률은 ...")
    with pytest.raises(NotesError, match="영어"):
        buddy.load(korean)
    with pytest.raises(NotesError, match="영어"):
        buddy.prepare("p-value가 뭐야?", "knn_retrieval", {})
    with pytest.raises(NotesError, match="영어"):
        buddy.add_lesson("검정", "rule")
    with pytest.raises(NotesError):
        buddy.load(tmp_path / "slides.pdf")


def test_profile_line_goes_into_grounded_prompts(notes_file):
    buddy = StudyBuddy()
    buddy.load(notes_file)
    ans = buddy.prepare("What is a p-value?", "knn_retrieval", {"target_exam": "capstone final", "style": "two sentences"})
    assert "Learner profile: preparing for capstone final; preferred explanation style: two sentences." in ans.prompt


def test_live_answer_uses_the_prepared_prompt(notes_file):
    buddy = StudyBuddy()
    buddy.load(notes_file)
    seen = {}

    def complete(prompt):
        seen["prompt"] = prompt
        return " Use Welch's t-test [1]. ", 321

    ans = buddy.ask("When are variances unequal?", "knn_retrieval", {}, complete)
    assert ans.source == "live" and ans.answer == "Use Welch's t-test [1]." and ans.prompt_tokens == 321
    assert seen["prompt"] == ans.prompt


def test_parse_study_compare_defaults_to_last_question():
    assert parse_study_compare("none knn", "What is a p-value?") == ("no_memory", "knn_retrieval", "What is a p-value?")
    assert parse_study_compare("knn cues why bootstrap") == ("knn_retrieval", "lesson_cues", "why bootstrap")
    with pytest.raises(UsageError):
        parse_study_compare("knn")
    with pytest.raises(UsageError):
        parse_study_compare("knn fewshot what")
    with pytest.raises(UsageError):
        parse_study_compare("knn cues")


@needs_memory
def test_repl_study_flow_offline(notes_file):
    script = "\n".join([
        f"/load {notes_file}",
        "When should I use McNemar's test for paired classifiers?",
        "/inspect context",
        '/record "Bonferroni correction" -> "divide alpha by the number of tests"',
        "/compare none cues",
        "/memory status",
        "/task medical",
        CUSTOM,
        "/inspect context",
        "/explain fewshot",
        "/exit",
    ]) + "\n"
    result = CliRunner().invoke(cli.app, ["--demo", "chat"], input=script)
    out = result.output
    assert result.exit_code == 0, out
    assert "4단락" in out and "McNemar test" in out and "OFFLINE RETRIEVAL PREVIEW" in out
    assert "RAW PROMPT" in out and "학습 기억에 추가" in out and "메모리에만 (--demo)" in out
    assert "lesson_cues" in out and "TF-IDF 유사도" in out and "/explain fewshot" in out
    assert "녹화된 응답이 없습니다" not in out
