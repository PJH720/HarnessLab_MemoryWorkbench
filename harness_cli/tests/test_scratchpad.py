"""Engine pool, prompt sections per system, session memory (/record), and frozen-data integrity. No LLM calls."""

import hashlib
import json

import pytest

from harness_cli import config, lego
from harness_cli.config import Settings
from harness_cli.engine import EnginePool, HarnessEngine, ReplayCache, with_system
from harness_cli.presets import PRESETS
from harness_cli.scratchpad import Scratchpad, ScratchpadError

pytestmark = pytest.mark.skipif(
    not all(b.memory_path.exists() for b in lego.BLOCKS.values()), reason="frozen memories not present"
)

# A pattern no frozen example has, so the new example must win retrieval on its own words.
NEW_SYMPTOMS = "Bright violet freckles spread across both kneecaps after swimming in a cold quarry lake."
QUERY = "Violet freckles appeared on my kneecaps after a cold quarry swim."


def _engine(name: str, llm=None) -> HarnessEngine:
    block = lego.resolve(name)
    return HarnessEngine(with_system(Settings.from_env(), block.name, block.memory_path), llm=llm, offline=llm is None)


def _pool(tmp_path) -> EnginePool:
    return EnginePool(Settings.from_env(), ReplayCache(tmp_path / "none.json"), live=False)


# --- prompt sections: one offline check per system ---------------------------------------------

EXPECT_NOTES = {
    "no_memory": None, "fewshot_all": None, "knn_top6": None, "confusion_contrast": None,
    "contrast_online": "Known confusions", "cue_bank": "Diagnostic cues", "cue_bank_verified": "Verified diagnostic cues",
    "reflexion_notes": "Reflections",
}


@pytest.mark.parametrize("name", list(lego.BLOCKS))
def test_sections_are_read_from_every_systems_prompt(name):
    engine = _engine(name)
    items = getattr(engine.system, "items", [])
    errs = getattr(engine.system, "error_ids", [])
    # query with a stored mistake so error-driven sections clear their retrieval thresholds
    query = items[0]["symptoms"] if items else engine.system.examples[errs[0]]["raw_question"] if errs else PRESETS[0].symptoms
    sec = engine.sections(engine.prompt_for(engine.wrap(query)))
    if name == "no_memory":
        assert sec.evidence == () and sec.notes == ()
        return
    assert sec.evidence, "every example-based system injects examples"
    assert all(ev.label in engine.labels for ev in sec.evidence)
    expected = EXPECT_NOTES[name]
    if expected:
        assert sec.notes_title.startswith(expected) and sec.notes
        assert not any(n.startswith("- ") or "none recorded" in n for n in sec.notes)
        assert any(ev.past_mistake for ev in sec.evidence)
    else:
        assert sec.notes == ()  # midterm confusion_contrast recorded no confusions offline


@pytest.mark.parametrize("name", list(lego.BLOCKS))
def test_displayed_prompt_is_the_prompt_sent(name):
    sent = []

    def stub(prompt):
        sent.append(prompt)
        return json.dumps({"reasoning": "r", "final_answer": "malaria"})

    engine = _engine(name, llm=stub)
    pred = engine.predict(PRESETS[1].symptoms)
    assert pred.diagnosis == "malaria" and len(sent) == 1
    assert sent[0] == engine.prompt_for(engine.wrap(PRESETS[1].symptoms))
    assert pred.prompt_chars == len(sent[0])


def test_fewshot_all_prompt_is_not_empty():
    pred = _engine("fewshot_all").preview(PRESETS[0].symptoms)
    assert len(pred.evidence) == 60 and pred.prompt_chars > 5_000


# --- engine pool --------------------------------------------------------------------------------

def test_pool_builds_each_block_once_and_resolves_aliases(tmp_path):
    pool = _pool(tmp_path)
    assert pool.get("knn") is pool.get("knn_top6")
    assert pool.loaded() == ("knn_top6",)
    with pytest.raises(lego.UnknownHarness):
        pool.get("knn_top7")


# --- scratchpad ---------------------------------------------------------------------------------

def test_record_puts_the_new_example_into_knn_top6_retrieval(tmp_path):
    pool = _pool(tmp_path)
    engine = pool.get("knn_top6").engine
    before = engine.preview(QUERY)
    assert NEW_SYMPTOMS not in [ev.question for ev in before.evidence]

    pool.scratchpad.add(NEW_SYMPTOMS, "Allergy", engine.labels)
    after = engine.preview(QUERY)
    top6 = [ev.question for ev in after.evidence]
    assert len(top6) == 6 and NEW_SYMPTOMS in top6
    hit = next(ev for ev in after.evidence if ev.question == NEW_SYMPTOMS)
    assert hit.session and hit.label == "allergy"


def test_status_counts_frozen_and_session_examples_per_system(tmp_path):
    pool = _pool(tmp_path)
    labels = pool.get("knn").engine.labels
    pool.get("no_memory")
    pool.scratchpad.add(NEW_SYMPTOMS, "allergy", labels)
    pool.get("cue_bank")  # loaded after /record: still gets the session example
    rows = {r.system: r for r in pool.scratchpad.status()}
    assert (rows["knn_top6"].frozen, rows["knn_top6"].session, rows["knn_top6"].affected) == (60, 1, True)
    assert (rows["cue_bank"].frozen, rows["cue_bank"].session) == (60, 1)
    assert (rows["no_memory"].frozen, rows["no_memory"].session, rows["no_memory"].affected) == (0, 0, False)
    assert len(pool.get("cue_bank").engine.system.examples) == 61


def test_reset_restores_the_frozen_state(tmp_path):
    pool = _pool(tmp_path)
    engine = pool.get("contrast_online").engine
    frozen = engine.system.get_state()
    pool.scratchpad.add(NEW_SYMPTOMS, "allergy", engine.labels)
    assert engine.system.get_state() != frozen and engine.modified
    assert pool.scratchpad.reset() == 1
    assert engine.system.get_state() == frozen and not engine.modified
    assert pool.scratchpad.records == ()


@pytest.mark.parametrize(
    ("symptoms", "disease", "message"),
    [
        (NEW_SYMPTOMS, "flu", "22개 질병"),
        (NEW_SYMPTOMS, "allergyy", "혹시 'allergy'"),
        ("무릎에 보라색 반점", "allergy", "영어"),
        ("too short", "allergy", "자여야"),
    ],
)
def test_invalid_records_are_rejected(symptoms, disease, message):
    pad = Scratchpad()
    labels = _engine("knn").labels
    with pytest.raises(ScratchpadError, match=message):
        pad.add(symptoms, disease, labels)
    assert pad.records == ()


def test_duplicate_record_is_rejected():
    pad, labels = Scratchpad(), _engine("knn").labels
    pad.add(NEW_SYMPTOMS, " ALLERGY ", labels)
    assert pad.records[0].disease == "allergy"
    with pytest.raises(ScratchpadError, match="이미"):
        pad.add(NEW_SYMPTOMS.upper(), "allergy", labels)


# --- replay after /record -----------------------------------------------------------------------

def test_demo_mode_never_replays_a_pre_record_answer(tmp_path):
    cache = ReplayCache(tmp_path / "replay.json")
    symptoms = PRESETS[0].symptoms
    cache.put("knn_top6", symptoms, {"response": json.dumps({"final_answer": "pneumonia"}), "seconds": 1.0})
    cache.put("no_memory", symptoms, {"response": json.dumps({"final_answer": "pneumonia"}), "seconds": 1.0})
    pool = EnginePool(Settings.from_env(), cache, live=False)
    knn, raw = pool.get("knn"), pool.get("no_memory")
    assert knn.predict(symptoms)[0].source == "replay"

    pool.scratchpad.add(NEW_SYMPTOMS, "allergy", knn.engine.labels)
    pred, note = knn.predict(symptoms)
    assert pred.source == "preview" and pred.diagnosis == "" and pred.evidence
    assert "실시간 서버" in note
    assert raw.predict(symptoms)[0].source == "replay"  # no_memory's prompt did not change

    pool.scratchpad.reset()
    assert knn.predict(symptoms)[0].source == "replay"


# --- frozen research data -----------------------------------------------------------------------

def _digest_logs() -> dict[str, str]:
    root = config.LOGS
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_scratchpad_never_touches_frozen_logs(tmp_path):
    before = _digest_logs()
    pool = _pool(tmp_path)
    for name in lego.BLOCKS:
        pool.get(name)
    labels = pool.get("knn").engine.labels
    pool.scratchpad.add(NEW_SYMPTOMS, "allergy", labels)
    for name in lego.BLOCKS:
        pool.get(name).predict(QUERY)
    pool.scratchpad.reset()
    assert _digest_logs() == before


# --- recorded replay covers every block x preset, for the prompts the CLI builds today ----------

@pytest.mark.parametrize("name", list(lego.BLOCKS))
def test_replay_covers_every_block_and_preset_with_the_current_prompt(name):
    from harness_cli.engine import sha

    cache, engine = ReplayCache(), _engine(name)
    for preset in PRESETS:
        entry = cache.get(name, preset.symptoms)
        assert entry is not None and entry["response"], (name, preset.key)
        assert entry["prompt_sha"] == sha(engine.prompt_for(engine.wrap(preset.symptoms))), (name, preset.key)
        assert entry["prompt_tokens"] > 0 and entry["seconds"] > 1  # real round trips, not cache hits


def test_compare_shows_the_nearest_examples_first_including_a_recorded_one(tmp_path):
    from harness_cli.ui import nearest_evidence

    pool = _pool(tmp_path)
    engine = pool.get("knn_top6").engine
    pool.scratchpad.add(NEW_SYMPTOMS, "allergy", engine.labels)
    pred = engine.preview(QUERY)
    shown = nearest_evidence(pred)
    assert shown[0] is pred.evidence[-1]  # nearest sits right before the problem in the prompt
    assert any(ev.session for ev in shown)
