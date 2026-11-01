"""Lego registry: names, aliases, and the frozen numbers checked against the files they were copied from."""

import csv
import json

import pytest

from harness_cli import config, lego


def test_registry_has_the_eight_blocks_in_order():
    assert lego.names() == (
        "no_memory", "fewshot_all", "knn_top6", "contrast_online",
        "cue_bank", "cue_bank_verified", "reflexion_notes", "confusion_contrast",
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [("knn", "knn_top6"), ("KNN-Top6", "knn_top6"), (" none ", "no_memory"), ("verified", "cue_bank_verified"),
     ("cc", "confusion_contrast"), ("reflexion", "reflexion_notes"), ("fewshot_all", "fewshot_all")],
)
def test_resolve_accepts_names_and_aliases(name, expected):
    assert lego.resolve(name).name == expected


def test_resolve_rejects_unknown_names_with_a_hint():
    with pytest.raises(lego.UnknownHarness, match="knn_top6"):
        lego.resolve("knn_top7")
    with pytest.raises(lego.UnknownHarness, match="사용 가능"):
        lego.resolve("zzz")


def test_memories_load_from_the_package_and_logs_follow_the_data_root(monkeypatch, tmp_path):
    assert lego.resolve("knn").memory_path == config.MEMORY_DIR / "knn_top6.json"
    assert lego.resolve("knn").log_path == config.LOGS / "online-v2/Symptom2Disease/knn_top6" / config.MODEL_DIR / "memory.json"
    monkeypatch.setattr(config, "LOGS", tmp_path)
    assert lego.resolve("cc").log_path.is_relative_to(tmp_path)
    assert lego.resolve("cc").memory_path == config.MEMORY_DIR / "confusion_contrast.json"


def test_rounded_means_match_the_reported_figures():
    means = {b.name: round(b.scores.mean_accuracy() * 100, 1) for b in lego.BLOCKS.values()}
    assert means == {
        "no_memory": 67.0, "fewshot_all": 84.6, "knn_top6": 83.0, "contrast_online": 81.6,
        "cue_bank": 82.2, "cue_bank_verified": 83.6, "reflexion_notes": 82.5, "confusion_contrast": 80.7,
    }
    assert 0.80 < lego.token_reduction("knn_top6") < 0.81
    assert not lego.resolve("confusion_contrast").multi_seed
    with pytest.raises(ValueError):  # midterm tokens are a different metric
        lego.token_reduction("confusion_contrast")


@pytest.mark.skipif(not lego.ACCURACY_CSV.exists(), reason="analysis CSV not present")
def test_online_v2_constants_match_accuracy_csv():
    with lego.ACCURACY_CSV.open() as fh:
        rows = {(r["sys"], int(r["seed"]), r["split"]): r for r in csv.DictReader(fh)}
    online = [b for b in lego.BLOCKS.values() if b.run == lego.ONLINE_V2]
    assert {k[0] for k in rows} == {b.name for b in online}
    seen = set()
    for block in online:
        for run in block.scores.runs:
            row = rows[(block.name, run.seed, run.split)]
            assert (run.n, run.correct, run.input_tokens) == (
                int(row["n"]), int(row["correct"]), int(row["cached_eval_input_tokens"])
            ), (block.name, run)
            seen.add((block.name, run.seed, run.split))
    assert seen == set(rows)


@pytest.mark.parametrize("block", [b for b in lego.BLOCKS.values() if b.run == lego.ONLINE_V2], ids=lambda b: b.name)
def test_loaded_memory_is_the_seed_42_run(block):
    """The memory the CLI loads comes from the same run as the seed-42 val number."""
    val = block.log_path.with_name("val.json")
    if not val.exists():
        pytest.skip("online-v2 val.json not present")
    log = json.loads(val.read_text())
    run = next(r for r in block.scores.split("val") if r.seed == 42)
    assert (log["seed"], log["correct"], log["total"]) == (42, run.correct, run.n)


def test_midterm_constants_match_result_files():
    block = lego.resolve("confusion_contrast")
    folder = block.log_path.parent
    if not (folder / "test.json").exists():
        pytest.skip("midterm logs not present")
    for run in block.scores.runs:
        log = json.loads((folder / f"{run.split}.json").read_text())
        assert (log["seed"], log["correct"], log["total"], log["llm_input_tokens"]) == (
            run.seed, run.correct, run.n, run.input_tokens
        )
