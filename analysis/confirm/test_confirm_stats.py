"""Synthetic checks of the preregistered decision rule; no data or model needed."""

import json
import sys
from pathlib import Path

import numpy as np

import confirm_stats
from confirm_stats import tost


def pair(n: int, only_sys: int, only_ref: int, both: int) -> tuple[np.ndarray, np.ndarray]:
    ref = np.zeros(n, bool)
    other = np.zeros(n, bool)
    ref[:both] = other[:both] = True
    other[both:both + only_sys] = True
    ref[both + only_sys:both + only_sys + only_ref] = True
    return ref, other


def write_run(run: Path, system: str, correct: np.ndarray, empty: int = 0) -> None:
    """Files of one finished seed-42 job; the last `empty` answers are failed calls."""
    folder = run / "eval" / system / f"seed{confirm_stats.PRIMARY_SEED}"
    folder.mkdir(parents=True)
    answered = len(correct) - empty
    rows = [{"index": i, "raw_question": f"question {i}", "target": "flu",
             "prediction": "flu" if i < answered else "", "was_correct": bool(ok)}
            for i, ok in enumerate(correct)]
    (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "test.json").write_text(json.dumps({"total": len(rows), "correct": int(sum(correct))}))


def run_main(tmp_path: Path, monkeypatch) -> tuple[int, dict]:
    monkeypatch.setattr(confirm_stats, "OUT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["confirm_stats.py", str(tmp_path / "run")])
    code = confirm_stats.main()
    return code, json.loads((tmp_path / "out" / "summary.json").read_text())


def test_balanced_discordance_is_equivalent():
    ref, other = pair(572, 30, 30, 450)
    r = tost(ref, other)
    assert r["diff"] == 0 and r["verdict"] == "equivalent"
    assert -0.05 < r["ci90_low"] < 0 < r["ci90_high"] < 0.05


def test_large_gap_is_different_not_equivalent():
    ref, other = pair(572, 10, 80, 400)
    r = tost(ref, other)
    assert r["diff"] < -0.10 and r["verdict"] == "different"


def test_small_sample_is_inconclusive():
    ref, other = pair(40, 4, 6, 25)
    assert tost(ref, other)["verdict"] == "inconclusive"


def test_mcnemar_counts_and_identical_arms():
    ref, other = pair(100, 7, 3, 60)
    r = tost(ref, other)
    assert (r["only_sys"], r["only_ref"]) == (7, 3)
    same = tost(ref, ref)
    assert same["p_mcnemar"] == 1.0 and same["verdict"] == "equivalent"


def test_missing_primary_pair_gives_no_inference(tmp_path, monkeypatch):
    ref, _ = pair(572, 30, 30, 450)
    write_run(tmp_path / "run", "knn_top6", ref)
    code, summary = run_main(tmp_path, monkeypatch)
    assert code == 1 and summary["status"] == "INCOMPLETE"
    assert "primary" not in summary and "hc1_verdict" not in summary


def test_primary_pair_alone_gives_hc1_and_defers_secondary(tmp_path, monkeypatch):
    ref, other = pair(572, 30, 30, 450)
    write_run(tmp_path / "run", "knn_top6", ref)
    write_run(tmp_path / "run", "cue_bank", other)
    code, summary = run_main(tmp_path, monkeypatch)
    assert code == 0 and summary["status"] == "PRIMARY_ONLY"
    assert summary["hc1_verdict"] == "equivalent" and summary["primary"]["n"] == 572
    assert summary["coverage"]["complete"] == 2
    assert not (tmp_path / "out" / "comparisons.csv").exists()


def test_failed_calls_above_two_percent_make_hc1_inconclusive(tmp_path, monkeypatch):
    ref, other = pair(572, 30, 30, 450)
    write_run(tmp_path / "run", "knn_top6", ref)
    write_run(tmp_path / "run", "cue_bank", other, empty=15)
    _, summary = run_main(tmp_path, monkeypatch)
    assert summary["primary"]["missing_rate"] > confirm_stats.MAX_MISSING
    assert summary["hc1_verdict"].startswith("inconclusive (missing")
