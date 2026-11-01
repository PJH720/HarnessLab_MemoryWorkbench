"""Numerical, replay-safety and pre-registration decision regression tests."""

import json
import math
import socket
import sys

import numpy as np
import pandas as pd
import pytest

import stats


@pytest.mark.parametrize(
    "b,c,p", [(7, 1, 0.0703125), (4, 4, 1.0), (0, 0, 1.0), (6, 0, 0.03125)]
)
def test_exact_mcnemar_hand_values(b, c, p):
    assert stats.exact_mcnemar(b, c) == p
    assert stats.exact_mcnemar(c, b) == p


def test_stratified_matched_mh_and_exact_three_strata():
    result = stats.matched_mh([(7, 1), (3, 2), (4, 1)])
    assert result["only_ref"] == 14
    assert result["only_sys"] == 4
    assert result["mh_statistic"] == pytest.approx(100 / 18)
    assert result["mh_p"] == pytest.approx(math.erfc(math.sqrt(100 / 18 / 2)))
    assert result["pooled_exact_p"] == 8096 / 262144
    assert stats.matched_mh([(0, 0)] * 3)["mh_p"] == 1.0


def test_holm_known_values_and_registered_missing_family():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.01, 0.04, None, None, None, None]) == [
        0.06,
        0.2,
        None,
        None,
        None,
        None,
    ]
    assert stats.holm([1, 1, 1]) == [1, 1, 1]


def test_bootstrap_deterministic_and_paired():
    strata = [np.array([-1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])]
    first = stats.paired_bootstrap(strata)
    assert first == stats.paired_bootstrap(strata)
    assert first["draws"] == 10_000
    assert first["rng_seed"] == 0
    assert first["diff_vs_ref"] == pytest.approx(1 / 6)
    assert first["ci_low"] <= first["diff_vs_ref"] <= first["ci_high"]
    assert stats.paired_bootstrap([np.zeros(8)])["ci_high"] == 0


def test_cache_miss_cannot_fall_back_to_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network fallback attempted")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    client = stats.CacheOnly([tmp_path])
    with pytest.raises(stats.MissingCache):
        client("uncached prompt")
    assert not list(tmp_path.iterdir())
    modules = stats.safe_modules()
    assert stats._PREFIX + ".llm" in sys.modules
    assert not hasattr(sys.modules[stats._PREFIX + ".llm"], "LLM")
    assert modules["no_memory"].NoMemory(client).get_state() == "{}"


def test_loader_discovers_embedded_predictions_and_does_not_divide_tokens(
    tmp_path, monkeypatch
):
    data = [
        dict(input="question one", target="allergy", raw_question="one"),
        dict(input="question two", target="malaria", raw_question="two"),
    ]
    monkeypatch.setattr(stats, "examples", lambda *_: data)
    path = tmp_path / "val.json"
    path.write_text(
        json.dumps(
            dict(
                seed=42,
                memory="agents/no_memory.py",
                total=2,
                correct=1,
                accuracy=0.5,
                llm_input_tokens=123,
                predictions=[dict(pred="allergy"), dict(pred="dengue")],
            )
        )
    )
    spec = dict(
        seed=42,
        split="val",
        sys="no_memory",
        result=path,
        memory=None,
        cache_dirs=[tmp_path / "no-cache"],
        block="E3",
    )
    rows, features, meta = stats.load_run(spec)
    assert meta["status"] == "complete"
    assert [r["correct"] for r in rows] == [True, False]
    assert all(r["input_tokens"] is None for r in rows)
    assert features == []


def test_loader_missing_cache_is_missing_outcome_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stats,
        "examples",
        lambda *_: [dict(input="missing", target="allergy", raw_question="one")],
    )
    path = tmp_path / "val.json"
    path.write_text(json.dumps(dict(total=1, correct=1, accuracy=1.0)))
    spec = dict(
        seed=42,
        split="val",
        sys="no_memory",
        result=path,
        memory=None,
        cache_dirs=[tmp_path / "empty"],
        block="E3",
    )
    rows, _, meta = stats.load_run(spec)
    assert meta["status"] == "missing_cache"
    assert rows[0]["correct"] is None
    assert rows[0]["pred"] is None


def test_pairing_uses_seed_and_item_identity():
    frame = pd.DataFrame(
        [
            dict(
                seed=42,
                split="val",
                qid=0,
                gold="a",
                item_id="item",
                sys="knn_top6",
                correct=True,
            ),
            dict(
                seed=42,
                split="val",
                qid=0,
                gold="a",
                item_id="other",
                sys="cue_bank",
                correct=False,
            ),
        ]
    )
    assert stats.paired(frame, "cue_bank").empty


def decision_tables(deltas, p):
    by_seed = pd.DataFrame(
        [
            dict(seed=s, split="test", sys="cue_bank", acc_ref=0.8, acc_sys=0.8 - d)
            for s, d in zip(stats.SEEDS, deltas)
        ]
    )
    pooled = pd.DataFrame([dict(split="test", sys="cue_bank", n_seeds=3, mh_p=p)])
    return by_seed, pooled


@pytest.mark.parametrize(
    "deltas,p,complete,decision",
    [
        ([0.1, 0.1, 0.1], 0.01, True, "SUPPORTED"),
        ([0.1, 0.1, 0.1], 0.08, True, "NOT SUPPORTED"),
        ([0.1, 0.1, -0.1], 0.01, True, "INCOMPLETE"),
        ([0.1, 0.1, 0.1], 0.01, False, "INCOMPLETE"),
        ([-0.1, -0.1, -0.1], 0.01, True, "NOT SUPPORTED"),
    ],
)
def test_preregistered_decision_and_overlap_rule(deltas, p, complete, decision):
    by_seed, pooled = decision_tables(deltas, p)
    assert stats.decide_h1(by_seed, pooled, complete)[0] == decision


def test_reference_replay_matches_original_features_and_all_pairs():
    table, features, _, _ = stats.load_all()
    result = stats.parity(table)
    assert result["passed"]
    assert result["byte_identical_csv"]
    expected = pd.read_csv(stats.ROOT / "analysis/out/failures.csv")
    actual = features[(features.seed == 42) & (features.split == "val")]
    actual = actual.sort_values(["system", "q"]).reset_index(drop=True)
    expected = expected.sort_values(["system", "q"]).reset_index(drop=True)
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(
                actual[column].astype(float), expected[column], atol=1e-12, rtol=1e-12
            )
        else:
            assert actual[column].tolist() == expected[column].tolist()


def test_clustered_h2_uses_question_clusters_with_current_pandas():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        [
            dict(
                seed=seed,
                split="val",
                sys=system,
                qid=q,
                correct=bool(rng.random() < 0.6),
                row_status="ok",
            )
            for seed in [42, 123]
            for q in range(15)
            for system in stats.SYSTEMS
        ]
    )
    text = stats.clustered_logit(frame, pd.DataFrame(), "H2")
    assert "clusters=30" in text
    assert "raw-minus-generated" in text
    assert "Covariance Type:              cluster" in text


def test_noise_reports_correctness_flips_separately_from_answer_changes():
    def record(rep, q, correct, pred):
        return dict(
            seed=42,
            split="val",
            sys="no_memory",
            qid=q,
            gold="a",
            item_id=str(q),
            row_status="ok",
            correct=correct,
            pred=pred,
            rep=rep,
        )

    baseline = pd.DataFrame([record(0, 0, True, "a"), record(0, 1, False, "b")])
    noise = pd.DataFrame(
        [
            record(1, 0, True, "a"),
            record(1, 1, False, "c"),
            record(2, 0, False, "b"),
            record(2, 1, False, "c"),
        ]
    )
    result = stats.noise_table(baseline, noise)
    assert result.iloc[0].correctness_flips == 1
    assert result.iloc[1].correctness_flips == 0
    assert result.iloc[1].raw_prediction_flips == 1


def test_inconsistent_result_counts_are_not_analysable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stats,
        "examples",
        lambda *_: [dict(input="one", target="a", raw_question="one")],
    )
    path = tmp_path / "val.json"
    path.write_text(
        json.dumps(dict(total=1, correct=0, accuracy=0.0, predictions=[dict(pred="a")]))
    )
    spec = dict(
        seed=42,
        split="val",
        sys="no_memory",
        result=path,
        memory=None,
        cache_dirs=[tmp_path / "empty"],
        block="E3",
    )
    rows, _, meta = stats.load_run(spec)
    assert meta["status"] == "aggregate_mismatch"
    assert rows[0]["row_status"] == "unvalidated_run"
    assert stats.usable(pd.DataFrame(rows)).empty


def test_engine_aggregates_matched_rows_by_seed_and_split():
    records = []
    for seed in stats.SEEDS:
        for q in range(4):
            for system in stats.SYSTEMS:
                correct = q in ([1, 2] if system == "cue_bank" else [0, 1])
                records.append(
                    dict(
                        seed=seed,
                        split="test",
                        sys=system,
                        qid=q,
                        gold="a",
                        item_id=str(q),
                        correct=correct,
                        row_status="ok",
                    )
                )
    by_seed, pooled, boot = stats.contrast_tables(pd.DataFrame(records))
    h1 = pooled[(pooled.split == "test") & (pooled.sys == "cue_bank")].iloc[0]
    assert h1.n == 12
    assert h1.n_seeds == 3
    assert h1.only_ref == h1.only_sys == 3
    assert h1.pooled_exact_p == 1.0
    assert h1.mh_p == 1.0
    assert h1.status == "complete"
    seed_h1 = by_seed[(by_seed.split == "test") & (by_seed.sys == "cue_bank")]
    assert seed_h1.n.tolist() == [4, 4, 4]
    assert (
        boot[(boot.split == "test") & (boot.sys == "cue_bank")].iloc[0].diff_vs_ref == 0
    )


def test_unstable_snapshot_is_retried_without_another_file_change():
    current = {"result.json": {"sha256": "same"}}
    assert not stats.needs_refresh(current, current, {"snapshot_stable": True})
    assert stats.needs_refresh(current, current, {"snapshot_stable": False})
    assert stats.needs_refresh(current, {}, {"snapshot_stable": True})


def test_execution_budgets_are_configurable_without_statistical_changes():
    assert stats.execution_limits({}) == (20, 12)
    assert stats.execution_limits(
        {"limits": {"max_iterations": 60, "max_watch_checks": 21}}
    ) == (60, 21)
    for invalid in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            stats.execution_limits({"limits": {"max_iterations": invalid}})
