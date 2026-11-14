"""Preregistered confirmatory analysis on the Symptom2Disease holdout (n=572).

Contract: report/analysis_20261112_2353_confirmatory_preregistration.md. No model is
called: inputs are the per-question predictions written by scripts/run_confirmatory.sh.
The exact McNemar, Holm, paired bootstrap and failure-feature code is imported from
analysis/multiseed/stats.py unchanged.

  usage: uv run --project harness_lab/text_classification --with scipy --with statsmodels \
           --with pandas python analysis/confirm/confirm_stats.py [RUN_DIR]
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT / "harness_lab/text_classification"
OUT = HERE / "out"
sys.path.insert(0, str(ROOT / "analysis/multiseed"))
sys.path.insert(0, str(ROOT / "harness_lab"))
import stats as ms  # noqa: E402  (analysis/multiseed/stats.py)

DATASET = "Symptom2DiseaseHoldout"
N_TEST = 572
SEEDS = [42, 123, 456]
PRIMARY_SEED = 42
REFERENCE = ms.REFERENCE  # knn_top6
SYSTEMS = ms.SYSTEMS
MARGIN = 0.05
ALPHA = 0.05
MAX_MISSING = 0.02
RAW = ["knn_top6", "fewshot_all", "contrast_online"]


def item_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_predictions(run: Path) -> tuple[pd.DataFrame, list[dict]]:
    """One row per (seed, system, question); a run is complete only if it matches test.json."""
    rows, runs = [], []
    for seed in SEEDS:
        for system in SYSTEMS:
            folder = run / "eval" / system / f"seed{seed}"
            preds, summary = folder / "predictions.jsonl", folder / "test.json"
            status = {"seed": seed, "sys": system, "status": "missing", "rows": 0}
            if preds.exists() and summary.exists():
                data = [json.loads(line) for line in preds.read_text().splitlines() if line.strip()]
                agg = json.loads(summary.read_text())
                correct = sum(bool(r["was_correct"]) for r in data)
                status.update(rows=len(data), correct=correct, empty_predictions=sum(not r["prediction"] for r in data))
                ok = len(data) == N_TEST == agg["total"] and correct == agg["correct"]
                status["status"] = "complete" if ok else "aggregate_mismatch"
                for r in data:
                    rows.append({
                        "seed": seed, "sys": system, "qid": r["index"], "item_id": item_id(r["raw_question"]),
                        "gold": r["target"], "pred": r["prediction"], "correct": bool(r["was_correct"]),
                        "failed_call": not r["prediction"],
                    })
            runs.append(status)
    return pd.DataFrame(rows), runs


def tost(ref: np.ndarray, other: np.ndarray, margin: float = MARGIN) -> dict:
    """Paired difference other - ref with a Wald SE, TOST at alpha each side, exact McNemar."""
    ref, other = np.asarray(ref, bool), np.asarray(other, bool)
    n = len(ref)
    b, c = int(np.sum(other & ~ref)), int(np.sum(ref & ~other))
    d = (b - c) / n
    se = math.sqrt(max(((b + c) / n - d * d) / n, 0.0))
    z = norm.ppf(1 - ALPHA)
    if se == 0:
        p_tost = 0.0 if abs(d) < margin else 1.0
    else:
        p_tost = max(norm.sf((d + margin) / se), norm.cdf((d - margin) / se))
    lo, hi = d - z * se, d + z * se
    p_diff = ms.exact_mcnemar(b, c)
    equivalent, different = (-margin < lo and hi < margin), p_diff < ALPHA
    verdict = ("different_but_within_margin" if equivalent and different else
               "equivalent" if equivalent else "different" if different else "inconclusive")
    return {"n": n, "only_sys": b, "only_ref": c, "diff": d, "se": se, "ci90_low": lo,
            "ci90_high": hi, "p_tost": p_tost, "p_mcnemar": p_diff, "verdict": verdict}


def paired_arrays(table: pd.DataFrame, seed: int, system: str) -> pd.DataFrame:
    sub = table[table.seed == seed]
    ref = sub[sub.sys == REFERENCE][["item_id", "correct", "failed_call"]]
    oth = sub[sub.sys == system][["item_id", "correct", "failed_call"]]
    return ref.merge(oth, on="item_id", suffixes=("_ref", "_sys"), validate="one_to_one")


def comparisons(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in [s for s in SYSTEMS if s != REFERENCE]:
        pair = paired_arrays(table, PRIMARY_SEED, system)
        res = tost(pair.correct_ref.to_numpy(), pair.correct_sys.to_numpy())
        delta = pair.correct_sys.astype(int).to_numpy() - pair.correct_ref.astype(int).to_numpy()
        boot = ms.paired_bootstrap([delta])
        rows.append({"system": system, **res, "boot_ci95_low": boot["ci_low"], "boot_ci95_high": boot["ci_high"],
                     "missing_rate": float((pair.failed_call_ref | pair.failed_call_sys).mean())})
    frame = pd.DataFrame(rows)
    frame["p_mcnemar_holm"] = ms.holm(frame.p_mcnemar.tolist())
    return frame


def seed_pooled(table: pd.DataFrame, system: str, draws: int = 10_000, seed: int = 0) -> dict:
    """Question-clustered bootstrap: the three seeds are repeated measures of one item."""
    wide = []
    for s in SEEDS:
        pair = paired_arrays(table, s, system).set_index("item_id")
        wide.append((pair.correct_sys.astype(int) - pair.correct_ref.astype(int)).rename(s))
    per_item = pd.concat(wide, axis=1).mean(axis=1).to_numpy()
    rng = np.random.default_rng(seed)
    means = per_item[rng.integers(0, len(per_item), size=(draws, len(per_item)))].mean(axis=1)
    return {"items": len(per_item), "diff": float(per_item.mean()),
            "ci95_low": float(np.quantile(means, 0.025)), "ci95_high": float(np.quantile(means, 0.975))}


def cluster_logit(y: pd.Series, x: pd.DataFrame, groups: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.Logit(y.astype(float), sm.add_constant(x.astype(float), has_constant="add")).fit(
            disp=False, cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def hc2(table: pd.DataFrame) -> dict:
    data = table[table.sys.isin(RAW + ms.ERROR_MEMORIES)]
    generated = data.sys.isin(ms.ERROR_MEMORIES)
    fit = cluster_logit(data.correct, generated.rename("generated").to_frame(), data.item_id)
    acc = data.groupby(generated).correct.mean()
    return {"rows": len(data), "clusters": int(data.item_id.nunique()), "coef_generated": float(fit.params["generated"]),
            "p_generated": float(fit.pvalues["generated"]), "acc_raw": float(acc[False]), "acc_generated": float(acc[True])}


def hc3(table: pd.DataFrame, run: Path) -> dict:
    from text_classification.data import load_dataset_splits_3way

    frames = []
    for seed in SEEDS:
        _, _, dataset, _ = load_dataset_splits_3way(DATASET, 60, 50, N_TEST, seed)
        memories = {
            s: ms.memory_for(s, (run / "frozen_memories" / s / f"seed{seed}" / "memory.json").read_text(), None)
            for s in ms.ERROR_MEMORIES
        }
        sub = table[(table.seed == seed) & table.sys.isin(ms.ERROR_MEMORIES)]
        feats = ms.failure_features(sub.rename(columns={"sys": "system", "qid": "q"}), memories, dataset)
        feats["seed"] = seed
        feats["item_id"] = [item_id(dataset[q]["raw_question"]) for q in feats.q]
        frames.append(feats)
    feats = pd.concat(frames, ignore_index=True)
    x = feats[["max_similarity", "target_in_cues", "n_other_diseases"]].copy()
    x["prompt_kchars"] = feats.prompt_chars / 1000
    x = pd.concat([x, pd.get_dummies(feats.system, drop_first=True, dtype=float)], axis=1)
    fit = cluster_logit(feats.failed, x, feats.item_id)
    return {"rows": len(feats), "clusters": int(feats.item_id.nunique()),
            "coef_target_in_cues": float(fit.params["target_in_cues"]),
            "p_target_in_cues": float(fit.pvalues["target_in_cues"]),
            "share_target_in_cues": float(feats.target_in_cues.mean())}


def pct(x: float) -> str:
    return f"{x * 100:+.2f}"


def write_summary(result: dict, comp: pd.DataFrame) -> None:
    p = result["primary"]
    lines = [
        "# Confirmatory analysis summary",
        "",
        f"- Coverage: {result['coverage']['complete']}/{result['coverage']['expected']} runs complete.",
        f"- HC1 (seed {PRIMARY_SEED}, n={p['n']}): cue_bank - knn_top6 = {pct(p['diff'])} pp, "
        f"90% CI [{pct(p['ci90_low'])}, {pct(p['ci90_high'])}], TOST p={p['p_tost']:.4g}, "
        f"exact McNemar p={p['p_mcnemar']:.4g} (only cue_bank {p['only_sys']}, only knn_top6 {p['only_ref']}). "
        f"Verdict: **{result['hc1_verdict']}**.",
        "",
        "| System vs knn_top6 | Diff (pp) | 90% CI | Bootstrap 95% CI | McNemar p | Holm p | TOST p | Verdict |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for r in comp.itertuples():
        lines.append(
            f"| {r.system} | {pct(r.diff)} | [{pct(r.ci90_low)}, {pct(r.ci90_high)}] | "
            f"[{pct(r.boot_ci95_low)}, {pct(r.boot_ci95_high)}] | {r.p_mcnemar:.4g} | {r.p_mcnemar_holm:.4g} | "
            f"{r.p_tost:.4g} | {r.verdict} |"
        )
    lines += ["", f"- Seed-pooled (auxiliary): {json.dumps(result['seed_pooled_primary'])}",
              f"- HC2: {json.dumps(result['hc2'])}", f"- HC3: {json.dumps(result['hc3'])}", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(lines))


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else LAB / "logs/confirm-20261112"
    table, runs = load_predictions(run)
    complete = [r for r in runs if r["status"] == "complete"]
    result = {"coverage": {"expected": len(runs), "complete": len(complete), "runs": runs}}
    OUT.mkdir(parents=True, exist_ok=True)
    done = {(r["seed"], r["sys"]) for r in complete}
    if not {(PRIMARY_SEED, REFERENCE), (PRIMARY_SEED, "cue_bank")} <= done:
        result["status"] = "INCOMPLETE"
        (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"INCOMPLETE: primary pair not complete ({len(complete)}/{len(runs)} runs); no inferential output.")
        return 1
    pair = paired_arrays(table, PRIMARY_SEED, "cue_bank")
    primary = tost(pair.correct_ref.to_numpy(), pair.correct_sys.to_numpy())
    primary["missing_rate"] = float((pair.failed_call_ref | pair.failed_call_sys).mean())
    hc1 = primary["verdict"] if primary["missing_rate"] <= MAX_MISSING else "inconclusive (missing > 2%)"
    result.update(primary=primary, hc1_verdict=hc1)
    if len(complete) != len(runs):
        # The primary test is final; secondary analyses wait for every preregistered run.
        result["status"] = "PRIMARY_ONLY"
        (OUT / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
        print(json.dumps({"hc1_verdict": hc1, **primary}, indent=2, default=float))
        print(f"Secondary analyses pending: {len(complete)}/{len(runs)} runs complete.")
        return 0
    comp = comparisons(table)
    result.update(
        status="COMPLETE",
        hc4=comp[comp.system == "fewshot_all"].iloc[0].to_dict(),
        seed_pooled_primary=seed_pooled(table, "cue_bank"), hc2=hc2(table), hc3=hc3(table, run),
        accuracy={f"{k[0]}/seed{k[1]}": round(v, 4) for k, v in table.groupby(["sys", "seed"]).correct.mean().items()},
    )
    comp.to_csv(OUT / "comparisons.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    write_summary(result, comp)
    print((OUT / "SUMMARY.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
