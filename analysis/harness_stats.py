"""Stage 1 diagnosis engine: statistics on the online-v2 harness comparison.

Per-question correctness is recomputed by replaying each saved memory over the
validation set; every LLM response comes from the litellm cache written by the run,
so no model calls are made (the script aborts if the replay looks uncached).

Outputs (analysis/out/):
  per_question.csv   question x system correctness and prediction
  mcnemar.csv        pairwise exact McNemar tests
  bootstrap.csv      paired bootstrap 95% CI of accuracy difference vs the reference
  failures.csv       per-question features for error-memory systems
  logit.txt          logistic regression of failure on those features
  *.png              figures

Usage:
  TC_DISABLE_THINKING=1 OPENAI_API_KEY=onprem-placeholder \
  TEXT_CLASSIFICATION_LLM_CACHE_DIR=$HOME/.cache/text-classification/litellm-online \
  uv run --project harness_lab/text_classification --with scipy --with statsmodels \
      --with matplotlib python analysis/harness_stats.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "harness_lab"
RUN = LAB / "text_classification/logs/online-v2/Symptom2Disease"
OUT = ROOT / "analysis/out"
MODEL_LEAF = "qwen3.8-flash-next-nvfp4"
REFERENCE = "knn_top6"
SYSTEMS = ["no_memory", "fewshot_all", "knn_top6", "contrast_online", "cue_bank", "reflexion_notes", "cue_bank_verified"]
ERROR_MEMORIES = ["cue_bank", "reflexion_notes", "cue_bank_verified"]
BOOTSTRAP_DRAWS = 10_000
SEED = 0

sys.path.insert(0, str(LAB))
from text_classification.agents._text_rank import rank_texts  # noqa: E402
from text_classification.data.api import load_dataset_splits_3way  # noqa: E402
from text_classification.inner_loop import load_memory_system  # noqa: E402
from text_classification.llm import LLM  # noqa: E402


def is_correct(raw) -> bool:
    return raw if isinstance(raw, bool) else bool(raw["was_correct"])


def replay(systems: list[str], val: list[dict], evaluator) -> tuple[pd.DataFrame, dict]:
    """Rebuild each system from memory.json and re-predict the validation set from cache."""
    llm = LLM(model="openai/Inferact/Qwen3.8-Flash-Next-NVFP4", api_base="https://model-endpoint.example.invalid/v1", temperature=0.0)
    rows, memories = [], {}
    for name in systems:
        mem_path = RUN / name / MODEL_LEAF / "memory.json"
        if not (RUN / name / MODEL_LEAF / "val.json").exists():
            print(f"skip {name}: no val.json yet")
            continue
        memory = load_memory_system(name, llm)
        memory.set_state(mem_path.read_text())
        memories[name] = memory
        started = time.time()
        for qi, ex in enumerate(val):
            pred, _ = memory.predict(ex["input"])
            rows.append({"q": qi, "system": name, "target": ex["target"], "pred": pred,
                         "correct": is_correct(evaluator(pred, ex["target"]))})
        if time.time() - started > 30:
            raise RuntimeError(f"{name} replay took >30s: responses are not coming from the cache")
    return pd.DataFrame(rows), memories


def check_against_logged(table: pd.DataFrame) -> None:
    """Replayed accuracy must equal the accuracy the run logged in val.json."""
    for name, acc in table.groupby("system")["correct"].mean().items():
        logged = pd.read_json(RUN / name / MODEL_LEAF / "val.json", typ="series")["accuracy"]
        if abs(acc - logged) > 1e-9:
            raise RuntimeError(f"{name}: replay {acc:.3f} != logged {logged:.3f}")


def mcnemar_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        only_a = int((wide[a] & ~wide[b]).sum())
        only_b = int((~wide[a] & wide[b]).sum())
        n = only_a + only_b
        p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
        rows.append({"a": a, "b": b, "acc_a": wide[a].mean(), "acc_b": wide[b].mean(),
                     "only_a_correct": only_a, "only_b_correct": only_b, "p_exact": p})
    return pd.DataFrame(rows)


def bootstrap_table(wide: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(wide), size=(BOOTSTRAP_DRAWS, len(wide)))
    ref = wide[REFERENCE].to_numpy()
    rows = []
    for name in wide.columns:
        if name == REFERENCE:
            continue
        diff = wide[name].to_numpy().astype(float) - ref
        draws = diff[idx].mean(axis=1)
        rows.append({"system": name, "diff_vs_ref": diff.mean(),
                     "ci_low": np.quantile(draws, 0.025), "ci_high": np.quantile(draws, 0.975)})
    return pd.DataFrame(rows)


def failure_features(table: pd.DataFrame, memories: dict, val: list[dict]) -> pd.DataFrame:
    """Features known before the model answers: what the harness injected for each question."""
    rows = []
    for name in ERROR_MEMORIES:
        if name not in memories:
            continue
        memory = memories[name]
        min_score = getattr(memory, "min_score", 0.0)
        for qi, ex in enumerate(val):
            ranked = rank_texts(ex["input"], [it["symptoms"] for it in memory.items])
            used = [(s, memory.items[i]) for s, i in ranked[: memory.top_items] if s > 0 and s >= min_score]
            diseases = {d.lower() for _, it in used for d in (it["right"], it.get("wrong", ""))} - {""}
            outcome = table[(table.system == name) & (table.q == qi)].iloc[0]
            rows.append({
                "system": name, "q": qi, "failed": int(not outcome.correct),
                "n_injected": len(used),
                "max_similarity": max((s for s, _ in used), default=0.0),
                # every question gets 3 cues, so "any other disease named" is constant;
                # these two vary: is the true disease among the cues, how many others are
                "target_in_cues": int(ex["target"].lower() in diseases),
                "n_other_diseases": len(diseases - {ex["target"].lower()}),
                "pred_is_injected_disease": int(outcome.pred.strip().lower() in diseases),
                "prompt_chars": len(memory._build_prompt(ex["input"])),
            })
    return pd.DataFrame(rows)


def fit_logit(feat: pd.DataFrame) -> str:
    x = feat[["max_similarity", "target_in_cues", "n_other_diseases"]].astype(float)
    x["prompt_kchars"] = feat["prompt_chars"] / 1000
    x = pd.concat([x, pd.get_dummies(feat["system"], drop_first=True, dtype=float)], axis=1)
    try:
        return str(sm.Logit(feat["failed"], sm.add_constant(x)).fit(disp=False).summary())
    except Exception as exc:  # small n can make the fit singular; report rather than hide
        return f"logit failed: {exc}"


def plots(mc: pd.DataFrame, boot: pd.DataFrame, feat: pd.DataFrame, order: list[str]) -> None:
    pmat = pd.DataFrame(np.nan, index=order, columns=order)
    for r in mc.itertuples():
        pmat.loc[r.a, r.b] = pmat.loc[r.b, r.a] = r.p_exact
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pmat.to_numpy(dtype=float), cmap="viridis_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), order)
    for i, j in itertools.product(range(len(order)), repeat=2):
        if i != j:
            ax.text(j, i, f"{pmat.iat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, label="exact McNemar p")
    ax.set_title("Pairwise McNemar p (validation, n=50)")
    fig.tight_layout()
    fig.savefig(OUT / "mcnemar_p.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 4))
    b = boot.sort_values("diff_vs_ref")
    ax.errorbar(b.diff_vs_ref * 100, range(len(b)),
                xerr=[(b.diff_vs_ref - b.ci_low) * 100, (b.ci_high - b.diff_vs_ref) * 100], fmt="o", capsize=4)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(b)), b.system)
    ax.set_xlabel(f"accuracy difference vs {REFERENCE} (%p), paired bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap_ci.png", dpi=160)

    if not feat.empty:
        rates = feat.groupby(["system", "target_in_cues"])["failed"].agg(["mean", "size"]).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, (name, g) in enumerate(rates.groupby("system")):
            for _, r in g.iterrows():
                xpos = k + (0.2 if not r.target_in_cues else -0.2)
                ax.bar(xpos, r["mean"] * 100, width=0.38, color="#c2410c" if not r.target_in_cues else "#9a948a")
                ax.text(xpos, r["mean"] * 100 + 1, f"n={int(r['size'])}", ha="center", fontsize=8)
        ax.set_xticks(range(rates.system.nunique()), sorted(rates.system.unique()))
        ax.set_ylabel("failure rate (%)")
        ax.set_title("Failure rate: true disease absent from injected cues (orange) vs present (grey)")
        fig.tight_layout()
        fig.savefig(OUT / "failure_by_distractor.png", dpi=160)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, val, _, evaluator = load_dataset_splits_3way("Symptom2Disease", 60, 50, 212, 42)
    table, memories = replay(SYSTEMS, val, evaluator)
    check_against_logged(table)
    table.to_csv(OUT / "per_question.csv", index=False)

    order = [s for s in SYSTEMS if s in memories]
    wide = table.pivot(index="q", columns="system", values="correct")[order]
    mc = mcnemar_table(wide)
    boot = bootstrap_table(wide)
    feat = failure_features(table, memories, val)
    mc.to_csv(OUT / "mcnemar.csv", index=False)
    boot.to_csv(OUT / "bootstrap.csv", index=False)
    feat.to_csv(OUT / "failures.csv", index=False)
    (OUT / "logit.txt").write_text(fit_logit(feat) if not feat.empty else "no error-memory systems")
    plots(mc, boot, feat, order)

    print(wide.mean().round(3).to_string())
    print("\nvs reference:")
    print(mc[(mc.a == REFERENCE) | (mc.b == REFERENCE)].round(4).to_string(index=False))
    print("\nbootstrap:")
    print(boot.round(3).to_string(index=False))
    if not feat.empty:
        print("\nfailure rate by target_in_cues:")
        print(feat.groupby(["system", "target_in_cues"])["failed"].agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
