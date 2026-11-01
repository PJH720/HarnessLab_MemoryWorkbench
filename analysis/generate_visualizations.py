"""Build the eight analysis figures and the gallery README in analysis/figures/.

Every number drawn or written is read from analysis/multiseed/out/*.csv,
logit_h3.txt or ds_tasks/runs/pilot-s0/trajectories.parquet at run time.
Nothing under analysis/multiseed/out/ is modified.

Run:
    uv run --project harness_lab/text_classification \
      --with matplotlib --with seaborn --with pandas --with pyarrow --with scipy \
      python analysis/generate_visualizations.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "multiseed" / "out"
TRAJ = ROOT / "ds_tasks" / "runs" / "pilot-s0" / "trajectories.parquet"
FIG = ROOT / "analysis" / "figures"

sys.path.insert(0, str(ROOT / "analysis" / "multiseed"))
from power_plan import exact_power

BLUE, CORAL, EMERALD, SLATE = "#2B59C3", "#E4572E", "#17A07A", "#6B7A8F"
GOLD, PLUM, SKY = "#D9A21B", "#8E5CA8", "#7FA7D9"
REF = "knn_top6"
ORDER = ["no_memory", "knn_top6", "fewshot_all", "contrast_online", "cue_bank", "cue_bank_verified", "reflexion_notes"]
COLOR = {
    "no_memory": SLATE,
    "knn_top6": BLUE,
    "fewshot_all": GOLD,
    "contrast_online": SKY,
    "cue_bank": CORAL,
    "cue_bank_verified": EMERALD,
    "reflexion_notes": PLUM,
}
DS_ORDER = ["ds_base", "ds_knn", "ds_cue_verified"]
DS_COLOR = {"ds_base": SLATE, "ds_knn": BLUE, "ds_cue_verified": EMERALD}
SEED_MARK = {42: "o", 123: "s", 456: "D"}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def note(ax: plt.Axes, text: str, y: float = -0.2) -> None:
    ax.text(0.0, y, text, transform=ax.transAxes, fontsize=7.5, color="#555", va="top")


acc = pd.read_csv(OUT / "accuracy_tokens.csv")
boot = pd.read_csv(OUT / "bootstrap.csv")
mcn = pd.read_csv(OUT / "mcnemar_by_seed.csv")
feat = pd.read_csv(OUT / "features.csv")
noise = pd.read_csv(OUT / "noise.csv")
power = pd.read_csv(OUT / "power_plan.csv")
traj = pd.read_parquet(TRAJ)
facts: dict[str, object] = {}


# Fig 1: accuracy vs tokens, Pareto frontier over seed means
def fig1() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, split in zip(axes, ("test", "val")):
        m = (
            acc[acc.split == split]
            .groupby("sys")
            .agg(acc=("accuracy", "mean"), lo=("accuracy", "min"), hi=("accuracy", "max"), tok=("cached_eval_input_tokens", "mean"))
            .reindex(ORDER)
        )
        frontier, best = [], -1.0
        for sys_, r in m.sort_values("tok").iterrows():
            if r.acc > best:
                frontier.append(sys_)
                best = r.acc
        f = m.loc[frontier].sort_values("tok")
        ax.step(f.tok, f.acc * 100, where="post", color="#999", lw=1, ls="--", zorder=1, label="Pareto frontier (seed means)")
        for sys_, r in m.iterrows():
            on = sys_ in frontier
            ax.errorbar(r.tok, r.acc * 100, yerr=[[max(0.0, 100 * (r.acc - r.lo))], [max(0.0, 100 * (r.hi - r.acc))]], fmt="o",
                        ms=9 if on else 6.5, color=COLOR[sys_], mec="black" if on else "white", mew=1.2,
                        capsize=3, elinewidth=1, zorder=3,
                        label=f"{sys_}  {r.acc*100:.1f}%  ({r.tok/1000:,.0f}k){'  ◆ frontier' if on else ''}")
        ax.set_xscale("log")
        ax.set_xlabel("Cached eval input tokens per run (log scale)")
        ax.set_ylabel("Accuracy (%) · mean of 3 seeds, bar = min–max")
        ax.set_title(f"{split.capitalize()} split (n={int(acc[acc.split == split].n.iloc[0])} per seed)")
        if split == "test":
            k, fa = m.loc["knn_top6"], m.loc["fewshot_all"]
            saving = 1 - k.tok / fa.tok
            b = boot[(boot.split == "test") & (boot.sys == "fewshot_all")].iloc[0]
            facts.update(f1_saving=saving, f1_knn=k.acc, f1_fa=fa.acc, f1_knn_tok=k.tok, f1_fa_tok=fa.tok,
                         f1_ci=(b.diff_vs_ref, b.ci_low, b.ci_high), f1_frontier=list(f.index))
            y = k.acc * 100 - 3.0
            ax.annotate("", xy=(k.tok, y), xytext=(fa.tok, y), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
            ax.text(np.sqrt(k.tok * fa.tok), y - 0.6,
                    f"−{saving*100:.0f}% tokens; Δacc(fewshot_all − knn) {b.diff_vs_ref*100:+.1f} pp\n"
                    f"95% CI [{b.ci_low*100:+.1f}, {b.ci_high*100:+.1f}] → not distinguishable",
                    ha="center", va="top", fontsize=7.3, color=BLUE)
        ax.legend(loc="lower right", fontsize=6.8, handletextpad=0.4)
    fig.suptitle("Fig 1. Accuracy vs input-token cost across 7 memory harnesses", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_accuracy_vs_tokens_pareto")


# Fig 2: bars of seed means with individual seeds overlaid
def fig2() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, split in zip(axes, ("val", "test")):
        d = acc[acc.split == split].assign(pct=lambda x: x.accuracy * 100)
        means = d.groupby("sys").pct.mean().reindex(ORDER)
        ax.bar(range(len(ORDER)), means.values, color=[COLOR[s] for s in ORDER], alpha=0.35, width=0.7,
               edgecolor=[COLOR[s] for s in ORDER], lw=1.2)
        for i, s in enumerate(ORDER):
            for seed, r in d[d.sys == s].set_index("seed").iterrows():
                ax.scatter(i + rng.uniform(-0.17, 0.17), r.pct, marker=SEED_MARK[seed], s=26, color=COLOR[s],
                           edgecolor="black", lw=0.5, zorder=3)
            spread = d[d.sys == s].pct
            ax.text(i, spread.max() + 0.8, f"{means[s]:.1f}", ha="center", va="bottom", fontsize=7.5)
            ax.text(i, 51.5, f"Δ{spread.max()-spread.min():.1f}", ha="center", fontsize=6.8, color="#666")
        ax.set_xticks(range(len(ORDER)), ORDER, rotation=35, ha="right")
        ax.set_ylim(50, 92)
        ax.set_title(f"{split.capitalize()} (n={int(d.n.iloc[0])} per seed)")
        ax.set_ylabel("Accuracy (%)")
    handles = [plt.Line2D([], [], marker=m, ls="", color="#444", label=f"seed {s}") for s, m in SEED_MARK.items()]
    axes[1].legend(handles=handles, loc="upper right", fontsize=7.5)
    note(axes[0], "Bars = mean of 3 seeds; markers = individual seeds; Δ = max−min across seeds (pp).\n"
                  "no_memory has no training memory, so all three seeds give identical answers (points overlap).", y=-0.36)
    fig.suptitle("Fig 2. System accuracy by split with seed-level variability", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_system_performance_by_split")
    facts["f2_spread"] = acc.groupby(["split", "sys"]).accuracy.agg(lambda s: (s.max() - s.min()) * 100).unstack(0)


# Fig 3: forest plot of bootstrap CIs vs knn_top6
def fig3() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    systems = [s for s in ORDER if s != REF]
    y = np.arange(len(systems))[::-1]
    for ax, split in zip(axes, ("val", "test")):
        b = boot[boot.split == split].set_index("sys").reindex(systems)
        for yi, (s, r) in zip(y, b.iterrows()):
            crosses = r.ci_low <= 0 <= r.ci_high
            ax.errorbar(r.diff_vs_ref * 100, yi, xerr=[[100 * (r.diff_vs_ref - r.ci_low)], [100 * (r.ci_high - r.diff_vs_ref)]],
                        fmt="o", color=COLOR[s], mfc="white" if crosses else COLOR[s], ms=7, capsize=3, lw=1.6)
            ax.text(17, yi, f"{r.diff_vs_ref*100:+.1f} [{r.ci_low*100:+.1f}, {r.ci_high*100:+.1f}]", va="center", ha="right",
                    fontsize=7.3)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlim(-21, 17.5)
        ax.set_yticks(y, systems)
        ax.set_xlabel(f"Accuracy difference vs {REF} (pp)")
        n = int(b.n.iloc[0])
        ax.set_title(f"{split.capitalize()} (n={n} = 3 seeds × {n // 3} items)")
    cb_y = y[systems.index("cue_bank")]
    cb = boot[(boot.split == "test") & (boot.sys == "cue_bank")].iloc[0]
    axes[1].annotate("H1 (cue_bank < knn_top6):\nNOT SUPPORTED, CI includes 0", xy=(cb.diff_vs_ref * 100, cb_y),
                     xytext=(-20, cb_y + 1.3), fontsize=7.5, color=CORAL, arrowprops=dict(arrowstyle="->", color=CORAL))
    note(axes[0], "Paired bootstrap, 10,000 draws, items resampled within seed (RNG seed 0). Open marker = CI includes 0.\n"
                  "Seeds re-evaluate the same items, so pooled rows are not independent; the primary test is per-seed McNemar.")
    fig.suptitle("Fig 3. Paired-bootstrap 95% CIs relative to knn_top6", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_bootstrap_forest_plot")
    facts["f3"] = boot.set_index(["split", "sys"])


# Fig 4: seed 42, same memory on 50 vs 212 items
def fig4() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    a42 = acc[acc.seed == 42].set_index(["split", "sys"]).accuracy
    ax = axes[0]
    for s in ("knn_top6", "cue_bank"):
        ys = [a42["val", s] * 100, a42["test", s] * 100]
        ax.plot([0, 1], ys, "-o", color=COLOR[s], lw=2, ms=8, label=s)
        for x, yv in zip([0, 1], ys):
            ax.text(x - 0.06 if x == 0 else x + 0.06, yv, f"{yv:.1f}%", ha="right" if x == 0 else "left", va="center",
                    fontsize=8, color=COLOR[s])
    gap = [(a42[sp, "cue_bank"] - a42[sp, "knn_top6"]) * 100 for sp in ("val", "test")]
    for x, gv in zip([0, 1], gap):
        ax.text(x, 66.0, f"Δ {gv:+.1f} pp", ha="center", fontsize=9, fontweight="bold", color=CORAL)
    ax.set_xticks([0, 1], ["Val (n=50)", "Test (n=212)"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(64, 88)
    ax.set_ylabel("Accuracy (%) · seed 42, same memory")
    ax.set_title("Same memory, two sample sizes")
    ax.legend(loc="upper center", fontsize=8)

    ax = axes[1]
    m = mcn[(mcn.seed == 42) & (mcn.sys == "cue_bank")].set_index("split")
    x = np.arange(2)
    ref_only = [int(m.loc[sp, "only_ref"]) for sp in ("val", "test")]
    sys_only = [int(m.loc[sp, "only_sys"]) for sp in ("val", "test")]
    ax.bar(x - 0.18, ref_only, 0.36, color=BLUE, label="only knn_top6 correct")
    ax.bar(x + 0.18, sys_only, 0.36, color=CORAL, label="only cue_bank correct")
    for i, sp in enumerate(("val", "test")):
        ax.text(i - 0.18, ref_only[i] + 0.3, str(ref_only[i]), ha="center", fontsize=8)
        ax.text(i + 0.18, sys_only[i] + 0.3, str(sys_only[i]), ha="center", fontsize=8)
        ax.text(i, max(ref_only[i], sys_only[i]) + 2.0, f"exact McNemar p = {m.loc[sp, 'p_exact']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Val (n=50)", "Test (n=212)"])
    ax.set_ylim(0, 18)
    ax.set_ylabel("Discordant items")
    ax.set_title("Discordant pairs behind the gap")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Fig 4. A {gap[0]:+.0f} pp gap on 50 items shrinks to {gap[1]:+.1f} pp on 212 items (seed 42)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_sample_size_reproducibility")
    facts["f4"] = dict(gap=gap, val=(ref_only[0], sys_only[0], m.loc["val", "p_exact"]),
                       test=(ref_only[1], sys_only[1], m.loc["test", "p_exact"]))


def parse_h3() -> dict[str, tuple[float, float]]:
    """Coefficient and p-value from the preregistered H3 test-split logit (not re-estimated here)."""
    block = (OUT / "logit_h3.txt").read_text().split("H3 test;", 1)[1].split("H3 val;", 1)[0]
    found = {}
    for name in ("target_in_cues", "max_similarity", "prompt_kchars"):
        mt = re.search(rf"^{name}\s+(-?[\d.]+)\s+[\d.]+\s+-?[\d.]+\s+([\d.]+)", block, re.MULTILINE)
        found[name] = (float(mt.group(1)), float(mt.group(2)))
    return found


# Fig 5: error rate by target_in_cues, and anchoring among errors
def fig5() -> None:
    systems = ["cue_bank", "cue_bank_verified", "reflexion_notes"]
    d = feat[(feat.split == "test") & (feat.feature_status == "ok") & feat.system.isin(systems)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    w = 0.36
    rows = []
    for i, s in enumerate(systems):
        for j, t in enumerate((0, 1)):
            sub = d[(d.system == s) & (d.target_in_cues == t)]
            k, n = int(sub.failed.sum()), len(sub)
            lo, hi = wilson(k, n)
            rate = k / n
            rows.append((s, t, k, n))
            xpos = i + (j - 0.5) * w
            ax.bar(xpos, rate * 100, w * 0.95, color=COLOR[s], alpha=0.4 if t == 0 else 0.95, edgecolor=COLOR[s])
            ax.errorbar(xpos, rate * 100, yerr=[[100 * (rate - lo)], [100 * (hi - rate)]], color="black", capsize=2.5, lw=1)
            ax.text(xpos, 1.0, f"{k}/{n}", ha="center", fontsize=6.8, color="white" if t else "#222", fontweight="bold")
    ax.set_xticks(range(len(systems)), systems)
    ax.set_ylabel("Error rate (%) · Wilson 95% CI")
    ax.set_title("Error rate: true label absent (light) / present (dark) in cues", fontsize=10)
    h3 = parse_h3()
    c, p = h3["target_in_cues"]
    ax.text(0.98, 0.97, f"H3 logit (test, cluster-robust SE):\ntarget_in_cues β = {c:+.3f}, p = {p:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, bbox=dict(fc="white", ec="#ccc"))

    ax = axes[1]
    fails = d[d.failed == 1]
    mat = fails.pivot_table(index="system", columns="target_in_cues", values="pred_is_injected_disease", aggfunc="mean").reindex(systems)
    cnt = fails.pivot_table(index="system", columns="target_in_cues", values="pred_is_injected_disease", aggfunc="size").reindex(systems)
    hits = fails.pivot_table(index="system", columns="target_in_cues", values="pred_is_injected_disease", aggfunc="sum").reindex(systems)
    ann = (mat * 100).round(0).astype(int).astype(str) + "%\n(" + hits.astype(int).astype(str) + "/" + cnt.astype(int).astype(str) + ")"
    sns.heatmap(mat * 100, annot=ann, fmt="", cmap="Reds", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "% of errors"},
                linewidths=1, linecolor="white", annot_kws={"fontsize": 8})
    ax.set_xticklabels(["label absent\nfrom cues", "label present\nin cues"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Among errors: predicted disease was named in the cues", fontsize=10)
    note(axes[0], "Test split, 3 seeds pooled (rows = seed × item, not independent).\n"
                  "knn_top6 injects raw examples rather than cues, so it has no feature rows.")
    fig.suptitle("Fig 5. Cognitive-anchor features of LLM-written cue memories", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_cognitive_anchor_features")
    facts["f5"] = dict(rows=rows, h3=h3, hits=hits, cnt=cnt)


# Fig 6: data-science agent pilot, held-out phase only
def fig6() -> None:
    ev = traj[traj.phase == "eval"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    ax = axes[0]
    res = []
    for i, h in enumerate(DS_ORDER):
        sub = ev[ev.harness == h]
        k, n = int(sub.correct.sum()), len(sub)
        lo, hi = wilson(k, n)
        res.append((h, k, n))
        ax.bar(i, k / n * 100, color=DS_COLOR[h], alpha=0.85, width=0.6)
        ax.errorbar(i, k / n * 100, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], color="black", capsize=3)
        ax.text(i, 3, f"{k}/{n}", ha="center", color="white", fontweight="bold", fontsize=8.5)
        ax.text(i, hi * 100 + 2, f"{k/n*100:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(range(3), DS_ORDER)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Held-out accuracy (%) · Wilson 95% CI")
    ax.set_title("Accuracy (eval phase)")

    ax = axes[1]
    sns.boxplot(data=ev, x="harness", y="steps", order=DS_ORDER, hue="harness", palette=DS_COLOR, legend=False, ax=ax,
                width=0.5, fliersize=0, boxprops=dict(alpha=0.35))
    sns.stripplot(data=ev, x="harness", y="steps", order=DS_ORDER, hue="correct", palette={True: EMERALD, False: CORAL},
                  ax=ax, size=4.5, jitter=0.18, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("")
    ax.set_ylabel("Agent steps per task")
    ax.set_title("Steps to finish")

    ax = axes[2]
    rng = np.random.default_rng(1)
    offset = {"ds_base": -0.12, "ds_knn": 0.0, "ds_cue_verified": 0.12}
    for h in DS_ORDER:
        sub = ev[ev.harness == h]
        yj = sub.correct.astype(int) + offset[h] + rng.uniform(-0.05, 0.05, len(sub))
        ax.scatter(sub.tokens_in, yj, color=DS_COLOR[h], s=22, alpha=0.8, edgecolor="black", lw=0.3, label=h)
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["failed", "correct"])
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("Input tokens per task (log scale)")
    ax.set_title("Token cost vs outcome")
    ax.legend(fontsize=7.5, loc="center right")
    for a in axes[:2]:
        a.tick_params(axis="x", labelrotation=20)
    axes[1].legend(title="correct", fontsize=7.5, title_fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.text(0.01, -0.04, "Pilot: seed 0, 2 folds, 31 held-out task cards per harness; memory-building (train) rows excluded. "
             "ds_knn finishes in a median of 2 steps; whether retrieved memory hands over near-identical answers is unchecked.",
             fontsize=7.5, color="#555")
    fig.suptitle(f"Fig 6. Data-science agent pilot ({len(traj)} trajectories, {len(ev)} held-out)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_ds_tasks_benchmark")
    facts["f6"] = dict(res=res, pooled=traj.groupby("harness").correct.agg(["sum", "size"]),
                       steps=ev.groupby("harness").steps.median(), tok=ev.groupby("harness").tokens_in.median(),
                       total=len(traj))


# Fig 7: exact McNemar power vs N, reusing analysis/multiseed/power_plan.py
def fig7() -> None:
    row = power.query("minimum_effect_pp == 5 and assumed_discordance == 0.2").iloc[0]
    assert int(row.independent_pairs_required) == 658
    assert abs(exact_power(658, 0.05, 0.2) - row.power_at_n) < 1e-9
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ns = np.unique(np.round(np.geomspace(20, 3000, 70)).astype(int))
    styles = {0.1: "-", 0.2: "--", 0.3: ":"}
    colors = {3.0: CORAL, 5.0: BLUE, 8.0: EMERALD}
    for eff in (3.0, 5.0, 8.0):
        for disc in (0.1, 0.2, 0.3):
            ax.plot(ns, [exact_power(int(n), eff / 100, disc) for n in ns], styles[disc], color=colors[eff], lw=1.5,
                    label=f"{eff:.0f} pp, discordance {disc:.0%}")
    ax.scatter(power.independent_pairs_required, power.power_at_n, color="black", zorder=4, s=18,
               label="power_plan.csv (N for 80%)")
    ax.axhline(0.8, color="#888", lw=0.8)
    for n, lab in ((50, "val n=50"), (212, "test n=212")):
        ax.axvline(n, color=SLATE, lw=1, ls="-.")
        ax.text(n * 1.04, 0.9, lab, rotation=90, fontsize=7.5, color=SLATE, va="top")
    ax.annotate("658 items:\n5 pp at 20% discordance", (658, row.power_at_n), xytext=(1100, 0.42), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Independent paired items N (log scale)")
    ax.set_ylabel("Power (1 − β)")
    ax.legend(fontsize=6.8, ncol=2, loc="upper left")
    ax.set_title("Fig 7. Exact two-sided McNemar power, α = 0.05", loc="left")
    fig.tight_layout()
    save(fig, "fig7_power_analysis_curve")
    facts["f7"] = dict(p50=exact_power(50, 0.05, 0.2), p212=exact_power(212, 0.05, 0.2), p212_8=exact_power(212, 0.08, 0.2))


# Fig 8: repeat-call flip rates
def fig8() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    x = np.arange(len(noise))
    ax.bar(x - 0.18, noise.correctness_flip_rate * 100, 0.36, color=BLUE, label="correctness flipped (right ↔ wrong)")
    ax.bar(x + 0.18, noise.raw_prediction_flip_rate * 100, 0.36, color=CORAL, label="predicted label changed")
    for i, r in noise.iterrows():
        ax.text(i - 0.18, r.correctness_flip_rate * 100 + 0.3, f"{r.correctness_flips}/{r.n}", ha="center", fontsize=8)
        ax.text(i + 0.18, r.raw_prediction_flip_rate * 100 + 0.3, f"{r.raw_prediction_flips}/{r.n}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{a} vs {b}" for a, b in zip(noise.a, noise.b)])
    ax.set_ylabel("Share of items (%)")
    ax.set_ylim(0, 18)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Fig 8. Repeat-call noise on the local LLM server (same input & settings, val n=50)", loc="left")
    fig.tight_layout()
    save(fig, "fig8_runtime_stochasticity")


def readme(stamp: str) -> str:
    f3, f4, f5, f6, f7, sp = facts["f3"], facts["f4"], facts["f5"], facts["f6"], facts["f7"], facts["f2_spread"]
    cb, nm, fa = f3.loc[("test", "cue_bank")], f3.loc[("test", "no_memory")], f3.loc[("test", "fewshot_all")]
    n5 = {(s, t): (k, n) for s, t, k, n in f5["rows"]}
    tic_c, tic_p = f5["h3"]["target_in_cues"]
    sim_c, sim_p = f5["h3"]["max_similarity"]
    pk_c, pk_p = f5["h3"]["prompt_kchars"]
    pooled = f6["pooled"]
    ci = facts["f1_ci"]

    def pp(x: float) -> str:
        return f"{x*100:+.1f}"

    lines = [
        "# 분석 그림 갤러리",
        "",
        f"- 생성: {stamp}. `analysis/generate_visualizations.py`가 그림 8종과 이 문서를 함께 만듭니다.",
        "- 원천: `analysis/multiseed/out/*.csv`, `logit_h3.txt`, `ds_tasks/runs/pilot-s0/trajectories.parquet`",
        "- 본문 숫자는 모두 위 파일에서 실행 시점에 계산했습니다. 각 그림은 PNG(300 dpi)와 PDF(벡터)로 저장됩니다.",
        "",
        "```bash",
        "uv run --project harness_lab/text_classification \\",
        "  --with matplotlib --with seaborn --with pandas --with pyarrow --with scipy \\",
        "  python analysis/generate_visualizations.py",
        "```",
        "",
        "## Fig 1. 정확도와 입력 토큰의 파레토 관계",
        "",
        "![Fig 1](fig1_accuracy_vs_tokens_pareto.png)",
        "",
        f"- test에서 knn_top6는 평균 {facts['f1_knn_tok']:,.0f} 토큰으로 {facts['f1_knn']*100:.1f}%, "
        f"fewshot_all은 {facts['f1_fa_tok']:,.0f} 토큰으로 {facts['f1_fa']*100:.1f}%입니다. knn_top6의 입력 토큰이 {facts['f1_saving']*100:.0f}% 적습니다.",
        f"- 정확도 차이(fewshot_all − knn_top6)는 {pp(ci[0])}%p, 95% CI [{pp(ci[1])}, {pp(ci[2])}]입니다. "
        "0을 포함하므로 \"같다\"가 아니라 \"이 규모로는 구별되지 않는다\"로 읽어야 합니다.",
        f"- test 파레토 경계(시드 평균 기준)는 {' → '.join(facts['f1_frontier'])}입니다. 경계 위 점들 사이의 정확도 차이는 모두 Fig 3의 CI 안에 있어, "
        "경계의 순서 자체도 이 표본 규모에서는 확정적이지 않습니다.",
        "",
        "## Fig 2. 분할별 성능과 시드 변동",
        "",
        "![Fig 2](fig2_system_performance_by_split.png)",
        "",
        f"- 시드 간 최대−최소 폭은 val에서 최대 {sp['val'].max():.1f}%p, test에서 최대 {sp['test'].max():.1f}%p입니다. 50문항에서는 한 문항이 2%p라 폭이 커집니다.",
        "- no_memory는 학습 기억이 없어 세 시드의 답이 같습니다(점 세 개가 겹침).",
        "",
        "## Fig 3. knn_top6 대비 페어드 부트스트랩 95% CI",
        "",
        "![Fig 3](fig3_bootstrap_forest_plot.png)",
        "",
        f"- 사전 등록 주 가설 H1(cue_bank < knn_top6)의 test 차이는 {pp(cb.diff_vs_ref)}%p, CI [{pp(cb.ci_low)}, {pp(cb.ci_high)}]로 0을 포함합니다. "
        "판정은 NOT SUPPORTED이며, 효과가 없다는 증명은 아닙니다.",
        f"- CI가 0에서 떨어진 시스템은 no_memory뿐입니다: test {pp(nm.diff_vs_ref)}%p [{pp(nm.ci_low)}, {pp(nm.ci_high)}]. "
        f"fewshot_all은 {pp(fa.diff_vs_ref)}%p [{pp(fa.ci_low)}, {pp(fa.ci_high)}]입니다.",
        "- 부트스트랩은 시드 안에서 문항을 재표집했습니다. 세 시드는 같은 212문항을 다시 평가한 것이라 n=636을 독립 표본으로 볼 수 없고, 주 검정은 시드별 McNemar입니다.",
        "",
        "## Fig 4. 소표본과 대표본의 재현성",
        "",
        "![Fig 4](fig4_sample_size_reproducibility.png)",
        "",
        f"- seed 42의 같은 기억으로 cue_bank − knn_top6는 val(50문항)에서 {f4['gap'][0]:+.1f}%p, test(212문항)에서 {f4['gap'][1]:+.1f}%p였습니다.",
        f"- 불일치 문항은 val {f4['val'][0]}:{f4['val'][1]} (exact p={f4['val'][2]:.3f}), test {f4['test'][0]}:{f4['test'][1]} (p={f4['test'][2]:.3f})입니다. "
        "val의 큰 격차도 불일치 8문항에서 나온 값이고 5% 수준에서 유의하지 않았습니다.",
        "",
        "## Fig 5. 단서 기억의 인지적 닻 피처",
        "",
        "![Fig 5](fig5_cognitive_anchor_features.png)",
        "",
        "- 대상은 LLM이 쓴 단서를 주입하는 세 시스템(cue_bank, cue_bank_verified, reflexion_notes)의 test 행입니다. knn_top6는 원시 예시를 넣으므로 피처 행이 없습니다.",
        "- 정답 질환이 단서에 있을 때 vs 없을 때 오답 수: "
        + "; ".join(f"{s} {n5[(s, 1)][0]}/{n5[(s, 1)][1]} vs {n5[(s, 0)][0]}/{n5[(s, 0)][1]}" for s in ("cue_bank", "cue_bank_verified", "reflexion_notes"))
        + ".",
        f"- 사전 등록 H3 로짓(문항 군집 강건 표준오차)에서 target_in_cues 계수는 {tic_c:+.3f} (p={tic_p:.3f})로 효과가 보이지 않습니다. "
        f"유의했던 항은 max_similarity {sim_c:+.3f} (p={sim_p:.3f})와 prompt_kchars {pk_c:+.3f} (p={pk_p:.3f})입니다.",
        "- 오른쪽 히트맵은 오답 중 예측 질환이 단서에 나온 질환이었던 비율입니다. 단서가 오답을 끌어당기는 양상을 서술적으로 보여 줄 뿐 인과 효과의 근거는 아닙니다.",
        "",
        "## Fig 6. 데이터사이언스 에이전트 파일럿",
        "",
        "![Fig 6](fig6_ds_tasks_benchmark.png)",
        "",
        f"- 궤적 {f6['total']}건 중 기억을 만드는 train 단계를 빼고, 처음 보는 과제(eval, 하네스당 31건)만 그렸습니다.",
        "- eval 정확도: " + ", ".join(f"{h} {k}/{n} = {k/n*100:.1f}%" for h, k, n in f6["res"]) + ".",
        "- 요청서의 74.2%(ds_knn)와 46.8%(ds_cue_verified)는 train과 eval을 합친 값("
        + ", ".join(f"{h} {int(pooled.loc[h, 'sum'])}/{int(pooled.loc[h, 'size'])}" for h in ("ds_knn", "ds_cue_verified"))
        + ")이라, eval 31건만 있는 ds_base와 분모가 다릅니다. 처음 보는 과제의 성능 비교에는 eval 값을 씁니다.",
        f"- ds_knn의 eval 중앙 스텝은 {f6['steps']['ds_knn']:.0f}, 중앙 입력 토큰은 {f6['tok']['ds_knn']:,.0f}입니다"
        f"(ds_base {f6['steps']['ds_base']:.0f}스텝, {f6['tok']['ds_base']:,.0f}토큰). 검색된 기억이 거의 같은 답을 넘겨주는지는 확인하지 않았습니다.",
        "- 시드 1개 파일럿이므로 하네스 간 차이를 결론으로 쓰지 않습니다.",
        "",
        "## Fig 7. 통계적 검정력 곡선",
        "",
        "![Fig 7](fig7_power_analysis_curve.png)",
        "",
        "- `analysis/multiseed/power_plan.py`의 정확 McNemar 검정력 함수로 곡선을 그렸습니다. 검은 점은 `power_plan.csv`의 80% 달성 표본 수이고, 스크립트가 5%p·불일치 20% → 658을 재확인합니다.",
        f"- 불일치율 20%에서 5%p 효과의 검정력은 n=50에서 {f7['p50']:.2f}, n=212에서 {f7['p212']:.2f}입니다. 8%p 효과도 n=212에서 {f7['p212_8']:.2f}입니다.",
        "- 이번 test 212문항은 5%p 안팎의 차이를 가려내도록 설계된 규모가 아닙니다. 확인 실험에는 독립 문항 수백 개 단위가 필요합니다.",
        "",
        "## Fig 8. 로컬 LLM 서빙의 반복 호출 잡음",
        "",
        "![Fig 8](fig8_runtime_stochasticity.png)",
        "",
        "- 같은 입력·같은 설정으로 다시 돌린 세 쌍(val 50문항)에서 정오가 바뀐 문항은 "
        + ", ".join(f"{r.correctness_flips}/{r.n}" for r in noise.itertuples())
        + f" ({noise.correctness_flip_rate.min()*100:.0f}~{noise.correctness_flip_rate.max()*100:.0f}%), 예측 라벨이 바뀐 문항은 "
        + ", ".join(f"{r.raw_prediction_flips}/{r.n}" for r in noise.itertuples())
        + f" ({noise.raw_prediction_flip_rate.min()*100:.0f}~{noise.raw_prediction_flip_rate.max()*100:.0f}%)입니다.",
        "- 50문항에서 몇 문항 차이는 이 잡음 범위 안에서 읽어야 합니다. 서버 쪽의 정확한 원인은 확인하지 못했습니다.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for build in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        build()
        print("built", build.__name__)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    (FIG / "README.md").write_text(readme(stamp), encoding="utf-8")
    print("wrote", FIG / "README.md")
