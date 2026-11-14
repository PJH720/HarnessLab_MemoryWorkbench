"""Fig 9: design of the confirmatory holdout experiment. Draws no result, only the
holdout construction (DEDUP_REPORT.json from scripts/build_holdout.py) and the planned power.

  usage: uv run --project harness_lab/text_classification --with matplotlib --with scipy \
           python analysis/confirm/plot_design.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/multiseed"))
from power_plan import exact_power  # noqa: E402

REPORT = ROOT / "harness_lab/text_classification/data/symptom_holdout/DEDUP_REPORT.json"
OUT = ROOT / "analysis/figures"
DELTA, DISCORDANCE = 0.05, 0.20
MARKS = {212: ("original test", (8, -34), "left"), 572: ("this holdout", (10, -46), "left"),
         658: ("80% power target", (-8, 12), "right")}
INK, MUTED, KEEP, DROP = "#1f2933", "#7b8794", "#0b7285", "#c92a2a"


def construction(ax, report: dict) -> None:
    ex = report["excluded"]
    steps = [
        ("Source train rows", report["source_rows"]["train"], MUTED),
        ("Already used (exact)", -ex["exact_vendored"], DROP),
        ("Near-duplicate of used", -ex["near_duplicate_vendored"], DROP),
        ("Duplicate inside holdout", -(ex["near_duplicate_within_holdout"] + ex["exact_within_holdout"]), DROP),
        ("Holdout test set", report["n_holdout"], KEEP),
    ]
    level = 0
    for row, (label, value, color) in enumerate(steps):
        left, width = (0, value) if value > 0 else (level + value, -value)
        ax.barh(row, width, left=left, color=color, height=0.62)
        ax.text(left + width + 12, row, f"{value:+d}" if value < 0 else f"{value}", va="center", color=INK, fontsize=10)
        level = value if row == 0 else level + value if value < 0 else level
    ax.set_yticks(range(len(steps)), [s[0] for s in steps])
    ax.invert_yaxis()
    ax.set_xlim(0, report["source_rows"]["train"] * 1.14)
    ax.set_xlabel("Questions")
    ax.set_title(f"A. Building the holdout (cosine cutoff {report['threshold']}, {report['labels']} labels)", loc="left")


def power(ax) -> None:
    grid = list(range(50, 1001, 25))
    ax.plot(grid, [exact_power(n, DELTA, DISCORDANCE) for n in grid], color=KEEP, lw=2)
    ax.axhline(0.80, color=MUTED, lw=1, ls="--")
    for n, (label, offset, align) in MARKS.items():
        p = exact_power(n, DELTA, DISCORDANCE)
        ax.scatter([n], [p], color=INK, zorder=3, s=28)
        ax.annotate(f"n={n}\n{label}\npower {p:.2f}", (n, p), textcoords="offset points",
                    xytext=offset, ha=align, fontsize=9, color=INK)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Independent paired questions")
    ax.set_ylabel("Power to detect a 5 %p difference")
    ax.set_title("B. Planned power (20% discordant pairs, two-sided alpha 0.05)", loc="left")


def main() -> int:
    report = json.loads(REPORT.read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    construction(axes[0], report)
    power(axes[1])
    fig.suptitle("Confirmatory experiment design (preregistered; no results yet)", x=0.01, ha="left", fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig9_confirmatory_design.{suffix}", dpi=300)
    return 0


if __name__ == "__main__":
    sys.exit(main())
