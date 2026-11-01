"""Generate task cards; every answer comes from reference.py.

Usage: uv run python build_cards.py   (writes cards/*.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import reference as ref

OUT = Path(__file__).parent / "cards"
PIPELINE = (
    "Data: Telco-Customer-Churn2025.csv (customers) and the Census ACS S1901 CSV "
    "(skip the second header row). ZIP = last 5 characters of GEO_ID as int; "
    "Area_Median_Income = S1901_C01_012E coerced to numeric. Left-join customers on 'Zip Code'; "
    "fill missing Area_Median_Income with the median of the Census column. "
    "Income_Charge_Ratio = Monthly Charges / (Area_Median_Income / 12). "
    "Area_Income_Level = bins [0, 40000, 75000, 120000, inf) labelled Low/Medium/High/Very High. "
    "Churn is 'Churn Value' (1 = churned)."
)
ABS_RATE = {"abs": 0.005}
REL_STAT = {"rel": 0.01}
P_VALUE = {"log10": 0.5}


def card(cid, question, answer, answer_type, tags, tolerance=None, note=None, needs_pipeline=True):
    return {"id": cid, "question": (PIPELINE + "\n\n" if needs_pipeline else "") + question,
            "answer": answer, "answer_type": answer_type, "tolerance": tolerance or {}, "tags": tags,
            "inputs": [ref.TELCO_CSV, ref.CENSUS_CSV], **({"note": note} if note else {})}


def build() -> list[dict]:
    df, rep = ref.build_dataset()
    cards = [
        card("join_unique_zips", "How many unique ZIP codes are in the customer file?", rep.telco_zips, "number", ["join"], needs_pipeline=False),
        card("join_census_zctas", "How many ZCTA rows does the Census file have (after skipping the description row)?", rep.census_zctas, "number", ["join"], needs_pipeline=False),
        card("join_matched_zips", "How many of the customers' unique ZIP codes exist in the Census file?", rep.matched_zips, "number", ["join", "trap:dtype"],
             note="Census ZIP must be cast to int before matching"),
        card("join_unmatched_zip", "Which customer ZIP code has no Census match? Answer the ZIP.", str(rep.unmatched_zips[0]), "category", ["join"]),
        card("join_missing_income", "After the left join and before filling, how many customers have missing Area_Median_Income?", rep.customers_missing_income, "number", ["join"]),
        card("describe_overall_churn", "What is the overall churn rate (fraction, 0-1)?", float(df[ref.TARGET].mean()), "number", ["describe"], ABS_RATE, needs_pipeline=False),
        card("derive_mean_icr", "What is the mean Income_Charge_Ratio over all customers?", float(df["Income_Charge_Ratio"].mean()), "number", ["derive"], {"rel": 0.005}),
        card("derive_low_count", "How many customers fall in Area_Income_Level 'Low'?", int((df["Area_Income_Level"] == "Low").sum()), "number", ["derive"]),
        card("describe_corr_icr", "What is the Pearson correlation between Income_Charge_Ratio and Churn Value?",
             float(df["Income_Charge_Ratio"].corr(df[ref.TARGET])), "number", ["describe", "derive"], {"abs": 0.005}),
    ]
    for level, rate in ref.churn_rate_by(df, "Area_Income_Level").items():
        cards.append(card(f"segment_income_{level.lower().replace(' ', '_')}", f"What is the churn rate (0-1) of customers in Area_Income_Level '{level}'?", rate, "number", ["segment", "derive"], ABS_RATE))
    for column, values in {"Contract": ["Month-to-month", "Two year"], "Internet Service": ["Fiber optic"],
                           "Payment Method": ["Electronic check"], "Senior Citizen": ["Yes"]}.items():
        rates = ref.churn_rate_by(df, column)
        for v in values:
            slug = f"{column}_{v}".lower().replace(" ", "_").replace("-", "_")
            cards.append(card(f"segment_{slug}", f"What is the churn rate (0-1) of customers with {column} = '{v}'?", rates[v], "number", ["segment"], ABS_RATE, needs_pipeline=False))
    for contract in ["Month-to-month", "Two year"]:
        sub = df[(df["Area_Income_Level"] == "Low") & (df["Contract"] == contract)]
        cards.append(card(f"cross_low_{contract.lower().replace(' ', '_').replace('-', '_')}",
                          f"What is the churn rate (0-1) of customers with Area_Income_Level 'Low' AND Contract '{contract}'?",
                          float(sub[ref.TARGET].mean()), "number", ["segment", "cross"], ABS_RATE))
    for column in ["Income_Charge_Ratio", "Monthly Charges", "Tenure Months"]:
        tt = ref.welch_ttest(df, column)
        slug = column.lower().replace(" ", "_")
        needs = column == "Income_Charge_Ratio"
        cards.append(card(f"ttest_{slug}_t", f"Welch's t-test (unequal variances), churned minus retained, on '{column}': what is the t statistic?", tt.t, "number", ["test"], REL_STAT, needs_pipeline=needs))
        cards.append(card(f"ttest_{slug}_p", f"Same Welch's t-test on '{column}': what is the two-sided p-value?", tt.p, "number", ["test"], P_VALUE, needs_pipeline=needs))
    tt = ref.welch_ttest(df, "Income_Charge_Ratio")
    cards.append(card("ttest_icr_cohens_d", "Cohen's d for Income_Charge_Ratio, churned minus retained, using the average of the two sample variances (ddof=1) as pooled variance?",
                      tt.cohens_d, "number", ["test", "effect-size"], REL_STAT))
    committed = ref.welch_ttest(pd.read_csv(ref.DATA_DIR / ref.COMMITTED_JOINED_CSV), "Income_Charge_Ratio")
    cards.append({**card("ttest_committed_csv_t", "Using only telco_churn_with_income.csv, drop rows with missing Income_Charge_Ratio and run Welch's t-test (churned vs retained). What is t?",
                         committed.t, "number", ["test", "trap:missing"], REL_STAT, needs_pipeline=False,
                         note="differs from the filled pipeline; tests whether the agent follows the stated missing-value rule"),
                  "inputs": [ref.COMMITTED_JOINED_CSV]})
    model = ref.train_lgbm(df)
    spec = ("Model: drop Churn Value (target), CLTV, Churn Label, Churn Score, Churn Reason, ID/geo columns "
            "(CustomerID, Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude) and Area_Income_Level; "
            "one-hot other categoricals; stratified 80/20 split random_state=42; "
            "LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight='balanced', random_state=42, deterministic=True).")
    cards += [
        card("model_roc_auc", spec + " What is the test ROC-AUC?", model.roc_auc, "number", ["model", "trap:leakage"], {"abs": 0.01},
             note="including Churn Score/Label inflates AUC toward 1"),
        card("model_recall", spec + " What is the test recall at threshold 0.5?", model.recall_at_05, "number", ["model"], {"abs": 0.02}),
        card("model_top_shap", spec + " Which 3 one-hot feature columns have the largest mean |SHAP| on the test set? (spaces replaced by underscores)",
             list(model.top_shap), "set", ["model", "explain"]),
    ]
    return cards


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cards = build()
    ids = [c["id"] for c in cards]
    assert len(ids) == len(set(ids)), "duplicate card ids"
    for c in cards:
        (OUT / f"{c['id']}.json").write_text(json.dumps(c, ensure_ascii=False, indent=2))
    print(f"wrote {len(cards)} cards")
    for c in cards:
        ans = c["answer"] if not isinstance(c["answer"], float) else round(c["answer"], 6)
        print(f"{c['id']:<34} {str(c['tags']):<34} {ans}")


if __name__ == "__main__":
    main()
