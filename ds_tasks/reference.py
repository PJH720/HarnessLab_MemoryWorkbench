"""Reference implementation of the churn-guard analysis, extracted from its notebooks.

Every task-card answer is computed here, deterministically. Sources:
- census_income_join.ipynb cells 3, 5, 11, 13  -> build_census_lookup, join_income, derive_features
- income_segmentation_churn_analysis.ipynb cell 10 -> welch_ttest
- ensemble/income notebooks (LightGBM + SHAP)      -> train_lgbm, top_shap_features

Fixes relative to the notebooks: probabilities are evaluated on a held-out split (the
retention notebook scored its own training data), and leakage columns are always dropped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(os.environ.get(
    "CHURN_DATA_DIR",
    Path(__file__).resolve().parent.parent / "Reference/churn-guard/data",
))
TELCO_CSV = "Telco-Customer-Churn2025.csv"
CENSUS_CSV = "ACSST5Y2024.S1901_2026-07-01T071953/ACSST5Y2024.S1901-Data.csv"
COMMITTED_JOINED_CSV = "telco_churn_with_income.csv"

CENSUS_COLUMNS = {
    "S1901_C01_001E": "Area_Total_Households",
    "S1901_C01_012E": "Area_Median_Income",
    "S1901_C01_013E": "Area_Mean_Income",
}
INCOME_BINS = [0, 40_000, 75_000, 120_000, float("inf")]
INCOME_LABELS = ["Low", "Medium", "High", "Very High"]
TARGET = "Churn Value"
LEAKAGE = ["Churn Label", "Churn Score", "Churn Reason"]
ID_AND_GEO = ["CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude"]
SEED = 42


def load_telco(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / TELCO_CSV, encoding="utf-8-sig")
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0.0)
    return df


def build_census_lookup(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw = pd.read_csv(data_dir / CENSUS_CSV, skiprows=[1])
    lookup = raw[["GEO_ID", *CENSUS_COLUMNS]].rename(columns=CENSUS_COLUMNS)
    lookup["Zip_Code"] = lookup["GEO_ID"].str[-5:].astype(int)
    for col in CENSUS_COLUMNS.values():
        # suppressed or top-coded values ("-", "N", "250,000+") become NaN
        lookup[col] = pd.to_numeric(lookup[col], errors="coerce")
    return lookup.drop(columns=["GEO_ID"])


@dataclass(frozen=True)
class JoinReport:
    telco_zips: int
    census_zctas: int
    matched_zips: int
    unmatched_zips: tuple[int, ...]
    customers_missing_income: int


def join_income(telco: pd.DataFrame, lookup: pd.DataFrame) -> tuple[pd.DataFrame, JoinReport]:
    """Left join on ZIP; missing income (unmatched or suppressed) filled with the Census median."""
    telco_zips, census_zips = set(telco["Zip Code"]), set(lookup["Zip_Code"])
    joined = telco.merge(lookup, left_on="Zip Code", right_on="Zip_Code", how="left").drop(columns=["Zip_Code"])
    missing = int(joined["Area_Median_Income"].isna().sum())
    for col in CENSUS_COLUMNS.values():
        joined[col] = joined[col].fillna(lookup[col].median())
    report = JoinReport(len(telco_zips), len(census_zips), len(telco_zips & census_zips),
                        tuple(sorted(telco_zips - census_zips)), missing)
    return joined, report


def derive_features(joined: pd.DataFrame) -> pd.DataFrame:
    out = joined.copy()
    out["Income_Charge_Ratio"] = out["Monthly Charges"] / (out["Area_Median_Income"] / 12)
    out["Area_Income_Level"] = pd.cut(out["Area_Median_Income"], bins=INCOME_BINS, labels=INCOME_LABELS)
    return out


def build_dataset(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, JoinReport]:
    joined, report = join_income(load_telco(data_dir), build_census_lookup(data_dir))
    return derive_features(joined), report


def churn_rate_by(df: pd.DataFrame, column: str) -> dict[str, float]:
    rates = df.groupby(column, observed=True)[TARGET].mean()
    return {str(k): float(v) for k, v in rates.items()}


@dataclass(frozen=True)
class TTest:
    mean_churned: float
    mean_retained: float
    t: float
    p: float
    cohens_d: float


def welch_ttest(df: pd.DataFrame, column: str) -> TTest:
    churned = df.loc[df[TARGET] == 1, column].dropna()
    retained = df.loc[df[TARGET] == 0, column].dropna()
    t, p = stats.ttest_ind(churned, retained, equal_var=False)
    pooled = np.sqrt((churned.var(ddof=1) + retained.var(ddof=1)) / 2)
    return TTest(float(churned.mean()), float(retained.mean()), float(t), float(p),
                 float((churned.mean() - retained.mean()) / pooled))


def model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features for churn modeling: leakage, identifiers and geography removed, categoricals one-hot."""
    x = df.drop(columns=[TARGET, "CLTV", *LEAKAGE, *ID_AND_GEO, "Area_Income_Level"], errors="ignore")
    x = pd.get_dummies(x, columns=list(x.select_dtypes(exclude="number").columns), dtype=float)
    x.columns = [c.replace(" ", "_") for c in x.columns]
    return x, df[TARGET]


@dataclass(frozen=True)
class ModelResult:
    roc_auc: float
    recall_at_05: float
    top_shap: tuple[str, ...]


def train_lgbm(df: pd.DataFrame, n_top: int = 3) -> ModelResult:
    x, y = model_matrix(df)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight="balanced",
                               random_state=SEED, n_jobs=1, deterministic=True, verbose=-1)
    model.fit(x_tr, y_tr)
    proba = model.predict_proba(x_te)[:, 1]
    # LightGBM's built-in TreeSHAP; last column is the bias term
    contrib = model.booster_.predict(x_te, pred_contrib=True)[:, :-1]
    order = np.argsort(-np.abs(contrib).mean(axis=0))[:n_top]
    return ModelResult(float(roc_auc_score(y_te, proba)),
                       float(recall_score(y_te, (proba >= 0.5).astype(int))),
                       tuple(x.columns[i] for i in order))
