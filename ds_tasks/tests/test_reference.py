"""The reference pipeline must reproduce churn-guard's saved notebook outputs."""

import pandas as pd
import pytest

import reference as ref


@pytest.fixture(scope="module")
def built():
    return ref.build_dataset()


def test_census_join_matches_notebook_printout(built):
    _, report = built
    # census_income_join.ipynb cell 9: 1,652 ZIPs, 1,651 matched, 95250 unmatched
    assert report.telco_zips == 1652
    assert report.matched_zips == 1651
    assert report.unmatched_zips == (95250,)
    # cell 11: 568 customers without income after the join
    assert report.customers_missing_income == 568


def test_income_levels_match_notebook_printout(built):
    df, _ = built
    counts = df["Area_Income_Level"].value_counts().sort_index().to_dict()
    # cell 13 printout (fixed bins)
    assert counts == {"Low": 150, "Medium": 1852, "High": 3393, "Very High": 1648}


def test_welch_ttest_reproduces_saved_output_on_committed_csv():
    # income_segmentation_churn_analysis.ipynb cell 10 ran on the committed CSV (blank incomes dropped)
    committed = pd.read_csv(ref.DATA_DIR / ref.COMMITTED_JOINED_CSV)
    result = ref.welch_ttest(committed, "Income_Charge_Ratio")
    assert result.t == pytest.approx(11.7208, abs=1e-4)
    assert result.p == pytest.approx(4.7677e-31, rel=1e-4)
    assert result.mean_churned == pytest.approx(0.0108, abs=1e-4)
    assert result.mean_retained == pytest.approx(0.0087, abs=1e-4)


def test_model_excludes_leakage_and_is_deterministic(built):
    df, _ = built
    x, _ = ref.model_matrix(df)
    assert not any(c.startswith(("Churn_Label", "Churn_Score", "Churn_Reason")) for c in x.columns)
    first, second = ref.train_lgbm(df), ref.train_lgbm(df)
    assert first == second
    assert 0.80 < first.roc_auc < 0.95
