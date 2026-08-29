"""
Regression tests for the Task 2 leakage controls.

These are the properties whose *silent* failure would invalidate every number
in the results table, so they are asserted in code rather than trusted to
convention: a random split, a label reaching the feature matrix, or a rolling
window peeking forward are all invisible in the metrics -- they just make them
better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features
from src.models import evaluation, splitting


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _toy_panel(n_loans: int = 40, n_months: int = 60) -> pd.DataFrame:
    """A small, well-formed panel: monthly rows per loan from 2018-01."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for i in range(n_loans):
        loan_id = f"L{i:06d}"
        start = pd.Timestamp("2018-01-01")
        balance = 300_000.0
        for month in range(n_months):
            dpd = int(rng.choice([0, 0, 0, 30, 60], p=[0.8, 0.08, 0.05, 0.05, 0.02]))
            balance *= 0.997
            rows.append(
                {
                    "loan_id": loan_id,
                    "month_index": month,
                    "reporting_month": start + pd.DateOffset(months=month),
                    "origination_month": start,
                    "loan_age_months": month,
                    "remaining_term_months": 360 - month,
                    "original_balance": 300_000.0,
                    "current_balance": balance,
                    "interest_rate": 4.5,
                    "credit_score_band": "700-739",
                    "ltv_band": "75-80%",
                    "state": "CA",
                    "current_status": "Current" if dpd == 0 else f"{dpd}-DPD",
                    "days_past_due": dpd,
                    "modification_flag": False,
                    "document_status": "Complete",
                    "servicer_name": "Atlas Mortgage Services",
                    "source_system": "CoreServicing",
                    "vintage_year": 2018,
                    "next_3m_delinquency_flag": float(rng.random() < 0.1),
                    "next_12m_default_flag": float(rng.random() < 0.08),
                    "next_state": "Current",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _toy_panel()


@pytest.fixture(scope="module")
def boundaries() -> splitting.SplitBoundaries:
    # Each window is 18 months wide, so a 12-month horizon still leaves rows
    # in every window after purging.
    return splitting.SplitBoundaries(
        train_end=pd.Timestamp("2020-12-01"),
        valid_end=pd.Timestamp("2022-06-01"),
        test_end=pd.Timestamp("2022-12-01"),
    )


# --------------------------------------------------------------------------
# Feature-level leakage
# --------------------------------------------------------------------------
def test_forbidden_columns_never_enter_the_feature_spec(panel):
    _, spec = features.build_feature_matrix(panel)
    assert not set(spec.full) & set(config.FORBIDDEN_FEATURES)
    assert not set(spec.baseline) & set(config.FORBIDDEN_FEATURES)


def test_leak_assertion_raises_on_a_target_column():
    with pytest.raises(ValueError, match="Leaky columns"):
        features.assert_no_leaky_features(["loan_age_months", "next_12m_default_flag"])


def test_rolling_features_use_no_future_rows(panel):
    """
    Truncating the panel after month t must not change month t's features.

    This is the operational definition of "backward-looking": if a feature at
    month t differs depending on whether months t+1.. exist, it saw the future.
    """
    full, spec = features.build_feature_matrix(panel)
    cutoff = pd.Timestamp("2020-06-01")
    truncated, _ = features.build_feature_matrix(panel[panel["reporting_month"] <= cutoff])

    rolling_columns = [
        c for c in spec.numeric
        if c.startswith(("dpd_", "paydown_")) or c.endswith(("_to_date", "_this_month"))
    ] + ["months_since_delinquency", "worst_status_to_date"]
    rolling_columns = [c for c in dict.fromkeys(rolling_columns) if c in full.columns]
    assert rolling_columns, "expected some rolling features to check"

    keys = ["loan_id", "reporting_month"]
    left = full[full["reporting_month"] <= cutoff].set_index(keys)[rolling_columns].sort_index()
    right = truncated.set_index(keys)[rolling_columns].sort_index()

    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_cross_sectional_features_use_only_the_same_month(panel):
    """The market-rate benchmark for month t must match month t's own rows."""
    engineered, _ = features.build_feature_matrix(panel)
    for month, group in engineered.groupby("reporting_month"):
        expected = group["interest_rate"].median()
        assert np.allclose(group["market_rate_median"].dropna(), expected)
        break


# --------------------------------------------------------------------------
# Split-level leakage
# --------------------------------------------------------------------------
def test_split_windows_are_time_ordered_and_disjoint(panel, boundaries):
    engineered, _ = features.build_feature_matrix(panel)
    split = splitting.make_time_split(
        engineered, "next_3m_delinquency_flag", boundaries=boundaries
    )
    assert split.train["reporting_month"].max() < split.valid["reporting_month"].min()
    assert split.valid["reporting_month"].max() < split.test["reporting_month"].min()


def test_training_window_is_purged_by_the_target_horizon(panel, boundaries):
    """
    A 12-month target must not train on rows whose window reaches validation.

    The last row kept in training has to be at least ``horizon`` months before
    the first validation month.
    """
    engineered, _ = features.build_feature_matrix(panel)
    split = splitting.make_time_split(engineered, "next_12m_default_flag", boundaries=boundaries)

    last_train = split.train["reporting_month"].max()
    first_valid = split.valid["reporting_month"].min()
    gap = (first_valid.year - last_train.year) * 12 + (first_valid.month - last_train.month)
    assert gap > split.horizon_months - 1


def test_audit_rejects_a_random_row_level_split(panel):
    """The audit must fail the exact split style Task 2 forbids."""
    engineered, _ = features.build_feature_matrix(panel)
    shuffled = engineered.sample(frac=1.0, random_state=config.RANDOM_SEED)
    third = len(shuffled) // 3
    with pytest.raises(ValueError, match="Time-aware split violated"):
        splitting.audit_split(shuffled[:third], shuffled[third : 2 * third], shuffled[2 * third :])


def test_absorbing_rows_are_excluded_from_every_window(panel, boundaries):
    engineered, _ = features.build_feature_matrix(panel)
    engineered.loc[engineered.index[:50], "current_status"] = "Prepaid"
    split = splitting.make_time_split(
        engineered, "next_3m_delinquency_flag", boundaries=boundaries
    )
    for window in (split.train, split.valid, split.test):
        assert not window["current_status"].isin(config.ABSORBING_STATES).any()


def test_empty_window_after_purge_raises_a_useful_error(panel):
    """A window no longer than the horizon must fail loudly, not silently."""
    engineered, _ = features.build_feature_matrix(panel)
    too_narrow = splitting.SplitBoundaries(
        train_end=pd.Timestamp("2020-12-01"),
        valid_end=pd.Timestamp("2021-12-01"),  # 12 months, equal to the horizon
        test_end=pd.Timestamp("2022-12-01"),
    )
    with pytest.raises(ValueError, match="empty after purging"):
        splitting.make_time_split(engineered, "next_12m_default_flag", boundaries=too_narrow)


# --------------------------------------------------------------------------
# Metric correctness
# --------------------------------------------------------------------------
def test_recall_at_precision_returns_zero_when_the_floor_is_unreachable():
    y_true = np.zeros(1000)
    y_true[:5] = 1
    y_prob = np.random.default_rng(0).random(1000)
    recall, threshold = evaluation.recall_at_precision(y_true, y_prob, floor=0.9)
    assert recall == 0.0
    assert np.isnan(threshold)


def test_multiclass_metrics_are_invariant_to_class_ordering():
    """
    Severity-ordered classes must score identically to alphabetical ones.

    scikit-learn binds probability columns to labels in sorted order inside
    ``log_loss`` and ``roc_auc_score``; this test is what caught that.
    """
    rng = np.random.default_rng(0)
    classes = ["Current", "30-DPD", "Default"]
    proba = rng.dirichlet(np.ones(3), size=500)
    y_true = np.asarray(classes)[np.argmax(proba, axis=1)]

    ordered = evaluation.multiclass_metrics(y_true, proba, classes)

    permutation = [2, 0, 1]
    shuffled_classes = [classes[i] for i in permutation]
    shuffled_proba = proba[:, permutation]
    shuffled = evaluation.multiclass_metrics(y_true, shuffled_proba, shuffled_classes)

    for key in ("macro_f1", "log_loss", "brier", "roc_auc_ovr_macro", "accuracy"):
        assert ordered[key] == pytest.approx(shuffled[key])
