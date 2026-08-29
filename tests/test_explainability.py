"""
Regression tests for the Task 6 explainability and governance pipeline.

The risk in a governance report is not that it crashes -- it is that it says
something confident and wrong. A disparity screen that flags every segment, a
SHAP matrix computed on a differently-encoded frame than the one scored, a
credit-band gap reported as a fairness finding: all of them produce a report
that looks rigorous and misleads a reader who trusts it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.explain import errors, fairness, shap_values


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def test_stratified_sample_preserves_a_rare_class():
    """
    A uniform sample of a 9%-positive panel spends its budget on non-events.

    Without stratification the beeswarm describes the model's behaviour on the
    quiet majority, which is not the behaviour anyone is reviewing.
    """
    outcome = np.zeros(10000, dtype=int)
    outcome[:900] = 1  # 9% positive
    frame = pd.DataFrame({"x": np.arange(10000)})

    positions = shap_values.stratified_sample(frame, outcome, n_rows=1000)
    sampled_rate = outcome[positions].mean()

    assert 900 <= len(positions) <= 1100
    assert sampled_rate == pytest.approx(0.09, abs=0.01)


def test_sampling_is_a_no_op_when_the_budget_exceeds_the_data():
    frame = pd.DataFrame({"x": np.arange(50)})
    positions = shap_values.stratified_sample(frame, np.zeros(50, dtype=int), n_rows=500)
    assert len(positions) == 50


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------
def test_predictions_are_classified_at_the_deployed_threshold():
    """
    Not at 0.5. The Task 2 threshold is tuned on validation, and classifying at
    0.5 would describe a model nobody deployed.
    """
    y_true = np.array([1, 1, 0, 0])
    probability = np.array([0.9, 0.1, 0.9, 0.1])
    outcome = errors.classify_predictions(y_true, probability, threshold=0.3)

    assert list(outcome) == [
        "true positive", "false negative", "false positive", "true negative"
    ]


def test_rates_carry_the_counts_behind_them():
    """
    A rate without its denominator cannot be significance tested, and an
    untested rate gap on twenty events is how a governance report cries wolf.
    """
    frame = pd.DataFrame({"segment": ["A"] * 10})
    outcome = pd.Series(
        ["true positive"] * 2 + ["false positive"] * 3 + ["false negative"] + ["true negative"] * 4
    )
    table = errors.error_rates_by_segment(frame, outcome, "segment", min_group=1)
    row = table.iloc[0]

    # The frame's schema is fixed: `segment` names the column, `group` the level.
    assert list(table.columns[:2]) == ["segment", "group"]

    for column in ("false_positives", "false_negatives", "true_positives",
                   "actual_negatives", "actual_positives"):
        assert column in table.columns

    assert row["false_positives"] == 3
    assert row["actual_negatives"] == 7
    assert row["false_positive_rate"] == pytest.approx(3 / 7)
    assert row["recall"] == pytest.approx(2 / 3)


def test_small_groups_are_suppressed():
    """A false positive rate computed on nine loans is noise with a decimal point."""
    frame = pd.DataFrame({"segment": ["big"] * 300 + ["tiny"] * 9})
    outcome = pd.Series(["true negative"] * 309)
    table = errors.error_rates_by_segment(frame, outcome, "segment", min_group=200)

    assert set(table["group"]) == {"big"}
    assert len(table) == 1


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------
def test_perfect_calibration_has_near_zero_error():
    rng = np.random.default_rng(config.RANDOM_SEED)
    probability = rng.uniform(0, 1, 40000)
    y_true = (rng.uniform(0, 1, 40000) < probability).astype(int)

    table = errors.reliability_table(y_true, probability)
    assert errors.expected_calibration_error(table) < 0.02


def test_calibration_error_is_weighted_by_bin_population():
    """
    A wild miss in a bin holding four records must not outweigh a small bias
    across the bulk of the book.
    """
    table = pd.DataFrame(
        {
            "records": [10000, 4],
            "mean_predicted": [0.10, 0.90],
            "observed_rate": [0.11, 0.10],
        }
    )
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    ece = errors.expected_calibration_error(table)

    assert ece < 0.02  # dominated by the 10,000-record bin, not the 4-record one


# --------------------------------------------------------------------------
# Disparity screen
# --------------------------------------------------------------------------
def _disparity_table(worst_events, worst_n, best_events, best_n, segment="state"):
    return pd.DataFrame(
        {
            segment: ["worst", "best"],
            "records": [worst_n, best_n],
            "actual_negatives": [worst_n, best_n],
            "false_positives": [worst_events, best_events],
            "false_positive_rate": [worst_events / worst_n, best_events / best_n],
        }
    )


def test_a_large_gap_on_few_events_is_not_escalated():
    """
    Sixteen false positives against four is a ratio of 0.25 and roughly nothing.

    This is the exact case that made an earlier version of the screen escalate
    eight of fifteen segment-metric pairs. A report that flags everything is one
    nobody reads.
    """
    table = _disparity_table(worst_events=16, worst_n=400, best_events=4, best_n=300)
    summary = fairness.disparity_summary(table, "state", "false_positive_rate")

    assert bool(summary["below_floor"].iloc[0]) is True
    assert bool(summary["escalate"].iloc[0]) is False


def test_a_large_gap_on_many_events_is_escalated():
    table = _disparity_table(worst_events=400, worst_n=4000, best_events=40, best_n=4000)
    summary = fairness.disparity_summary(table, "state", "false_positive_rate")

    assert bool(summary["significant"].iloc[0]) is True
    assert bool(summary["escalate"].iloc[0]) is True


def test_a_credit_band_gap_is_never_escalated():
    """
    A model that flagged sub-620 and 800+ borrowers alike would be broken.

    Reporting that as a fairness finding confuses a risk model doing its job
    with a risk model doing harm, and buries the findings that matter.
    """
    table = _disparity_table(400, 4000, 40, 4000, segment="credit_score_band")
    summary = fairness.disparity_summary(table, "credit_score_band", "false_positive_rate")

    assert bool(summary["significant"].iloc[0]) is True
    assert bool(summary["escalate"].iloc[0]) is False
    assert summary["kind"].iloc[0].startswith("risk factor")


def test_vintage_is_treated_as_a_risk_factor():
    """
    Within one reporting window, vintage is almost pure loan age -- 54 months
    for the 2018 cohort against 3.5 for 2023 -- and seasoning is a legitimate
    driver of default hazard.
    """
    assert "vintage_year" in fairness.RISK_FACTOR_SEGMENTS
    assert fairness.segment_kind("vintage_year").startswith("risk factor")
    assert fairness.segment_kind("state") == "screen for disparity"


def test_disparity_is_suppressed_when_the_model_flags_most_of_the_book():
    """
    A model flagging half the book produces enormous group gaps that describe
    the threshold, not the treatment. The prepayment head does exactly this.
    """
    table = _disparity_table(400, 4000, 40, 4000)
    summary = fairness.disparity_summary(
        table, "state", "false_positive_rate", overall_selection_rate=0.54
    )

    assert bool(summary["interpretable"].iloc[0]) is False
    assert bool(summary["escalate"].iloc[0]) is False


def test_two_proportion_test_matches_a_known_case():
    """Identical rates must be indistinguishable; a wide gap must not be."""
    assert fairness._two_proportion_p_value(50, 1000, 50, 1000) == pytest.approx(1.0)
    assert fairness._two_proportion_p_value(200, 1000, 50, 1000) < 1e-10


# --------------------------------------------------------------------------
# Local explanations
# --------------------------------------------------------------------------
def _fake_result(n=200, features=("a", "b", "c")):
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, (n, len(features)))
    X = pd.DataFrame(rng.normal(0, 1, (n, len(features))), columns=list(features))
    frame = pd.DataFrame({config.ID_COL: [f"L{i}" for i in range(n)], config.TIME_COL: "2023-01-01"})
    return shap_values.ShapResult(
        target="t", label="demo", values=values, base_value=-2.0,
        X=X, frame=frame, feature_names=list(features), positions=np.arange(n),
    )


def test_local_explanation_sums_to_the_model_output():
    """
    The waterfall has to add up. If the "all other features" residual is wrong
    the chart shows a total the model never produced.
    """
    result = _fake_result()
    explanation = shap_values.local_explanation(result, position=7, top_n=2)

    reconstructed = (
        explanation.attrs["base_value"]
        + explanation["shap_value"].sum()
        + explanation.attrs["other_features"]
    )
    assert reconstructed == pytest.approx(explanation.attrs["log_odds"])


def test_demo_loans_are_not_a_highlight_reel():
    """
    Picking only confident hits demonstrates the model on the records where
    nothing was ever in doubt.
    """
    result = _fake_result(n=500)
    rng = np.random.default_rng(1)
    probability = rng.uniform(0, 1, 500)
    outcome = (rng.uniform(0, 1, 500) < 0.3).astype(int)

    picks = shap_values.pick_demo_loans(result, probability, outcome)
    cases = set(picks["case"])

    assert "confident false positive" in cases or "missed event (false negative)" in cases
    assert len(picks) >= 2
