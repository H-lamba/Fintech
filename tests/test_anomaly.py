"""
Regression tests for the Task 4 anomaly and exception pipeline.

The pipeline's failure modes are quiet ones: a rule that silently never
evaluates, an ablation that smuggles back the signal it claims to remove, a
class label that a CSV round-trip turns into NaN. None of those raise, and each
of them makes the reported numbers mean something other than what they say.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.anomaly import curation, evaluation, explain, features as fm, models, signals
from src.profiling import rules


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _row(loan_id, age, status, balance=250_000.0, origination="2021-01-01", dpd=0):
    start = pd.Timestamp(origination)
    return {
        "loan_id": loan_id,
        "reporting_month": start + pd.DateOffset(months=age),
        "origination_month": start,
        "loan_age_months": age,
        "remaining_term_months": 360 - age,
        "original_balance": 250_000.0,
        "current_balance": balance,
        "interest_rate": 5.0,
        "original_term_months": 360,
        "current_status": status,
        "days_past_due": dpd,
        "modification_flag": False,
        "document_status": "Complete",
        "source_system": "CoreServicing",
        "servicer_name": "Atlas Mortgage Services",
        "prepayment_flag": 1 if status == "Prepaid" else 0,
        "default_flag": 1 if status == "Default" else 0,
        "exception_required": False,
        "exception_type": np.nan,
    }


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """
    Four loans, one per case the detectors must separate.

    ``clean``   an ordinary performing loan
    ``zombie``  prepays at month 3, then emits two more active rows
    ``jumper``  Current -> 90-DPD with no intermediate buckets
    ``ladder``  a legitimate 30 -> 60 -> 90 escalation, which must NOT fire
    """
    rows = []
    rows += [_row("clean", age, "Current") for age in range(6)]

    rows += [_row("zombie", age, "Current") for age in range(3)]
    rows += [_row("zombie", 3, "Prepaid", balance=0.0)]
    rows += [_row("zombie", age, "Current") for age in (4, 5)]

    rows += [_row("jumper", age, "Current") for age in range(3)]
    rows += [_row("jumper", 3, "90-DPD", dpd=95)]

    rows += [_row("ladder", 0, "Current")]
    rows += [_row("ladder", 1, "30-DPD", dpd=35)]
    rows += [_row("ladder", 2, "60-DPD", dpd=65)]
    rows += [_row("ladder", 3, "90-DPD", dpd=95)]

    return pd.DataFrame(rows).reset_index(drop=True)


def _fired(matrix: pd.DataFrame, panel: pd.DataFrame, column: str, loan: str, age: int) -> bool:
    mask = (panel["loan_id"] == loan) & (panel["loan_age_months"] == age)
    return bool(matrix.loc[mask, column].iloc[0])


# --------------------------------------------------------------------------
# The rule parser
# --------------------------------------------------------------------------
def test_expression_columns_ignore_string_literals_and_operators():
    """
    Identifiers inside quotes and bare operator words are not columns.

    This is the bug that made four of the organiser's own rules report as "not
    applicable to this data": ``'90-DPD'`` contributed ``DPD`` to the required
    column list, and ``and`` / ``not`` contributed themselves, so no real frame
    could ever satisfy the requirement and the rule was skipped in silence.
    """
    referenced = rules._referenced_columns(
        "not (current_status == '90-DPD' and days_past_due < 90)"
    )
    assert referenced == ["current_status", "days_past_due"]


def test_expression_rules_are_applicable_against_a_real_frame(toy_panel):
    rule_set = rules.build_rule_set(
        [{"name": "T", "expression": "not (default_flag == 1 and prepayment_flag == 1)"}]
    )
    assert rule_set[0].applicable(toy_panel)


# --------------------------------------------------------------------------
# Sequence-aware detectors
# --------------------------------------------------------------------------
def test_post_absorbing_activity_fires_only_after_a_terminal_row(toy_panel):
    matrix, _ = signals.sequence_detectors(toy_panel)
    column = "seq__post_absorbing_activity"

    assert _fired(matrix, toy_panel, column, "zombie", 4)
    assert _fired(matrix, toy_panel, column, "zombie", 5)
    # The terminal row itself is not the defect; the rows after it are.
    assert not _fired(matrix, toy_panel, column, "zombie", 3)
    assert not matrix.loc[toy_panel["loan_id"] == "clean", column].any()


def test_illegal_transition_fires_on_a_bucket_skip_but_not_a_ladder(toy_panel):
    """
    The detector must separate ``Current -> 90-DPD`` from ``60-DPD -> 90-DPD``.

    Without the second half of this assertion the detector could simply flag
    every 90-DPD row and still look perfect on the injected defects.
    """
    matrix, _ = signals.sequence_detectors(toy_panel)
    column = "seq__illegal_status_transition"

    assert _fired(matrix, toy_panel, column, "jumper", 3)
    assert not _fired(matrix, toy_panel, column, "ladder", 3)
    assert not matrix.loc[toy_panel["loan_id"] == "ladder", column].any()


def test_sequence_detectors_preserve_the_caller_row_order(toy_panel):
    """The detectors sort internally by loan and month; the result must not."""
    shuffled = toy_panel.sample(frac=1.0, random_state=config.RANDOM_SEED)
    matrix, _ = signals.sequence_detectors(shuffled)

    assert list(matrix.index) == list(shuffled.index)
    zombie_late = (shuffled["loan_id"] == "zombie") & (shuffled["loan_age_months"] == 5)
    assert bool(matrix.loc[zombie_late, "seq__post_absorbing_activity"].iloc[0])


# --------------------------------------------------------------------------
# Score combination
# --------------------------------------------------------------------------
def test_rule_score_ranks_one_high_above_many_lows():
    """
    A noisy-OR, not a sum.

    Under a plain count of violations, three ``missing_document_status`` flags
    would outrank one balance that exceeds origination -- exactly backwards for
    a reviewer queue.
    """
    matrix = pd.DataFrame(
        {"rule__a": [True, False], "rule__b": [False, True], "rule__c": [False, True], "rule__d": [False, True]}
    )
    severities = {"rule__a": "high", "rule__b": "low", "rule__c": "low", "rule__d": "low"}
    score = signals.rule_score(matrix, severities)
    assert score.iloc[0] > score.iloc[1]


def test_rule_score_is_bounded_and_monotone():
    matrix = pd.DataFrame({"rule__a": [False, True, True], "rule__b": [False, False, True]})
    severities = {"rule__a": "high", "rule__b": "medium"}
    score = signals.rule_score(matrix, severities)
    assert score.iloc[0] == 0.0
    assert score.is_monotonic_increasing
    assert (score <= 1.0).all()


def test_hybrid_never_falls_below_the_rule_score():
    """
    Rules set a floor the model cannot argue down.

    A weighted average would let a confident model talk away a hard violation;
    this test is what stops someone changing the combination to one.
    """
    rule = np.array([0.0, 0.9, 0.9, 0.15])
    ml = np.array([0.0, 0.0, 0.99, 0.5])
    hybrid = models.hybrid_score(rule, ml)
    assert (hybrid >= rule - 1e-12).all()
    assert (hybrid >= ml - 1e-12).all()
    assert (hybrid <= 1.0).all()


# --------------------------------------------------------------------------
# Leakage and ablation integrity
# --------------------------------------------------------------------------
def test_exception_labels_never_enter_the_feature_matrix(toy_panel):
    matrix, severities, _ = signals.build_signal_matrix(toy_panel)
    built = fm.build_features(toy_panel, matrix, signals.rule_score(matrix, severities))

    assert fm.EXCEPTION_FLAG not in built.model_columns
    assert fm.EXCEPTION_TYPE not in built.model_columns
    assert not set(built.model_columns) & set(config.FORBIDDEN_FEATURES)


def test_leak_assertion_raises_on_a_target_column():
    with pytest.raises(ValueError, match="Leaky columns"):
        fm.assert_no_leaky_features(["current_balance", "exception_required"])


def test_record_state_ablation_carries_no_sequence_information(toy_panel):
    """
    The strict ablation must remove sequence information, not just its prefix.

    ``rule_score`` is a noisy-OR over *every* signal including the sequence
    detectors, so leaving it in would smuggle their evidence straight back into
    a model the report describes as sequence-blind.
    """
    matrix, severities, _ = signals.build_signal_matrix(toy_panel)
    built = fm.build_features(toy_panel, matrix, signals.rule_score(matrix, severities))

    columns = built.record_state_columns
    assert not any(c.startswith("seq__") for c in columns)
    assert "rule_score" not in columns
    assert not set(columns) & set(built.sequence_context_columns)
    # The row-level aggregate is the permitted substitute.
    assert "rule_score_row_level" in columns


def test_no_sequence_flag_ablation_keeps_context_but_drops_detectors(toy_panel):
    matrix, severities, _ = signals.build_signal_matrix(toy_panel)
    built = fm.build_features(toy_panel, matrix, signals.rule_score(matrix, severities))

    columns = built.no_sequence_flag_columns
    assert not any(c.startswith("seq__") for c in columns)
    assert "rule_score" not in columns
    assert set(built.sequence_context_columns) <= set(columns)


def test_isolation_forest_is_not_shown_the_rule_indicators(toy_panel):
    """Otherwise it rediscovers the rules and reports them back as a discovery."""
    matrix, severities, _ = signals.build_signal_matrix(toy_panel)
    built = fm.build_features(toy_panel, matrix, signals.rule_score(matrix, severities))

    unsupervised = set(built.unsupervised_columns)
    assert not any(c.startswith(("rule__", "date__", "seq__")) for c in unsupervised)
    assert "rule_score" not in unsupervised


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
def test_zero_spread_column_does_not_produce_an_infinite_z():
    """
    A column where nearly every row shares one value has a MAD of exactly zero.

    Dividing by it reported deviations of 71 million MAD in the reviewer queue
    -- a formatting accident presented to a human as a finding.
    """
    frame = pd.DataFrame({"flat": [1.0] * 99 + [5.0], "spread": np.linspace(0, 10, 100)})
    deviation = explain.RobustDeviation(frame, ["flat", "spread"])
    z = deviation.z_scores(frame)
    assert np.isfinite(z["flat"]).all() or z["flat"].isna().all()
    assert z.to_numpy()[np.isfinite(z.to_numpy())].max() < 1e6


def test_layer_classification_reads_aggregates_as_deterministic():
    """``rule_score`` is a rule aggregate, not learned record state."""
    context = {"transition_rarity", "months_after_absorbing"}
    assert explain.classify_layer("rule_score", context) == "rule"
    assert explain.classify_layer("rule__balance_exceeds_original", context) == "rule"
    assert explain.classify_layer("seq__post_absorbing_activity", context) == "sequence detector"
    assert explain.classify_layer("transition_rarity", context) == "sequence context"
    assert explain.classify_layer("current_balance", context) == "record state"


# --------------------------------------------------------------------------
# Curation
# --------------------------------------------------------------------------
def _fake_queue_inputs(n: int = 200):
    rng = np.random.default_rng(config.RANDOM_SEED)
    frame = pd.DataFrame(
        {
            "loan_id": [f"L{i:05d}" for i in range(n)],
            "reporting_month": pd.Timestamp("2023-01-01"),
            "current_status": "Current",
            "current_balance": rng.uniform(50_000, 400_000, n),
        }
    )
    scores = pd.DataFrame(
        {
            "hybrid_score": rng.uniform(0, 1, n),
            "rule_score": rng.uniform(0, 1, n),
            "ml_score": rng.uniform(0, 1, n),
            "exception_probability": rng.uniform(0, 1, n),
        }
    )
    triggered = pd.Series(["balance_exceeds_original" if i % 3 else "" for i in range(n)])
    drivers = pd.Series(["current balance=1.00"] * n)
    types = pd.Series(rng.choice(["No exception", "Zombie Loan", "Time Travel"], n))
    return frame, scores, triggered, drivers, types


def test_queue_meets_the_task_floor_and_carries_every_reviewer_column():
    frame, scores, triggered, drivers, types = _fake_queue_inputs()
    queue = curation.build_review_queue(frame, scores, triggered, drivers, types, n_examples=25)

    assert len(queue) >= 20
    for column in ("loan_id", "hybrid_score", "triggered_rules", "top_drivers", "suggested_action"):
        assert column in queue.columns
    assert queue["suggested_action"].str.len().gt(0).all()


def test_queue_reserves_slots_for_records_no_rule_supports():
    """
    Those are the only rows in the queue that can teach the rule set something.

    Without a reservation they are crowded out by whichever detector is most
    confident, and the queue becomes a demonstration of one rule working.
    """
    frame, scores, triggered, drivers, types = _fake_queue_inputs()
    queue = curation.build_review_queue(
        frame, scores, triggered, drivers, types, n_examples=25, n_unsupported=5
    )
    assert (queue["triggered_rules"].str.len() == 0).sum() >= 5


def test_clean_class_label_survives_a_csv_round_trip(tmp_path):
    """
    The clean class must not be the literal string ``None``.

    pandas reads that back from CSV as NaN, so every clean record in the
    delivered ``anomaly_examples.csv`` would silently lose its label.
    """
    path = tmp_path / "queue.csv"
    pd.DataFrame({"predicted_exception_type": [fm.NO_EXCEPTION, "Zombie Loan"]}).to_csv(path, index=False)
    reloaded = pd.read_csv(path)

    assert reloaded["predicted_exception_type"].isna().sum() == 0
    assert fm.NO_EXCEPTION in set(reloaded["predicted_exception_type"])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_flag_top_k_selects_exactly_k_highest():
    score = np.array([0.1, 0.9, 0.5, 0.7])
    flagged = evaluation.flag_top_k(score, 2)
    assert flagged.sum() == 2
    assert flagged[1] and flagged[3]


def test_precision_at_k_matches_a_hand_count():
    y_true = np.array([1, 0, 1, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    result = evaluation.precision_at_k(y_true, score, 3)
    assert result["precision@3"] == pytest.approx(2 / 3)
    assert result["recall@3"] == pytest.approx(2 / 3)
