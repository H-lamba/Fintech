"""
Regression tests for Phase 9: inference, submission shape, and the no-refit
guarantee.

The failures these guard against are all silent. A submission with the columns
in a different order is not obviously wrong on inspection and is completely
wrong to a scorer that joins on position. A cell holding the string "None"
passes every in-memory null check and comes back as NaN the moment anyone opens
the file. A preprocessing object refitted on the test set produces slightly
better numbers and invalidates the whole entry.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src import config
from src.submission import build as builder
from src.submission import inference


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["A", "A", "B"],
            "reporting_month": ["2024-01-01", "2024-02-01", "2024-01-01"],
            "prob_next_3m_delinquency": [np.nan] * 3,
            "next_state": [np.nan] * 3,
            "exception_required": [np.nan] * 3,
            "exception_type": [np.nan] * 3,
            "anomaly_score": [np.nan] * 3,
            "top_drivers": [np.nan] * 3,
            "action": [np.nan] * 3,
            "confidence": [np.nan] * 3,
        }
    )


@pytest.fixture
def scored() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["B", "A", "A"],  # deliberately out of template order
            "reporting_month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
            "prob_next_3m_delinquency": [0.10, 0.20, 0.30],
            "next_state": ["Current", "30-DPD", "Current"],
            "confidence": [0.9, 0.8, 0.7],
            "anomaly_score": [0.1, 0.95, 0.2],
            "exception_probability": [0.01, 0.90, 0.02],
            "exception_threshold": [0.5, 0.5, 0.5],
            "predicted_exception_type": ["No exception", "Zombie Loan", "No exception"],
            "triggered_rules": ["", "post_absorbing_activity; illegal_status_transition", ""],
        }
    )


# --------------------------------------------------------------------------
# Submission shape
# --------------------------------------------------------------------------
def test_columns_match_the_template_exactly_and_in_order(scored, template):
    submission = builder.build_submission(scored, template)
    assert list(submission.columns) == list(template.columns)


def test_rows_are_aligned_by_key_not_by_position(scored, template):
    """
    A submission whose rows are in a different order than the template's is not
    obviously wrong on inspection and is completely wrong to a scorer that
    joins on index.
    """
    submission = builder.build_submission(scored, template)

    assert list(submission["loan_id"]) == list(template["loan_id"])
    row = submission[
        (submission["loan_id"] == "A") & (submission["reporting_month"] == "2024-02-01")
    ].iloc[0]
    assert row["next_state"] == "30-DPD"
    assert row["prob_next_3m_delinquency"] == pytest.approx(0.20)


def test_no_cell_reads_back_as_nan_after_a_csv_round_trip(scored, template):
    """
    The literal string "None" passes an in-memory null check and comes back as
    NaN the moment anyone reads the file -- which is the only state the
    organiser ever sees. The same bug bit the Phase 5 reviewer queue.
    """
    submission = builder.build_submission(scored, template)
    round_tripped = builder._round_trip(submission)

    assert int(round_tripped.isna().sum().sum()) == 0
    assert "None" not in set(round_tripped["exception_type"])
    assert builder.NO_EXCEPTION in set(round_tripped["exception_type"])


def test_a_cleared_row_gets_a_no_action_message(scored, template):
    """
    The rule-based action text assumes something fired. A row the model clears
    needs "no action required", not an instruction to write the rule that would
    have caught it.
    """
    submission = builder.build_submission(scored, template)
    cleared = submission[submission["exception_required"] == 0]

    assert len(cleared) == 2
    assert (cleared["action"] == builder.NO_ACTION).all()
    assert (cleared["top_drivers"] == builder.NO_DRIVERS).all()


def test_the_flagged_row_carries_drivers_and_a_specific_action(scored, template):
    submission = builder.build_submission(scored, template)
    flagged = submission[submission["exception_required"] == 1]

    assert len(flagged) == 1
    assert flagged["exception_type"].iloc[0] == "Zombie Loan"
    assert "post_absorbing_activity" in flagged["top_drivers"].iloc[0]
    assert "=" not in flagged["top_drivers"].iloc[0]  # names only, no values
    assert flagged["action"].iloc[0] != builder.NO_ACTION


def test_the_tuned_threshold_is_used_not_a_hardcoded_half(scored, template):
    """
    The exception head's probabilities stay compressed below 0.52 because early
    stopping halts within ten rounds; a fixed 0.5 cut flagged one row in 78,409
    against a 2.6% base rate. The threshold travels with the predictions.
    """
    low = scored.copy()
    low["exception_probability"] = [0.01, 0.31, 0.02]
    low["exception_threshold"] = 0.30

    submission = builder.build_submission(low, template)
    assert submission["exception_required"].sum() == 1


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_validation_catches_a_reordered_column(scored, template):
    submission = builder.build_submission(scored, template)
    scrambled = submission[list(reversed(submission.columns))]
    report = builder.validate(scrambled, template)

    assert not report.passed
    assert any("in order" in c["check"] for c in report.failures())


def test_validation_catches_a_probability_outside_the_unit_interval(scored, template):
    submission = builder.build_submission(scored, template)
    submission.loc[0, "confidence"] = 1.4
    report = builder.validate(submission, template)

    assert any("within [0, 1]" in c["check"] and not c["passed"] for c in report.checks)


def test_a_structurally_invalid_submission_is_not_written(scored, template, tmp_path):
    """
    A file with the wrong columns is not a partially-good submission; writing it
    anyway just moves the discovery later.
    """
    submission = builder.build_submission(scored, template).drop(columns=["confidence"])
    report = builder.validate(submission, template)

    with pytest.raises(ValueError, match="Refusing to write"):
        builder.write_submission(submission, report, tmp_path / "submission.csv")
    assert not (tmp_path / "submission.csv").exists()


# --------------------------------------------------------------------------
# The no-refit guarantee
# --------------------------------------------------------------------------
def _fitted_state(model) -> dict:
    """Every fitted statistic a scaler or imputer holds, as plain arrays."""
    state = {}
    pipeline = getattr(model.estimator, "named_steps", None)
    if not pipeline:
        return state
    for name, step in pipeline.items():
        for attribute in ("statistics_", "mean_", "scale_", "var_", "categories_"):
            value = getattr(step, attribute, None)
            if value is not None:
                state[f"{name}.{attribute}"] = np.asarray(value, dtype=object)
    return state


def test_scoring_does_not_refit_the_preprocessing(tmp_path):
    """
    Inference must call ``transform``, never ``fit``.

    Refitting a scaler or imputer on the test set is the classic MLOps leak: it
    produces slightly better numbers and invalidates the entry. This scores a
    frame with a deliberately shifted distribution and asserts every fitted
    statistic is unchanged afterwards.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(config.RANDOM_SEED)
    train = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 2, 200)})
    y = (train["a"] > 0).astype(int)

    pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")),
         ("scale", StandardScaler()),
         ("clf", LogisticRegression(max_iter=500))]
    )
    pipeline.fit(train, y)

    class _Model:
        estimator = pipeline

    before = copy.deepcopy(_fitted_state(_Model()))

    # A wildly different distribution: if anything refits, the state moves.
    shifted = pd.DataFrame({"a": rng.normal(100, 30, 500), "b": rng.normal(-50, 10, 500)})
    pipeline.predict_proba(shifted)

    after = _fitted_state(_Model())
    assert set(before) == set(after) and before
    for key in before:
        assert np.array_equal(
            np.asarray(before[key], dtype=float), np.asarray(after[key], dtype=float)
        ), f"{key} changed during scoring"


def test_category_levels_are_frozen_on_train():
    """
    A level first seen at scoring time must become NaN, not silently re-encode
    every other level.
    """
    from src.models.estimators import CategoryHarmoniser

    train = pd.DataFrame({"servicer": ["A", "B", "A", "C"]})
    harmoniser = CategoryHarmoniser.fit(train, ["servicer"])

    scored = harmoniser.transform(pd.DataFrame({"servicer": ["A", "B", "NEW"]}))
    assert list(scored["servicer"].cat.categories) == ["A", "B", "C"]
    assert pd.isna(scored["servicer"].iloc[2])


# --------------------------------------------------------------------------
# Driver formatting
# --------------------------------------------------------------------------
def test_driver_names_are_stripped_of_values():
    """
    The submission column is specified as feature names. The Phase 5 driver
    string carries values too, which is right for a reviewer queue and wrong
    here.
    """
    raw = "post absorbing activity=1; transition rarity=0.00125; rule score=0.99"
    cleaned = inference._clean_driver_names(raw, limit=3)

    assert "=" not in cleaned
    assert cleaned.startswith("post absorbing activity")
    assert len(cleaned.split(",")) == 3


def test_driver_names_are_capped_and_deduplicated():
    raw = "; ".join(["a=1", "a=2", "b=3", "c=4", "d=5"])
    assert inference._clean_driver_names(raw, limit=2) == "a, b"
