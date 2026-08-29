"""
Regression tests for the Task 3 censoring and competing-risk logic.

Censoring bugs are the dangerous kind: they do not raise, they do not look
wrong, and they move every curve in the same direction. Each test below pins
one decision from ``src/survival/dataset.py`` that would otherwise be a
convention someone could quietly change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.survival import baselines, dataset, models

CUTOFF = pd.Timestamp("2023-12-01")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _loan_rows(
    loan_id: str,
    origination: str,
    n_months: int,
    final_status: str = "Current",
    start_age: int = 0,
    extra_rows: list[dict] | None = None,
) -> list[dict]:
    """A loan's monthly rows, optionally ending in an absorbing state."""
    start = pd.Timestamp(origination)
    rows = []
    for offset in range(n_months):
        age = start_age + offset
        status = "Current"
        if offset == n_months - 1:
            status = final_status
        rows.append(
            {
                "loan_id": loan_id,
                "reporting_month": start + pd.DateOffset(months=age),
                "origination_month": start,
                "loan_age_months": age,
                "current_status": status,
                "days_past_due": 0,
                "current_balance": 250_000.0,
                "original_balance": 250_000.0,
                "interest_rate": 5.0,
                "credit_score_band": "700-739",
                "vintage_year": start.year,
            }
        )
    rows.extend(extra_rows or [])
    return rows


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """
    Six loans, one per case the transformation has to get right.

    ``active``      still performing at the cutoff -> administratively censored
    ``defaulted``   defaults at age 9
    ``prepaid``     prepays at age 5
    ``dropped``     stops reporting well before the cutoff -> lost to follow-up
    ``zombie``      defaults at age 6, then emits two more "Current" rows
    ``late_entry``  first observed at age 12 -> left-truncated
    """
    rows: list[dict] = []
    rows += _loan_rows("active", "2022-01-01", 24)
    rows += _loan_rows("defaulted", "2022-01-01", 10, final_status="Default")
    rows += _loan_rows("prepaid", "2022-01-01", 6, final_status="Prepaid")
    rows += _loan_rows("dropped", "2021-01-01", 8)

    zombie_extra = [
        {
            "loan_id": "zombie",
            "reporting_month": pd.Timestamp("2022-01-01") + pd.DateOffset(months=age),
            "origination_month": pd.Timestamp("2022-01-01"),
            "loan_age_months": age,
            "current_status": "Current",
            "days_past_due": 0,
            "current_balance": 0.0,
            "original_balance": 250_000.0,
            "interest_rate": 5.0,
            "credit_score_band": "700-739",
            "vintage_year": 2022,
        }
        for age in (7, 8)
    ]
    rows += _loan_rows("zombie", "2022-01-01", 7, final_status="Default", extra_rows=zombie_extra)
    rows += _loan_rows("late_entry", "2022-01-01", 12, start_age=12)

    return pd.DataFrame(rows)


@pytest.fixture
def toy_frame(toy_panel) -> dataset.SurvivalFrame:
    return dataset.build_survival_frame(toy_panel, cutoff=CUTOFF)


def _record(frame: dataset.SurvivalFrame, loan_id: str) -> pd.Series:
    return frame.data.set_index("loan_id").loc[loan_id]


# --------------------------------------------------------------------------
# Censoring
# --------------------------------------------------------------------------
def test_active_loans_are_censored_not_dropped(toy_frame):
    """
    The single most consequential decision in the whole task.

    A loan still performing at the cutoff is ``event_code == 0`` at its last
    observed age -- it is *not* a "no default" outcome, and it is *not*
    excluded. Dropping it would leave only resolved loans in the sample and
    roughly double every incidence estimate.
    """
    record = _record(toy_frame, "active")
    assert record["event_code"] == config.EVENT_CENSORED
    assert record["exit_month"] == 23
    assert record["censoring_type"] == "administrative"
    assert "active" in set(toy_frame.data["loan_id"])


def test_administrative_and_lost_to_followup_are_distinguished(toy_frame):
    """Both are right-censoring; only one is non-informative by construction."""
    assert _record(toy_frame, "active")["censoring_type"] == "administrative"
    assert _record(toy_frame, "dropped")["censoring_type"] == "lost_to_followup"
    # ``active`` and ``late_entry`` both run to the cutoff; only ``dropped``
    # stops reporting early.
    assert toy_frame.censoring["censored_administrative"] == 2
    assert toy_frame.censoring["censored_lost_to_followup"] == 1


def test_events_are_recorded_at_the_event_month(toy_frame):
    assert _record(toy_frame, "defaulted")["event_code"] == config.EVENT_DEFAULT
    assert _record(toy_frame, "defaulted")["exit_month"] == 9
    assert _record(toy_frame, "prepaid")["event_code"] == config.EVENT_PREPAID
    assert _record(toy_frame, "prepaid")["exit_month"] == 5


def test_zombie_rows_do_not_resurrect_a_resolved_loan(toy_frame):
    """
    A ``Current`` row after a ``Default`` row is a data defect, not a recovery.

    ``groupby().last()`` -- the obvious implementation -- would read this loan
    as still active and move it from the event count into the censored count.
    """
    record = _record(toy_frame, "zombie")
    assert record["event_code"] == config.EVENT_DEFAULT
    assert record["exit_month"] == 6
    assert toy_frame.censoring["zombie_rows_dropped"] == 2
    assert toy_frame.censoring["zombie_loans_affected"] == 1


# --------------------------------------------------------------------------
# Left truncation
# --------------------------------------------------------------------------
def test_delayed_entry_is_recorded(toy_frame):
    record = _record(toy_frame, "late_entry")
    assert record["entry_month"] == 12
    assert record["exposure_months"] == record["exit_month"] - 12
    assert toy_frame.censoring["left_truncated_loans"] == 1


def test_risk_set_excludes_loans_not_yet_observed(toy_frame):
    """
    The point of the ``entry`` argument: a loan first seen at month 12 is not
    in the risk set at month 6, because it was not being watched then.
    """
    table = baselines.risk_table(toy_frame)
    at_month_6 = table.loc[table["month"] == 6, "at_risk"].iloc[0]
    at_month_13 = table.loc[table["month"] == 13, "at_risk"].iloc[0]

    ids_at_6 = toy_frame.data[
        (toy_frame.data["entry_month"] < 6) & (toy_frame.data["exit_month"] >= 6)
    ]["loan_id"]
    assert "late_entry" not in set(ids_at_6)
    assert at_month_6 == len(ids_at_6)
    assert at_month_13 >= 1


def test_disabling_left_truncation_changes_the_risk_set(toy_panel):
    """The ablation switch has to actually do something, or it proves nothing."""
    truncated = dataset.build_survival_frame(toy_panel, cutoff=CUTOFF, left_truncation=True)
    naive = dataset.build_survival_frame(toy_panel, cutoff=CUTOFF, left_truncation=False)

    assert (naive.data["entry_month"] == 0).all()
    early = 3
    n_truncated = baselines.risk_table(truncated).query("month == @early")["at_risk"].iloc[0]
    n_naive = baselines.risk_table(naive).query("month == @early")["at_risk"].iloc[0]
    assert n_naive == n_truncated + 1


# --------------------------------------------------------------------------
# Competing risks
# --------------------------------------------------------------------------
def test_cif_and_survival_sum_to_one(toy_frame):
    """
    ``CIF_default(t) + CIF_prepaid(t) + S(t) = 1`` at every t.

    This identity is what separates a competing-risk estimate from two
    independent survival curves, which can happily sum past 1.
    """
    default = baselines.aalen_johansen_cif(toy_frame, config.EVENT_DEFAULT)
    prepaid = baselines.aalen_johansen_cif(toy_frame, config.EVENT_PREPAID)
    total = default["cif"] + prepaid["cif"] + default["overall_survival"]
    assert np.allclose(total, 1.0)


def test_naive_km_overstates_incidence(toy_frame):
    """
    ``1 - cause-specific KM`` is never below the Aalen-Johansen CIF.

    That is the direction of the bias competing risks introduce, and the reason
    both numbers are reported side by side rather than only the flattering one.
    """
    for cause in (config.EVENT_DEFAULT, config.EVENT_PREPAID):
        curve = baselines.aalen_johansen_cif(toy_frame, cause)
        assert (curve["naive_1_minus_km"] >= curve["cif"] - 1e-12).all()


def test_aalen_johansen_matches_lifelines_on_untied_data():
    """
    Cross-check the hand-rolled estimator against ``lifelines``.

    The estimator here is implemented directly because monthly durations are
    heavily tied and ``AalenJohansenFitter`` resolves ties by random jitter.
    On untied data the two must agree exactly -- which is what makes the
    substitution safe rather than merely convenient.
    """
    from lifelines import AalenJohansenFitter

    rng = np.random.default_rng(config.RANDOM_SEED)
    n = 300
    durations = np.sort(rng.uniform(1, 60, n))
    events = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])

    frame = dataset.SurvivalFrame(
        data=pd.DataFrame(
            {
                "loan_id": [f"L{i}" for i in range(n)],
                "entry_month": 0.0,
                "exit_month": durations,
                "event_code": events,
            }
        ),
        censoring={},
    )
    ours = baselines.aalen_johansen_cif(frame, config.EVENT_DEFAULT, timeline=durations)

    fitter = AalenJohansenFitter(calculate_variance=False)
    fitter.fit(durations, events, event_of_interest=config.EVENT_DEFAULT)
    theirs = fitter.cumulative_density_.reindex(durations, method="ffill").to_numpy().ravel()

    assert np.allclose(ours["cif"].to_numpy(), theirs, atol=1e-8)


# --------------------------------------------------------------------------
# Baseline model
# --------------------------------------------------------------------------
def test_constant_hazard_is_occurrence_over_exposure(toy_frame):
    model = baselines.ConstantHazardModel.fit(toy_frame)
    expected_exposure = float(
        (toy_frame.data["exit_month"] - toy_frame.data["entry_month"]).sum()
    )
    assert model.exposure_months == pytest.approx(expected_exposure)
    assert model.hazards[config.EVENT_DEFAULT] == pytest.approx(2 / expected_exposure)


def test_censored_loans_contribute_exposure_to_the_baseline(toy_frame):
    """
    Dropping censored loans -- the classic mistake -- inflates the hazard.

    The test states the direction of the error, so a future change that quietly
    filters them out fails here rather than in a report nobody re-derives.
    """
    full = baselines.ConstantHazardModel.fit(toy_frame)
    events_only = baselines.ConstantHazardModel.fit(
        toy_frame.subset(toy_frame.data["event_code"] != config.EVENT_CENSORED)
    )
    assert events_only.hazards[config.EVENT_DEFAULT] > full.hazards[config.EVENT_DEFAULT]


def test_constant_hazard_cif_respects_the_competing_risk(toy_frame):
    """The two CIFs must not sum past 1, however long the horizon."""
    model = baselines.ConstantHazardModel.fit(toy_frame)
    times = np.arange(1, 600)
    total = model.cumulative_incidence(config.EVENT_DEFAULT, times) + model.cumulative_incidence(
        config.EVENT_PREPAID, times
    )
    assert (total <= 1.0 + 1e-9).all()
    assert total[-1] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Curves and splitting
# --------------------------------------------------------------------------
def test_effective_timeline_stops_where_the_risk_set_thins(toy_frame):
    """A curve must end where the evidence ends, not where the axis does."""
    short = baselines.effective_timeline(toy_frame, min_at_risk=5)
    long = baselines.effective_timeline(toy_frame, min_at_risk=1)
    assert short[-1] < long[-1]


def test_vintage_split_is_disjoint_and_forward_in_time(toy_panel):
    static = pd.DataFrame(
        {
            "loan_id": toy_panel["loan_id"].unique(),
            "origination_month": [
                toy_panel.loc[toy_panel["loan_id"] == lid, "origination_month"].iloc[0]
                for lid in toy_panel["loan_id"].unique()
            ],
        }
    )
    frame = dataset.build_survival_frame(toy_panel, static, cutoff=CUTOFF)
    train, test = dataset.vintage_split(frame, "2021-06-01")

    assert not set(train.data["loan_id"]) & set(test.data["loan_id"])
    if len(train) and len(test):
        assert train.data["origination_month"].max() < test.data["origination_month"].min()


# --------------------------------------------------------------------------
# Cox / competing-risk assembly
# --------------------------------------------------------------------------
def test_cox_cif_reduces_to_the_marginal_estimate_at_the_mean_profile():
    """
    With covariates that carry no signal, the Cox-assembled CIF must land on
    the non-parametric Aalen-Johansen curve. If the assembly arithmetic is
    wrong -- a missing ``S(s-)`` factor, hazards not coupled across causes --
    the two diverge.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = 1200
    durations = rng.integers(1, 48, n).astype(float)
    events = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])

    data = pd.DataFrame(
        {
            "loan_id": [f"L{i}" for i in range(n)],
            "entry_month": 0.0,
            "exit_month": durations,
            "event_code": events,
            "credit_score": rng.normal(700, 40, n),
            "ltv": rng.normal(78, 8, n),
        }
    )
    frame = dataset.SurvivalFrame(data=data, censoring={})

    spec = models.DesignSpec.fit(frame, numeric=["credit_score", "ltv"], categorical=[])
    fitted = {
        cause: models.fit_cause_specific_cox(frame, cause, spec)
        for cause in (config.EVENT_DEFAULT, config.EVENT_PREPAID)
    }

    times = np.arange(1, 37)
    cox_cif = models.mean_profile_cif(fitted, frame, times)[config.EVENT_DEFAULT]
    marginal = baselines.aalen_johansen_cif(frame, config.EVENT_DEFAULT, timeline=times)["cif"]

    assert np.max(np.abs(cox_cif - marginal.to_numpy())) < 0.02
