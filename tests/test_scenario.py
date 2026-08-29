"""
Regression tests for the Task 5 scenario and stress pipeline.

The failure modes here are all silent ones: a band left stale under a shifted
score, a calibration that clamps at its search bound and reports the clamp as
an answer, a "stated multiplier" attributed to a file that never stated it.
None of them raise, and each turns a projection into a number that means
something other than what the report says it means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.scenario import project, stress
from src.scenario.macro import MacroScenarios, ScenarioMonth


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def _month(scenario="Adverse-Credit", horizon=24, hpi=80.0, rate=7.0, mult=2.0, prepay=0.8):
    return ScenarioMonth(
        scenario=scenario,
        projection_month=pd.Timestamp("2025-12-01"),
        horizon_month=horizon,
        mortgage_rate=rate,
        unemployment_rate=6.5,
        hpi_index=hpi,
        default_multiplier=mult,
        prepayment_multiplier=prepay,
    )


ANCHOR = _month(horizon=1, hpi=100.0, rate=6.6, mult=1.0, prepay=1.0)


@pytest.fixture
def portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": [f"L{i}" for i in range(5)],
            "credit_score": [590.0, 645.0, 690.0, 730.0, 810.0],
            "credit_score_band": ["<620", "620-659", "660-699", "700-739", "800+"],
            "ltv": [55.0, 70.0, 78.0, 85.0, 95.0],
            "ltv_band": ["<60%", "60-75%", "75-80%", "80-90%", "90-97%"],
            "dti": [25.0, 33.0, 40.0, 45.0, 28.0],
            "dti_band": ["<30%", "30-36%", "36-43%", ">43%", "<30%"],
            "interest_rate": [7.2, 6.8, 6.4, 6.0, 5.6],
            "rate_spread": [0.6, 0.2, -0.2, -0.6, -1.0],
        }
    )


# --------------------------------------------------------------------------
# The macro file is the source of truth
# --------------------------------------------------------------------------
def test_missing_column_stops_the_run(tmp_path):
    """
    A partial scenario file must fail loudly.

    The whole design rests on the file being authoritative; silently
    substituting a default for a missing stress variable would make the
    projection a statement about this code's assumptions instead.
    """
    path = tmp_path / "macro.csv"
    pd.DataFrame({"scenario": ["Baseline"], "projection_month": ["2024-01-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        MacroScenarios.load(path)


def test_baseline_is_identified_by_flat_multipliers_not_by_name(tmp_path):
    """A differently-named reference scenario must still be found."""
    path = tmp_path / "macro.csv"
    rows = []
    for name, mult in (("Steady-State", 1.0), ("Stress", 2.0)):
        for horizon in (1, 2):
            rows.append(
                {
                    "scenario": name, "projection_month": f"2024-0{horizon}-01",
                    "horizon_month": horizon, "mortgage_rate": 6.0,
                    "unemployment_rate": 4.0, "hpi_index": 100.0,
                    "default_multiplier": 1.0 if mult == 1.0 else horizon,
                    "prepayment_multiplier": 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    assert MacroScenarios.load(path).baseline_name == "Steady-State"


def test_no_multiplier_is_invented_for_delinquency():
    """
    The file states default and prepayment multipliers and nothing else.

    Reusing the default multiplier for delinquency would put a number in the
    report and credit it to a source that never gave it.
    """
    month = _month()
    assert month.multiplier_for("default_12m") == pytest.approx(2.0)
    assert month.multiplier_for("prepayment_12m") == pytest.approx(0.8)
    assert month.multiplier_for("delinquency_3m") is None


# --------------------------------------------------------------------------
# Stress transformations
# --------------------------------------------------------------------------
def test_falling_house_prices_raise_ltv_mechanically(portfolio):
    """LTV is debt over value: a 20% fall in the index is a 1.25x rise in LTV."""
    result = stress.apply_market_channels(portfolio, _month(hpi=80.0), ANCHOR)
    assert result.applied["hpi_factor_on_ltv"] == pytest.approx(1.25)
    assert result.data["ltv"].iloc[1] == pytest.approx(70.0 * 1.25)


def test_rate_spread_is_measured_against_the_scenario_market_rate(portfolio):
    result = stress.apply_market_channels(portfolio, _month(rate=4.2), ANCHOR)
    expected = portfolio["interest_rate"] - 4.2
    assert np.allclose(result.data["rate_spread"], expected)


def test_bands_are_rebuilt_from_the_shifted_values(portfolio):
    """
    The banded column must never contradict the value underneath it.

    A record whose score says 545 and whose band says ``700-739`` is one the
    model will happily score, and the number it returns is meaningless.
    """
    # -195 puts every score below 620 *after* the 500-point clip floor, so the
    # assertion tests the band rebuild rather than the clip.
    result = stress.apply_scenario(portfolio, _month(hpi=100.0), ANCHOR, credit_shift=-195.0)
    shifted = result.data

    assert (shifted["credit_score"] < 620).all()
    assert (shifted["credit_score_band"] == "<620").all()

    # And the LTV bands track their own source too.
    recomputed = stress.recompute_bands(shifted)
    assert (recomputed["ltv_band"] == shifted["ltv_band"]).all()


def test_stressed_features_are_clipped_to_plausible_ranges(portfolio):
    """
    A projection that pushes a score to 380 is extrapolation, not stress.

    The clip is reported rather than applied silently, so a scenario that needs
    an impossible input is visible.
    """
    result = stress.apply_credit_shift(portfolio, -400.0)
    low, high = config.STRESS_BOUNDS["credit_score"]
    assert result.data["credit_score"].min() >= low
    assert result.clipped["credit_score"] == len(portfolio)


def test_no_stress_leaves_the_frame_alone(portfolio):
    """The baseline scenario at its own anchor must be a no-op."""
    result = stress.apply_scenario(portfolio, ANCHOR, ANCHOR, credit_shift=0.0)
    assert np.allclose(result.data["ltv"], portfolio["ltv"])
    assert np.allclose(result.data["credit_score"], portfolio["credit_score"])
    assert (result.data["credit_score_band"] == portfolio["credit_score_band"]).all()


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def test_calibration_recovers_a_known_shift(portfolio):
    """
    Given a scorer with a known response, the solver must find its inverse.

    The scorer below doubles the rate for every 100 points of deterioration, so
    a 2x multiplier has an exact answer of -100 points.
    """
    # Scores chosen so a -100 shift stays clear of the 500-point clip floor;
    # otherwise the solver correctly overshoots to compensate for the clipping
    # and the "known answer" is no longer -100.
    unclipped = portfolio.assign(credit_score=[640.0, 680.0, 700.0, 730.0, 810.0])
    reference_mean = unclipped["credit_score"].mean()

    def score(frame: pd.DataFrame) -> float:
        shift = frame["credit_score"].mean() - reference_mean
        return 0.05 * (2.0 ** (-shift / 100.0))

    shift, diagnostics = stress.calibrate_credit_shift(
        score, unclipped, _month(hpi=100.0, mult=2.0), ANCHOR,
        baseline_rate=0.05, target_multiplier=2.0,
    )
    assert diagnostics["converged"]
    assert shift == pytest.approx(-100.0, abs=1.0)


def test_unreachable_multiplier_is_reported_not_clamped_silently(portfolio):
    """
    A stated multiplier the model cannot produce is the most important thing a
    stress framework can surface, and the easiest to hide behind a search bound.
    """
    def score(frame: pd.DataFrame) -> float:
        return 0.05  # completely insensitive to any shift

    search_range = (-250.0, 100.0)
    shift, diagnostics = stress.calibrate_credit_shift(
        score, portfolio, _month(mult=3.0), ANCHOR,
        baseline_rate=0.05, target_multiplier=3.0, search_range=search_range,
    )
    assert diagnostics["converged"] is False
    assert diagnostics["attainable_multiplier"] == pytest.approx(1.0)
    # The returned shift is the closer endpoint, reported so the clamp is visible.
    assert shift in search_range


def test_saturation_summary_names_the_shortfall():
    scenarios = MacroScenarios.load()
    calibration = {
        (scenarios.baseline_name, 12): {"credit_score_shift": 0.0, "converged": True,
                                        "attainable_multiplier": 1.0},
        ("Adverse-Credit", 48): {"credit_score_shift": -250.0, "converged": False,
                                 "attainable_multiplier": 1.88},
    }
    table = project.saturation_summary(calibration, scenarios)

    assert scenarios.baseline_name not in set(table["scenario"])
    row = table[table["horizon_month"] == 48].iloc[0]
    assert row["reached"] is False or row["reached"] == False  # noqa: E712
    assert row["shortfall"] > 0


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def test_baseline_delta_is_zero_for_the_baseline_scenario():
    grouped = pd.DataFrame(
        {
            "scenario": ["Base", "Base", "Stress", "Stress"],
            "horizon_month": [12, 24, 12, 24],
            "default_12m": [0.10, 0.10, 0.15, 0.20],
        }
    )
    out = project._attach_baseline_delta(grouped, ["default_12m"])
    base = out[out["scenario"] == "Base"]["default_12m_vs_baseline_pp"]

    assert np.allclose(base, 0.0)
    stress_rows = out[out["scenario"] == "Stress"].sort_values("horizon_month")
    assert np.allclose(stress_rows["default_12m_vs_baseline_pp"], [5.0, 10.0])


def test_latest_position_takes_one_row_per_loan():
    frame = pd.DataFrame(
        {
            "loan_id": ["A", "A", "A", "B", "B"],
            "reporting_month": pd.to_datetime(
                ["2024-01-01", "2024-03-01", "2024-02-01", "2024-05-01", "2024-04-01"]
            ),
            "value": [1, 3, 2, 5, 4],
        }
    )
    latest = project.latest_position(frame)
    assert len(latest) == 2
    assert set(latest["value"]) == {3, 5}
