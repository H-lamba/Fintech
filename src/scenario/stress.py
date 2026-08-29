"""
Applying a macro scenario to the feature matrix.

Three transmission channels, in descending order of how much they assume.

**1. House prices to loan-to-value (mechanical).**
LTV is debt over value. If the index says values fall 28%, current LTV rises by
``100 / hpi_index``. No elasticity, no fitted relationship -- it is arithmetic,
and it is the strongest link in the chain.

**2. Market rate to refinance incentive (mechanical).**
``rate_spread`` is the loan's note rate less the prevailing market rate, and the
scenario states the market rate directly. A borrower paying 7% in a 4.2% market
has an incentive the model already knows how to read.

**3. Labour market to credit quality (calibrated, not assumed).**
The file gives an unemployment path and a ``default_multiplier``, but no
elasticity connecting them to credit scores. Inventing one -- "assume 40 points
of FICO per point of unemployment" -- would make the projection a restatement
of that invented number, and the scenario file would no longer be the source of
truth it is meant to be.

So the shift is **solved for**: find the portfolio-wide credit-score shift that
makes the model reproduce the scenario's own stated default multiplier. The
file remains authoritative, the model supplies the transmission, and the answer
is reportable in a unit a credit officer recognises -- *"a 2.6x default
multiplier is equivalent to the whole book losing N points of FICO."*

Bounds
------
Every stressed feature is clipped to ``config.STRESS_BOUNDS`` and every banded
column is **recomputed** from the shifted value underneath it. Shifting ``ltv``
while leaving ``ltv_band`` at its original level hands the model a record that
contradicts itself, and the model will happily score it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .. import config
from .macro import ScenarioMonth

# The continuous feature each banded column is derived from.
BAND_SOURCES = {
    "credit_score_band": ("credit_score", config.CREDIT_SCORE_EDGES, config.CREDIT_SCORE_BANDS),
    "ltv_band": ("ltv", config.LTV_EDGES, config.LTV_BANDS),
    "dti_band": ("dti", config.DTI_EDGES, config.DTI_BANDS),
}


@dataclass
class StressResult:
    """A stressed feature frame plus what was actually done to it."""

    data: pd.DataFrame
    applied: dict = field(default_factory=dict)
    clipped: dict = field(default_factory=dict)


def recompute_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild every banded column from its continuous source.

    Called after any stress that moves a continuous feature. Without it the
    band columns are stale and the model sees a record whose score says 640 and
    whose band says ``740-799``.
    """
    out = frame.copy()
    for band_column, (source, edges, labels) in BAND_SOURCES.items():
        if band_column in out.columns and source in out.columns:
            out[band_column] = pd.cut(
                out[source], bins=list(edges), labels=list(labels), right=False
            ).astype(str)
    return out


def _clip(series: pd.Series, feature: str) -> tuple[pd.Series, int]:
    """Clip to the plausible range, reporting how many records were bound."""
    bounds = config.STRESS_BOUNDS.get(feature)
    if bounds is None:
        return series, 0
    low, high = bounds
    n_clipped = int(((series < low) | (series > high)).sum())
    return series.clip(low, high), n_clipped


def apply_market_channels(
    frame: pd.DataFrame, month: ScenarioMonth, anchor: ScenarioMonth
) -> StressResult:
    """
    The two mechanical channels: house prices to LTV, market rate to spread.

    ``anchor`` is the scenario's first projection month, so the HPI shift is
    measured relative to the scenario's own starting point rather than to an
    arbitrary index level.
    """
    out = frame.copy()
    applied: dict = {}
    clipped: dict = {}

    # --- house prices -> LTV ----------------------------------------------
    if "ltv" in out.columns and month.hpi_index > 0:
        hpi_factor = anchor.hpi_index / month.hpi_index
        out["ltv"] = out["ltv"].astype("float64") * hpi_factor
        out["ltv"], clipped["ltv"] = _clip(out["ltv"], "ltv")
        applied["hpi_factor_on_ltv"] = round(hpi_factor, 4)

    # --- market rate -> refinance incentive --------------------------------
    if "rate_spread" in out.columns and "interest_rate" in out.columns:
        out["market_rate_median"] = month.mortgage_rate
        out["rate_spread"] = out["interest_rate"].astype("float64") - month.mortgage_rate
        applied["market_rate"] = month.mortgage_rate

    return StressResult(data=out, applied=applied, clipped=clipped)


def apply_credit_shift(frame: pd.DataFrame, shift: float) -> StressResult:
    """Move every credit score by ``shift`` points, then rebuild the bands."""
    out = frame.copy()
    applied = {"credit_score_shift": round(float(shift), 2)}
    clipped: dict = {}

    if "credit_score" in out.columns and shift != 0.0:
        out["credit_score"] = out["credit_score"].astype("float64") + shift
        out["credit_score"], clipped["credit_score"] = _clip(out["credit_score"], "credit_score")

    return StressResult(data=out, applied=applied, clipped=clipped)


def apply_scenario(
    frame: pd.DataFrame,
    month: ScenarioMonth,
    anchor: ScenarioMonth,
    credit_shift: float = 0.0,
) -> StressResult:
    """All channels, bounds-checked, with the banded columns rebuilt at the end."""
    market = apply_market_channels(frame, month, anchor)
    credit = apply_credit_shift(market.data, credit_shift)

    stressed = recompute_bands(credit.data)

    applied = {**market.applied, **credit.applied}
    clipped = {**market.clipped, **credit.clipped}
    return StressResult(data=stressed, applied=applied, clipped=clipped)


# --------------------------------------------------------------------------
# Calibrating the credit channel
# --------------------------------------------------------------------------
def calibrate_credit_shift(
    score_fn,
    frame: pd.DataFrame,
    month: ScenarioMonth,
    anchor: ScenarioMonth,
    baseline_rate: float,
    target_multiplier: float,
    search_range: tuple[float, float] = (-250.0, 100.0),
    tolerance: float = 1e-4,
) -> tuple[float, dict]:
    """
    Solve for the credit-score shift that reproduces a stated multiplier.

    Parameters
    ----------
    score_fn:
        ``frame -> mean predicted probability``. The Phase 3 model, wrapped.
    baseline_rate:
        The portfolio's mean predicted rate under the baseline scenario at this
        horizon. The target is ``baseline_rate * target_multiplier``.

    Returns ``(shift, diagnostics)``. The shift is in FICO points and is
    negative for a deterioration.

    A root-find rather than a grid search because the mapping is monotone --
    lowering credit scores cannot lower the modelled default rate -- and Brent's
    method converges in a handful of model evaluations where a grid would need
    hundreds. Where the target lies outside what any shift in ``search_range``
    can reach, the closest attainable endpoint is returned together with the
    residual, so an unreachable scenario is visible as a number rather than as
    a silent clamp.
    """
    target_rate = baseline_rate * target_multiplier

    def residual(shift: float) -> float:
        stressed = apply_scenario(frame, month, anchor, credit_shift=shift)
        return score_fn(stressed.data) - target_rate

    low, high = search_range
    residual_low, residual_high = residual(low), residual(high)

    diagnostics = {
        "target_multiplier": target_multiplier,
        "baseline_rate": baseline_rate,
        "target_rate": target_rate,
        "attainable_low": target_rate + residual_low,
        "attainable_high": target_rate + residual_high,
    }

    if residual_low * residual_high > 0:
        # The target is outside the reachable range at either end. This is a
        # finding, not a failure: it means the credit channel *saturates* --
        # even moving the whole book to the floor of the score range does not
        # produce the multiplier the scenario states. The attainable ceiling is
        # returned so the shortfall can be reported as a number.
        shift = low if abs(residual_low) < abs(residual_high) else high
        attained = target_rate + (residual_low if shift == low else residual_high)
        diagnostics["converged"] = False
        diagnostics["residual"] = residual_low if shift == low else residual_high
        diagnostics["attained_rate"] = attained
        diagnostics["attainable_multiplier"] = (
            attained / baseline_rate if baseline_rate else float("nan")
        )
        return shift, diagnostics

    shift = float(brentq(residual, low, high, xtol=tolerance, rtol=1e-8, maxiter=60))
    diagnostics["converged"] = True
    diagnostics["residual"] = residual(shift)
    diagnostics["attained_rate"] = target_rate + diagnostics["residual"]
    diagnostics["attainable_multiplier"] = target_multiplier
    return shift, diagnostics
