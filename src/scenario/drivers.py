"""
Why a scenario moves the numbers.

A scenario's effect on a projected rate decomposes exactly, because the model
is a tree ensemble and LightGBM will return per-feature contributions to the
predicted log-odds. Score the portfolio twice -- baseline and stressed -- and
the change in each feature's mean contribution is that feature's share of the
change in the rate. Nothing is inferred from correlation, and nothing is
asserted from the narrative: if the report says house prices did the work, the
arithmetic says so too.

This matters because the plausible story and the true one come apart. The
Adverse-Credit narrative is about unemployment, but unemployment is not a
feature -- it reaches the model only through the calibrated credit-score shift.
Whether the resulting rate change is actually driven by credit score, by the
LTV that house prices moved, or by the rate spread is an empirical question,
and this module answers it rather than guessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def _contributions(model, frame: pd.DataFrame) -> pd.DataFrame | None:
    """Per-row feature contributions to the predicted log-odds, or None."""
    if getattr(model, "backend", None) != "lightgbm":
        return None

    from ..models import estimators

    X = estimators.prepare_matrix(frame, model.numeric, model.categorical, model.harmoniser)
    raw = np.asarray(model.estimator.booster_.predict(X, pred_contrib=True))
    n_features = len(model.feature_names)
    return pd.DataFrame(raw[:, :n_features], columns=model.feature_names, index=frame.index)


def scenario_drivers(
    model,
    baseline_frame: pd.DataFrame,
    stressed_frame: pd.DataFrame,
    target_label: str,
    scenario: str,
    horizon: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Rank features by how much they moved the projected rate.

    Returns one row per feature: the mean contribution under each state, the
    change, and that change as a share of the total absolute movement. Positive
    ``delta_contribution`` means the feature pushed the rate up.
    """
    baseline_contributions = _contributions(model, baseline_frame)
    stressed_contributions = _contributions(model, stressed_frame)
    if baseline_contributions is None or stressed_contributions is None:
        return pd.DataFrame()

    delta = stressed_contributions.mean() - baseline_contributions.mean()
    total_movement = delta.abs().sum()

    frame = pd.DataFrame(
        {
            "feature": delta.index,
            "baseline_contribution": baseline_contributions.mean().to_numpy(),
            "stressed_contribution": stressed_contributions.mean().to_numpy(),
            "delta_contribution": delta.to_numpy(),
        }
    )
    frame["share_of_movement"] = frame["delta_contribution"].abs() / total_movement if total_movement else np.nan
    frame.insert(0, "scenario", scenario)
    frame.insert(1, "horizon_month", horizon)
    frame.insert(2, "measure", target_label)

    frame = frame.reindex(frame["delta_contribution"].abs().sort_values(ascending=False).index)
    return frame.head(top_n).reset_index(drop=True)


def feature_movement(
    baseline_frame: pd.DataFrame, stressed_frame: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """
    How far each stressed feature actually moved, in its own units.

    The contribution table says which feature mattered; this says what happened
    to it. A reviewer needs both -- "LTV explains 60% of the rise" is only
    actionable alongside "LTV went from 76% to 105%".
    """
    rows = []
    for feature in features:
        if feature not in baseline_frame.columns or feature not in stressed_frame.columns:
            continue
        base = pd.to_numeric(baseline_frame[feature], errors="coerce")
        stressed = pd.to_numeric(stressed_frame[feature], errors="coerce")
        if base.isna().all():
            continue
        rows.append(
            {
                "feature": feature,
                "baseline_mean": float(base.mean()),
                "stressed_mean": float(stressed.mean()),
                "change": float(stressed.mean() - base.mean()),
                "pct_change": float(stressed.mean() / base.mean() - 1.0) if base.mean() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def narrate(drivers: pd.DataFrame, movement: pd.DataFrame, rate_change_pp: float) -> str:
    """
    A written explanation of one scenario's effect, generated from the numbers.

    Written from the attribution rather than from the scenario's name, so the
    sentence cannot drift away from what the model actually did. Every figure
    quoted is read out of the two frames passed in.
    """
    if drivers.empty:
        return "No attribution available for this backend."

    movement_lookup = movement.set_index("feature") if not movement.empty else pd.DataFrame()
    direction = "raises" if rate_change_pp >= 0 else "lowers"

    parts = [
        f"The scenario {direction} the projected rate by "
        f"{abs(rate_change_pp):.2f} percentage points. Attribution over the portfolio:"
    ]
    for _, row in drivers.iterrows():
        feature = row["feature"]
        share = row["share_of_movement"]
        sign = "up" if row["delta_contribution"] > 0 else "down"
        detail = ""
        if feature in movement_lookup.index:
            moved = movement_lookup.loc[feature]
            detail = (
                f" ({moved['baseline_mean']:,.2f} -> {moved['stressed_mean']:,.2f},"
                f" {moved['pct_change']:+.1%})"
            )
        parts.append(
            f"- **{feature.replace('_', ' ')}** pushes it {sign}, "
            f"{share:.0%} of the total movement{detail}."
        )
    return "\n".join(parts)


def collect(
    models: dict,
    portfolio: pd.DataFrame,
    scenarios,
    calibration: dict,
    horizons: tuple[int, ...] | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Driver attribution for every (scenario, horizon, target).

    Returns ``(drivers, movements)``.
    """
    from .stress import apply_scenario

    horizons = horizons or config.SCENARIO_HORIZONS
    baseline_name = scenarios.baseline_name

    stressed_features = ["credit_score", "ltv", "rate_spread", "dti", "interest_rate"]
    driver_frames: list[pd.DataFrame] = []
    movement_frames: list[pd.DataFrame] = []

    for scenario in scenarios.scenarios:
        if scenario == baseline_name:
            continue
        anchor = scenarios.anchor(scenario)
        baseline_anchor = scenarios.anchor(baseline_name)
        available = set(scenarios.horizons(scenario))

        for horizon in [h for h in horizons if h in available]:
            month = scenarios.month(scenario, horizon)
            baseline_month = scenarios.month(baseline_name, horizon)
            shift = calibration.get((scenario, horizon), {}).get("credit_score_shift", 0.0)

            baseline_frame = apply_scenario(portfolio, baseline_month, baseline_anchor, 0.0).data
            stressed_frame = apply_scenario(portfolio, month, anchor, shift).data

            movement = feature_movement(baseline_frame, stressed_frame, stressed_features)
            movement.insert(0, "scenario", scenario)
            movement.insert(1, "horizon_month", horizon)
            movement_frames.append(movement)

            for target, label in config.SCENARIO_TARGETS.items():
                model = models.get(target)
                if model is None:
                    continue
                driver_frames.append(
                    scenario_drivers(
                        model, baseline_frame, stressed_frame, label, scenario, horizon, top_n
                    )
                )

    drivers = pd.concat(driver_frames, ignore_index=True) if driver_frames else pd.DataFrame()
    movements = pd.concat(movement_frames, ignore_index=True) if movement_frames else pd.DataFrame()
    return drivers, movements
