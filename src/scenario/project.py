"""
Projecting the portfolio under each scenario.

What is being projected
-----------------------
For each scenario and projection month, the portfolio's feature matrix is
stressed to that month's assumptions and re-scored through the Phase 3 models.
The result is the **conditional forward rate as at that month**: at projection
month 24 under Adverse-Credit, ``default_12m`` is the probability a loan
defaults over the following twelve months, given the book and the macro state
at month 24.

That is a stress-sensitivity projection, and it is deliberately not a cash-flow
run-off. The Phase 3 models predict a fixed forward window from a record's own
month; they do not compound, and this pipeline does not pretend they do. The
portfolio is held at its last observed position rather than amortised, defaulted
and prepaid loans are not removed as the horizon extends, and no balance is
rolled forward. What the numbers answer is "how much worse does this book look
under that macro state", which is the question a stress test is for. What they
do not answer is "what are the cumulative losses", which needs the Phase 4
hazards and a run-off engine.

Scoring population
------------------
The **latest observed record per loan** across the whole panel: the book as it
stands at the cutoff. Scoring every historical month would weight the projection
towards loans that happen to have long histories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .. import config
from .macro import MacroScenarios, ScenarioMonth
from .stress import apply_scenario, calibrate_credit_shift


@dataclass
class ProjectionInputs:
    """The portfolio, the models and the scenario table, ready to project."""

    portfolio: pd.DataFrame
    models: dict
    scenarios: MacroScenarios
    calibration: dict = field(default_factory=dict)


def load_models(targets: dict[str, str] | None = None, variant: str = "improved") -> dict:
    """
    Load the Phase 3 models named in ``config.SCENARIO_TARGETS``.

    The calibrated estimator is used where one was fitted, because a scenario
    projection is read as a *rate* -- a level, not a ranking -- and an
    uncalibrated reweighted model is out by roughly an order of magnitude on
    exactly that quantity.
    """
    targets = targets or config.SCENARIO_TARGETS
    models = {}
    for target in targets:
        path = config.MODELS_DIR / f"{target}__{variant}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No Phase 3 model at {path}. Run `make predict` first -- the scenario "
                "pipeline projects the trained models rather than refitting them."
            )
        models[target] = joblib.load(path)
    return models


def latest_position(frame: pd.DataFrame) -> pd.DataFrame:
    """The most recent observed record per loan: the book as it stands."""
    ordered = frame.sort_values([config.ID_COL, config.TIME_COL])
    return ordered.groupby(config.ID_COL, sort=False).tail(1).reset_index(drop=True)


def _portfolio_scorer(model):
    """``frame -> mean predicted probability`` across the portfolio."""

    def score(frame: pd.DataFrame) -> float:
        return float(model.predict_proba(frame)[:, 1].mean())

    return score


def calibrate_scenarios(
    inputs: ProjectionInputs,
    calibration_target: str = "next_12m_default_flag",
    horizons: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Solve the credit-score shift for every (scenario, horizon).

    Calibrated once against the default model and then reused for every target,
    because the shift describes a state of the world -- the book's credit
    quality at that horizon -- not a per-model tuning knob. Letting each target
    solve its own shift would produce three mutually inconsistent portfolios
    and call them one scenario.
    """
    model = inputs.models[calibration_target]
    score = _portfolio_scorer(model)
    baseline_name = inputs.scenarios.baseline_name

    # Only the horizons that will actually be projected. Each calibration is a
    # root-find costing a handful of full-portfolio scorings, so solving all 48
    # months to report five of them is 90% wasted work.
    requested = set(horizons or config.SCENARIO_HORIZONS)

    # The baseline frame at a given horizon is identical for every scenario, so
    # it is scored once and reused.
    baseline_rates: dict[int, float] = {}
    baseline_anchor = inputs.scenarios.anchor(baseline_name)

    shifts: dict[tuple[str, int], dict] = {}
    for scenario in inputs.scenarios.scenarios:
        anchor = inputs.scenarios.anchor(scenario)
        for horizon in sorted(set(inputs.scenarios.horizons(scenario)) & requested):
            month = inputs.scenarios.month(scenario, horizon)

            # The reference is the *baseline* scenario at the same horizon, so
            # the multiplier is measured against the same point in time rather
            # than against today.
            if horizon not in baseline_rates:
                baseline_month = inputs.scenarios.month(baseline_name, horizon)
                baseline_frame = apply_scenario(
                    inputs.portfolio, baseline_month, baseline_anchor, credit_shift=0.0
                ).data
                baseline_rates[horizon] = score(baseline_frame)
            baseline_rate = baseline_rates[horizon]

            if scenario == baseline_name or abs(month.default_multiplier - 1.0) < 1e-9:
                shifts[(scenario, horizon)] = {
                    "credit_score_shift": 0.0,
                    "baseline_rate": baseline_rate,
                    "converged": True,
                    "residual": 0.0,
                    "target_multiplier": month.default_multiplier,
                }
                continue

            shift, diagnostics = calibrate_credit_shift(
                score, inputs.portfolio, month, anchor,
                baseline_rate=baseline_rate,
                target_multiplier=month.default_multiplier,
            )
            shifts[(scenario, horizon)] = {"credit_score_shift": shift, **diagnostics}

            if verbose and horizon in config.SCENARIO_HORIZONS:
                status = "" if diagnostics.get("converged") else "  [not attainable]"
                print(
                    f"  {scenario:<16} h={horizon:>2}  multiplier={month.default_multiplier:.2f}"
                    f"  ->  credit shift {shift:+7.1f} pts{status}"
                )

    return shifts


def project(
    inputs: ProjectionInputs,
    horizons: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score the portfolio under every scenario and horizon.

    Returns ``(record_level, portfolio_level)``. The record-level frame carries
    one row per (loan, scenario, horizon, target) and is what the segment
    aggregations and driver attribution consume; the portfolio-level frame is
    the headline table.
    """
    horizons = horizons or config.SCENARIO_HORIZONS
    records: list[pd.DataFrame] = []

    identifiers = [config.ID_COL, *config.SCENARIO_SEGMENTS]
    identifiers = [c for c in identifiers if c in inputs.portfolio.columns]

    for scenario in inputs.scenarios.scenarios:
        anchor = inputs.scenarios.anchor(scenario)
        available = set(inputs.scenarios.horizons(scenario))
        for horizon in [h for h in horizons if h in available]:
            month = inputs.scenarios.month(scenario, horizon)
            shift = inputs.calibration.get((scenario, horizon), {}).get("credit_score_shift", 0.0)
            stressed = apply_scenario(inputs.portfolio, month, anchor, credit_shift=shift)

            frame = inputs.portfolio[identifiers].copy()
            frame["scenario"] = scenario
            frame["horizon_month"] = horizon
            frame["projection_month"] = month.projection_month
            frame["credit_score_shift"] = shift
            frame["stressed_ltv"] = stressed.data["ltv"].to_numpy() if "ltv" in stressed.data else np.nan
            frame["stressed_credit_score"] = (
                stressed.data["credit_score"].to_numpy() if "credit_score" in stressed.data else np.nan
            )

            for target, label in config.SCENARIO_TARGETS.items():
                model = inputs.models.get(target)
                if model is None:
                    continue
                frame[label] = model.predict_proba(stressed.data)[:, 1]

            records.append(frame)

        if verbose:
            print(f"  projected {scenario}")

    record_level = pd.concat(records, ignore_index=True)
    return record_level, portfolio_projection(record_level, inputs.scenarios)


def portfolio_projection(record_level: pd.DataFrame, scenarios=None) -> pd.DataFrame:
    """
    Mean projected rate per scenario and horizon, by **two** methods.

    ``<label>`` is the feature-stress projection: the portfolio's features moved
    to the scenario's macro state and re-scored through the Phase 3 model.

    ``<label>_stated`` is the scenario file's own view: the baseline rate at the
    same horizon multiplied by the file's stated hazard multiplier.

    Both are reported because they answer the same question from different
    directions, and because the credit channel is known to saturate -- there is
    a point past which no shift in credit score reproduces the stated
    multiplier, and a report that showed only one method would hide it. The gap
    between the columns is a headline finding, not a rounding difference.
    """
    labels = [label for label in config.SCENARIO_TARGETS.values() if label in record_level.columns]
    grouped = (
        record_level.groupby(["scenario", "horizon_month", "projection_month"], as_index=False)
        .agg({**{label: "mean" for label in labels}, "credit_score_shift": "first"})
    )
    grouped["loans"] = record_level.groupby(
        ["scenario", "horizon_month", "projection_month"]
    ).size().to_numpy()

    grouped = _attach_baseline_delta(grouped, labels)
    if scenarios is not None:
        grouped = _attach_stated_projection(grouped, labels, scenarios)
    return grouped


def _attach_stated_projection(
    grouped: pd.DataFrame, labels: list[str], scenarios
) -> pd.DataFrame:
    """Baseline rate times the scenario file's own stated multiplier."""
    baseline_name = scenarios.baseline_name
    reference = grouped[grouped["scenario"] == baseline_name].set_index("horizon_month")

    out = grouped.copy()
    for label in labels:
        stated = []
        for _, row in out.iterrows():
            horizon = int(row["horizon_month"])
            multiplier = (
                scenarios.month(row["scenario"], horizon).multiplier_for(label)
                if horizon in reference.index
                else None
            )
            stated.append(
                reference.loc[horizon, label] * multiplier if multiplier is not None else np.nan
            )
        # Only carry the column where the file actually states a multiplier.
        if not all(pd.isna(v) for v in stated):
            out[f"{label}_stated"] = stated
    return out


def segment_projection(record_level: pd.DataFrame, segment: str) -> pd.DataFrame:
    """
    Projected rates cut by one segment, with the change against baseline.

    Segment-level deltas are what turn a portfolio number into an action: a
    30bp rise spread evenly is a different problem from the same rise
    concentrated in one servicer's book.
    """
    if segment not in record_level.columns:
        return pd.DataFrame()

    labels = [label for label in config.SCENARIO_TARGETS.values() if label in record_level.columns]
    grouped = (
        record_level.groupby(["scenario", "horizon_month", segment], as_index=False)
        .agg({label: "mean" for label in labels})
    )
    grouped["loans"] = record_level.groupby(
        ["scenario", "horizon_month", segment]
    ).size().to_numpy()
    return _attach_baseline_delta(grouped, labels, keys=["horizon_month", segment])


def _attach_baseline_delta(
    grouped: pd.DataFrame, labels: list[str], keys: list[str] | None = None
) -> pd.DataFrame:
    """Add ``<label>_vs_baseline`` columns, in percentage points."""
    keys = keys or ["horizon_month"]
    baseline_name = grouped.loc[
        grouped[labels].std(axis=1).notna(), "scenario"
    ].iloc[0] if grouped.empty else None

    # The reference is whichever scenario is flat -- identified the same way
    # MacroScenarios does, so the two never disagree.
    spread = grouped.groupby("scenario")[labels].std().fillna(0.0)
    baseline_name = spread.sum(axis=1).idxmin() if len(spread) else None

    reference = grouped[grouped["scenario"] == baseline_name].set_index(keys)[labels]
    out = grouped.copy()
    for label in labels:
        mapped = out.set_index(keys).index.map(reference[label])
        out[f"{label}_vs_baseline_pp"] = 100.0 * (out[label].to_numpy() - np.asarray(mapped, dtype=float))
    return out


def multiplier_check(
    portfolio_level: pd.DataFrame, scenarios: MacroScenarios
) -> pd.DataFrame:
    """
    The model's realised multiplier against the scenario file's stated one.

    Two independent views of the same scenario: the file states a hazard
    multiplier, and the feature-space stress produces one. They should agree on
    default -- the credit channel was calibrated to make them agree -- and they
    are free to disagree on prepayment, which no channel was calibrated
    against. That disagreement is the interesting column: it is the model's own
    view of the refinance response, derived only from the rate path.
    """
    baseline_name = scenarios.baseline_name
    reference = portfolio_level[portfolio_level["scenario"] == baseline_name].set_index("horizon_month")

    rows = []
    for _, row in portfolio_level.iterrows():
        horizon = int(row["horizon_month"])
        if horizon not in reference.index:
            continue
        month = scenarios.month(row["scenario"], horizon)
        base = reference.loc[horizon]
        for label in config.SCENARIO_TARGETS.values():
            stated = month.multiplier_for(label)
            if stated is None or label not in portfolio_level.columns or base[label] == 0:
                continue
            realised = row[label] / base[label]
            rows.append(
                {
                    "scenario": row["scenario"],
                    "horizon_month": horizon,
                    "measure": label,
                    "stated_multiplier": stated,
                    "model_multiplier": realised,
                    "gap": realised - stated,
                    "calibrated": label == "default_12m",
                }
            )
    return pd.DataFrame(rows)


def saturation_summary(calibration: dict, scenarios) -> pd.DataFrame:
    """
    Where the credit channel runs out of room.

    A scenario multiplier the model cannot reach by *any* credit-score shift is
    the most important thing a stress framework can tell you, and it is easy to
    hide: a naive calibration clamps at the search bound, reports a number, and
    the projection quietly under-states the scenario. This table names the
    shortfall in the file's own units.
    """
    rows = []
    for (scenario, horizon), diagnostics in sorted(calibration.items()):
        if scenario == scenarios.baseline_name:
            continue
        month = scenarios.month(scenario, horizon)
        rows.append(
            {
                "scenario": scenario,
                "horizon_month": horizon,
                "stated_multiplier": month.default_multiplier,
                "attainable_multiplier": diagnostics.get("attainable_multiplier", np.nan),
                "credit_score_shift": diagnostics.get("credit_score_shift", np.nan),
                "reached": bool(diagnostics.get("converged", False)),
                "shortfall": month.default_multiplier
                - diagnostics.get("attainable_multiplier", np.nan),
            }
        )
    return pd.DataFrame(rows)
