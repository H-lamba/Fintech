"""
Evaluating survival models: discrimination, calibration, and the baseline
comparison Task 3 asks for.

Three families of number, because no single one is sufficient:

* **Concordance (Harrell's C)** -- does the model rank loans by risk? The
  constant-hazard baseline scores 0.5 here by construction; that is the point
  of including it, not a defect.
* **Time-dependent Brier score / IBS** -- are the predicted *probabilities*
  right at each horizon? This is where the constant-hazard model is actually
  competitive on level and loses on shape, and where a model that ignores
  censoring looks best and is most wrong.
* **Calibration by risk decile** -- predicted cumulative incidence against the
  Aalen-Johansen estimate observed in that decile. The table a credit committee
  reads before it trusts a number.

Censoring in the metrics
------------------------
The Brier score is inverse-probability-of-censoring weighted (IPCW): a loan
censored at month 14 contributes to the month-12 score and not to the month-24
one, and the loans that *are* still observed at 24 are up-weighted to stand in
for it. Without that weighting, a model is rewarded for the censoring pattern
rather than the risk. The censoring distribution is estimated on TRAIN and
applied to TEST, exactly as the event model is.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from .. import config
from .baselines import ConstantHazardModel, aalen_johansen_cif
from .dataset import SurvivalFrame
from .models import CauseSpecificCox, cause_specific_survival


# --------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------
def concordance(model: CauseSpecificCox, frame: SurvivalFrame) -> float:
    """
    Harrell's C for one cause: the share of comparable pairs ranked correctly.

    The risk score is the partial hazard, negated because ``concordance_index``
    expects a *predicted survival time*: a higher hazard means a shorter time.
    """
    risk = model.partial_hazard(frame)
    return float(
        concordance_index(
            frame.data[frame.duration_col],
            -risk,
            frame.cause_indicator(model.cause),
        )
    )


# --------------------------------------------------------------------------
# Time-dependent Brier score
# --------------------------------------------------------------------------
def _to_structured(frame: SurvivalFrame, cause: int):
    from sksurv.util import Surv

    return Surv.from_arrays(
        event=frame.cause_indicator(cause).astype(bool),
        time=frame.data[frame.duration_col].to_numpy(dtype=float),
    )


def _usable_horizons(frame: SurvivalFrame, cause: int, horizons) -> np.ndarray:
    """
    Keep only horizons the holdout can actually score.

    IPCW needs at least one loan still at risk beyond the horizon and at least
    one event at or before it; a horizon past the end of follow-up produces an
    undefined weight, not a large error.
    """
    durations = frame.data[frame.duration_col].to_numpy(dtype=float)
    events = frame.cause_indicator(cause)
    usable = [
        float(h)
        for h in horizons
        if (durations > h).any() and ((durations <= h) & (events == 1)).any()
    ]
    return np.asarray(sorted(set(usable)), dtype=float)


def brier_scores(
    survival_estimates: dict[str, np.ndarray],
    train: SurvivalFrame,
    test: SurvivalFrame,
    cause: int,
    horizons=config.SURVIVAL_HORIZONS,
) -> pd.DataFrame:
    """
    IPCW Brier score at each horizon, plus the integrated score, per model.

    ``survival_estimates`` maps a model name to an (n_test, n_horizons) array
    of cause-specific survival probabilities, evaluated on the same horizons
    this function keeps.
    """
    from sksurv.metrics import brier_score, integrated_brier_score

    times = _usable_horizons(test, cause, horizons)
    if len(times) == 0:
        return pd.DataFrame()

    if (train.data[train.entry_col] > 0).any() or (test.data[test.entry_col] > 0).any():
        warnings.warn(
            "IPCW Brier scores ignore delayed entry; the reported values assume "
            "every loan was under observation from month 0.",
            RuntimeWarning,
            stacklevel=2,
        )

    y_train = _to_structured(train, cause)
    y_test = _to_structured(test, cause)

    rows = []
    for name, estimate in survival_estimates.items():
        estimate = np.asarray(estimate, dtype=float)[:, : len(times)]
        try:
            _, scores = brier_score(y_train, y_test, estimate, times)
            integrated = (
                float(integrated_brier_score(y_train, y_test, estimate, times))
                if len(times) > 1
                else float(scores[0])
            )
        except ValueError as exc:  # horizon outside the censoring support
            rows.append({"model": name, "cause": config.EVENT_LABELS[cause], "error": str(exc)})
            continue

        row = {"model": name, "cause": config.EVENT_LABELS[cause], "integrated_brier": integrated}
        for horizon, score in zip(times, scores):
            row[f"brier_{int(horizon)}m"] = float(score)
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Survival estimates for each competing model, on one grid
# --------------------------------------------------------------------------
def build_survival_estimates(
    cox: CauseSpecificCox,
    constant: ConstantHazardModel,
    train: SurvivalFrame,
    test: SurvivalFrame,
    cause: int,
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Cause-specific survival curves from each model, aligned on ``times``.

    All three are the *same quantity* -- P(no event of this cause by t) -- so
    the Brier comparison is like-for-like:

    - ``cox``: per-loan, covariate-driven.
    - ``constant_hazard``: one exponential curve, identical for every loan.
    - ``kaplan_meier``: the marginal training curve, identical for every loan.
      A model with covariates that cannot beat this has learned nothing.
    """
    n_test = len(test)
    times = np.asarray(times, dtype=float)

    cox_survival = cause_specific_survival(cox, test, times)

    hazard = constant.hazards[cause]
    constant_survival = np.tile(np.exp(-hazard * times), (n_test, 1))

    marginal = aalen_johansen_cif(train, cause, timeline=times)
    km_curve = np.cumprod(1.0 - marginal["cause_hazard"].to_numpy())
    km_survival = np.tile(km_curve, (n_test, 1))

    return {
        "cox": cox_survival,
        "constant_hazard": constant_survival,
        "kaplan_meier": km_survival,
    }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibration_by_risk_decile(
    predicted_cif: np.ndarray,
    frame: SurvivalFrame,
    cause: int,
    horizon: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Predicted vs. observed cumulative incidence at one horizon, by risk decile.

    "Observed" is the Aalen-Johansen estimate computed *inside* each decile, so
    the comparison respects censoring on both sides. A naive alternative --
    counting events in the decile and dividing by loans -- would understate
    incidence in exactly the deciles with the shortest follow-up.
    """
    predictions = np.asarray(predicted_cif, dtype=float)
    data = frame.data.copy()
    data["_predicted"] = predictions

    ranks = data["_predicted"].rank(method="first")
    data["_decile"] = pd.qcut(ranks, min(n_bins, data["_predicted"].nunique()), labels=False) + 1

    timeline = np.arange(1, horizon + 1)
    rows = []
    for decile, group in data.groupby("_decile"):
        segment = SurvivalFrame(data=group, censoring={})
        curve = aalen_johansen_cif(segment, cause, timeline=timeline)
        observed = float(curve["cif"].iloc[-1]) if len(curve) else float("nan")
        rows.append(
            {
                "decile": int(decile),
                "loans": len(group),
                "mean_predicted": float(group["_predicted"].mean()),
                "observed_cif": observed,
                "gap": float(group["_predicted"].mean()) - observed,
            }
        )

    out = pd.DataFrame(rows)
    out.insert(0, "cause", config.EVENT_LABELS[cause])
    out.insert(1, "horizon_months", horizon)
    return out


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------
def results_frame(rows: list[dict]) -> pd.DataFrame:
    """One row per (cause, model): concordance, Brier, IBS."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    model_order = {"constant_hazard": 0, "kaplan_meier": 1, "cox": 2}
    cause_order = {"default": 0, "prepaid": 1}
    frame["_c"] = frame["cause"].map(cause_order).fillna(9)
    frame["_m"] = frame["model"].map(model_order).fillna(9)

    leading = ["cause", "model", "concordance", "integrated_brier"]
    ordered = [c for c in leading if c in frame.columns]
    rest = [c for c in frame.columns if c not in ordered and not c.startswith("_")]
    frame = frame.sort_values(["_c", "_m"])[[*ordered, *rest]]
    return frame.reset_index(drop=True)


def results_markdown(frame: pd.DataFrame, float_format: str = ".4f") -> str:
    """Render the comparison table, with inapplicable cells as an explicit dash."""
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        values = display[column].dropna()
        spec = "d" if len(values) and (values % 1 == 0).all() else float_format
        display[column] = display[column].map(
            lambda v, spec=spec: "--" if pd.isna(v) else format(int(v) if spec == "d" else v, spec)
        )
    return display.to_markdown(index=False)
