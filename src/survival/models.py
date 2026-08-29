"""
Cause-specific Cox proportional-hazards models and the competing-risk CIF
built from them.

The modelling strategy
----------------------
Two Cox models, one per cause. Each is fitted with the *other* cause treated as
censoring -- that is the cause-specific hazard, and it is the quantity a Cox
model can legitimately estimate under competing risks. What it cannot do on its
own is answer "what fraction of loans will have defaulted by month 36", because
that depends on how many are still around to default. So the two fitted models
are combined into a cumulative incidence function:

    S(t|x)     = exp( -(H_default(t|x) + H_prepaid(t|x)) )
    CIF_k(t|x) = sum over s <= t of  h_k(s|x) * S(s-|x)

This is the "competing-risk approximation" route: cause-specific hazards from
Cox, cumulative incidence assembled from them. It reduces to the Aalen-Johansen
estimator when the covariate vector is the sample mean, which is the check in
``tests/test_survival_censoring.py``.

Covariates are **origination-time only**, standardised so each hazard ratio
reads as "per one standard deviation". Nothing measured after month 0 is
admitted: the covariate vector of a survival model is what was known when the
clock started, and a value read off month 18 would be the survival-analysis
form of the leakage Task 2 forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from .. import config
from .dataset import SurvivalFrame


# --------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------
@dataclass
class DesignSpec:
    """Column list and standardisation constants, learned on TRAIN only."""

    numeric: list[str]
    categorical: list[str]
    means: pd.Series
    scales: pd.Series
    dummy_columns: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        frame: SurvivalFrame,
        numeric: list[str] | None = None,
        categorical: list[str] | None = None,
    ) -> "DesignSpec":
        data = frame.data
        numeric = [c for c in (numeric or config.SURVIVAL_NUMERIC_COVARIATES) if c in data.columns]
        categorical = [
            c for c in (categorical or config.SURVIVAL_CATEGORICAL_COVARIATES) if c in data.columns
        ]

        values = data[numeric].apply(pd.to_numeric, errors="coerce")
        means = values.mean()
        scales = values.std(ddof=0).replace(0, 1.0)

        spec = cls(numeric=numeric, categorical=categorical, means=means, scales=scales)
        spec.dummy_columns = list(spec.transform(frame).columns)
        return spec

    def transform(self, frame: SurvivalFrame) -> pd.DataFrame:
        """Standardised numerics + one-hot categoricals, aligned to the fitted columns."""
        data = frame.data
        numeric = data[self.numeric].apply(pd.to_numeric, errors="coerce")
        standardised = (numeric - self.means) / self.scales
        standardised = standardised.fillna(0.0)

        if self.categorical:
            dummies = pd.get_dummies(
                data[self.categorical].astype("object"), drop_first=True, dtype=float
            )
            design = pd.concat([standardised, dummies], axis=1)
        else:
            design = standardised

        design.columns = [_safe(c) for c in design.columns]
        if self.dummy_columns:
            design = design.reindex(columns=self.dummy_columns, fill_value=0.0)
        return design


def _safe(name: str) -> str:
    """lifelines formulas dislike spaces and slashes in column names."""
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("+", "plus")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace("%", "pct")
    )


# --------------------------------------------------------------------------
# Cause-specific Cox
# --------------------------------------------------------------------------
@dataclass
class CauseSpecificCox:
    """A fitted Cox model for one cause, plus the design it was fitted on."""

    cause: int
    fitter: CoxPHFitter
    spec: DesignSpec

    @property
    def label(self) -> str:
        return config.EVENT_LABELS[self.cause]

    def partial_hazard(self, frame: SurvivalFrame) -> np.ndarray:
        """``exp((x - xbar)'beta)`` -- the multiplier on the baseline hazard."""
        design = self.spec.transform(frame)
        return self.fitter.predict_partial_hazard(design).to_numpy()

    def cumulative_hazard(self, frame: SurvivalFrame, times: np.ndarray) -> np.ndarray:
        """``H_k(t|x)`` as an (n_loans, n_times) array."""
        baseline = _baseline_cumulative_hazard(self.fitter, times)
        return np.outer(self.partial_hazard(frame), baseline)

    def summary_frame(self) -> pd.DataFrame:
        """Hazard ratios with confidence intervals, per one standard deviation."""
        summary = self.fitter.summary.reset_index().rename(columns={"covariate": "covariate"})
        keep = {
            "covariate": "covariate",
            "exp(coef)": "hazard_ratio",
            "exp(coef) lower 95%": "hr_lower_95",
            "exp(coef) upper 95%": "hr_upper_95",
            "z": "z",
            "p": "p_value",
        }
        available = {k: v for k, v in keep.items() if k in summary.columns}
        out = summary[list(available)].rename(columns=available)
        out.insert(0, "cause", self.label)
        return out.sort_values("hazard_ratio", ascending=False).reset_index(drop=True)


def _baseline_cumulative_hazard(fitter: CoxPHFitter, times: np.ndarray) -> np.ndarray:
    """Step-interpolate the fitted baseline cumulative hazard onto a monthly grid."""
    baseline = fitter.baseline_cumulative_hazard_
    column = baseline.columns[0]
    known_times = baseline.index.to_numpy(dtype=float)
    known_values = baseline[column].to_numpy(dtype=float)
    # A cumulative hazard is a right-continuous step function: carry the last
    # known value forward rather than interpolating between event times.
    indices = np.searchsorted(known_times, np.asarray(times, dtype=float), side="right") - 1
    out = np.where(indices >= 0, known_values[np.clip(indices, 0, None)], 0.0)
    return out


def fit_cause_specific_cox(
    frame: SurvivalFrame,
    cause: int,
    spec: DesignSpec | None = None,
    penalizer: float = 0.01,
) -> CauseSpecificCox:
    """
    Fit one cause-specific Cox model.

    A small ridge ``penalizer`` is on by default. It is not hyperparameter
    tuning -- it keeps the fit finite when a segment is close to separated
    (here, the ``800+`` credit band records almost no defaults at all), which
    would otherwise send a coefficient to infinity and take the standard errors
    with it.

    Delayed entry is passed to lifelines via ``entry_col`` so the risk set at
    month *t* excludes loans not yet observed at *t*.
    """
    spec = spec or DesignSpec.fit(frame)
    design = spec.transform(frame)

    fit_frame = design.copy()
    fit_frame["_duration"] = frame.data[frame.duration_col].to_numpy()
    fit_frame["_event"] = frame.cause_indicator(cause)
    fit_frame["_entry"] = frame.data[frame.entry_col].to_numpy()

    fitter = CoxPHFitter(penalizer=penalizer)
    fitter.fit(
        fit_frame,
        duration_col="_duration",
        event_col="_event",
        entry_col="_entry",
    )
    return CauseSpecificCox(cause=cause, fitter=fitter, spec=spec)


def check_proportional_hazards(model: CauseSpecificCox, frame: SurvivalFrame) -> pd.DataFrame:
    """
    Schoenfeld-residual test of the proportional-hazards assumption.

    Reported rather than acted on. A violation means the hazard ratio for that
    covariate is not constant over loan age -- worth stating in the model card,
    because it is the assumption the whole Cox apparatus rests on and quietly
    breaking it is a standard way survival results go wrong.
    """
    design = model.spec.transform(frame)
    fit_frame = design.copy()
    fit_frame["_duration"] = frame.data[frame.duration_col].to_numpy()
    fit_frame["_event"] = frame.cause_indicator(model.cause)
    fit_frame["_entry"] = frame.data[frame.entry_col].to_numpy()

    try:
        result = proportional_hazard_test(model.fitter, fit_frame, time_transform="rank")
    except Exception as exc:  # pragma: no cover - diagnostic only
        return pd.DataFrame([{"cause": model.label, "error": str(exc)}])

    out = result.summary.reset_index()
    rename = {out.columns[0]: "covariate"}
    out = out.rename(columns=rename)
    out.insert(0, "cause", model.label)
    out["violates_ph_at_0.01"] = out["p"] < 0.01
    return out.sort_values("p").reset_index(drop=True)


# --------------------------------------------------------------------------
# Competing-risk cumulative incidence from the two cause-specific models
# --------------------------------------------------------------------------
def cumulative_incidence(
    models: dict[int, CauseSpecificCox],
    frame: SurvivalFrame,
    times: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Per-loan cumulative incidence for every cause, as (n_loans, n_times) arrays.

    ``S(t|x)`` uses the hazards of **both** causes -- that coupling is what
    makes this a competing-risk estimate rather than two independent survival
    curves that can sum past 1.
    """
    times = np.asarray(times, dtype=float)
    cumulative = {cause: model.cumulative_hazard(frame, times) for cause, model in models.items()}

    total = sum(cumulative.values())
    survival = np.exp(-total)
    # S(s-) : survival through the previous grid point.
    survival_lagged = np.concatenate([np.ones((survival.shape[0], 1)), survival[:, :-1]], axis=1)

    out = {}
    for cause, cum_hazard in cumulative.items():
        increments = np.diff(cum_hazard, axis=1, prepend=0.0)
        out[cause] = np.cumsum(increments * survival_lagged, axis=1)
    return out


def survival_matrix(
    models: dict[int, CauseSpecificCox], frame: SurvivalFrame, times: np.ndarray
) -> np.ndarray:
    """All-cause survival ``S(t|x)`` as (n_loans, n_times)."""
    total = sum(model.cumulative_hazard(frame, times) for model in models.values())
    return np.exp(-total)


def cause_specific_survival(
    model: CauseSpecificCox, frame: SurvivalFrame, times: np.ndarray
) -> np.ndarray:
    """
    Survival from one cause alone, competing event censored.

    This is the quantity the Brier-score machinery expects, because that
    machinery evaluates a single binary event against a single survival curve.
    """
    return np.exp(-model.cumulative_hazard(frame, times))


def mean_profile_cif(
    models: dict[int, CauseSpecificCox], frame: SurvivalFrame, times: np.ndarray
) -> dict[int, np.ndarray]:
    """Portfolio-average CIF: the per-loan curves averaged, not the curve of the average loan."""
    per_loan = cumulative_incidence(models, frame, times)
    return {cause: matrix.mean(axis=0) for cause, matrix in per_loan.items()}
