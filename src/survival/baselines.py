"""
Non-parametric estimators and the naive baseline.

Three things live here, in increasing order of how much they assume:

* :class:`ConstantHazardModel` -- one exponential hazard per cause, fitted by
  occurrence/exposure. This is the naive baseline Task 3 asks for: it says
  every loan, at every age, faces the same monthly risk. Everything else has
  to beat it.
* :func:`kaplan_meier` -- the non-parametric survival curve, left-truncation
  aware.
* :func:`aalen_johansen_cif` -- the **cumulative incidence function** under
  competing risks.

Why the CIF and not ``1 - KM``
------------------------------
Under competing risks, running a Kaplan-Meier on "default, treating prepayment
as censored" and reporting ``1 - S(t)`` overstates default incidence, because
it answers "what would default incidence be in a world where nobody could
prepay?" -- a world that does not exist. The Aalen-Johansen estimator weights
each cause-specific hazard by the probability of still being in the portfolio
to experience it:

    CIF_k(t) = sum over s <= t of  [ d_k(s) / n(s) ] * S(s-)

where ``S`` is the *all-cause* Kaplan-Meier and ``n(s)`` the risk set. Both are
computed here and both are reported, because the size of the gap between them
is the clearest single demonstration that competing risks were handled rather
than assumed away.

The estimator is implemented directly rather than via
``lifelines.AalenJohansenFitter`` for one reason: monthly durations are heavily
tied, and that fitter resolves ties by jittering the durations with a random
seed. The discrete-time formula above is exact for tied times and
deterministic. ``tests/test_survival_censoring.py`` cross-checks it against
lifelines on jitter-free data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter

from .. import config
from .dataset import SurvivalFrame


# --------------------------------------------------------------------------
# Risk sets
# --------------------------------------------------------------------------
def risk_table(frame: SurvivalFrame, timeline: np.ndarray | None = None) -> pd.DataFrame:
    """
    Risk set and event counts at each month, honouring delayed entry.

    ``n(t) = #{ entry < t <= exit }`` -- the left-truncation-aware risk set.
    With no delayed entry this collapses to ``#{ exit >= t }``, so the same
    code path serves both cases and there is no second, subtly different
    implementation to keep in step.
    """
    data = frame.data
    entry = data[frame.entry_col].to_numpy()
    exit_ = data[frame.duration_col].to_numpy()
    event = data[frame.event_col].to_numpy()

    if timeline is None:
        timeline = np.arange(1, int(np.nanmax(exit_)) + 1)
    timeline = np.asarray(timeline, dtype=float)

    at_risk = np.array([((entry < t) & (exit_ >= t)).sum() for t in timeline], dtype=float)
    events_default = np.array(
        [((exit_ == t) & (event == config.EVENT_DEFAULT)).sum() for t in timeline], dtype=float
    )
    events_prepaid = np.array(
        [((exit_ == t) & (event == config.EVENT_PREPAID)).sum() for t in timeline], dtype=float
    )
    censored = np.array(
        [((exit_ == t) & (event == config.EVENT_CENSORED)).sum() for t in timeline], dtype=float
    )

    return pd.DataFrame(
        {
            "month": timeline,
            "at_risk": at_risk,
            "events_default": events_default,
            "events_prepaid": events_prepaid,
            "events_any": events_default + events_prepaid,
            "censored": censored,
        }
    )


# --------------------------------------------------------------------------
# Baseline: constant hazard (exponential)
# --------------------------------------------------------------------------
@dataclass
class ConstantHazardModel:
    """
    The naive baseline: one constant monthly hazard per cause.

    Fitted by occurrence over exposure -- the maximum-likelihood estimator for
    an exponential with right-censored, left-truncated data, and the reason
    censored loans still contribute: they add exposure to the denominator even
    though they add no event to the numerator. Dropping them, the classic
    mistake, would inflate every hazard.

    It has no covariates, so its concordance is 0.5 by construction. Its value
    as a baseline is in the *calibration* metrics: it fixes a level, and the
    Cox models have to beat that level, not just rank better than a coin.
    """

    hazards: dict[int, float]
    exposure_months: float
    events: dict[int, int]

    @classmethod
    def fit(cls, frame: SurvivalFrame) -> "ConstantHazardModel":
        data = frame.data
        exposure = float((data[frame.duration_col] - data[frame.entry_col]).sum())
        events = {
            cause: int((data[frame.event_col] == cause).sum())
            for cause in (config.EVENT_DEFAULT, config.EVENT_PREPAID)
        }
        hazards = {c: (n / exposure if exposure > 0 else 0.0) for c, n in events.items()}
        return cls(hazards=hazards, exposure_months=exposure, events=events)

    @property
    def total_hazard(self) -> float:
        return float(sum(self.hazards.values()))

    def survival_function(self, times: np.ndarray) -> np.ndarray:
        """All-cause survival ``S(t) = exp(-Lambda t)``."""
        return np.exp(-self.total_hazard * np.asarray(times, dtype=float))

    def cumulative_incidence(self, cause: int, times: np.ndarray) -> np.ndarray:
        """
        Closed-form competing-risk CIF for constant hazards::

            CIF_k(t) = (lambda_k / Lambda) * (1 - exp(-Lambda t))

        The ``lambda_k / Lambda`` factor is the competing-risk correction: even
        at infinite time only that share of loans ever reaches cause ``k``.
        """
        total = self.total_hazard
        if total == 0:
            return np.zeros_like(np.asarray(times, dtype=float))
        share = self.hazards[cause] / total
        return share * (1.0 - np.exp(-total * np.asarray(times, dtype=float)))

    def summary(self) -> pd.DataFrame:
        rows = []
        for cause, hazard in self.hazards.items():
            rows.append(
                {
                    "cause": config.EVENT_LABELS[cause],
                    "events": self.events[cause],
                    "exposure_months": self.exposure_months,
                    "monthly_hazard": hazard,
                    "annualised_rate": 1 - (1 - hazard) ** 12,
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Non-parametric curves
# --------------------------------------------------------------------------
def kaplan_meier(
    frame: SurvivalFrame, cause: int | None = None, label: str | None = None
) -> KaplanMeierFitter:
    """
    Kaplan-Meier survival curve, with delayed entry passed through.

    ``cause=None`` gives all-cause survival ("still performing and still on the
    book"). A specific cause gives the *cause-specific* KM, in which the
    competing event is censored -- correct for hazard estimation, and reported
    alongside the CIF precisely to show what it over-states.
    """
    data = frame.data
    if cause is None:
        observed = (data[frame.event_col] != config.EVENT_CENSORED).astype(int)
        label = label or "any event"
    else:
        observed = frame.cause_indicator(cause)
        label = label or config.EVENT_LABELS[cause]

    fitter = KaplanMeierFitter(label=label)
    fitter.fit(
        durations=data[frame.duration_col],
        event_observed=observed,
        entry=data[frame.entry_col],
    )
    return fitter


def nelson_aalen(frame: SurvivalFrame, cause: int) -> NelsonAalenFitter:
    """Cause-specific cumulative hazard -- the shape the Cox model assumes is shared."""
    fitter = NelsonAalenFitter(label=config.EVENT_LABELS[cause])
    fitter.fit(
        durations=frame.data[frame.duration_col],
        event_observed=frame.cause_indicator(cause),
        entry=frame.data[frame.entry_col],
    )
    return fitter


def aalen_johansen_cif(
    frame: SurvivalFrame, cause: int, timeline: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Cumulative incidence for one cause under competing risks.

    Returns a frame with ``month``, ``at_risk``, ``cause_hazard``,
    ``overall_survival`` and ``cif``. The ``naive_1_minus_km`` column carries
    the cause-specific ``1 - KM`` for the same data, so the overstatement is
    visible in the table rather than only in prose.
    """
    table = risk_table(frame, timeline)
    at_risk = table["at_risk"].to_numpy()
    safe_at_risk = np.where(at_risk > 0, at_risk, np.nan)

    events_cause = table[f"events_{config.EVENT_LABELS[cause]}"].to_numpy()
    hazard_cause = np.nan_to_num(events_cause / safe_at_risk)
    hazard_any = np.nan_to_num(table["events_any"].to_numpy() / safe_at_risk)

    # All-cause survival, lagged: S(s-) is survival through the *previous* month.
    survival = np.cumprod(1.0 - hazard_any)
    survival_lagged = np.concatenate([[1.0], survival[:-1]])

    cif = np.cumsum(hazard_cause * survival_lagged)

    # Cause-specific KM for the same cause, competing event treated as censored.
    naive_survival = np.cumprod(1.0 - hazard_cause)

    out = table.copy()
    out["cause_hazard"] = hazard_cause
    out["overall_survival"] = survival
    out["cif"] = cif
    out["naive_1_minus_km"] = 1.0 - naive_survival
    out["cause"] = config.EVENT_LABELS[cause]
    return out


def effective_timeline(
    frame: SurvivalFrame, min_at_risk: int = 50, max_month: int | None = None
) -> np.ndarray:
    """
    The months over which a curve is worth drawing.

    A Kaplan-Meier or Aalen-Johansen tail computed on a handful of remaining
    loans is a step function made of noise, and it flattens out in a way that
    reads as "the risk stopped" rather than "we ran out of data". This returns
    the grid up to the last month where at least ``min_at_risk`` loans are
    still under observation, so the plotted curve ends where the evidence does.
    """
    exit_ = frame.data[frame.duration_col].to_numpy()
    entry = frame.data[frame.entry_col].to_numpy()
    limit = int(max_month or np.nanmax(exit_))

    last_supported = 0
    for month in range(1, limit + 1):
        if ((entry < month) & (exit_ >= month)).sum() >= min_at_risk:
            last_supported = month
        else:
            break
    return np.arange(1, max(last_supported, 1) + 1)


def cif_by_segment(
    frame: SurvivalFrame,
    segment_col: str,
    cause: int,
    timeline: np.ndarray | None = None,
    min_loans: int = 50,
    min_at_risk: int = 50,
) -> pd.DataFrame:
    """
    Cumulative incidence curves cut by a categorical feature.

    Two guards, both about not drawing what is not there:

    * Segments with fewer than ``min_loans`` are dropped. A CIF built on twenty
      loans is a step function pretending to be a curve, and putting it on the
      same axes as a well-populated segment invites the wrong reading.
    * Each segment's curve is **truncated where its own risk set falls below**
      ``min_at_risk``. This matters most for recent vintages: a 2023 cohort has
      15 months of follow-up, and drawing its CIF flat out to month 59 says
      "these loans stopped defaulting" when it means "we stopped watching".
    """
    data = frame.data
    if segment_col not in data.columns:
        raise ValueError(f"{segment_col!r} is not a column of the survival frame")

    if timeline is None:
        timeline = np.arange(1, int(data[frame.duration_col].max()) + 1)

    frames = []
    for value, group in data.groupby(segment_col, dropna=True):
        if len(group) < min_loans:
            continue
        segment = SurvivalFrame(data=group, censoring={})
        supported = effective_timeline(segment, min_at_risk=min_at_risk, max_month=timeline[-1])
        curve = aalen_johansen_cif(segment, cause, timeline=timeline)
        curve = curve[curve["month"] <= supported[-1]]
        curve[segment_col] = value
        curve["n_loans"] = len(group)
        curve["follow_up_months"] = int(supported[-1])
        frames.append(curve)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def incidence_at_horizons(
    curve: pd.DataFrame, horizons: tuple[int, ...] = config.SURVIVAL_HORIZONS
) -> dict[int, float]:
    """Read a CIF off at fixed months on book, for the summary tables."""
    out = {}
    for horizon in horizons:
        rows = curve.loc[curve["month"] <= horizon, "cif"]
        out[horizon] = float(rows.iloc[-1]) if len(rows) else float("nan")
    return out
