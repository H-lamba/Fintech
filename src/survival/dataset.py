"""
Turning a monthly performance panel into a survival dataset.

The panel has one row per loan-month. A survival model wants one row per loan:
how long it was observed, and what ended the observation. Getting that
transformation right *is* most of Task 3 -- every downstream curve inherits its
mistakes -- so each decision is made explicitly here and recorded in a
censoring report rather than left implicit in a groupby.

The clock
---------
Time is **months on book** (``loan_age_months``), not calendar time. A loan
originated in 2018 and one originated in 2022 are compared at the same *age*,
which is the whole point of a survival model and the reason vintages with
short follow-up do not look artificially safe.

The three ways a loan leaves the risk set
-----------------------------------------
1. **Default** -- the event of interest (code 1).
2. **Prepayment** -- the competing event (code 2). It is not censoring: a
   prepaid loan cannot subsequently default, so treating it as "still at risk,
   just unobserved" would overstate default incidence. Cause-specific hazards
   and the Aalen-Johansen estimator both need it coded as its own event.
3. **Right-censoring** (code 0) -- observation ended with the loan still alive.
   Split further into:
   - *administrative* censoring: the loan was still active at the data cutoff.
     This is the dominant case (~80% of censored loans here) and is
     non-informative by construction -- the cutoff has nothing to do with the
     loan.
   - *loss to follow-up*: the panel simply stops reporting the loan before the
     cutoff. Recorded separately because informative censoring here would bias
     the estimates, and the only honest thing to do is show how much of it
     there is.

Left truncation (delayed entry)
-------------------------------
Where a loan is first observed at age 7 rather than age 0 it is
**left-truncated**: it is only in the sample *because* it survived to its entry
age, so counting it in the risk set from month 0 understates early hazards.
Every estimator here therefore takes an ``entry`` argument and builds its risk
set as ``{entry < t <= exit}``.

On *this* pack, delayed entry turns out not to occur: every loan has an
age-0 row. 1,528 loans nonetheless have a calendar-first row at age 7, 10, ...
because their age-0 row carries a corrupted ``reporting_month`` -- the Phase 1
"Time Travel" defect. Reading entry off the calendar-first row would invent
1,528 left-truncated loans out of a data-quality artifact, so survival times
are taken on the **age axis** (min/max ``loan_age_months``), never the calendar
axis, and the disagreement between the two orderings is counted and reported
rather than silently absorbed. The truncation machinery stays in place because
a real servicing extract routinely does start mid-life.

Zombie rows
-----------
``Default`` and ``Prepaid`` are absorbing: a loan emits one row in that state
and none afterwards. Where a later active row exists it is the Phase 1
"Zombie Loan" defect, not a resurrection. The event is therefore read from the
**first** absorbing row and every row after it is dropped. Taking the last row
instead -- the obvious groupby -- silently reclassifies 691 loans in this pack
from resolved to still-active.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config


@dataclass
class SurvivalFrame:
    """
    One row per loan, plus the bookkeeping that explains how it was built.

    Attributes
    ----------
    data:
        Columns include ``entry_month`` (left-truncation time), ``exit_month``
        (event or censoring time), ``event_code`` (0/1/2), ``exposure_months``
        and the origination-time covariates.
    censoring:
        Counts behind every decision above, rendered into the report.
    """

    data: pd.DataFrame
    censoring: dict = field(default_factory=dict)

    entry_col: str = "entry_month"
    duration_col: str = "exit_month"
    event_col: str = "event_code"

    def __len__(self) -> int:
        return len(self.data)

    def cause_indicator(self, cause: int) -> np.ndarray:
        """
        1 where this loan experienced ``cause``, 0 otherwise.

        This is the **cause-specific** view: the competing event is treated as
        censoring *for the purpose of estimating that cause's hazard*, which is
        correct for hazard estimation and wrong for cumulative incidence. The
        CIF is built separately, in :mod:`src.survival.baselines`.
        """
        return (self.data[self.event_col].to_numpy() == cause).astype(int)

    def subset(self, mask: pd.Series | np.ndarray) -> "SurvivalFrame":
        return SurvivalFrame(data=self.data.loc[mask].copy(), censoring=dict(self.censoring))


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def _drop_rows_after_absorption(panel: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Keep each loan's history up to and including its first absorbing row.

    Returns the trimmed panel, the number of rows dropped, and the number of
    loans affected.
    """
    absorbing = panel["current_status"].isin(config.ABSORBING_TO_EVENT)
    first_absorbing = (
        panel.loc[absorbing]
        .groupby(config.ID_COL)["loan_age_months"]
        .min()
        .rename("absorbing_age")
    )
    merged = panel.merge(first_absorbing, on=config.ID_COL, how="left")

    zombie = merged["absorbing_age"].notna() & (merged["loan_age_months"] > merged["absorbing_age"])
    n_rows = int(zombie.sum())
    n_loans = int(merged.loc[zombie, config.ID_COL].nunique())

    return merged.loc[~zombie].drop(columns=["absorbing_age"]), n_rows, n_loans


def build_survival_frame(
    panel: pd.DataFrame,
    static: pd.DataFrame | None = None,
    cutoff: pd.Timestamp | str | None = None,
    left_truncation: bool = True,
) -> SurvivalFrame:
    """
    Collapse the monthly panel into one duration/event record per loan.

    Parameters
    ----------
    cutoff:
        Observation cutoff. Loans still active in this month are
        *administratively* censored; loans that stop reporting earlier are
        recorded as lost to follow-up. Defaults to the panel's last month.
    left_truncation:
        Honour delayed entry. With ``False`` every loan is assumed observed
        from month 0, which is what most naive pipelines do and what biases
        early hazards downward. Kept as a switch so the effect is measurable
        rather than asserted -- see ``tests/test_survival_censoring.py``.
    """
    panel = panel.copy()
    panel[config.TIME_COL] = pd.to_datetime(panel[config.TIME_COL], errors="coerce")
    panel = panel.sort_values([config.ID_COL, config.TIME_COL])

    cutoff = pd.Timestamp(cutoff) if cutoff is not None else panel[config.TIME_COL].max()

    trimmed, zombie_rows, zombie_loans = _drop_rows_after_absorption(panel)

    grouped = trimmed.groupby(config.ID_COL, sort=False)
    records = grouped.agg(
        first_age=("loan_age_months", "min"),
        last_age=("loan_age_months", "max"),
        first_month=(config.TIME_COL, "min"),
        last_month=(config.TIME_COL, "max"),
        n_observed_months=(config.TIME_COL, "size"),
    ).reset_index()

    # How often do calendar order and age order disagree? Every such loan is a
    # Phase 1 "Time Travel" candidate; the count goes into the censoring report
    # so the choice of axis is visible rather than buried.
    calendar_first = trimmed.groupby(config.ID_COL, sort=False).head(1).set_index(config.ID_COL)
    age_order_anomalies = int(
        (records[config.ID_COL].map(calendar_first["loan_age_months"]).to_numpy()
         != records["first_age"].to_numpy()).sum()
    )

    # The outcome is read off the *earliest absorbing row by loan age*, not off
    # whichever row happens to come last in the file. Both the zombie defect
    # and the time-travel defect can put a non-absorbing row after an absorbing
    # one; neither is a resurrection.
    absorbing_rows = trimmed[trimmed["current_status"].isin(config.ABSORBING_TO_EVENT)]
    earliest_absorbing = (
        absorbing_rows.sort_values([config.ID_COL, "loan_age_months"])
        .groupby(config.ID_COL, sort=False)
        .head(1)
        .set_index(config.ID_COL)
    )
    records["final_status"] = records[config.ID_COL].map(earliest_absorbing["current_status"])
    records["event_code"] = (
        records["final_status"].map(config.ABSORBING_TO_EVENT).fillna(config.EVENT_CENSORED)
    ).astype(int)
    records["absorbing_age"] = records[config.ID_COL].map(earliest_absorbing["loan_age_months"])

    records["entry_month"] = records["first_age"].astype(float) if left_truncation else 0.0
    # Event loans exit at the event; censored loans exit at their last observation.
    records["exit_month"] = (
        records["absorbing_age"].fillna(records["last_age"]).astype(float)
    )

    # --- degenerate exposure ----------------------------------------------
    # A record whose entry equals its exit contributes no exposure. For an
    # event, give it the one month it was observed for so the event is not
    # thrown away; for a censored record there is nothing to keep -- a
    # censoring time equal to the entry time adds no information to any
    # risk-set estimator, and lifelines rejects it outright.
    degenerate = records["exit_month"] <= records["entry_month"]
    events_bumped = int((degenerate & (records["event_code"] != config.EVENT_CENSORED)).sum())
    records.loc[degenerate & (records["event_code"] != config.EVENT_CENSORED), "exit_month"] = (
        records["entry_month"] + 1.0
    )

    drop_mask = degenerate & (records["event_code"] == config.EVENT_CENSORED)
    censored_dropped = int(drop_mask.sum())
    records = records.loc[~drop_mask].copy()

    records["exposure_months"] = records["exit_month"] - records["entry_month"]

    # --- censoring taxonomy ------------------------------------------------
    is_event = records["event_code"] != config.EVENT_CENSORED
    at_cutoff = records["last_month"] >= cutoff
    records["censoring_type"] = np.select(
        [is_event, at_cutoff],
        ["event", "administrative"],
        default="lost_to_followup",
    )

    if static is not None and not static.empty:
        records = _attach_covariates(records, static, trimmed)

    censoring = {
        "loans": int(len(records)),
        "observation_cutoff": f"{cutoff:%Y-%m}",
        "events_default": int((records["event_code"] == config.EVENT_DEFAULT).sum()),
        "events_prepaid": int((records["event_code"] == config.EVENT_PREPAID).sum()),
        "censored_total": int((records["event_code"] == config.EVENT_CENSORED).sum()),
        "censored_administrative": int((records["censoring_type"] == "administrative").sum()),
        "censored_lost_to_followup": int((records["censoring_type"] == "lost_to_followup").sum()),
        "left_truncated_loans": int((records["entry_month"] > 0).sum()),
        "left_truncation_enabled": left_truncation,
        "calendar_vs_age_order_anomalies": age_order_anomalies,
        "zombie_rows_dropped": zombie_rows,
        "zombie_loans_affected": zombie_loans,
        "degenerate_events_given_one_month": events_bumped,
        "degenerate_censored_dropped": censored_dropped,
        "total_exposure_months": float(records["exposure_months"].sum()),
        "median_follow_up_months": float(records["exposure_months"].median()),
    }
    censoring["censoring_rate"] = censoring["censored_total"] / max(censoring["loans"], 1)

    return SurvivalFrame(data=records.reset_index(drop=True), censoring=censoring)


def _attach_covariates(
    records: pd.DataFrame, static: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """
    Join origination-time covariates only.

    Anything measured after month 0 is excluded on principle: a survival
    model's covariate vector is what was known when the clock started. Using
    a later panel observation would be the survival-analysis form of the
    leakage Task 2 forbids.
    """
    wanted = [
        "credit_score", "credit_score_band", "ltv", "ltv_band", "dti", "dti_band",
        "original_balance", "original_term_months", "interest_rate",
        "loan_purpose", "property_type", "occupancy_type", "servicer_name",
        "state", "vintage_year", "vintage_quarter", "origination_month",
    ]
    available = [c for c in wanted if c in static.columns]
    merged = records.merge(
        static[[config.ID_COL, *available]].drop_duplicates(config.ID_COL),
        on=config.ID_COL,
        how="left",
    )

    # Backfill anything the static file lacks from the loan's own first panel row.
    missing = [c for c in wanted if c not in merged.columns and c in panel.columns]
    if missing:
        first_rows = panel.groupby(config.ID_COL, sort=False).head(1)
        merged = merged.merge(
            first_rows[[config.ID_COL, *missing]], on=config.ID_COL, how="left"
        )

    if "original_balance" in merged.columns:
        merged["log_original_balance"] = np.log(merged["original_balance"].clip(lower=1.0))
    if "origination_month" in merged.columns:
        merged["origination_month"] = pd.to_datetime(merged["origination_month"], errors="coerce")

    return merged


# --------------------------------------------------------------------------
# Time-aware split
# --------------------------------------------------------------------------
def vintage_split(
    frame: SurvivalFrame, cutoff: str | pd.Timestamp | None = None
) -> tuple[SurvivalFrame, SurvivalFrame]:
    """
    Split by **origination month**: older vintages train, newer ones test.

    This is the survival analogue of Task 2's time-aware split. Splitting on
    reporting month would cut individual loans in half, which a duration model
    cannot represent; splitting on vintage keeps each loan's history intact and
    still guarantees the holdout is made of loans the model has never seen,
    originated after everything it was fitted on.

    The holdout inevitably has shorter follow-up -- that is what makes it a
    forward-looking test, and it is exactly the situation where a model that
    mishandles censoring will look good and be wrong.
    """
    cutoff = pd.Timestamp(cutoff or config.SURVIVAL_TRAIN_VINTAGE_END)
    data = frame.data
    if "origination_month" not in data.columns:
        raise ValueError("vintage_split needs an 'origination_month' column")

    is_train = data["origination_month"] <= cutoff
    train, test = frame.subset(is_train), frame.subset(~is_train)
    for part, name in ((train, "train"), (test, "test")):
        part.censoring = dict(part.censoring)
        part.censoring["split"] = name
        part.censoring["vintage_cutoff"] = f"{cutoff:%Y-%m}"
    return train, test


def censoring_report(frame: SurvivalFrame) -> pd.DataFrame:
    """The censoring bookkeeping as a table, for the report and the model card."""
    rows = [{"metric": k, "value": v} for k, v in frame.censoring.items()]
    return pd.DataFrame(rows)


def outcome_summary(frame: SurvivalFrame, by: str | None = None) -> pd.DataFrame:
    """Event / censoring counts, optionally cut by a segment column."""
    data = frame.data.copy()
    data["outcome"] = data["event_code"].map(config.EVENT_LABELS)

    if by is None:
        counts = data["outcome"].value_counts().rename_axis("outcome").reset_index(name="loans")
        counts["share"] = counts["loans"] / counts["loans"].sum()
        return counts

    table = (
        data.pivot_table(index=by, columns="outcome", values=config.ID_COL, aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    table["loans"] = table.sum(axis=1)
    table["median_exposure_months"] = data.groupby(by)["exposure_months"].median()
    return table.reset_index()
