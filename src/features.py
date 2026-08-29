"""
Phase 2 -- feature engineering for the loan performance panel (feeds Task 2).

Three families of feature are built here:

* **Static** -- attributes fixed at origination (credit score, LTV, DTI, term,
  purpose, geography). Joined from ``loan_static_attributes.csv`` where the
  panel only carries the banded version.
* **Contemporaneous** -- the state of the loan *as at* the reporting month
  (balance, days past due, status, document status).
* **Rolling / path-dependent** -- what the loan's own history up to and
  including the reporting month says about it: delinquency trend, paydown
  speed, status churn, servicer transfers, time since last delinquency.

Leakage guardrails, stated explicitly because Task 2 is graded on them
-----------------------------------------------------------------------
1. Every rolling window is **backward-looking and inclusive of month t only**.
   No ``shift(-k)``, no ``rolling(...).shift(-1)``, no reverse cumulative
   anywhere in this module.
2. Rolling windows are computed on a **complete monthly grid** per loan, so a
   "3-month window" is three *calendar* months even where the panel has a
   missing month. Position-based windows would silently reach further back
   than advertised on gapped loans.
3. The label columns, the Task 4 exception labels, ``loss_severity_band``
   (populated only on an absorbing row) and the contemporaneous
   ``default_flag`` / ``prepayment_flag`` are hard-excluded via
   ``config.FORBIDDEN_FEATURES`` and the exclusion is *asserted*, not assumed
   -- see :func:`assert_no_leaky_features`.
4. Cross-sectional features (e.g. rate spread versus the market) are computed
   within a single reporting month, so they use no information from the
   future either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

# Windows, in calendar months, used by the rolling feature block.
ROLLING_WINDOWS = (3, 6, 12)
LAGS = (1, 3, 6)


# --------------------------------------------------------------------------
# Feature specification
# --------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    """
    Which columns a model may consume, split by dtype family.

    ``baseline_*`` is the deliberately thin set given to the baseline model:
    the handful of fields a credit analyst would reach for without any feature
    engineering at all. The contrast with the full set is the point of the
    baseline-versus-improved comparison.
    """

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    baseline_numeric: list[str] = field(default_factory=list)
    baseline_categorical: list[str] = field(default_factory=list)

    @property
    def full(self) -> list[str]:
        return self.numeric + self.categorical

    @property
    def baseline(self) -> list[str]:
        return self.baseline_numeric + self.baseline_categorical

    def columns_for(self, variant: str) -> list[str]:
        return self.baseline if variant == "baseline" else self.full

    def split_for(self, variant: str) -> tuple[list[str], list[str]]:
        if variant == "baseline":
            return self.baseline_numeric, self.baseline_categorical
        return self.numeric, self.categorical


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _period(months: pd.Series) -> pd.Series:
    """Calendar month as a monotone integer, so lags are exact across gaps."""
    dt = pd.to_datetime(months, errors="coerce")
    return dt.dt.year * 12 + dt.dt.month


def _ordinal(series: pd.Series, order: list[str]) -> pd.Series:
    lookup = {value: i for i, value in enumerate(order)}
    return series.map(lookup).astype("float64")


def _monthly_grid(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Reindex the panel onto a gap-free (loan_id, period) grid.

    Rolling windows are then position-based on the grid, which makes them
    calendar-exact. Months absent from the source panel come back as NaN and
    are skipped by ``min_periods=1`` aggregations rather than shifting the
    window's reach.
    """
    bounds = panel.groupby(config.ID_COL)["period"].agg(["min", "max"])
    spans = (bounds["max"] - bounds["min"] + 1).astype(int)
    loans = np.repeat(bounds.index.to_numpy(), spans)
    offsets = np.concatenate([np.arange(n) for n in spans]) if len(spans) else np.array([])
    periods = np.repeat(bounds["min"].to_numpy(), spans) + offsets

    grid = pd.DataFrame({config.ID_COL: loans, "period": periods.astype("int64")})
    return grid.merge(
        panel[[config.ID_COL, "period", *columns]],
        on=[config.ID_COL, "period"],
        how="left",
    )


def _align(grouped: pd.Series, like: pd.DataFrame) -> np.ndarray:
    """
    Flatten a groupby-rolling result back onto the frame's own row order.

    ``groupby(...).rolling(...)`` returns a (group, position) MultiIndex whose
    concatenation order is not contractually the frame's order; dropping the
    group level and reindexing makes the alignment explicit instead of lucky.
    """
    if isinstance(grouped.index, pd.MultiIndex):
        grouped = grouped.reset_index(level=0, drop=True)
    return grouped.reindex(like.index).to_numpy()


def _expected_balance(
    original_balance: pd.Series,
    annual_rate_pct: pd.Series,
    term_months: pd.Series,
    age_months: pd.Series,
) -> pd.Series:
    """
    Scheduled amortised balance under a level-payment mortgage.

    ``B_a = P * ((1+r)^n - (1+r)^a) / ((1+r)^n - 1)``

    The gap between this and the reported balance is the single most
    informative engineered feature for both prepayment (paying down faster
    than schedule) and servicing error (balance discrepancy).
    """
    r = (annual_rate_pct.astype("float64") / 100.0) / 12.0
    n = term_months.astype("float64")
    a = age_months.astype("float64").clip(lower=0)
    a = np.minimum(a, n)

    with np.errstate(over="ignore", invalid="ignore"):
        growth_n = np.power(1.0 + r, n)
        growth_a = np.power(1.0 + r, a)
        scheduled = original_balance * (growth_n - growth_a) / (growth_n - 1.0)

    # Zero-rate loans amortise linearly; the formula above is undefined there.
    linear = original_balance * (1.0 - a / n.replace(0, np.nan))
    return pd.Series(np.where(r > 0, scheduled, linear), index=original_balance.index)


# --------------------------------------------------------------------------
# Ingestion / assembly
# --------------------------------------------------------------------------
def attach_static(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """
    Join origination attributes the panel does not already carry.

    The panel ships the *banded* credit score / LTV / DTI; the static file
    carries the underlying continuous values, which a gradient booster can
    split far more finely than a six-level band.
    """
    if static is None or static.empty:
        return panel

    wanted = [
        "credit_score",
        "ltv",
        "dti",
        "original_term_months",
        "vintage_quarter",
    ]
    available = [c for c in wanted if c in static.columns and c not in panel.columns]
    if not available:
        return panel

    return panel.merge(
        static[[config.ID_COL, *available]].drop_duplicates(config.ID_COL),
        on=config.ID_COL,
        how="left",
    )


def attach_dq_score(panel: pd.DataFrame, dq_scores: pd.DataFrame | None) -> pd.DataFrame:
    """
    Join the Phase 1 record-level data-quality score, if it has been produced.

    It is a legitimate feature -- a record the profiler distrusts is a record
    whose reported status may be wrong -- and it is computed from month t's own
    contents only, so it carries no forward information.
    """
    if dq_scores is None or dq_scores.empty or "dq_score" not in dq_scores.columns:
        return panel

    keys = [config.ID_COL, config.TIME_COL]
    if not set(keys).issubset(dq_scores.columns):
        return panel

    scores = dq_scores[[*keys, "dq_score"]].copy()
    scores[config.TIME_COL] = pd.to_datetime(scores[config.TIME_COL], errors="coerce")
    scores = scores.drop_duplicates(keys)

    merged = panel.merge(scores, on=keys, how="left")
    return merged


# --------------------------------------------------------------------------
# Feature blocks
# --------------------------------------------------------------------------
def add_contemporaneous_features(panel: pd.DataFrame) -> pd.DataFrame:
    """State of the loan as at month t. No history, no future."""
    df = panel.copy()

    df["status_ordinal"] = _ordinal(df["current_status"], config.STATUS_ORDER)
    if "document_status" in df.columns:
        df["doc_status_ordinal"] = _ordinal(df["document_status"], config.DOC_STATUS_ORDER)

    df["balance_ratio"] = df["current_balance"] / df["original_balance"].replace(0, np.nan)

    term = df.get("original_term_months")
    if term is None:
        term = df["loan_age_months"] + df["remaining_term_months"]
        df["original_term_months"] = term
    df["term_progress"] = df["loan_age_months"] / term.replace(0, np.nan)

    expected = _expected_balance(
        df["original_balance"], df["interest_rate"], term, df["loan_age_months"]
    )
    df["expected_balance"] = expected
    df["balance_vs_expected"] = df["current_balance"] / expected.replace(0, np.nan)
    df["balance_gap_abs"] = df["current_balance"] - expected

    df["is_delinquent"] = (df["days_past_due"] > 0).astype("int8")

    # Calendar seasonality, encoded cyclically so December and January are
    # adjacent. The raw time index is deliberately *not* a feature: a tree
    # cannot extrapolate it past the training window.
    month_of_year = pd.to_datetime(df[config.TIME_COL]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month_of_year / 12)
    df["month_cos"] = np.cos(2 * np.pi * month_of_year / 12)

    return df


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Where the loan sits relative to everything else reporting the same month.

    Computed within a reporting month, so no future information is used. The
    rate spread is the refinance incentive in disguise, and it is the main
    driver of prepayment.
    """
    df = panel.copy()
    by_month = df.groupby(config.TIME_COL)

    df["market_rate_median"] = by_month["interest_rate"].transform("median")
    df["rate_spread"] = df["interest_rate"] - df["market_rate_median"]
    df["balance_pctile_in_month"] = by_month["current_balance"].rank(pct=True)

    return df


def add_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Path-dependent features from each loan's own history up to month t.

    Computed on a gap-free monthly grid (see :func:`_monthly_grid`) so every
    window means exactly what its name says in calendar months.
    """
    df = panel.copy()
    df["period"] = _period(df[config.TIME_COL])

    carried = ["days_past_due", "current_balance", "status_ordinal"]
    for optional in ("servicer_name", "source_system", "document_status", "modification_flag"):
        if optional in df.columns:
            carried.append(optional)

    grid = _monthly_grid(df, carried)
    by_loan = grid.groupby(config.ID_COL, sort=False)

    out = grid[[config.ID_COL, "period"]].copy()

    # --- delinquency trend -------------------------------------------------
    dpd = grid["days_past_due"]
    for window in ROLLING_WINDOWS:
        rolled = by_loan["days_past_due"].rolling(window, min_periods=1)
        out[f"dpd_max_{window}m"] = _align(rolled.max(), grid)
        out[f"dpd_mean_{window}m"] = _align(rolled.mean(), grid)
    for lag in LAGS:
        lagged = by_loan["days_past_due"].shift(lag)
        out[f"dpd_lag_{lag}m"] = lagged.to_numpy()
        out[f"dpd_delta_{lag}m"] = (dpd - lagged).to_numpy()

    # --- paydown speed -----------------------------------------------------
    balance = grid["current_balance"]
    for lag in LAGS:
        lagged = by_loan["current_balance"].shift(lag)
        out[f"paydown_{lag}m"] = (balance / lagged.replace(0, np.nan) - 1.0).to_numpy()

    # --- cumulative history ------------------------------------------------
    delinquent = (grid["days_past_due"] > 0).astype("float64")
    delinquent[grid["days_past_due"].isna()] = np.nan
    out["delinquent_months_to_date"] = (
        delinquent.groupby(grid[config.ID_COL], sort=False).cumsum().to_numpy()
    )

    # Months since the most recent delinquent month (0 if delinquent now,
    # NaN if the loan has never been delinquent).
    delinquent_period = grid["period"].where(grid["days_past_due"] > 0)
    last_delinquent = delinquent_period.groupby(grid[config.ID_COL], sort=False).cummax()
    out["months_since_delinquency"] = (grid["period"] - last_delinquent).to_numpy()

    status = grid["status_ordinal"]
    out["worst_status_to_date"] = (
        status.groupby(grid[config.ID_COL], sort=False).cummax().to_numpy()
    )
    changed = (status != by_loan["status_ordinal"].shift(1)) & status.notna()
    changed &= by_loan["status_ordinal"].shift(1).notna()
    out["status_changes_to_date"] = (
        changed.astype("float64").groupby(grid[config.ID_COL], sort=False).cumsum().to_numpy()
    )
    out["status_changed_this_month"] = changed.astype("float64").to_numpy()

    # --- servicing churn ---------------------------------------------------
    for column, name in (
        ("servicer_name", "servicer_transfers_to_date"),
        ("source_system", "source_changes_to_date"),
        ("document_status", "doc_status_changes_to_date"),
    ):
        if column not in grid.columns:
            continue
        current = grid[column]
        previous = by_loan[column].shift(1)
        switched = (current != previous) & current.notna() & previous.notna()
        out[name] = (
            switched.astype("float64").groupby(grid[config.ID_COL], sort=False).cumsum().to_numpy()
        )

    if "modification_flag" in grid.columns:
        modified = grid["modification_flag"].astype("float64")
        out["modified_ever"] = (
            modified.groupby(grid[config.ID_COL], sort=False).cummax().to_numpy()
        )
        modified_period = grid["period"].where(grid["modification_flag"].astype("boolean").fillna(False).astype(bool))
        last_modified = modified_period.groupby(grid[config.ID_COL], sort=False).cummax()
        out["months_since_modification"] = (grid["period"] - last_modified).to_numpy()

    # Grid rows that never existed in the panel are dropped by the inner join.
    merged = df.merge(out, on=[config.ID_COL, "period"], how="left")
    return merged.drop(columns=["period"])


# --------------------------------------------------------------------------
# Leakage audit
# --------------------------------------------------------------------------
def assert_no_leaky_features(columns: list[str]) -> None:
    """
    Hard stop if a label, a label-derived column, or an absorbing-row-only
    column has found its way into the feature list.

    Asserted rather than assumed: the disqualification criterion for Task 2 is
    a *silent* leak, and a silent leak is exactly what a convention-only rule
    produces the first time someone adds a column.
    """
    leaked = sorted(set(columns) & set(config.FORBIDDEN_FEATURES))
    if leaked:
        raise ValueError(
            f"Leaky columns present in the feature matrix: {leaked}. "
            "These are labels or label-derived fields (see config.FORBIDDEN_FEATURES)."
        )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_feature_matrix(
    panel: pd.DataFrame,
    static: pd.DataFrame | None = None,
    dq_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, FeatureSpec]:
    """
    Turn a raw monthly panel into a modelling frame plus its feature spec.

    Returns the frame with every engineered column appended (labels and ids
    retained for downstream splitting) and the :class:`FeatureSpec` naming the
    columns a model is allowed to see.
    """
    df = panel.copy()
    df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL], errors="coerce")
    df = df.sort_values([config.ID_COL, config.TIME_COL]).reset_index(drop=True)

    df = attach_static(df, static)
    df = add_contemporaneous_features(df)
    df = add_cross_sectional_features(df)
    df = add_rolling_features(df)
    df = attach_dq_score(df, dq_scores)

    spec = build_feature_spec(df)
    assert_no_leaky_features(spec.full)
    return df, spec


def build_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    """Name the modelling columns, keeping only those actually present."""

    def present(candidates: list[str]) -> list[str]:
        return [c for c in candidates if c in df.columns]

    numeric = present(
        [
            # static / origination
            "credit_score",
            "ltv",
            "dti",
            "original_balance",
            "original_term_months",
            "interest_rate",
            "vintage_year",
            # contemporaneous
            "loan_age_months",
            "remaining_term_months",
            "current_balance",
            "days_past_due",
            "status_ordinal",
            "doc_status_ordinal",
            "balance_ratio",
            "term_progress",
            "balance_vs_expected",
            "balance_gap_abs",
            "is_delinquent",
            "month_sin",
            "month_cos",
            # cross-sectional
            "rate_spread",
            "balance_pctile_in_month",
            # rolling
            *[f"dpd_max_{w}m" for w in ROLLING_WINDOWS],
            *[f"dpd_mean_{w}m" for w in ROLLING_WINDOWS],
            *[f"dpd_lag_{lag}m" for lag in LAGS],
            *[f"dpd_delta_{lag}m" for lag in LAGS],
            *[f"paydown_{lag}m" for lag in LAGS],
            "delinquent_months_to_date",
            "months_since_delinquency",
            "worst_status_to_date",
            "status_changes_to_date",
            "status_changed_this_month",
            "servicer_transfers_to_date",
            "source_changes_to_date",
            "doc_status_changes_to_date",
            "modified_ever",
            "months_since_modification",
            # data quality
            "dq_score",
        ]
    )

    categorical = present(
        [
            "credit_score_band",
            "ltv_band",
            "dti_band",
            "state",
            "loan_purpose",
            "property_type",
            "occupancy_type",
            "servicer_name",
            "current_status",
            "document_status",
            "source_system",
        ]
    )

    # The baseline sees only what is on the face of the record: no rolling
    # history, no amortisation schedule, no cross-sectional context.
    baseline_numeric = present(
        ["loan_age_months", "current_balance", "days_past_due", "interest_rate", "balance_ratio"]
    )
    baseline_categorical = present(["credit_score_band", "current_status"])

    return FeatureSpec(
        numeric=numeric,
        categorical=categorical,
        baseline_numeric=baseline_numeric,
        baseline_categorical=baseline_categorical,
    )


# --------------------------------------------------------------------------
# Feature dictionary
# --------------------------------------------------------------------------
# One entry per engineered column: what it is, which family it belongs to, and
# -- the part that matters for grading -- the information window it is computed
# over. "as-at t" means the column uses only month t's own record; "t-k..t"
# means a backward-looking window ending at t inclusive. No entry may reference
# a month after t; :func:`assert_no_leaky_features` and the truncation test in
# tests/test_leakage_controls.py enforce that claim rather than trusting it.
FEATURE_FAMILIES = {
    "static": "Fixed at origination; identical for every month of a loan's life.",
    "contemporaneous": "The state of the loan as at the reporting month.",
    "cross_sectional": "The loan's position relative to every other loan reporting that month.",
    "rolling": "Derived from the loan's own history up to and including month t.",
    "quality": "Phase 1 data-quality signal for the record itself.",
}

FEATURE_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    # name: (family, window, definition)
    "credit_score": ("static", "origination", "Borrower FICO at origination (500-850)."),
    "ltv": ("static", "origination", "Loan-to-value at origination, percent."),
    "dti": ("static", "origination", "Debt-to-income at origination, percent."),
    "original_balance": ("static", "origination", "Unpaid principal balance at origination, USD."),
    "original_term_months": ("static", "origination", "Contractual term at origination."),
    "interest_rate": ("static", "origination", "Annual fixed note rate, percent."),
    "vintage_year": ("static", "origination", "Calendar year of origination."),
    "credit_score_band": ("static", "origination", "Credit score bucket."),
    "ltv_band": ("static", "origination", "LTV bucket."),
    "dti_band": ("static", "origination", "DTI bucket."),
    "state": ("static", "origination", "US state of the mortgaged property."),
    "loan_purpose": ("static", "origination", "Purchase, rate/term refinance, or cash-out."),
    "property_type": ("static", "origination", "Single-family, condominium, or PUD."),
    "occupancy_type": ("static", "origination", "Primary residence, investment, or second home."),

    "loan_age_months": ("contemporaneous", "as-at t", "Months elapsed since origination."),
    "remaining_term_months": ("contemporaneous", "as-at t", "Contractual months remaining."),
    "current_balance": ("contemporaneous", "as-at t", "Unpaid principal balance, USD."),
    "days_past_due": ("contemporaneous", "as-at t", "Days delinquent at the reporting month."),
    "current_status": ("contemporaneous", "as-at t", "Performance state: Current / 30 / 60 / 90-DPD."),
    "document_status": ("contemporaneous", "as-at t", "Document file completeness."),
    "servicer_name": ("contemporaneous", "as-at t", "Institution servicing the loan this month."),
    "source_system": ("contemporaneous", "as-at t", "System of record that produced the row."),
    "status_ordinal": ("contemporaneous", "as-at t", "current_status encoded worst-last, so a tree can split on severity."),
    "doc_status_ordinal": ("contemporaneous", "as-at t", "document_status encoded Complete < Pending < Missing."),
    "balance_ratio": ("contemporaneous", "as-at t", "current_balance / original_balance: how much principal is left."),
    "term_progress": ("contemporaneous", "as-at t", "loan_age_months / original_term_months: position in the amortisation schedule."),
    "expected_balance": ("contemporaneous", "as-at t", "Scheduled balance under level-payment amortisation at this age."),
    "balance_vs_expected": ("contemporaneous", "as-at t", "current_balance / expected_balance. Below 1 means paying down faster than schedule; a large deviation is also the Balance Discrepancy defect signature."),
    "balance_gap_abs": ("contemporaneous", "as-at t", "current_balance - expected_balance, in USD."),
    "is_delinquent": ("contemporaneous", "as-at t", "1 where days_past_due > 0."),
    "month_sin": ("contemporaneous", "as-at t", "Cyclical encoding of the calendar month, so December and January are adjacent."),
    "month_cos": ("contemporaneous", "as-at t", "Second component of the cyclical month encoding."),

    "market_rate_median": ("cross_sectional", "month t", "Median note rate across every loan reporting in month t."),
    "rate_spread": ("cross_sectional", "month t", "interest_rate minus the month's median rate: the refinance incentive, and the main driver of prepayment."),
    "balance_pctile_in_month": ("cross_sectional", "month t", "Percentile rank of current_balance within month t."),

    "delinquent_months_to_date": ("rolling", "0..t", "Count of months with days_past_due > 0 so far."),
    "months_since_delinquency": ("rolling", "0..t", "Months since the most recent delinquent month; 0 if delinquent now, null if never."),
    "worst_status_to_date": ("rolling", "0..t", "Running maximum of status_ordinal."),
    "status_changes_to_date": ("rolling", "0..t", "Count of month-on-month status transitions so far."),
    "status_changed_this_month": ("rolling", "t-1..t", "1 where the status differs from last month."),
    "servicer_transfers_to_date": ("rolling", "0..t", "Count of servicer changes so far."),
    "source_changes_to_date": ("rolling", "0..t", "Count of source-system changes so far."),
    "doc_status_changes_to_date": ("rolling", "0..t", "Count of document-status changes so far."),
    "modified_ever": ("rolling", "0..t", "1 once the loan has received a loss-mitigation modification."),
    "months_since_modification": ("rolling", "0..t", "Months since the modification; null if never modified."),

    "dq_score": ("quality", "as-at t", "Phase 1 record-level data-quality score (100 = clean). A record the profiler distrusts is a record whose reported status may be wrong."),
}

# Generated names follow a template rather than being listed one by one.
FEATURE_PATTERNS: list[tuple[str, str, str, str]] = [
    # (prefix, suffix, family, definition template using {k})
    ("dpd_max_", "m", "rolling", "Maximum days_past_due over the last {k} calendar months, inclusive of t."),
    ("dpd_mean_", "m", "rolling", "Mean days_past_due over the last {k} calendar months, inclusive of t."),
    ("dpd_lag_", "m", "rolling", "days_past_due as at month t-{k}."),
    ("dpd_delta_", "m", "rolling", "Change in days_past_due between month t-{k} and month t."),
    ("paydown_", "m", "rolling", "Proportional change in current_balance between month t-{k} and month t."),
]


def describe_feature(name: str) -> tuple[str, str, str]:
    """(family, window, definition) for one feature name, template-expanded."""
    if name in FEATURE_DEFINITIONS:
        return FEATURE_DEFINITIONS[name]
    for prefix, suffix, family, template in FEATURE_PATTERNS:
        if name.startswith(prefix) and name.endswith(suffix):
            middle = name[len(prefix) : len(name) - len(suffix)]
            if middle.isdigit():
                return family, f"t-{middle}..t", template.format(k=middle)
    return "unclassified", "unknown", "No definition recorded."


def feature_dictionary(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """
    The feature dictionary: every modelling column with its family, information
    window, definition, coverage and whether the baseline model sees it.

    Generated from the *actual* feature matrix rather than maintained by hand,
    so it cannot drift away from what the models are trained on. A feature with
    no recorded definition shows up as ``unclassified``, which is the signal to
    document it rather than a silent omission.
    """
    baseline = set(spec.baseline)
    rows = []
    for name in spec.full:
        family, window, definition = describe_feature(name)
        series = df[name] if name in df.columns else pd.Series(dtype="float64")
        rows.append(
            {
                "feature": name,
                "family": family,
                "information_window": window,
                "dtype": "categorical" if name in spec.categorical else "numeric",
                "in_baseline": name in baseline,
                "pct_present": round(100 * float(series.notna().mean()), 2) if len(series) else float("nan"),
                "n_unique": int(series.nunique()) if len(series) else 0,
                "definition": definition,
            }
        )

    order = {name: i for i, name in enumerate(FEATURE_FAMILIES)}
    frame = pd.DataFrame(rows)
    frame["_f"] = frame["family"].map(order).fillna(99)
    return frame.sort_values(["_f", "feature"]).drop(columns=["_f"]).reset_index(drop=True)


def feature_dictionary_markdown(dictionary: pd.DataFrame, spec: FeatureSpec) -> str:
    """Render the dictionary as a report section, grouped by family."""
    lines = [
        "# Feature dictionary",
        "",
        f"{len(spec.full)} modelling features ({len(spec.numeric)} numeric, "
        f"{len(spec.categorical)} categorical). The baseline model sees "
        f"{len(spec.baseline)} of them; the improved model sees all of them.",
        "",
        "**Information window** is the guarantee that matters. `as-at t` uses only the "
        "reporting month's own record; `t-k..t` is a backward-looking window ending at "
        "month t inclusive; `month t` is cross-sectional across loans within the same "
        "month. No feature reads a month after t. That claim is enforced, not asserted: "
        "`features.assert_no_leaky_features` hard-fails on any label-derived column, and "
        "`tests/test_leakage_controls.py` rebuilds the matrix on a panel truncated after "
        "month t and requires every rolling feature at t to be unchanged.",
        "",
        "Rolling windows are computed on a gap-free monthly grid, so a \"3-month window\" "
        "is three *calendar* months even where the panel is missing a month.",
        "",
    ]
    for family, blurb in FEATURE_FAMILIES.items():
        subset = dictionary[dictionary["family"] == family]
        if subset.empty:
            continue
        lines.append(f"## {family.replace('_', '-')} ({len(subset)})")
        lines.append("")
        lines.append(f"_{blurb}_")
        lines.append("")
        display = subset[
            ["feature", "information_window", "dtype", "in_baseline", "pct_present", "definition"]
        ].rename(columns={"pct_present": "% present", "in_baseline": "baseline"})
        lines.append(display.to_markdown(index=False))
        lines.append("")

    leftover = dictionary[~dictionary["family"].isin(FEATURE_FAMILIES)]
    if not leftover.empty:
        lines.append("## Undocumented")
        lines.append("")
        lines.append(
            "These reached the feature matrix without a dictionary entry. Add one to "
            "`FEATURE_DEFINITIONS` in `src/features.py`."
        )
        lines.append("")
        lines.append(leftover[["feature", "dtype"]].to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)
