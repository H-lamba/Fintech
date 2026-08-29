"""Task 1c: outlier detection and invalid date relationships."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def iqr_outliers(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Classic Tukey fences per numeric column."""
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if s.empty or s.nunique() <= 1:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        rows.append(
            {
                "column": col,
                "method": f"IQR (k={k})",
                "lower_fence": round(float(lo), 4),
                "upper_fence": round(float(hi), 4),
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(100.0 * mask.mean(), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True)


def zscore_outliers(df: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """Robust z-score (median/MAD) -- less sensitive to the outliers it's hunting."""
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if s.empty or s.nunique() <= 1:
            continue
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            continue
        robust_z = 0.6745 * (df[col] - med) / mad
        mask = robust_z.abs() > threshold
        rows.append(
            {
                "column": col,
                "method": f"robust z (|z|>{threshold})",
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(100.0 * mask.mean(), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True)


def outlier_flag_matrix(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """
    Boolean matrix, one column per numeric field, marking IQR outliers.
    Feeds the record-level data-quality score.
    """
    flags = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col]
        clean = s.dropna()
        if clean.empty or clean.nunique() <= 1:
            continue
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        flags[f"outlier__{col}"] = ((s < q1 - k * iqr) | (s > q3 + k * iqr)).fillna(False)
    return flags


def date_relationship_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Invalid date relationships specific to loan panel data.

    Each check returns a boolean Series; we report the violation count and keep
    the mask so the record-level score can use it.
    """
    checks: dict[str, pd.Series] = {}
    cols = df.columns

    if {"origination_month", "reporting_month"} <= set(cols):
        checks["origination_after_reporting"] = (
            df["origination_month"] > df["reporting_month"]
        ).fillna(False)

    if {"last_updated_at", "reporting_month"} <= set(cols):
        checks["last_update_before_reporting_month"] = (
            df["last_updated_at"] < df["reporting_month"]
        ).fillna(False)

    if {"loan_age_months", "origination_month", "reporting_month"} <= set(cols):
        implied = (
            (df["reporting_month"].dt.year - df["origination_month"].dt.year) * 12
            + (df["reporting_month"].dt.month - df["origination_month"].dt.month)
        )
        checks["loan_age_inconsistent_with_dates"] = (
            (df["loan_age_months"] - implied).abs() > 1
        ).fillna(False)

    if "reporting_month" in cols:
        checks["reporting_month_in_future"] = (
            df["reporting_month"] > pd.Timestamp.today()
        ).fillna(False)

    rows = [
        {
            "check": name,
            "n_violations": int(mask.sum()),
            "pct_violations": round(100.0 * mask.mean(), 3),
        }
        for name, mask in checks.items()
    ]
    return pd.DataFrame(rows)


def date_violation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Same checks as above, returned as a per-record boolean matrix."""
    out = pd.DataFrame(index=df.index)
    cols = df.columns

    if {"origination_month", "reporting_month"} <= set(cols):
        out["date__origination_after_reporting"] = (
            df["origination_month"] > df["reporting_month"]
        ).fillna(False)

    if {"last_updated_at", "reporting_month"} <= set(cols):
        out["date__last_update_before_reporting"] = (
            df["last_updated_at"] < df["reporting_month"]
        ).fillna(False)

    if {"loan_age_months", "origination_month", "reporting_month"} <= set(cols):
        implied = (
            (df["reporting_month"].dt.year - df["origination_month"].dt.year) * 12
            + (df["reporting_month"].dt.month - df["origination_month"].dt.month)
        )
        out["date__loan_age_inconsistent"] = (
            (df["loan_age_months"] - implied).abs() > 1
        ).fillna(False)

    return out


def staleness_flags(df: pd.DataFrame, stale_days: int = config.STALE_DAYS) -> pd.Series:
    """
    A record whose servicer last touched it long before the reporting month is
    'stale' -- explicitly called out in the problem statement's servicer_updates
    description.
    """
    if not {"last_updated_at", "reporting_month"} <= set(df.columns):
        return pd.Series(False, index=df.index, name="stale_record")
    age_days = (df["reporting_month"] - df["last_updated_at"]).dt.days
    return (age_days > stale_days).fillna(False).rename("stale_record")


def impossible_reporting_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reporting months that predate every origination in the book.

    A performance row cannot describe a month before any loan existed. Rows
    dated earlier are corrupted timestamps, not early history -- and because
    they sit at the edge of the calendar they are easy to mistake for a thin
    but legitimate warm-up period. Surfacing them as their own finding is what
    turns "the data-quality score is low in 2017" into "every row dated 2017 is
    a defect".
    """
    if config.TIME_COL not in df.columns or "origination_month" not in df.columns:
        return pd.DataFrame()

    reporting = pd.to_datetime(df[config.TIME_COL], errors="coerce")
    origination = pd.to_datetime(df["origination_month"], errors="coerce")
    if reporting.isna().all() or origination.isna().all():
        return pd.DataFrame()

    earliest_origination = origination.min()
    impossible = reporting < earliest_origination
    if not impossible.any():
        return pd.DataFrame()

    affected = df.loc[impossible]
    return (
        affected.assign(_period=reporting[impossible].dt.to_period("M").astype(str))
        .groupby("_period")
        .agg(
            rows=(config.ID_COL, "size"),
            loans=(config.ID_COL, "nunique"),
            earliest_origination_in_book=(config.TIME_COL, lambda _: f"{earliest_origination:%Y-%m}"),
        )
        .reset_index()
        .rename(columns={"_period": "reporting_month"})
    )
