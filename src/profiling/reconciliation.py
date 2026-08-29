"""
Source-conflict detection against servicer_updates.csv.

The problem statement is explicit that this second file carries "partial or
conflicting updates used for source conflict detection, stale record logic,
and reconciliation" -- so conflicts here are a first-class data-quality signal
and later become anomaly/exception evidence in Task 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def _join_keys(main: pd.DataFrame, updates: pd.DataFrame) -> list[str]:
    """Prefer loan_id + reporting_month; fall back to loan_id alone."""
    keys = [k for k in (config.ID_COL, config.TIME_COL) if k in main.columns and k in updates.columns]
    return keys or ([config.ID_COL] if config.ID_COL in main.columns and config.ID_COL in updates.columns else [])


def reconcile(
    main: pd.DataFrame,
    updates: pd.DataFrame,
    numeric_tolerance: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare overlapping fields between the panel and the servicer feed.

    Returns (summary_by_field, conflict_records).
    Numeric fields use a relative tolerance so float noise isn't reported as a
    conflict; categorical fields compare on a case-folded string.
    """
    if main.empty or updates.empty:
        return pd.DataFrame(), pd.DataFrame()

    keys = _join_keys(main, updates)
    if not keys:
        return pd.DataFrame(), pd.DataFrame()

    compare_cols = [
        c
        for c in updates.columns
        if c in main.columns and c not in keys and c not in ("source_system", "last_updated_at")
    ]
    if not compare_cols:
        return pd.DataFrame(), pd.DataFrame()

    merged = main.merge(
        updates[keys + compare_cols + [c for c in ("last_updated_at",) if c in updates.columns]],
        on=keys,
        how="inner",
        suffixes=("", "__servicer"),
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    conflict_mask_total = pd.Series(False, index=merged.index)

    for col in compare_cols:
        other = f"{col}__servicer"
        if other not in merged.columns:
            continue

        left, right = merged[col], merged[other]
        both_present = left.notna() & right.notna()

        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            denom = right.abs().where(right.abs() > 0, 1.0)
            differs = ((left - right).abs() / denom > numeric_tolerance) & both_present
        else:
            differs = (
                left.astype(str).str.strip().str.lower()
                != right.astype(str).str.strip().str.lower()
            ) & both_present

        differs = differs.fillna(False)
        merged[f"conflict__{col}"] = differs
        conflict_mask_total |= differs

        summary_rows.append(
            {
                "field": col,
                "n_compared": int(both_present.sum()),
                "n_conflicts": int(differs.sum()),
                "pct_conflicts": round(100.0 * differs.mean(), 4),
            }
        )

    merged["has_source_conflict"] = conflict_mask_total

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("n_conflicts", ascending=False).reset_index(drop=True)

    conflict_cols = keys + [c for c in merged.columns if c.startswith("conflict__")]
    detail_cols = []
    for col in compare_cols:
        if f"{col}__servicer" in merged.columns:
            detail_cols += [col, f"{col}__servicer"]

    conflicts = merged.loc[merged["has_source_conflict"], conflict_cols + detail_cols]
    return summary, conflicts.reset_index(drop=True)


def conflict_flags(main: pd.DataFrame, updates: pd.DataFrame) -> pd.Series:
    """
    Per-record boolean aligned to `main`'s index, for the data-quality score.
    """
    empty = pd.Series(False, index=main.index, name="source_conflict")
    if main.empty or updates.empty:
        return empty

    keys = _join_keys(main, updates)
    if not keys:
        return empty

    _, conflicts = reconcile(main, updates)
    if conflicts.empty:
        return empty

    conflicted_keys = conflicts[keys].drop_duplicates()
    conflicted_keys["_conflict"] = True
    marked = main[keys].merge(conflicted_keys, on=keys, how="left")
    return marked["_conflict"].fillna(False).astype(bool).set_axis(main.index).rename("source_conflict")


def duplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact duplicate panel rows on (loan_id, reporting_month) -- a loan should
    appear once per month, so anything else breaks the panel structure and will
    quietly corrupt a time-aware split.
    """
    keys = [k for k in (config.ID_COL, config.TIME_COL) if k in df.columns]
    if not keys or df.empty:
        return pd.DataFrame()
    dupes = df[df.duplicated(subset=keys, keep=False)]
    if dupes.empty:
        return pd.DataFrame()
    return dupes.sort_values(keys).reset_index(drop=True)
