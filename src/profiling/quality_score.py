"""
Task 1g: record-level and batch-level data-quality scores.

Design:
  * Every defect signal (rule violation, date violation, outlier, missingness,
    staleness, source conflict) contributes a weighted penalty.
  * The raw penalty is mapped to a 0-100 score, 100 == clean.
  * The per-record breakdown is kept so a reviewer can see *why* a record
    scored badly -- which is what makes this usable in Task 4 and Task 7
    rather than an opaque number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def compute_record_scores(
    df: pd.DataFrame,
    rule_matrix: pd.DataFrame | None = None,
    date_matrix: pd.DataFrame | None = None,
    outlier_matrix: pd.DataFrame | None = None,
    stale_flags: pd.Series | None = None,
    conflict_flags: pd.Series | None = None,
    weights: dict[str, float] | None = None,
    exclude_missing_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns a frame indexed like `df` with the component counts, the total
    penalty, the 0-100 score, a quality band, and a human-readable reason
    string listing the specific defects on that record.
    """
    w = {**config.DQ_WEIGHTS, **(weights or {})}
    n = len(df)
    idx = df.index

    def _zeros() -> pd.Series:
        return pd.Series(0, index=idx, dtype=int)

    rule_hits = (
        rule_matrix.sum(axis=1).astype(int) if rule_matrix is not None and not rule_matrix.empty else _zeros()
    )
    date_hits = (
        date_matrix.sum(axis=1).astype(int) if date_matrix is not None and not date_matrix.empty else _zeros()
    )
    outlier_hits = (
        outlier_matrix.sum(axis=1).astype(int)
        if outlier_matrix is not None and not outlier_matrix.empty
        else _zeros()
    )

    # Columns that are blank by design (conditionally populated) must not be
    # penalised -- otherwise a business rule reads as a data defect and the
    # score collapses toward zero for the whole portfolio.
    excluded = set(exclude_missing_cols or []) | set(config.TARGET_COLS)
    critical = [c for c in config.CRITICAL_COLUMNS if c in df.columns and c not in excluded]
    non_critical = [c for c in df.columns if c not in critical and c not in excluded]
    missing_critical = df[critical].isna().sum(axis=1).astype(int) if critical else _zeros()
    missing_non_critical = (
        df[non_critical].isna().sum(axis=1).astype(int) if non_critical else _zeros()
    )

    stale = (
        stale_flags.reindex(idx).fillna(False).astype(int)
        if stale_flags is not None
        else _zeros()
    )
    conflicts = (
        conflict_flags.reindex(idx).fillna(False).astype(int)
        if conflict_flags is not None
        else _zeros()
    )

    penalty = (
        w["rule_violation"] * (rule_hits + date_hits)
        + w["outlier"] * outlier_hits
        + w["missing_critical"] * missing_critical
        + w["missing_non_critical"] * missing_non_critical
        + w["stale_record"] * stale
        + w["source_conflict"] * conflicts
    )

    # Map penalty -> score with an exponential decay: cheap, bounded, and
    # keeps a clean record at exactly 100 without a hand-tuned max penalty.
    score = 100.0 * np.exp(-penalty / 10.0)

    out = pd.DataFrame(
        {
            "n_rule_violations": rule_hits,
            "n_date_violations": date_hits,
            "n_outliers": outlier_hits,
            "n_missing_critical": missing_critical,
            "n_missing_non_critical": missing_non_critical,
            "is_stale": stale.astype(bool),
            "has_source_conflict": conflicts.astype(bool),
            "penalty": penalty.round(3),
            "dq_score": score.round(2),
        },
        index=idx,
    )

    out["dq_band"] = pd.cut(
        out["dq_score"],
        bins=[-0.1, 40, 70, 90, 100.1],
        labels=["critical", "poor", "fair", "good"],
    )

    out["dq_reasons"] = _build_reasons(rule_matrix, date_matrix, outlier_matrix, stale, conflicts, out)

    for key in (config.ID_COL, config.TIME_COL):
        if key in df.columns:
            out.insert(0, key, df[key])

    return out


def _build_reasons(
    rule_matrix: pd.DataFrame | None,
    date_matrix: pd.DataFrame | None,
    outlier_matrix: pd.DataFrame | None,
    stale: pd.Series,
    conflicts: pd.Series,
    scores: pd.DataFrame,
) -> pd.Series:
    """Human-readable defect list per record -- the reviewer-facing part."""
    parts: list[pd.Series] = []

    for matrix, prefix in ((rule_matrix, "rule"), (date_matrix, "date"), (outlier_matrix, "outlier")):
        if matrix is None or matrix.empty:
            continue
        names = {c: c.split("__", 1)[-1] for c in matrix.columns}
        labelled = matrix.apply(
            lambda row, _n=names, _p=prefix: [
                f"{_p}:{_n[c]}" for c in row.index if bool(row[c])
            ],
            axis=1,
        )
        parts.append(labelled)

    if parts:
        combined = parts[0]
        for extra in parts[1:]:
            combined = combined.combine(extra, lambda a, b: list(a) + list(b))
    else:
        combined = pd.Series([[] for _ in range(len(scores))], index=scores.index)

    combined = combined.combine(
        stale.astype(bool), lambda lst, flag: list(lst) + (["stale_record"] if flag else [])
    )
    combined = combined.combine(
        conflicts.astype(bool),
        lambda lst, flag: list(lst) + (["source_conflict"] if flag else []),
    )

    return combined.apply(lambda lst: "; ".join(lst[:8]) if lst else "clean")


def batch_summary(record_scores: pd.DataFrame) -> dict:
    """Batch-level data-quality score plus the headline composition."""
    if record_scores.empty:
        return {}

    scores = record_scores["dq_score"]
    band_counts = record_scores["dq_band"].value_counts(dropna=False).to_dict()

    return {
        "n_records": int(len(record_scores)),
        "batch_dq_score": round(float(scores.mean()), 2),
        "median_dq_score": round(float(scores.median()), 2),
        "p05_dq_score": round(float(scores.quantile(0.05)), 2),
        "pct_clean": round(100.0 * float((record_scores["penalty"] == 0).mean()), 2),
        "pct_critical": round(
            100.0 * float((record_scores["dq_band"] == "critical").mean()), 2
        ),
        "pct_poor_or_worse": round(
            100.0 * float(record_scores["dq_band"].isin(["critical", "poor"]).mean()), 2
        ),
        "n_with_rule_violation": int((record_scores["n_rule_violations"] > 0).sum()),
        "n_with_date_violation": int((record_scores["n_date_violations"] > 0).sum()),
        "n_stale": int(record_scores["is_stale"].sum()),
        "n_source_conflict": int(record_scores["has_source_conflict"].sum()),
        "band_counts": {str(k): int(v) for k, v in band_counts.items()},
    }


def batch_summary_by_segment(
    df: pd.DataFrame, record_scores: pd.DataFrame, segment_col: str
) -> pd.DataFrame:
    """
    Batch score sliced by servicer / vintage / state.

    This is the view that turns a portfolio-level number into an action:
    "servicer X's feed is the problem", not "the data is 92% clean".
    """
    if segment_col not in df.columns or record_scores.empty:
        return pd.DataFrame()

    joined = record_scores.copy()
    joined[segment_col] = df[segment_col].values

    grouped = joined.groupby(segment_col, dropna=False).agg(
        n_records=("dq_score", "size"),
        mean_dq_score=("dq_score", "mean"),
        pct_with_rule_violation=("n_rule_violations", lambda s: 100.0 * (s > 0).mean()),
        n_stale=("is_stale", "sum"),
        n_source_conflict=("has_source_conflict", "sum"),
    )
    return grouped.round(2).sort_values("mean_dq_score").reset_index()


def worst_records(
    df: pd.DataFrame, record_scores: pd.DataFrame, n: int = 25
) -> pd.DataFrame:
    """The N worst records with their reasons -- seeds Task 4's reviewer examples."""
    if record_scores.empty:
        return pd.DataFrame()

    cols = [c for c in (config.ID_COL, config.TIME_COL) if c in record_scores.columns]
    keep = cols + ["dq_score", "dq_band", "penalty", "dq_reasons"]
    keep = [c for c in keep if c in record_scores.columns]
    return record_scores.nsmallest(n, "dq_score")[keep].reset_index(drop=True)
