"""Task 1a: column distribution profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per column: dtype, cardinality, missingness, and the distribution
    summary appropriate to the column's type.
    """
    rows = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        row: dict[str, object] = {
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100.0 * s.isna().mean(), 3) if n else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }

        if pd.api.types.is_bool_dtype(s):
            # Booleans satisfy is_numeric_dtype but have no meaningful
            # quantiles -- summarise them as a rate instead.
            row.update(
                {
                    "n_true": int(s.sum()),
                    "pct_true": round(100.0 * s.mean(), 3) if s.notna().any() else 0.0,
                }
            )
        elif pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            row.update(
                {
                    "mean": round(float(desc.get("mean", np.nan)), 4),
                    "std": round(float(desc.get("std", np.nan)), 4),
                    "min": float(desc.get("min", np.nan)),
                    "p01": float(s.quantile(0.01)) if s.notna().any() else np.nan,
                    "p25": float(desc.get("25%", np.nan)),
                    "p50": float(desc.get("50%", np.nan)),
                    "p75": float(desc.get("75%", np.nan)),
                    "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
                    "max": float(desc.get("max", np.nan)),
                    "skew": round(float(s.skew()), 4) if s.notna().sum() > 2 else np.nan,
                    "n_zero": int((s == 0).sum()),
                    "n_negative": int((s < 0).sum()),
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            row.update(
                {
                    "min": str(s.min()),
                    "max": str(s.max()),
                }
            )
        else:
            vc = s.value_counts(dropna=True)
            top = vc.head(5)
            row.update(
                {
                    "top_values": "; ".join(
                        f"{idx}={cnt} ({100.0 * cnt / n:.1f}%)" for idx, cnt in top.items()
                    )
                    if n
                    else "",
                    "is_constant": bool(row["n_unique"] <= 1),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def numeric_histograms(df: pd.DataFrame, bins: int = 20) -> dict[str, pd.DataFrame]:
    """Binned counts per numeric column, for plotting in the report."""
    out: dict[str, pd.DataFrame] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if s.empty or s.nunique() <= 1:
            continue
        counts, edges = np.histogram(s, bins=bins)
        out[col] = pd.DataFrame(
            {"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts}
        )
    return out
