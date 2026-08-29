"""
Task 1e: correlations and highly dependent fields.

Numeric-numeric uses Spearman (monotone, robust to the skew that loan balances
always have). Categorical-categorical uses Cramer's V with the bias correction,
because plain chi-square inflates on high-cardinality fields like `state`.
Numeric-categorical uses the correlation ratio (eta).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def numeric_correlations(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(method=method).round(4)


def top_correlated_pairs(
    corr: pd.DataFrame, threshold: float = 0.7, top_n: int = 30
) -> pd.DataFrame:
    """Flatten the correlation matrix to the strongly-related pairs only."""
    if corr.empty:
        return pd.DataFrame(columns=["column_a", "column_b", "correlation"])
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = corr.loc[a, b]
            if pd.notna(val) and abs(val) >= threshold:
                rows.append({"column_a": a, "column_b": b, "correlation": round(float(val), 4)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.reindex(out["correlation"].abs().sort_values(ascending=False).index)
    return out.head(top_n).reset_index(drop=True)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V for two categorical series."""
    confusion = pd.crosstab(x, y)
    if confusion.size == 0 or confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.to_numpy().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))


def categorical_associations(
    df: pd.DataFrame, max_cardinality: int = 60, threshold: float = 0.3
) -> pd.DataFrame:
    """Pairwise Cramer's V across categorical columns of manageable cardinality."""
    cats = [
        c
        for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if 1 < df[c].nunique(dropna=True) <= max_cardinality
    ]
    rows = []
    for i, a in enumerate(cats):
        for b in cats[i + 1 :]:
            sub = df[[a, b]].dropna()
            if sub.empty:
                continue
            v = cramers_v(sub[a], sub[b])
            if pd.notna(v) and v >= threshold:
                rows.append({"column_a": a, "column_b": b, "cramers_v": round(v, 4)})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["column_a", "column_b", "cramers_v"])
    return out.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta: how much of a numeric field's variance a categorical field explains."""
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df.empty or df["cat"].nunique() < 2:
        return np.nan
    overall_mean = df["val"].mean()
    numerator = sum(
        len(g) * (g["val"].mean() - overall_mean) ** 2 for _, g in df.groupby("cat")
    )
    denominator = ((df["val"] - overall_mean) ** 2).sum()
    if denominator == 0:
        return np.nan
    return float(np.sqrt(numerator / denominator))


def mixed_associations(
    df: pd.DataFrame, max_cardinality: int = 60, threshold: float = 0.3
) -> pd.DataFrame:
    """Numeric-vs-categorical dependency via the correlation ratio."""
    cats = [
        c
        for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if 1 < df[c].nunique(dropna=True) <= max_cardinality
    ]
    nums = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].nunique() > 1]
    rows = []
    for c in cats:
        for n in nums:
            eta = correlation_ratio(df[c], df[n])
            if pd.notna(eta) and eta >= threshold:
                rows.append({"categorical": c, "numeric": n, "eta": round(eta, 4)})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["categorical", "numeric", "eta"])
    return out.sort_values("eta", ascending=False).reset_index(drop=True)


def redundancy_candidates(corr: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """
    Columns that duplicate another column almost exactly. Dropping one of each
    pair keeps tree models from splitting importance across twins -- and a
    near-perfect correlation with a target is a leakage smell worth checking.
    """
    if corr.empty:
        return []
    drop: set[str] = set()
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = corr.loc[a, b]
            if pd.notna(val) and abs(val) >= threshold and b not in drop:
                drop.add(b)
    return sorted(drop)
