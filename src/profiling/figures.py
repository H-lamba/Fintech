"""
Figures for the Data Intelligence Report.

Every chart here answers a question the tables can only answer by being read
carefully:

* *Where are the holes, and do they come in blocks?* -> missingness bar +
  segment heatmap. A block pattern means one upstream process dropped a group
  of fields, which is a different problem from scattered gaps.
* *What do the numeric fields actually look like, and where do the fences
  fall?* -> distributions with the Tukey bounds drawn on.
* *Which fields have moved between train and test?* -> PSI bars against the
  conventional bands.
* *How is data quality distributed, and is it concentrated anywhere?* -> score
  histogram plus the worst-scoring segment.

Chart forms follow ``src/viz.py``: sequential single-hue for magnitude,
reserved status colours for the PSI bands (always with a text label, never
hue alone), and recessive chrome throughout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, viz


def _shorten(labels, limit: int = 26) -> list[str]:
    return [str(x) if len(str(x)) <= limit else str(x)[: limit - 1] + "…" for x in labels]


# --------------------------------------------------------------------------
# 1. Missingness
# --------------------------------------------------------------------------
def plot_missingness(
    df: pd.DataFrame,
    missing_summary: pd.DataFrame,
    outdir: Path,
    segment_col: str = "current_status",
    age_col: str = "loan_age_months",
    top_n: int = 15,
) -> Path | None:
    """
    Left: missing rate per column. Right: the same columns cut by performance
    state, as a heatmap.

    The right panel is the one that matters. A column missing at 100% in one
    state and 0% everywhere else is *structural* -- a business rule, not a
    defect -- and penalising it would drag nearly every record below a passing
    quality score. The heatmap makes that pattern visible in one look instead
    of requiring a cross-tab per column.
    """
    columns = (
        missing_summary[missing_summary["n_missing"] > 0]
        .sort_values("pct_missing", ascending=False)
        .head(top_n)["column"]
        .tolist()
    )
    if not columns:
        return None

    viz.style()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.5, 0.42 * len(columns) + 2.6), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    # --- per-column rate ---------------------------------------------------
    rates = df[columns].isna().mean().sort_values()
    positions = np.arange(len(rates))
    left.barh(positions, rates.to_numpy(), color=viz.SEQUENTIAL[4], height=0.62)
    left.set_yticks(positions)
    left.set_yticklabels(_shorten(rates.index), fontsize=8)
    viz.percent_axis(left, which="x")
    viz.finish(left, "Missing rate by column", xlabel="Share of rows missing", grid_axis="x")
    for position, value in zip(positions, rates.to_numpy()):
        left.annotate(
            f"{value:.1%}", xy=(value, position), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=7.5, color=viz.INK_SECONDARY,
        )
    left.set_xlim(0, max(rates.max() * 1.18, 0.01))

    # --- by segment --------------------------------------------------------
    if segment_col in df.columns:
        matrix = df.groupby(segment_col)[columns].apply(lambda g: g.isna().mean())
    elif age_col in df.columns:
        buckets = pd.cut(df[age_col], [-1, 6, 12, 24, 36, 60, 10**6],
                         labels=["0-6m", "7-12m", "13-24m", "25-36m", "37-60m", "60m+"])
        matrix = df.groupby(buckets, observed=True)[columns].apply(lambda g: g.isna().mean())
    else:
        matrix = pd.DataFrame()

    if matrix.empty:
        right.set_visible(False)
    else:
        matrix = matrix[rates.index[::-1]]
        image = right.imshow(
            matrix.to_numpy(), aspect="auto", cmap=_sequential_cmap(), vmin=0, vmax=1
        )
        right.set_xticks(np.arange(matrix.shape[1]))
        right.set_xticklabels(_shorten(matrix.columns, 30), rotation=35, ha="right", fontsize=7.5)
        right.set_yticks(np.arange(matrix.shape[0]))
        right.set_yticklabels(_shorten(matrix.index), fontsize=8)
        # Hairline cell separators: a near-zero cell is deliberately close to the
        # surface colour, which loses the grid unless the cells are outlined.
        right.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
        right.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
        right.grid(which="minor", color=viz.SURFACE, linewidth=2.0)
        right.grid(which="major", visible=False)
        right.tick_params(which="minor", length=0)
        for spine in right.spines.values():
            spine.set_visible(False)
        right.set_title(
            f"Missing rate by {segment_col.replace('_', ' ')}",
            color=viz.INK_PRIMARY, loc="left", pad=10, fontweight="medium",
        )
        bar = fig.colorbar(image, ax=right, fraction=0.035, pad=0.02)
        bar.set_label("Share missing", fontsize=8, color=viz.INK_SECONDARY)
        bar.ax.tick_params(labelsize=7.5, color=viz.INK_MUTED)
        bar.outline.set_visible(False)

    viz.suptitle(
        fig,
        "Where the data is missing",
        "A column missing in one state and present in every other is structural, not a "
        "defect: it is excluded from the quality penalty.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return viz.save(fig, outdir / "missingness.png")


def _sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("project_sequential", [viz.SURFACE, *viz.SEQUENTIAL])


# --------------------------------------------------------------------------
# 2. Distributions
# --------------------------------------------------------------------------
def plot_distributions(
    df: pd.DataFrame, outdir: Path, columns: list[str] | None = None, bins: int = 40
) -> Path | None:
    """
    Small multiples of the key numeric fields, with the Tukey fences drawn on.

    The fences are the same 1.5x IQR bounds the outlier table counts against,
    so the reader can see *why* a column has a 4% outlier rate rather than
    taking the number on faith.
    """
    candidates = columns or [
        "current_balance", "original_balance", "interest_rate", "days_past_due",
        "loan_age_months", "remaining_term_months", "credit_score", "ltv", "dti",
    ]
    present = [
        c for c in candidates
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]
    if not present:
        return None

    viz.style()
    ncols = min(3, len(present))
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.9 * nrows), squeeze=False)

    for index, column in enumerate(present):
        ax = axes[index // ncols][index % ncols]
        values = df[column].dropna()
        ax.hist(values, bins=bins, color=viz.SEQUENTIAL[3], edgecolor=viz.SURFACE, linewidth=0.5)

        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        share_outside = float(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean())
        for fence in (q1 - 1.5 * iqr, q3 + 1.5 * iqr):
            if values.min() <= fence <= values.max():
                ax.axvline(fence, color=viz.STATUS["serious"], linewidth=1.4, linestyle=(0, (4, 3)))

        viz.finish(ax, f"{column}   {share_outside:.1%} outside fences", ylabel="Rows")
        ax.tick_params(axis="x", labelrotation=0)

    for index in range(len(present), nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)

    viz.suptitle(
        fig,
        "Numeric distributions, with Tukey fences",
        "Dashed lines are the 1.5x IQR bounds the outlier table counts against.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return viz.save(fig, outdir / "distributions.png")


# --------------------------------------------------------------------------
# 3. Train / test drift
# --------------------------------------------------------------------------
def plot_drift(drift_table: pd.DataFrame, outdir: Path, top_n: int = 18) -> Path | None:
    """
    PSI per feature against the conventional 0.10 / 0.25 bands.

    Drawn as a **dot plot on a log axis**, not bars. PSI here spans four orders
    of magnitude (0.000 to 4.2); on a linear bar chart every stable field
    collapses to a zero-length stub and the reader learns only that two fields
    are large. A bar has to run from zero to be honest, so the fix is to change
    the mark rather than the scale: position encodes the value, and the bands
    are shaded behind it.

    Band colour is a *status*, not a series identity, so it comes from the
    reserved status palette and always ships with the band name in the
    legend -- hue never carries the meaning alone.
    """
    if drift_table is None or drift_table.empty or "psi" not in drift_table.columns:
        return None

    table = drift_table.dropna(subset=["psi"]).nlargest(top_n, "psi").sort_values("psi")
    if table.empty:
        return None

    def band(psi: float) -> str:
        if psi >= config.PSI_MAJOR:
            return "critical"
        if psi >= config.PSI_MINOR:
            return "warning"
        return "good"

    values = table["psi"].to_numpy()
    colours = [viz.STATUS[band(v)] for v in values]

    viz.style()
    fig, ax = plt.subplots(figsize=(10.0, 0.36 * len(table) + 2.4))
    positions = np.arange(len(table))

    # A log axis cannot show an exact zero; floor at half the smallest non-zero
    # value so a perfectly stable field still plots, at the left edge.
    positive = values[values > 0]
    floor = float(positive.min() / 2) if len(positive) else 1e-4
    plotted = np.clip(values, floor, None)

    ax.set_xscale("log")
    ax.set_xlim(floor * 0.7, max(values.max() * 2.2, config.PSI_MAJOR * 3))

    ax.hlines(positions, floor * 0.7, plotted, color=viz.GRIDLINE, linewidth=1.4, zorder=1)
    ax.scatter(
        plotted, positions, s=70, color=colours, zorder=3,
        edgecolor=viz.SURFACE, linewidth=1.5,
    )

    for threshold, label, height in (
        (config.PSI_MINOR, "0.10  minor", len(table) - 0.2),
        (config.PSI_MAJOR, "0.25  major", len(table) - 1.1),
    ):
        ax.axvline(threshold, color=viz.AXIS, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(
            label, xy=(threshold, height), xytext=(5, 0), textcoords="offset points",
            fontsize=7.5, color=viz.INK_MUTED, va="center",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(_shorten(table["column"]), fontsize=8)
    ax.set_ylim(-0.8, len(table) + 0.4)
    for position, value, plot_x in zip(positions, values, plotted):
        ax.annotate(
            f"{value:.3f}", xy=(plot_x, position), xytext=(9, 0), textcoords="offset points",
            va="center", fontsize=7.5, color=viz.INK_SECONDARY,
        )

    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    viz.finish(ax, "", xlabel="Population stability index (log scale)", grid_axis="x")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=viz.STATUS[key], label=text)
        for key, text in (
            ("good", "No material drift (< 0.10)"),
            ("warning", "Minor drift (0.10-0.25)"),
            ("critical", "Major drift (> 0.25)"),
        )
    ]
    ax.legend(handles=handles, loc="lower right")

    viz.suptitle(
        fig,
        "Train vs. test drift",
        "PSI compares each field's test distribution against train, binned on train quantiles.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return viz.save(fig, outdir / "drift.png")


# --------------------------------------------------------------------------
# 4. Data-quality score
# --------------------------------------------------------------------------
def plot_quality_scores(
    record_scores: pd.DataFrame, outdir: Path, segment_col: str = "dq_band"
) -> Path | None:
    """
    Left: the score distribution. Right: mean score by reporting month.

    The time series is the one that changes decisions: a quality score that is
    flat is a stable feed, and one that steps down in a particular month points
    at a specific upstream change rather than at diffuse noise.
    """
    if record_scores is None or record_scores.empty or "dq_score" not in record_scores.columns:
        return None

    viz.style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.2))

    scores = record_scores["dq_score"].dropna()
    left.hist(scores, bins=40, color=viz.SEQUENTIAL[3], edgecolor=viz.SURFACE, linewidth=0.5)
    clean = float((scores >= 99.99).mean())
    left.axvline(float(scores.mean()), color=viz.INK_SECONDARY, linewidth=1.6, linestyle=(0, (4, 3)))
    left.annotate(
        f"mean {scores.mean():.1f}", xy=(scores.mean(), left.get_ylim()[1] * 0.92),
        xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=8, color=viz.INK_SECONDARY,
    )
    viz.finish(left, f"Record score distribution   {clean:.0%} defect-free",
               xlabel="Data-quality score (100 = clean)", ylabel="Records")

    time_col = config.TIME_COL if config.TIME_COL in record_scores.columns else None
    if time_col:
        by_month = (
            record_scores.assign(_m=pd.to_datetime(record_scores[time_col], errors="coerce"))
            .dropna(subset=["_m"])
            .groupby("_m")["dq_score"]
            .mean()
        )
        right.plot(by_month.index, by_month.to_numpy(), color=viz.CATEGORICAL[0])
        right.set_ylim(0, 100)
        viz.finish(right, "Mean score by reporting month", ylabel="Mean data-quality score")
        right.tick_params(axis="x", labelrotation=30)
    else:
        right.set_visible(False)

    viz.suptitle(
        fig,
        "Data quality",
        "Score is 100 * exp(-penalty/10); a clean record scores exactly 100.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return viz.save(fig, outdir / "quality_scores.png")


# --------------------------------------------------------------------------
# 5. Rule violations
# --------------------------------------------------------------------------
def plot_rule_violations(rule_summary: pd.DataFrame, outdir: Path, top_n: int = 18) -> Path | None:
    """Violation counts per rule -- what the rule engine actually caught, ranked."""
    if rule_summary is None or rule_summary.empty:
        return None

    count_col = next(
        (c for c in ("n_violations", "violations", "n_failed", "count") if c in rule_summary.columns),
        None,
    )
    name_col = next(
        (c for c in ("rule", "name", "rule_name", "description") if c in rule_summary.columns), None
    )
    if count_col is None or name_col is None:
        return None

    table = rule_summary[rule_summary[count_col] > 0].nlargest(top_n, count_col)
    if table.empty:
        return None
    table = table.sort_values(count_col)

    viz.style()
    fig, ax = plt.subplots(figsize=(10.0, 0.38 * len(table) + 2.2))
    positions = np.arange(len(table))
    ax.barh(positions, table[count_col].to_numpy(), color=viz.SEQUENTIAL[4], height=0.62)
    ax.set_yticks(positions)
    ax.set_yticklabels(_shorten(table[name_col], 44), fontsize=8)
    for position, value in zip(positions, table[count_col].to_numpy()):
        ax.annotate(
            f"{int(value):,}", xy=(value, position), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=7.5, color=viz.INK_SECONDARY,
        )
    ax.set_xlim(0, table[count_col].max() * 1.16)
    viz.finish(ax, "", xlabel="Rows violating the rule", grid_axis="x")

    viz.suptitle(
        fig,
        "Validation rule violations",
        "Organiser rules plus the project's own domain checks, ranked by rows affected.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return viz.save(fig, outdir / "rule_violations.png")
