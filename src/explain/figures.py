"""
Explainability figures.

The SHAP library ships its own plots. They are not used here, for one reason:
this report sits alongside five others that share a validated palette, and a
figure that arrives in a different colour language reads as pasted in from
somewhere else. The values come from ``shap``; the rendering comes from
``src/viz.py``, so a reader who has learned the palette in the profiling report
can read this one without relearning it.

Form choices worth stating:

* **Beeswarm** for the global view rather than a bar chart alone. A bar says
  how much a feature matters; the beeswarm also says *which way* and *for which
  values*, and the sequential ramp on feature value is what carries that.
* **Waterfall** for the local view, running from the base rate to this loan's
  score, because a single prediction is a sum and the reader should see it add
  up.
* **Reliability** as predicted against observed with a bin-population histogram
  underneath, because a point on a calibration curve built from nine records is
  not a point anyone should read.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, viz

# Ordered categoricals whose colour can carry meaning on a beeswarm. A nominal
# category like `state` has no order, so colouring it by an arbitrary code
# would invent a gradient that is not there -- those stay neutral grey.
ORDINAL_CATEGORIES = {
    "credit_score_band": list(config.CREDIT_SCORE_BANDS),
    "ltv_band": list(config.LTV_BANDS),
    "dti_band": list(config.DTI_BANDS),
    "document_status": list(config.DOC_STATUS_ORDER),
    "current_status": list(config.STATUS_ORDER),
}


def _colour_scale(column: pd.Series, name: str) -> np.ndarray | None:
    """
    A numeric reading of a column for the beeswarm's colour ramp, or None.

    Numeric columns use their own values; known ordinal categoricals use their
    declared order; nominal categoricals return None and are drawn neutral.
    """
    order = ORDINAL_CATEGORIES.get(name)
    if order is not None:
        lookup = {level: i for i, level in enumerate(order)}
        mapped = column.astype(str).map(lookup)
        return mapped.to_numpy(dtype=float) if mapped.notna().any() else None

    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
    return values if np.isfinite(values).sum() > 1 and np.nanstd(values) > 0 else None

RAISES = viz.CATEGORICAL[1]  # orange: pushes risk up
LOWERS = viz.CATEGORICAL[0]  # blue: pushes risk down


def _pretty(name: str) -> str:
    return str(name).replace("_", " ")


# --------------------------------------------------------------------------
# Global
# --------------------------------------------------------------------------
def plot_beeswarm(
    result, outdir: Path, top_n: int = 15, max_points: int = 4000, seed: int = 0
) -> Path | None:
    """
    SHAP value against feature, coloured by the feature's own value.

    Reading it: horizontal spread is impact on the prediction; colour is
    whether that impact came from a high or low value of the feature. A band
    that runs dark-on-the-right means "high values of this feature raise risk".
    """
    if result is None or len(result) == 0:
        return None

    importance = np.abs(result.values).mean(axis=0)
    order = np.argsort(-importance)[:top_n][::-1]

    rng = np.random.default_rng(seed)
    rows = (
        rng.choice(len(result), size=max_points, replace=False)
        if len(result) > max_points
        else np.arange(len(result))
    )

    viz.style()
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(order) + 2.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("shap_seq", viz.SEQUENTIAL)

    for position, feature_index in enumerate(order):
        shap_column = result.values[rows, feature_index]
        name = result.feature_names[feature_index]
        raw = _colour_scale(result.X.iloc[rows, feature_index], name)

        # Rank-normalise the colour so one extreme value cannot flatten the
        # whole ramp; a column with no meaningful order goes neutral grey.
        if raw is not None:
            ranks = pd.Series(raw).rank(pct=True).to_numpy()
            colours = cmap(np.nan_to_num(ranks, nan=0.5))
        else:
            colours = np.tile(matplotlib.colors.to_rgba(viz.INK_MUTED), (len(shap_column), 1))

        jitter = rng.uniform(-0.18, 0.18, size=len(shap_column))
        ax.scatter(shap_column, position + jitter, s=6, c=colours, alpha=0.6, linewidths=0)

    ax.axvline(0.0, color=viz.AXIS, linewidth=1.2, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_pretty(result.feature_names[i]) for i in order], fontsize=8)
    ax.set_ylim(-0.7, len(order) - 0.3)
    viz.finish(ax, "", xlabel="SHAP value (impact on the model's log-odds)", grid_axis="x")

    mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    bar = fig.colorbar(mappable, ax=ax, fraction=0.02, pad=0.02, ticks=[0, 1])
    bar.ax.set_yticklabels(["low", "high"], fontsize=7.5)
    bar.set_label("feature value (percentile)", fontsize=8, color=viz.INK_SECONDARY)
    bar.outline.set_visible(False)

    viz.suptitle(
        fig,
        f"What drives the {result.label} model",
        f"One point per loan ({len(rows):,} sampled). Right of the line raises the predicted "
        "risk; colour is whether that came from a high or low value of the feature.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return viz.save(fig, outdir / f"shap_beeswarm_{result.label}.png")


def plot_global_importance(importances: dict, outdir: Path, top_n: int = 14) -> Path | None:
    """Mean absolute SHAP per feature, one panel per model, for side-by-side reading."""
    usable = {label: table for label, table in importances.items() if table is not None and not table.empty}
    if not usable:
        return None

    viz.style()
    fig, axes = plt.subplots(
        1, len(usable), figsize=(5.2 * len(usable), 0.34 * top_n + 2.6), squeeze=False
    )

    for index, (label, table) in enumerate(usable.items()):
        ax = axes[0][index]
        subset = table.head(top_n).sort_values("mean_abs_shap")
        positions = np.arange(len(subset))
        # Colour by the direction the feature usually pushes, so the chart
        # carries sign as well as magnitude.
        colours = [RAISES if v > 0 else LOWERS for v in subset["mean_signed_shap"]]

        ax.barh(positions, subset["mean_abs_shap"].to_numpy(), color=colours, height=0.62)
        ax.set_yticks(positions)
        ax.set_yticklabels([_pretty(f) for f in subset["feature"]], fontsize=8)
        for position, value, share in zip(positions, subset["mean_abs_shap"], subset["share"]):
            ax.annotate(
                f"{share:.0%}", xy=(value, position), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7.5, color=viz.INK_SECONDARY,
            )
        ax.set_xlim(0, subset["mean_abs_shap"].max() * 1.18)
        viz.finish(ax, label, xlabel="Mean |SHAP|", grid_axis="x")

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=RAISES, label="raises risk on average"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=LOWERS, label="lowers risk on average"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))

    viz.suptitle(fig, "Global feature importance", "Share labels are of total mean |SHAP|.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return viz.save(fig, outdir / "shap_global_importance.png")


# --------------------------------------------------------------------------
# Local
# --------------------------------------------------------------------------
def plot_waterfall(explanation: pd.DataFrame, outdir: Path, filename: str, subtitle: str = "") -> Path | None:
    """
    One loan, from the portfolio base rate to its own prediction.

    Drawn in log-odds because that is the space the contributions are additive
    in; the probability equivalent of each end is annotated so a reader who
    thinks in rates is not stranded.
    """
    if explanation is None or explanation.empty:
        return None

    base = explanation.attrs.get("base_value", 0.0)
    other = explanation.attrs.get("other_features", 0.0)
    total = explanation.attrs.get("log_odds", base + explanation["shap_value"].sum())

    steps = explanation.iloc[::-1].copy()  # smallest at the bottom of the plot
    labels = [f"{_pretty(r.feature)} = {_format(r.feature_value)}" for r in steps.itertuples()]
    values = steps["shap_value"].to_numpy()

    if abs(other) > 1e-9:
        labels = ["all other features"] + labels
        values = np.concatenate([[other], values])

    viz.style()
    fig, ax = plt.subplots(figsize=(11.0, 0.42 * len(values) + 3.0))

    running = base
    for position, value in enumerate(values):
        colour = RAISES if value > 0 else LOWERS
        ax.barh(position, value, left=running, color=colour, height=0.6)
        ax.annotate(
            f"{value:+.2f}",
            xy=(running + value, position),
            xytext=(5 if value > 0 else -5, 0), textcoords="offset points",
            ha="left" if value > 0 else "right", va="center",
            fontsize=7.5, color=viz.INK_SECONDARY,
        )
        running += value

    ax.axvline(base, color=viz.AXIS, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        f"base {base:.2f}  ({_probability(base):.1%})",
        xy=(base, len(values) - 0.3), xytext=(4, 6), textcoords="offset points",
        fontsize=7.5, color=viz.INK_MUTED,
    )
    ax.axvline(total, color=viz.INK_PRIMARY, linewidth=1.4, zorder=1)
    ax.annotate(
        f"this loan {total:.2f}  ({_probability(total):.1%})",
        xy=(total, -0.9), xytext=(4, 0), textcoords="offset points",
        fontsize=8, color=viz.INK_PRIMARY, fontweight="medium",
    )

    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-1.4, len(values) - 0.3)
    viz.finish(ax, "", xlabel="Model log-odds", grid_axis="x")

    viz.suptitle(fig, "Why this loan scored the way it did", subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return viz.save(fig, outdir / filename)


def _format(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    return str(value)


def _probability(log_odds: float) -> float:
    return float(1.0 / (1.0 + np.exp(-log_odds)))


# --------------------------------------------------------------------------
# Reliability and confidence
# --------------------------------------------------------------------------
def plot_reliability(tables: dict, outdir: Path) -> Path | None:
    """Predicted against observed, with the bin populations underneath."""
    usable = {label: table for label, table in tables.items() if table is not None and not table.empty}
    if not usable:
        return None

    viz.style()
    fig, axes = plt.subplots(
        2, len(usable), figsize=(5.0 * len(usable), 5.8), squeeze=False,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for index, (label, table) in enumerate(usable.items()):
        top, bottom = axes[0][index], axes[1][index]
        limit = max(float(table["mean_predicted"].max()), float(table["observed_rate"].max())) * 1.1

        top.plot([0, limit], [0, limit], color=viz.AXIS, linewidth=1.2, zorder=1,
                 label="perfect calibration")
        top.plot(
            table["mean_predicted"], table["observed_rate"],
            color=viz.CATEGORICAL[0], marker="o", markersize=7,
            markeredgecolor=viz.SURFACE, markeredgewidth=1.5, zorder=3, label=label,
        )
        top.set_xlim(0, limit)
        top.set_ylim(0, limit)
        viz.percent_axis(top, which="x")
        viz.percent_axis(top, which="y")
        viz.finish(top, label, xlabel="", ylabel="Observed event rate", grid_axis="y")
        top.grid(axis="x")
        top.legend(loc="upper left")

        bottom.bar(
            table["mean_predicted"], table["records"],
            width=max(limit / max(len(table), 1) * 0.7, 1e-3),
            color=viz.SEQUENTIAL[3],
        )
        bottom.set_xlim(0, limit)
        bottom.set_yscale("log")
        viz.percent_axis(bottom, which="x")
        viz.finish(bottom, "", xlabel="Predicted probability", ylabel="Records")

    viz.suptitle(
        fig,
        "Reliability: does a predicted rate mean what it says?",
        "Points on the diagonal are well calibrated. The histogram shows how many records "
        "sit behind each point -- a point built on a handful is not a finding.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return viz.save(fig, outdir / "reliability.png")


# --------------------------------------------------------------------------
# Errors and disparity
# --------------------------------------------------------------------------
def plot_error_rates(group_table: pd.DataFrame, outdir: Path, segment: str) -> Path | None:
    """
    False positive and false negative rate side by side, per group.

    Both, because they trade off: a segment can look clean on one and carry all
    the cost on the other, and showing only the flattering one is how a
    governance report becomes a marketing document.
    """
    if group_table is None or group_table.empty:
        return None
    subset = group_table[group_table["segment"] == segment]
    if subset.empty or subset["group"].nunique() > 14:
        return None

    subset = subset.sort_values("false_positive_rate", ascending=True)
    positions = np.arange(len(subset))

    viz.style()
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(subset) + 2.6))
    height = 0.38

    ax.barh(positions + height / 2, subset["false_positive_rate"], height=height,
            color=RAISES, label="false positive rate")
    ax.barh(positions - height / 2, subset["false_negative_rate"], height=height,
            color=LOWERS, label="false negative rate")

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{g}  (n={int(n):,})" for g, n in zip(subset["group"], subset["records"])], fontsize=8
    )
    viz.percent_axis(ax, which="x")
    viz.finish(ax, "", xlabel="Rate", grid_axis="x")
    ax.legend(loc="lower right")

    viz.suptitle(
        fig,
        f"Where the model is wrong, by {_pretty(segment)}",
        "False positives are the cost borne by borrowers who performed; false negatives are "
        "the risk the lender absorbs.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return viz.save(fig, outdir / f"error_rates_{segment}.png")
