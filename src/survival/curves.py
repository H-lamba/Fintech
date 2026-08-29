"""
Event-curve figures for the survival report.

Design notes (why the charts look the way they do)
--------------------------------------------------
* **Colour follows the entity, everywhere.** Default is blue ``#2a78d6``,
  prepayment is orange ``#eb6834``, in every figure in this report. A reader
  who learns the mapping once never has to re-read a legend. The pair passes
  the CVD and normal-vision separation gates on the light chart surface.
* **Segments are small multiples, not twelve lines on one axis.** Six credit
  bands times two causes is twelve curves; overlaid they are unreadable, and a
  six-step single-hue ramp cannot hold six distinguishable steps at the
  required lightness gaps. Faceting keeps the entity colours intact and makes
  the ordinal comparison a left-to-right scan.
* **Competing risks are drawn as a composition.** The stacked area is the
  honest picture: at every loan age the portfolio is exactly
  ``survived + defaulted + prepaid``, and the bands sum to 1 by construction.
* **Grid and axes recede**; the data is the only thing at full contrast.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402
from ..viz import (  # noqa: E402
    AXIS,
    CATEGORICAL,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    finish as _base_finish,
    percent_axis as _percent_axis,
    save as _save,
    style as _style,
    suptitle,
)
from .baselines import aalen_johansen_cif, cif_by_segment  # noqa: E402
from .dataset import SurvivalFrame  # noqa: E402

# Default is categorical slot 1, prepayment slot 2, in every figure in the
# project. A reader who learns the mapping once never re-reads a legend.
CAUSE_COLOR = {config.EVENT_DEFAULT: CATEGORICAL[0], config.EVENT_PREPAID: CATEGORICAL[1]}
CAUSE_LABEL = {config.EVENT_DEFAULT: "Default", config.EVENT_PREPAID: "Prepayment"}
SURVIVOR_FILL = GRIDLINE


def _finish(ax, title: str = "", xlabel: str = "Months on book", ylabel: str = "") -> None:
    _base_finish(ax, title=title, xlabel=xlabel, ylabel=ylabel)


ACRONYMS = {"ltv": "LTV", "dti": "DTI", "pud": "PUD"}


def _pretty_covariate(name: str) -> str:
    """``occupancy_type_Second_Home`` -> ``occupancy type Second Home``, with acronyms kept."""
    words = str(name).replace("_", " ").split()
    return " ".join(ACRONYMS.get(word.lower(), word) for word in words)


# --------------------------------------------------------------------------
# 1. Marginal competing-risk picture
# --------------------------------------------------------------------------
def plot_competing_risk_overview(
    frame: SurvivalFrame, outdir: Path, timeline: np.ndarray | None = None
) -> Path:
    """
    Left: portfolio composition by loan age (the bands sum to 1 by construction).
    Right: the two cumulative incidence curves, with the naive ``1 - KM`` that
    ignores the competing risk drawn dashed on top of each, so the
    overstatement is visible rather than only described.
    """
    _style()
    timeline = timeline if timeline is not None else np.arange(1, int(frame.data[frame.duration_col].max()) + 1)

    default = aalen_johansen_cif(frame, config.EVENT_DEFAULT, timeline=timeline)
    prepaid = aalen_johansen_cif(frame, config.EVENT_PREPAID, timeline=timeline)
    months = default["month"].to_numpy()

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- composition -------------------------------------------------------
    survivors = default["overall_survival"].to_numpy()
    cif_d = default["cif"].to_numpy()
    cif_p = prepaid["cif"].to_numpy()

    left.stackplot(
        months,
        survivors,
        cif_d,
        cif_p,
        colors=[SURVIVOR_FILL, CAUSE_COLOR[config.EVENT_DEFAULT], CAUSE_COLOR[config.EVENT_PREPAID]],
        labels=["Still performing", "Defaulted", "Prepaid"],
        edgecolor=SURFACE,
        linewidth=2.0,  # the 2px surface gap between stacked segments
    )
    left.set_ylim(0, 1)
    left.set_xlim(months.min(), months.max())
    _percent_axis(left)
    _finish(left, "Portfolio composition by loan age", ylabel="Share of loans")
    left.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, -0.32))

    # --- cumulative incidence ---------------------------------------------
    for curve, cause in ((default, config.EVENT_DEFAULT), (prepaid, config.EVENT_PREPAID)):
        color = CAUSE_COLOR[cause]
        right.plot(months, curve["cif"], color=color, label=f"{CAUSE_LABEL[cause]} (CIF)")
        right.plot(
            months,
            curve["naive_1_minus_km"],
            color=color,
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            alpha=0.75,
            label=f"{CAUSE_LABEL[cause]} (naive 1-KM)",
        )
        # Direct label at the curve end -- identity never rests on colour alone.
        right.annotate(
            CAUSE_LABEL[cause],
            xy=(months[-1], curve["cif"].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
        )

    right.set_xlim(months.min(), months.max() * 1.12)
    right.set_ylim(0, None)
    _percent_axis(right)
    _finish(right, "Cumulative incidence, competing risks vs. naive", ylabel="Cumulative probability")
    right.legend(loc="upper left")

    suptitle(
        fig,
        "Competing risks over loan age",
        "Dashed: 1 - cause-specific Kaplan-Meier, which treats the competing event as "
        "censoring and so overstates incidence.",
    )
    return _save(fig, outdir / "cif_overview.png")


# --------------------------------------------------------------------------
# 2. Segmented curves (small multiples)
# --------------------------------------------------------------------------
def plot_segmented_cif(
    frame: SurvivalFrame,
    segment_col: str,
    outdir: Path,
    timeline: np.ndarray | None = None,
    order: list | None = None,
    min_loans: int = 50,
) -> Path | None:
    """
    One panel per segment, both causes in each, with the portfolio-wide curve
    behind as a reference.

    Every panel shares its axes, so the comparison is a scan rather than an
    arithmetic exercise, and the pooled reference makes "above or below the
    book" readable without cross-referencing panels.
    """
    _style()
    timeline = timeline if timeline is not None else np.arange(1, int(frame.data[frame.duration_col].max()) + 1)

    segments = {}
    for cause in (config.EVENT_DEFAULT, config.EVENT_PREPAID):
        curves = cif_by_segment(frame, segment_col, cause, timeline=timeline, min_loans=min_loans)
        if curves.empty:
            return None
        segments[cause] = curves

    values = list(segments[config.EVENT_DEFAULT][segment_col].unique())
    if order:
        values = [v for v in order if v in values] + [v for v in values if v not in order]
    else:
        values = sorted(values, key=lambda v: (str(type(v)), v))

    pooled = {
        cause: aalen_johansen_cif(frame, cause, timeline=timeline) for cause in segments
    }

    n = len(values)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), sharex=True, sharey=True, squeeze=False
    )

    ymax = max(
        float(segments[c][segments[c][segment_col].isin(values)]["cif"].max()) for c in segments
    )

    for index, value in enumerate(values):
        ax = axes[index // ncols][index % ncols]
        n_loans = None
        follow_up = 0
        for cause, curves in segments.items():
            pooled_curve = pooled[cause]
            # The book-wide curve for the *same cause*, so "above or below the
            # portfolio" is legible without cross-referencing another panel.
            ax.plot(
                pooled_curve["month"], pooled_curve["cif"],
                color=CAUSE_COLOR[cause], linewidth=1.2, alpha=0.40,
                linestyle=(0, (4, 3)), zorder=1,
            )
            subset = curves[curves[segment_col] == value]
            n_loans = int(subset["n_loans"].iloc[0])
            follow_up = max(follow_up, int(subset["follow_up_months"].iloc[0]))
            ax.plot(
                subset["month"], subset["cif"],
                color=CAUSE_COLOR[cause], label=CAUSE_LABEL[cause], zorder=2,
            )
        ax.set_ylim(0, ymax * 1.05)
        _percent_axis(ax)
        # Naming the follow-up length makes the truncated curve self-explaining:
        # a short line is a short observation window, not a low risk.
        _finish(
            ax,
            f"{value}   {n_loans:,} loans, {follow_up}m follow-up",
            xlabel="", ylabel="",
        )

    for index in range(n, nrows * ncols):
        axes[index // ncols][index % ncols].set_visible(False)

    for col in range(ncols):
        axes[nrows - 1][col].set_xlabel("Months on book")
    for row in range(nrows):
        axes[row][0].set_ylabel("Cumulative incidence")

    handles = [
        plt.Line2D([], [], color=CAUSE_COLOR[c], linewidth=2, label=CAUSE_LABEL[c])
        for c in segments
    ] + [
        plt.Line2D(
            [], [], color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
            label="Whole portfolio (same cause)",
        )
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))

    pretty = segment_col.replace("_", " ")
    suptitle(fig, f"Cumulative incidence by {pretty}")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _save(fig, outdir / f"cif_by_{segment_col}.png")


# --------------------------------------------------------------------------
# 3. Model comparison on the holdout
# --------------------------------------------------------------------------
def plot_model_comparison(
    observed: dict[int, pd.DataFrame],
    predicted: dict[str, dict[int, np.ndarray]],
    timeline: np.ndarray,
    outdir: Path,
) -> Path:
    """
    Observed (Aalen-Johansen) against each model's portfolio-average CIF on the
    holdout vintages.

    The observed curve is drawn as a thick neutral reference rather than as a
    third coloured series: it is the target, not a competitor.
    """
    _style()
    causes = sorted(observed)
    fig, axes = plt.subplots(1, len(causes), figsize=(6.2 * len(causes), 4.4), squeeze=False)

    styles = {
        "cox": {"color": CAUSE_COLOR[config.EVENT_DEFAULT], "label": "Cox (cause-specific)"},
        "constant_hazard": {"color": CAUSE_COLOR[config.EVENT_PREPAID], "label": "Constant hazard"},
    }

    for index, cause in enumerate(causes):
        ax = axes[0][index]
        curve = observed[cause]
        ax.plot(
            curve["month"], curve["cif"],
            color=INK_SECONDARY, linewidth=3.0, alpha=0.9,
            label="Observed (Aalen-Johansen)", zorder=1,
        )
        for name, style in styles.items():
            if name not in predicted or cause not in predicted[name]:
                continue
            ax.plot(
                timeline, predicted[name][cause],
                color=style["color"], linewidth=2.0, linestyle="-", label=style["label"], zorder=2,
            )
        _percent_axis(ax)
        _finish(ax, f"{CAUSE_LABEL[cause]}, holdout vintages", ylabel="Cumulative incidence")
        ax.legend(loc="upper left")

    suptitle(fig, "Predicted vs. observed cumulative incidence")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, outdir / "model_comparison.png")


# --------------------------------------------------------------------------
# 4. Hazard ratios
# --------------------------------------------------------------------------
def plot_hazard_ratios(summaries: dict[int, pd.DataFrame], outdir: Path, top_n: int = 10) -> Path:
    """
    Forest plot of hazard ratios with 95% intervals, one panel per cause.

    Log x-axis, because a hazard ratio is multiplicative: 0.5 and 2.0 are the
    same size of effect in opposite directions and should be equidistant from
    the reference line at 1.
    """
    _style()
    causes = sorted(summaries)
    fig, axes = plt.subplots(
        1, len(causes), figsize=(6.4 * len(causes), 4.6), squeeze=False, sharex=True
    )

    # The panels share an x-axis so the two causes are directly comparable, so
    # the tick set has to be chosen across *both* -- computing it per panel
    # lets whichever panel is drawn last silently crop the other one's ticks.
    tables = {}
    for cause in causes:
        table = summaries[cause].copy()
        table["_effect"] = np.log(table["hazard_ratio"]).abs()
        tables[cause] = table.nlargest(top_n, "_effect").sort_values("hazard_ratio")

    low = min(t["hr_lower_95"].min() for t in tables.values())
    high = max(t["hr_upper_95"].max() for t in tables.values())
    ticks = [t for t in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0) if low / 1.25 <= t <= high * 1.25]

    for index, cause in enumerate(causes):
        ax = axes[0][index]
        color = CAUSE_COLOR[cause]
        table = tables[cause]

        positions = np.arange(len(table))
        ax.hlines(
            positions,
            table["hr_lower_95"], table["hr_upper_95"],
            color=color, linewidth=2.0, alpha=0.45,
        )
        ax.scatter(table["hazard_ratio"], positions, s=64, color=color, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(1.0, color=AXIS, linewidth=1.2, zorder=1)

        ax.set_yticks(positions)
        ax.set_yticklabels([_pretty_covariate(c) for c in table["covariate"]], fontsize=8)
        ax.set_xscale("log")
        # A log axis defaults to 10^0 notation, unreadable for ratios that all
        # live between 0.4 and 2.
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}x" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Hazard ratio (per 1 SD, log scale)")
        ax.set_title(CAUSE_LABEL[cause], color=INK_PRIMARY, loc="left", pad=10, fontweight="medium")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        for pos, value in zip(positions, table["hazard_ratio"]):
            ax.annotate(
                f"{value:.2f}", xy=(value, pos), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=7.5, color=INK_SECONDARY,
            )

    suptitle(
        fig,
        "Cause-specific hazard ratios",
        "Right of the line: raises the hazard. Continuous covariates are standardised, "
        "so each ratio is the effect of a one-standard-deviation move.",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, outdir / "hazard_ratios.png")


# --------------------------------------------------------------------------
# 5. Calibration
# --------------------------------------------------------------------------
def plot_calibration(tables: dict[int, pd.DataFrame], outdir: Path, horizon: int) -> Path:
    """Predicted vs. observed cumulative incidence by risk decile, per cause."""
    _style()
    causes = sorted(tables)
    fig, axes = plt.subplots(1, len(causes), figsize=(5.4 * len(causes), 4.4), squeeze=False)

    for index, cause in enumerate(causes):
        ax = axes[0][index]
        table = tables[cause]
        color = CAUSE_COLOR[cause]
        limit = max(table["mean_predicted"].max(), table["observed_cif"].max()) * 1.08

        ax.plot([0, limit], [0, limit], color=AXIS, linewidth=1.2, zorder=1, label="Perfect calibration")
        ax.plot(
            table["mean_predicted"], table["observed_cif"],
            color=color, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=2, label=f"{CAUSE_LABEL[cause]} deciles",
        )
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
        )
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        _percent_axis(ax)
        _finish(
            ax,
            f"{CAUSE_LABEL[cause]} at {horizon} months",
            xlabel="Predicted cumulative incidence",
            ylabel="Observed (Aalen-Johansen)",
        )
        ax.grid(axis="x")
        ax.legend(loc="upper left")

    suptitle(fig, f"Calibration by risk decile, {horizon}-month horizon")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, outdir / "calibration.png")
