"""
Figures for the scenario report.

Three charts:

* *Where does each scenario take the book?* -> projected rate against horizon,
  one panel per measure. A line chart, because the x-axis is time and the
  reader's question is about a path rather than a level.
* *Who absorbs it?* -> the change against baseline by segment. A dot plot on a
  shared axis: the spread matters more than any single value, and a bar chart
  of signed changes reads as magnitude when it is direction.
* *Why?* -> the feature attribution, as a diverging bar around zero, because
  the sign is the point.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, viz

# Ordinal segments must be plotted in their own order, not alphabetically:
# "<620, 800+, 740-799, ..." is a credit axis in scrambled order.
SEGMENT_ORDER = {
    "credit_score_band": list(config.CREDIT_SCORE_BANDS),
    "ltv_band": list(config.LTV_BANDS),
    "dti_band": list(config.DTI_BANDS),
}

MEASURE_LABEL = {
    "delinquency_3m": "Delinquency (next 3m)",
    "default_12m": "Default (next 12m)",
    "prepayment_12m": "Prepayment (next 12m)",
}


def _scenario_colors(scenarios: list[str], baseline: str | None = None) -> dict[str, str]:
    """
    Baseline in neutral ink, stressed scenarios in categorical slots.

    The baseline is a reference, not a competitor, so it is drawn as chrome
    rather than as a third series fighting for attention.
    """
    colors = {}
    slot = 0
    for name in scenarios:
        if baseline is not None and name == baseline:
            colors[name] = viz.INK_SECONDARY
        else:
            colors[name] = viz.CATEGORICAL[slot % len(viz.CATEGORICAL)]
            slot += 1
    return colors


def plot_projection_paths(portfolio_level: pd.DataFrame, outdir: Path) -> Path | None:
    """Projected rate against horizon, one panel per measure."""
    measures = [m for m in config.SCENARIO_TARGETS.values() if m in portfolio_level.columns]
    if portfolio_level.empty or not measures:
        return None

    scenarios = list(portfolio_level["scenario"].unique())
    # The flattest scenario is the baseline, identified the same way everywhere.
    spread = portfolio_level.groupby("scenario")[measures].std().fillna(0.0)
    baseline = spread.sum(axis=1).idxmin() if len(spread) else None
    colors = _scenario_colors(scenarios, baseline)

    viz.style()
    fig, axes = plt.subplots(1, len(measures), figsize=(5.4 * len(measures), 4.4), squeeze=False)

    for index, measure in enumerate(measures):
        ax = axes[0][index]
        for scenario in scenarios:
            subset = portfolio_level[portfolio_level["scenario"] == scenario].sort_values("horizon_month")
            if subset.empty:
                continue
            is_baseline = scenario == baseline
            ax.plot(
                subset["horizon_month"], subset[measure],
                color=colors[scenario], label=scenario,
                linewidth=2.6 if is_baseline else 2.0,
                linestyle="-" if not is_baseline else (0, (4, 3)),
                zorder=2 if is_baseline else 3,
            )

            # The scenario file's own stated view, where it differs from what
            # the feature stress produces. The gap between the solid and dotted
            # lines is the credit channel running out of room -- the single most
            # important thing on this chart, so it is drawn rather than
            # described.
            stated_column = f"{measure}_stated"
            if not is_baseline and stated_column in subset.columns:
                stated = subset[stated_column]
                if stated.notna().any() and not np.allclose(
                    stated.fillna(0), subset[measure].fillna(0), rtol=0.02
                ):
                    ax.plot(
                        subset["horizon_month"], stated,
                        color=colors[scenario], linewidth=1.4, linestyle=(0, (1, 2)),
                        alpha=0.85, zorder=3,
                        label=f"{scenario} (stated multiplier)",
                    )
        viz.percent_axis(ax, decimals=1)
        viz.finish(ax, MEASURE_LABEL.get(measure, measure),
                   xlabel="Projection month", ylabel="Projected rate")
        if index == 0:
            ax.legend(loc="best")

    viz.suptitle(
        fig,
        "Projected rates under each macro scenario",
        "Solid: the model re-scored on stressed features. Dotted: the scenario file's own "
        "stated multiplier applied to the baseline. Conditional forward rate at each "
        "projection month, not a cumulative run-off.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return viz.save(fig, outdir / "projection_paths.png")


def plot_segment_impact(
    segment_tables: dict[str, pd.DataFrame], outdir: Path, measure: str = "default_12m"
) -> Path | None:
    """
    Change against baseline by segment, at the longest horizon.

    A dot plot with a zero line: the reader's question is which segments move
    and by how much relative to each other, and dots on a shared axis answer it
    without implying the magnitudes stack.
    """
    delta_column = f"{measure}_vs_baseline_pp"
    usable = {
        name: table for name, table in segment_tables.items()
        if not table.empty and delta_column in table.columns and table[name].nunique() <= 12
    }
    if not usable:
        return None

    viz.style()
    fig, axes = plt.subplots(
        1, len(usable), figsize=(4.8 * len(usable), 4.8), squeeze=False, sharex=True
    )

    for index, (segment, table) in enumerate(usable.items()):
        ax = axes[0][index]
        horizon = int(table["horizon_month"].max())
        subset = table[(table["horizon_month"] == horizon) & (table[delta_column].abs() > 1e-12)]
        if subset.empty:
            ax.set_visible(False)
            continue

        scenarios = list(subset["scenario"].unique())
        colors = _scenario_colors(scenarios)
        present = set(subset[segment].astype(str))
        declared = SEGMENT_ORDER.get(segment)
        if declared:
            groups = [g for g in declared if g in present]
        else:
            groups = sorted(present)
        positions = {name: pos for pos, name in enumerate(groups)}

        for scenario in scenarios:
            rows = subset[subset["scenario"] == scenario]
            ax.scatter(
                rows[delta_column],
                [positions[str(v)] for v in rows[segment]],
                s=70, color=colors[scenario], label=scenario,
                edgecolor=viz.SURFACE, linewidth=1.5, zorder=3,
            )

        ax.axvline(0.0, color=viz.AXIS, linewidth=1.2, zorder=1)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=8)
        viz.finish(ax, segment.replace("_", " "),
                   xlabel="Change vs. baseline (pp)", grid_axis="x")
        if index == 0:
            ax.legend(loc="lower right")

    viz.suptitle(
        fig,
        f"Who absorbs the stress: {MEASURE_LABEL.get(measure, measure)}",
        "Change against the baseline scenario at the longest projected horizon, "
        "in percentage points.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return viz.save(fig, outdir / "segment_impact.png")


def plot_drivers(drivers: pd.DataFrame, outdir: Path, measure: str = "default_12m") -> Path | None:
    """
    Feature attribution as a diverging bar around zero, one panel per scenario.

    Diverging because the sign carries the meaning: a feature that pulls the
    rate down under a stress scenario is the interesting one, and a magnitude
    chart would hide it.
    """
    if drivers is None or drivers.empty:
        return None

    subset = drivers[drivers["measure"] == measure]
    if subset.empty:
        return None
    horizon = int(subset["horizon_month"].max())
    subset = subset[subset["horizon_month"] == horizon]

    scenarios = list(subset["scenario"].unique())
    viz.style()
    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(5.6 * len(scenarios), 4.4), squeeze=False, sharex=True
    )

    up, down = viz.CATEGORICAL[1], viz.CATEGORICAL[0]

    for index, scenario in enumerate(scenarios):
        ax = axes[0][index]
        rows = subset[subset["scenario"] == scenario].sort_values("delta_contribution")
        positions = np.arange(len(rows))
        colors = [up if v > 0 else down for v in rows["delta_contribution"]]

        ax.barh(positions, rows["delta_contribution"].to_numpy(), color=colors, height=0.6)
        ax.axvline(0.0, color=viz.AXIS, linewidth=1.2, zorder=2)
        ax.set_yticks(positions)
        ax.set_yticklabels([f.replace("_", " ") for f in rows["feature"]], fontsize=8)

        for position, value, share in zip(positions, rows["delta_contribution"], rows["share_of_movement"]):
            offset = 4 if value > 0 else -4
            ax.annotate(
                f"{share:.0%}", xy=(value, position), xytext=(offset, 0),
                textcoords="offset points", va="center",
                ha="left" if value > 0 else "right",
                fontsize=7.5, color=viz.INK_SECONDARY,
            )

        # Symmetric limits with padding: the share labels sit outside the bar
        # end, and a tight axis pushes them onto the y-tick labels.
        extent = float(subset["delta_contribution"].abs().max()) * 1.35
        ax.set_xlim(-extent, extent)
        viz.finish(ax, f"{scenario}, month {horizon}",
                   xlabel="Change in mean contribution to log-odds", grid_axis="x")

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=up, label="raises the rate"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=down, label="lowers the rate"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))

    viz.suptitle(
        fig,
        f"What moves each scenario: {MEASURE_LABEL.get(measure, measure)}",
        "Change in each feature's mean contribution between the baseline and stressed "
        "portfolios. Labels are the share of total movement.",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    return viz.save(fig, outdir / "scenario_drivers.png")
