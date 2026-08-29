"""
Figures for the anomaly report.

Two charts, each answering a question the tables answer only by arithmetic:

* *Does adding a layer actually help, and where?* -> the precision/recall
  trade-off of each detector, with queue size on the axis a reviewer thinks in.
* *Which layer is the model actually leaning on?* -> contribution by signal
  layer, so "hybrid" is a measured claim rather than an architectural one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import viz

# Ordered deterministic-first, so the legend reads as a progression from
# "a rule said so" to "the model inferred it".
LAYER_COLOR = {
    "rule": viz.CATEGORICAL[0],
    "date": viz.CATEGORICAL[1],
    "sequence detector": viz.CATEGORICAL[2],
    "sequence context": viz.CATEGORICAL[3],
    "record state": viz.INK_MUTED,
}


def plot_detector_comparison(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    fixed_points: dict[str, tuple[int, float]],
    outdir: Path,
) -> Path | None:
    """
    Precision within the top-k records, as the queue lengthens.

    A precision/recall scatter was the obvious form and the wrong one: at a
    fixed queue size four of the detectors here land on the same point by
    construction -- they all flag the same 9,584 records and capture nearly
    every exception -- so the chart showed four overlapping dots and illegible
    labels. What actually separates them is *ranking*: how fast precision
    decays as the reviewer works further down the list. That is the question a
    queue owner asks, and this is the chart that answers it.

    ``fixed_points`` carries the deterministic detectors, which produce a set
    rather than a ranking and so appear as single markers rather than curves.
    """
    if not scores and not fixed_points:
        return None

    y_true = np.asarray(y_true).astype(int)
    total_positives = int(y_true.sum())
    if total_positives == 0:
        return None

    viz.style()
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    grid = np.unique(np.geomspace(20, max(len(y_true) // 4, 100), 60).astype(int))
    colours = viz.CATEGORICAL

    for position, (name, score) in enumerate(scores.items()):
        order = np.argsort(-np.asarray(score, dtype=float))
        hits = np.cumsum(y_true[order])
        precision = hits[grid - 1] / grid
        # No end-of-line direct labels: all four curves converge at the right
        # edge, where a stack of labels is unreadable. The legend carries
        # identity, and the curves are separated where it matters.
        ax.plot(grid, precision, color=colours[position % len(colours)], label=name, zorder=3)

    # Labels go to the left of their marker, not above or below it. The two
    # deterministic points sit close together and differ mainly in height, so a
    # vertical offset puts each label nearer the other point than its own.
    for name, (flagged, precision) in fixed_points.items():
        ax.scatter(
            [flagged], [precision], s=90, marker="D", zorder=4,
            color=viz.INK_SECONDARY, edgecolor=viz.SURFACE, linewidth=1.5,
        )
        ax.annotate(
            f"{name} ({flagged:,})",
            xy=(flagged, precision), xytext=(-12, 0), textcoords="offset points",
            ha="right", va="center", fontsize=8, color=viz.INK_SECONDARY,
        )

    base_rate = y_true.mean()
    ax.axhline(base_rate, color=viz.AXIS, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        f"base rate {base_rate:.1%}", xy=(grid[0], base_rate), xytext=(0, 6),
        textcoords="offset points", ha="left", fontsize=7.5, color=viz.INK_MUTED,
    )

    ax.set_xscale("log")
    ax.set_xlim(grid[0] * 0.8, grid[-1] * 1.35)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    viz.percent_axis(ax, which="y")
    viz.finish(ax, "", xlabel="Reviewer queue size (records, log scale)",
               ylabel="Precision within the queue")
    ax.grid(axis="x")
    ax.legend(loc="center left")

    viz.suptitle(
        fig,
        "What each detector layer buys",
        "Higher is better at every queue length. The deterministic detectors produce a set "
        "rather than a ranking, so they appear as single points.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return viz.save(fig, outdir / "detector_comparison.png")


def plot_driver_layers(importance: pd.DataFrame, outdir: Path, top_n: int = 18) -> Path | None:
    """
    Top features by mean absolute contribution, coloured by signal layer.

    The colour is the finding: if the bars are all one layer, the word "hybrid"
    is describing the architecture rather than the behaviour.
    """
    if importance is None or importance.empty or "layer" not in importance.columns:
        return None

    table = importance.nlargest(top_n, "mean_abs_contribution").sort_values("mean_abs_contribution")
    if table.empty:
        return None

    viz.style()
    fig, ax = plt.subplots(figsize=(10.0, 0.38 * len(table) + 2.4))
    positions = np.arange(len(table))
    colours = [LAYER_COLOR.get(layer, viz.INK_MUTED) for layer in table["layer"]]

    ax.barh(positions, table["mean_abs_contribution"].to_numpy(), color=colours, height=0.62)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [name.replace("rule__json__", "").replace("rule__", "").replace("date__date__", "")
         .replace("seq__", "").replace("_", " ") for name in table["feature"]],
        fontsize=8,
    )
    for position, value, share in zip(positions, table["mean_abs_contribution"], table["share"]):
        ax.annotate(
            f"{share:.1%}", xy=(value, position), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=7.5, color=viz.INK_SECONDARY,
        )
    ax.set_xlim(0, table["mean_abs_contribution"].max() * 1.16)
    viz.finish(ax, "", xlabel="Mean absolute contribution to predicted log-odds", grid_axis="x")

    present = [layer for layer in LAYER_COLOR if layer in set(table["layer"])]
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                   color=LAYER_COLOR[layer], label=layer)
        for layer in present
    ]
    ax.legend(handles=handles, loc="lower right", title="signal layer")

    viz.suptitle(
        fig,
        "What the exception model leans on",
        "Colour is the layer each signal comes from: deterministic rules, date checks, "
        "sequence-aware detectors, or learned record state.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return viz.save(fig, outdir / "driver_layers.png")
