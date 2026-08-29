"""
One chart style for the whole project.

Every figure this repo emits -- profiling, survival, and whatever Phases 5-7
add -- draws from the tokens below, so a reader who learns the palette in the
Data Intelligence Report can read the survival report without relearning it.

The palette is validated rather than chosen by eye:

* **Categorical** slots are assigned in a fixed order and never cycled. The
  first three clear the colour-vision-deficiency and normal-vision separation
  gates on this light surface for *all* pairs; past three, facet instead of
  adding hues.
* **Sequential** encoding uses one hue, light to dark, with the lightest step
  still clearing 2:1 against the surface so "near zero" recedes without
  disappearing.
* **Status** colours (good / warning / serious / critical) are reserved for
  state and never reused as a series colour. They always ship with a label,
  because roughly one reader in twelve cannot separate them by hue.

Chrome recedes: hairline horizontal grid, no top or right spine, muted ticks.
The data is the only thing at full contrast.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots, in assignment order. Never cycle past the end -- fold the
# tail into "Other" or facet.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# Single-hue ramp for magnitude. Light end clears 2:1 on the surface.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Reserved for state. Always accompanied by a text label.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def style() -> None:
    """Apply the project chart style. Idempotent; call at the top of any figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "lines.linewidth": 2.0,
        }
    )


def finish(ax, title: str = "", xlabel: str = "", ylabel: str = "", grid_axis: str = "y") -> None:
    """Recessive chrome: drop the top/right spines, keep one hairline grid axis."""
    if title:
        ax.set_title(title, color=INK_PRIMARY, loc="left", pad=10, fontweight="medium")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)


def percent_axis(ax, which: str = "y", decimals: int = 0) -> None:
    """Percent ticks on round steps -- 0/5/10/15, never 0/3/5/8."""
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    axis.set_major_formatter(lambda v, _: f"{v:.{decimals}%}")


def suptitle(fig, text: str, subtitle: str = "") -> None:
    """Left-aligned figure title, with an optional muted line beneath the figure."""
    fig.suptitle(
        text, x=0.008, ha="left", fontsize=13, color=INK_PRIMARY, fontweight="semibold"
    )
    if subtitle:
        fig.text(0.008, -0.03, subtitle, ha="left", fontsize=8, color=INK_MUTED)


def sequential_steps(n: int) -> list[str]:
    """``n`` evenly spaced steps of the sequential ramp, light to dark."""
    if n <= 1:
        return [SEQUENTIAL[3]]
    span = len(SEQUENTIAL) - 1
    return [SEQUENTIAL[round(i * span / (n - 1))] for i in range(n)]


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
