"""Assembles the Task 5 scenario and stress report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from ..profiling.report import ReportBuilder
from .drivers import narrate

METHOD_NOTE = """
**The scenario file is the only source of stress assumptions.** Nothing in this
pipeline invents an unemployment path, a rate shock or an elasticity.
`data/macro_scenarios.csv` supplies, per scenario and month, a mortgage rate, an
unemployment rate, a house price index and the scenario's own stated default and
prepayment multipliers. A missing column stops the run rather than falling back
to a default.

**Three transmission channels, in descending order of how much they assume.**

*House prices to loan-to-value* is mechanical. LTV is debt over value, so a
house price index at 72 against a starting 100 raises current LTV by a factor of
1.39. No elasticity, no fitted relationship -- arithmetic.

*Market rate to refinance incentive* is mechanical. `rate_spread` is the loan's
note rate less the prevailing market rate, and the scenario states that rate. A
borrower paying 7% in a 4.2% market has an incentive the Phase 3 model already
knows how to read.

*Labour market to credit quality* is *calibrated, not assumed*. The file gives
an unemployment path and a default multiplier but no elasticity connecting them
to credit scores. Hard-coding one -- "40 points of FICO per point of
unemployment" -- would make the projection a restatement of that invented
number. Instead the shift is solved for: find the portfolio-wide credit-score
move that makes the model reproduce the scenario's *own* stated default
multiplier. The file stays authoritative, the model supplies the transmission,
and the answer lands in a unit a credit officer recognises.

**Bounds are enforced and banded columns are rebuilt.** Every stressed feature is
clipped to a plausible range, and `credit_score_band`, `ltv_band` and `dti_band`
are recomputed from the shifted values underneath them. Moving `ltv` while
leaving `ltv_band` at its original level hands the model a record that
contradicts itself -- and the model will score it without complaint.

**One credit shift per scenario-month, shared across all three targets.** The
shift describes a state of the world, not a per-model tuning knob. Letting each
target solve its own would produce three mutually inconsistent portfolios and
call them one scenario.
""".strip()

LIMITS_NOTE = """
**The credit channel saturates, and the projection says so.** Beyond a point,
no shift in credit score reproduces the default multiplier the scenario file
states. Even moving the entire book to the floor of the observable score range
(500) leaves the Adverse-Credit multiplier short at the longer horizons. Three
readings, all worth stating:

1. The Phase 3 model's sensitivity to credit score is bounded by the range it
   was fitted on. A 2.6x default multiplier is outside what this book's credit
   distribution can express, however hard it is pushed.
2. The scenario's severity is therefore carried by channels the model does not
   see. Unemployment at 8.1% is not a feature; it reaches the model only
   through the calibrated shift, and the shift has run out of room.
3. A naive calibration would clamp at its search bound, report a number and
   move on -- and the projection would quietly under-state the scenario. The
   saturation table names the shortfall in the file's own units instead.

This is why both projection methods are reported. The **feature-stress** column
is what the model says when its inputs are moved; the **stated-multiplier**
column is the scenario file's own view applied directly to the baseline rate.
Where they diverge, the feature-stress figure is a floor, not a forecast.

**Prepayment barely responds to the rate path.** The High-Prepayment scenario
states a 2.9x prepayment multiplier by month 48; the feature-stress projection
produces roughly 1.05x. That is consistent rather than anomalous: Task 2 found
prepayment close to unpredictable in this pack (ROC-AUC 0.52) and Task 3 found
a cause-specific Cox model barely beating a constant hazard on it (C = 0.558).
A model with no prepayment signal cannot acquire one under stress. The
stated-multiplier column is the usable projection for prepayment here.
""".strip()

SCOPE_NOTE = """
**What these numbers are.** For each scenario and projection month, the
portfolio's features are stressed to that month's assumptions and re-scored
through the Phase 3 models. A figure at horizon 24 is the *conditional forward
rate as at that month*: given the book and the macro state two years out, the
probability a loan defaults over the following twelve months.

**What they are not.** This is a stress-sensitivity projection, not a cash-flow
run-off. The Phase 3 models predict a fixed forward window from a record's own
month; they do not compound, and this pipeline does not pretend they do. The
portfolio is held at its last observed position rather than amortised, loans
that default or prepay are not removed as the horizon extends, and no balance is
rolled forward. The question answered is "how much worse does this book look
under that macro state" -- which is what a stress test is for. The question left
open is "what are the cumulative losses", which needs the Phase 4 hazards and a
run-off engine.

**Scoring population.** The latest observed record per loan: the book as it
stands at the data cutoff. Scoring every historical month would weight the
projection towards loans that happen to have long histories.
""".strip()


def projection_markdown(portfolio_level: pd.DataFrame, float_format: str = ".4f") -> str:
    """The headline projection table, rendered."""
    columns = [
        "scenario", "horizon_month", "projection_month", "loans", "credit_score_shift",
        *[label for label in config.SCENARIO_TARGETS.values() if label in portfolio_level.columns],
        *[c for c in portfolio_level.columns if c.endswith("_vs_baseline_pp")],
    ]
    display = portfolio_level[[c for c in columns if c in portfolio_level.columns]].copy()
    if "projection_month" in display.columns:
        display["projection_month"] = pd.to_datetime(display["projection_month"]).dt.strftime("%Y-%m")

    for column in display.select_dtypes(include="number").columns:
        values = display[column].dropna()
        spec = "d" if len(values) and (values % 1 == 0).all() else float_format
        display[column] = display[column].map(
            lambda v, spec=spec: "--" if pd.isna(v) else format(int(v) if spec == "d" else v, spec)
        )
    return display.to_markdown(index=False)


def build_narratives(
    drivers: pd.DataFrame,
    movements: pd.DataFrame,
    portfolio_level: pd.DataFrame,
    scenarios,
    horizons: tuple[int, ...] | None = None,
) -> dict[str, str]:
    """
    A written explanation per (scenario, measure), generated from the attribution.

    Written from the numbers rather than from the scenario's name, so the prose
    cannot drift away from what the model actually did.
    """
    if drivers.empty:
        return {}

    horizons = horizons or config.SCENARIO_HORIZONS
    longest = max(h for h in horizons if h in set(drivers["horizon_month"]))
    baseline_name = scenarios.baseline_name
    reference = portfolio_level[portfolio_level["scenario"] == baseline_name].set_index("horizon_month")

    narratives: dict[str, str] = {}
    for scenario in drivers["scenario"].unique():
        for measure in drivers["measure"].unique():
            subset = drivers[
                (drivers["scenario"] == scenario)
                & (drivers["measure"] == measure)
                & (drivers["horizon_month"] == longest)
            ]
            if subset.empty:
                continue

            projected = portfolio_level[
                (portfolio_level["scenario"] == scenario)
                & (portfolio_level["horizon_month"] == longest)
            ]
            if projected.empty or longest not in reference.index:
                continue
            change_pp = 100.0 * (projected[measure].iloc[0] - reference.loc[longest, measure])

            movement = movements[
                (movements["scenario"] == scenario) & (movements["horizon_month"] == longest)
            ]
            narratives[f"{scenario} / {measure}"] = narrate(subset, movement, change_pp)

    return narratives


def build_report(
    assumptions: pd.DataFrame,
    portfolio_level: pd.DataFrame,
    calibration: pd.DataFrame,
    checks: pd.DataFrame,
    saturation: pd.DataFrame,
    segment_tables: dict[str, pd.DataFrame],
    drivers: pd.DataFrame,
    movements: pd.DataFrame,
    narratives: dict[str, str],
    figures: dict[str, Path],
    reports_dir: Path,
    n_loans: int,
    variant: str,
) -> ReportBuilder:
    """Assemble every section of the Task 5 deliverable."""
    builder = ReportBuilder(title="Scenario & Stress Simulation Report (Task 5)")

    builder.add_text(
        "What this projects",
        f"The book's {n_loans:,} loans at their latest observed position, re-scored under "
        f"each macro scenario through the Phase 3 `{variant}` models. Delinquency, default "
        "and prepayment, portfolio-wide and by segment, with the movement attributed to "
        "features.",
    )
    builder.add_text("Scope and limitations", SCOPE_NOTE)

    builder.add_table(
        "Scenario assumptions",
        assumptions,
        note="Read directly from `data/macro_scenarios.csv`. Nothing here is a project "
        "assumption.",
    )

    builder.add_table(
        "Portfolio projection",
        portfolio_level.assign(
            projection_month=pd.to_datetime(portfolio_level["projection_month"]).dt.strftime("%Y-%m")
        ),
        note="Mean projected rate per scenario and horizon, with the change against the "
        "baseline scenario at the same horizon in percentage points.",
        max_rows=40,
    )

    builder.add_text("Method", METHOD_NOTE)
    builder.add_text("Where this projection runs out of road", LIMITS_NOTE)

    builder.add_table(
        "Calibrated credit channel",
        calibration,
        note="The portfolio-wide credit-score shift that makes the model reproduce each "
        "scenario's stated default multiplier. `converged = False` marks a multiplier the "
        "model cannot reach by any shift in the search range -- reported as a number rather "
        "than silently clamped.",
        max_rows=40,
    )

    builder.add_table(
        "Credit channel saturation",
        saturation,
        note="`reached = False` marks a stated multiplier the model cannot produce by any "
        "credit-score shift. `attainable_multiplier` is the ceiling it reaches at the floor "
        "of the observable score range, and `shortfall` is the gap the feature-stress "
        "projection therefore under-states by.",
        max_rows=40,
    )

    builder.add_table(
        "Stated vs. modelled multipliers",
        checks,
        note="Two independent views of the same scenario. Default agrees by construction -- "
        "the credit channel was calibrated to make it agree. **Prepayment was not calibrated "
        "against anything**, so its column is the model's own view of the refinance response, "
        "derived only from the rate path. Where it disagrees with the file, that gap is a "
        "finding about the model, not an error in the projection.",
        max_rows=40,
    )

    for segment, table in segment_tables.items():
        builder.add_table(
            f"Impact by {segment.replace('_', ' ')}",
            table.sort_values(["horizon_month", "scenario"]),
            note="A portfolio-level move spread evenly is a different problem from the same "
            "move concentrated in one segment.",
            level=3,
            max_rows=30,
        )

    if narratives:
        blocks = [f"### {key}\n\n{text}\n" for key, text in narratives.items()]
        builder.add_text(
            "Scenario drivers",
            "Generated from the model's own per-feature contributions: the change in each "
            "feature's mean contribution between the baseline and stressed portfolios is "
            "that feature's share of the change in the rate.\n\n" + "\n".join(blocks),
        )

    if not drivers.empty:
        builder.add_table("Driver attribution", drivers, level=3, max_rows=40)
    if not movements.empty:
        builder.add_table(
            "How far each stressed feature moved",
            movements,
            note="The attribution says which feature mattered; this says what happened to it.",
            level=3,
            max_rows=40,
        )

    if figures:
        lines = []
        for caption, path in figures.items():
            relative = Path(path).relative_to(reports_dir)
            lines.append(f"**{caption}**\n\n![{caption}]({relative.as_posix()})\n")
        builder.add_text("Figures", "\n".join(lines))

    return builder
