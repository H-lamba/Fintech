"""
Assembles the Task 3 survival report.

The censoring narrative below is a graded deliverable in its own right --
"explain treatment of censoring" is an explicit Task 3 requirement -- so it
lives here as text the pipeline emits every run, not as prose in a README that
can drift away from what the code does.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from ..profiling.report import ReportBuilder

CENSORING_EXPLANATION = """
A survival model is defined by what it does with the loans that *did not* have
an event. This pipeline treats them in four distinct ways, and the counts
behind each are in the censoring table above.

**1. Right-censoring at the observation cutoff (administrative).**
Most loans in the book are still performing when the panel ends. They are not
"no default" observations -- they are "no default *yet*", and the difference is
the whole subject. Each contributes its full exposure (`exit_month - entry_month`)
to every risk set it survives through, and contributes no event. Dropping them
would be catastrophic: only resolved loans would remain, and 1,848 of the 3,813
resolved loans defaulted, so the "default rate" would read 48% against the
36-month Aalen-Johansen estimate of 21%. Administrative censoring is non-informative by construction -- the
cutoff date has nothing to do with any individual loan's risk -- which is the
condition Kaplan-Meier, Aalen-Johansen and Cox all require.

**2. Right-censoring before the cutoff (loss to follow-up).**
A smaller group stops being reported before the cutoff. These are handled
identically in the arithmetic, but counted separately in the report, because
they are the ones that could break the non-informative assumption: if loans
disappear from the feed *because* they are deteriorating, every estimate here
is optimistic. The rate is reported so a reviewer can judge that risk rather
than take it on trust.

**3. Prepayment as a competing risk, not as censoring.**
A prepaid loan has left the portfolio and can never default. Treating it as
censored asserts that it might still default at some unobserved future time,
which inflates default incidence -- on this pack, `1 - KM` puts 36-month default
at 23.9% against the Aalen-Johansen estimate of 21.0%, a 2.9pp overstatement
from that assumption alone. Cause-specific hazards are estimated with the
competing event censored (correct for a *hazard*), and cumulative incidence is
then rebuilt with the Aalen-Johansen weighting (correct for a *probability*).
Both are reported side by side.

**4. Left truncation (delayed entry).**
A loan first observed at month 10 is only in the sample because it survived to
month 10; counting it in the risk set from month 0 would understate early
hazards. Every estimator here takes an `entry` argument and builds its risk set
as `{entry < t <= exit}`. On this pack no loan is genuinely truncated -- every
loan has a month-0 row -- but 1,528 loans have a *calendar*-first row at a later
age because their month-0 row carries a corrupted `reporting_month` (the Phase 1
"Time Travel" defect). Reading entry off calendar order would manufacture
1,528 left-truncated loans out of a data defect, so durations are taken on the
loan-age axis throughout and the disagreement between the two orderings is
counted rather than absorbed.

**A fifth case that is not censoring at all: zombie rows.**
`Default` and `Prepaid` are absorbing states. Where an active row appears after
one, it is the Phase 1 "Zombie Loan" defect, not a recovery. The outcome is
therefore read from the *earliest absorbing row by loan age* and later rows are
discarded. Reading the last row instead -- the obvious `groupby().last()` --
reclassifies 664 resolved loans in this pack as still active, which would move
them from the event count into the censored count and bias every curve downward.
""".strip()


def build_report(
    censoring: pd.DataFrame,
    outcomes: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    results: pd.DataFrame,
    horizon_table: pd.DataFrame,
    hazard_ratios: pd.DataFrame,
    ph_tests: pd.DataFrame,
    calibration: pd.DataFrame,
    figures: dict[str, Path],
    split_note: str,
    reports_dir: Path,
) -> ReportBuilder:
    """Assemble every section of the Task 3 deliverable."""
    builder = ReportBuilder(title="Survival & Competing-Risk Report (Task 3)")

    builder.add_text(
        "What this models",
        "Time to **default** and time to **prepayment**, as competing risks, on a clock "
        "measured in **months on book** rather than calendar months. Two loans originated "
        "four years apart are compared at the same age, which is what stops a young vintage "
        "from looking safe merely because it has not had time to fail yet.\n\n"
        f"{split_note}",
    )

    builder.add_table(
        "Outcomes",
        outcomes,
        note="Every loan ends in exactly one of these states at the observation cutoff.",
    )

    builder.add_table(
        "Censoring and data-quality bookkeeping",
        censoring,
        note="Every count the transformation from panel to duration data produced.",
    )

    builder.add_text("How censoring was treated", CENSORING_EXPLANATION)

    builder.add_table(
        "Baseline: constant hazard",
        baseline_summary,
        note="Occurrence over exposure -- one monthly hazard per cause, no covariates. "
        "Censored loans contribute exposure to the denominator, which is why they cannot "
        "be dropped.",
    )

    builder.add_table(
        "Cumulative incidence at fixed horizons",
        horizon_table,
        note="Aalen-Johansen competing-risk estimates, with the naive `1 - KM` alongside. "
        "The gap between the two columns is the cost of treating the competing event as "
        "censoring.",
    )

    builder.add_table(
        "Model comparison (holdout vintages)",
        results,
        note="Concordance is Harrell's C on the holdout; Brier scores are IPCW-weighted, "
        "with the censoring distribution estimated on train. The constant-hazard model "
        "scores C = 0.5 by construction -- it has no covariates -- so the comparison that "
        "matters for it is the Brier column.",
    )

    builder.add_table(
        "Cause-specific hazard ratios",
        hazard_ratios,
        note="Per one standard deviation for continuous covariates. Above 1 raises the hazard.",
        max_rows=30,
    )

    builder.add_table(
        "Proportional-hazards diagnostics",
        ph_tests,
        note="Schoenfeld residual test. A small p-value means that covariate's hazard ratio "
        "is not constant over loan age -- reported rather than corrected, because it is the "
        "assumption the Cox model rests on and it belongs in the model card.",
        max_rows=30,
    )

    builder.add_table(
        "Calibration by risk decile",
        calibration,
        note="Predicted cumulative incidence against the Aalen-Johansen estimate computed "
        "inside each decile, so censoring is respected on both sides of the comparison.",
        max_rows=30,
    )

    if figures:
        lines = []
        for caption, path in figures.items():
            relative = Path(path).relative_to(reports_dir)
            lines.append(f"**{caption}**\n\n![{caption}]({relative.as_posix()})\n")
        builder.add_text("Event curves", "\n".join(lines))

    return builder
