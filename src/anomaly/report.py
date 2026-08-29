"""Assembles the Task 4 anomaly and exception report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..profiling.report import ReportBuilder

HEADLINE_CAVEAT = """
**Read the near-perfect scores below as a property of the data, not of the model.**
The supervised head reaches 0.999 precision at 0.999 recall and a perfect
per-class F1 on all five exception types. That is not a modelling achievement;
it is what happens when every defect class carries a near-deterministic
fingerprint:

| defect class | fingerprint | separability |
| :--- | :--- | :--- |
| Balance Discrepancy | balance above origination | caught exactly by two row-level rules |
| Time Travel | reporting month before origination | caught exactly by the date checks |
| Zombie Loan | active row after a terminal row | caught exactly by a sequence detector |
| Impossible State Transition | delinquency bucket skipped | caught by a sequence detector |

Each was *injected* by a generator, and an injection leaves a cleaner trace
than a real servicing error does. Real defects arrive partially, inconsistently
and mixed with legitimate rarities. **The numbers here measure whether the
pipeline is wired correctly; they do not forecast performance on a real
servicer feed.** The parts that would transfer are the layering, the
noisy-OR combination, the sequence detectors and the curation -- not the scores.

The ablation table is included precisely so this is checkable rather than
asserted: it shows how much of the result each layer is responsible for.
""".strip()

METHOD_NOTE = """
**The two layers do different jobs, and the split is measurable.**

Row-level rules -- the organiser's `validation_rules.json` plus this project's
own domain checks -- decide from a single record. They catch every Balance
Discrepancy and every Time Travel defect in this pack, and roughly one in ten
Impossible State Transitions and Zombie Loans. That is not a tuning failure: an
expression evaluated against one row cannot see last month's status, and cannot
know the loan was already terminal.

**Sequence-aware detectors** close that gap. `post_absorbing_activity` and
`illegal_status_transition` take recall on the two invisible classes from 12%
and 9% to 100% and 99%, lifting overall recall from 52% to 99.7%. They are
still deterministic -- they are rules, just rules that need the loan's history
-- so they are reported as part of the deterministic layer, not as an ML
result.

**What is left for the model is precision.** With the sequence detectors in
place the deterministic layer reaches 99.7% recall at 15.1% precision: a queue
of 9,584 records to find 1,449 exceptions, most of the excess coming from one
low-severity rule that fires on 11.8% of the book. The supervised head cuts
that queue to 1,450 records at 99.9% precision -- the same exceptions, a
sixth of the reviewing.

**The learned layer did find something the rules did not.** The ablation row
*supervised (record state only)* sees no sequence detector flags and no
month-on-month context, yet still reaches 0.979 PR-AUC. Its dominant feature is
`balance_vs_scheduled`, the ratio of reported balance to the amortisation
schedule, and inspecting it explains why:

* **Zombie Loan** rows carry a stale balance -- mean ratio 0.77 against 0.99
  for clean records, with 85% deviating more than 10% from schedule.
* **Impossible State Transition** rows sit at *exactly* 1.000 while genuine
  90-DPD records sit at 1.005 or above, because a real three-months-delinquent
  loan has accrued arrears and the injected row was written with a performing
  loan's balance.

Neither fingerprint was anticipated by any hand-written rule. This is the
honest case for the learned layer: not that it beat the rules on the headline
metric, but that it located a row-level signature of two defects that the rule
author believed were only visible in sequence.

**The unsupervised layer's unsupported flags are benign -- and that is the
finding.** Every one of the five highest-scoring records with no rule violation
is an ordinary loan termination: `Default` or `Prepaid` with a zero balance,
which is statistically extreme and operationally correct. An Isolation Forest
run alone on this book would put clean terminations at the top of the reviewer
queue. That is the concrete case for the hybrid: unsupervised novelty detection
finds what is *unusual*, and only the rule layer knows what is *wrong*.

**Scores combine as a noisy-OR, not a weighted average.**
`hybrid = 1 - (1 - rule_score) * (1 - ml_score)`. A fired high-severity rule
sets a floor the model cannot argue down; the model can only add suspicion on
top. A weighted average would let a confident model talk away a hard
violation, which is not a trade a servicer would accept.

**Continuous detectors are cut at a fixed queue size, not a fixed threshold.**
A rank-normalised score has no natural cut point -- a 0.5 threshold on the
Isolation Forest flags half the book. Every row of the ablation is therefore
evaluated at the same queue size the full deterministic layer produces, so the
comparison is at equal reviewer cost.
""".strip()

CENSORING_NOTE = """
The exception labels are **contemporaneous**: `exception_required` describes
the record it sits on, not a future outcome. So unlike Task 2 there is no
forward window to purge -- but the split is still strictly by
`reporting_month`, and for the same reason: a random split would put one month
of a loan in training and the next month of the same loan in test, and the
sequence features (`months_after_absorbing`, `status_severity_delta`) are
explicitly built from adjacent months. Absorbing-state rows are **kept**, not
dropped as they are in Tasks 2 and 3 -- a terminal row can itself be the defect.
""".strip()


def build_report(
    ablation: pd.DataFrame,
    signal_coverage: pd.DataFrame,
    type_metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
    importance: pd.DataFrame,
    queue: pd.DataFrame,
    composition: pd.DataFrame,
    split_note: str,
    reports_dir: Path,
    figures: dict[str, Path] | None = None,
) -> ReportBuilder:
    """Assemble every section of the Task 4 deliverable."""
    builder = ReportBuilder(title="Anomaly & Exception Report (Task 4)")

    builder.add_text(
        "What this detects",
        "Record-level data defects in the monthly performance panel: a continuous "
        "anomaly score for every row, a predicted probability and type for the "
        "`exception_required` / `exception_type` labels, the drivers behind each "
        "flag, and a curated queue a reviewer can work without opening the pipeline."
        f"\n\n{split_note}",
    )

    builder.add_text("How to read these numbers", HEADLINE_CAVEAT)

    builder.add_table(
        "Detector comparison",
        ablation,
        note="Each detector measured on the held-out window. `flagged` is the size of the "
        "reviewer queue it implies; `precision@k` is what fraction of the first k records "
        "are real exceptions.",
    )

    builder.add_text("How the hybrid is built", METHOD_NOTE)
    builder.add_text("Splitting", CENSORING_NOTE, level=3)

    builder.add_table(
        "Deterministic signal coverage",
        signal_coverage,
        note="Every rule, date check and sequence detector: how often it fires and how "
        "often it is right. A signal that fires constantly and is almost never an "
        "exception is a reviewer's time being spent, so it is named rather than "
        "averaged away.",
        max_rows=40,
    )

    builder.add_table(
        "Exception type classification",
        type_metrics,
        note="Five-way over every record, `None` included. Macro-F1 leads because `None` "
        "is 97.4% of the panel and accuracy is maximised by never predicting an exception.",
    )
    builder.add_table("Per-class performance", per_class, level=3)
    builder.add_table("Confusion matrix", confusion, level=3)

    builder.add_table(
        "What drives the model",
        importance.head(20),
        note="Mean absolute contribution to the predicted log-odds, from the booster's own "
        "per-row attributions. The `layer` column shows how much of the model's decision "
        "rests on deterministic evidence versus learned pattern.",
    )

    builder.add_table(
        "Curated reviewer queue",
        queue,
        note="Stratified rather than top-N: a guaranteed block per exception type plus "
        "reserved slots for high-scoring records with no rule violation, which are the "
        "only rows in the queue that can teach the rule set something. Full file: "
        "`reports/anomaly_examples.csv`.",
        max_rows=30,
    )
    builder.add_table("Queue composition", composition, level=3)

    if figures:
        lines = []
        for caption, path in figures.items():
            relative = Path(path).relative_to(reports_dir)
            lines.append(f"**{caption}**\n\n![{caption}]({relative.as_posix()})\n")
        builder.add_text("Figures", "\n".join(lines))

    return builder
