"""
Turning scores into a reviewer queue.

Task 4 asks for at least 20 reviewer-ready examples. The number is the easy
part; the useful part is *which* 20. Taking the top 20 by score would return
twenty near-identical Balance Discrepancy rows, because that is the class the
deterministic layer is most confident about -- a queue that demonstrates one
detector working and says nothing about the other three.

So the queue is **stratified**: a guaranteed allocation per detected exception
type, plus a block of high-scoring records that carry *no* deterministic
violation at all. That last group is the one worth arguing about -- it is
where the unsupervised layer is making a claim the rules cannot support, and
it is the only part of the queue capable of surfacing a defect class nobody has
written a rule for yet.

Each row carries what a reviewer needs to act without opening the pipeline:
the identifiers, the score, what fired, why, and a suggested action.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .features import NO_EXCEPTION

# What to tell the reviewer to do, keyed on the dominant deterministic signal.
# Ordered: the first pattern that matches a record's triggered rules wins, so
# the more specific remediations come first.
SUGGESTED_ACTIONS: list[tuple[str, str]] = [
    (
        "post_absorbing_activity",
        "Loan already reached a terminal state. Confirm the termination date with the "
        "servicer and suppress the later rows; do not re-activate the loan.",
    ),
    (
        "illegal_status_transition",
        "Delinquency bucket skipped without an intermediate month. Request the missing "
        "month's file from the servicer before accepting the status.",
    ),
    (
        "delinquency_bucket_skip",
        "Status deteriorated by more than one bucket in a month. Verify days_past_due "
        "against the servicer's own ledger.",
    ),
    (
        "origination_after_reporting",
        "Reporting month precedes origination. Correct the timestamp at source; the row "
        "cannot be used for age-based analytics until it is fixed.",
    ),
    (
        "loan_age_inconsistent",
        "Loan age does not agree with the origination and reporting dates. Recompute age "
        "from the dates and reconcile with the servicer.",
    ),
    (
        "TEMPORAL_ORDERING",
        "Date fields are out of order. Correct at source before this record enters any "
        "time-aware model.",
    ),
    (
        "balance_exceeds_original",
        "Balance exceeds origination with no modification recorded. Request the "
        "modification agreement, or treat the balance as a servicing error.",
    ),
    (
        "BALANCE_CEILING",
        "Balance breaches the ceiling rule. Reconcile the principal against the "
        "amortisation schedule.",
    ),
    (
        "balance_increase_without_modification",
        "Principal rose month-on-month with no modification flag. Check for a capitalised "
        "arrears event that was never recorded.",
    ),
    (
        "loan_age_regression",
        "Loan age fell while the calendar advanced. Usually a knock-on of a corrupted "
        "date on the neighbouring row; check both months together.",
    ),
    (
        "missing_document_status",
        "Document status is blank. Low severity on its own -- chase the file only if the "
        "record is queued for another reason.",
    ),
]

UNSUPPORTED_ACTION = (
    "No rule fired; the model flagged this on its pattern alone. Review manually and, if "
    "it is a genuine defect, write the rule that would have caught it."
)


def _suggest_action(triggered: str) -> str:
    if not triggered:
        return UNSUPPORTED_ACTION
    for pattern, action in SUGGESTED_ACTIONS:
        if pattern in triggered:
            return action
    return "Review the triggered rules against the servicer's source file."


def build_review_queue(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    triggered: pd.Series,
    drivers: pd.Series,
    predicted_type: pd.Series | None = None,
    n_examples: int = 25,
    n_unsupported: int = 5,
) -> pd.DataFrame:
    """
    Assemble the curated reviewer queue.

    Parameters
    ----------
    scores:
        Frame with at least ``hybrid_score``; ``rule_score``, ``ml_score`` and
        ``exception_probability`` are carried through when present.
    n_examples:
        Total rows to return. Task 4's floor is 20; the default leaves headroom
        so the stratification below has something to allocate.
    n_unsupported:
        How many of those rows must be high-scoring records with **no**
        deterministic violation. Reserving slots for them is deliberate: they
        are the only rows in the queue that can teach the rule set something.
    """
    queue = pd.DataFrame(index=frame.index)
    queue[config.ID_COL] = frame[config.ID_COL]
    queue[config.TIME_COL] = pd.to_datetime(frame[config.TIME_COL], errors="coerce").dt.strftime("%Y-%m")
    queue["current_status"] = frame.get("current_status")
    queue["current_balance"] = frame.get("current_balance")

    for column in ("hybrid_score", "rule_score", "ml_score", "exception_probability"):
        if column in scores.columns:
            queue[column] = scores[column].to_numpy()

    queue["triggered_rules"] = triggered.reindex(frame.index).fillna("")
    queue["top_drivers"] = drivers.reindex(frame.index).fillna("")
    if predicted_type is not None:
        queue["predicted_exception_type"] = predicted_type.reindex(frame.index)
    queue["suggested_action"] = queue["triggered_rules"].map(_suggest_action)

    selected = _stratified_selection(queue, n_examples=n_examples, n_unsupported=n_unsupported)
    return selected.sort_values("hybrid_score", ascending=False).reset_index(drop=True)


def _stratified_selection(
    queue: pd.DataFrame, n_examples: int, n_unsupported: int
) -> pd.DataFrame:
    """
    Guarantee coverage of every detected type, then fill by score.

    Without this the queue is monotone in whatever the most confident detector
    happens to be, and a reviewer reading it would conclude the system only
    finds one kind of problem.
    """
    ranked = queue.sort_values("hybrid_score", ascending=False)
    has_rule = ranked["triggered_rules"].str.len() > 0

    chosen: list[pd.Index] = []

    # 1. Rule-unsupported records first, so they are never crowded out.
    unsupported = ranked[~has_rule].head(n_unsupported)
    chosen.append(unsupported.index)

    # 2. A guaranteed block per predicted exception type.
    if "predicted_exception_type" in ranked.columns:
        types = [
            value for value in ranked["predicted_exception_type"].dropna().unique()
            if value != NO_EXCEPTION
        ]
        per_type = max(2, (n_examples - n_unsupported) // max(len(types), 1))
        for value in types:
            block = ranked[
                (ranked["predicted_exception_type"] == value)
                & ~ranked.index.isin(np.concatenate([c.to_numpy() for c in chosen]) if chosen else [])
            ].head(per_type)
            chosen.append(block.index)

    taken = pd.Index(np.concatenate([c.to_numpy() for c in chosen])).unique() if chosen else pd.Index([])

    # 3. Fill the remainder by score.
    remaining = n_examples - len(taken)
    if remaining > 0:
        filler = ranked[~ranked.index.isin(taken)].head(remaining)
        taken = taken.append(filler.index)

    return queue.loc[taken.unique()[:n_examples]]


def queue_composition(queue: pd.DataFrame) -> pd.DataFrame:
    """What the curated queue is made of -- for the report, and as a sanity check."""
    rows = []
    if "predicted_exception_type" in queue.columns:
        for value, count in queue["predicted_exception_type"].value_counts().items():
            rows.append({"dimension": "predicted type", "value": value, "examples": int(count)})
    supported = (queue["triggered_rules"].str.len() > 0).sum()
    rows.append({"dimension": "evidence", "value": "rule-supported", "examples": int(supported)})
    rows.append({"dimension": "evidence", "value": "model-only", "examples": int(len(queue) - supported)})
    return pd.DataFrame(rows)
