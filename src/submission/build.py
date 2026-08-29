"""
Assembling and validating ``submission.csv``.

The template is the contract
----------------------------
``data/submission_template.csv`` is read at run time and its column list is
taken as binding -- not a hardcoded list in this file. A change the organiser
makes to the template then surfaces as a validation failure on the next run
instead of as a silently wrong submission, which is the failure mode that costs
a whole entry.

Validation runs before the file is written, not after. Every check below has
been chosen because its failure would be invisible in a spot check: a column in
the wrong order, a probability of 1.4, a row count that is short by the loans
whose history happened to be missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..anomaly.curation import UNSUPPORTED_ACTION, _suggest_action
from .inference import _clean_driver_names

# Fallback only. The real operating point is tuned on a held-back window and
# travels with the predictions in `exception_threshold`.
EXCEPTION_THRESHOLD = 0.5

# Never the bare string "None": pandas reads it back from CSV as NaN, so every
# clean row in the delivered file would silently lose its label. The same bug
# was fixed in the Phase 5 reviewer queue.
NO_EXCEPTION = "No exception"
NO_DRIVERS = "none"
NO_ACTION = "No exception detected; no action required."


@dataclass
class ValidationReport:
    """Every check run against the assembled submission."""

    checks: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)

    def failures(self) -> list:
        return [c for c in self.checks if not c["passed"]]


def load_template(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path or config.SUBMISSION_TEMPLATE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"No submission template at {path}. The template defines the required "
            "columns and row set; the submission cannot be validated without it."
        )
    return pd.read_csv(path)


def build_submission(scored: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the submission in the template's own column order.

    Rows are aligned to the template by ``(loan_id, reporting_month)`` rather
    than by position: a submission whose rows are in a different order than the
    template's is not obviously wrong on inspection and is completely wrong to a
    scorer that joins on index.
    """
    frame = scored.copy()
    frame[config.TIME_COL] = pd.to_datetime(frame[config.TIME_COL]).dt.strftime("%Y-%m-%d")

    # --- derived columns ---------------------------------------------------
    triggered = frame.get("triggered_rules", pd.Series("", index=frame.index)).fillna("")

    # The model's judgement where it exists, the rule flag only as a fallback.
    # "Any rule fired" flags 13.8% of the book against a 2.6% base rate, almost
    # all of it one low-severity check.
    if "exception_probability" in frame.columns and frame["exception_probability"].notna().any():
        # The threshold travels with the predictions, tuned on a held-back
        # window rather than assumed. See inference._supervised_exceptions.
        threshold = (
            float(frame["exception_threshold"].dropna().iloc[0])
            if "exception_threshold" in frame.columns and frame["exception_threshold"].notna().any()
            else EXCEPTION_THRESHOLD
        )
        flagged = frame["exception_probability"].fillna(0.0) >= threshold
    else:
        flagged = triggered.str.len() > 0
    frame["exception_required"] = flagged.astype(int)

    predicted = frame.get("predicted_exception_type")
    frame["exception_type"] = _resolve_exception_type(predicted, triggered, flagged)
    frame["top_drivers"] = _drivers_column(triggered, flagged)
    frame["action"] = [
        _action_for(rules, is_flagged, score)
        for rules, is_flagged, score in zip(
            triggered, flagged, frame.get("anomaly_score", pd.Series(0.0, index=frame.index))
        )
    ]

    keys = [config.ID_COL, config.TIME_COL]
    template = template.copy()
    template[config.TIME_COL] = pd.to_datetime(template[config.TIME_COL]).dt.strftime("%Y-%m-%d")

    available = [c for c in template.columns if c in frame.columns and c not in keys]
    merged = template[keys].merge(frame[[*keys, *available]], on=keys, how="left")

    # Preserve the template's exact column list and order.
    for column in template.columns:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[list(template.columns)]


# Rule names map to the defect taxonomy the panel uses. Ordered most specific
# first: a row can trigger several rules, and the sequence-aware detectors are
# the ones that identify a defect class rather than a symptom of one.
_EXCEPTION_RULES = [
    ("post_absorbing_activity", "Zombie Loan"),
    ("illegal_status_transition", "Impossible State Transition"),
    ("delinquency_bucket_skip", "Impossible State Transition"),
    ("origination_after_reporting", "Time Travel"),
    ("loan_age_inconsistent", "Time Travel"),
    ("TEMPORAL_ORDERING", "Time Travel"),
    ("balance_exceeds_original", "Balance Discrepancy"),
    ("BALANCE_CEILING", "Balance Discrepancy"),
    ("balance_increase_without_modification", "Balance Discrepancy"),
]


def _resolve_exception_type(
    predicted: pd.Series | None, triggered: pd.Series, flagged: pd.Series
) -> pd.Series:
    """
    The model's predicted class, falling back to the rule mapping.

    A row the model clears is labelled ``No exception`` whatever rules fired on
    it -- otherwise a single low-severity document check would put every clean
    loan into a defect class.
    """
    if predicted is not None and predicted.notna().any():
        resolved = predicted.fillna(NO_EXCEPTION).astype(str)
    else:
        resolved = _infer_exception_type(triggered)
    return resolved.where(flagged, NO_EXCEPTION)


def _drivers_column(triggered: pd.Series, flagged: pd.Series) -> pd.Series:
    """Driver names for flagged rows; an explicit placeholder for the rest."""
    drivers = triggered.map(_clean_driver_names)
    drivers = drivers.where(flagged & drivers.str.len().gt(0), NO_DRIVERS)
    return drivers.replace("", NO_DRIVERS)


def _action_for(rules: str, flagged: bool, anomaly_score: float) -> str:
    """
    What a reviewer should do, which depends on whether anything was flagged.

    The rule-based text assumes something fired. A clean row scored by the
    model as clean needs "no action required", not an instruction to go and
    write the rule that would have caught it.
    """
    if not flagged:
        return NO_ACTION
    if isinstance(rules, str) and rules.strip():
        return _suggest_action(rules)
    return UNSUPPORTED_ACTION


def _infer_exception_type(triggered: pd.Series) -> pd.Series:
    """
    The defect class implied by the rules that fired.

    Rule-based rather than modelled, deliberately. The Phase 5 classifier
    reaches perfect per-class F1 on this pack precisely because each defect
    carries a near-deterministic rule fingerprint -- so the rule mapping is the
    same answer with an audit trail attached, and it degrades honestly on a
    defect class nobody has written a rule for instead of guessing one of four.
    """
    def classify(rules: str) -> str:
        if not isinstance(rules, str) or not rules.strip():
            return NO_EXCEPTION
        for pattern, label in _EXCEPTION_RULES:
            if pattern in rules:
                return label
        return "Other"

    return triggered.map(classify)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate(submission: pd.DataFrame, template: pd.DataFrame) -> ValidationReport:
    """Run every structural and value check before the file is written."""
    report = ValidationReport()

    report.add(
        "columns match the template exactly, in order",
        list(submission.columns) == list(template.columns),
        f"expected {list(template.columns)}, got {list(submission.columns)}",
    )
    report.add(
        "row count matches the template",
        len(submission) == len(template),
        f"template {len(template):,}, submission {len(submission):,}",
    )

    keys = [config.ID_COL, config.TIME_COL]
    if set(keys).issubset(submission.columns):
        report.add(
            "no duplicate (loan_id, reporting_month) rows",
            not submission.duplicated(keys).any(),
            f"{int(submission.duplicated(keys).sum())} duplicates",
        )
        template_keys = set(
            zip(template[config.ID_COL], pd.to_datetime(template[config.TIME_COL]).dt.strftime("%Y-%m-%d"))
        )
        submission_keys = set(zip(submission[config.ID_COL], submission[config.TIME_COL]))
        report.add(
            "row set matches the template's",
            template_keys == submission_keys,
            f"{len(template_keys - submission_keys)} missing, "
            f"{len(submission_keys - template_keys)} unexpected",
        )

    # A probability outside [0, 1] is the kind of error that survives a spot
    # check and fails a scorer.
    for column in config.SUBMISSION_UNIT_INTERVAL_COLUMNS:
        if column not in submission.columns:
            continue
        values = pd.to_numeric(submission[column], errors="coerce")
        outside = int(((values < 0) | (values > 1)).sum())
        report.add(f"`{column}` within [0, 1]", outside == 0, f"{outside} values outside")

    # Unfilled cells are the most likely silent failure: a loan whose history
    # was missing scores as NaN and the row is submitted blank.
    for column in submission.columns:
        if column in keys:
            continue
        missing = int(submission[column].isna().sum())
        report.add(
            f"`{column}` fully populated",
            missing == 0,
            f"{missing:,} missing ({missing / max(len(submission), 1):.1%})",
        )

    # Validate the CSV round trip, not the in-memory frame. A cell holding the
    # literal string "None" or "" passes an in-memory null check and comes back
    # as NaN the moment anyone reads the file -- which is the only state the
    # organiser will ever see.
    round_tripped = _round_trip(submission)
    for column in round_tripped.columns:
        if column in keys:
            continue
        missing = int(round_tripped[column].isna().sum())
        report.add(
            f"`{column}` survives a CSV round trip",
            missing == 0,
            f"{missing:,} cells read back as NaN",
        )

    if "next_state" in submission.columns:
        unexpected = sorted(set(submission["next_state"].dropna()) - set(config.STATUS_ORDER))
        report.add(
            "`next_state` values are valid performance states",
            not unexpected,
            f"unexpected: {unexpected}",
        )

    return report


def _round_trip(submission: pd.DataFrame) -> pd.DataFrame:
    """Write to a buffer and read back, exactly as a consumer would."""
    import io

    buffer = io.StringIO()
    submission.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)


def write_submission(
    submission: pd.DataFrame, report: ValidationReport, path: Path | str | None = None
) -> Path:
    """
    Write the submission, refusing if a structural check failed.

    Value-level failures are warned about; structural ones stop the write. A
    file with the wrong columns is not a partially-good submission, it is an
    unreadable one, and writing it anyway just moves the discovery later.
    """
    structural = {
        "columns match the template exactly, in order",
        "row count matches the template",
        "no duplicate (loan_id, reporting_month) rows",
        "row set matches the template's",
    }
    blocking = [c for c in report.failures() if c["check"] in structural]
    if blocking:
        detail = "; ".join(f"{c['check']}: {c['detail']}" for c in blocking)
        raise ValueError(f"Refusing to write an invalid submission -- {detail}")

    path = Path(path or config.SUBMISSION_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path
