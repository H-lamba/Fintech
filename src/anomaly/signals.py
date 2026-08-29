"""
The deterministic half of the hybrid: every rule-based signal, per record.

Two layers, and the distinction is the whole point of this module.

**Row-level rules** come straight from Phase 1: the organiser's
``validation_rules.json`` plus the project's own domain checks, all of which
decide from one record in isolation. They are exact, cheap and completely
blind to anything that requires the loan's history.

**Sequence-aware detectors** live here because a row-level engine structurally
cannot express them. Phase 1 measured the cost of that blindness: row-level
rules catch 100% of Balance Discrepancy and Time Travel and roughly 10% of
Impossible State Transition and Zombie Loan. The two invisible classes are
invisible for a reason -- one is a statement about the *previous* month's
status, the other about whether the loan was already terminal -- and neither
fits in an expression evaluated against a single row.

Both layers produce the same shape: a boolean matrix, a severity-weighted
score, and a human-readable string naming what fired. That string is what a
reviewer actually reads, so it is built here rather than reconstructed later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..profiling import outliers, rules

# A violation is evidence, not proof. These are the probabilities the noisy-OR
# below treats each severity as carrying on its own; they are deliberately
# short of 1.0 so that even a high-severity rule leaves room for the record to
# turn out benign.
SEVERITY_WEIGHT = {"high": 0.90, "medium": 0.50, "low": 0.15}

# Legal one-month moves between performance states. Delinquency escalates one
# bucket at a time; it may cure to Current from anywhere; terminal states are
# reachable from the states a servicer can reach them from.
LEGAL_TRANSITIONS: dict[str, set[str]] = {
    "Current": {"Current", "30-DPD", "Prepaid"},
    "30-DPD": {"Current", "30-DPD", "60-DPD", "Prepaid"},
    "60-DPD": {"Current", "30-DPD", "60-DPD", "90-DPD", "Prepaid"},
    "90-DPD": {"Current", "30-DPD", "60-DPD", "90-DPD", "Default", "Prepaid"},
    "Default": set(),
    "Prepaid": set(),
}


def _severity_map(rule_set: list[rules.Rule]) -> dict[str, str]:
    return {f"rule__{rule.name}": rule.severity for rule in rule_set}


# --------------------------------------------------------------------------
# Sequence-aware detectors
# --------------------------------------------------------------------------
def sequence_detectors(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Rule-strength checks that need the loan's own history.

    Returns a boolean matrix aligned to ``panel.index`` and a severity map.

    ``seq__post_absorbing_activity``
        A row at a loan age after the loan's first ``Default`` or ``Prepaid``.
        Those states are terminal by definition, so a later active row is not a
        recovery -- it is the Zombie Loan defect.
    ``seq__illegal_status_transition``
        A month-on-month move that the state machine does not allow, e.g.
        ``Current -> 90-DPD`` without passing through 30 and 60.
    ``seq__delinquency_bucket_skip``
        The same event measured as a magnitude: more than one bucket of
        deterioration in a single month. Kept separate from the legality check
        because it still fires when the state machine definition is wrong or
        incomplete, which on unfamiliar data it usually is.
    ``seq__balance_increase_without_modification``
        Principal rising month-on-month with no modification recorded.
    ``seq__loan_age_regression``
        Loan age falling while the calendar advances.
    """
    frame = panel.copy()
    frame["_order"] = np.arange(len(frame))
    frame[config.TIME_COL] = pd.to_datetime(frame[config.TIME_COL], errors="coerce")
    frame = frame.sort_values([config.ID_COL, config.TIME_COL])

    by_loan = frame.groupby(config.ID_COL, sort=False)
    previous_status = by_loan["current_status"].shift(1)

    # --- post-absorption activity -----------------------------------------
    absorbing = frame["current_status"].isin(config.ABSORBING_STATES)
    absorbing_age = (
        frame["loan_age_months"].where(absorbing).groupby(frame[config.ID_COL]).transform("min")
    )
    post_absorbing = absorbing_age.notna() & (frame["loan_age_months"] > absorbing_age)

    # --- transition legality ----------------------------------------------
    allowed = previous_status.map(LEGAL_TRANSITIONS)
    illegal = pd.Series(
        [
            bool(prev is not np.nan and isinstance(allow, set) and cur not in allow)
            for prev, allow, cur in zip(previous_status, allowed, frame["current_status"])
        ],
        index=frame.index,
    )
    illegal &= previous_status.notna()

    # --- bucket skip, as a magnitude --------------------------------------
    order = {state: i for i, state in enumerate(config.STATUS_ORDER)}
    severity_now = frame["current_status"].map(order)
    severity_prev = previous_status.map(order)
    bucket_skip = (severity_now - severity_prev) > 1

    # --- balance and age monotonicity -------------------------------------
    previous_balance = by_loan["current_balance"].shift(1)
    modified = (
        frame["modification_flag"].astype("boolean").fillna(False).astype(bool)
        if "modification_flag" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    balance_up = (frame["current_balance"] > previous_balance * 1.001) & ~modified

    age_regression = (by_loan["loan_age_months"].diff() < 0).fillna(False)

    detectors = pd.DataFrame(
        {
            "seq__post_absorbing_activity": post_absorbing.fillna(False),
            "seq__illegal_status_transition": illegal.fillna(False),
            "seq__delinquency_bucket_skip": bucket_skip.fillna(False),
            "seq__balance_increase_without_modification": balance_up.fillna(False),
            "seq__loan_age_regression": age_regression,
        }
    ).astype(bool)

    # Restore the caller's row order.
    detectors["_order"] = frame["_order"].to_numpy()
    detectors = detectors.sort_values("_order").drop(columns=["_order"])
    detectors.index = panel.index

    severities = {
        "seq__post_absorbing_activity": "high",
        "seq__illegal_status_transition": "high",
        "seq__delinquency_bucket_skip": "medium",
        "seq__balance_increase_without_modification": "medium",
        "seq__loan_age_regression": "high",
    }
    return detectors, severities


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_signal_matrix(
    panel: pd.DataFrame, json_rules: list[dict] | None = None
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """
    Every deterministic signal for every record.

    Returns ``(matrix, severities, rule_summary)`` where ``matrix`` is boolean
    with one column per signal, prefixed ``rule__`` (row-level),
    ``date__`` (temporal consistency) or ``seq__`` (sequence-aware).
    """
    rule_set = rules.build_rule_set(json_rules)
    rule_summary, rule_matrix = rules.evaluate_rules(panel, rule_set)
    severities = _severity_map(rule_set)

    date_matrix = outliers.date_violation_matrix(panel)
    if not date_matrix.empty:
        date_matrix = date_matrix.add_prefix("date__")
        # A date that contradicts the loan's own origination is a hard defect,
        # not a borderline one.
        severities.update({column: "high" for column in date_matrix.columns})

    sequence_matrix, sequence_severities = sequence_detectors(panel)
    severities.update(sequence_severities)

    parts = [part for part in (rule_matrix, date_matrix, sequence_matrix) if not part.empty]
    matrix = pd.concat(parts, axis=1).fillna(False).astype(bool)
    severities = {column: severities.get(column, "medium") for column in matrix.columns}

    return matrix, severities, rule_summary


def rule_score(matrix: pd.DataFrame, severities: dict[str, str]) -> pd.Series:
    """
    Severity-weighted violation score in [0, 1], combined as a noisy-OR.

    ``1 - prod(1 - w_i)`` over the rules that fired. Each violation is treated
    as independent evidence that the record is an exception, so two medium
    violations outrank one, and no amount of low-severity noise ever reaches
    the level of a single high-severity hit. A plain sum would let three
    ``missing_document_status`` flags outrank a balance that exceeds
    origination, which is exactly backwards.
    """
    if matrix.empty:
        return pd.Series(0.0, index=matrix.index)

    weights = np.array([SEVERITY_WEIGHT.get(severities.get(c, "medium"), 0.5) for c in matrix.columns])
    complement = 1.0 - matrix.to_numpy(dtype=float) * weights
    return pd.Series(1.0 - complement.prod(axis=1), index=matrix.index)


def triggered_rules(matrix: pd.DataFrame, severities: dict[str, str], max_rules: int = 4) -> pd.Series:
    """
    Human-readable list of what fired, worst severity first.

    This is the column a reviewer reads before anything else, so it is built
    from the same matrix that drives the score -- there is no second code path
    that could describe a different set of rules than the one scored.
    """
    if matrix.empty:
        return pd.Series("", index=matrix.index)

    rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(matrix.columns, key=lambda c: rank.get(severities.get(c, "medium"), 1))
    values = matrix[ordered].to_numpy()
    names = np.array([c.split("__", 1)[1] for c in ordered])

    out = []
    for row in values:
        fired = names[row]
        if len(fired) == 0:
            out.append("")
        elif len(fired) <= max_rules:
            out.append("; ".join(fired))
        else:
            out.append("; ".join(fired[:max_rules]) + f"; (+{len(fired) - max_rules} more)")
    return pd.Series(out, index=matrix.index)


def signal_coverage(
    matrix: pd.DataFrame, severities: dict[str, str], labels: pd.Series | None = None
) -> pd.DataFrame:
    """
    Per-signal firing rate and, where labels exist, its precision.

    A signal that fires often and is almost never an exception is a reviewer's
    time being spent; naming it is more useful than burying it in an aggregate.
    """
    rows = []
    for column in matrix.columns:
        fired = matrix[column]
        row = {
            "signal": column,
            "layer": column.split("__", 1)[0],
            "severity": severities.get(column, "medium"),
            "n_fired": int(fired.sum()),
            "pct_fired": round(100.0 * float(fired.mean()), 4),
        }
        if labels is not None and fired.any():
            row["precision_vs_exception"] = round(float(labels[fired].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_fired", ascending=False).reset_index(drop=True)
