"""
The feature matrix both the unsupervised and the supervised detectors consume.

Three groups:

* **Signal indicators** -- every deterministic rule, date check and
  sequence-aware detector as a 0/1 column, plus the severity-weighted rule
  score. Handing these to the models is what makes the system *hybrid* rather
  than two systems reported side by side: the learned model can discover that
  ``missing_document_status`` fires on 11.8% of records and is almost never an
  exception, while ``post_absorbing_activity`` fires on 0.5% and always is.
* **Sequence context** -- the numeric form of the loan's month-on-month
  behaviour: severity moves, balance moves, transition rarity. These give the
  models something continuous to work with where the detectors give a
  threshold.
* **Record state** -- the panel's own numerics, plus the Phase 1 data-quality
  score.

Everything is **backward-looking or contemporaneous**. An exception label is a
statement about the record itself, so a detector could in principle read the
following month too; it deliberately does not. A reviewer queue that only
works once the next month's file has landed is a reviewer queue that is always
a month late, and the conservative choice keeps this model usable on the
current reporting cycle.

The Task 2 forward-looking targets and the exception labels themselves are
hard-excluded and the exclusion is asserted, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config

EXCEPTION_FLAG = "exception_required"
EXCEPTION_TYPE = "exception_type"
# Not the string "None": pandas reads that back from CSV as NaN, so the
# reviewer queue would silently lose the label on every clean record.
NO_EXCEPTION = "No exception"


@dataclass
class AnomalyFeatures:
    """The modelling frame plus the column lists each model is allowed to see."""

    data: pd.DataFrame
    signal_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)

    @property
    def model_columns(self) -> list[str]:
        return [*self.signal_columns, *self.numeric_columns, *self.categorical_columns]

    # Continuous features derived from the loan's adjacent months. Named
    # explicitly so the ablation can remove *sequence information* rather than
    # whatever happens to match a string prefix.
    SEQUENCE_CONTEXT = (
        "status_severity_delta",
        "dpd_delta",
        "balance_delta_pct",
        "transition_rarity",
        "months_after_absorbing",
        "months_observed_to_date",
    )

    @property
    def sequence_context_columns(self) -> list[str]:
        return [c for c in self.SEQUENCE_CONTEXT if c in self.numeric_columns]

    def row_level_signal_columns(self) -> list[str]:
        """
        Signal columns that decide from one record, plus the row-level rule score.

        ``rule_score`` is deliberately swapped for ``rule_score_row_level``:
        the headline score is a noisy-OR over *every* signal including the
        sequence detectors, so leaving it in an ablation that claims to remove
        sequence information would smuggle that information straight back in.
        This is the difference between an ablation and a decoration.
        """
        return [
            *[c for c in self.signal_columns if not c.startswith("seq__") and c != "rule_score"],
            *(["rule_score_row_level"] if "rule_score_row_level" in self.data.columns else []),
        ]

    @property
    def no_sequence_flag_columns(self) -> list[str]:
        """Detector flags removed; continuous month-on-month context retained."""
        return [*self.row_level_signal_columns(), *self.numeric_columns]

    @property
    def record_state_columns(self) -> list[str]:
        """
        Everything the model can see from one record in isolation.

        The strictest ablation: no sequence-aware detector flags, no aggregate
        that includes them, and no month-on-month context. What survives is
        what a model can learn without ever being told the loan has a history.
        """
        context = set(self.sequence_context_columns)
        return [
            *self.row_level_signal_columns(),
            *[c for c in self.numeric_columns if c not in context],
        ]

    @property
    def unsupervised_columns(self) -> list[str]:
        """
        Numeric-only view for the Isolation Forest.

        The forest is asked to find records that are *unusual*, so it is given
        the continuous state of the record and the sequence context -- not the
        rule indicators. Feeding it the deterministic flags would let it
        rediscover the rules and report them back as a discovery; keeping them
        out is what lets the ablation table say something meaningful about how
        much the unsupervised layer adds on its own.
        """
        return self.numeric_columns


def assert_no_leaky_features(columns: list[str]) -> None:
    """Hard stop if a label or a forward-looking target reached the matrix."""
    leaked = sorted(set(columns) & set(config.FORBIDDEN_FEATURES))
    if leaked:
        raise ValueError(
            f"Leaky columns in the anomaly feature matrix: {leaked}. "
            "These are labels or forward-looking targets (config.FORBIDDEN_FEATURES)."
        )


# Severity is carried on the signal matrix's producer, not its columns, so the
# row-level score reconstructs it from the same convention signals.py uses.
_HIGH_PREFIXES = ("date__", "seq__")


def _severity_of(column: str) -> str:
    """Severity for a signal column, matching signals.build_signal_matrix."""
    if column.startswith(_HIGH_PREFIXES):
        return "high"
    if "missing_document_status" in column or "DOCUMENT_STATUS" in column:
        return "low"
    if any(token in column for token in ("DPD_RANGE", "RATE_RANGE", "TERM_NON_NEGATIVE",
                                         "current_status_but_high_dpd", "implausible_interest_rate",
                                         "dpd_implausibly_large")):
        return "medium"
    return "high"


def _transition_rarity(panel: pd.DataFrame, previous_status: pd.Series) -> pd.Series:
    """
    Empirical frequency of this month's (previous -> current) state move.

    Learned from the panel rather than declared, so it needs no hand-written
    state machine and degrades gracefully on data whose legal moves differ from
    this pack's. Rare is not the same as illegal -- ``Current -> Default``
    occurs twice here and is legitimate -- which is precisely why this is a
    feature for a model to weigh rather than a rule that fires.
    """
    pairs = previous_status.fillna("<start>") + " -> " + panel["current_status"].astype(str)
    frequency = pairs.value_counts(normalize=True)
    return pairs.map(frequency).astype("float64")


def build_features(
    panel: pd.DataFrame,
    signal_matrix: pd.DataFrame,
    rule_score: pd.Series,
    dq_scores: pd.DataFrame | None = None,
) -> AnomalyFeatures:
    """Assemble the anomaly modelling frame from the panel and its signals."""
    frame = panel.copy()
    frame[config.TIME_COL] = pd.to_datetime(frame[config.TIME_COL], errors="coerce")

    ordered = frame.sort_values([config.ID_COL, config.TIME_COL])
    by_loan = ordered.groupby(config.ID_COL, sort=False)

    order = {state: i for i, state in enumerate(config.STATUS_ORDER)}
    severity_now = ordered["current_status"].map(order).astype("float64")
    severity_prev = by_loan["current_status"].shift(1).map(order).astype("float64")
    previous_status = by_loan["current_status"].shift(1)

    context = pd.DataFrame(index=ordered.index)
    context["status_severity"] = severity_now
    context["status_severity_delta"] = severity_now - severity_prev
    context["dpd_delta"] = ordered["days_past_due"] - by_loan["days_past_due"].shift(1)
    context["balance_delta_pct"] = (
        ordered["current_balance"] / by_loan["current_balance"].shift(1).replace(0, np.nan) - 1.0
    )
    context["transition_rarity"] = _transition_rarity(ordered, previous_status)
    context["months_observed_to_date"] = by_loan.cumcount().astype("float64")

    absorbing = ordered["current_status"].isin(config.ABSORBING_STATES)
    absorbing_age = (
        ordered["loan_age_months"].where(absorbing).groupby(ordered[config.ID_COL]).transform("min")
    )
    context["months_after_absorbing"] = (ordered["loan_age_months"] - absorbing_age).clip(lower=0)

    # Scheduled-balance deviation: the Balance Discrepancy defect in continuous
    # form, and a useful signal for servicing error generally.
    rate = (ordered["interest_rate"].astype("float64") / 100.0) / 12.0
    term = ordered.get("original_term_months")
    if term is None:
        term = ordered["loan_age_months"] + ordered["remaining_term_months"]
    age = ordered["loan_age_months"].astype("float64").clip(lower=0)
    with np.errstate(over="ignore", invalid="ignore"):
        growth_n = np.power(1.0 + rate, term.astype("float64"))
        growth_a = np.power(1.0 + rate, np.minimum(age, term.astype("float64")))
        scheduled = ordered["original_balance"] * (growth_n - growth_a) / (growth_n - 1.0)
    context["balance_vs_scheduled"] = ordered["current_balance"] / scheduled.replace(0, np.nan)

    context = context.reindex(frame.index)
    frame = pd.concat([frame, context], axis=1)

    # --- deterministic signals as features ---------------------------------
    indicators = signal_matrix.reindex(frame.index).fillna(False).astype("int8")
    frame = pd.concat([frame, indicators], axis=1)
    frame["rule_score"] = rule_score.reindex(frame.index).astype("float64")

    # A second score over the row-level signals only, so the ablations below
    # can remove sequence information without the aggregate carrying it back.
    from .signals import SEVERITY_WEIGHT  # local import: avoids a cycle

    row_level = [c for c in signal_matrix.columns if not c.startswith("seq__")]
    if row_level:
        weights = np.array([SEVERITY_WEIGHT.get(_severity_of(c), 0.5) for c in row_level])
        complement = 1.0 - signal_matrix[row_level].reindex(frame.index).fillna(False).to_numpy(dtype=float) * weights
        frame["rule_score_row_level"] = 1.0 - complement.prod(axis=1)
    else:
        frame["rule_score_row_level"] = 0.0

    # --- Phase 1 data-quality score ----------------------------------------
    if dq_scores is not None and not dq_scores.empty and "dq_score" in dq_scores.columns:
        keys = [config.ID_COL, config.TIME_COL]
        if set(keys).issubset(dq_scores.columns):
            scores = dq_scores[[*keys, "dq_score"]].copy()
            scores[config.TIME_COL] = pd.to_datetime(scores[config.TIME_COL], errors="coerce")
            frame = frame.merge(scores.drop_duplicates(keys), on=keys, how="left")

    # --- column lists ------------------------------------------------------
    signal_columns = list(indicators.columns) + ["rule_score"]

    numeric_candidates = [
        "current_balance", "original_balance", "days_past_due", "loan_age_months",
        "remaining_term_months", "interest_rate", "credit_score", "ltv", "dti",
        "status_severity", "status_severity_delta", "dpd_delta", "balance_delta_pct",
        "transition_rarity", "months_observed_to_date", "months_after_absorbing",
        "balance_vs_scheduled", "dq_score",
    ]
    numeric_columns = [
        c for c in numeric_candidates
        if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])
    ]

    categorical_columns = [
        c for c in ("current_status", "document_status", "source_system", "servicer_name")
        if c in frame.columns
    ]

    features = AnomalyFeatures(
        data=frame,
        signal_columns=signal_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    assert_no_leaky_features(features.model_columns)
    return features


def prepare_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """
    The two supervised targets, with the multiclass one made total.

    ``exception_type`` is null for every clean record. Left as null it would be
    dropped by any splitter and the model would train only on exceptions, which
    is not the question -- at scoring time nobody knows yet whether a record is
    an exception. Filling the nulls with an explicit ``None`` class makes the
    task what it actually is: a five-way classification over every record.
    """
    targets = pd.DataFrame(index=panel.index)
    if EXCEPTION_FLAG in panel.columns:
        targets[EXCEPTION_FLAG] = panel[EXCEPTION_FLAG].astype("boolean").fillna(False).astype(int)
    if EXCEPTION_TYPE in panel.columns:
        targets[EXCEPTION_TYPE] = panel[EXCEPTION_TYPE].fillna(NO_EXCEPTION).astype(str)
    return targets


def exception_classes(targets: pd.DataFrame) -> list[str]:
    """Class order for the multiclass head: the clean class first, then alphabetical."""
    if EXCEPTION_TYPE not in targets.columns:
        return []
    observed = sorted(set(targets[EXCEPTION_TYPE].unique()) - {NO_EXCEPTION})
    return [NO_EXCEPTION, *observed]
