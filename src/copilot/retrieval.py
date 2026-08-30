"""
Assembling grounded context for the reviewer copilot.

The copilot is a **summariser of things the pipeline already computed**, never
a predictor. That is not only a design preference: the challenge's
qualification rule disqualifies a solution that sends records to an LLM for
classification. So the context assembled here is deliberately complete -- the
loan's own record, the definitions of the fields in it, the rules that fired,
and the models' own outputs -- and the prompt built from it asks only for a
restatement a human reviewer can act on.

Retrieval, not recall
---------------------
Every fact the model is allowed to use is placed in the prompt. Nothing relies
on the model's parametric knowledge of mortgages, of this schema, or of what a
"Zombie Loan" is -- the data dictionary entry is supplied. A model asked about
a field it was not given should say so, and the guardrails in
:mod:`src.copilot.guardrails` check that it did.

Context budget
--------------
The panel has 39 columns and the feature matrix 59 more; a naive dump would
push a long prompt for no benefit and bury the fields that matter. Only the
fields referenced by a triggered rule, named in the model's own top drivers, or
on the core identity/state list are included, and each carries its dictionary
definition inline.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config

# Always included: a reviewer cannot act on a note that does not say which loan,
# which month, and what state it is in.
# The default ask: summarise, restate, decide nothing.
NOTE_TASK = (
    "Write a short reviewer note (3-5 sentences) summarising the situation above "
    "for a human loan reviewer. Restate only what is in this context. "
    "Do not estimate, predict, or infer any number that is not written above. "
    "Do not state a decision or a recommended outcome for the loan."
)

# The explorer's "recommend next steps" ask. Phrased as *verification*, not as
# a decision, because the decision guardrail is not a formality to be worked
# around -- a copilot that recommends an outcome is doing the reviewer's job,
# and the challenge's qualification rule is explicit that the LLM may not.
ACTION_TASK = (
    "List the concrete verification steps a human loan reviewer should take on this "
    "record, as 3-5 short bullet points. Each bullet must name the specific field, "
    "rule or document to check and why it is worth checking, using only what is "
    "written above. Do not estimate, predict, or infer any number that is not "
    "written above. Do not state a decision, an outcome, or a recommendation about "
    "the loan itself -- describe only what should be verified, and by what evidence."
)

CORE_FIELDS = [
    "loan_id",
    "reporting_month",
    "origination_month",
    "current_status",
    "days_past_due",
    "current_balance",
    "original_balance",
    "loan_age_months",
    "interest_rate",
    "credit_score_band",
    "ltv_band",
    "state",
    "servicer_name",
    "document_status",
]


@dataclass
class LoanContext:
    """Everything the copilot is allowed to know about one loan-month."""

    loan_id: str
    reporting_month: str
    record: dict = field(default_factory=dict)
    definitions: dict = field(default_factory=dict)
    triggered_rules: list = field(default_factory=list)
    model_outputs: dict = field(default_factory=dict)
    anomaly: dict = field(default_factory=dict)
    # What the model is asked to produce from this context. Defaults to the
    # reviewer note; the explorer's "recommend next steps" asks a different
    # question of the *same* grounded context, and the grounding is the part
    # that must not vary between them.
    task: str = ""

    def grounded_numbers(self) -> set[str]:
        """
        Every number that appears anywhere in the supplied context.

        The guardrail layer uses this to check the model's output for figures it
        was never given. A number in a reviewer note that is not in this set was
        invented.
        """
        found: set[str] = set()

        def collect(value) -> None:
            if isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, (pd.Timestamp, datetime.date, datetime.datetime)):
                # Dates in the record arrive as Timestamps, not strings. Left
                # unhandled they ground nothing, and a note faithfully quoting
                # an origination date is accused of inventing it.
                stamp = pd.Timestamp(value)
                found.update({stamp.strftime("%Y-%m-%d"), stamp.strftime("%Y-%m"),
                              str(stamp.year)})
            elif isinstance(value, (int, float, np.integer, np.floating)):
                if not pd.isna(value):
                    found.update(_number_forms(float(value)))
            elif isinstance(value, str):
                # Normalised on this side too: the grounded set and the output
                # scan must fold characters identically, or a value that is in
                # the context still fails to match the note that quotes it.
                value = _normalise(value)
                # Dates are grounded as whole strings and as year-month, so a
                # note quoting "2023-06-01" or "2023-06" matches either way.
                for date in _DATE_PATTERN.findall(value):
                    found.update({date, date[:7]})
                for token in _NUMBER_PATTERN.findall(value):
                    try:
                        found.update(_number_forms(float(token.replace(",", ""))))
                    except ValueError:
                        continue

        # The loan's own identifiers are context too.
        found.update({self.reporting_month, self.reporting_month[:7]})
        collect(self.record)
        collect(self.model_outputs)
        collect(self.anomaly)
        collect(self.triggered_rules)

        # The prompt's own instruction text is context too. A model echoing
        # "3-5 sentences" back is quoting the request, not inventing a figure.
        for token in _NUMBER_PATTERN.findall(_normalise(self.to_prompt())):
            try:
                found.update(_number_forms(float(token.replace(",", ""))))
            except ValueError:
                continue
        return found

    def to_prompt(self) -> str:
        """Render the context as the user message."""
        blocks = [
            f"## Loan under review\n\nLoan ID: {self.loan_id}\nReporting month: {self.reporting_month}",
            "## Record\n\n" + _render_record(self.record, self.definitions),
        ]

        if self.triggered_rules:
            blocks.append("## Validation rules that fired\n\n" + _render_rules(self.triggered_rules))
        else:
            blocks.append("## Validation rules that fired\n\nNone.")

        if self.model_outputs:
            blocks.append("## Model outputs\n\n" + _render_kv(self.model_outputs))
        if self.anomaly:
            blocks.append("## Anomaly detection output\n\n" + _render_kv(self.anomaly))

        blocks.append("## Your task\n\n" + (self.task or NOTE_TASK))
        return "\n\n".join(blocks)


import re  # noqa: E402  (used by the helpers below)

_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")
_DATE_PATTERN = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")

# The same folding the guardrail applies, so both sides of the comparison
# normalise identically. Importing it keeps them from drifting apart.
from .guardrails import normalise as _normalise  # noqa: E402


def _number_forms(value: float) -> set[str]:
    """
    The string forms a number might legitimately be written in.

    A note that renders 250000.0 as "250,000" or "250000" is quoting the
    context faithfully, and a grounding check that failed it would be
    unusable. Rounded forms are included for the same reason: "8.6%" is a fair
    rendering of 0.08612.
    """
    forms = {
        f"{value:g}",
        f"{value:.0f}",
        f"{value:.1f}",
        f"{value:.2f}",
        f"{abs(value):g}",
        f"{abs(value):.0f}",
    }
    if 0 <= abs(value) <= 1:
        percent = value * 100
        forms.update({f"{percent:.0f}", f"{percent:.1f}", f"{percent:.2f}", f"{percent:g}"})
    return {f.rstrip(".") for f in forms}


def _render_record(record: dict, definitions: dict) -> str:
    lines = ["| Field | Value | Definition |", "| :--- | :--- | :--- |"]
    for key, value in record.items():
        definition = definitions.get(key, "")
        lines.append(f"| `{key}` | {_format(value)} | {definition} |")
    return "\n".join(lines)


def _render_rules(rules: list) -> str:
    lines = ["| Rule | Severity | What it means |", "| :--- | :--- | :--- |"]
    for rule in rules:
        lines.append(
            f"| `{rule.get('rule', '')}` | {rule.get('severity', '')} | {rule.get('description', '')} |"
        )
    return "\n".join(lines)


def _render_kv(mapping: dict) -> str:
    lines = ["| Output | Value |", "| :--- | :--- |"]
    for key, value in mapping.items():
        lines.append(f"| `{key}` | {_format(value)} |")
    return "\n".join(lines)


def _format(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "not available"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_context(
    row: pd.Series,
    definitions: dict,
    rule_specs: list,
    triggered: str = "",
    model_outputs: dict | None = None,
    anomaly: dict | None = None,
    extra_fields: list | None = None,
    task: str = "",
) -> LoanContext:
    """
    Assemble one loan's grounded context.

    ``triggered`` is the semicolon-separated rule string the Phase 5 pipeline
    produces; it is matched back to the rule specifications so the note can
    carry each rule's own description rather than its bare name.
    """
    fields = list(dict.fromkeys([*CORE_FIELDS, *(extra_fields or [])]))
    record = {f: row[f] for f in fields if f in row.index}

    # A queue row with no triggered rules reads back from CSV as float NaN, and
    # NaN is *truthy* -- so `triggered or ""` passes it through and `str()` turns
    # it into a rule literally named "nan". The model then faithfully reports
    # that a rule called `nan` fired with severity "unknown": a fabricated
    # finding produced by perfectly correct, grounded behaviour, and one the
    # guardrails cannot catch because the string really was in its context.
    # Guarded here because every caller funnels through this function.
    if triggered is None or (isinstance(triggered, float) and pd.isna(triggered)):
        triggered = ""

    rule_lookup = {str(spec.get("name", "")): spec for spec in rule_specs or []}
    rules = []
    for name in [part.strip() for part in str(triggered).split(";") if part.strip()]:
        clean = name.split("(")[0].strip()
        spec = rule_lookup.get(clean) or rule_lookup.get(clean.replace("json__", ""))
        rules.append(
            {
                "rule": clean,
                "severity": (spec or {}).get("severity", "unknown"),
                "description": (spec or {}).get("description", "Project-defined consistency check."),
            }
        )

    return LoanContext(
        loan_id=str(row.get(config.ID_COL, "unknown")),
        reporting_month=str(row.get(config.TIME_COL, "unknown"))[:10],
        record=record,
        definitions={k: v for k, v in (definitions or {}).items() if k in record},
        triggered_rules=rules,
        model_outputs=model_outputs or {},
        anomaly=anomaly or {},
        task=task,
    )


def context_to_json(context: LoanContext) -> str:
    """The context as JSON, for the audit log."""
    payload = {
        "loan_id": context.loan_id,
        "reporting_month": context.reporting_month,
        "record": {k: _jsonable(v) for k, v in context.record.items()},
        "triggered_rules": context.triggered_rules,
        "model_outputs": {k: _jsonable(v) for k, v in context.model_outputs.items()},
        "anomaly": {k: _jsonable(v) for k, v in context.anomaly.items()},
    }
    return json.dumps(payload, default=str)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
