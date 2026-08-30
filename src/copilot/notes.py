"""
Generating reviewer notes from grounded context.

The copilot sits **strictly downstream** of every model in this repository. It
receives the anomaly score, the exception type, the predicted probabilities and
the triggered rules as *inputs*, and it restates them. It never computes one.

That ordering is a qualification requirement, not a preference: the challenge
disqualifies a solution that sends records to an LLM for classification. The
system prompt says so, the guardrails check it, and the pipeline structure
makes it true -- there is no code path here that reaches a model output the
statistical pipeline did not already produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import guardrails
from .llm_client import LLMResponse, call_llm, log_verdict
from .retrieval import ACTION_TASK, LoanContext, build_context, context_to_json


@dataclass
class ReviewerNote:
    """One generated note, its verdict, and whether it may reach a reviewer."""

    loan_id: str
    reporting_month: str
    note: str
    raw_output: str
    verdict: guardrails.GuardrailVerdict
    response: LLMResponse
    context: LoanContext = None
    released: bool = True

    def to_row(self) -> dict:
        return {
            "loan_id": self.loan_id,
            "reporting_month": self.reporting_month,
            "call_id": self.response.call_id,
            "model": self.response.model,
            "provider": self.response.provider,
            "status": self.response.status,
            "guardrails_passed": self.verdict.passed,
            "guardrail_detail": self.verdict.summary(),
            "released_to_reviewer": self.released,
            "note": self.note,
        }


def generate_note(
    row: pd.Series,
    definitions: dict,
    rule_specs: list,
    triggered: str = "",
    model_outputs: dict | None = None,
    anomaly: dict | None = None,
    offline: bool | None = None,
    task: str = "",
    purpose: str = "reviewer_note",
    **call_kwargs,
) -> ReviewerNote:
    """
    Assemble context, call the model, check the output, wrap it.

    A note that fails a guardrail is **not** silently rewritten. It is wrapped
    with the failure stated, marked as withheld, and kept in full -- because a
    governance layer that quietly patches its model's mistakes has destroyed
    the evidence that it makes them.
    """
    context = build_context(
        row, definitions, rule_specs,
        triggered=triggered, model_outputs=model_outputs, anomaly=anomaly, task=task,
    )

    response = call_llm(
        context.to_prompt(),
        purpose=purpose,
        context_json=context_to_json(context),
        offline=offline,
        **call_kwargs,
    )

    verdict = (
        guardrails.check(response.output, context.grounded_numbers())
        if response.status != "error"
        else guardrails.GuardrailVerdict(passed=False, notes=[response.error or "call failed"])
    )
    response.guardrails = verdict.to_dict()
    response.human_review = {
        "released_to_reviewer": verdict.passed,
        "action": "released" if verdict.passed else "withheld from the reviewer queue",
    }
    log_verdict(response)

    return ReviewerNote(
        loan_id=context.loan_id,
        reporting_month=context.reporting_month,
        note=guardrails.wrap(response.output, verdict),
        raw_output=response.output,
        verdict=verdict,
        response=response,
        context=context,
        released=verdict.passed,
    )


def generate_action(
    row: pd.Series,
    definitions: dict,
    rule_specs: list,
    triggered: str = "",
    model_outputs: dict | None = None,
    anomaly: dict | None = None,
    offline: bool | None = None,
    **call_kwargs,
) -> ReviewerNote:
    """
    Suggested **verification steps** for one loan-month.

    Deliberately not "the recommended action on the loan". The rule layer
    already emits a suggested action, and it is deterministic and auditable;
    what the model adds is the reviewer's checklist for confirming or refuting
    it. Framing it as verification is what keeps this call on the correct side
    of the decision guardrail rather than fighting it -- the guardrail still
    runs, and a response that slips into recommending an outcome is withheld
    exactly like any other.
    """
    return generate_note(
        row, definitions, rule_specs,
        triggered=triggered, model_outputs=model_outputs, anomaly=anomaly,
        offline=offline, task=ACTION_TASK, purpose="verification_steps",
        **call_kwargs,
    )


def generate_batch(
    queue: pd.DataFrame,
    panel: pd.DataFrame,
    definitions: dict,
    rule_specs: list,
    limit: int = 8,
    offline: bool | None = None,
    verbose: bool = True,
) -> list[ReviewerNote]:
    """
    Notes for the top of the Phase 5 reviewer queue.

    One call per loan, sequentially. Not parallelised on purpose: the point of
    this layer is a complete, ordered audit trail, and concurrent writes to an
    append-only log buy a few seconds at the cost of the thing the log is for.
    """
    notes: list[ReviewerNote] = []
    # Sorted so the repeated .loc lookups below hit a lexsorted index rather
    # than falling back to a scan per row.
    lookup = panel.set_index(
        ["loan_id", panel["reporting_month"].astype(str).str[:7]]
    ).sort_index()

    for position, item in enumerate(queue.head(limit).itertuples(), start=1):
        key = (item.loan_id, str(item.reporting_month)[:7])
        if key not in lookup.index:
            continue
        row = lookup.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row = row.copy()
        row["loan_id"] = item.loan_id
        row["reporting_month"] = item.reporting_month

        model_outputs = {
            field: getattr(item, field, None)
            for field in ("hybrid_score", "exception_probability", "predicted_exception_type")
            if hasattr(item, field)
        }
        anomaly = {
            "top_drivers": getattr(item, "top_drivers", None),
            "suggested_action_from_rules": getattr(item, "suggested_action", None),
        }

        note = generate_note(
            row, definitions, rule_specs,
            triggered=getattr(item, "triggered_rules", "") or "",
            model_outputs=model_outputs, anomaly=anomaly, offline=offline,
        )
        notes.append(note)

        if verbose:
            state = "released" if note.released else f"WITHHELD ({note.verdict.summary()})"
            print(f"  [{position}] {note.loan_id} {note.reporting_month} -- {state}")

    return notes


def notes_frame(notes: list) -> pd.DataFrame:
    return pd.DataFrame([note.to_row() for note in notes]) if notes else pd.DataFrame()
