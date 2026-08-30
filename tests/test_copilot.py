"""
Regression tests for the Task 7 reviewer copilot.

The controls here are the deliverable. A prompt that *asks* a model not to
invent numbers is not a control; the check that catches it when it does is, and
these tests are what stop that check from quietly weakening.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.copilot import failures, guardrails, llm_client, retrieval


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------
def test_prediction_language_is_blocked():
    """
    Using an LLM to classify is a disqualification condition, so a note that
    produces its own forecast must never reach a reviewer.
    """
    verdict = guardrails.check("Based on the trend, this loan will default next year.", set())
    assert verdict.passed is False
    assert verdict.prediction_language


def test_decision_language_is_blocked_including_gerunds():
    """"I recommend denying" is the same act as "I recommend deny"."""
    for text in (
        "I recommend denying the modification.",
        "You should approve the loan.",
        "Reject this application.",
    ):
        assert guardrails.check(text, set()).passed is False, text


def test_describing_what_to_verify_is_allowed():
    """
    The copilot may tell a reviewer what to check. Blocking that would leave it
    unable to say anything useful.
    """
    verdict = guardrails.check(
        "The reviewer should verify the reported balance against the servicer file.",
        {"1"},
    )
    assert verdict.passed is True


def test_a_number_not_in_the_context_is_caught():
    """
    The failure a reader is least able to spot: a confident, well-formatted
    note containing a balance nobody supplied.
    """
    verdict = guardrails.check("The balance is 987,654 and the loan is 108 days past due.",
                               {"108", "250000"})
    assert verdict.passed is False
    assert "987,654" in verdict.ungrounded_numbers


def test_reformatted_context_numbers_are_not_flagged():
    """
    A note rendering 250000.0 as "250,000" is quoting faithfully. A grounding
    check that failed it would be unusable and would be switched off.
    """
    grounded = retrieval.LoanContext(
        loan_id="L1", reporting_month="2023-06",
        record={"current_balance": 250000.0, "days_past_due": 108},
    ).grounded_numbers()

    verdict = guardrails.check("Balance 250,000 with 108 days past due.", grounded)
    assert verdict.ungrounded_numbers == []
    assert verdict.passed is True


def test_vagueness_is_recorded_but_does_not_block():
    """
    Hedging makes a note weak, not unsafe. Failing on it would suppress output
    for a stylistic reason.
    """
    verdict = guardrails.check("Generally speaking, the loan looks fine.", {"1"})
    assert verdict.vague_language
    assert verdict.passed is True


def test_disclaimer_is_attached_to_both_ends():
    """
    A reviewer scanning a queue reads the first line; one pasting a note into a
    case file carries the last one with it.
    """
    wrapped = guardrails.wrap("Some note text.", guardrails.GuardrailVerdict(passed=True))
    assert wrapped.startswith("[RECOMMENDATION, NOT A DECISION")
    assert wrapped.rstrip().endswith("]")
    assert wrapped.count("RECOMMENDATION, NOT A DECISION") == 2


def test_a_failed_note_says_so_and_is_not_rewritten():
    """
    A governance layer that quietly patches its model's mistakes has destroyed
    the evidence that it makes them.
    """
    verdict = guardrails.check("This loan will default.", set())
    wrapped = guardrails.wrap("This loan will default.", verdict)
    assert "GUARDRAIL FAILED" in wrapped
    assert "This loan will default." in wrapped  # kept in full, not edited


# --------------------------------------------------------------------------
# Grounded context
# --------------------------------------------------------------------------
@pytest.fixture
def loan_row() -> pd.Series:
    return pd.Series(
        {
            "loan_id": "ABC123456789",
            "reporting_month": "2023-06-01",
            "current_status": "90-DPD",
            "days_past_due": 108,
            "current_balance": 250000.0,
            "credit_score_band": "620-659",
            "state": "CA",
        }
    )


def test_context_carries_definitions_only_for_included_fields(loan_row):
    context = retrieval.build_context(
        loan_row,
        definitions={"days_past_due": "Days delinquent.", "unused_field": "Not in the record."},
        rule_specs=[],
    )
    assert "days_past_due" in context.definitions
    assert "unused_field" not in context.definitions


def test_triggered_rules_are_matched_to_their_descriptions(loan_row):
    context = retrieval.build_context(
        loan_row,
        definitions={},
        rule_specs=[{"name": "BALANCE_CEILING", "severity": "high", "description": "Balance ceiling."}],
        triggered="json__BALANCE_CEILING; some_project_rule",
    )
    assert len(context.triggered_rules) == 2
    assert context.triggered_rules[0]["description"] == "Balance ceiling."
    assert context.triggered_rules[0]["severity"] == "high"


def test_prompt_forbids_prediction_and_decision(loan_row):
    prompt = retrieval.build_context(loan_row, {}, []).to_prompt()
    lowered = prompt.lower()
    assert "do not estimate, predict" in lowered
    assert "do not state a decision" in lowered


def test_grounded_numbers_cover_the_supplied_record(loan_row):
    grounded = retrieval.build_context(loan_row, {}, []).grounded_numbers()
    assert "108" in grounded
    assert "250000" in grounded


# --------------------------------------------------------------------------
# Client, logging and parsing
# --------------------------------------------------------------------------
def test_offline_mode_makes_no_call_and_is_labelled(tmp_path):
    """
    An offline response must never be presentable as a real generation.
    """
    log = tmp_path / "log.jsonl"
    response = llm_client.call_llm("Loan ID: X\nReporting month: 2023-01", offline=True, log_path=log)

    assert response.status == "offline_stub"
    assert response.provider == "offline-stub"
    assert "OFFLINE STUB" in response.output


def test_every_call_is_logged_with_the_audit_fields(tmp_path):
    log = tmp_path / "log.jsonl"
    llm_client.call_llm("prompt text", offline=True, log_path=log)
    records = llm_client.load_prompt_log(log)

    assert len(records) == 1
    for key in ("call_id", "timestamp", "model", "prompt", "system_prompt", "output", "status"):
        assert key in records[0], key


def test_the_verdict_is_a_second_linked_record_not_a_mutation(tmp_path):
    """
    Append-only is what makes the log evidence. The call record is written the
    instant the call returns, so a crash before the check cannot erase the fact
    that it happened.
    """
    log = tmp_path / "log.jsonl"
    response = llm_client.call_llm("prompt", offline=True, log_path=log)
    response.guardrails = {"passed": False}
    llm_client.log_verdict(response, log)

    records = llm_client.load_prompt_log(log)
    assert [r["record_type"] for r in records] == ["call", "review"]
    assert records[0]["call_id"] == records[1]["call_id"]
    assert records[1]["guardrails"]["passed"] is False


def test_a_malformed_log_line_does_not_hide_the_rest(tmp_path):
    """The log is evidence; a reader that refuses to open it is worse than a partial read."""
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"call_id": "a"}) + "\n{ truncated\n" + json.dumps({"call_id": "b"}) + "\n")
    records = llm_client.load_prompt_log(log)
    assert [r["call_id"] for r in records] == ["a", "b"]


def test_json_is_recovered_from_a_fenced_or_prose_wrapped_reply():
    """Models wrap JSON in prose often enough that a bare json.loads is not a parser."""
    fenced = llm_client.LLMResponse(
        call_id="1", timestamp="t", model="m", provider="p", system_prompt="", prompt="",
        output='Here you go:\n```json\n{"risk": "high"}\n```\nHope that helps.',
    )
    payload, error = llm_client.parse_json_output(fenced)
    assert payload == {"risk": "high"} and error is None

    broken = llm_client.LLMResponse(
        call_id="2", timestamp="t", model="m", provider="p", system_prompt="", prompt="",
        output="no json here at all",
    )
    payload, error = llm_client.parse_json_output(broken)
    assert payload is None and error


# --------------------------------------------------------------------------
# Failure probes
# --------------------------------------------------------------------------
def test_every_probe_declares_its_trap_and_its_correction():
    """
    A failure example without the human decision beside it is an anecdote.
    Task 7 is graded on the correction, not on the failure.
    """
    assert len(failures.PROBES) >= 3
    for probe in failures.PROBES:
        assert probe.why_it_is_a_trap.strip()
        assert probe.correct_behaviour.strip()
        assert probe.human_correction.strip()


def test_probes_cover_the_three_named_failure_modes():
    modes = " ".join(p.failure_mode for p in failures.PROBES).lower()
    assert "overconfident" in modes
    assert "hallucinated" in modes
    assert "vague" in modes


def test_refusal_detection_reads_a_plain_refusal():
    assert failures.looks_like_refusal("Employment status is not in the context provided.")
    assert not failures.looks_like_refusal("The borrower earns 90,000 per year.")


# --------------------------------------------------------------------------
# Grounding the rule block
# --------------------------------------------------------------------------
@pytest.mark.parametrize("empty", [float("nan"), None, "", "   "])
def test_a_record_with_no_triggered_rules_grounds_no_rules(empty):
    """
    A queue row with no triggered rules reads back from CSV as float NaN, and
    **NaN is truthy** -- so `triggered or ""` passes it through and `str()`
    turns it into a rule literally named "nan". The model then faithfully
    reports that a rule called `nan` fired with severity "unknown".

    That is the worst class of defect this layer can produce: a fabricated
    finding generated by correct, grounded behaviour, which the guardrails
    cannot catch because the string genuinely was in the supplied context.
    """
    row = pd.Series({"loan_id": "L1", "reporting_month": "2024-01-01",
                     "current_status": "Current"})

    context = retrieval.build_context(row, {}, [], triggered=empty)

    assert context.triggered_rules == []
    assert "## Validation rules that fired\n\nNone." in context.to_prompt()


def test_a_real_triggered_rule_still_reaches_the_prompt():
    """The guard above must not silence genuine rule hits."""
    row = pd.Series({"loan_id": "L1", "reporting_month": "2024-01-01"})
    specs = [{"name": "BALANCE_CEILING", "severity": "high", "description": "Balance check."}]

    context = retrieval.build_context(
        row, {}, specs, triggered="json__BALANCE_CEILING; other_rule"
    )

    assert [r["rule"] for r in context.triggered_rules] == ["json__BALANCE_CEILING", "other_rule"]
    assert context.triggered_rules[0]["severity"] == "high"


# --------------------------------------------------------------------------
# The explorer's second ask
# --------------------------------------------------------------------------
def test_the_action_task_asks_for_verification_never_a_decision():
    """
    The explorer's "Recommend action" button must not become a way around the
    decision guardrail. It asks what a reviewer should *verify*; the rule layer
    is what emits a suggested action, and it is deterministic and auditable.
    """
    task = retrieval.ACTION_TASK.lower()

    assert "verif" in task
    assert "do not state a decision" in task
    assert "do not estimate, predict, or infer" in task


def test_the_two_asks_share_one_grounded_context():
    """
    The question may vary between the note and the checklist. The grounding may
    not -- otherwise the two answers are checked against different facts.
    """
    row = pd.Series({"loan_id": "L1", "reporting_month": "2024-01-01",
                     "current_balance": 250000.0})

    note = retrieval.build_context(row, {}, [], task="")
    action = retrieval.build_context(row, {}, [], task=retrieval.ACTION_TASK)

    assert note.record == action.record
    assert note.grounded_numbers() - {"3", "5"} <= action.grounded_numbers() | {"3", "5"}
    assert retrieval.ACTION_TASK in action.to_prompt()
    assert retrieval.NOTE_TASK in note.to_prompt()
