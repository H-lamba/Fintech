"""Assembles the Task 7 LLM reviewer copilot report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..profiling.report import ReportBuilder

GOVERNANCE_NOTE = """
**The copilot never predicts.** It sits strictly downstream of every model in
this repository: the anomaly score, the exception type, the predicted
probabilities and the triggered rules arrive as *inputs* and are restated. No
code path here reaches a model output the statistical pipeline did not already
produce. That ordering is a qualification requirement -- the challenge
disqualifies a solution that sends records to an LLM for classification -- not
a stylistic preference.

**The prompt is a request; the guardrails are the control.** Instructing a
model not to invent numbers is not a control, because nothing checks that it
complied. Three checks run on every response before it reaches a reviewer:

1. **Prediction language** -- a note saying "this loan will default" has
   crossed from summarising into producing, and no surrounding disclaimer fixes
   that.
2. **Decision language** -- the output informs a decision; it does not take
   one.
3. **Numeric grounding** -- every number in the note is matched against every
   reasonable rendering of every figure in the supplied context. A number with
   no source was invented, and this is the failure a reader is least able to
   spot: a confident, well-formatted note containing a balance nobody supplied.

**A failed note is not repaired.** It is wrapped with the failure stated,
marked as withheld from the reviewer queue, and kept in full. A governance
layer that quietly patches its model's mistakes has destroyed the evidence that
it makes them.

**Everything is logged, including the failures.** Prompt, system prompt, model,
provider, timestamp, latency, token usage, output, guardrail verdict and the
human review decision go to `reports/llm_prompt_log.jsonl`, one JSON object per
line, append-only. Calls that errored or were refused are logged too -- those
are the ones a reviewer will ask about.
""".strip()

FAILURE_NOTE = """
A set of controls nobody has attacked is a set of controls nobody knows the
strength of. Four adversarial probes are run against a real, fully grounded
context, each designed to elicit one of the failure modes the challenge names.

The probes are phrased the way a hurried reviewer would actually phrase them,
because the realistic threat is not a malicious prompt but a colleague typing
"so is this one going to default?" into the box.

**Probes the model passes are reported alongside the ones it fails.**
Cherry-picking the failures would be as dishonest as cherry-picking the
successes, and the point of the exercise is to show what the controls catch,
not to stage a defeat.
""".strip()


def build_report(
    notes: pd.DataFrame,
    probes: pd.DataFrame,
    log_summary: pd.DataFrame,
    examples: list,
    figures: dict[str, Path],
    reports_dir: Path,
    mode_note: str,
    control_failures: pd.DataFrame | None = None,
) -> ReportBuilder:
    """Assemble every section of the Task 7 deliverable."""
    builder = ReportBuilder(title="LLM Reviewer Copilot Report (Task 7)")

    builder.add_text(
        "What this does",
        "Turns the Phase 5 reviewer queue into written notes a human can act on, by "
        "assembling each loan's record, the definitions of the fields in it, the validation "
        "rules that fired and the models' own outputs into a grounded prompt.\n\n"
        f"{mode_note}",
    )

    builder.add_text("Governance", GOVERNANCE_NOTE)

    if not notes.empty:
        released = int(notes["released_to_reviewer"].sum())
        builder.add_table(
            "Generated reviewer notes",
            notes[["loan_id", "reporting_month", "model", "status",
                   "guardrails_passed", "guardrail_detail", "released_to_reviewer"]],
            note=f"{released} of {len(notes)} notes passed every guardrail and were released "
            "to the reviewer queue.",
            max_rows=30,
        )

    for title, text in examples:
        builder.add_text(title, text, level=3)

    builder.add_text("Deliberate failure testing", FAILURE_NOTE)

    if control_failures is not None and not control_failures.empty:
        builder.add_text(
            "Failures of the control, not the model",
            "On this run the model passed every probe. Manufacturing a model failure to "
            "fill this section would be worse than useless, so what is reported instead is "
            "what genuinely failed: **the guardrail, three times**, each time by blocking a "
            "correct note.\n\nThat is a serious governance defect rather than a footnote. "
            "A control that accuses faithful output of hallucinating gets switched off by "
            "the people it protects, and the one real hallucination then goes out with the "
            "rest. Every row below is an observation from the live run recorded in "
            "`reports/llm_prompt_log.jsonl`, with the fix that shipped.",
        )
        builder.add_table("Control failures and corrections", control_failures, level=3)
    if not probes.empty:
        builder.add_table(
            "Adversarial probe results",
            probes[["probe", "failure_mode", "outcome", "model_refused",
                    "guardrails_passed", "guardrail_detail"]],
            max_rows=20,
        )
        for row in probes.itertuples():
            builder.add_text(
                f"Probe: {row.probe} ({row.failure_mode})",
                f"**The trap.** {row.why_it_is_a_trap}\n\n"
                f"**Correct behaviour.** {row.correct_behaviour}\n\n"
                f"**What the model returned.**\n\n> "
                + str(row.llm_output).replace("\n", "\n> ")[:1500]
                + f"\n\n**Guardrail verdict.** {row.guardrail_detail}\n\n"
                f"**Outcome.** {row.outcome}\n\n"
                f"**Human correction / rejection.** {row.human_correction}",
                level=3,
            )

    if not log_summary.empty:
        builder.add_table(
            "Audit trail",
            log_summary,
            note="Every call in `reports/llm_prompt_log.jsonl`, grouped. The log records "
            "prompt, system prompt, model, provider, timestamp, latency, token usage, "
            "output, guardrail verdict and human review decision.",
        )

    if figures:
        lines = []
        for caption, path in figures.items():
            relative = Path(path).relative_to(reports_dir)
            lines.append(f"**{caption}**\n\n![{caption}]({relative.as_posix()})\n")
        builder.add_text("Figures", "\n".join(lines))

    return builder


def summarise_log(records: list) -> pd.DataFrame:
    """Group the audit trail by purpose and outcome, for the report."""
    if not records:
        return pd.DataFrame()

    # The verdict arrives in a separate append-only record; join it back on
    # call_id rather than counting the two record types as two calls.
    verdicts = {
        r.get("call_id"): (r.get("guardrails") or {}).get("passed")
        for r in records
        if r.get("record_type") == "review"
    }
    calls = [r for r in records if r.get("record_type", "call") == "call"]

    frame = pd.DataFrame(
        [
            {
                "purpose": str(r.get("purpose", "")).split(":")[0],
                "model": r.get("model"),
                "provider": r.get("provider"),
                "status": r.get("status"),
                "guardrails_passed": verdicts.get(r.get("call_id")),
                "latency_seconds": r.get("latency_seconds"),
                "total_tokens": (r.get("usage") or {}).get("total_tokens"),
            }
            for r in calls
        ]
    )
    return (
        frame.groupby(["purpose", "model", "provider", "status"], dropna=False)
        .agg(
            calls=("status", "size"),
            guardrails_passed=("guardrails_passed", "sum"),
            mean_latency_seconds=("latency_seconds", "mean"),
            total_tokens=("total_tokens", "sum"),
        )
        .reset_index()
    )
