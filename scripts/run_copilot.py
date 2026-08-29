"""
Phase 8 pipeline: Task 7 -- the LLM-assisted reviewer copilot.

    python scripts/run_copilot.py --offline     # no API calls; exercises everything
    python scripts/run_copilot.py --live        # real calls, costs API credits
    python scripts/run_copilot.py --live --notes 3 --no-probes

Requires the Phase 5 reviewer queue: run `make anomaly` first.

**--offline is the default.** This script sends loan data to a third-party API
and spends the account's credits, so a live run has to be asked for explicitly
rather than being what happens if you forget a flag.

Outputs:
    reports/llm_prompt_log.jsonl       <- the mandatory audit trail (append-only)
    reports/copilot_report.md / .html  <- the graded deliverable
    reports/copilot/*.csv              <- notes, probe results, log summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_io  # noqa: E402
from src.copilot import failures, llm_client, notes as notes_module, report  # noqa: E402
from src.copilot.retrieval import build_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 8 (Task 7) LLM reviewer copilot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", default=True,
                      help="Deterministic stubs, no API calls. The default.")
    mode.add_argument("--live", action="store_true",
                      help="Make real API calls. Sends loan data to the configured "
                           "provider and spends account credits.")
    parser.add_argument("--notes", type=int, default=6,
                        help="Reviewer notes to generate from the top of the Phase 5 queue.")
    parser.add_argument("--no-probes", action="store_true",
                        help="Skip the adversarial failure probes.")
    parser.add_argument("--model", default=None, help="Override the configured model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offline = not args.live
    reports_dir = config.REPORTS_DIR
    outdir = reports_dir / "copilot"
    outdir.mkdir(parents=True, exist_ok=True)

    if offline:
        mode_note = (
            "**This run was offline.** No API call was made; responses are deterministic "
            "stubs marked `provider: offline-stub` in the audit trail, and nothing in this "
            "report presents one as a real model generation. Re-run with `--live` and a key "
            "in `.env` for genuine responses."
        )
        print("Mode: OFFLINE (no API calls). Use --live for real responses.")
    else:
        if not llm_client.has_credentials():
            raise SystemExit(
                "--live needs a key. Copy .env.example to .env and set XAI_API_KEY."
            )
        mode_note = (
            f"**This run was live** against `{llm_client.LLM_BASE_URL}` using "
            f"`{args.model or llm_client.LLM_MODEL}`. Every call is in the audit trail."
        )
        print(
            f"Mode: LIVE against {llm_client.LLM_BASE_URL} "
            f"({args.model or llm_client.LLM_MODEL})\n"
            f"  key from .env:{llm_client.ENV_KEY_USED} -> provider detected: "
            f"{llm_client.PROVIDER_NAME}"
        )

    call_kwargs = {"model": args.model} if args.model else {}

    # ---------------------------------------------------------- 1. context
    print("Loading grounding sources...")
    queue_path = reports_dir / "anomaly_examples.csv"
    if not queue_path.exists():
        raise FileNotFoundError(
            f"No reviewer queue at {queue_path}. Run `make anomaly` first -- the copilot "
            "summarises the pipeline's output rather than producing its own."
        )
    queue = pd.read_csv(queue_path)
    panel = data_io.load_train()
    definitions = data_io.load_data_dictionary()
    rule_specs = data_io.load_validation_rules()
    print(f"  queue={len(queue)} rows | dictionary={len(definitions)} fields | "
          f"rules={len(rule_specs)}")

    # ----------------------------------------------------- 2. reviewer notes
    print(f"\nGenerating {min(args.notes, len(queue))} reviewer notes...")
    generated = notes_module.generate_batch(
        queue, panel, definitions, rule_specs,
        limit=args.notes, offline=offline, **call_kwargs,
    )
    notes_table = notes_module.notes_frame(generated)

    if not notes_table.empty:
        released = int(notes_table["released_to_reviewer"].sum())
        print(f"  {released}/{len(notes_table)} released; "
              f"{len(notes_table) - released} withheld by guardrails")

    # ------------------------------------------------------ 3. failure probes
    probe_table = pd.DataFrame()
    if not args.no_probes and generated:
        print("\nRunning adversarial probes...")
        results = failures.run_all(
            generated[0].context, offline=offline, **call_kwargs
        )
        probe_table = failures.results_frame(results)

    # ------------------------------------------------------------ 4. outputs
    examples = []
    for note in generated[:2]:
        examples.append((f"Example note: loan {note.loan_id} ({note.reporting_month})",
                         "```\n" + note.note + "\n```"))

    records = llm_client.load_prompt_log()
    log_summary = report.summarise_log(records)
    n_calls = sum(1 for r in records if r.get("record_type", "call") == "call")
    n_reviews = len(records) - n_calls
    print(f"\nAudit trail: {n_calls} call record(s) + {n_reviews} verdict record(s) "
          f"in {llm_client.LOG_PATH}")

    for name, table in {
        "reviewer_notes": notes_table,
        "adversarial_probes": probe_table,
        "log_summary": log_summary,
        "control_failures": failures.control_failures_frame(),
    }.items():
        if not table.empty:
            table.to_csv(outdir / f"{name}.csv", index=False)

    builder = report.build_report(
        notes=notes_table, probes=probe_table, log_summary=log_summary,
        examples=examples, figures={}, reports_dir=reports_dir, mode_note=mode_note,
        control_failures=failures.control_failures_frame(),
    )
    builder.save(reports_dir / "copilot_report.md", reports_dir / "copilot_report.html")
    print(f"Wrote {reports_dir / 'copilot_report.md'}")


if __name__ == "__main__":
    main()
