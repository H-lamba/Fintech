"""
Loan Performance Intelligence Engine -- single-command entrypoint.

    python main.py                  # every phase, ending in submission/submission.csv
    python main.py --live-copilot   # same, but Phase 8 calls the real LLM provider
    python main.py --submission     # inference and submission only (needs trained models)
    python main.py --model-card     # regenerate the model card from existing reports
    python main.py --sample 2000    # fast pass on 2,000 loans

Phase order is not arbitrary. Profiling produces the record-level data-quality
score the feature matrix consumes; prediction produces both the models the
scenario and submission phases load and the feature dictionary the profiling
report folds in; the anomaly queue is what the copilot summarises. Profiling
therefore runs at both ends -- that is what resolves the one circular
reference, not redundancy.

The copilot runs **offline** here. A default entrypoint must not send data to a
third party or spend an account's credits because someone ran it without
reading the flags; `make copilot-live` is the deliberate act.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

PYTHON = sys.executable


def _phase(name: str, script: str, args: list[str]) -> tuple[str, float, bool]:
    """Run one phase as a subprocess so a crash names the phase that failed."""
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    started = time.perf_counter()
    result = subprocess.run([PYTHON, str(ROOT / "scripts" / script), *args], cwd=ROOT)
    elapsed = time.perf_counter() - started
    ok = result.returncode == 0
    if not ok:
        print(f"\n[FAILED] {name} exited {result.returncode}")
    return name, elapsed, ok


def run_pipeline(
    sample: int | None, skip: set, no_figures: bool, live_copilot: bool = False
) -> list:
    """Every phase, in the order that resolves their dependencies."""
    sample_args = ["--sample", str(sample)] if sample else []
    figure_args = ["--no-figures"] if no_figures else []

    # Offline stays the default -- see the module docstring. `--live-copilot`
    # is the deliberate act that spends credits and sends data to a provider.
    copilot_flag = "--live" if live_copilot else "--offline"
    copilot_label = "live" if live_copilot else "offline"

    plan = [
        ("Phase 1  Data intelligence & profiling (Task 1)", "run_profiling.py",
         (["--sample", str(sample * 30)] if sample else []) + figure_args),
        ("Phase 2-3  Features & loan performance prediction (Task 2)", "run_prediction.py",
         sample_args + ["--score-test"]),
        ("Phase 4  Survival & competing risks (Task 3)", "run_survival.py",
         sample_args + figure_args),
        ("Phase 5  Anomaly & exception detection (Task 4)", "run_anomaly.py",
         sample_args + figure_args),
        ("Phase 6  Scenario & stress simulation (Task 5)", "run_scenario.py",
         sample_args + figure_args),
        ("Phase 7  Explainability & responsible AI (Task 6)", "run_explainability.py",
         sample_args + figure_args),
        (f"Phase 8  LLM reviewer copilot, {copilot_label} (Task 7)", "run_copilot.py",
         [copilot_flag, "--notes", "4"]),
        ("Phase 1b  Re-profile, folding in the feature dictionary", "run_profiling.py",
         (["--sample", str(sample * 30)] if sample else []) + figure_args),
    ]

    timings = []
    for name, script, args in plan:
        key = script.replace("run_", "").replace(".py", "")
        if key in skip:
            print(f"\n[skipped] {name}")
            continue
        timings.append(_phase(name, script, args))
        if not timings[-1][2]:
            raise SystemExit(f"Pipeline stopped: {name} failed.")
    return timings


def build_submission(sample: int | None, verbose: bool = True) -> Path:
    """Score the unlabelled panel and write a validated submission."""
    from src.submission import build as builder
    from src.submission import inference

    print(f"\n{'=' * 72}\nPhase 9  Inference & submission\n{'=' * 72}")
    result = inference.run_inference(sample_loans=sample, verbose=verbose)

    template = builder.load_template()
    submission = builder.build_submission(result.scored, template)
    report = builder.validate(submission, template)

    print("\nValidation:")
    for check in report.checks:
        mark = "ok  " if check["passed"] else "FAIL"
        detail = f"  -- {check['detail']}" if not check["passed"] else ""
        print(f"  [{mark}] {check['check']}{detail}")

    path = builder.write_submission(submission, report)
    report.to_frame().to_csv(config.REPORTS_DIR / "submission_validation.csv", index=False)

    print(f"\nWrote {path} ({len(submission):,} rows x {submission.shape[1]} columns)")
    if not report.passed:
        print("  NOTE: value-level checks failed above; the file was written but is not clean.")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loan Performance Intelligence Engine -- end-to-end pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Run every phase on N loans, for a fast end-to-end check.")
    parser.add_argument("--submission", action="store_true",
                        help="Only run inference and write the submission (needs trained models).")
    parser.add_argument("--model-card", action="store_true",
                        help="Only regenerate the model card from the existing reports.")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Phase keys to skip, e.g. --skip survival scenario copilot.")
    parser.add_argument("--no-figures", action="store_true", help="Skip chart rendering.")
    parser.add_argument("--live-copilot", action="store_true",
                        help="Run Phase 8 against the real LLM provider. Sends loan data "
                             "to a third party and spends account credits; without it the "
                             "copilot runs offline with deterministic stubs.")
    args = parser.parse_args()

    from src.submission import model_card

    if args.model_card:
        print(f"Wrote {model_card.write()}")
        return

    started = time.perf_counter()
    timings = []
    if not args.submission:
        timings = run_pipeline(
            args.sample, set(args.skip), args.no_figures, live_copilot=args.live_copilot
        )

    build_submission(args.sample)
    card = model_card.write()

    print(f"\n{'=' * 72}\nComplete in {time.perf_counter() - started:.0f}s\n{'=' * 72}")
    if timings:
        for name, elapsed, _ in timings:
            print(f"  {elapsed:6.1f}s  {name}")
    print("\nDeliverables:")
    for path in (
        config.SUBMISSION_PATH,
        card,
        config.REPORTS_DIR / "data_intelligence_report.md",
        config.REPORTS_DIR / "task2_model_results.md",
        config.REPORTS_DIR / "survival_report.md",
        config.REPORTS_DIR / "anomaly_report.md",
        config.REPORTS_DIR / "scenario_report.md",
        config.REPORTS_DIR / "explainability_report.md",
        config.REPORTS_DIR / "copilot_report.md",
        config.REPORTS_DIR / "llm_prompt_log.jsonl",
    ):
        marker = "ok" if Path(path).exists() else "--"
        print(f"  [{marker}] {Path(path).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
