"""
Phase 7 pipeline: Task 6 -- explainability and responsible AI.

    python scripts/run_explainability.py                # all three models, ~60s
    python scripts/run_explainability.py --sample 2000  # 2,000 loans, fast pass
    python scripts/run_explainability.py --shap-rows 5000

Requires the Phase 3 models: run `make predict` first.

Outputs:
    reports/explainability_report/           <- every plot and table
    reports/explainability_report.md / .html <- the graded deliverable
    reports/model_card.md                    <- the section 11 requirement
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_io, features as feature_module  # noqa: E402
from src.explain import errors, fairness, figures, report, shap_values  # noqa: E402
from src.models import estimators, splitting  # noqa: E402
from src.scenario.project import load_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 7 (Task 6) explainability and responsible AI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Use N randomly chosen LOANS (whole histories kept) for a fast pass.")
    parser.add_argument("--shap-rows", type=int, default=config.SHAP_SAMPLE_ROWS,
                        help="Rows sampled (stratified on the outcome) for SHAP values.")
    parser.add_argument("--no-figures", action="store_true", help="Skip chart rendering.")
    return parser.parse_args()


def _sample_loans(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    loans = panel[config.ID_COL].unique()
    return panel[panel[config.ID_COL].isin(rng.choice(loans, size=min(n, len(loans)), replace=False))]


def main() -> None:
    args = parse_args()
    outdir = config.EXPLAINABILITY_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    reports_dir = config.REPORTS_DIR

    # ------------------------------------------------------------ 1. inputs
    print("Rebuilding the Phase 3 feature matrix...")
    panel = data_io.load_train()
    if panel.empty:
        raise FileNotFoundError(f"No panel found at {config.TRAIN_PATH}")
    if args.sample:
        panel = _sample_loans(panel, args.sample)

    static = data_io.load_static()
    dq = pd.read_csv(config.DQ_SCORES_PATH, low_memory=False) if config.DQ_SCORES_PATH.exists() else None
    frame, _ = feature_module.build_feature_matrix(panel, static=static, dq_scores=dq)

    models = load_models(targets=config.EXPLAIN_TARGETS, variant="improved")
    boundaries = splitting.SplitBoundaries.from_config()

    global_frames, verification_rows = [], []
    reliability_tables, calibration_rows, confidence_rows = {}, [], []
    importances, results = {}, {}
    confusion_rows, error_segment_frames, characterisation_frames = [], [], []
    fairness_group_frames, disparity_frames, monotonicity_rows = [], [], []
    local_examples_frames, local_tables = [], {}
    rendered: dict[str, Path] = {}

    for target, label in config.EXPLAIN_TARGETS.items():
        model = models[target]
        print(f"\n== {label} ({target}) ==")

        # The exact Phase 3 holdout: same boundaries, same purge horizon.
        split = splitting.make_time_split(frame, target, boundaries=boundaries)
        test = split.test.reset_index(drop=True)
        y_true = test[target].astype(int).to_numpy()

        # Deployed model: calibrated probability at the tuned threshold.
        probability = model.predict_proba(test)[:, 1]
        outcome = errors.classify_predictions(y_true, probability, model.threshold)
        print(f"  test={len(test):,} rows, {y_true.mean():.2%} positive, threshold={model.threshold:.3f}")

        # ------------------------------------------------------- 2. SHAP
        result = shap_values.explain_model(
            model, test, target, label, outcome=y_true, n_rows=args.shap_rows
        )
        results[label] = result
        check = shap_values.verify_against_booster(result, model)
        verification_rows.append({"model": label, "rows_explained": len(result), **check})
        print(f"  SHAP on {len(result):,} rows | matches booster: {check.get('agrees')}")

        importance = shap_values.global_importance(result)
        importances[label] = importance
        global_frames.append(importance.head(15))

        # ------------------------------------------- 3. local explanations
        # `result.positions` indexes back into `test`, so probabilities and
        # outcomes line up with the sampled rows without any index guesswork.
        picks = shap_values.pick_demo_loans(
            result, probability[result.positions], y_true[result.positions]
        )
        if not picks.empty:
            picks.insert(0, "model", label)
            print(f"  local explanations: {', '.join(picks['case'])}")
            local_examples_frames.append(picks)
            for row in picks.itertuples():
                explanation = shap_values.local_explanation(result, int(row.position))
                name = f"{label} -- {row.case}"
                local_tables[name] = explanation
                # The full case, not its first word: "confident true positive"
                # and "confident false positive" both start with "confident",
                # and a collision here silently drops one of the four demo cases.
                slug = re.sub(r"[^a-z0-9]+", "_", row.case.lower()).strip("_")
                explanation.to_csv(outdir / f"local_{label}_{slug}.csv", index=False)
                if not args.no_figures:
                    path = figures.plot_waterfall(
                        explanation, outdir,
                        f"waterfall_{label}_{slug}.png",
                        subtitle=f"{label} model | {row.case} | loan {getattr(row, config.ID_COL)} | "
                                 f"predicted {row.predicted_probability:.1%}, actual "
                                 f"{'event' if row.actual_outcome else 'no event'}",
                    )
                    if path:
                        rendered[f"Local explanation: {name}"] = path

        # ------------------------------------------------- 4. error analysis
        summary = errors.confusion_summary(outcome)
        summary.insert(0, "model", label)
        confusion_rows.append(summary)

        for segment in config.EXPLAIN_SEGMENTS:
            table = errors.error_rates_by_segment(test, outcome, segment)
            if not table.empty:
                table.insert(0, "model", label)
                error_segment_frames.append(table)

        characterisation = errors.characterise_errors(test, outcome, model.numeric)
        if not characterisation.empty:
            characterisation.insert(0, "model", label)
            characterisation_frames.append(characterisation)

        # --------------------------------------------- 5. reliability
        table = errors.reliability_table(y_true, probability)
        reliability_tables[label] = table
        ece = errors.expected_calibration_error(table)
        calibration_rows.append(
            {
                "model": label,
                "expected_calibration_error": ece,
                "mean_predicted": float(probability.mean()),
                "observed_rate": float(y_true.mean()),
                "bias": float(probability.mean() - y_true.mean()),
            }
        )
        confidence = errors.confidence_profile(probability, model.threshold)
        confidence.insert(0, "model", label)
        confidence_rows.append(confidence)
        print(f"  ECE={ece:.4f} | mean predicted {probability.mean():.2%} vs observed {y_true.mean():.2%}")

        # ------------------------------------------------- 6. disparity
        groups, disparity = fairness.audit(test, outcome)
        if not groups.empty:
            groups.insert(0, "model", label)
            fairness_group_frames.append(groups)
            monotonicity_rows.append(
                {"model": label, **fairness.monotonicity_check(
                    groups, "credit_score_band", list(config.CREDIT_SCORE_BANDS)
                )}
            )
        if not disparity.empty:
            disparity.insert(0, "model", label)
            disparity_frames.append(disparity)
            escalated = int(disparity["escalate"].sum())
            print(f"  disparity: {escalated} finding(s) escalated on non-risk-factor segments")

        if not args.no_figures:
            path = figures.plot_beeswarm(result, outdir, max_points=config.SHAP_PLOT_ROWS)
            if path:
                rendered[f"What drives the {label} model"] = path

    # ---------------------------------------------------------- assemble
    def _concat(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    global_importance = _concat(global_frames)
    verification = pd.DataFrame(verification_rows)
    confusion = _concat(confusion_rows)
    error_segments = _concat(error_segment_frames)
    characterisation = _concat(characterisation_frames)
    calibration_summary = pd.DataFrame(calibration_rows)
    confidence_table = _concat(confidence_rows)
    fairness_groups = _concat(fairness_group_frames)
    disparity_table = _concat(disparity_frames)
    monotonicity = pd.DataFrame(monotonicity_rows)
    local_examples = _concat(local_examples_frames)

    reliability = pd.concat(
        [t.assign(model=label) for label, t in reliability_tables.items()], ignore_index=True
    ) if reliability_tables else pd.DataFrame()

    if not args.no_figures:
        print("\nRendering figures...")
        for caption, path in (
            ("Global feature importance", figures.plot_global_importance(importances, outdir)),
            ("Reliability", figures.plot_reliability(reliability_tables, outdir)),
        ):
            if path:
                rendered[caption] = path
        for segment in ("credit_score_band", "vintage_year", "servicer_name"):
            subset = fairness_groups[fairness_groups.get("model") == "default"] if not fairness_groups.empty else pd.DataFrame()
            path = figures.plot_error_rates(subset, outdir, segment)
            if path:
                rendered[f"Error rates by {segment.replace('_', ' ')} (default model)"] = path

    for name, table in {
        "global_importance": global_importance,
        "shap_verification": verification,
        "local_examples": local_examples,
        "confusion_summary": confusion,
        "error_rates_by_segment": error_segments,
        "false_positive_characterisation": characterisation,
        "reliability": reliability,
        "calibration_summary": calibration_summary,
        "confidence_profile": confidence_table,
        "fairness_groups": fairness_groups,
        "disparity_summary": disparity_table,
        "monotonicity_check": monotonicity,
    }.items():
        if not table.empty:
            table.to_csv(outdir / f"{name}.csv", index=False)

    builder = report.build_report(
        global_importance=global_importance,
        verification=verification,
        local_examples=local_examples,
        local_tables=local_tables,
        confusion=confusion,
        error_segments=error_segments,
        error_characterisation=characterisation,
        reliability=reliability,
        calibration_summary=calibration_summary,
        confidence=confidence_table,
        fairness_groups=fairness_groups,
        disparity=disparity_table,
        monotonicity=monotonicity,
        figures=rendered,
        reports_dir=reports_dir,
    )
    builder.save(reports_dir / "explainability_report.md", reports_dir / "explainability_report.html")

    card = report.build_model_card(
        global_importance, disparity_table, calibration_summary, error_segments,
        results_path=outdir / "global_importance.csv",
    )
    (reports_dir / "model_card.md").write_text(card)

    print(f"\nWrote {reports_dir / 'explainability_report.md'} and {reports_dir / 'model_card.md'}")


if __name__ == "__main__":
    main()
