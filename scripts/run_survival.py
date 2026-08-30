"""
Phase 4 pipeline: Task 3 -- time-to-event / survival modelling.

    python scripts/run_survival.py                    # full run, ~30s
    python scripts/run_survival.py --sample 2000      # 2,000 loans, fast pass
    python scripts/run_survival.py --no-left-truncation   # ablation

Outputs (all under reports/):
    survival_report.md / .html    <- graded deliverable, includes the censoring note
    survival/*.png                <- event curves, segmented and compared
    survival/*.csv                <- every table behind the report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_io  # noqa: E402
from src.survival import baselines, curves, dataset, evaluation, models, report  # noqa: E402

CAUSES = (config.EVENT_DEFAULT, config.EVENT_PREPAID)

# Credit and LTV bands are ordinal; plotting them in their natural order is the
# difference between a scan and a puzzle.
SEGMENT_ORDERS = {
    "credit_score_band": ["<620", "620-659", "660-699", "700-739", "740-799", "800+"],
    "ltv_band": ["<60%", "60-75%", "75-80%", "80-90%", "90-97%"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 (Task 3) survival and competing-risk modelling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Use N randomly chosen loans (whole histories kept) for a fast pass.")
    parser.add_argument("--vintage-cutoff", default=config.SURVIVAL_TRAIN_VINTAGE_END,
                        help="Last origination month in the training set; later vintages are the holdout.")
    parser.add_argument("--no-left-truncation", action="store_true",
                        help="Assume every loan was observed from month 0 (ablation).")
    parser.add_argument("--penalizer", type=float, default=0.01,
                        help="Ridge penalty on the Cox models.")
    parser.add_argument("--calibration-horizon", type=int, default=24,
                        help="Months on book at which the calibration table is built.")
    parser.add_argument("--no-figures", action="store_true", help="Skip chart rendering.")
    return parser.parse_args()


def _sample_loans(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    loans = panel[config.ID_COL].unique()
    keep = rng.choice(loans, size=min(n, len(loans)), replace=False)
    return panel[panel[config.ID_COL].isin(keep)]


def main() -> None:
    args = parse_args()
    reports_dir = config.REPORTS_DIR
    outdir = reports_dir / "survival"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading panel...")
    panel = data_io.load_train()
    static = data_io.load_static()
    if panel.empty:
        raise FileNotFoundError(f"No panel found at {config.TRAIN_PATH}")
    if args.sample:
        panel = _sample_loans(panel, args.sample)

    # ---------------------------------------------------------------- 1. prep
    frame = dataset.build_survival_frame(
        panel, static, left_truncation=not args.no_left_truncation
    )
    print(
        f"{frame.censoring['loans']:,} loans | "
        f"{frame.censoring['events_default']:,} default, "
        f"{frame.censoring['events_prepaid']:,} prepaid, "
        f"{frame.censoring['censored_total']:,} censored "
        f"({frame.censoring['censoring_rate']:.1%})"
    )

    train, test = dataset.vintage_split(frame, args.vintage_cutoff)
    print(f"Vintage split at {args.vintage_cutoff}: train={len(train):,} loans, test={len(test):,}")

    timeline = np.arange(1, int(frame.data[frame.duration_col].max()) + 1)

    # ------------------------------------------------------- 2. baseline model
    constant = baselines.ConstantHazardModel.fit(train)
    print(constant.summary().to_string(index=False))

    marginal = {c: baselines.aalen_johansen_cif(frame, c, timeline=timeline) for c in CAUSES}
    horizon_rows = []
    for cause, curve in marginal.items():
        for horizon in config.SURVIVAL_HORIZONS:
            window = curve[curve["month"] <= horizon]
            if window.empty:
                continue
            horizon_rows.append(
                {
                    "cause": config.EVENT_LABELS[cause],
                    "months_on_book": horizon,
                    "cif_aalen_johansen": float(window["cif"].iloc[-1]),
                    "naive_1_minus_km": float(window["naive_1_minus_km"].iloc[-1]),
                    "overstatement_pp": 100
                    * (float(window["naive_1_minus_km"].iloc[-1]) - float(window["cif"].iloc[-1])),
                }
            )
    horizon_table = pd.DataFrame(horizon_rows)

    # ------------------------------------------------------- 3. advanced model
    print("\nFitting cause-specific Cox models...")
    spec = models.DesignSpec.fit(train)
    cox = {
        cause: models.fit_cause_specific_cox(train, cause, spec, penalizer=args.penalizer)
        for cause in CAUSES
    }

    hazard_ratios = pd.concat([m.summary_frame() for m in cox.values()], ignore_index=True)
    ph_tests = pd.concat(
        [models.check_proportional_hazards(m, train) for m in cox.values()], ignore_index=True
    )

    # ---------------------------------------------------------- 4. evaluation
    print("Evaluating on the holdout vintages...")
    horizons = tuple(h for h in config.SURVIVAL_HORIZONS if h <= test.data[test.duration_col].max())
    horizons = horizons or (12,)

    metric_rows = []
    for cause in CAUSES:
        estimates = evaluation.build_survival_estimates(
            cox[cause], constant, train, test, cause, np.asarray(horizons, dtype=float)
        )
        brier = evaluation.brier_scores(estimates, train, test, cause, horizons)
        concordance = {
            "cox": evaluation.concordance(cox[cause], test),
            # No covariates -> every loan gets the same score -> no ranking.
            "constant_hazard": 0.5,
            "kaplan_meier": 0.5,
        }
        for _, row in brier.iterrows():
            record = row.to_dict()
            record["concordance"] = concordance.get(record["model"], float("nan"))
            metric_rows.append(record)

    results = evaluation.results_frame(metric_rows)
    print("\n" + evaluation.results_markdown(results))

    calibration_horizon = min(args.calibration_horizon, int(test.data[test.duration_col].max()))
    cif_grid = np.arange(1, calibration_horizon + 1)
    predicted_cif = models.cumulative_incidence(cox, test, cif_grid)
    calibration_tables = {
        cause: evaluation.calibration_by_risk_decile(
            predicted_cif[cause][:, -1], test, cause, calibration_horizon
        )
        for cause in CAUSES
    }
    calibration = pd.concat(calibration_tables.values(), ignore_index=True)

    # ------------------------------------------------------------- 5. figures
    figures: dict[str, Path] = {}
    if not args.no_figures:
        print("Rendering figures...")
        figures["Competing risks over loan age"] = curves.plot_competing_risk_overview(
            frame, outdir, timeline=timeline
        )
        for segment in config.SURVIVAL_SEGMENTS:
            if segment not in frame.data.columns:
                continue
            path = curves.plot_segmented_cif(
                frame, segment, outdir, timeline=timeline, order=SEGMENT_ORDERS.get(segment)
            )
            if path is not None:
                figures[f"Cumulative incidence by {segment.replace('_', ' ')}"] = path

        # Stop the holdout curves where the risk set thins out: past that point
        # a flat Aalen-Johansen tail says "no data left", not "no more risk".
        holdout_timeline = baselines.effective_timeline(test, min_at_risk=100)
        observed = {c: baselines.aalen_johansen_cif(test, c, timeline=holdout_timeline) for c in CAUSES}
        predicted = {
            "cox": models.mean_profile_cif(cox, test, holdout_timeline),
            "constant_hazard": {
                c: constant.cumulative_incidence(c, holdout_timeline) for c in CAUSES
            },
        }
        figures["Predicted vs observed on the holdout"] = curves.plot_model_comparison(
            observed, predicted, holdout_timeline, outdir
        )
        figures["Cause-specific hazard ratios"] = curves.plot_hazard_ratios(
            {c: m.summary_frame() for c, m in cox.items()}, outdir
        )
        figures["Calibration by risk decile"] = curves.plot_calibration(
            calibration_tables, outdir, calibration_horizon
        )

    # -------------------------------------------------------------- 6. output
    censoring = dataset.censoring_report(frame)
    outcomes = dataset.outcome_summary(frame)

    for name, table in {
        "censoring_summary": censoring,
        "outcomes": outcomes,
        "constant_hazard_baseline": constant.summary(),
        "cumulative_incidence_horizons": horizon_table,
        "model_comparison": results,
        "hazard_ratios": hazard_ratios,
        "proportional_hazards_tests": ph_tests,
        "calibration_deciles": calibration,
    }.items():
        table.to_csv(outdir / f"{name}.csv", index=False)

    for cause, curve in marginal.items():
        curve.to_csv(outdir / f"cif_{config.EVENT_LABELS[cause]}.csv", index=False)
    for segment in config.SURVIVAL_SEGMENTS:
        if segment not in frame.data.columns:
            continue
        for cause in CAUSES:
            seg = baselines.cif_by_segment(frame, segment, cause, timeline=timeline)
            if not seg.empty:
                seg.to_csv(outdir / f"cif_{config.EVENT_LABELS[cause]}_by_{segment}.csv", index=False)

    split_note = (
        f"Models are fitted on vintages originated up to **{pd.Timestamp(args.vintage_cutoff):%Y-%m}** "
        f"({len(train):,} loans) and evaluated on later vintages ({len(test):,} loans) that the "
        "model has never seen. Splitting on origination rather than reporting month keeps each "
        "loan's history intact -- a duration model cannot represent half a loan -- while still "
        "guaranteeing the holdout is strictly forward in time."
    )

    builder = report.build_report(
        censoring=censoring,
        outcomes=outcomes,
        baseline_summary=constant.summary(),
        results=results,
        horizon_table=horizon_table,
        hazard_ratios=hazard_ratios,
        ph_tests=ph_tests,
        calibration=calibration,
        figures=figures,
        split_note=split_note,
        reports_dir=reports_dir,
    )
    builder.save(reports_dir / "survival_report.md", reports_dir / "survival_report.html")

    (outdir / "censoring_summary.json").write_text(
        json.dumps(frame.censoring, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {reports_dir / 'survival_report.md'}")


if __name__ == "__main__":
    main()
