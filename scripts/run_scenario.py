"""
Phase 6 pipeline: Task 5 -- scenario and stress simulation.

    python scripts/run_scenario.py                 # all scenarios, ~40s
    python scripts/run_scenario.py --sample 2000   # 2,000 loans, fast pass
    python scripts/run_scenario.py --horizons 12 24

Requires the Phase 3 models: run `make predict` first. This pipeline projects
the trained models under stress; it never refits them.

Outputs (all under reports/):
    scenario_report.md / .html    <- graded deliverable
    scenario_report.csv           <- portfolio projection, one row per scenario-horizon
    scenario/*.csv                <- segment tables, drivers, calibration, checks
    scenario/*.png                <- projection paths, segment impact
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_io, features as feature_module  # noqa: E402
from src.scenario import drivers as driver_module  # noqa: E402
from src.scenario import figures, project, report  # noqa: E402
from src.scenario.macro import MacroScenarios  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 6 (Task 5) scenario and stress simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Use N randomly chosen LOANS (whole histories kept) for a fast pass.")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(config.SCENARIO_HORIZONS),
                        help="Projection months to report.")
    parser.add_argument("--variant", default="improved", choices=["improved", "baseline"],
                        help="Which Phase 3 model variant to project.")
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
    outdir = reports_dir / "scenario"
    outdir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(args.horizons)

    # ------------------------------------------------------- 1. assumptions
    print("Loading macro scenarios...")
    scenarios = MacroScenarios.load()
    print(f"  {len(scenarios.scenarios)} scenarios, baseline = {scenarios.baseline_name!r}, "
          f"horizons 1..{max(scenarios.horizons())}")

    # ------------------------------------------------------- 2. the portfolio
    print("Building the portfolio's current position...")
    train = data_io.load_train()
    test = data_io.load_test()
    static = data_io.load_static()
    if train.empty:
        raise FileNotFoundError(f"No panel found at {config.TRAIN_PATH}")

    panel = pd.concat([train, test], ignore_index=True, sort=False) if not test.empty else train
    if args.sample:
        panel = _sample_loans(panel, args.sample)

    dq_scores = pd.read_csv(config.DQ_SCORES_PATH, low_memory=False) if config.DQ_SCORES_PATH.exists() else None
    engineered, _ = feature_module.build_feature_matrix(panel, static=static, dq_scores=dq_scores)
    portfolio = project.latest_position(engineered)

    if "vintage_year" not in portfolio.columns and "origination_month" in portfolio.columns:
        portfolio["vintage_year"] = pd.to_datetime(portfolio["origination_month"]).dt.year
    print(f"  {len(portfolio):,} loans at their latest observed position "
          f"({portfolio[config.TIME_COL].min():%Y-%m} .. {portfolio[config.TIME_COL].max():%Y-%m})")

    # ---------------------------------------------------------- 3. the models
    print(f"Loading Phase 3 models ({args.variant})...")
    models = project.load_models(variant=args.variant)

    inputs = project.ProjectionInputs(portfolio=portfolio, models=models, scenarios=scenarios)

    # --------------------------------------------------- 4. calibrate credit
    print("Calibrating the credit channel against the stated default multipliers...")
    inputs.calibration = project.calibrate_scenarios(inputs, horizons=horizons)
    calibration_table = pd.DataFrame(
        [{"scenario": s, "horizon_month": h, **d} for (s, h), d in inputs.calibration.items()]
    ).sort_values(["scenario", "horizon_month"])

    # ------------------------------------------------------------ 5. project
    print("Projecting the portfolio...")
    record_level, portfolio_level = project.project(inputs, horizons=horizons)
    print()
    print(report.projection_markdown(portfolio_level))

    checks = project.multiplier_check(portfolio_level, scenarios)
    saturation = project.saturation_summary(inputs.calibration, scenarios)
    unreached = int((~saturation["reached"]).sum()) if not saturation.empty else 0
    if unreached:
        print(f"  [note] {unreached} scenario-month(s) state a default multiplier the "
              "credit channel cannot reach; see the saturation table")

    # ------------------------------------------------------- 6. segmentation
    print("\nAggregating by segment...")
    segment_tables = {}
    for segment in config.SCENARIO_SEGMENTS:
        table = project.segment_projection(record_level, segment)
        if not table.empty:
            segment_tables[segment] = table
            print(f"  {segment}: {table[segment].nunique()} groups")

    # ------------------------------------------------------------ 7. drivers
    print("Attributing the movement to features...")
    drivers, movements = driver_module.collect(
        models, portfolio, scenarios, inputs.calibration, horizons=horizons
    )

    narratives = report.build_narratives(
        drivers, movements, portfolio_level, scenarios, horizons=horizons
    )

    # ------------------------------------------------------------ 8. figures
    rendered: dict[str, Path] = {}
    if not args.no_figures:
        print("Rendering figures...")
        for caption, path in (
            ("Projected rates by scenario", figures.plot_projection_paths(portfolio_level, outdir)),
            ("Segment impact under stress", figures.plot_segment_impact(segment_tables, outdir)),
            ("What moves each scenario", figures.plot_drivers(drivers, outdir)),
        ):
            if path:
                rendered[caption] = path

    # ------------------------------------------------------------ 9. outputs
    portfolio_level.to_csv(reports_dir / "scenario_report.csv", index=False)

    tables = {
        "portfolio_projection": portfolio_level,
        "scenario_assumptions": scenarios.summary(),
        "credit_calibration": calibration_table,
        "multiplier_check": checks,
        "credit_saturation": saturation,
        "scenario_drivers": drivers,
        "feature_movement": movements,
        "record_level_projection": record_level,
    }
    for segment, table in segment_tables.items():
        tables[f"segment_{segment}"] = table
    for name, table in tables.items():
        table.to_csv(outdir / f"{name}.csv", index=False)

    builder = report.build_report(
        assumptions=scenarios.summary(),
        portfolio_level=portfolio_level,
        calibration=calibration_table,
        checks=checks,
        saturation=saturation,
        segment_tables=segment_tables,
        drivers=drivers,
        movements=movements,
        narratives=narratives,
        figures=rendered,
        reports_dir=reports_dir,
        n_loans=len(portfolio),
        variant=args.variant,
    )
    builder.save(reports_dir / "scenario_report.md", reports_dir / "scenario_report.html")
    print(f"\nWrote {reports_dir / 'scenario_report.csv'} and {reports_dir / 'scenario_report.md'}")


if __name__ == "__main__":
    main()
