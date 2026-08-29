"""
Single entrypoint for the synthetic loan-performance benchmark suite.

    python scripts/generate_synthetic_suite.py
    python scripts/generate_synthetic_suite.py --loans 10000 --out data
    python scripts/generate_synthetic_suite.py --parquet --seed 7

Writes to ./data/ by default:
    loan_static_attributes.csv
    loan_monthly_performance_train.csv     reporting_month <  2024-01-01, labelled
    loan_monthly_performance_test.csv      reporting_month >= 2024-01-01, unlabelled
    servicer_updates.csv
    macro_scenarios.csv
    validation_rules.json
    data_dictionary.md
    submission_template.csv
    _injected_anomalies.csv                ground truth for anomaly detection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datagen import GenerationConfig, SyntheticDataPipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic loan-performance benchmark suite."
    )
    parser.add_argument("--loans", type=int, default=50_000, help="Cohort size (default 50000).")
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed.")
    parser.add_argument("--out", type=str, default="data", help="Output directory.")
    parser.add_argument("--max-months", type=int, default=60, help="Cap on months on book.")
    parser.add_argument("--split-date", type=str, default="2024-01-01",
                        help="Out-of-time train/test cutoff.")
    parser.add_argument("--anomaly-rate", type=float, default=0.025,
                        help="Share of rows carrying an injected defect.")
    parser.add_argument("--parquet", action="store_true",
                        help="Also write parquet copies of the panel files.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the diagnostic report.")
    return parser.parse_args()


def _report(result, out_dir: Path) -> None:
    stats, timings = result.stats, result.timings

    print("\n" + "=" * 72)
    print("SYNTHETIC LOAN PERFORMANCE SUITE")
    print("=" * 72)

    print(f"\nCohort           {stats['n_loans']:>12,} loans")
    print(f"Panel rows       {stats['n_panel_rows']:>12,}")
    print(f"  train          {stats['n_train_rows']:>12,}")
    print(f"  test           {stats['n_test_rows']:>12,}")
    print(f"Mean months/loan {stats['mean_months_on_book']:>12}")

    print("\n--- Terminal outcomes ---")
    total = max(stats["n_loans"], 1)
    print(f"  Defaulted      {stats['terminal_default_loans']:>8,}  "
          f"({100.0 * stats['terminal_default_loans'] / total:5.2f}%)")
    print(f"  Prepaid        {stats['terminal_prepaid_loans']:>8,}  "
          f"({100.0 * stats['terminal_prepaid_loans'] / total:5.2f}%)")

    print("\n--- Copula: achieved correlations (target: -0.45 credit/LTV, -0.45 credit/DTI) ---")
    print(result.correlations.to_string())

    print("\n--- Targets ---")
    print(result.target_summary.to_string(index=False))

    print("\n--- Injected anomalies (ground truth) ---")
    print(result.anomaly_ledger.to_string(index=False))
    print(f"  total {stats['anomaly_rows']:,} rows ({stats['anomaly_rate_pct']}% of panel)")

    print("\n--- Servicer feed ---")
    print(f"  rows              {stats['servicer_rows']:>8,}")
    print(f"  balance conflicts {stats['balance_conflicts']:>8,}")
    print(f"  status conflicts  {stats['status_conflicts']:>8,}")
    print(f"  stale rows        {stats['stale_rows']:>8,}")

    print("\n--- Timings (seconds) ---")
    for phase, seconds in timings.items():
        print(f"  {phase:<32} {seconds:>7.2f}")

    print(f"\nWritten to {out_dir.resolve()}")
    for name in stats.get("files_written", []):
        size_mb = (out_dir / name).stat().st_size / 1e6
        print(f"  {name:<44} {size_mb:>8.1f} MB")
    print()


def main() -> int:
    args = _parse_args()
    pd.set_option("display.width", 120)

    config = GenerationConfig(
        n_loans=args.loans,
        seed=args.seed,
        output_dir=Path(args.out),
        max_months_on_book=args.max_months,
        split_date=args.split_date,
        anomaly_rate=args.anomaly_rate,
        write_parquet=args.parquet,
    )

    result = SyntheticDataPipeline(config).run(write=True)

    if not args.quiet:
        _report(result, Path(args.out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
