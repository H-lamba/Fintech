"""
Phase 3 pipeline: Task 2 -- loan performance prediction.

    python scripts/run_prediction.py                      # full run, all targets
    python scripts/run_prediction.py --sample 2000        # 2,000 loans, fast pass
    python scripts/run_prediction.py --targets next_12m_default_flag
    python scripts/run_prediction.py --score-test         # also score the unlabelled panel

Outputs (all under reports/):
    task2_model_results.csv / .md   <- baseline vs improved comparison table
    task2_split_audit.csv           <- evidence the split is time-aware and purged
    task2/*.csv                     <- reliability, lift, per-class, importances
    task2_test_predictions.csv      <- only with --score-test
    models/*.joblib                 <- fitted models + manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.models import predict, splitting  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 (Task 2) loan performance prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Train on N randomly chosen LOANS (whole histories kept) for a fast pass.",
    )
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "lightgbm", "xgboost", "hist"],
        help="Boosted-tree backend for the improved model.",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        help="Subset of targets to run. Default: all five.",
    )
    parser.add_argument("--train-end", default=config.SPLIT_TRAIN_END,
                        help="Last reporting month in the training window (inclusive).")
    parser.add_argument("--valid-end", default=config.SPLIT_VALID_END,
                        help="Last reporting month in the validation window (inclusive).")
    parser.add_argument("--test-end", default=config.SPLIT_TEST_END,
                        help="Last reporting month in the test window (inclusive).")
    parser.add_argument("--precision-floor", type=float, default=config.PRECISION_FLOOR,
                        help="Precision floor for the recall-at-fixed-precision metric.")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Skip probability calibration (for an ablation).")
    parser.add_argument("--no-save-models", action="store_true",
                        help="Do not persist fitted models to models/.")
    parser.add_argument("--score-test", action="store_true",
                        help="Also score the unlabelled test panel and write predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    boundaries = splitting.SplitBoundaries(
        train_end=pd.Timestamp(args.train_end),
        valid_end=pd.Timestamp(args.valid_end),
        test_end=pd.Timestamp(args.test_end),
    )

    predict.run_task2(
        sample_loans=args.sample,
        backend=args.backend,
        targets=args.targets,
        boundaries=boundaries,
        calibrate_models=not args.no_calibration,
        precision_floor=args.precision_floor,
        save_models=not args.no_save_models,
        score_test=args.score_test,
    )


if __name__ == "__main__":
    main()
