"""
Phase 5 pipeline: Task 4 -- anomaly and exception detection.

    python scripts/run_anomaly.py                  # full run, ~60s
    python scripts/run_anomaly.py --sample 2000    # 2,000 loans, fast pass
    python scripts/run_anomaly.py --examples 40    # a longer reviewer queue

Outputs (all under reports/):
    anomaly_report.md / .html   <- graded deliverable
    anomaly_examples.csv        <- the curated reviewer queue (>= 20 rows)
    anomaly/*.csv               <- ablation, signal coverage, per-class, importances
    anomaly/*.png               <- detector comparison, driver layers
    anomaly_scores.csv          <- record-level scores for the scored window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_io  # noqa: E402
from src.anomaly import (  # noqa: E402
    curation,
    evaluation,
    features as feature_module,
    figures,
    models,
    report,
    signals,
)
from src.models import splitting  # noqa: E402

EXCEPTION_FLAG = feature_module.EXCEPTION_FLAG
EXCEPTION_TYPE = feature_module.EXCEPTION_TYPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 (Task 4) anomaly and exception detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Use N randomly chosen LOANS (whole histories kept) for a fast pass.")
    parser.add_argument("--examples", type=int, default=25,
                        help="Rows in the curated reviewer queue (Task 4 requires at least 20).")
    parser.add_argument("--unsupported", type=int, default=5,
                        help="Queue slots reserved for high-scoring records with no rule violation.")
    parser.add_argument("--queue-size", type=int, default=500,
                        help="Queue size used for the precision@k columns.")
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
    outdir = reports_dir / "anomaly"
    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- 1. ingest
    print("Loading panel...")
    panel = data_io.load_train()
    if panel.empty:
        raise FileNotFoundError(f"No panel found at {config.TRAIN_PATH}")
    if args.sample:
        panel = _sample_loans(panel, args.sample)
    panel = panel.reset_index(drop=True)

    dq_scores = pd.read_csv(config.DQ_SCORES_PATH, low_memory=False) if config.DQ_SCORES_PATH.exists() else None

    # ------------------------------------------- 2. deterministic signals
    print("Evaluating deterministic signals (rules, dates, sequence detectors)...")
    signal_matrix, severities, rule_summary = signals.build_signal_matrix(
        panel, data_io.load_validation_rules()
    )
    rule_scores = signals.rule_score(signal_matrix, severities)
    triggered = signals.triggered_rules(signal_matrix, severities)

    targets = feature_module.prepare_targets(panel)
    labels = targets[EXCEPTION_FLAG].to_numpy()
    coverage = signals.signal_coverage(signal_matrix, severities, targets[EXCEPTION_FLAG].astype(bool))
    print(f"  {signal_matrix.shape[1]} signals; {int(signal_matrix.any(axis=1).sum()):,} records fire at least one")

    # ------------------------------------------------------- 3. features
    print("Building the anomaly feature matrix...")
    anomaly_features = feature_module.build_features(panel, signal_matrix, rule_scores, dq_scores)
    frame = anomaly_features.data
    for column in targets.columns:
        frame[column] = targets[column].to_numpy()
    classes = feature_module.exception_classes(targets)

    # ------------------------------------------------- 4. time-aware split
    # The exception label is contemporaneous, so the purge horizon is zero --
    # but the split is still by reporting month, and absorbing rows are kept,
    # because a terminal row can itself be the defect.
    boundaries = splitting.SplitBoundaries.from_config()
    split = splitting.make_time_split(
        frame, EXCEPTION_FLAG, boundaries=boundaries, horizon_months=0, drop_absorbing=False
    )
    train, valid, test = split.train, split.valid, split.test
    print(
        f"Split by reporting_month: train={len(train):,} valid={len(valid):,} test={len(test):,} "
        f"({split.audit['train_months']} | {split.audit['valid_months']} | {split.audit['test_months']})"
    )

    y_train = train[EXCEPTION_FLAG].to_numpy()
    y_valid = valid[EXCEPTION_FLAG].to_numpy()
    y_test = test[EXCEPTION_FLAG].to_numpy()

    # ----------------------------------------------- 5. unsupervised layer
    print("Fitting the Isolation Forest (no labels, no rule indicators)...")
    forest = models.IsolationForestDetector.fit(train, anomaly_features.unsupervised_columns)
    train_raw = forest.raw_score(train)
    ml_test = forest.score(test, reference=train_raw)

    rule_test = train.iloc[0:0]  # placeholder to keep the name obvious below
    rule_test = test["rule_score"].to_numpy()
    hybrid_test = models.hybrid_score(rule_test, ml_test)

    # ------------------------------------------------- 6. supervised heads
    print("Fitting the supervised exception heads...")
    numeric = [*anomaly_features.signal_columns, *anomaly_features.numeric_columns]
    categorical = anomaly_features.categorical_columns
    dtypes = models.category_dtypes(train, categorical)

    binary_head = models.fit_supervised_head(
        train, y_train, valid, y_valid, numeric, categorical, dtypes
    )

    # Two ablation heads. Without them the headline supervised number is
    # unreadable: the sequence detectors are near-perfect indicators of two
    # defect classes by construction, so a model handed them scores well
    # without having learned anything. These separate "the model learned
    # something" from "the model learned to trust a detector".
    #
    #   no sequence flags  -- detector indicators removed, continuous
    #                         month-on-month context retained
    #   record state only  -- no sequence information of any kind
    no_flag_head = models.fit_supervised_head(
        train, y_train, valid, y_valid,
        anomaly_features.no_sequence_flag_columns, categorical, dtypes,
    )
    no_flag_probability = no_flag_head.predict_proba(test, dtypes)[:, 1]

    record_only_head = models.fit_supervised_head(
        train, y_train, valid, y_valid,
        anomaly_features.record_state_columns, categorical, dtypes,
    )
    record_only_probability = record_only_head.predict_proba(test, dtypes)[:, 1]
    probability_valid = binary_head.predict_proba(valid, dtypes)[:, 1]
    probability_test = binary_head.predict_proba(test, dtypes)[:, 1]
    threshold = evaluation.tune_threshold_for_f1(y_valid, probability_valid)

    type_head = models.fit_supervised_head(
        train, train[EXCEPTION_TYPE].to_numpy(), valid, valid[EXCEPTION_TYPE].to_numpy(),
        numeric, categorical, dtypes, multiclass=True, class_order=classes,
    )
    type_proba = type_head.predict_proba(test, dtypes)

    # ----------------------------------------------------- 7. ablation
    print("Scoring the detector ablation...")
    rule_only = signal_matrix.loc[test.index, [c for c in signal_matrix.columns if not c.startswith("seq__")]].any(axis=1)
    all_deterministic = signal_matrix.loc[test.index].any(axis=1)

    # The continuous detectors are cut at the queue size the full deterministic
    # layer produces, so every row of the table costs the reviewer the same.
    queue_size = int(all_deterministic.sum())

    ablation_rows = [
        {"detector": "row-level rules", "labels_used": "no",
         **evaluation.binary_at_threshold(y_test, rule_only.to_numpy())},
        {"detector": "+ sequence detectors", "labels_used": "no",
         **evaluation.binary_at_threshold(y_test, all_deterministic.to_numpy())},
        {"detector": "isolation forest", "labels_used": "no",
         **evaluation.binary_at_threshold(y_test, evaluation.flag_top_k(ml_test, queue_size)),
         **evaluation.score_metrics(y_test, ml_test, (args.queue_size,))},
        {"detector": "hybrid (rules + forest)", "labels_used": "no",
         **evaluation.binary_at_threshold(y_test, evaluation.flag_top_k(hybrid_test, queue_size)),
         **evaluation.score_metrics(y_test, hybrid_test, (args.queue_size,))},
        {"detector": "supervised (record state only)", "labels_used": "yes",
         **evaluation.binary_at_threshold(y_test, evaluation.flag_top_k(record_only_probability, queue_size)),
         **evaluation.score_metrics(y_test, record_only_probability, (args.queue_size,))},
        {"detector": "supervised (no sequence flags)", "labels_used": "yes",
         **evaluation.binary_at_threshold(y_test, evaluation.flag_top_k(no_flag_probability, queue_size)),
         **evaluation.score_metrics(y_test, no_flag_probability, (args.queue_size,))},
        {"detector": "supervised (all signals)", "labels_used": "yes",
         **evaluation.binary_at_threshold(y_test, probability_test >= threshold),
         **evaluation.score_metrics(y_test, probability_test, (args.queue_size,)),
         "brier": evaluation.brier(y_test, probability_test)},
    ]
    ablation = evaluation.ablation_frame(ablation_rows)
    print("\n" + evaluation.results_markdown(ablation) + "\n")

    type_metrics = pd.DataFrame([evaluation.multiclass_metrics(test[EXCEPTION_TYPE].to_numpy(), type_proba, classes)])
    per_class = evaluation.per_class_report(test[EXCEPTION_TYPE].to_numpy(), type_proba, classes)
    confusion = evaluation.confusion_frame(test[EXCEPTION_TYPE].to_numpy(), type_proba, classes)
    print(evaluation.results_markdown(per_class))

    # ------------------------------------------------------ 8. explanations
    print("\nExtracting drivers...")
    from src.anomaly import explain

    contributions = explain.supervised_contributions(binary_head, test, dtypes)
    if contributions is not None:
        drivers = explain.top_drivers_from_contributions(contributions, test)
    else:
        deviation = explain.RobustDeviation(train, anomaly_features.unsupervised_columns)
        drivers = deviation.top_drivers(test)
    importance = explain.global_importance(
        binary_head, contributions, set(anomaly_features.sequence_context_columns)
    )

    # Unsupervised drivers are always computed: they are what a reviewer reads
    # for the rule-clean records, where the supervised model has no rule to
    # point at and "the pattern was unusual" needs a unit.
    deviation = explain.RobustDeviation(train, anomaly_features.unsupervised_columns)
    unsupervised_drivers = deviation.top_drivers(test)

    # ---------------------------------------------------------- 9. curation
    print("Curating the reviewer queue...")
    scores = pd.DataFrame(
        {
            "rule_score": rule_test,
            "ml_score": ml_test,
            "hybrid_score": hybrid_test,
            "exception_probability": probability_test,
        },
        index=test.index,
    )
    predicted_type = pd.Series(np.asarray(classes)[np.argmax(type_proba, axis=1)], index=test.index)

    combined_drivers = drivers.copy()
    rule_clean = triggered.reindex(test.index).fillna("").str.len() == 0
    combined_drivers[rule_clean] = unsupervised_drivers[rule_clean]

    queue = curation.build_review_queue(
        test, scores, triggered.reindex(test.index), combined_drivers,
        predicted_type=predicted_type, n_examples=args.examples, n_unsupported=args.unsupported,
    )
    composition = curation.queue_composition(queue)

    if len(queue) < 20:
        raise ValueError(f"Reviewer queue has {len(queue)} rows; Task 4 requires at least 20.")
    print(f"  {len(queue)} reviewer-ready examples "
          f"({int((queue['triggered_rules'].str.len() > 0).sum())} rule-supported)")

    # ---------------------------------------------------------- 10. outputs
    queue.to_csv(reports_dir / "anomaly_examples.csv", index=False)

    record_scores = scores.copy()
    record_scores.insert(0, config.ID_COL, test[config.ID_COL].to_numpy())
    record_scores.insert(1, config.TIME_COL, pd.to_datetime(test[config.TIME_COL]).dt.strftime("%Y-%m-%d"))
    record_scores["predicted_exception_type"] = predicted_type.to_numpy()
    record_scores["triggered_rules"] = triggered.reindex(test.index).to_numpy()
    record_scores.to_csv(reports_dir / "anomaly_scores.csv", index=False)

    rendered: dict[str, Path] = {}
    if not args.no_figures:
        print("Rendering figures...")
        for caption, path in (
            ("What each detector layer buys", figures.plot_detector_comparison(
                y_test,
                {
                    "isolation forest": ml_test,
                    "hybrid (rules + forest)": hybrid_test,
                    "supervised (record state only)": record_only_probability,
                    "supervised (all signals)": probability_test,
                },
                {
                    "row-level rules": (int(rule_only.sum()), float(y_test[rule_only.to_numpy()].mean())),
                    "+ sequence detectors": (queue_size, float(y_test[all_deterministic.to_numpy()].mean())),
                },
                outdir,
            )),
            ("What the exception model leans on", figures.plot_driver_layers(importance, outdir)),
        ):
            if path:
                rendered[caption] = path

    for name, table in {
        "detector_ablation": ablation,
        "signal_coverage": coverage,
        "rule_summary": rule_summary,
        "exception_type_metrics": type_metrics,
        "exception_type_per_class": per_class,
        "exception_type_confusion": confusion,
        "driver_importance": importance,
        "queue_composition": composition,
    }.items():
        table.to_csv(outdir / f"{name}.csv", index=False)

    split_note = (
        f"Fitted on `{split.audit['train_months']}` ({len(train):,} records), tuned on "
        f"`{split.audit['valid_months']}` ({len(valid):,}) and reported on "
        f"`{split.audit['test_months']}` ({len(test):,}), which no detector was fitted or "
        "thresholded on."
    )
    builder = report.build_report(
        ablation=ablation,
        signal_coverage=coverage,
        type_metrics=type_metrics,
        per_class=per_class,
        confusion=confusion,
        importance=importance,
        queue=queue,
        composition=composition,
        split_note=split_note,
        reports_dir=reports_dir,
        figures=rendered,
    )
    builder.save(reports_dir / "anomaly_report.md", reports_dir / "anomaly_report.html")
    print(f"\nWrote {reports_dir / 'anomaly_examples.csv'} and {reports_dir / 'anomaly_report.md'}")


if __name__ == "__main__":
    main()
