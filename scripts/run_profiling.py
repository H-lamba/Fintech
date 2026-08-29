"""
Phase 1 pipeline: runs every Task 1 requirement and writes the
Data Intelligence Report plus the record-level data-quality scores.

    python scripts/run_profiling.py

Outputs (all under reports/):
    data_intelligence_report.md / .html   <- graded deliverable
    dq_scores_train.csv                   <- record-level scores, feeds Task 4
    profiling/*.csv                       <- every intermediate table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_io  # noqa: E402
from src.profiling import (  # noqa: E402
    distributions,
    drift,
    missingness,
    outliers,
    quality_score,
    reconciliation,
    relationships,
    rules,
)
from src.profiling import figures  # noqa: E402
from src.profiling.report import ReportBuilder  # noqa: E402


def _save(df: pd.DataFrame, name: str, outdir: Path) -> None:
    if df is not None and not df.empty:
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 data profiling.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Profile only N rows (fast iteration on big files).")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip chart rendering (tables and scores only).")
    args = parser.parse_args()

    reports_dir = config.REPORTS_DIR
    tables_dir = reports_dir / "profiling"
    charts_dir = reports_dir / "profiling" / "charts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Path] = {}

    print("Loading data pack...")
    train = data_io.load_train()
    test = data_io.load_test()
    static = data_io.load_static()
    servicer = data_io.load_servicer_updates()
    json_rules = data_io.load_validation_rules()
    data_dict = data_io.load_data_dictionary()

    if train.empty:
        print(f"ERROR: no training data at {config.TRAIN_PATH}")
        print("Drop the organiser's data pack into data/, or generate a stand-in:")
        print("    python scripts/generate_synthetic_suite.py")
        sys.exit(1)

    if args.sample:
        train = train.head(args.sample)
        if not test.empty:
            test = test.head(args.sample)

    # merge static attributes in if they add columns
    if not static.empty and config.ID_COL in static.columns:
        new_cols = [c for c in static.columns if c not in train.columns or c == config.ID_COL]
        if len(new_cols) > 1:
            train = train.merge(static[new_cols], on=config.ID_COL, how="left")
            if not test.empty:
                test = test.merge(static[new_cols], on=config.ID_COL, how="left")

    print(f"  train: {len(train):,} rows x {train.shape[1]} cols")
    print(f"  test:  {len(test):,} rows x {test.shape[1] if not test.empty else 0} cols")

    rb = ReportBuilder("Data Intelligence Report - Loan Performance Intelligence Engine")

    # ---------------------------------------------------------------- overview
    schema_train = data_io.schema_report(train, "train")
    schema_test = data_io.schema_report(test, "test") if not test.empty else {}

    rb.add_text(
        "Scope",
        "Profiling of the loan-level monthly panel prior to any model training. "
        "Covers distributions, missingness, outliers and invalid date relationships, "
        "cross-column relationship breaks, field dependencies, train/test drift, "
        "source reconciliation, and record- and batch-level data-quality scoring.\n\n"
        "Every finding below is reproduced by `python scripts/run_profiling.py`.",
    )

    rb.add_kv(
        "Dataset overview",
        {
            "Train rows": f"{len(train):,}",
            "Train columns": train.shape[1],
            "Test rows": f"{len(test):,}" if not test.empty else "n/a",
            "Unique loans (train)": f"{train[config.ID_COL].nunique():,}"
            if config.ID_COL in train.columns else "n/a",
            "Reporting period (train)": f"{train[config.TIME_COL].min()} to {train[config.TIME_COL].max()}"
            if config.TIME_COL in train.columns else "n/a",
            "Reporting period (test)": f"{test[config.TIME_COL].min()} to {test[config.TIME_COL].max()}"
            if not test.empty and config.TIME_COL in test.columns else "n/a",
            "Data dictionary fields parsed": len(data_dict),
            "Organiser validation rules loaded": len(json_rules),
        },
    )

    if schema_train.get("expected_but_missing"):
        rb.add_text(
            "Schema differences vs. the published field list",
            "Expected but absent: `" + "`, `".join(schema_train["expected_but_missing"]) + "`\n\n"
            "Present but not in the published list: `"
            + "`, `".join(schema_train["present_but_unexpected"][:25]) + "`",
        )

    # ------------------------------------------------------- 1. distributions
    print("Profiling distributions...")
    col_profile = distributions.profile_columns(train)
    _save(col_profile, "column_profile_train", tables_dir)
    rb.add_table(
        "1. Column distributions",
        col_profile,
        "One row per column: dtype, cardinality, missingness, and the distribution "
        "summary appropriate to its type.",
        max_rows=60,
    )

    constant_cols = col_profile.loc[col_profile["n_unique"] <= 1, "column"].tolist()
    if constant_cols:
        rb.add_text(
            "Constant columns",
            "Zero variance, so they carry no predictive signal and should be dropped "
            "before training: `" + "`, `".join(constant_cols) + "`",
            level=3,
        )

    # ---------------------------------------------------------- 2. missingness
    print("Analysing missingness...")
    miss = missingness.missingness_summary(train)
    miss_pairs = missingness.missingness_co_occurrence(train)
    _save(miss, "missingness_train", tables_dir)
    _save(miss_pairs, "missingness_co_occurrence", tables_dir)

    rb.add_table(
        "2. Missing-value patterns",
        miss[miss["n_missing"] > 0],
        "Columns with at least one missing value, worst first.",
    )
    rb.add_table(
        "Missingness co-occurrence",
        miss_pairs.head(15),
        "Columns whose missing-indicators correlate: a high value means the fields go "
        "missing together, which points at one upstream process dropping a block of "
        "fields rather than independent gaps.",
        level=3,
    )

    if "current_status" in train.columns:
        structural = missingness.structured_missingness_flags(train, "current_status")
        _save(structural, "structured_missingness", tables_dir)
        rb.add_table(
            "Structured (by-design) missingness",
            structural,
            "Near-total missingness within a segment is almost always intentional "
            "(e.g. loss severity is only populated once a loan defaults). These are "
            "excluded from the defect narrative so they don't inflate the quality penalty.",
            level=3,
        )

    # --------------------------------------------- 3. outliers & date validity
    print("Detecting outliers and date violations...")
    iqr = outliers.iqr_outliers(train)
    robust = outliers.zscore_outliers(train)
    date_checks = outliers.date_relationship_checks(train)
    _save(iqr, "outliers_iqr", tables_dir)
    _save(robust, "outliers_robust_z", tables_dir)
    _save(date_checks, "date_relationship_checks", tables_dir)

    rb.add_table("3. Outliers (Tukey IQR fences)", iqr,
                 "Numeric columns ranked by the share of rows outside 1.5x IQR.")
    rb.add_table("Outliers (robust z-score)", robust,
                 "Median/MAD-based second opinion, less sensitive to the outliers it is hunting.",
                 level=3)
    rb.add_table("Invalid date relationships", date_checks,
                 "Loan-specific temporal consistency checks.", level=3)

    impossible_periods = outliers.impossible_reporting_periods(train)
    if not impossible_periods.empty:
        _save(impossible_periods, "impossible_reporting_periods", tables_dir)
        n_rows = int(impossible_periods["rows"].sum())
        rb.add_table(
            "Reporting months that predate the book",
            impossible_periods,
            f"**{n_rows:,} rows** are dated before the earliest origination in the entire "
            "portfolio, so they describe months in which no loan existed. These are "
            "corrupted timestamps, not early history. They are easy to miss because they "
            "sit at the edge of the calendar and look like a thin warm-up period -- and "
            "they are why the mean data-quality score is depressed for the earliest "
            "reporting periods rather than for any particular servicer or field.",
            level=3,
        )

    # -------------------------------------------- 4. cross-column rule breaks
    print("Evaluating validation rules...")
    rule_set = rules.build_rule_set(json_rules)
    rule_summary, rule_matrix = rules.evaluate_rules(train, rule_set)
    _save(rule_summary, "rule_violations", tables_dir)

    n_json = sum(1 for r in rule_set if r.name.startswith("json__"))
    rb.add_table(
        "4. Cross-column relationship breaks",
        rule_summary,
        f"{n_json} rule(s) loaded from the organiser's `validation_rules.json` plus "
        f"{len(rule_set) - n_json} additional domain rules written for this engine. "
        "A violation means two fields on the same record contradict each other.",
        max_rows=50,
    )

    # ------------------------------------------------------- 5. relationships
    print("Computing correlations and dependencies...")
    corr = relationships.numeric_correlations(train)
    top_pairs = relationships.top_correlated_pairs(corr)
    cat_assoc = relationships.categorical_associations(train)
    mixed = relationships.mixed_associations(train)
    redundant = relationships.redundancy_candidates(corr)

    _save(corr.reset_index(names="column") if not corr.empty else corr, "correlation_matrix", tables_dir)
    _save(top_pairs, "top_correlated_pairs", tables_dir)
    _save(cat_assoc, "categorical_associations", tables_dir)
    _save(mixed, "mixed_associations", tables_dir)

    rb.add_table("5. Strongly correlated numeric fields", top_pairs,
                 "Spearman rank correlation (monotone, robust to the skew loan balances always carry). "
                 "|rho| >= 0.7 only.")
    rb.add_table("Categorical dependencies", cat_assoc.head(20),
                 "Bias-corrected Cramer's V. Plain chi-square inflates on high-cardinality "
                 "fields like `state`, so the correction matters here.", level=3)
    rb.add_table("Numeric-vs-categorical dependencies", mixed.head(20),
                 "Correlation ratio (eta): the share of a numeric field's variance explained "
                 "by a categorical field.", level=3)

    if redundant:
        rb.add_text(
            "Redundancy candidates",
            "Near-duplicate columns (|rho| >= 0.95). Keeping both splits importance across "
            "twins in tree models, and a near-perfect correlation with a target is a "
            "leakage smell worth checking before training: `" + "`, `".join(redundant) + "`",
            level=3,
        )

    # -------------------------------------------------------------- 6. drift
    print("Measuring train/test drift...")
    if not test.empty:
        drift_tbl = drift.drift_report(train, test)
        _save(drift_tbl, "train_test_drift", tables_dir)
        major = drift_tbl[drift_tbl["drift_band"] == "MAJOR shift"] if not drift_tbl.empty else pd.DataFrame()
        rb.add_table(
            "6. Train vs. test drift",
            drift_tbl,
            "Population Stability Index per shared column, with a KS test (numeric) or "
            "category-share comparison (categorical) as a second opinion. "
            "Conventional bands: PSI < 0.10 stable, 0.10-0.25 moderate, > 0.25 major.",
            max_rows=50,
        )
        if not major.empty:
            rb.add_text(
                "Columns with major drift",
                "These shifted materially between train and test. Either the feature needs "
                "re-binning, or the model needs the scoring period represented in training - "
                "otherwise the time-aware validation will look optimistic relative to the "
                "actual scoring run: `" + "`, `".join(major["column"].tolist()) + "`",
                level=3,
            )

        if not args.no_figures:
            path = figures.plot_drift(drift_tbl, charts_dir)
            if path:
                rendered["Train vs. test drift"] = path
    else:
        rb.add_text("6. Train vs. test drift", "_No test file present; skipped._")

    if config.TIME_COL in train.columns:
        temporal = drift.temporal_drift(train, config.TIME_COL)
        _save(temporal, "temporal_drift", tables_dir)
        if not temporal.empty:
            worst = temporal.sort_values("psi_vs_first_period", ascending=False).head(20)
            rb.add_table(
                "Drift within the training window",
                worst,
                "Each reporting period measured against the first. This is the check that "
                "tells you whether a time-aware split will behave, and which vintages look "
                "unlike the scoring period.",
                level=3,
            )

    # ----------------------------------------------- 7. source reconciliation
    print("Reconciling against servicer updates...")
    conflict_series = None
    if not servicer.empty:
        recon_summary, conflicts = reconciliation.reconcile(train, servicer)
        conflict_series = reconciliation.conflict_flags(train, servicer)
        _save(recon_summary, "source_conflicts_summary", tables_dir)
        _save(conflicts.head(500), "source_conflict_examples", tables_dir)
        rb.add_table(
            "7. Source conflicts vs. servicer feed",
            recon_summary,
            "Fields where the monthly panel and the servicer feed disagree on the same "
            "(loan, month). Numeric fields use a 1% relative tolerance so float noise "
            "isn't reported as a conflict.",
        )
        rb.add_table("Example conflicting records", conflicts.head(15), level=3)
    else:
        rb.add_text("7. Source conflicts vs. servicer feed", "_No servicer file present; skipped._")

    dupes = reconciliation.duplicate_records(train)
    if not dupes.empty:
        _save(dupes.head(200), "duplicate_records", tables_dir)
        rb.add_text(
            "Duplicate panel rows",
            f"**{len(dupes):,} rows** duplicate an existing (loan_id, reporting_month) pair. "
            "A loan should appear once per month; anything else breaks the panel structure "
            "and will quietly corrupt a time-aware split if not de-duplicated first.",
            level=3,
        )

    if not args.no_figures:
        print("Rendering figures...")
        for caption, path in (
            ("Where the data is missing", figures.plot_missingness(train, miss, charts_dir)),
            ("Numeric distributions", figures.plot_distributions(train, charts_dir)),
            ("Validation rule violations", figures.plot_rule_violations(rule_summary, charts_dir)),
        ):
            if path:
                rendered[caption] = path

    # ------------------------------------------------- 8. data-quality scores
    print("Scoring data quality...")
    date_matrix = outliers.date_violation_matrix(train)
    outlier_matrix = outliers.outlier_flag_matrix(train)
    stale = outliers.staleness_flags(train)

    conditional_cols = missingness.detect_conditional_columns(train)
    record_scores = quality_score.compute_record_scores(
        train,
        rule_matrix=rule_matrix,
        date_matrix=date_matrix,
        outlier_matrix=outlier_matrix,
        stale_flags=stale,
        conflict_flags=conflict_series,
        exclude_missing_cols=conditional_cols,
    )
    record_scores.to_csv(reports_dir / "dq_scores_train.csv", index=False)

    if not args.no_figures:
        path = figures.plot_quality_scores(record_scores, charts_dir)
        if path:
            rendered["Data quality"] = path

    batch = quality_score.batch_summary(record_scores)
    rb.add_kv(
        "8. Batch-level data-quality score",
        {
            "Batch DQ score (0-100, mean)": batch.get("batch_dq_score"),
            "Median record score": batch.get("median_dq_score"),
            "5th percentile record score": batch.get("p05_dq_score"),
            "Records with zero defects": f"{batch.get('pct_clean')}%",
            "Records in 'critical' band": f"{batch.get('pct_critical')}%",
            "Records 'poor' or worse": f"{batch.get('pct_poor_or_worse')}%",
            "Records with a rule violation": f"{batch.get('n_with_rule_violation'):,}",
            "Records with a date violation": f"{batch.get('n_with_date_violation'):,}",
            "Stale records": f"{batch.get('n_stale'):,}",
            "Records with a source conflict": f"{batch.get('n_source_conflict'):,}",
        },
    )

    rb.add_text(
        "Scoring method",
        "Each defect signal contributes a weighted penalty (rule violation 3.0, source "
        "conflict 2.5, missing critical field 2.0, stale record 1.5, outlier 1.0, missing "
        "non-critical field 0.5). The total penalty maps to a 0-100 score via "
        "`100 * exp(-penalty / 10)`, so a clean record scores exactly 100 and the scale "
        "degrades smoothly without a hand-tuned maximum. Weights live in `src/config.py` "
        "and should be re-tuned once the real defect mix is known.\n\n"
        "The per-record reason string is kept alongside the score, so a reviewer sees "
        "*which* defects drove it - that is what makes this score usable as an anomaly "
        "feature in Task 4 and as grounding for the copilot in Task 7, rather than an "
        "opaque number.\n\n"
        + (
            "Conditionally-populated columns excluded from the missingness penalty "
            "(blank by design, not a defect): `" + "`, `".join(conditional_cols) + "`."
            if conditional_cols
            else "No conditionally-populated columns were detected."
        ),
        level=3,
    )

    for seg in ("servicer_name", "source_system", "state", "credit_score_band"):
        if seg in train.columns:
            seg_tbl = quality_score.batch_summary_by_segment(train, record_scores, seg)
            _save(seg_tbl, f"dq_by_{seg}", tables_dir)
            rb.add_table(
                f"Data quality by {seg}",
                seg_tbl,
                "Worst-scoring segment first. This is the view that turns a portfolio-level "
                "number into an action.",
                level=3,
            )
            break

    worst = quality_score.worst_records(train, record_scores, n=25)
    _save(worst, "worst_records", tables_dir)
    rb.add_table(
        "Worst 25 records",
        worst,
        "Lowest data-quality scores with their specific defects. These seed the "
        "reviewer-ready anomaly examples required in Task 4.",
        level=3,
    )

    # -------------------------------------------------------------- figures
    if rendered:
        order = [
            "Where the data is missing",
            "Numeric distributions",
            "Validation rule violations",
            "Train vs. test drift",
            "Data quality",
        ]
        lines = []
        for caption in [c for c in order if c in rendered] + [
            c for c in rendered if c not in order
        ]:
            relative = rendered[caption].relative_to(reports_dir)
            lines.append(f"**{caption}**\n\n![{caption}]({relative.as_posix()})\n")
        rb.add_text("9. Figures", "\n".join(lines))

    # --------------------------------------------------- 10. feature dictionary
    # Written by scripts/run_prediction.py, which is where the feature matrix is
    # actually built. Included here when it exists so the Data Intelligence
    # Report stays the single document describing the data and what is derived
    # from it -- and stays runnable on its own when it does not.
    feature_dict_path = reports_dir / "feature_dictionary.md"
    if feature_dict_path.exists():
        body = feature_dict_path.read_text()
        body = body.split("\n", 1)[1].strip() if body.startswith("#") else body.strip()
        rb.add_text("10. Feature dictionary", body)
    else:
        rb.add_text(
            "10. Feature dictionary",
            "_Not yet generated. Run `python scripts/run_prediction.py` to build the "
            "feature matrix and emit `reports/feature_dictionary.md`, then re-run this "
            "script to fold it in._",
        )

    # ---------------------------------------------------------- top findings
    findings = []
    applicable = rule_summary[rule_summary["applicable"] == True]  # noqa: E712
    top_rules = applicable.nlargest(5, "n_violations") if not applicable.empty else pd.DataFrame()
    for _, r in top_rules.iterrows():
        if r["n_violations"] and r["n_violations"] > 0:
            findings.append(
                f"**{r['rule']}** ({r['severity']}): {int(r['n_violations']):,} records "
                f"({r['pct_violations']}%) - {r['description']}"
            )
    if not miss.empty:
        worst_miss = miss.nlargest(3, "pct_missing")
        for _, r in worst_miss.iterrows():
            if r["pct_missing"] > 0:
                findings.append(f"**{r['column']}** is {r['pct_missing']}% missing.")
    if not impossible_periods.empty:
        findings.append(
            f"**{int(impossible_periods['rows'].sum()):,} rows are dated before the earliest "
            f"origination in the book** ({', '.join(impossible_periods['reporting_month'])}) - "
            "corrupted timestamps describing months in which no loan existed."
        )
    if batch.get("n_stale"):
        findings.append(f"**{batch['n_stale']:,} stale records** (servicer update older than "
                        f"{config.STALE_DAYS} days relative to the reporting month).")
    if batch.get("n_source_conflict"):
        findings.append(f"**{batch['n_source_conflict']:,} records conflict with the servicer feed.**")

    rb.sections.insert(
        2,
        (2, "Top data-quality issues",
         "\n".join(f"{i}. {f}" for i, f in enumerate(findings[:10], 1)) or "_None detected._"),
    )

    # ------------------------------------------------------------------ save
    md_path = reports_dir / "data_intelligence_report.md"
    html_path = reports_dir / "data_intelligence_report.html"
    rb.save(md_path, html_path)

    with open(reports_dir / "batch_quality_summary.json", "w") as fh:
        json.dump(batch, fh, indent=2, default=str)

    print("\nDone.")
    print(f"  {md_path}")
    print(f"  {html_path}")
    print(f"  {reports_dir / 'dq_scores_train.csv'}")
    print(f"  {tables_dir}/ ({len(list(tables_dir.glob('*.csv')))} tables)")
    print(f"\nBatch data-quality score: {batch.get('batch_dq_score')}/100 "
          f"({batch.get('pct_clean')}% of records defect-free)")


if __name__ == "__main__":
    main()
