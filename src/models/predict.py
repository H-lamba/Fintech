"""
Phase 3 orchestration -- Task 2, loan performance prediction.

Pipeline, end to end::

    ingest -> engineer features -> time-aware purged split -> train baseline
    -> train improved -> handle imbalance -> calibrate -> evaluate -> report

One target at a time, baseline and improved side by side, every number
measured on a test window that is strictly later in calendar time than
anything either model saw.

Run it with ``python scripts/run_prediction.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config, data_io, features as feature_module
from . import calibration, estimators, evaluation, splitting


# --------------------------------------------------------------------------
# 1. Ingestion
# --------------------------------------------------------------------------
def load_panel(sample_loans: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the labelled panel, the static attributes and the Phase 1 DQ scores.

    ``sample_loans`` subsets by **loan**, never by row: sampling rows would
    punch holes in the very histories the rolling features are built from.
    """
    panel = data_io.load_train()
    static = data_io.load_static()

    dq_scores = pd.DataFrame()
    if config.DQ_SCORES_PATH.exists():
        dq_scores = pd.read_csv(config.DQ_SCORES_PATH, low_memory=False)

    if sample_loans is not None and not panel.empty:
        rng = np.random.default_rng(config.RANDOM_SEED)
        loans = panel[config.ID_COL].unique()
        keep = rng.choice(loans, size=min(sample_loans, len(loans)), replace=False)
        panel = panel[panel[config.ID_COL].isin(keep)]

    return panel, static, dq_scores


def build_modelling_frame(
    panel: pd.DataFrame, static: pd.DataFrame, dq_scores: pd.DataFrame
) -> tuple[pd.DataFrame, feature_module.FeatureSpec]:
    """Feature engineering + the leakage assertion, in one call."""
    return feature_module.build_feature_matrix(panel, static=static, dq_scores=dq_scores)


# --------------------------------------------------------------------------
# 2. Result containers
# --------------------------------------------------------------------------
@dataclass
class FittedModel:
    """A trained model plus everything needed to reproduce its predictions."""

    target: str
    variant: str  # "baseline" | "improved"
    backend: str
    estimator: Any
    calibrated: Any | None
    calibration_method: str | None
    harmoniser: estimators.CategoryHarmoniser
    numeric: list[str]
    categorical: list[str]
    threshold: float = 0.5
    classes: list | None = None
    best_iteration: int | None = None

    @property
    def feature_names(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    def _active(self, calibrated: bool = True) -> Any:
        return self.calibrated if (calibrated and self.calibrated is not None) else self.estimator

    def predict_proba(self, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Raw probability matrix, in the fitted estimator's own class order."""
        X = estimators.prepare_matrix(df, self.numeric, self.categorical, self.harmoniser)
        return self._active(calibrated).predict_proba(X)

    def predict_class_proba(self, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """
        Multiclass probabilities re-ordered into ``self.classes``.

        Multiclass labels are integer-encoded before fitting (LightGBM's
        holdout sets are not passed through the wrapper's label encoder, so a
        string label reaches the C++ layer and fails there). This maps the
        estimator's integer class order back onto the state names.
        """
        if self.classes is None:
            raise ValueError("predict_class_proba is only defined for multiclass models")
        model = self._active(calibrated)
        proba = self.predict_proba(df, calibrated=calibrated)
        model_labels = [self.classes[int(i)] for i in model.classes_]
        return _align_proba(proba, model_labels, self.classes)


@dataclass
class TargetOutcome:
    """Everything produced for one target: models, metrics rows, diagnostics."""

    target: str
    task: str
    split_audit: dict
    models: dict[str, FittedModel] = field(default_factory=dict)
    metric_rows: list[dict] = field(default_factory=list)
    diagnostics: dict[str, pd.DataFrame] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 3. Training -- binary targets
# --------------------------------------------------------------------------
def train_binary_target(
    frame: pd.DataFrame,
    spec: feature_module.FeatureSpec,
    target: str,
    backend: str,
    boundaries: splitting.SplitBoundaries,
    precision_floor: float = config.PRECISION_FLOOR,
    calibrate_models: bool = True,
    verbose: bool = True,
) -> TargetOutcome:
    """
    Train, calibrate and evaluate the baseline and improved models for one
    binary target.

    The validation window does three jobs -- early stopping, calibration
    fitting, and threshold selection -- and the test window does none of them.
    That is the containment: validation is spent, test is clean, and the
    reported numbers are the ones a deployment would actually see.
    """
    split = splitting.make_time_split(frame, target, boundaries=boundaries)
    outcome = TargetOutcome(target=target, task="binary", split_audit=split.audit)

    y_train = split.train[target].astype(int).to_numpy()
    y_valid = split.valid[target].astype(int).to_numpy()
    y_test = split.test[target].astype(int).to_numpy()

    if verbose:
        print(
            f"  split: train={len(y_train):,} ({y_train.mean():.2%} pos) "
            f"valid={len(y_valid):,} ({y_valid.mean():.2%} pos) "
            f"test={len(y_test):,} ({y_test.mean():.2%} pos)"
        )

    for variant in ("baseline", "improved"):
        numeric, categorical = spec.split_for(variant)
        feature_module.assert_no_leaky_features([*numeric, *categorical])

        harmoniser = estimators.CategoryHarmoniser.fit(split.train, categorical)
        X_train = estimators.prepare_matrix(split.train, numeric, categorical, harmoniser)
        X_valid = estimators.prepare_matrix(split.valid, numeric, categorical, harmoniser)
        X_test = estimators.prepare_matrix(split.test, numeric, categorical, harmoniser)

        started = time.perf_counter()
        if variant == "baseline":
            model = estimators.make_baseline_model(numeric, categorical)
            model.fit(X_train, y_train)
            used_backend = "logistic-regression"
            best_iter = None
        else:
            weight = estimators.compute_scale_pos_weight(y_train)
            model = estimators.make_improved_model(backend, n_classes=2, scale_pos_weight=weight)
            model = estimators.fit_improved_model(
                model, backend, X_train, y_train, X_valid, y_valid
            )
            used_backend = backend
            best_iter = estimators.best_iteration(model, backend)
        fit_seconds = time.perf_counter() - started

        raw_valid = model.predict_proba(X_valid)[:, 1]
        raw_test = model.predict_proba(X_test)[:, 1]

        calibrated_model, method = (None, None)
        if calibrate_models and len(np.unique(y_valid)) == 2:
            calibrated_model, method = calibration.calibrate(model, X_valid, y_valid)
            cal_valid = calibrated_model.predict_proba(X_valid)[:, 1]
            cal_test = calibrated_model.predict_proba(X_test)[:, 1]
        else:
            cal_valid, cal_test = raw_valid, raw_test

        # Threshold chosen on validation, on the same (calibrated) scale the
        # test window will be scored on, then frozen.
        threshold = evaluation.tune_threshold_for_f1(y_valid, cal_valid)

        metrics = evaluation.binary_metrics(
            y_test, cal_test, threshold, y_prob_uncalibrated=raw_test, precision_floor=precision_floor
        )
        row = {
            "target": target,
            "task": "binary",
            "model": variant,
            "backend": used_backend,
            "n_features": len(numeric) + len(categorical),
            "test_n": metrics["n"],
            "test_positive_rate": metrics["positive_rate"],
            "calibration_method": method or "none",
            "best_iteration": best_iter,
            "fit_seconds": round(fit_seconds, 2),
            **{k: v for k, v in metrics.items() if k not in {"n", "positive_rate"}},
        }
        outcome.metric_rows.append(row)

        fitted = FittedModel(
            target=target,
            variant=variant,
            backend=used_backend,
            estimator=model,
            calibrated=calibrated_model,
            calibration_method=method,
            harmoniser=harmoniser,
            numeric=numeric,
            categorical=categorical,
            threshold=threshold,
            best_iteration=best_iter,
        )
        outcome.models[variant] = fitted

        outcome.diagnostics[f"reliability_{variant}"] = calibration.reliability_table(y_test, cal_test)
        if variant == "improved":
            outcome.diagnostics["lift"] = evaluation.lift_table(y_test, cal_test)
            importance = _feature_importance(model, X_train.columns.tolist())
            if importance is not None:
                outcome.diagnostics["importance"] = importance

        if verbose:
            print(
                f"    {variant:<9} {used_backend:<20} "
                f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
                f"F1={metrics['f1']:.4f} Brier {metrics.get('brier_uncalibrated', float('nan')):.4f}"
                f"->{metrics['brier']:.4f}  ({fit_seconds:.1f}s)"
            )

    return outcome


# --------------------------------------------------------------------------
# 4. Training -- multiclass next_state
# --------------------------------------------------------------------------
def train_multiclass_target(
    frame: pd.DataFrame,
    spec: feature_module.FeatureSpec,
    backend: str,
    boundaries: splitting.SplitBoundaries,
    target: str = config.MULTICLASS_TARGET,
    calibrate_models: bool = True,
    verbose: bool = True,
) -> TargetOutcome:
    """
    Train baseline and improved classifiers for the next-month state.

    Macro-F1 is the headline rather than accuracy: ``Current`` is ~90% of the
    panel, so accuracy is maximised by a model that never predicts a
    delinquency transition at all -- exactly the model that is useless.
    """
    split = splitting.make_time_split(frame, target, boundaries=boundaries)
    outcome = TargetOutcome(target=target, task="multiclass", split_audit=split.audit)

    y_train = split.train[target].astype(str).to_numpy()
    y_valid = split.valid[target].astype(str).to_numpy()
    y_test = split.test[target].astype(str).to_numpy()

    observed = set(y_train) | set(y_valid) | set(y_test)
    classes = [state for state in config.STATUS_ORDER if state in observed]
    classes += sorted(observed - set(classes))

    # Integer-encode against the fixed class order: LightGBM's sklearn wrapper
    # label-encodes the training labels but hands the holdout labels to the C++
    # layer untouched, so string states fail there. Encoding once here also
    # fixes the probability column order for every backend.
    class_index = {label: i for i, label in enumerate(classes)}
    enc_train = np.array([class_index[label] for label in y_train])
    enc_valid = np.array([class_index[label] for label in y_valid])

    if verbose:
        print(f"  split: train={len(y_train):,} valid={len(y_valid):,} test={len(y_test):,}")
        print(f"  classes: {classes}")

    for variant in ("baseline", "improved"):
        numeric, categorical = spec.split_for(variant)
        feature_module.assert_no_leaky_features([*numeric, *categorical])

        harmoniser = estimators.CategoryHarmoniser.fit(split.train, categorical)
        X_train = estimators.prepare_matrix(split.train, numeric, categorical, harmoniser)
        X_valid = estimators.prepare_matrix(split.valid, numeric, categorical, harmoniser)
        X_test = estimators.prepare_matrix(split.test, numeric, categorical, harmoniser)

        started = time.perf_counter()
        if variant == "baseline":
            model = estimators.make_baseline_model(numeric, categorical)
            model.fit(X_train, enc_train)
            used_backend = "logistic-regression"
            best_iter = None
        else:
            model = estimators.make_improved_model(backend, n_classes=len(classes))
            model = estimators.fit_improved_model(
                model, backend, X_train, enc_train, X_valid, enc_valid
            )
            used_backend = backend
            best_iter = estimators.best_iteration(model, backend)
        fit_seconds = time.perf_counter() - started

        def _named(fitted: Any, X: pd.DataFrame) -> np.ndarray:
            labels = [classes[int(i)] for i in fitted.classes_]
            return _align_proba(fitted.predict_proba(X), labels, classes)

        raw_test = _named(model, X_test)

        calibrated_model, method = (None, None)
        if calibrate_models:
            calibrated_model, method = calibration.calibrate(model, X_valid, enc_valid)
            cal_test = _named(calibrated_model, X_test)
        else:
            cal_test = raw_test

        metrics = evaluation.multiclass_metrics(
            y_test, cal_test, classes, y_prob_uncalibrated=raw_test
        )
        row = {
            "target": target,
            "task": "multiclass",
            "model": variant,
            "backend": used_backend,
            "n_features": len(numeric) + len(categorical),
            "test_n": metrics["n"],
            "calibration_method": method or "none",
            "best_iteration": best_iter,
            "fit_seconds": round(fit_seconds, 2),
            **{k: v for k, v in metrics.items() if k != "n"},
        }
        outcome.metric_rows.append(row)

        outcome.models[variant] = FittedModel(
            target=target,
            variant=variant,
            backend=used_backend,
            estimator=model,
            calibrated=calibrated_model,
            calibration_method=method,
            harmoniser=harmoniser,
            numeric=numeric,
            categorical=categorical,
            classes=classes,
            best_iteration=best_iter,
        )

        outcome.diagnostics[f"per_class_{variant}"] = evaluation.per_class_report(
            y_test, cal_test, classes
        )
        if variant == "improved":
            outcome.diagnostics["confusion"] = evaluation.confusion_frame(y_test, cal_test, classes)
            importance = _feature_importance(model, X_train.columns.tolist())
            if importance is not None:
                outcome.diagnostics["importance"] = importance

        if verbose:
            print(
                f"    {variant:<9} {used_backend:<20} "
                f"macro-F1={metrics['macro_f1']:.4f} bal-acc={metrics['balanced_accuracy']:.4f} "
                f"Brier {metrics.get('brier_uncalibrated', float('nan')):.4f}->{metrics['brier']:.4f} "
                f"({fit_seconds:.1f}s)"
            )

    return outcome


def _align_proba(proba: np.ndarray, model_classes: list, wanted: list) -> np.ndarray:
    """Reorder a probability matrix into a fixed class order, zero-filling absentees."""
    out = np.zeros((proba.shape[0], len(wanted)), dtype="float64")
    lookup = {label: i for i, label in enumerate(model_classes)}
    for j, label in enumerate(wanted):
        if label in lookup:
            out[:, j] = proba[:, lookup[label]]
    return out


def _warn_on_missing_coverage(
    outcomes: dict[str, TargetOutcome], to_score: pd.DataFrame, variant: str, threshold: float = 0.99
) -> None:
    """
    Warn where a model feature is almost entirely null on the scoring panel.

    The realistic case is ``dq_score``: the Phase 1 profiler runs on the
    labelled panel, so an unlabelled panel scored before the profiler has seen
    it arrives with that column empty. The model still predicts -- boosters
    treat it as missing -- but the reviewer should know a feature the model
    was trained with is silently absent rather than discover it in a drifted
    prediction distribution.
    """
    reported: set[str] = set()
    for outcome in outcomes.values():
        model = outcome.models.get(variant)
        if model is None:
            continue
        for column in model.feature_names:
            if column in reported or column not in to_score.columns:
                continue
            if to_score[column].isna().mean() >= threshold:
                reported.add(column)
    if reported:
        print(
            f"[warn] scoring panel has no usable values for: {sorted(reported)} "
            "-- these features were available at training time"
        )


def _feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    """Gain/split importances where the backend exposes them (feeds Phase 7)."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return None
    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    total = frame["importance"].sum()
    frame["importance_pct"] = frame["importance"] / total if total else np.nan
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. Scoring an unlabelled panel
# --------------------------------------------------------------------------
def score_panel(
    outcomes: dict[str, TargetOutcome],
    history: pd.DataFrame,
    scoring_panel: pd.DataFrame,
    static: pd.DataFrame,
    dq_scores: pd.DataFrame | None = None,
    variant: str = "improved",
) -> pd.DataFrame:
    """
    Score an unlabelled panel (the organiser's test file) with the trained models.

    History matters: the rolling features for a 2024-01 row are computed from
    that loan's 2023 months, so the labelled panel is concatenated in *before*
    feature engineering and filtered out again afterwards. It supplies past
    context only -- no label from it is used, and no row of it is scored.

    Rows already in an absorbing state get a deterministic override rather than
    a model score: a prepaid loan does not later become delinquent.
    """
    history = history.copy()
    scoring_panel = scoring_panel.copy()
    history["_score_row"] = False
    scoring_panel["_score_row"] = True

    combined = pd.concat([history, scoring_panel], ignore_index=True, sort=False)
    engineered, _ = feature_module.build_feature_matrix(combined, static=static, dq_scores=dq_scores)
    to_score = engineered[engineered["_score_row"]].copy()

    absorbing = to_score["current_status"].isin(config.ABSORBING_STATES)
    output = to_score[[config.ID_COL, config.TIME_COL, "current_status"]].copy()

    _warn_on_missing_coverage(outcomes, to_score, variant)

    for target, outcome in outcomes.items():
        model = outcome.models.get(variant)
        if model is None:
            continue

        if outcome.task == "binary":
            column = f"prob_{target}"
            output[column] = model.predict_proba(to_score)[:, 1]
            # A resolved loan cannot experience a new forward event.
            output.loc[absorbing, column] = 0.0
        else:
            aligned = model.predict_class_proba(to_score)
            output["next_state"] = np.asarray(model.classes)[np.argmax(aligned, axis=1)]
            output["next_state_confidence"] = aligned.max(axis=1)
            # Absorbing states are self-transitions by construction.
            output.loc[absorbing, "next_state"] = to_score.loc[absorbing, "current_status"]
            output.loc[absorbing, "next_state_confidence"] = 1.0

    return output.drop(columns=["current_status"])


# --------------------------------------------------------------------------
# 6. End-to-end run
# --------------------------------------------------------------------------
def run_task2(
    sample_loans: int | None = None,
    backend: str = "auto",
    targets: list[str] | None = None,
    boundaries: splitting.SplitBoundaries | None = None,
    calibrate_models: bool = True,
    precision_floor: float = config.PRECISION_FLOOR,
    save_models: bool = True,
    score_test: bool = False,
    reports_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the whole of Task 2 and write every artifact.

    Returns a dict with the results table, the split audit, and the fitted
    outcomes, so a notebook or the Phase 6/7 code can consume them directly
    instead of re-reading the CSVs.
    """
    reports_dir = Path(reports_dir or config.REPORTS_DIR)
    task_dir = reports_dir / "task2"
    task_dir.mkdir(parents=True, exist_ok=True)

    resolved_backend = estimators.resolve_backend(backend)
    boundaries = boundaries or splitting.SplitBoundaries.from_config()
    requested = targets or [*config.BINARY_TARGETS, config.MULTICLASS_TARGET]

    if verbose:
        print(f"Backend: {resolved_backend}")
        print(
            f"Time-aware split -- train<= {boundaries.train_end:%Y-%m}, "
            f"valid<= {boundaries.valid_end:%Y-%m}, test<= {boundaries.test_end:%Y-%m} "
            "(purged by each target's horizon; no random split anywhere)"
        )

    panel, static, dq_scores = load_panel(sample_loans=sample_loans)
    if panel.empty:
        raise FileNotFoundError(f"No labelled panel found at {config.TRAIN_PATH}")

    if verbose:
        print(f"Loaded {len(panel):,} rows / {panel[config.ID_COL].nunique():,} loans")
        print("Engineering features...")

    frame, spec = build_modelling_frame(panel, static, dq_scores)
    if verbose:
        print(
            f"Feature matrix: {len(spec.full)} model features "
            f"({len(spec.numeric)} numeric, {len(spec.categorical)} categorical); "
            f"baseline uses {len(spec.baseline)}"
        )

    # The feature dictionary is generated from the matrix that was actually
    # built, so it cannot drift away from what the models are trained on.
    # run_profiling.py folds it into the Data Intelligence Report when present.
    dictionary = feature_module.feature_dictionary(frame, spec)
    dictionary.to_csv(reports_dir / "feature_dictionary.csv", index=False)
    (reports_dir / "feature_dictionary.md").write_text(
        feature_module.feature_dictionary_markdown(dictionary, spec), encoding="utf-8"
    )
    undocumented = dictionary.loc[dictionary["family"] == "unclassified", "feature"].tolist()
    if undocumented:
        print(f"[warn] {len(undocumented)} feature(s) have no dictionary entry: {undocumented}")

    outcomes: dict[str, TargetOutcome] = {}
    metric_rows: list[dict] = []

    for target in requested:
        if target not in frame.columns:
            print(f"[skip] {target}: not present in the panel")
            continue
        if verbose:
            print(f"\n== {target} ==")

        if target == config.MULTICLASS_TARGET:
            outcome = train_multiclass_target(
                frame, spec, resolved_backend, boundaries,
                calibrate_models=calibrate_models, verbose=verbose,
            )
        else:
            outcome = train_binary_target(
                frame, spec, target, resolved_backend, boundaries,
                precision_floor=precision_floor, calibrate_models=calibrate_models, verbose=verbose,
            )

        outcomes[target] = outcome
        metric_rows.extend(outcome.metric_rows)

        for name, table in outcome.diagnostics.items():
            table.to_csv(task_dir / f"{target}__{name}.csv", index=False)

    results = evaluation.results_frame(metric_rows)
    split_audit = splitting.split_summary_frame(
        {t: _outcome_to_split(o) for t, o in outcomes.items()}
    )

    results.to_csv(reports_dir / "task2_model_results.csv", index=False)
    split_audit.to_csv(reports_dir / "task2_split_audit.csv", index=False)
    (reports_dir / "task2_model_results.md").write_text(
        _results_document(results, split_audit, resolved_backend, spec), encoding="utf-8"
    )

    if save_models:
        _save_models(outcomes)

    predictions = None
    if score_test:
        test_panel = data_io.load_test()
        if test_panel.empty:
            print("[skip] scoring: no test panel found")
        else:
            if verbose:
                print(f"\nScoring {len(test_panel):,} unlabelled rows...")
            predictions = score_panel(outcomes, panel, test_panel, static, dq_scores)
            predictions.to_csv(reports_dir / "task2_test_predictions.csv", index=False)

    if verbose:
        print("\n" + evaluation.results_markdown(results))
        print(f"\nWrote {reports_dir / 'task2_model_results.csv'}")

    return {
        "results": results,
        "split_audit": split_audit,
        "outcomes": outcomes,
        "feature_spec": spec,
        "backend": resolved_backend,
        "predictions": predictions,
    }


def _outcome_to_split(outcome: TargetOutcome) -> splitting.TimeSplit:
    """Adapter so the audit summariser can read an outcome's stored audit dict."""
    empty = pd.DataFrame()
    return splitting.TimeSplit(
        target=outcome.target,
        horizon_months=outcome.split_audit["horizon_months"],
        train=empty,
        valid=empty,
        test=empty,
        audit=outcome.split_audit,
    )


def _save_models(outcomes: dict[str, TargetOutcome]) -> None:
    """Persist each fitted model with its feature list, harmoniser and threshold."""
    import joblib

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for target, outcome in outcomes.items():
        for variant, model in outcome.models.items():
            path = config.MODELS_DIR / f"{target}__{variant}.joblib"
            joblib.dump(model, path)
            manifest.append(
                {
                    "target": target,
                    "variant": variant,
                    "backend": model.backend,
                    "n_features": len(model.feature_names),
                    "threshold": model.threshold,
                    "calibration": model.calibration_method,
                    "path": str(path.relative_to(config.PROJECT_ROOT)),
                }
            )
    (config.MODELS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _results_document(
    results: pd.DataFrame,
    split_audit: pd.DataFrame,
    backend: str,
    spec: feature_module.FeatureSpec,
) -> str:
    """The Task 2 results page: comparison table, split evidence, method notes."""
    lines = [
        "# Task 2 -- Loan Performance Prediction",
        "",
        "Baseline versus improved model, every target, measured on a held-out",
        "window that is strictly later in calendar time than any training row.",
        "",
        "## Results",
        "",
        evaluation.results_markdown(results),
        "",
        "## Time-aware split",
        "",
        "Split is by `reporting_month`. **No random row-level split is used anywhere.**",
        "A monthly panel repeats each loan dozens of times, so a random split would put",
        "one month of a loan in training and the next month of the same loan in test.",
        "",
        "Each target's training window is additionally **purged** by its own forward",
        "horizon: a row labelled over the next 12 months is dropped from training if",
        "those 12 months reach into the validation window.",
        "",
        split_audit.to_markdown(index=False),
        "",
        "## Method notes",
        "",
        f"- **Backend**: {backend} for the improved model; logistic regression for the baseline.",
        f"- **Features**: {len(spec.full)} for the improved model "
        f"({len(spec.numeric)} numeric, {len(spec.categorical)} categorical); "
        f"{len(spec.baseline)} for the baseline.",
        "- **Imbalance**: `scale_pos_weight = negatives/positives` on the booster, "
        "`class_weight='balanced'` on the baseline. Resampling was rejected: synthetic "
        "minority rows would carry rolling histories belonging to no real loan.",
        "- **Calibration**: fitted on the validation window with the base model frozen; "
        "isotonic where the minority class supports it, Platt scaling otherwise. Brier is "
        "reported before and after.",
        "- **Threshold**: tuned for F1 on validation, then frozen before test is scored. "
        "0.5 is not a meaningful cut on a reweighted model with a low base rate.",
        "- **Absorbing states**: rows already Default or Prepaid are excluded from training "
        "and evaluation and answered by a deterministic rule at scoring time.",
        "",
    ]
    return "\n".join(lines)
