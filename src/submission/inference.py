"""
Scoring the unlabelled test panel.

Every model used here was fitted in an earlier phase and is loaded from disk.
Nothing is refitted, and nothing is fitted on the test data -- not a model, not
an imputer, not a scaler, not a categorical encoding. That is not a convention
this module follows by care; it is a property of how the artefacts were saved.
Each :class:`~src.models.predict.FittedModel` carries the *fitted* sklearn
pipeline and the ``CategoryHarmoniser`` whose level set was learned on the
training window, and its ``predict_proba`` calls ``transform`` on both. There
is no code path that reaches ``fit`` at inference time, and
``tests/test_submission.py`` asserts it by checking the fitted statistics are
unchanged after scoring.

History without labels
----------------------
The rolling features for a 2024-01 row are computed from that loan's 2023
months, so the labelled panel is concatenated in *before* feature engineering
and filtered out again afterwards. It supplies backward-looking context only:
no label from it enters a feature, ``assert_no_leaky_features`` fails the run
if one does, and no row of it is scored or submitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config, data_io, features as feature_module
from ..anomaly import features as anomaly_features
from ..anomaly import evaluation as anomaly_evaluation
from ..anomaly import models as anomaly_models
from ..anomaly import signals as anomaly_signals
from ..models import estimators
from ..scenario.project import load_models


@dataclass
class InferenceResult:
    """Scored rows plus the provenance a reviewer needs to trust them."""

    scored: pd.DataFrame
    provenance: dict = field(default_factory=dict)


def _clean_driver_names(raw: str, limit: int = config.SUBMISSION_TOP_DRIVERS) -> str:
    """
    ``top_drivers`` as comma-separated feature names.

    The Phase 5 driver string carries values as well -- "post absorbing
    activity=1; transition rarity=0.00125" -- which is right for a reviewer
    reading a queue and wrong for a submission column specified as feature
    names. Values are stripped here rather than at source so the reviewer queue
    keeps them.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    names = []
    for part in raw.split(";"):
        name = part.split("=")[0].strip()
        name = name.split("(")[0].strip()
        if name and name not in names:
            names.append(name)
    return ", ".join(names[:limit])


def build_scoring_frame(
    history: pd.DataFrame,
    test: pd.DataFrame,
    static: pd.DataFrame,
    dq_scores: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer features over history + test, then return only the test rows.

    Returns ``(scoring_rows, engineered_all)``. The second is kept because the
    anomaly layer's sequence detectors need the same combined view.
    """
    history = history.copy()
    test = test.copy()
    history["_score_row"] = False
    test["_score_row"] = True

    combined = pd.concat([history, test], ignore_index=True, sort=False)
    engineered, spec = feature_module.build_feature_matrix(
        combined, static=static, dq_scores=dq_scores
    )
    feature_module.assert_no_leaky_features(spec.full)

    scoring = engineered[engineered["_score_row"]].copy()
    return scoring, engineered


def score_targets(scoring: pd.DataFrame, variant: str = "improved") -> pd.DataFrame:
    """
    Predicted probability per binary target, plus the multiclass next state.

    Absorbing rows get a deterministic override rather than a model score: a
    prepaid loan cannot subsequently become delinquent, and asking a model to
    say so is asking it to disagree with the definition of the state.
    """
    targets = {**config.SCENARIO_TARGETS, "next_6m_delinquency_flag": "delinquency_6m"}
    models = load_models(targets={t: t for t in config.SUBMISSION_PROBABILITY_COLUMNS}, variant=variant)

    out = pd.DataFrame(index=scoring.index)
    absorbing = scoring["current_status"].isin(config.ABSORBING_STATES)

    for target, column in config.SUBMISSION_PROBABILITY_COLUMNS.items():
        model = models.get(target)
        if model is None:
            out[column] = np.nan
            continue
        out[column] = model.predict_proba(scoring)[:, 1]
        out.loc[absorbing, column] = 0.0

    state_model = load_models(targets={config.MULTICLASS_TARGET: "next_state"}, variant=variant)[
        config.MULTICLASS_TARGET
    ]
    classes = list(state_model.classes or [])
    proba = state_model.predict_proba(scoring)

    if classes:
        labels = [classes[int(i)] for i in state_model._active().classes_]
        aligned = np.zeros((len(scoring), len(classes)))
        lookup = {label: i for i, label in enumerate(labels)}
        for position, label in enumerate(classes):
            if label in lookup:
                aligned[:, position] = proba[:, lookup[label]]
    else:
        aligned = proba
        classes = list(state_model._active().classes_)

    out["next_state"] = np.asarray(classes)[np.argmax(aligned, axis=1)]
    out["confidence"] = aligned.max(axis=1)

    # An absorbing state is a self-transition by construction, and the model is
    # not asked to rediscover that.
    out.loc[absorbing, "next_state"] = scoring.loc[absorbing, "current_status"]
    out.loc[absorbing, "confidence"] = 1.0

    return out


def score_anomalies(
    engineered: pd.DataFrame, scoring_mask: pd.Series, dq_scores: pd.DataFrame | None
) -> pd.DataFrame:
    """
    Hybrid anomaly score, exception flag and type for the test rows.

    The deterministic signals are evaluated over the **combined** panel because
    the sequence detectors need each loan's history -- a post-absorption row in
    2024 is only detectable against the terminal row that preceded it, which may
    sit in the labelled panel.
    """
    matrix, severities, _ = anomaly_signals.build_signal_matrix(
        engineered, data_io.load_validation_rules()
    )
    rule_scores = anomaly_signals.rule_score(matrix, severities)
    triggered = anomaly_signals.triggered_rules(matrix, severities)

    built = anomaly_features.build_features(engineered, matrix, rule_scores, dq_scores)
    frame = built.data

    # The forest is fitted on the labelled rows only; the test rows are scored,
    # never fitted on.
    fit_rows = frame.loc[~scoring_mask]
    forest = anomaly_models.IsolationForestDetector.fit(
        fit_rows, built.unsupervised_columns,
        max_samples=min(8192, max(len(fit_rows), 2)),
    )
    reference = forest.raw_score(fit_rows)

    scoring = frame.loc[scoring_mask]
    ml_score = forest.score(scoring, reference=reference)
    hybrid = anomaly_models.hybrid_score(scoring["rule_score"].to_numpy(), ml_score)

    out = pd.DataFrame(
        {
            "anomaly_score": hybrid,
            "rule_score": scoring["rule_score"].to_numpy(),
            "triggered_rules": triggered.loc[scoring.index].to_numpy(),
        },
        index=scoring.index,
    )

    # --- supervised exception heads ---------------------------------------
    # `exception_required` must be the *model's* judgement, not "any rule
    # fired". The raw rule flag fires on 13.8% of the book against a true base
    # rate of 2.6%, almost all of it one low-severity check; Phase 5 measured
    # the supervised head cutting that queue by a factor of six at 99.9%
    # precision. Fitted here on the labelled panel and applied to the test rows.
    labelled = frame.loc[~scoring_mask]
    targets = anomaly_features.prepare_targets(engineered.loc[~scoring_mask])
    if anomaly_features.EXCEPTION_FLAG in targets.columns and targets[
        anomaly_features.EXCEPTION_FLAG
    ].nunique() > 1:
        out = out.join(
            _supervised_exceptions(labelled, targets, scoring, built), how="left"
        )
    else:
        out["exception_probability"] = np.nan
        out["exception_threshold"] = np.nan
        out["predicted_exception_type"] = "No exception"

    return out


def _supervised_exceptions(
    labelled: pd.DataFrame, targets: pd.DataFrame, scoring: pd.DataFrame, built
) -> pd.DataFrame:
    """
    Fit the Phase 5 exception heads on the labelled panel and score the test rows.

    Refitting here rather than loading a saved artefact is deliberate: Phase 5's
    heads are fitted on a *time-split* of the labelled panel for honest
    evaluation, whereas a deployed scorer should use every labelled month it
    has. The fit uses labelled rows only -- no test row contributes to it.
    """
    numeric = [*built.signal_columns, *built.numeric_columns]
    categorical = built.categorical_columns
    dtypes = anomaly_models.category_dtypes(labelled, categorical)

    # A chronological tail of the labelled panel acts as the early-stopping
    # window, so the number of trees is not tuned on the rows being scored.
    ordered = labelled.sort_values(config.TIME_COL)
    cut = int(len(ordered) * 0.85)
    fit_rows, stop_rows = ordered.iloc[:cut], ordered.iloc[cut:]
    y_fit = targets.loc[fit_rows.index, anomaly_features.EXCEPTION_FLAG].to_numpy()
    y_stop = targets.loc[stop_rows.index, anomaly_features.EXCEPTION_FLAG].to_numpy()

    binary = anomaly_models.fit_supervised_head(
        fit_rows, y_fit, stop_rows, y_stop, numeric, categorical, dtypes
    )
    probability = binary.predict_proba(scoring, dtypes)[:, 1]

    # The threshold is tuned on the held-back window, never assumed to be 0.5.
    # The sequence detectors are near-perfect indicators, so average precision
    # saturates within ten boosting rounds and early stopping correctly halts;
    # the model ranks well but its probabilities stay compressed below 0.52. A
    # fixed 0.5 cut flagged one row in 78,409 against a 2.6% base rate -- the
    # model was right and the operating point was wrong.
    threshold = anomaly_evaluation.tune_threshold_for_f1(
        y_stop, binary.predict_proba(stop_rows, dtypes)[:, 1]
    )

    classes = anomaly_features.exception_classes(targets)
    type_head = anomaly_models.fit_supervised_head(
        fit_rows, targets.loc[fit_rows.index, anomaly_features.EXCEPTION_TYPE].to_numpy(),
        stop_rows, targets.loc[stop_rows.index, anomaly_features.EXCEPTION_TYPE].to_numpy(),
        numeric, categorical, dtypes, multiclass=True, class_order=classes,
    )
    type_proba = type_head.predict_proba(scoring, dtypes)

    out = pd.DataFrame(
        {
            "exception_probability": probability,
            "exception_threshold": threshold,
            "predicted_exception_type": np.asarray(classes)[np.argmax(type_proba, axis=1)],
        },
        index=scoring.index,
    )
    out.attrs["threshold"] = threshold
    out.attrs["best_iteration"] = anomaly_models.best_iteration(binary)
    return out


def run_inference(
    sample_loans: int | None = None, variant: str = "improved", verbose: bool = True
) -> InferenceResult:
    """Score the unlabelled test panel end to end."""
    if verbose:
        print("Loading the data pack...")
    history = data_io.load_train()
    test = data_io.load_test()
    static = data_io.load_static()
    if test.empty:
        raise FileNotFoundError(f"No test panel at {config.TEST_PATH}")

    dq_scores = (
        pd.read_csv(config.DQ_SCORES_PATH, low_memory=False)
        if config.DQ_SCORES_PATH.exists()
        else None
    )

    if sample_loans is not None:
        rng = np.random.default_rng(config.RANDOM_SEED)
        loans = test[config.ID_COL].unique()
        keep = set(rng.choice(loans, size=min(sample_loans, len(loans)), replace=False))
        test = test[test[config.ID_COL].isin(keep)]
        history = history[history[config.ID_COL].isin(keep)]

    if verbose:
        print(f"  history={len(history):,} rows | test={len(test):,} rows "
              f"({test[config.ID_COL].nunique():,} loans)")
        print("Engineering features over history + test...")

    scoring, engineered = build_scoring_frame(history, test, static, dq_scores)

    if verbose:
        print(f"Scoring {len(scoring):,} test rows with the saved {variant} models...")
    predictions = score_targets(scoring, variant=variant)

    if verbose:
        print("Running anomaly detection...")
    anomalies = score_anomalies(engineered, engineered["_score_row"], dq_scores)

    scored = pd.concat(
        [
            scoring[[config.ID_COL, config.TIME_COL, "current_status"]].reset_index(drop=True),
            predictions.reset_index(drop=True),
            anomalies.reset_index(drop=True),
        ],
        axis=1,
    )

    provenance = {
        "model_variant": variant,
        "history_rows": len(history),
        "test_rows": len(test),
        "scored_rows": len(scored),
        "loans_scored": int(scored[config.ID_COL].nunique()),
        "reporting_months": f"{scored[config.TIME_COL].min():%Y-%m} .. "
                            f"{scored[config.TIME_COL].max():%Y-%m}",
    }
    return InferenceResult(scored=scored, provenance=provenance)
