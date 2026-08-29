"""
Metrics, threshold selection and the results table.

Every metric Task 2 asks for is computed here, on the held-out test window,
for every (target, model) pair:

============================  ==============================================
ROC-AUC                       ranking quality, imbalance-insensitive
PR-AUC (average precision)    ranking quality where the positives are rare --
                              the metric that actually moves when a model
                              stops finding the minority class
F1                            at a threshold tuned on validation, not 0.5
Recall @ fixed precision      the operating point a servicer can staff: "if
                              we only work queues that are >= 50% precise,
                              what share of true events do we catch?"
Brier score                   probability quality, before and after calibration
Macro-F1                      the multiclass ``next_state`` headline, so the
                              rare states are not drowned by ``Current``
============================  ==============================================

A note on thresholds: 0.5 is meaningless on a reweighted model with a 3% base
rate, so the decision threshold is tuned to maximise F1 on the validation
window and then *fixed* before the test window is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .. import config


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------
def tune_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising F1 on the data given (intended: the validation window)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one more point than thresholds.
    denominator = precision[:-1] + recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0, 2 * precision[:-1] * recall[:-1] / denominator, 0.0)
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.nanargmax(f1))])


def recall_at_precision(
    y_true: np.ndarray, y_prob: np.ndarray, floor: float = config.PRECISION_FLOOR
) -> tuple[float, float]:
    """
    Best achievable recall subject to ``precision >= floor``.

    Returns ``(recall, threshold)``; ``(0.0, nan)`` when the floor is
    unreachable at any threshold, which is itself a finding worth reporting.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.0, float("nan")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    feasible = precision[:-1] >= floor
    if not feasible.any():
        return 0.0, float("nan")

    best = int(np.argmax(np.where(feasible, recall[:-1], -1.0)))
    return float(recall[best]), float(thresholds[best])


# --------------------------------------------------------------------------
# Binary metrics
# --------------------------------------------------------------------------
def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    y_prob_uncalibrated: np.ndarray | None = None,
    precision_floor: float = config.PRECISION_FLOOR,
) -> dict:
    """
    Every binary metric for one (target, model) pair on one window.

    ``y_prob`` is the calibrated probability where calibration was applied;
    ``y_prob_uncalibrated`` is kept alongside so the Brier before/after pair
    can be reported from a single call.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype="float64")
    y_pred = (y_prob >= threshold).astype(int)

    single_class = len(np.unique(y_true)) < 2
    recall_at_floor, threshold_at_floor = recall_at_precision(y_true, y_prob, precision_floor)

    metrics = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"recall_at_precision_{precision_floor:g}": recall_at_floor,
        "threshold": float(threshold),
        "threshold_at_precision_floor": threshold_at_floor,
        "brier": float(brier_score_loss(y_true, y_prob)),
    }

    if y_prob_uncalibrated is not None:
        raw = np.asarray(y_prob_uncalibrated, dtype="float64")
        metrics["brier_uncalibrated"] = float(brier_score_loss(y_true, raw))
        metrics["brier_calibrated"] = metrics["brier"]
        metrics["brier_improvement"] = metrics["brier_uncalibrated"] - metrics["brier_calibrated"]

    return metrics


# --------------------------------------------------------------------------
# Multiclass metrics
# --------------------------------------------------------------------------
def multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray, classes: list) -> float:
    """
    Multiclass Brier score: mean squared error against the one-hot outcome,
    summed over classes. Lower is better; 0 is perfect.
    """
    y_true = np.asarray(y_true)
    onehot = np.zeros_like(y_prob, dtype="float64")
    index = {label: i for i, label in enumerate(classes)}
    for row, label in enumerate(y_true):
        if label in index:
            onehot[row, index[label]] = 1.0
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list,
    y_prob_uncalibrated: np.ndarray | None = None,
) -> dict:
    """Headline metrics for ``next_state``, led by macro-F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(classes)[np.argmax(y_prob, axis=1)]

    # ``roc_auc_score`` and ``log_loss`` both bind probability *columns* to
    # labels in sorted order (via LabelBinarizer) regardless of the order the
    # ``labels`` argument is given in. This pipeline keeps ``classes`` in
    # severity order for readability, so the columns are re-sorted here before
    # those two metrics see them -- without this they silently score the wrong
    # column against each class.
    order = np.argsort(np.asarray(classes, dtype=object))
    sorted_classes = [classes[i] for i in order]
    proba_sorted = y_prob[:, order]

    try:
        roc_auc = float(
            roc_auc_score(
                y_true, proba_sorted, multi_class="ovr", average="macro", labels=sorted_classes
            )
        )
    except ValueError:
        # A class absent from the evaluation window makes OvR AUC undefined.
        roc_auc = float("nan")

    metrics = {
        "n": int(len(y_true)),
        "n_classes": len(classes),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=classes, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc_ovr_macro": roc_auc,
        "log_loss": float(log_loss(y_true, proba_sorted, labels=sorted_classes)),
        "brier": multiclass_brier(y_true, y_prob, classes),
    }

    if y_prob_uncalibrated is not None:
        metrics["brier_uncalibrated"] = multiclass_brier(y_true, y_prob_uncalibrated, classes)
        metrics["brier_calibrated"] = metrics["brier"]
        metrics["brier_improvement"] = metrics["brier_uncalibrated"] - metrics["brier_calibrated"]

    return metrics


def per_class_report(y_true: np.ndarray, y_prob: np.ndarray, classes: list) -> pd.DataFrame:
    """Precision / recall / F1 per performance state -- where macro-F1 comes from."""
    y_pred = np.asarray(classes)[np.argmax(y_prob, axis=1)]
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    rows = [
        {"class": label, **{k: report[label][k] for k in ("precision", "recall", "f1-score", "support")}}
        for label in classes
        if label in report
    ]
    return pd.DataFrame(rows).rename(columns={"f1-score": "f1"})


def confusion_frame(y_true: np.ndarray, y_prob: np.ndarray, classes: list) -> pd.DataFrame:
    """Confusion matrix as a labelled frame, for the model card."""
    y_pred = np.asarray(classes)[np.argmax(y_prob, axis=1)]
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    return pd.DataFrame(matrix, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])


# --------------------------------------------------------------------------
# Results table
# --------------------------------------------------------------------------
RESULTS_COLUMNS = [
    "target",
    "task",
    "model",
    "backend",
    "n_features",
    "test_n",
    "test_positive_rate",
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    f"recall_at_precision_{config.PRECISION_FLOOR:g}",
    "macro_f1",
    "brier_uncalibrated",
    "brier_calibrated",
    "brier_improvement",
    "calibration_method",
    "threshold",
]


def results_frame(rows: list[dict]) -> pd.DataFrame:
    """
    Assemble the baseline-versus-improved comparison table.

    One row per (target, model). Columns that do not apply to a given task --
    macro-F1 on a binary target, PR-AUC on the multiclass one -- are left NaN
    rather than dropped, so the table stays rectangular and diffable.
    """
    frame = pd.DataFrame(rows)
    for column in RESULTS_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    ordered = [c for c in RESULTS_COLUMNS if c in frame.columns]
    extra = [c for c in frame.columns if c not in ordered]
    frame = frame[[*ordered, *extra]]

    target_order = {t: i for i, t in enumerate([*config.BINARY_TARGETS, config.MULTICLASS_TARGET])}
    frame["_t"] = frame["target"].map(target_order).fillna(99)
    frame["_m"] = frame["model"].map({"baseline": 0, "improved": 1}).fillna(9)
    return frame.sort_values(["_t", "_m"]).drop(columns=["_t", "_m"]).reset_index(drop=True)


def results_markdown(frame: pd.DataFrame, float_format: str = ".4f") -> str:
    """Render the comparison table as markdown for the README / model card."""
    display = frame.copy()
    drop_if_empty = ["macro_f1", "pr_auc", "roc_auc"]
    for column in drop_if_empty:
        if column in display.columns and display[column].isna().all():
            display = display.drop(columns=[column])
    # NaN is correct in the CSV (the metric simply does not apply to that
    # task) but reads as a defect in a rendered table, so numeric columns are
    # pre-formatted and blanks become an explicit dash.
    for column in display.select_dtypes(include="number").columns:
        values = display[column].dropna()
        # Counts (row totals, tree counts) read badly as 74.0000.
        spec = "d" if len(values) and (values % 1 == 0).all() else float_format
        display[column] = display[column].map(
            lambda v, spec=spec: "--" if pd.isna(v) else format(int(v) if spec == "d" else v, spec)
        )
    return display.to_markdown(index=False)


def lift_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Event rate by predicted-risk decile -- the "does the ranking work" sanity
    check a credit reviewer trusts more than an AUC.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False) + 1
    grouped = frame.groupby("decile").agg(n=("y", "size"), events=("y", "sum"), mean_prob=("p", "mean"))
    grouped["event_rate"] = grouped["events"] / grouped["n"]
    grouped["lift"] = grouped["event_rate"] / frame["y"].mean()
    return grouped.reset_index().sort_values("decile", ascending=False)
