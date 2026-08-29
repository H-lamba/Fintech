"""
Measuring the detectors, and the ablation that justifies the hybrid.

The headline table is an ablation, not a single score, because the interesting
claim in Task 4 is *that the combination beats either half* -- and that claim
is only credible if both halves are measured on their own first:

===========================  ===================================================
row-level rules              the Phase 1 engine, unchanged
+ sequence detectors         adds the two checks a row-level engine cannot express
isolation forest             unsupervised, no labels, no rule indicators
hybrid (rules OR forest)     the noisy-OR combination
supervised                   learned over all signals; the deployed ranking
===========================  ===================================================

Precision and recall at a fixed queue size matter more here than AUC. A
reviewer works a finite list: "of the 500 records we can look at this month,
how many are real?" is the operational question, and precision@k answers it
where an AUC does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_at_threshold(y_true: np.ndarray, flagged: np.ndarray) -> dict:
    """Precision / recall / F1 for a hard flag, plus the queue size it implies."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(flagged).astype(int)
    return {
        "flagged": int(flagged.sum()),
        "flagged_pct": float(flagged.mean()),
        "precision": float(precision_score(y_true, flagged, zero_division=0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0)),
        "f1": float(f1_score(y_true, flagged, zero_division=0)),
    }


def flag_top_k(score: np.ndarray, k: int) -> np.ndarray:
    """
    Flag the ``k`` highest-scoring records.

    A label-free score has no natural cut point, and a fixed one is arbitrary:
    the Isolation Forest score here is a rank percentile, so any threshold near
    0.5 flags half the book. Holding the *queue size* fixed instead makes the
    detectors comparable at equal reviewer cost, which is the comparison a
    servicer actually cares about -- "for the same number of records we can
    look at, which detector puts the real ones at the top?"
    """
    score = np.asarray(score, dtype=float)
    flagged = np.zeros(len(score), dtype=bool)
    if k > 0:
        flagged[np.argsort(-score)[: min(k, len(score))]] = True
    return flagged


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> dict:
    """
    Precision and recall in the top ``k`` records by score.

    This is the reviewer's actual constraint: the queue has a length, and the
    question is what fraction of it is worth opening.
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(score, dtype=float))[:k]
    hits = int(y_true[order].sum())
    return {
        f"precision@{k}": hits / max(len(order), 1),
        f"recall@{k}": hits / max(int(y_true.sum()), 1),
    }


def score_metrics(
    y_true: np.ndarray, score: np.ndarray, queue_sizes: tuple[int, ...] = (100, 500, 1000)
) -> dict:
    """Ranking quality for a continuous anomaly score."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)

    metrics: dict = {"n": int(len(y_true)), "positives": int(y_true.sum())}
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, score))
        metrics["pr_auc"] = float(average_precision_score(y_true, score))
    for k in queue_sizes:
        metrics.update(precision_at_k(y_true, score, min(k, len(y_true))))
    return metrics


def tune_threshold_for_f1(y_true: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising F1 on the window given (intended: validation)."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    denominator = precision[:-1] + recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0, 2 * precision[:-1] * recall[:-1] / denominator, 0.0)
    return float(thresholds[int(np.nanargmax(f1))]) if len(f1) else 0.5


def ablation_frame(rows: list[dict]) -> pd.DataFrame:
    """The detector comparison table, ordered simplest to most capable."""
    order = {
        "row-level rules": 0,
        "+ sequence detectors": 1,
        "isolation forest": 2,
        "hybrid (rules + forest)": 3,
        "supervised (record state only)": 4,
        "supervised (no sequence flags)": 5,
        "supervised (all signals)": 6,
    }
    frame = pd.DataFrame(rows)
    frame["_o"] = frame["detector"].map(order).fillna(99)
    leading = [
        "detector", "labels_used", "flagged", "flagged_pct", "precision", "recall", "f1",
        "roc_auc", "pr_auc",
    ]
    ordered = [c for c in leading if c in frame.columns]
    rest = [c for c in frame.columns if c not in ordered and not c.startswith("_")]
    return frame.sort_values("_o")[[*ordered, *rest]].reset_index(drop=True)


def multiclass_metrics(y_true: np.ndarray, proba: np.ndarray, classes: list) -> dict:
    """
    Macro-F1 led, because ``None`` is 97.4% of records.

    Accuracy on this task is maximised by a model that never predicts an
    exception at all, which is the one model guaranteed to be useless.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=classes, zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()),
    }


def per_class_report(y_true: np.ndarray, proba: np.ndarray, classes: list) -> pd.DataFrame:
    """Precision / recall / F1 per exception type."""
    y_pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    rows = [
        {"exception_type": label, **{k: report[label][k] for k in ("precision", "recall", "f1-score", "support")}}
        for label in classes if label in report
    ]
    return pd.DataFrame(rows).rename(columns={"f1-score": "f1"})


def confusion_frame(y_true: np.ndarray, proba: np.ndarray, classes: list) -> pd.DataFrame:
    y_pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    return pd.DataFrame(
        matrix, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]
    ).reset_index(names="")


def brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(brier_score_loss(np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)))


def results_markdown(frame: pd.DataFrame, float_format: str = ".4f") -> str:
    """Render a metrics table, with inapplicable cells as an explicit dash."""
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        values = display[column].dropna()
        spec = "d" if len(values) and (values % 1 == 0).all() else float_format
        display[column] = display[column].map(
            lambda v, spec=spec: "--" if pd.isna(v) else format(int(v) if spec == "d" else v, spec)
        )
    return display.to_markdown(index=False)
