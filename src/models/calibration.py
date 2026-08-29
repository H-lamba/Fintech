"""
Probability calibration.

Why this step is not optional here
----------------------------------
``scale_pos_weight`` / ``class_weight="balanced"`` fixes the ranking problem
that severe imbalance creates, and breaks the probabilities while doing it: a
model trained on a reweighted objective emits scores roughly an order of
magnitude above the true event rate. For a securitisation use case the
*probability itself* is the deliverable -- it feeds expected-loss arithmetic
and the Task 6 scenario projections -- so a well-ranked but badly scaled score
is only half a model. Brier score is reported before and after so the size of
the correction is visible rather than asserted.

The calibrator is fitted on the **validation window** with the base model
frozen, and every calibrated number in the results table is then measured on
the untouched **test window**.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator

    _HAS_FROZEN = True
except ImportError:  # pragma: no cover - older scikit-learn
    FrozenEstimator = None  # type: ignore[assignment]
    _HAS_FROZEN = False

# Below this many minority events, isotonic regression overfits its own step
# function and sigmoid (Platt) is the safer, lower-variance choice.
ISOTONIC_MIN_EVENTS = 1000


def choose_method(y: np.ndarray, method: str = "auto") -> str:
    """Pick isotonic where there are enough events to support it, else Platt."""
    if method != "auto":
        return method
    y = np.asarray(y)
    if y.ndim > 1 or len(np.unique(y)) > 2:
        counts = pd.Series(y).value_counts()
        events = int(counts.min())
    else:
        events = int((y == 1).sum())
    return "isotonic" if events >= ISOTONIC_MIN_EVENTS else "sigmoid"


def calibrate(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    method: str = "auto",
) -> tuple[Any, str]:
    """
    Wrap a *fitted* model in a calibrator fitted on held-out validation data.

    Returns the calibrated estimator and the method actually used.
    """
    chosen = choose_method(y_valid, method)

    if _HAS_FROZEN:
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=chosen)
    else:  # pragma: no cover - older scikit-learn
        calibrator = CalibratedClassifierCV(model, method=chosen, cv="prefit")

    calibrator.fit(X_valid, y_valid)
    return calibrator, chosen


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Predicted versus observed event rate, by probability decile.

    This is the table a reviewer actually reads to decide whether a 4%
    prediction means 4%; the Brier score compresses it to one number.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frame["bin"] = pd.cut(frame["p"], bins=edges, include_lowest=True)

    grouped = frame.groupby("bin", observed=True).agg(
        n=("y", "size"),
        mean_predicted=("p", "mean"),
        observed_rate=("y", "mean"),
    )
    grouped["gap"] = grouped["mean_predicted"] - grouped["observed_rate"]
    return grouped.reset_index()
