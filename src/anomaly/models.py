"""
The learned half of the hybrid: one unsupervised detector, two supervised heads.

**Isolation Forest** (unsupervised). Scores every record on how easily it can
be isolated in the continuous feature space, with no labels at all. This is the
layer that would still work on a book where nobody has ever written down what
an exception looks like -- which is most books. It is deliberately *not* shown
the rule indicators; see ``AnomalyFeatures.unsupervised_columns``.

**Exception probability** (supervised, binary). Predicts
``exception_required`` from the rule indicators, the unsupervised score and the
record's state. Its job on this pack is precision, not recall: the
deterministic layer already reaches 99.8% recall, at 17.9% precision, which is
a queue of 39,000 records to find 7,000 exceptions. Ranking that queue is worth
more than finding anything new in it.

**Exception type** (supervised, multiclass). Five-way over
``{None, Balance Discrepancy, Impossible State Transition, Time Travel, Zombie
Loan}``, so the reviewer gets a suggested category rather than only a score.

The hybrid score
----------------
``hybrid = 1 - (1 - rule_score) * (1 - ml_score)`` -- a noisy-OR, the same
combination used inside ``rule_score`` itself. It has the property the domain
needs: a fired high-severity rule sets a floor that no amount of ML calm can
talk down, and the unsupervised score can only ever add suspicion on top. A
weighted average would let a confident model argue away a hard violation,
which is not a trade a servicer would accept.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config


# --------------------------------------------------------------------------
# Unsupervised
# --------------------------------------------------------------------------
@dataclass
class IsolationForestDetector:
    """Fitted Isolation Forest plus the columns and scaling it was fitted on."""

    pipeline: Pipeline
    columns: list[str]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        columns: list[str],
        contamination: float | str = "auto",
        n_estimators: int = 300,
        max_samples: int | str = 8192,
        seed: int = config.RANDOM_SEED,
    ) -> "IsolationForestDetector":
        """
        Fit on the *training window only*.

        ``contamination="auto"`` by default: the contamination parameter only
        moves sklearn's internal decision threshold, and this pipeline never
        uses that threshold -- it uses the continuous score and sets its own
        operating point downstream. Passing a guessed contamination rate would
        bake an assumption about how dirty the book is into a model whose whole
        purpose is to find that out.
        """
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "forest",
                    IsolationForest(
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        contamination=contamination,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(frame[columns])
        return cls(pipeline=pipeline, columns=list(columns))

    def raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        """``-score_samples``: higher means more anomalous."""
        return -self.pipeline.score_samples(frame[self.columns])

    def score(self, frame: pd.DataFrame, reference: np.ndarray | None = None) -> np.ndarray:
        """
        Anomaly score in [0, 1], as a percentile of the reference distribution.

        Rank-normalised rather than min-max scaled: the raw isolation score has
        no meaningful units and a single extreme record would otherwise
        compress everything else into the bottom of the range. ``reference``
        defaults to the scored frame itself; pass the training scores to keep
        the scale fixed across scoring runs.
        """
        raw = self.raw_score(frame)
        baseline = raw if reference is None else np.asarray(reference)
        return np.searchsorted(np.sort(baseline), raw, side="right") / max(len(baseline), 1)


def hybrid_score(rule_score: np.ndarray | pd.Series, ml_score: np.ndarray | pd.Series) -> np.ndarray:
    """
    Noisy-OR of the deterministic and learned scores.

    Rules set a floor; the model can only add. See the module docstring for why
    that asymmetry is the right one for a reviewer queue.
    """
    rules = np.clip(np.asarray(rule_score, dtype=float), 0.0, 1.0)
    ml = np.clip(np.asarray(ml_score, dtype=float), 0.0, 1.0)
    return 1.0 - (1.0 - rules) * (1.0 - ml)


# --------------------------------------------------------------------------
# Supervised heads
# --------------------------------------------------------------------------
def _lightgbm_available() -> bool:
    if importlib.util.find_spec("lightgbm") is None:
        return False
    try:
        importlib.import_module("lightgbm")
    except Exception:  # a missing OpenMP runtime, most commonly
        return False
    return True


@dataclass
class SupervisedHead:
    """A fitted classifier plus everything needed to reproduce its predictions."""

    estimator: Any
    backend: str
    numeric: list[str]
    categorical: list[str]
    classes: list = None
    threshold: float = 0.5
    # Class label order for a multiclass head. Labels are integer-encoded
    # before fitting (see fit_supervised_head) so this maps back to names.
    class_order: list = None

    @property
    def feature_names(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    def prepare(self, frame: pd.DataFrame, dtypes: dict | None = None) -> pd.DataFrame:
        X = frame.loc[:, self.feature_names].copy()
        for column in self.numeric:
            X[column] = pd.to_numeric(X[column], errors="coerce").astype("float64")
        for column in self.categorical:
            X[column] = X[column].astype(dtypes[column]) if dtypes and column in dtypes else X[column].astype("category")
        return X

    def predict_proba(self, frame: pd.DataFrame, dtypes: dict | None = None) -> np.ndarray:
        """Probabilities, with multiclass columns reordered into ``class_order``."""
        proba = self.estimator.predict_proba(self.prepare(frame, dtypes))
        if not self.class_order:
            return proba
        out = np.zeros((proba.shape[0], len(self.class_order)))
        for position, encoded in enumerate(self.classes):
            index = int(encoded)
            if 0 <= index < len(self.class_order):
                out[:, index] = proba[:, position]
        return out


def category_dtypes(frame: pd.DataFrame, columns: list[str]) -> dict:
    """
    Freeze categorical levels on TRAIN so a level first seen in the scoring
    window cannot change the encoding. Unseen levels become NaN, which every
    backend here treats as missing.
    """
    return {
        column: pd.CategoricalDtype(categories=pd.Index(frame[column].dropna().unique()).sort_values())
        for column in columns
    }


def fit_supervised_head(
    train: pd.DataFrame,
    y_train: np.ndarray,
    valid: pd.DataFrame,
    y_valid: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    dtypes: dict,
    multiclass: bool = False,
    class_order: list | None = None,
    seed: int = config.RANDOM_SEED,
) -> SupervisedHead:
    """
    Fit the exception classifier, with early stopping on the later window.

    Class weighting rather than resampling, for the same reason Task 2 gave:
    the minority rows here carry sequence context (months after absorption, a
    status transition) that an interpolated synthetic row would not represent
    coherently.

    Multiclass labels are **integer-encoded against ``class_order``** before
    fitting. LightGBM's sklearn wrapper label-encodes the training labels but
    hands the holdout labels to the C++ layer untouched, so a string class
    fails there; encoding once here also fixes the probability column order
    for every backend.
    """
    head = SupervisedHead(
        estimator=None, backend="", numeric=numeric, categorical=categorical,
        class_order=list(class_order) if multiclass and class_order else None,
    )

    if head.class_order:
        index = {label: position for position, label in enumerate(head.class_order)}
        y_train = np.array([index[label] for label in y_train])
        y_valid = np.array([index[label] for label in y_valid])
    n_classes = len(head.class_order) if head.class_order else 2
    X_train = head.prepare(train, dtypes)
    X_valid = head.prepare(valid, dtypes)

    if _lightgbm_available():
        import lightgbm as lgb

        params = dict(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            cat_smooth=20,
            min_data_per_group=200,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        if multiclass:
            params.update(
                objective="multiclass", num_class=n_classes,
                class_weight="balanced", metric="multi_logloss",
            )
        else:
            positives = float((y_train == 1).sum())
            negatives = float((y_train == 0).sum())
            params.update(
                objective="binary",
                scale_pos_weight=(negatives / positives) if positives else 1.0,
                metric="average_precision",
            )
        model = lgb.LGBMClassifier(**params)
        callbacks = [lgb.early_stopping(60, first_metric_only=True, verbose=False), lgb.log_evaluation(0)]
        import inspect

        if "eval_X" in inspect.signature(model.fit).parameters:
            model.fit(X_train, y_train, eval_X=(X_valid,), eval_y=(y_valid,), callbacks=callbacks)
        else:  # pragma: no cover - lightgbm < 4.7
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=callbacks)
        head.estimator, head.backend = model, "lightgbm"
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features="from_dtype",
            early_stopping=False,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(X_train, y_train)
        head.estimator, head.backend = model, "hist"

    head.classes = list(model.classes_)
    return head


def best_iteration(head: SupervisedHead) -> int | None:
    if head.backend == "lightgbm":
        return getattr(head.estimator, "best_iteration_", None)
    return getattr(head.estimator, "n_iter_", None)
