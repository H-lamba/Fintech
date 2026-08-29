"""
SHAP values for the Phase 3 models.

What is being explained, and what is not
---------------------------------------
TreeExplainer decomposes the **booster's log-odds output**, not the calibrated
probability that the pipeline actually deploys. The calibrator is a monotone
transform fitted on top, so it cannot change the *ranking* of feature
contributions -- an attribution that says credit score dominates is true of the
deployed model too -- but the additive decomposition sums to the base model's
log-odds, not to the calibrated probability. Stating that plainly matters more
than it might seem: a model card that claims SHAP values explain the deployed
probability is claiming something the arithmetic does not support.

Everything downstream that is about *decisions* -- error analysis, reliability,
disparity -- runs on the calibrated probability at the tuned threshold, because
that is what a borrower would actually experience.

On sampling
-----------
``feature_perturbation="tree_path_dependent"`` needs **no background dataset**.
It walks the trees using the training-time cover counts already stored in the
model, so there is nothing to sample and nothing to hold in memory beyond the
output matrix. The alternative, ``interventional``, does need a background set
and is where the usual out-of-memory advice comes from; it is not used here.

Sampling is still applied, for two honest reasons that are not memory:

* **Scale headroom.** The values matrix is ``n_rows x n_features`` floats, and
  ``n_rows x n_features x n_classes`` for a multiclass head. At this data size
  that is 27 MB and three seconds; on a pack ten times larger it is not.
* **Legibility.** A beeswarm of 58,000 points is a solid block of ink. The plot
  sample is smaller again than the analysis sample for that reason alone.

Sampling is **stratified on the outcome**, because the positives are 8-11% of
the panel and a uniform sample of a rare class explains mostly negatives.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..models import estimators


@dataclass
class ShapResult:
    """SHAP values for one model, with the rows they were computed on."""

    target: str
    label: str
    values: np.ndarray  # (n_rows, n_features)
    base_value: float
    X: pd.DataFrame  # the encoded matrix the model saw
    frame: pd.DataFrame  # the source rows, with ids and segments
    feature_names: list[str]
    # Positional indices into the frame that was passed in. Carried explicitly
    # so callers can line up probabilities and outcomes with the sampled rows
    # without re-deriving the sample or relying on index alignment.
    positions: np.ndarray = None

    def __len__(self) -> int:
        return len(self.values)

    def predicted_log_odds(self) -> np.ndarray:
        """Base value plus the row's contributions -- the model's own output."""
        return self.base_value + self.values.sum(axis=1)


def stratified_sample(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    n_rows: int,
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """
    Positional indices of a sample that preserves the outcome balance.

    A uniform sample of a 9%-positive panel spends 91% of its budget explaining
    records nobody asks about. Stratifying keeps the rare class represented at
    its true rate, so the beeswarm shows the model's behaviour on events rather
    than on the quiet majority.
    """
    if n_rows >= len(frame):
        return np.arange(len(frame))

    rng = np.random.default_rng(seed)
    outcome = np.asarray(outcome)
    indices: list[np.ndarray] = []

    for value in np.unique(outcome):
        positions = np.flatnonzero(outcome == value)
        share = len(positions) / len(outcome)
        take = max(1, int(round(n_rows * share)))
        take = min(take, len(positions))
        indices.append(rng.choice(positions, size=take, replace=False))

    return np.sort(np.concatenate(indices))


def explain_model(
    model,
    frame: pd.DataFrame,
    target: str,
    label: str,
    outcome: np.ndarray | None = None,
    n_rows: int = config.SHAP_SAMPLE_ROWS,
    seed: int = config.RANDOM_SEED,
) -> ShapResult:
    """
    Compute SHAP values for one Phase 3 model over a stratified sample.

    Only LightGBM and XGBoost backends are explainable this way; a logistic
    regression baseline returns ``None`` from the caller's perspective rather
    than a silently different kind of attribution.
    """
    import shap

    if getattr(model, "backend", None) not in {"lightgbm", "xgboost"}:
        raise TypeError(
            f"TreeExplainer needs a tree backend; {target!r} was fitted with "
            f"{getattr(model, 'backend', 'unknown')!r}."
        )

    positions = (
        stratified_sample(frame, outcome, n_rows, seed)
        if outcome is not None
        else np.arange(min(n_rows, len(frame)))
    )
    sampled = frame.iloc[positions]
    X = estimators.prepare_matrix(sampled, model.numeric, model.categorical, model.harmoniser)

    with warnings.catch_warnings():
        # shap warns that LightGBM binary output shape changed; both shapes are
        # handled below, so the warning is noise in a pipeline log.
        warnings.simplefilter("ignore", UserWarning)
        explainer = shap.TreeExplainer(model.estimator, feature_perturbation="tree_path_dependent")
        values = np.asarray(explainer.shap_values(X))

    base = explainer.expected_value
    if values.ndim == 3:  # (rows, features, classes) -- keep the positive class
        values = values[:, :, 1]
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])

    return ShapResult(
        target=target,
        label=label,
        values=values,
        base_value=float(base),
        X=X,
        frame=sampled,
        feature_names=list(X.columns),
        positions=positions,
    )


def verify_against_booster(result: ShapResult, model, atol: float = 1e-6) -> dict:
    """
    Cross-check the SHAP values against LightGBM's own ``pred_contrib``.

    Both run TreeSHAP, so they must agree exactly. Where they do not, one of the
    two is being handed a different matrix than the other -- a stale category
    encoding or a reordered column -- and every attribution in the report is
    describing a model that was never scored. Cheap to check, and it is the
    kind of error that produces a plausible-looking chart.
    """
    if getattr(model, "backend", None) != "lightgbm":
        return {"checked": False, "reason": f"backend {getattr(model, 'backend', None)!r}"}

    contributions = np.asarray(model.estimator.booster_.predict(result.X, pred_contrib=True))
    contributions = contributions[:, : len(result.feature_names)]
    max_difference = float(np.abs(result.values - contributions).max())

    return {
        "checked": True,
        "agrees": bool(max_difference <= atol),
        "max_abs_difference": max_difference,
    }


# --------------------------------------------------------------------------
# Global view
# --------------------------------------------------------------------------
def global_importance(result: ShapResult, top_n: int | None = None) -> pd.DataFrame:
    """
    Mean absolute SHAP value per feature -- the standard global ranking.

    ``mean_signed`` is carried alongside because the two answer different
    questions: magnitude says how much a feature matters, sign says which way
    it usually pushes. A feature can be important and directionless.
    """
    frame = pd.DataFrame(
        {
            "feature": result.feature_names,
            "mean_abs_shap": np.abs(result.values).mean(axis=0),
            "mean_signed_shap": result.values.mean(axis=0),
            "max_abs_shap": np.abs(result.values).max(axis=0),
        }
    )
    total = frame["mean_abs_shap"].sum()
    frame["share"] = frame["mean_abs_shap"] / total if total else np.nan
    frame.insert(0, "model", result.label)
    frame = frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return frame.head(top_n) if top_n else frame


# --------------------------------------------------------------------------
# Local view
# --------------------------------------------------------------------------
def local_explanation(result: ShapResult, position: int, top_n: int = 12) -> pd.DataFrame:
    """
    One loan's contributions, largest first, with the feature's own value.

    Returns a frame ready to render as a waterfall: ``base_value`` plus the
    ``shap_value`` column sums to the model's log-odds for that row.
    """
    row_values = result.values[position]
    order = np.argsort(-np.abs(row_values))[:top_n]

    frame = pd.DataFrame(
        {
            "feature": [result.feature_names[i] for i in order],
            "feature_value": [result.X.iloc[position, i] for i in order],
            "shap_value": row_values[order],
        }
    )
    frame["direction"] = np.where(frame["shap_value"] > 0, "raises risk", "lowers risk")

    other = float(row_values.sum() - row_values[order].sum())
    frame.attrs.update(
        {
            "base_value": result.base_value,
            "other_features": other,
            "log_odds": float(result.base_value + row_values.sum()),
            "model": result.label,
        }
    )
    return frame


def pick_demo_loans(
    result: ShapResult,
    probability: np.ndarray,
    outcome: np.ndarray,
    id_col: str = config.ID_COL,
) -> pd.DataFrame:
    """
    Four loans worth showing, chosen so the demo is not a highlight reel.

    A confident hit, a confident miss, a false positive and a borderline case.
    Picking only the confident hit produces a demo that proves the model works
    on the records where nothing was ever in doubt.
    """
    probability = np.asarray(probability)
    outcome = np.asarray(outcome).astype(int)

    picks = {
        "confident true positive": np.flatnonzero((outcome == 1) & (probability > 0.8)),
        "confident false positive": np.flatnonzero((outcome == 0) & (probability > 0.8)),
        "missed event (false negative)": np.flatnonzero((outcome == 1) & (probability < 0.15)),
        "borderline": np.flatnonzero(np.abs(probability - 0.5) < 0.03),
    }

    rows = []
    for description, candidates in picks.items():
        if len(candidates) == 0:
            continue
        # The most extreme example of each case, so the waterfall is legible.
        position = int(candidates[np.argmax(np.abs(probability[candidates] - 0.5))])
        rows.append(
            {
                "case": description,
                "position": position,
                id_col: result.frame.iloc[position][id_col] if id_col in result.frame.columns else None,
                config.TIME_COL: result.frame.iloc[position].get(config.TIME_COL),
                "predicted_probability": float(probability[position]),
                "actual_outcome": int(outcome[position]),
            }
        )
    return pd.DataFrame(rows)
