"""
Why a record was flagged.

Two attribution paths, because the two detectors answer different questions.

**Supervised drivers** use LightGBM's native per-row contributions
(``pred_contrib=True``). These are exact Shapley values for a tree ensemble,
computed by TreeSHAP inside the booster in a single pass -- no ``shap``
dependency, no background dataset to sample, and no quadratic blow-up. This is
the path used whenever labels exist, because "the model raised this record
because of X" is only meaningful relative to a model that was trained on
outcomes.

**Unsupervised drivers** use robust per-feature deviation: how many median
absolute deviations the record sits from the population median, on the same
features the Isolation Forest saw. Running TreeSHAP against an Isolation Forest
is possible but wrong-headed here -- the forest's output is an isolation depth,
not a probability, so an exact attribution of it is an exact attribution of
something a reviewer cannot interpret. A robust z-score answers the question
the reviewer is actually asking ("what about this record is unusual?") in a
unit they already understand.

Both paths produce the same output shape: an ordered, human-readable string of
the top contributing features with their values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this, a deviation is not worth a reviewer's attention.
MIN_ROBUST_Z = 2.0

# Aggregate scores are excluded from the driver string. They are the model's
# strongest features and would fill every slot, but "rule score = 1.00" tells a
# reviewer nothing that the triggered-rules column has not already said. The
# drivers should name the *specific* evidence.
AGGREGATE_FEATURES = frozenset({"rule_score", "rule_score_row_level", "dq_score"})


def _pretty(name: str) -> str:
    for prefix in ("rule__json__", "rule__", "date__date__", "date__", "seq__"):
        if name.startswith(prefix):
            return name[len(prefix) :].replace("_", " ")
    return name.replace("_", " ")


def _format_value(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}" if abs(value) >= 0.01 else f"{value:.3g}"
    return str(value)


# --------------------------------------------------------------------------
# Supervised: exact tree contributions
# --------------------------------------------------------------------------
def supervised_contributions(head, frame: pd.DataFrame, dtypes: dict | None = None) -> pd.DataFrame | None:
    """
    Per-row feature contributions to the predicted log-odds.

    Returns a frame aligned to ``frame.index`` with one column per feature, or
    ``None`` where the backend cannot produce them.
    """
    if head.backend != "lightgbm":
        return None

    X = head.prepare(frame, dtypes)
    raw = head.estimator.booster_.predict(X, pred_contrib=True)
    raw = np.asarray(raw)

    n_features = len(head.feature_names)
    if raw.shape[1] == n_features + 1:
        contributions = raw[:, :n_features]
    else:
        # Multiclass returns (n_features + 1) columns per class; collapse to the
        # magnitude of the contribution across classes, which is what "this
        # feature mattered here" means when the classes are mutually exclusive.
        blocks = raw.reshape(len(raw), -1, n_features + 1)
        contributions = np.abs(blocks[:, :, :n_features]).max(axis=1)

    return pd.DataFrame(contributions, columns=head.feature_names, index=frame.index)


def top_drivers_from_contributions(
    contributions: pd.DataFrame, frame: pd.DataFrame, top_k: int = 3
) -> pd.Series:
    """The ``top_k`` features pushing each record's score up, with their values."""
    specific = [c for c in contributions.columns if c not in AGGREGATE_FEATURES]
    contributions = contributions[specific] if specific else contributions

    values = contributions.to_numpy()
    names = np.array(contributions.columns)
    order = np.argsort(-values, axis=1)[:, :top_k]

    out = []
    for row_position, (row_index, row_order) in enumerate(zip(contributions.index, order)):
        parts = []
        for column_position in row_order:
            if values[row_position, column_position] <= 0:
                continue
            name = names[column_position]
            raw_value = frame.at[row_index, name] if name in frame.columns else np.nan
            parts.append(f"{_pretty(name)}={_format_value(raw_value)}")
        out.append("; ".join(parts) if parts else "no positive contribution")
    return pd.Series(out, index=contributions.index)


# --------------------------------------------------------------------------
# Unsupervised: robust deviation
# --------------------------------------------------------------------------
class RobustDeviation:
    """
    Median / MAD reference computed on the training window.

    Robust statistics on purpose: the mean and standard deviation of a column
    that contains the anomalies are themselves dragged by those anomalies, so a
    plain z-score understates exactly the records it is meant to surface.
    """

    def __init__(self, frame: pd.DataFrame, columns: list[str]) -> None:
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        self.columns = list(columns)
        self.median = values.median()

        # 1.4826 makes the MAD a consistent estimator of sigma for normal data.
        mad = (values - self.median).abs().median() * 1.4826

        # A column where most rows share one value has a MAD of exactly zero --
        # ``balance_vs_scheduled`` is 1.0 for the great majority of records --
        # and dividing by it reports deviations in the tens of millions of MAD,
        # which is a formatting accident rather than a finding. The IQR is the
        # fallback scale; a column with no spread at all contributes no z.
        iqr = (values.quantile(0.75) - values.quantile(0.25)) / 1.349
        self.scale = mad.where(mad > 0, iqr).replace(0, np.nan)

    def z_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame[self.columns].apply(pd.to_numeric, errors="coerce")
        return ((values - self.median) / self.scale).abs()

    def top_drivers(self, frame: pd.DataFrame, top_k: int = 3) -> pd.Series:
        """The ``top_k`` most deviant features per record, above ``MIN_ROBUST_Z``."""
        z = self.z_scores(frame).fillna(0.0)
        z = z[[c for c in z.columns if c not in AGGREGATE_FEATURES]]
        values = z.to_numpy()
        names = np.array(z.columns)
        order = np.argsort(-values, axis=1)[:, :top_k]

        out = []
        for row_position, (row_index, row_order) in enumerate(zip(z.index, order)):
            parts = []
            for column_position in row_order:
                score = values[row_position, column_position]
                if score < MIN_ROBUST_Z:
                    continue
                name = names[column_position]
                raw_value = frame.at[row_index, name] if name in frame.columns else np.nan
                # Past ~50 robust sigma the exact number carries no extra
                # information for a reviewer, and a long one crowds the cell.
                shown = f"{score:.0f}" if score < 50 else ">50"
                parts.append(f"{_pretty(name)}={_format_value(raw_value)} ({shown} sigma)")
            out.append("; ".join(parts) if parts else "no feature beyond 2 MAD")
        return pd.Series(out, index=z.index)


# --------------------------------------------------------------------------
# Global view
# --------------------------------------------------------------------------
def classify_layer(feature: str, sequence_context: set[str] | None = None) -> str:
    """
    Which layer a feature belongs to.

    Prefix matching alone gets this wrong in both directions: ``rule_score`` is
    a deterministic aggregate with no ``rule__`` prefix, and the sequence
    *context* features (``months_after_absorbing``, ``transition_rarity``) are
    derived from adjacent months without carrying the ``seq__`` prefix of the
    detectors. Mislabelling them makes the chart claim the model leans on
    learned record state when it is leaning on rules.
    """
    if feature.startswith("rule__") or feature in {"rule_score", "rule_score_row_level"}:
        return "rule"
    if feature.startswith("date__"):
        return "date"
    if feature.startswith("seq__"):
        return "sequence detector"
    if sequence_context and feature in sequence_context:
        return "sequence context"
    return "record state"


def global_importance(
    head, contributions: pd.DataFrame | None, sequence_context: set[str] | None = None
) -> pd.DataFrame:
    """
    Mean absolute contribution per feature -- what drives the model overall.

    Falls back to the backend's own importances where per-row contributions are
    unavailable.
    """
    if contributions is not None and not contributions.empty:
        importance = contributions.abs().mean().sort_values(ascending=False)
        frame = importance.rename("mean_abs_contribution").rename_axis("feature").reset_index()
    else:
        values = getattr(head.estimator, "feature_importances_", None)
        if values is None:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {"feature": head.feature_names, "mean_abs_contribution": values}
        ).sort_values("mean_abs_contribution", ascending=False)

    total = frame["mean_abs_contribution"].sum()
    frame["share"] = frame["mean_abs_contribution"] / total if total else np.nan
    frame["layer"] = [classify_layer(f, sequence_context) for f in frame["feature"]]
    return frame.reset_index(drop=True)
