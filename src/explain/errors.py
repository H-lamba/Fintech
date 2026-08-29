"""
Where the model is wrong, and on what kind of loan.

Everything here runs on the **calibrated probability at the tuned threshold**,
not on the booster's raw score. SHAP explains the base model; error analysis
has to describe the deployed one, because a false positive is a decision, and
the decision is made by the calibrated probability crossing a threshold that
Task 2 tuned on the validation window.

Two questions, and they are different:

* *Is the model wrong in the right proportions?* -> reliability. A model that
  says 20% and is right 20% of the time is well calibrated even if it never
  reaches high confidence.
* *Is the model wrong disproportionately somewhere?* -> error rates by segment.
  A uniform 8% false-positive rate is a cost of doing business; the same rate
  concentrated in one vintage is a defect with a name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

OUTCOME_ORDER = ["true positive", "false positive", "false negative", "true negative"]


def classify_predictions(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> pd.Series:
    """Label each record TP / FP / FN / TN at the deployed threshold."""
    y_true = np.asarray(y_true).astype(int)
    predicted = (np.asarray(probability) >= threshold).astype(int)

    labels = np.where(
        (y_true == 1) & (predicted == 1), "true positive",
        np.where(
            (y_true == 0) & (predicted == 1), "false positive",
            np.where((y_true == 1) & (predicted == 0), "false negative", "true negative"),
        ),
    )
    return pd.Series(labels, name="outcome")


def confusion_summary(outcome: pd.Series) -> pd.DataFrame:
    """Counts and rates for the four cells, in a fixed order."""
    counts = outcome.value_counts().reindex(OUTCOME_ORDER, fill_value=0)
    frame = counts.rename("records").rename_axis("outcome").reset_index()
    frame["share"] = frame["records"] / max(len(outcome), 1)
    return frame


def _rates(group: pd.DataFrame) -> pd.Series:
    """Precision, recall and the two error rates for one group of records."""
    outcome = group["outcome"]
    tp = int((outcome == "true positive").sum())
    fp = int((outcome == "false positive").sum())
    fn = int((outcome == "false negative").sum())
    tn = int((outcome == "true negative").sum())

    actual_positive = tp + fn
    actual_negative = fp + tn
    flagged = tp + fp

    return pd.Series(
        {
            "records": len(group),
            "actual_positives": actual_positive,
            "actual_negatives": actual_negative,
            "flagged": flagged,
            # Raw counts are carried alongside the rates so a disparity test
            # downstream can ask whether a gap is distinguishable from noise.
            # A rate without its denominator cannot be tested, and an untested
            # rate gap on twenty events is how a governance report cries wolf.
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "selection_rate": flagged / len(group) if len(group) else np.nan,
            "precision": tp / flagged if flagged else np.nan,
            "recall": tp / actual_positive if actual_positive else np.nan,
            # FPR and FNR are the pair a fairness review reads: one is the cost
            # borne by borrowers who did nothing wrong, the other the risk the
            # lender absorbs.
            "false_positive_rate": fp / actual_negative if actual_negative else np.nan,
            "false_negative_rate": fn / actual_positive if actual_positive else np.nan,
        }
    )


def error_rates_by_segment(
    frame: pd.DataFrame,
    outcome: pd.Series,
    segment: str,
    min_group: int = config.MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """
    Error rates per segment level, worst false-positive rate first.

    Groups below ``min_group`` are dropped rather than reported: a false
    positive rate computed on nine loans is noise with a decimal point, and
    putting it in a governance table invites someone to act on it.
    """
    if segment not in frame.columns:
        return pd.DataFrame()

    working = frame[[segment]].copy()
    working["outcome"] = outcome.to_numpy()

    table = (
        working.groupby(segment, dropna=True, observed=True)
        .apply(_rates, include_groups=False)
        .reset_index()
    )
    table = table[table["records"] >= min_group]

    # Always ``segment`` (the column's name) and ``group`` (the level within
    # it). Returning the segment's own name as the column made the frame's
    # schema depend on the data -- and collided outright when a segment column
    # was itself called "segment".
    table = table.rename(columns={segment: "group"})
    table.insert(0, "segment", segment)
    return table.sort_values("false_positive_rate", ascending=False).reset_index(drop=True)


def characterise_errors(
    frame: pd.DataFrame,
    outcome: pd.Series,
    features: list[str],
    top_n: int = 12,
) -> pd.DataFrame:
    """
    How false positives differ from the records the model got right.

    For each feature, the mean among false positives against the mean among
    true negatives, standardised by the true-negative spread. A large
    standardised gap says "the model is flagging loans that look like *this*",
    which is the actionable form of an error analysis -- more so than a list of
    loan ids.
    """
    working = frame.copy()
    working["outcome"] = outcome.to_numpy()

    false_positives = working[working["outcome"] == "false positive"]
    true_negatives = working[working["outcome"] == "true negative"]
    false_negatives = working[working["outcome"] == "false negative"]
    true_positives = working[working["outcome"] == "true positive"]

    if false_positives.empty or true_negatives.empty:
        return pd.DataFrame()

    rows = []
    for feature in features:
        if feature not in working.columns:
            continue
        values = pd.to_numeric(working[feature], errors="coerce")
        if values.notna().sum() == 0:
            continue

        spread = values.std(ddof=0)
        if not np.isfinite(spread) or spread == 0:
            continue

        fp_mean = pd.to_numeric(false_positives[feature], errors="coerce").mean()
        tn_mean = pd.to_numeric(true_negatives[feature], errors="coerce").mean()
        fn_mean = pd.to_numeric(false_negatives[feature], errors="coerce").mean()
        tp_mean = pd.to_numeric(true_positives[feature], errors="coerce").mean()

        rows.append(
            {
                "feature": feature,
                "false_positive_mean": fp_mean,
                "true_negative_mean": tn_mean,
                "standardised_gap_fp_vs_tn": (fp_mean - tn_mean) / spread,
                "false_negative_mean": fn_mean,
                "true_positive_mean": tp_mean,
                "standardised_gap_fn_vs_tp": (fn_mean - tp_mean) / spread,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return (
        table.reindex(table["standardised_gap_fp_vs_tn"].abs().sort_values(ascending=False).index)
        .head(top_n)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------
def reliability_table(
    y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Predicted against observed event rate, by equal-width probability bin.

    Equal-width rather than equal-count on purpose: the question a reliability
    diagram answers is "when the model says 70%, is it right 70% of the time",
    and that question is asked about a probability level, not about a quantile
    of the model's own output.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(probability)})
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frame["bin"] = pd.cut(frame["p"], bins=edges, include_lowest=True)

    table = (
        frame.groupby("bin", observed=True)
        .agg(records=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    table["bin"] = table["bin"].astype(str)
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """
    Weighted mean absolute gap between predicted and observed -- the ECE.

    Weighted by bin population, so a wild miss in a bin holding four records
    does not outweigh a small bias across the bulk of the book.
    """
    if table.empty:
        return float("nan")
    weights = table["records"] / table["records"].sum()
    return float((weights * table["gap"].abs()).sum())


def confidence_profile(probability: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    How much of the book the model is actually confident about.

    A model that never leaves 0.05-0.15 is technically calibrated and
    operationally useless: nothing is ever decided. This reports the share of
    records in confident and uncertain bands, where "uncertain" is defined
    around the deployed threshold rather than around 0.5.
    """
    probability = np.asarray(probability)
    near = np.abs(probability - threshold) < 0.05

    return pd.DataFrame(
        [
            {"band": "confident, flagged", "share": float((probability >= threshold + 0.2).mean())},
            {"band": "uncertain (within 0.05 of threshold)", "share": float(near.mean())},
            {"band": "confident, cleared", "share": float((probability <= threshold - 0.2).mean())},
            {"band": "max predicted probability", "share": float(probability.max())},
        ]
    )
