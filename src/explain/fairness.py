"""
Disparity screening across borrower groups.

What this is, and what it is not
--------------------------------
This is a **disparity screen**, not a legal fairness test, and the distinction
is not pedantry -- reporting it as the latter would be the single most
misleading thing in this repository.

* The panel contains **no protected attribute**. There is no race, sex, age or
  national origin field, so no disparate-treatment or disparate-impact analysis
  in the legal sense is possible from this data. What is available is
  geography (``state``) and servicer, which are at best coarse proxies, and
  credit characteristics, which are not proxies at all.
* **Credit-score band disparity is expected and desirable.** A model that
  flagged sub-620 and 800+ borrowers at the same rate would be broken. Reporting
  a "disparity" there and treating it as a finding confuses a risk model doing
  its job with a risk model doing harm. The credit-band table is included as a
  *sanity check on monotonicity*, not as a fairness result.
* **Geography is where a real question lives.** State is not a legitimate risk
  factor in the way credit score is, and in US mortgage lending it correlates
  with protected classes. A state-level disparity in *error rates* -- being
  wrongly flagged more often in one state than another, conditional on actually
  performing -- is the finding worth escalating, and it is what this module
  ranks.

The metric that matters
-----------------------
Selection-rate parity is the wrong lens for a risk model: groups genuinely
differ in risk, so equal flag rates would be the anomaly. The comparison here
is **error-rate parity conditional on outcome** -- among borrowers who did not
default, is one group flagged more often than another? That question has no
legitimate risk-based answer, which is what makes a gap in it actionable.

The 0.80 ratio floor is borrowed from the US "four-fifths rule" as a screening
threshold. It is a trigger for a human to look, never a verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy import stats

from .. import config
from .errors import error_rates_by_segment

# Numerator and denominator behind each rate, so a gap can be significance
# tested rather than eyeballed.
RATE_COUNTS = {
    "false_positive_rate": ("false_positives", "actual_negatives"),
    "false_negative_rate": ("false_negatives", "actual_positives"),
    "selection_rate": ("flagged", "records"),
}

# Below this many events in a group the normal approximation behind the
# two-proportion test is not trustworthy, and neither is the rate.
MIN_EVENTS_FOR_TEST = 10

# A gap has to clear this to be called a finding rather than sampling noise.
SIGNIFICANCE_LEVEL = 0.01

# Segments where a gap is a *finding* rather than the model working correctly.
# Credit and collateral bands are legitimate risk factors; geography and
# servicer are not, and are where a disparity needs an explanation.
#
# `vintage_year` belongs here for a reason worth spelling out: in a single
# reporting window it is almost pure **loan age**. In the 2023 test window mean
# age runs from 54 months for the 2018 vintage to 3.5 months for 2023, and
# seasoning is a legitimate driver of default hazard. The 2018 cohort's high
# flag rate is also survivorship -- only 232 of its loans are still on the book
# and 41% of them actually default. Escalating that as a disparity would be
# reporting the model getting it right.
RISK_FACTOR_SEGMENTS = {"credit_score_band", "ltv_band", "dti_band", "vintage_year"}

# Above this overall flag rate, group disparity metrics stop being
# interpretable: a model that flags half the book produces enormous
# false-positive-rate gaps between any two groups, and the gap describes the
# threshold rather than the treatment. Findings are still reported, but not
# escalated, and the reason is carried in the table.
MAX_INTERPRETABLE_SELECTION_RATE = 0.40


def segment_kind(segment: str) -> str:
    return "risk factor (gap expected)" if segment in RISK_FACTOR_SEGMENTS else "screen for disparity"


def _two_proportion_p_value(x1: int, n1: int, x2: int, n2: int) -> float:
    """
    Two-sided p-value for the difference between two rates.

    A ratio alone cannot distinguish a real gap from a small sample: sixteen
    false positives in California against four in New York is a ratio of 0.36
    and roughly nothing at all. Pooled-variance two-proportion z-test.
    """
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    pooled = (x1 + x2) / (n1 + n2)
    if pooled in (0.0, 1.0):
        return 1.0
    standard_error = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if standard_error == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / standard_error
    return float(2 * stats.norm.sf(abs(z)))


def disparity_summary(
    table: pd.DataFrame,
    segment: str,
    metric: str = "false_positive_rate",
    floor: float = config.DISPARITY_RATIO_FLOOR,
    overall_selection_rate: float | None = None,
) -> pd.DataFrame:
    """
    Best-to-worst ratio on one metric, with a significance test attached.

    ``ratio = min(metric) / max(metric)``. A low ratio means one group carries
    much more of the error -- but only if the gap survives a two-proportion
    test. Escalation requires **three** conditions, not one:

    1. the ratio is below the screening floor,
    2. the gap is significant at 1% and both groups carry enough events for the
       test to mean anything,
    3. the segment is not a legitimate risk factor.

    Without (2) this screen escalated eight of fifteen segment-metric pairs on
    a small sample, most of them on twenty events or fewer. A governance report
    that flags everything is one nobody reads.
    """
    if table.empty or metric not in table.columns:
        return pd.DataFrame()

    values = table[metric].dropna()
    if values.empty or values.max() == 0:
        return pd.DataFrame()

    level = "group" if "group" in table.columns else segment
    worst = table.loc[values.idxmax()]
    best = table.loc[values.idxmin()]
    ratio = float(values.min() / values.max())
    is_risk_factor = segment in RISK_FACTOR_SEGMENTS

    numerator, denominator = RATE_COUNTS.get(metric, (None, None))
    p_value, testable = float("nan"), False
    if numerator in table.columns and denominator in table.columns:
        worst_events, worst_n = int(worst[numerator]), int(worst[denominator])
        best_events, best_n = int(best[numerator]), int(best[denominator])
        testable = worst_events >= MIN_EVENTS_FOR_TEST and best_n > 0 and worst_n > 0
        p_value = _two_proportion_p_value(worst_events, worst_n, best_events, best_n)

    significant = bool(testable and p_value < SIGNIFICANCE_LEVEL)
    interpretable = (
        overall_selection_rate is None
        or overall_selection_rate <= MAX_INTERPRETABLE_SELECTION_RATE
    )

    return pd.DataFrame(
        [
            {
                "segment": segment,
                "kind": segment_kind(segment),
                "metric": metric,
                "groups_compared": int(len(values)),
                "worst_group": worst[level],
                "worst_value": float(worst[metric]),
                "worst_group_records": int(worst["records"]),
                "worst_group_events": int(worst[numerator]) if numerator in table.columns else None,
                "best_group": best[level],
                "best_value": float(best[metric]),
                "ratio_best_to_worst": ratio,
                "below_floor": bool(ratio < floor),
                "p_value": p_value,
                "enough_events_to_test": testable,
                "significant": significant,
                "interpretable": interpretable,
                # Four conditions. A gap on a legitimate risk factor is never
                # escalated however large, and neither is one produced by a
                # model that flags half the book -- both would bury the findings
                # that matter.
                "escalate": bool(
                    ratio < floor and significant and interpretable and not is_risk_factor
                ),
            }
        ]
    )


def audit(
    frame: pd.DataFrame,
    outcome: pd.Series,
    segments: list[str] | None = None,
    metrics: tuple[str, ...] = ("false_positive_rate", "false_negative_rate", "selection_rate"),
    min_group: int = config.MIN_GROUP_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the screen across every segment and metric.

    Returns ``(group_table, disparity_table)``: the per-group rates, and the
    one-row-per-(segment, metric) summary that a reviewer reads first.
    """
    segments = segments or config.EXPLAIN_SEGMENTS

    # A model flagging most of the book makes every group comparison extreme.
    overall_selection_rate = float(
        outcome.isin(["true positive", "false positive"]).mean()
    )

    group_frames: list[pd.DataFrame] = []
    disparity_frames: list[pd.DataFrame] = []

    for segment in segments:
        table = error_rates_by_segment(frame, outcome, segment, min_group=min_group)
        if table.empty:
            continue
        table = table.copy()
        table.insert(1, "kind", segment_kind(segment))
        group_frames.append(table)

        raw = table
        for metric in metrics:
            disparity_frames.append(
                disparity_summary(
                    raw, segment, metric, overall_selection_rate=overall_selection_rate
                )
            )

    groups = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    disparity = (
        pd.concat([d for d in disparity_frames if not d.empty], ignore_index=True)
        if disparity_frames
        else pd.DataFrame()
    )
    if not disparity.empty:
        disparity = disparity.sort_values(
            ["escalate", "significant", "below_floor", "ratio_best_to_worst"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    if not disparity.empty:
        disparity["overall_selection_rate"] = overall_selection_rate
    return groups, disparity


def monotonicity_check(group_table: pd.DataFrame, segment: str, order: list[str]) -> dict:
    """
    Does the flag rate fall as credit quality rises?

    Not a fairness question -- a correctness one. If the model flags 740-799
    borrowers more often than 620-659 borrowers, something is inverted, and no
    amount of SHAP commentary is worth as much as noticing it.
    """
    subset = group_table[group_table["segment"] == segment]
    if subset.empty:
        return {"segment": segment, "checked": False}

    ordered = subset.set_index("group").reindex([g for g in order if g in set(subset["group"])])
    rates = ordered["selection_rate"].dropna()
    if len(rates) < 3:
        return {"segment": segment, "checked": False}

    return {
        "segment": segment,
        "checked": True,
        "monotone_decreasing": bool(rates.is_monotonic_decreasing),
        "first_group": rates.index[0],
        "first_rate": float(rates.iloc[0]),
        "last_group": rates.index[-1],
        "last_rate": float(rates.iloc[-1]),
    }
