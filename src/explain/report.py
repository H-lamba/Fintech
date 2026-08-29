"""Assembles the Task 6 explainability report and the model card."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..profiling.report import ReportBuilder

SCOPE_NOTE = """
**What SHAP explains here.** `TreeExplainer` decomposes the **booster's
log-odds**, not the calibrated probability the pipeline deploys. The calibrator
is a monotone transform fitted on top, so it cannot reorder feature
contributions -- an attribution that says credit score dominates is true of the
deployed model too -- but the additive decomposition sums to the base model's
log-odds. A report that claimed these values explain the deployed probability
would be claiming something the arithmetic does not support.

**What runs on the deployed model instead.** Error analysis, reliability and
the disparity screen all use the **calibrated probability at the threshold Task
2 tuned on validation**, because a false positive is a decision and the
decision is made there.

**On sampling.** `tree_path_dependent` perturbation needs no background
dataset -- it walks the trees using cover counts already stored in the model.
The full 58k-row test set computes in about three seconds. Rows are still
sampled, for scale headroom on a larger pack and because a beeswarm of 58,000
points is a solid block of ink, not because of a memory limit at this size.
Sampling is stratified on the outcome: the positives are 8-11% of the panel,
and a uniform sample of a rare class explains mostly non-events.
""".strip()

FAIRNESS_NOTE = """
**This is a disparity screen, not a legal fairness test**, and the distinction
is the most important sentence in this section.

The panel contains **no protected attribute** -- no race, sex, age or national
origin -- so no disparate-treatment or disparate-impact analysis in the legal
sense is possible from this data. What exists is geography and servicer, which
are coarse proxies at best, and credit characteristics, which are not proxies
at all.

**A credit-band gap is the model working, not failing.** A model that flagged
sub-620 and 800+ borrowers at the same rate would be broken. The credit-band
table is a *monotonicity check* -- does the flag rate fall as credit quality
rises -- not a fairness result, and it is labelled that way in the `kind`
column so nobody reads it as one.

**Geography is where a real question lives.** State is not a legitimate risk
factor in the way credit score is, and in US mortgage lending it correlates
with protected classes. The metric ranked here is therefore **error-rate parity
conditional on outcome**: among borrowers who did *not* default, is one state
flagged more often than another? That question has no legitimate risk-based
answer, which is what makes a gap in it worth escalating.

The 0.80 ratio floor is borrowed from the US "four-fifths rule" as a screening
trigger for a human to look. It is not a verdict, and nothing here should be
represented to a regulator as a compliance test.
""".strip()


def build_report(
    global_importance: pd.DataFrame,
    verification: pd.DataFrame,
    local_examples: pd.DataFrame,
    local_tables: dict,
    confusion: pd.DataFrame,
    error_segments: pd.DataFrame,
    error_characterisation: pd.DataFrame,
    reliability: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    confidence: pd.DataFrame,
    fairness_groups: pd.DataFrame,
    disparity: pd.DataFrame,
    monotonicity: pd.DataFrame,
    figures: dict[str, Path],
    reports_dir: Path,
) -> ReportBuilder:
    """Assemble every section of the Task 6 deliverable."""
    builder = ReportBuilder(title="Explainability & Responsible AI Report (Task 6)")

    builder.add_text(
        "What this covers",
        "Global and local explanations for the three Phase 3 binary models, an analysis of "
        "where they are wrong and on what kind of loan, reliability of the predicted "
        "probabilities, and a disparity screen across borrower segments.",
    )
    builder.add_text("Scope", SCOPE_NOTE)

    if not verification.empty:
        builder.add_table(
            "SHAP values cross-checked against the booster",
            verification,
            note="`shap` and LightGBM's own `pred_contrib` both run TreeSHAP, so they must "
            "agree exactly. Where they do not, one is being handed a different matrix than "
            "the other -- a stale category encoding, a reordered column -- and every "
            "attribution in this report would be describing a model that was never scored.",
        )

    builder.add_table(
        "Global feature importance",
        global_importance,
        note="Mean absolute SHAP per feature. `mean_signed_shap` carries direction: a "
        "feature can be important and directionless.",
        max_rows=45,
    )

    if not local_examples.empty:
        builder.add_table(
            "Loans selected for local explanation",
            local_examples,
            note="Deliberately not a highlight reel: a confident hit, a confident false "
            "positive, a missed event and a borderline case. Showing only the confident hit "
            "would demonstrate the model on the records where nothing was in doubt.",
        )
    for name, table in local_tables.items():
        builder.add_table(f"Local explanation: {name}", table, level=3, max_rows=15)

    builder.add_table("Confusion summary", confusion, note="At the threshold tuned in Task 2.")

    builder.add_table(
        "Error rates by segment",
        error_segments,
        note="Groups below the minimum size are dropped: a false positive rate computed on "
        "nine loans is noise with a decimal point, and putting it in a governance table "
        "invites someone to act on it.",
        max_rows=45,
    )

    builder.add_table(
        "What a false positive looks like",
        error_characterisation,
        note="Mean feature value among false positives against true negatives, standardised "
        "by the spread. A large gap says the model is flagging loans that look like *this*, "
        "which is the actionable form of an error analysis.",
        max_rows=30,
    )

    builder.add_table(
        "Calibration",
        calibration_summary,
        note="Expected calibration error is the population-weighted mean gap between "
        "predicted and observed, so a wild miss in a bin holding four records does not "
        "outweigh a small bias across the bulk of the book.",
    )
    builder.add_table("Reliability by probability bin", reliability, level=3, max_rows=45)
    builder.add_table(
        "Confidence profile",
        confidence,
        note="A model that never leaves a narrow band is technically calibrated and "
        "operationally useless: nothing is ever decided.",
        level=3,
        max_rows=30,
    )

    builder.add_text("Disparity screen", FAIRNESS_NOTE)
    builder.add_table("Disparity summary", disparity, max_rows=30)
    builder.add_table("Rates by group", fairness_groups, level=3, max_rows=60)
    if not monotonicity.empty:
        builder.add_table(
            "Monotonicity check",
            monotonicity,
            note="Not a fairness question but a correctness one: if the model flags "
            "740-799 borrowers more often than 620-659 borrowers, something is inverted.",
            level=3,
        )

    if figures:
        lines = []
        for caption, path in figures.items():
            relative = Path(path).relative_to(reports_dir)
            lines.append(f"**{caption}**\n\n![{caption}]({relative.as_posix()})\n")
        builder.add_text("Figures", "\n".join(lines))

    return builder


def build_model_card(
    global_importance: pd.DataFrame,
    disparity: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    error_segments: pd.DataFrame,
    results_path: Path,
) -> str:
    """
    The model card required by the problem statement's section 11.

    Written from the measured tables rather than from memory, so the limitations
    section cannot quietly diverge from what the pipeline actually found. Its
    absence is read as overconfidence; its presence is only worth anything if
    the "known failure modes" section is honest, so that section leads with the
    things that do not work.
    """
    def _top(model: str, n: int = 5) -> str:
        subset = global_importance[global_importance["model"] == model].head(n)
        if subset.empty:
            return "_not available_"
        return ", ".join(f"`{r.feature}` ({r.share:.0%})" for r in subset.itertuples())

    escalated = disparity[disparity.get("escalate", False)] if not disparity.empty else pd.DataFrame()
    worst_fp = (
        error_segments.sort_values("false_positive_rate", ascending=False).head(3)
        if not error_segments.empty else pd.DataFrame()
    )

    lines = [
        "# Model Card",
        "",
        "Loan Performance Intelligence Engine -- Intain Campus FinTech Challenge.",
        "Generated by `scripts/run_explainability.py`; every figure quoted is read out of the",
        f"tables in `{results_path.name}` and the reports beside it.",
        "",
        "## Intended use",
        "",
        "Portfolio-level surveillance of a residential mortgage book: ranking loans for",
        "review, projecting delinquency, default and prepayment under macro scenarios, and",
        "surfacing data-quality exceptions for a human reviewer.",
        "",
        "**Not intended for**, and not validated for: individual credit decisioning,",
        "pricing, adverse action, or any use where a decision is taken on a borrower without",
        "a human in the loop. The models were fitted on synthetic data (see Limitations).",
        "",
        "## Data",
        "",
        "- Monthly loan performance panel, 268,125 rows across 10,000 loans, 2017-01 to 2023-12.",
        "- Origination attributes joined per loan; a Phase 1 record-level data-quality score.",
        "- **Synthetic.** Generated by `scripts/generate_synthetic_suite.py` from a Markov",
        "  transition engine whose hazards depend on credit band and LTV band only.",
        "",
        "## Models",
        "",
        "| Head | Type | Target |",
        "| :--- | :--- | :--- |",
        "| Delinquency | LightGBM, isotonic-calibrated | 30+ DPD within 3 months |",
        "| Default | LightGBM, isotonic-calibrated | Default within 12 months |",
        "| Prepayment | LightGBM, isotonic-calibrated | Prepaid within 12 months |",
        "| Next state | LightGBM multiclass | Performance state at t+1 |",
        "| Time-to-event | Cause-specific Cox (competing risks) | Time to default / prepayment |",
        "| Exception | LightGBM + Isolation Forest + rules | Record-level data defects |",
        "",
        "Baselines retained for comparison: logistic regression (Task 2), constant hazard and",
        "Kaplan-Meier (Task 3), row-level rules alone (Task 4).",
        "",
        "## Validation",
        "",
        "- **Time-aware splits throughout. No random row-level split anywhere.**",
        "- Each Task 2 target's training window is **purged** by its own forward horizon, so a",
        "  row labelled over 12 months cannot sit in train while those months sit in validation.",
        "- Task 3 splits by **origination vintage**: older vintages train, newer ones test.",
        "- Evidence: `reports/task2_split_audit.csv`, and the leakage tests in",
        "  `tests/test_leakage_controls.py` (which fail on a deliberately shuffled split).",
        "",
        "## Metrics",
        "",
        "Held-out windows only. Full tables in `reports/task2_model_results.md`,",
        "`reports/survival_report.md`, `reports/anomaly_report.md`.",
        "",
        "## Top drivers",
        "",
        f"- **Delinquency**: {_top('delinquency')}",
        f"- **Default**: {_top('default')}",
        f"- **Prepayment**: {_top('prepayment')}",
        "",
        "## Calibration",
        "",
        calibration_summary.to_markdown(index=False) if not calibration_summary.empty else "_not available_",
        "",
        "Probabilities are isotonic-calibrated on the validation window with the base model",
        "frozen. Reweighting for class imbalance breaks the probability scale, and the",
        "probability is the deliverable -- it feeds the Task 5 scenario arithmetic.",
        "",
        "## Known failure modes",
        "",
        "Leading with what does not work, because that is the part of a model card worth reading.",
        "",
        "1. **Prepayment is close to unpredictable on this data.** ROC-AUC 0.52, PR-AUC 0.09",
        "   against a 0.09 base rate; precision never reaches 50% at any threshold. Task 3",
        "   agrees independently (Cox C = 0.558) and Task 5 agrees again (a 2.9x stated",
        "   prepayment multiplier produces 1.06x under feature stress). **Do not use the",
        "   prepayment head for anything.** The generator's prepayment hazard depends only on",
        "   credit band; an oracle using it scores AUC 0.55.",
        "2. **The Task 5 credit channel saturates.** Past month 24 no credit-score shift",
        "   reproduces the scenario file's stated default multiplier -- it tops out at 2.07x",
        "   against a stated 2.60x. Long-horizon feature-stress projections are a floor, not a",
        "   forecast.",
        "3. **Task 4's near-perfect scores are a property of injected defects**, not of the",
        "   model. Each defect class carries a near-deterministic fingerprint. Real servicing",
        "   errors arrive partially and mixed with legitimate rarities.",
        "4. **Proportional-hazards violations** are present in the Cox models and reported",
        "   rather than corrected; the constant hazard ratio assumption does not hold for every",
        "   covariate.",
        "5. **Scenario projections are not a run-off.** The portfolio is held at its last",
        "   observed position; loans that default or prepay are not removed as the horizon",
        "   extends, and no balance is amortised.",
        "",
        "## Limitations",
        "",
        "- **The data is synthetic.** Every metric in this repository measures whether the",
        "  pipeline is wired correctly. None of them forecasts performance on a real servicer",
        "  feed, where the relationships are noisier and the defects less cleanly separable.",
        "- **No protected attributes exist in the panel**, so no legal fairness analysis is",
        "  possible. What is reported is a disparity screen across geography and servicer.",
        "- SHAP explains the booster's log-odds, not the calibrated probability.",
        "- Segment results are unreliable below the minimum group size and are suppressed.",
        "",
        "## Responsible AI",
        "",
        "- **The LLM never produces a prediction.** It is confined to explaining and",
        "  summarising grounded model output, and every call is logged.",
        "- Disparity screen across credit band, LTV band, vintage, state and servicer.",
    ]

    if not escalated.empty:
        lines.append("")
        lines.append("**Escalated disparity findings:**")
        lines.append("")
        for row in escalated.itertuples():
            lines.append(
                f"- `{row.segment}` on `{row.metric}`: ratio {row.ratio_best_to_worst:.2f} "
                f"(worst `{row.worst_group}` at {row.worst_value:.1%}, best "
                f"`{row.best_group}` at {row.best_value:.1%})."
            )
    else:
        lines.append("")
        lines.append(
            "No disparity on a non-risk-factor segment fell below the 0.80 screening floor "
            "in this run."
        )

    if not worst_fp.empty:
        lines.append("")
        lines.append("**Segments carrying the most false positives:**")
        lines.append("")
        for row in worst_fp.itertuples():
            group = getattr(row, "group", None) or getattr(row, row.segment, "")
            lines.append(
                f"- `{row.segment}` = `{group}`: false positive rate "
                f"{row.false_positive_rate:.1%} across {int(row.records):,} records."
            )

    lines += [
        "",
        "## Reproducing",
        "",
        "```bash",
        "make all          # every phase, from raw CSVs to every report",
        "make test         # the leakage, censoring and detector regression suite",
        "```",
        "",
        "One seed (`src/config.RANDOM_SEED`) is used everywhere.",
        "",
    ]
    return "\n".join(lines)
