"""
The dashboard's pages, one per phase, ordered to match the demo flow.

Section 14 of the problem statement specifies the order a demo should follow:
dataset and targets, profiling, data-quality issues, features, the time-aware
split, baseline then improved model, survival output, anomaly examples,
scenario output, a local explanation, an LLM note, a rejected LLM output, the
submission, and the development log. The navigation below *is* that order, so
the app can be walked top to bottom on camera without deciding what comes next.

Each page shows the pipeline's own tables and figures. Where a page would
otherwise present a number that cannot support the reading a viewer will give
it -- Task 4's near-perfect scores, the prepayment head -- the caveat sits
beside the number rather than in a footnote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import data, theme


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _missing(phase: str, command: str) -> None:
    st.info(f"**{phase} has not been run yet.** Run `{command}` and reload this page.")


def _table(frame: pd.DataFrame, height: int | None = None, **kwargs) -> None:
    """
    One table helper for the whole app, so a Streamlit API change lands in one
    place. ``height`` is omitted rather than passed as None: Streamlit >= 1.60
    rejects None and accepts only a positive integer, "stretch" or "content".
    """
    if frame.empty:
        st.caption("No data.")
        return
    if height is not None:
        kwargs["height"] = height
    st.dataframe(frame, width="stretch", hide_index=True, **kwargs)


def _tiles(items: list) -> None:
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        column.markdown(theme.tile(**item), unsafe_allow_html=True)


def _gallery(figures: list, columns: int = 1) -> None:
    if not figures:
        st.caption("No figures; the phase may have been run with `--no-figures`.")
        return
    for index in range(0, len(figures), columns):
        for column, (title, path) in zip(st.columns(columns), figures[index : index + columns]):
            column.image(str(path), caption=title, width="stretch")


# --------------------------------------------------------------------------
# 1. Overview
# --------------------------------------------------------------------------
def overview() -> None:
    st.title("Loan Performance Intelligence Engine")
    st.caption(
        "Intain Campus FinTech Challenge -- AI Track. Every figure on this page is read "
        "from a file the pipeline wrote; nothing is recomputed in the app."
    )

    quality = data.read_csv("reports/dq_scores_train.csv")
    task2 = data.read_csv("reports/task2_model_results.csv")
    survival = data.read_csv("reports/survival/model_comparison.csv")
    anomaly = data.read_csv("reports/anomaly/detector_ablation.csv")
    submission = data.read_csv("submission/submission.csv")
    validation = data.read_csv("reports/submission_validation.csv")

    tiles = []
    if not quality.empty:
        tiles.append({
            "label": "Batch data quality", "value": f"{quality['dq_score'].mean():.1f}",
            "unit": "/ 100",
            "note": f"{(quality['dq_score'] >= 99.99).mean():.0%} of records defect-free",
            "tone": "good",
        })
    if not task2.empty:
        row = task2[(task2.target == "next_12m_default_flag") & (task2.model == "improved")]
        if not row.empty:
            tiles.append({
                "label": "12-month default", "value": f"{row.roc_auc.iloc[0]:.3f}", "unit": "ROC-AUC",
                "note": f"PR-AUC {row.pr_auc.iloc[0]:.3f} · Brier {row.brier_calibrated.iloc[0]:.3f} calibrated",
                "tone": "good",
            })
    if not survival.empty:
        row = survival[(survival.cause == "default") & (survival.model == "cox")]
        if not row.empty:
            tiles.append({
                "label": "Time to default", "value": f"{row.concordance.iloc[0]:.3f}",
                "unit": "Harrell's C", "note": "cause-specific Cox, holdout vintages", "tone": "good",
            })
    if not anomaly.empty:
        row = anomaly[anomaly.detector == "supervised (all signals)"]
        if not row.empty:
            tiles.append({
                "label": "Exception detection", "value": f"{row.precision.iloc[0]:.3f}",
                "unit": "precision", "note": "defects were injected -- see the caveat below",
                "tone": "warning",
            })
    if not submission.empty:
        failed = int((~validation.passed.astype(bool)).sum()) if not validation.empty else 0
        tiles.append({
            "label": "Submission", "value": f"{len(submission):,}", "unit": "rows",
            "note": "all checks passed" if failed == 0 else f"{failed} check(s) failed",
            "tone": "good" if failed == 0 else "critical",
        })

    if tiles:
        _tiles(tiles)

    st.markdown("")
    st.markdown(
        theme.caveat(
            "<b>Read every score on this dashboard as a property of the data.</b> The panel "
            "is synthetic, generated by a Markov engine whose hazards depend on credit band "
            "and LTV band, with data defects injected deterministically. These numbers "
            "measure whether the pipeline is wired correctly. They do not forecast "
            "performance on a real servicer feed, where relationships are noisier and "
            "defects far less separable. The parts that transfer are the leakage controls, "
            "the layering and the governance -- not the metrics."
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Required deliverables")
        frame = data.deliverables()
        _table(
            frame.assign(Present=frame.Present.map({True: "yes", False: "MISSING"})),
            height=390,
        )
    with right:
        st.subheader("Pipeline")
        st.code(
            "make setup        # venv + pinned dependencies\n"
            "make data         # generate the synthetic pack\n"
            "python main.py    # all nine phases -> submission.csv\n"
            "make test         # 108 regression tests\n\n"
            "streamlit run app.py   # this dashboard",
            language="bash",
        )
        state = data.pipeline_state()
        st.markdown("**Phases with output on disk**")
        st.markdown(
            " ".join(
                theme.pill(name, "ok" if ok else "bad") for name, ok in state.items()
            ),
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# 2. Data intelligence (Task 1)
# --------------------------------------------------------------------------
def data_intelligence() -> None:
    st.title("Data intelligence")
    st.caption("Task 1 -- profiling, missingness, outliers, drift and record-level quality scoring.")

    scores = data.read_csv("reports/dq_scores_train.csv")
    if scores.empty:
        return _missing("Profiling", "make profile")

    _tiles([
        {"label": "Records profiled", "value": f"{len(scores):,}", "note": "monthly panel rows"},
        {"label": "Mean quality score", "value": f"{scores.dq_score.mean():.1f}", "unit": "/ 100",
         "note": "100 * exp(-penalty / 10)", "tone": "good"},
        {"label": "Defect-free", "value": f"{(scores.dq_score >= 99.99).mean():.1%}",
         "note": "no rule, date, outlier or conflict flag"},
        {"label": "Rule violations", "value": f"{int(scores.n_rule_violations.gt(0).sum()):,}",
         "note": "records tripping at least one rule", "tone": "warning"},
    ])

    st.subheader("Figures")
    _gallery(data.figures("reports/profiling/charts"))

    st.subheader("Worst-scoring records")
    worst = data.read_csv("reports/profiling/worst_records.csv")
    _table(worst.head(15))

    st.subheader("Full report")
    st.markdown(
        "The complete Data Intelligence Report -- 10 sections, 20 tables -- is at "
        "`reports/data_intelligence_report.html`."
    )


# --------------------------------------------------------------------------
# 3. Features and the time-aware split
# --------------------------------------------------------------------------
def features() -> None:
    st.title("Features & the time-aware split")
    st.caption("Phase 2 -- what the models see, and the guarantee that none of it reads forward.")

    dictionary = data.read_csv("reports/feature_dictionary.csv")
    split = data.read_csv("reports/task2_split_audit.csv")
    if dictionary.empty:
        return _missing("Feature engineering", "make predict")

    _tiles([
        {"label": "Model features", "value": f"{len(dictionary)}", "note": "improved model"},
        {"label": "Baseline features", "value": f"{int(dictionary.in_baseline.sum())}",
         "note": "what a analyst sees without engineering"},
        {"label": "Families", "value": f"{dictionary.family.nunique()}",
         "note": ", ".join(dictionary.family.unique()[:4])},
        {"label": "Undocumented", "value": f"{int((dictionary.family == 'unclassified').sum())}",
         "note": "features with no dictionary entry", "tone": "good"},
    ])

    st.subheader("No random split, anywhere")
    st.markdown(
        "A monthly panel repeats each loan dozens of times. A random split puts month *t* of "
        "a loan in training and month *t+1* of the **same loan** in test, so the metrics "
        "measure recall of the training set rather than forecasting skill.\n\n"
        "Each target's training window is additionally **purged** by its own forward horizon: "
        "a row labelled over the next 12 months is dropped from training if those months "
        "reach into validation."
    )
    _table(split)

    st.subheader("Feature dictionary")
    family = st.selectbox("Family", ["all", *sorted(dictionary.family.unique())])
    view = dictionary if family == "all" else dictionary[dictionary.family == family]
    _table(
        view[["feature", "family", "information_window", "dtype", "in_baseline", "definition"]],
        height=430,
    )
    st.caption(
        "`information_window` is the leakage claim, stated per feature. `as-at t` uses only "
        "month t's own record; `t-k..t` is backward-looking and inclusive of t. No feature "
        "reads a month after t -- a test rebuilds the matrix on a panel truncated after t "
        "and requires every rolling feature to be unchanged."
    )


# --------------------------------------------------------------------------
# 4. Prediction (Task 2)
# --------------------------------------------------------------------------
def prediction() -> None:
    st.title("Loan performance prediction")
    st.caption("Task 2 -- baseline versus improved, five targets, on a strictly later test window.")

    results = data.read_csv("reports/task2_model_results.csv")
    if results.empty:
        return _missing("Prediction", "make predict")

    metric_columns = [
        "target", "model", "backend", "n_features", "roc_auc", "pr_auc", "f1",
        "recall_at_precision_0.5", "macro_f1", "brier_uncalibrated", "brier_calibrated",
    ]
    available = [c for c in metric_columns if c in results.columns]

    st.subheader("Baseline vs improved")
    _table(results[available].round(4))

    st.markdown(
        theme.caveat(
            "<b>Two results worth stating rather than burying.</b> On "
            "<code>next_3m_delinquency_flag</code> the improved model's ROC-AUC is slightly "
            "<i>worse</i> than the baseline's (0.753 vs 0.760) while its PR-AUC is materially "
            "better (0.472 vs 0.443) -- on an 11% base rate, PR-AUC is the metric that "
            "reflects usable ranking. And <b>the prepayment head does not work</b>: ROC-AUC "
            "0.52 against a 0.09 base rate, with precision never reaching 50% at any "
            "threshold. Three phases reach that conclusion independently."
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Calibration")
    st.markdown(
        "Reweighting for class imbalance fixes ranking and breaks the probability scale. "
        "The probability is the deliverable here -- it feeds the scenario arithmetic -- so "
        "every model is isotonic-calibrated on the validation window with the base model "
        "frozen, and Brier is reported before and after."
    )
    if {"brier_uncalibrated", "brier_calibrated"}.issubset(results.columns):
        improvement = results[results.model == "improved"][
            ["target", "brier_uncalibrated", "brier_calibrated"]
        ].copy()
        improvement["improvement"] = (
            improvement.brier_uncalibrated - improvement.brier_calibrated
        )
        _table(improvement.round(4))


# --------------------------------------------------------------------------
# 5. Survival (Task 3)
# --------------------------------------------------------------------------
def survival() -> None:
    st.title("Time to event")
    st.caption("Task 3 -- default and prepayment as competing risks, on a months-on-book clock.")

    comparison = data.read_csv("reports/survival/model_comparison.csv")
    censoring = data.read_csv("reports/survival/censoring_summary.csv")
    horizons = data.read_csv("reports/survival/cumulative_incidence_horizons.csv")
    if comparison.empty:
        return _missing("Survival modelling", "make survival")

    if not censoring.empty:
        lookup = dict(zip(censoring.metric, censoring.value))
        _tiles([
            {"label": "Loans", "value": f"{int(float(lookup.get('loans', 0))):,}"},
            {"label": "Defaults", "value": f"{int(float(lookup.get('events_default', 0))):,}"},
            {"label": "Prepayments", "value": f"{int(float(lookup.get('events_prepaid', 0))):,}"},
            {"label": "Right-censored", "value": f"{float(lookup.get('censoring_rate', 0)):.1%}",
             "note": "still performing at the cutoff -- kept, never dropped", "tone": "good"},
        ])

    st.subheader("Baseline vs advanced")
    _table(comparison.round(4))
    st.caption(
        "The constant-hazard model scores C = 0.5 by construction -- it has no covariates -- "
        "so the comparison that matters for it is the Brier column, where it still sets a "
        "real level to beat."
    )

    st.subheader("Competing risks are not censoring")
    st.markdown(
        "A prepaid loan has left the portfolio and can never default. Treating it as censored "
        "asserts it might still default at some unobserved time, which inflates default "
        "incidence:"
    )
    if not horizons.empty:
        _table(horizons.round(4))

    st.subheader("Event curves")
    _gallery(data.figures("reports/survival"))


# --------------------------------------------------------------------------
# 6. Anomalies (Task 4) -- with the live threshold control
# --------------------------------------------------------------------------
def anomalies() -> None:
    st.title("Anomaly & exception detection")
    st.caption("Task 4 -- deterministic rules, sequence-aware detectors and unsupervised ML.")

    ablation = data.read_csv("reports/anomaly/detector_ablation.csv")
    coverage = data.read_csv("reports/anomaly/signal_coverage.csv")
    queue = data.read_csv("reports/anomaly_examples.csv")
    if ablation.empty:
        return _missing("Anomaly detection", "make anomaly")

    st.markdown(
        theme.caveat(
            "<b>The near-perfect scores below are a property of injected defects, not of the "
            "model.</b> Each class carries a near-deterministic fingerprint because a "
            "generator put it there. What transfers to a real feed is the layering -- rules "
            "for what is <i>wrong</i>, unsupervised detection for what is <i>unusual</i> -- "
            "not the numbers."
        ),
        unsafe_allow_html=True,
    )

    st.subheader("What each detector layer buys")
    _table(ablation.round(4))
    st.caption(
        "Row-level rules catch every Balance Discrepancy and Time Travel defect and roughly "
        "one in ten Impossible State Transitions and Zombie Loans -- an expression evaluated "
        "against one row cannot see last month's status. Two sequence-aware detectors take "
        "overall recall from 52% to 99.7%."
    )

    if not coverage.empty:
        st.subheader("Signal coverage")
        st.caption(
            "A signal that fires constantly and is almost never an exception is a reviewer's "
            "time being spent, so it is named rather than averaged away."
        )
        _table(coverage.round(4), height=330)

    if not queue.empty:
        st.subheader("Reviewer queue")
        left, right = st.columns([1, 2])
        with left:
            kinds = ["all", *sorted(queue.predicted_exception_type.dropna().unique())]
            kind = st.selectbox("Predicted type", kinds)
        with right:
            floor = st.slider("Minimum hybrid score", 0.0, 1.0, 0.0, 0.01)

        view = queue if kind == "all" else queue[queue.predicted_exception_type == kind]
        view = view[view.hybrid_score >= floor]
        st.caption(f"{len(view)} of {len(queue)} curated examples")
        _table(
            view[["loan_id", "reporting_month", "predicted_exception_type", "hybrid_score",
                  "exception_probability", "triggered_rules", "top_drivers", "suggested_action"]],
            height=380,
        )
        st.caption(
            "Stratified, not top-N: a guaranteed block per type plus five slots reserved for "
            "high-scoring records with **no** rule violation -- the only rows in the queue "
            "that can teach the rule set something."
        )

    st.subheader("Figures")
    _gallery(data.figures("reports/anomaly"))


# --------------------------------------------------------------------------
# 7. Scenarios (Task 5) -- interactive
# --------------------------------------------------------------------------
def scenarios() -> None:
    st.title("Scenario & stress simulation")
    st.caption("Task 5 -- the Phase 3 models re-scored under each macro scenario.")

    projection = data.read_csv("reports/scenario_report.csv")
    saturation = data.read_csv("reports/scenario/credit_saturation.csv")
    if projection.empty:
        return _missing("Scenario simulation", "make scenario")

    measures = {
        "Default (next 12m)": "default_12m",
        "Delinquency (next 3m)": "delinquency_3m",
        "Prepayment (next 12m)": "prepayment_12m",
    }
    left, right = st.columns([1, 1])
    with left:
        label = st.selectbox("Measure", list(measures))
    with right:
        horizon = st.select_slider(
            "Projection month", sorted(projection.horizon_month.unique()),
            value=int(projection.horizon_month.max()),
        )
    measure = measures[label]

    at_horizon = projection[projection.horizon_month == horizon]
    baseline = at_horizon[at_horizon.scenario == "Baseline"]
    base_rate = float(baseline[measure].iloc[0]) if not baseline.empty else np.nan

    tiles = []
    for _, row in at_horizon.iterrows():
        delta = (row[measure] - base_rate) * 100
        tiles.append({
            "label": row.scenario,
            "value": f"{row[measure]:.2%}",
            "note": ("baseline" if row.scenario == "Baseline"
                     else f"{delta:+.2f} pp vs baseline · credit shift "
                          f"{row.credit_score_shift:+.0f} pts"),
            "tone": "critical" if delta > 1 else ("good" if delta < -0.5 else ""),
        })
    if tiles:
        _tiles(tiles)

    st.subheader(f"{label} by projection month")
    pivot = projection.pivot_table(index="horizon_month", columns="scenario", values=measure)
    stated = f"{measure}_stated"
    if stated in projection.columns:
        extra = projection[projection.scenario != "Baseline"].pivot_table(
            index="horizon_month", columns="scenario", values=stated
        )
        extra.columns = [f"{c} (stated multiplier)" for c in extra.columns]
        pivot = pivot.join(extra)
    st.line_chart(pivot, height=340)
    st.caption(
        "Solid lines are the model re-scored on stressed features. Where a dotted "
        "*stated multiplier* line diverges, the credit channel has run out of room and the "
        "feature-stress figure is a floor, not a forecast."
    )

    if not saturation.empty and "reached" in saturation.columns:
        unreached = saturation[~saturation.reached.astype(bool)]
        if not unreached.empty:
            st.markdown(
                theme.caveat(
                    "<b>The credit channel saturates.</b> Past month 24 no credit-score shift "
                    "reproduces the scenario file's stated default multiplier -- even moving "
                    "the whole book to the floor of the observable score range tops out at "
                    f"{unreached.attainable_multiplier.max():.2f}x against a stated "
                    f"{unreached.stated_multiplier.max():.2f}x. A naive calibration would "
                    "clamp at its search bound and report the clamp as an answer."
                ),
                unsafe_allow_html=True,
            )
            _table(saturation.round(4))

    st.subheader("Segment impact")
    segment = st.selectbox(
        "Segment", ["credit_score_band", "vintage_year", "state", "servicer_name"]
    )
    table = data.read_csv(f"reports/scenario/segment_{segment}.csv")
    if not table.empty:
        _table(table[table.horizon_month == horizon].round(4), height=330)

    st.subheader("Figures")
    _gallery(data.figures("reports/scenario"))


# --------------------------------------------------------------------------
# 8. Explainability (Task 6)
# --------------------------------------------------------------------------
def explainability() -> None:
    st.title("Explainability & responsible AI")
    st.caption("Task 6 -- what drives the models, where they are wrong, and on whom.")

    importance = data.read_csv("reports/explainability_report/global_importance.csv")
    calibration = data.read_csv("reports/explainability_report/calibration_summary.csv")
    disparity = data.read_csv("reports/explainability_report/disparity_summary.csv")
    errors = data.read_csv("reports/explainability_report/error_rates_by_segment.csv")
    if importance.empty:
        return _missing("Explainability", "make explain")

    st.markdown(
        "**SHAP explains the booster, not the deployed probability.** `TreeExplainer` "
        "decomposes the base model's log-odds; the isotonic calibrator sits on top. Being "
        "monotone it cannot reorder contributions, but the decomposition does not sum to the "
        "calibrated probability. Error analysis, reliability and the disparity screen all run "
        "on the *calibrated* probability at the tuned threshold, because that is what a "
        "borrower experiences."
    )

    model = st.selectbox("Model", sorted(importance.model.unique()))

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Top drivers")
        top = importance[importance.model == model].head(12)
        _table(top[["feature", "mean_abs_shap", "share"]].round(4))
    with right:
        st.subheader("Beeswarm")
        matches = [f for f in data.figures("reports/explainability_report")
                   if "beeswarm" in f[1].name and model in f[1].name]
        if matches:
            st.image(str(matches[0][1]), width="stretch")

    if not calibration.empty:
        st.subheader("Calibration")
        _table(calibration.round(4))
        st.caption(
            "Expected calibration error is population-weighted, so a wild miss in a bin "
            "holding four records does not outweigh a small bias across the bulk of the book."
        )

    st.subheader("Local explanations")
    st.markdown(
        "Chosen so the demo is not a highlight reel: a confident hit, a confident **false "
        "positive**, a missed event and a borderline case. Showing only the confident hit "
        "would demonstrate the model on the records where nothing was ever in doubt."
    )
    waterfalls = [f for f in data.figures("reports/explainability_report")
                  if "waterfall" in f[1].name and model in f[1].name]
    if waterfalls:
        choice = st.selectbox("Case", [t for t, _ in waterfalls])
        st.image(str(dict(waterfalls)[choice]), width="stretch")

    st.subheader("Disparity screen")
    st.markdown(
        theme.caveat(
            "<b>This is a disparity screen, not a legal fairness test.</b> The panel contains "
            "no protected attribute, so no disparate-impact analysis in the legal sense is "
            "possible. A credit-band gap is the model working correctly and is never "
            "escalated; <code>vintage_year</code> is treated the same way because within one "
            "reporting window it is almost pure loan age. What is ranked is <b>error-rate "
            "parity conditional on outcome</b> across geography and servicer -- among "
            "borrowers who did <i>not</i> default, is one group flagged more often?"
        ),
        unsafe_allow_html=True,
    )
    if not disparity.empty and "escalate" in disparity.columns:
        escalated = disparity[disparity.escalate.astype(bool)]
        st.caption(
            f"{len(escalated)} of {len(disparity)} segment-metric pairs escalated. A "
            "two-proportion significance test and a minimum event count cut this from 19; a "
            "screen that flags everything is one nobody reads."
        )
        _table(
            escalated[["model", "segment", "metric", "worst_group", "worst_value",
                       "best_group", "best_value", "ratio_best_to_worst", "p_value"]].round(5),
            height=300,
        )

    if not errors.empty:
        st.subheader("Error rates by segment")
        segment = st.selectbox("Segment", sorted(errors.segment.unique()), key="err_seg")
        view = errors[(errors.segment == segment) & (errors.model == model)]
        _table(view.round(4), height=300)


# --------------------------------------------------------------------------
# 9. Copilot (Task 7)
# --------------------------------------------------------------------------
def copilot() -> None:
    st.title("LLM reviewer copilot")
    st.caption("Task 7 -- grounded summarisation, guardrails, and a mandatory audit trail.")

    notes = data.read_csv("reports/copilot/reviewer_notes.csv")
    probes = data.read_csv("reports/copilot/adversarial_probes.csv")
    controls = data.read_csv("reports/copilot/control_failures.csv")
    log = data.read_jsonl("reports/llm_prompt_log.jsonl")
    if not log:
        return _missing("The copilot", "make copilot")

    calls = [r for r in log if r.get("record_type", "call") == "call"]
    live = [r for r in calls if r.get("status") == "ok"]
    _tiles([
        {"label": "Logged calls", "value": f"{len(calls):,}",
         "note": "prompt, model, timestamp, output, verdict"},
        {"label": "Live responses", "value": f"{len(live):,}",
         "note": f"{live[-1]['model'] if live else '-'}"},
        {"label": "Notes released", "value": f"{int(notes.released_to_reviewer.sum()) if not notes.empty else 0}"
                                             f" / {len(notes)}",
         "note": "the rest withheld by guardrails", "tone": "good"},
        {"label": "Adversarial probes", "value": f"{len(probes)}",
         "note": "prediction, hallucination, decision, false premise, authority, vagueness"},
    ])

    st.markdown(
        "**The copilot never predicts.** It sits strictly downstream of every model here: "
        "scores and probabilities arrive as *inputs* and are restated. No code path reaches "
        "a model output the statistical pipeline did not already produce -- that is the "
        "challenge's qualification rule, not a preference.\n\n"
        "**The prompt is a request; the guardrails are the control.** Instructing a model not "
        "to invent numbers checks nothing. Prediction language, decision language and "
        "**numeric grounding** are checked on every response before it reaches a reviewer."
    )

    st.subheader("Generated reviewer notes")
    if not notes.empty:
        choice = st.selectbox(
            "Loan", notes.loan_id + "  ·  " + notes.reporting_month.astype(str)
        )
        row = notes.iloc[list(notes.loan_id + "  ·  " + notes.reporting_month.astype(str)).index(choice)]
        st.markdown(
            theme.pill("released" if row.released_to_reviewer else "WITHHELD",
                       "ok" if row.released_to_reviewer else "bad")
            + theme.pill(f"guardrails: {row.guardrail_detail}", "info")
            + theme.pill(row.model, "info"),
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="note-box">{row.note}</div>', unsafe_allow_html=True)

    st.subheader("Deliberate failure testing")
    st.markdown(
        "Probes phrased the way a hurried reviewer would phrase them, because the realistic "
        "threat is not a malicious prompt but a colleague typing *\"so is this one going to "
        "default?\"* into the box."
    )
    if not probes.empty:
        _table(probes[["probe", "failure_mode", "outcome", "model_refused",
                       "guardrails_passed"]])
        probe = st.selectbox("Inspect a probe", probes.probe.tolist())
        row = probes[probes.probe == probe].iloc[0]
        st.markdown(f"**The trap.** {row.why_it_is_a_trap}")
        st.markdown(f"**What the model returned.**")
        st.markdown(f'<div class="note-box">{str(row.llm_output)[:1800]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"**Outcome.** {row.outcome}")
        st.markdown(f"**Human correction.** {row.human_correction}")

    if not controls.empty:
        st.subheader("Failures of the control, not the model")
        st.markdown(
            theme.caveat(
                "The model passed every probe on the live run. Manufacturing a model failure "
                "to fill this section would be worse than useless, so what is reported is "
                "what genuinely failed: <b>the guardrail, three times</b>, each time by "
                "blocking a correct note. A control that accuses faithful output of "
                "hallucinating gets switched off by the people it protects, and the one real "
                "hallucination then goes out with the rest."
            ),
            unsafe_allow_html=True,
        )
        for _, row in controls.iterrows():
            with st.expander(row.failure):
                st.markdown(f"**Observed.** {row.observed}")
                st.markdown(f"**Why it matters.** {row.why_it_matters}")
                st.markdown(f"**Correction.** {row.correction}")

    st.subheader("Audit trail")
    st.caption(
        f"{len(log):,} records in `reports/llm_prompt_log.jsonl` -- append-only, one JSON "
        "object per line, with the guardrail verdict as a second record joined on `call_id`."
    )
    if calls:
        recent = pd.DataFrame([
            {"timestamp": r.get("timestamp"), "purpose": r.get("purpose"),
             "model": r.get("model"), "status": r.get("status"),
             "latency_s": r.get("latency_seconds"),
             "tokens": (r.get("usage") or {}).get("total_tokens")}
            for r in calls[-25:]
        ])
        _table(recent, height=300)


# --------------------------------------------------------------------------
# 10. Loan explorer -- the interactive centrepiece
# --------------------------------------------------------------------------
def explorer() -> None:
    st.title("Loan explorer")
    st.caption(
        "Everything the pipeline concluded about one loan-month, from the submission it "
        "actually wrote."
    )

    submission = data.read_csv("submission/submission.csv")
    if submission.empty:
        return _missing("Inference", "python main.py --submission")

    # The loan's observed state is not in the submission but a reviewer needs it:
    # without it a deterministic override reads as a model prediction.
    panel = data.read_csv("data/loan_monthly_performance_test.csv")
    if not panel.empty and "current_status" in panel.columns:
        panel = panel[["loan_id", "reporting_month", "current_status"]].copy()
        panel["reporting_month"] = pd.to_datetime(panel.reporting_month).dt.strftime("%Y-%m-%d")
        submission = submission.merge(panel, on=["loan_id", "reporting_month"], how="left")

    queue = data.read_csv("reports/anomaly_examples.csv")

    st.markdown("**Pick a loan**")
    left, middle, right = st.columns([1.2, 1, 1])
    with left:
        source = st.radio(
            "Start from", ["Highest anomaly score", "Reviewer queue", "Search by loan ID"],
            label_visibility="collapsed",
        )
    with middle:
        month_filter = st.selectbox(
            "Reporting month", ["any", *sorted(submission.reporting_month.unique())]
        )
    with right:
        state_filter = st.selectbox(
            "Predicted next state", ["any", *sorted(submission.next_state.dropna().unique())]
        )

    pool = submission
    if month_filter != "any":
        pool = pool[pool.reporting_month == month_filter]
    if state_filter != "any":
        pool = pool[pool.next_state == state_filter]

    if source == "Search by loan ID":
        query = st.text_input("Loan ID contains", "")
        pool = pool[pool.loan_id.str.contains(query, case=False, na=False)] if query else pool.head(0)
    elif source == "Reviewer queue" and not queue.empty:
        pool = pool[pool.loan_id.isin(queue.loan_id)]
    else:
        pool = pool.sort_values("anomaly_score", ascending=False)

    if pool.empty:
        st.info("No loans match. Widen the filters or search for a different ID.")
        return

    options = (pool.loan_id + "  ·  " + pool.reporting_month.astype(str)).head(300).tolist()
    choice = st.selectbox(f"{len(pool):,} matching rows", options)
    loan_id, month = [part.strip() for part in choice.split("·")]
    row = pool[(pool.loan_id == loan_id) & (pool.reporting_month == month)].iloc[0]

    st.markdown("---")
    st.subheader(f"{loan_id} · {month}")

    status = row.get("current_status")
    absorbing = str(status) in ("Default", "Prepaid")

    # A 0.0% forward probability on an already-terminal loan is a deterministic
    # override, not a confident model prediction. Saying which is which is the
    # difference between a reviewer trusting the number and misreading it.
    forward_note = (
        f"deterministic: the loan is already {status}, so no new forward event is possible"
        if absorbing
        else f"3m delinquency {row.prob_next_3m_delinquency:.1%}"
    )

    _tiles([
        {"label": "Observed status", "value": str(status) if status else "unknown",
         "note": "as at this reporting month"},
        {"label": "Predicted next state", "value": str(row.next_state),
         "note": ("self-transition: an absorbing state is terminal by definition"
                  if absorbing else f"confidence {row.confidence:.1%}")},
        {"label": "12-month default", "value": f"{row.prob_next_12m_default:.1%}",
         "note": forward_note,
         "tone": "critical" if row.prob_next_12m_default > 0.3 else ""},
        {"label": "12-month prepayment", "value": f"{row.prob_next_12m_prepayment:.1%}",
         "note": "the prepayment head is not reliable -- see Task 2", "tone": "warning"},
        {"label": "Anomaly score", "value": f"{row.anomaly_score:.3f}",
         "note": f"exception: {row.exception_type}",
         "tone": "critical" if row.exception_required else "good"},
    ])

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Top drivers**")
        st.markdown(f'<div class="note-box">{row.top_drivers}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("**Recommended action**")
        st.markdown(f'<div class="note-box">{row.action}</div>', unsafe_allow_html=True)

    if not queue.empty and loan_id in set(queue.loan_id):
        st.markdown("**This loan is in the curated reviewer queue**")
        _table(queue[queue.loan_id == loan_id])

    notes = data.read_csv("reports/copilot/reviewer_notes.csv")
    if not notes.empty and loan_id in set(notes.loan_id):
        st.markdown("**LLM reviewer note**")
        note = notes[notes.loan_id == loan_id].iloc[0]
        st.markdown(
            theme.pill("released" if note.released_to_reviewer else "WITHHELD",
                       "ok" if note.released_to_reviewer else "bad"),
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="note-box">{note.note}</div>', unsafe_allow_html=True)

    with st.expander("Full submission row"):
        st.json({k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()})


# --------------------------------------------------------------------------
# 11. Submission & reproducibility
# --------------------------------------------------------------------------
def submission_page() -> None:
    st.title("Submission & reproducibility")
    st.caption("Phase 9 -- the graded file, and the checks run before it was written.")

    submission = data.read_csv("submission/submission.csv")
    validation = data.read_csv("reports/submission_validation.csv")
    if submission.empty:
        return _missing("Inference", "python main.py --submission")

    failed = int((~validation.passed.astype(bool)).sum()) if not validation.empty else 0
    _tiles([
        {"label": "Rows", "value": f"{len(submission):,}", "note": "one per test panel row"},
        {"label": "Columns", "value": f"{submission.shape[1]}", "note": "matching the template exactly"},
        {"label": "Validation checks", "value": f"{len(validation) - failed} / {len(validation)}",
         "note": "run before the file was written", "tone": "good" if failed == 0 else "critical"},
        {"label": "Exceptions flagged", "value": f"{submission.exception_required.mean():.2%}",
         "note": "against a 2.63% base rate in the labelled panel"},
    ])

    st.subheader("Validation")
    st.caption(
        "Every check exists because its failure would be invisible in a spot check: columns "
        "in the wrong order look correct and are wrong to a scorer that joins on position; a "
        "cell holding the string \"None\" passes every in-memory null test and reads back as "
        "NaN the moment anyone opens the file."
    )
    if not validation.empty:
        _table(validation.assign(passed=validation.passed.map({True: "ok", False: "FAIL"})),
               height=330)

    st.subheader("Detection against the injection ledger")
    st.markdown(
        "The predicted exception counts on the *unlabelled* panel match the generator's "
        "ground truth exactly. Read it as an end-to-end wiring check, not a performance "
        "claim -- each defect class carries a near-deterministic fingerprint because it was "
        "injected."
    )
    counts = submission.exception_type.value_counts().rename_axis("exception_type")
    _table(counts.reset_index(name="predicted_in_test"))

    st.subheader("Sample rows")
    only_flagged = st.checkbox("Only flagged exceptions", value=True)
    view = submission[submission.exception_required == 1] if only_flagged else submission
    _table(view.head(200), height=380)

    st.subheader("Reproducing")
    st.code(
        "make setup        # venv + pinned dependencies\n"
        "make data         # generate the synthetic benchmark pack\n"
        "python main.py    # all nine phases -> submission/submission.csv\n"
        "make test         # 108 regression tests",
        language="bash",
    )
    st.caption(
        f"One seed (`src/config.RANDOM_SEED`) is used everywhere. CI runs the suite plus a "
        "small-sample smoke run of every pipeline on each push."
    )


# --------------------------------------------------------------------------
# 12. Model card & development log
# --------------------------------------------------------------------------
def documents() -> None:
    st.title("Model card & development log")

    tab_card, tab_log = st.tabs(["Model card", "AI development log"])
    with tab_card:
        card = data.read_text("reports/model_card.md")
        if card:
            st.markdown(card)
        else:
            _missing("The model card", "python main.py --model-card")
    with tab_log:
        log = data.read_text("ai_dev_log/log.md")
        if log:
            st.markdown(log)
        else:
            st.info("No development log found at `ai_dev_log/log.md`.")
