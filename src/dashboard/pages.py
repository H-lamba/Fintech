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

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from . import data, theme


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _missing(phase: str, command: str) -> None:
    st.info(f"**{phase} has not been run yet.** Run `{command}` and reload this page.")


def _report_viewer(html_relative: str, label: str, height: int = 720) -> None:
    """
    Three ways to reach a generated report: its own page, inline, or the file.

    Streamlit serves the app from its own origin and does not expose the working
    directory over HTTP, so neither a relative `<a href>` nor a `file://` link
    can reach `reports/*.html` -- a browser blocks the second outright. Two
    routes are therefore built rather than linked. The report is published as a
    self-contained page into Streamlit's served `static/` directory, which gives
    it a real URL that opens in a new tab; and the same document is rendered
    into a sandboxed iframe for reading without leaving the page.
    """
    html = data.report_html(html_relative)
    if not html:
        st.caption(f"`{html_relative}` has not been generated yet.")
        return

    st.markdown(f"#### {label}")

    url = data.publish_report(html_relative)
    open_col, download_col, note_col = st.columns([1, 1, 2.4])
    with open_col:
        if url:
            st.link_button("Open as a page", url, width="stretch", type="primary")
        else:
            st.button("Open as a page", width="stretch", disabled=True)
    with download_col:
        st.download_button(
            "Download",
            data=data.read_bytes(html_relative),
            file_name=html_relative.rsplit("/", 1)[-1],
            mime="text/html",
            width="stretch",
        )
    with note_col:
        st.caption(
            f"Generated at `{html_relative}`. **Open as a page** loads it in a new tab as "
            "a standalone document; **Read inline** keeps you here. Every figure is "
            "embedded either way, so the page works on its own."
        )

    with st.expander("Read inline", expanded=False):
        components.html(html, height=height, scrolling=True)


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
    """A row of stat tiles, entering in sequence so the eye lands on the first."""
    columns = st.columns(len(items))
    for position, (column, item) in enumerate(zip(columns, items)):
        column.markdown(theme.tile(**item, delay=position), unsafe_allow_html=True)


def _gallery(figures: list, columns: int = 1) -> None:
    if not figures:
        st.caption("No figures; the phase may have been run with `--no-figures`.")
        return
    for index in range(0, len(figures), columns):
        for column, (title, path) in zip(st.columns(columns), figures[index : index + columns]):
            column.image(str(path), caption=title, width="stretch")


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
# Built with Altair rather than st.bar_chart so the palette is the *same* one
# `src/viz.py` gives the matplotlib figures these pages embed beside them. Two
# colour languages on one page reads as two products stapled together.
_AXIS_LABEL = alt.Axis(labelColor=theme.INK_SECONDARY, titleColor=theme.INK_SECONDARY,
                       gridColor=theme.GRID, domainColor=theme.AXIS, tickColor=theme.AXIS)


def _chart_base(frame: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(frame)
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=11)
        .configure_legend(labelFontSize=11, titleFontSize=11, labelColor=theme.INK_SECONDARY,
                          titleColor=theme.INK_SECONDARY)
    )


def _grouped_bars(frame: pd.DataFrame, x: str, y: str, color: str, title: str = "",
                  domain: list | None = None, height: int = 300) -> None:
    """Two series side by side -- the baseline-vs-improved shape, reused."""
    if frame.empty:
        return
    scale = alt.Scale(range=[theme.INK_MUTED, theme.BLUE])
    if domain:
        scale = alt.Scale(domain=domain, range=[theme.INK_MUTED, theme.BLUE])
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(f"{x}:N", title=None, axis=_AXIS_LABEL,
                    sort=list(dict.fromkeys(frame[x]))),
            y=alt.Y(f"{y}:Q", title=title or y, axis=_AXIS_LABEL),
            color=alt.Color(f"{color}:N", title=None, scale=scale),
            xOffset=f"{color}:N",
            tooltip=list(frame.columns),
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def _ranked_bars(frame: pd.DataFrame, label: str, value: str, title: str = "",
                 color: str | None = None, height: int = 320) -> None:
    """A horizontal ranking -- importances, coverage, error rates."""
    if frame.empty:
        return
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3,
                  color=color or theme.BLUE)
        .encode(
            x=alt.X(f"{value}:Q", title=title or value, axis=_AXIS_LABEL),
            y=alt.Y(f"{label}:N", title=None, sort="-x", axis=_AXIS_LABEL),
            tooltip=list(frame.columns),
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


# --------------------------------------------------------------------------
# 1. Overview
# --------------------------------------------------------------------------
def overview() -> None:
    st.markdown(
        theme.hero(
            theme.PRODUCT_CONTEXT,
            f"{theme.PRODUCT_NAME} — {theme.PRODUCT_TAGLINE}",
            "Profiling, multi-outcome prediction, competing-risk survival, anomaly "
            "detection, macro scenarios, explainability and a governed LLM copilot. "
            "Every figure on this page is read from a file the pipeline wrote; nothing "
            "is recomputed in the app.",
        ),
        unsafe_allow_html=True,
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
        required = frame[frame.Task == "Section 11"]
        present = int(required.Present.sum())
        st.markdown(
            theme.meter(
                "Section 11 deliverables present", present,
                display=f"{present} / {len(required)}",
                color=(theme.STATUS["good"] if present == len(required)
                       else theme.STATUS["warning"]),
                scale=len(required),
            ),
            unsafe_allow_html=True,
        )
        missing = required[~required.Present]
        if not missing.empty:
            st.markdown(
                theme.caveat(
                    "<b>Outstanding: "
                    + ", ".join(missing.Deliverable)
                    + ".</b> Listed here rather than omitted -- a checklist that shows "
                    "only what was produced cannot tell anyone what is still to do."
                ),
                unsafe_allow_html=True,
            )
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
            "make test         # the regression suite\n\n"
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
    st.markdown(
        theme.hero(
            "Task 1 · 15 points",
            "Data intelligence",
            "Column distributions, missingness patterns, outliers and invalid dates, "
            "cross-column relationship breaks, train-versus-test drift, servicer-feed "
            "reconciliation, and a record-level data-quality score that feeds both the "
            "feature matrix and the anomaly layer.",
        ),
        unsafe_allow_html=True,
    )

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

    drift = data.read_csv("reports/profiling/train_test_drift.csv")
    if not drift.empty and {"column", "psi"}.issubset(drift.columns):
        st.subheader("Where train and test diverge")
        st.caption(
            "Population stability index per feature. Conventional bands: below 0.10 stable, "
            "0.10-0.25 moderate, above 0.25 material. Hover for the underlying test."
        )
        top_drift = drift.dropna(subset=["psi"]).nlargest(15, "psi")
        _ranked_bars(top_drift, "column", "psi", title="PSI (train vs test)",
                     color=theme.ORANGE)

    st.subheader("Worst-scoring records")
    worst = data.read_csv("reports/profiling/worst_records.csv")
    _table(worst.head(15))

    st.markdown("---")
    _report_viewer(
        "reports/data_intelligence_report.html",
        "The complete Data Intelligence Report -- 10 sections, 20 tables",
    )


# --------------------------------------------------------------------------
# 3. Features and the time-aware split
# --------------------------------------------------------------------------
def features() -> None:
    st.markdown(
        theme.hero(
            "Phase 2 · feeds Task 2",
            "Features & the time-aware split",
            "What the models see, and the guarantee that none of it reads forward. "
            "Every feature carries an information window, and the split is purged by "
            "each target's own forward horizon.",
        ),
        unsafe_allow_html=True,
    )

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
    st.markdown(
        theme.hero(
            "Task 2 · 20 points",
            "Loan performance prediction",
            "Baseline versus improved across five targets -- 3- and 6-month delinquency, "
            "12-month default, 12-month prepayment and next state -- scored on a window "
            "strictly later than anything either model saw, with isotonic calibration "
            "and tuned, frozen thresholds.",
        ),
        unsafe_allow_html=True,
    )

    results = data.read_csv("reports/task2_model_results.csv")
    if results.empty:
        return _missing("Prediction", "make predict")

    metric_columns = [
        "target", "model", "backend", "n_features", "roc_auc", "pr_auc", "f1",
        "recall_at_precision_0.5", "macro_f1", "brier_uncalibrated", "brier_calibrated",
    ]
    available = [c for c in metric_columns if c in results.columns]

    st.subheader("Baseline vs improved")
    metric_choice = st.selectbox(
        "Compare on",
        [c for c in ["pr_auc", "roc_auc", "f1", "recall_at_precision_0.5",
                     "brier_calibrated", "macro_f1"] if c in results.columns],
    )
    chart_frame = results[["target", "model", metric_choice]].dropna()
    _grouped_bars(
        chart_frame, "target", metric_choice, "model",
        title=metric_choice, domain=["baseline", "improved"],
    )
    st.caption(
        "PR-AUC is the honest default on these base rates: ROC-AUC is dominated by the "
        "majority class nobody will action. Brier is lower-is-better -- the one metric "
        "here where a shorter bar is the good outcome."
    )
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
    st.markdown(
        theme.hero(
            "Task 3 · 15 points",
            "Time to event",
            "Default and prepayment as competing risks on a months-on-book clock. "
            "Cause-specific Cox against a constant-hazard and a Kaplan-Meier baseline, "
            "with censoring treated explicitly rather than dropped.",
        ),
        unsafe_allow_html=True,
    )

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

    st.markdown("---")
    _report_viewer("reports/survival_report.html",
                   "The complete Task 3 report, including the censoring treatment")


# --------------------------------------------------------------------------
# 6. Anomalies (Task 4) -- with the live threshold control
# --------------------------------------------------------------------------
def anomalies() -> None:
    st.markdown(
        theme.hero(
            "Task 4 · 10 points",
            "Anomaly & exception detection",
            "Deterministic rules for what is wrong, sequence-aware detectors for what a "
            "single row cannot see, and an Isolation Forest for what is merely unusual -- "
            "combined as a noisy-OR so a confident model cannot argue away a hard violation.",
        ),
        unsafe_allow_html=True,
    )

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
    if {"detector", "precision", "recall"}.issubset(ablation.columns):
        long = ablation.melt(
            id_vars="detector", value_vars=["precision", "recall"],
            var_name="metric", value_name="score",
        )
        _grouped_bars(long, "detector", "score", "metric",
                      title="score", domain=["precision", "recall"], height=320)
        st.caption(
            "Read the pair together. Rules alone are precise-ish and miss half the "
            "exceptions; adding sequence detectors takes recall to ~1.0 and leaves a large "
            "queue; the supervised head keeps the recall and cuts the queue six-fold."
        )
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

    st.markdown("---")
    _report_viewer("reports/anomaly_report.html", "The complete Task 4 report")


# --------------------------------------------------------------------------
# 7. Scenarios (Task 5) -- interactive
# --------------------------------------------------------------------------
def scenarios() -> None:
    st.markdown(
        theme.hero(
            "Task 5 · 10 points",
            "Scenario & stress simulation",
            "The Phase 3 models re-scored under base, adverse-credit and high-prepayment "
            "macro states. Two mechanical channels plus a credit channel that is solved "
            "for rather than assumed -- and reported where it saturates.",
        ),
        unsafe_allow_html=True,
    )

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

    st.markdown("---")
    _report_viewer("reports/scenario_report.html", "The complete Task 5 report")


# --------------------------------------------------------------------------
# 8. Explainability (Task 6)
# --------------------------------------------------------------------------
def explainability() -> None:
    st.markdown(
        theme.hero(
            "Task 6 · 10 points",
            "Explainability & responsible AI",
            "What drives the models, where they are wrong, and on whom. SHAP globals and "
            "locals, false positives and false negatives characterised rather than "
            "counted, calibration, and a disparity screen with a significance test.",
        ),
        unsafe_allow_html=True,
    )

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
        _ranked_bars(top, "feature", "share", title="share of |SHAP|", height=340)
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
        if not view.empty and {"group", "false_positive_rate"}.issubset(view.columns):
            _ranked_bars(view.nlargest(15, "false_positive_rate"), "group",
                         "false_positive_rate", title="false positive rate",
                         color=theme.ORANGE, height=300)
        _table(view.round(4), height=300)

    st.markdown("---")
    _report_viewer("reports/explainability_report.html", "The complete Task 6 report")


# --------------------------------------------------------------------------
# 9. Copilot (Task 7)
# --------------------------------------------------------------------------
def copilot() -> None:
    st.markdown(
        theme.hero(
            "Task 7 · 10 points",
            "LLM reviewer copilot",
            "Grounded summarisation, three classes of guardrail, six adversarial probes "
            "and a mandatory append-only audit trail. The copilot restates what the "
            "models produced; it never produces a prediction of its own.",
        ),
        unsafe_allow_html=True,
    )

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
        _render_generated_text(
            row.note,
            heading=f"{row.loan_id} · {row.reporting_month}",
            subtitle="Generated by the Phase 8 batch run.",
            released=bool(row.released_to_reviewer),
            verdict=str(row.guardrail_detail),
            model=str(row.model),
        )

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
        st.markdown("**What the model returned.**")
        _render_generated_text(str(row.llm_output))
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

        mix = (
            pd.DataFrame([{"model": r.get("model"), "status": r.get("status")} for r in calls])
            .value_counts()
            .reset_index(name="calls")
        )
        st.caption("Every call ever made, by model and outcome -- including the failures.")
        _ranked_bars(mix.assign(label=mix.model + "  ·  " + mix.status),
                     "label", "calls", title="logged calls", height=220)

    st.markdown("---")
    _report_viewer("reports/copilot_report.html", "The complete Task 7 report")


# --------------------------------------------------------------------------
# 10. Loan explorer -- the interactive centrepiece
# --------------------------------------------------------------------------
def explorer() -> None:
    st.markdown(
        theme.hero(
            "Demo · the interactive centrepiece",
            "Loan explorer",
            "Everything the pipeline concluded about one loan-month, read from the "
            "submission it actually wrote -- and, on demand, a grounded LLM reviewer "
            "note and verification checklist generated live against that same record.",
        ),
        unsafe_allow_html=True,
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

    st.markdown("**Probabilities at a glance**")
    st.markdown(
        theme.meter("3-month delinquency", row.prob_next_3m_delinquency,
                    display=f"{row.prob_next_3m_delinquency:.1%}", color=theme.AQUA)
        + theme.meter("6-month delinquency", row.prob_next_6m_delinquency,
                      display=f"{row.prob_next_6m_delinquency:.1%}", color=theme.AQUA)
        + theme.meter("12-month default", row.prob_next_12m_default,
                      display=f"{row.prob_next_12m_default:.1%}", color=theme.BLUE)
        + theme.meter("12-month prepayment", row.prob_next_12m_prepayment,
                      display=f"{row.prob_next_12m_prepayment:.1%}", color=theme.ORANGE)
        + theme.meter("Anomaly score", row.anomaly_score,
                      display=f"{row.anomaly_score:.3f}", color=theme.MAGENTA),
        unsafe_allow_html=True,
    )
    st.caption(
        "Bars are on a common 0-1 scale, so their lengths are comparable to each other "
        "rather than each being stretched to its own maximum."
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Top drivers**")
        st.markdown(f'<div class="note-box">{row.top_drivers}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("**Recommended action** · deterministic, from the rule layer")
        st.markdown(f'<div class="note-box">{row.action}</div>', unsafe_allow_html=True)

    if not queue.empty and loan_id in set(queue.loan_id):
        st.markdown("**This loan is in the curated reviewer queue**")
        _table(queue[queue.loan_id == loan_id])

    _copilot_panel(row, loan_id, month, queue)

    with st.expander("Full submission row"):
        st.json({k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()})


# --------------------------------------------------------------------------
# The explorer's copilot panel
# --------------------------------------------------------------------------
def _copilot_panel(row: pd.Series, loan_id: str, month: str, queue: pd.DataFrame) -> None:
    """
    Analyse this loan with the LLM, on demand.

    Two asks, one grounded context: a **reviewer note** that restates the
    record, and a **verification checklist** naming what a human should confirm.
    Both run the same guardrails as the batch pipeline and both land in the same
    append-only audit trail -- an interactive call that skipped either would be
    a governance hole opened for the sake of a demo button.

    The copilot stays strictly downstream: it is handed the probabilities, the
    anomaly score and the triggered rules as *inputs*. It is never asked what
    the loan will do.
    """
    st.markdown("---")
    live_available, _ = data.copilot_status()

    # The status sits on the description line rather than opposite the heading.
    # Floated to the right of a short title it read as a stray tag with nothing
    # balancing it; inline with the sentence it qualifies, it is read as part of
    # that sentence.
    st.subheader("Analyse with AI")
    st.markdown(
        theme.status_line(
            theme.live_badge(live_available),
            "The copilot summarises what the models already produced. It never predicts, "
            "never decides, and every response is checked for prediction language, "
            "decision language and numeric grounding before it is shown.",
        ),
        unsafe_allow_html=True,
    )

    definitions, rule_specs = data.grounding_sources()
    panel_record = data.panel_row(loan_id, month)

    if panel_record is None:
        st.info(
            "The loan's own record was not found in `data/loan_monthly_performance_test.csv`, "
            "so there is nothing to ground a note in. Run `make data` and `python main.py`."
        )
        return

    triggered = ""
    anomaly_extra: dict = {}
    if not queue.empty and loan_id in set(queue.loan_id):
        item = queue[queue.loan_id == loan_id].iloc[0]
        # `pd.isna` rather than a truthiness test: a queue row with no triggered
        # rules reads back from CSV as float NaN, which is *truthy*, so `or ""`
        # lets it through and `str()` turns it into a rule literally named "nan".
        # The model then faithfully reports that a rule called nan fired -- a
        # fabricated finding produced by correct, grounded behaviour.
        raw = item.get("triggered_rules")
        triggered = "" if pd.isna(raw) else str(raw)
        anomaly_extra = {"suggested_action_from_rules": item.get("suggested_action")}

    # The pipeline's own outputs, handed over as facts to restate.
    model_outputs = {
        "prob_next_3m_delinquency": row.prob_next_3m_delinquency,
        "prob_next_6m_delinquency": row.prob_next_6m_delinquency,
        "prob_next_12m_default": row.prob_next_12m_default,
        "prob_next_12m_prepayment": row.prob_next_12m_prepayment,
        "predicted_next_state": row.next_state,
        "confidence": row.confidence,
    }
    anomaly = {
        "anomaly_score": row.anomaly_score,
        "exception_required": row.exception_required,
        "exception_type": row.exception_type,
        "top_drivers": row.top_drivers,
        "suggested_action_from_rules": row.action,
        **anomaly_extra,
    }

    if not live_available:
        st.markdown(
            theme.caveat(
                "<b>No API key is configured, so these buttons will return a deterministic "
                "offline stub</b> marked as such, not a model response. Add a key to "
                "<code>.env</code> and reload to run this live. The guardrails, grounding and "
                "audit logging all execute either way."
            ),
            unsafe_allow_html=True,
        )

    batch = data.read_csv("reports/copilot/reviewer_notes.csv")
    if not batch.empty and loan_id in set(batch.loan_id):
        prior = batch[batch.loan_id == loan_id].iloc[0]
        with st.expander("The note Phase 8 already generated for this loan", expanded=False):
            _render_generated_text(
                prior.note,
                heading="Reviewer note",
                subtitle="From the batch pipeline, not this page.",
                released=bool(prior.released_to_reviewer),
                verdict=str(prior.get("guardrail_detail", "")),
            )
            st.caption(
                "Regenerating below makes a fresh call and appends a fresh record to the "
                "audit trail."
            )

    note_col, action_col, clear_col = st.columns([1, 1, 1])
    with note_col:
        want_note = st.button("Analyse with AI", width="stretch", type="primary")
    with action_col:
        want_action = st.button("Recommend action", width="stretch")
    with clear_col:
        if st.button("Clear", width="stretch"):
            for key in ("copilot_note", "copilot_action"):
                st.session_state.pop(key, None)

    key = f"{loan_id}·{month}"

    def _run(kind: str, generator, spinner: str) -> None:
        from src.copilot import notes as notes_module

        with st.spinner(spinner):
            try:
                result = generator(
                    notes_module, panel_record, definitions, rule_specs,
                    triggered, model_outputs, anomaly, not live_available,
                )
            except Exception as exc:  # noqa: BLE001 -- surfaced, never swallowed
                st.session_state[kind] = {"key": key, "error": f"{type(exc).__name__}: {exc}"}
                return
        st.session_state[kind] = {
            "key": key,
            # The model's own text, kept separate from the wrapped form. The
            # disclaimer is deliberately repeated top and bottom by
            # `guardrails.wrap` so it survives a copy-paste into a case file --
            # correct for a text artefact, but rendered on screen it is the
            # same sentence twice around three sentences of content. The UI
            # shows the body and states the disclaimer once, in its own right.
            "body": result.raw_output,
            "note": result.note,
            "released": bool(result.released),
            "verdict": result.verdict.summary(),
            "model": result.response.model,
            "status": result.response.status,
            "latency": result.response.latency_seconds,
            "call_id": result.response.call_id,
            "prompt": result.response.prompt,
        }

    if want_note:
        _run(
            "copilot_note",
            lambda m, r, d, s, t, mo, an, off: m.generate_note(
                r, d, s, triggered=t, model_outputs=mo, anomaly=an, offline=off
            ),
            "Assembling grounded context and calling the model...",
        )
    if want_action:
        _run(
            "copilot_action",
            lambda m, r, d, s, t, mo, an, off: m.generate_action(
                r, d, s, triggered=t, model_outputs=mo, anomaly=an, offline=off
            ),
            "Drafting verification steps from the same grounded context...",
        )

    _render_copilot_result(
        st.session_state.get("copilot_note"), key,
        "Reviewer note", "A plain-language summary of this record.",
    )
    _render_copilot_result(
        st.session_state.get("copilot_action"), key,
        "Suggested verification steps", "What to confirm before acting, and against what.",
    )


def _render_copilot_result(payload: dict | None, key: str, heading: str,
                           subtitle: str = "") -> None:
    """
    One generated response, as a card a reviewer can actually read.

    The body is rendered as markdown rather than dropped into a monospace
    block: the model answers in prose or in bullets, and a checklist displayed
    as preformatted text is a checklist nobody works through. The governance
    metadata -- verdict, disclaimer, call id, the prompt itself -- sits below
    the content in its own register, present and inspectable without competing
    with the thing the reviewer opened the panel to read.
    """
    if not payload or payload.get("key") != key:
        return

    with st.container(border=True):
        if payload.get("error"):
            st.markdown(f"**{heading}**")
            st.error(f"The call failed: {payload['error']}")
            st.caption(
                "The failure is in `reports/llm_prompt_log.jsonl` too -- a call that "
                "errored is exactly the one a reviewer will ask about."
            )
            return

        released = payload["released"]
        title, status = st.columns([2.2, 1])
        with title:
            st.markdown(f"**{heading}**")
            if subtitle:
                st.caption(subtitle)
        with status:
            # The model identifier is recorded against this call_id in the audit
            # trail and shown on the Task 7 page; it is not repeated here.
            st.markdown(
                theme.pill("released" if released else "WITHHELD BY GUARDRAILS",
                           "ok" if released else "bad")
                + theme.pill(f"{payload['latency']}s", "info"),
                unsafe_allow_html=True,
            )

        if not released:
            st.markdown(
                theme.caveat(
                    "<b>This response was withheld, and is shown below in full on "
                    "purpose.</b> A governance layer that quietly rewrote its model's "
                    "output would destroy the evidence that the model produces output "
                    "like this."
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="ai-body">{_markdown_body(payload["body"])}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            theme.disclaimer_note(
                "Recommendation, not a decision.",
                "Generated by an LLM from model output and loan data supplied to it. It "
                "contains no independent prediction and no credit decision. A human "
                f"reviewer is responsible for any action taken. Guardrails: "
                f"<b>{payload['verdict']}</b>.",
            ),
            unsafe_allow_html=True,
        )

        with st.expander("Show the grounded prompt and audit record"):
            st.caption(f"call_id `{payload['call_id']}` · appended to "
                       "`reports/llm_prompt_log.jsonl`")
            st.code(payload["prompt"], language="markdown")


def _unwrap_note(text: str) -> tuple[str, str]:
    """
    Split a stored note into its body and any guardrail banner.

    `guardrails.wrap` brackets the disclaimer above *and* below the note so it
    survives a copy-paste into a case file. Correct for the artefact; on screen
    it prints one long sentence twice around three sentences of content. Only
    lines that are wholly a bracketed wrapper are stripped -- the offline
    stub's own "[OFFLINE STUB -- not a model response] Loan ..." opener is
    content and stays.
    """
    def is_wrapper(line: str) -> bool:
        stripped = line.strip()
        return (
            stripped.startswith("[") and stripped.endswith("]")
            and ("RECOMMENDATION, NOT A DECISION" in stripped
                 or "GUARDRAIL FAILED" in stripped)
        )

    lines = (text or "").strip().splitlines()
    banner = ""
    while lines and (is_wrapper(lines[0]) or not lines[0].strip()):
        line = lines.pop(0).strip()
        if "GUARDRAIL FAILED" in line:
            banner = line.strip("[]")
    while lines and (is_wrapper(lines[-1]) or not lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip(), banner


def _render_generated_text(text: str, heading: str = "", subtitle: str = "",
                           released: bool | None = None, verdict: str = "",
                           model: str = "") -> None:
    """
    Any stored LLM output, rendered as content rather than as a file.

    The models answer in markdown -- bold labels, bulleted checklists -- and
    dropping that into a monospace block shows the reader the asterisks instead
    of the emphasis. One renderer for every place generated text is displayed,
    so a note reads the same on the copilot page as in the explorer.
    """
    body, banner = _unwrap_note(text)
    if not body:
        st.caption("No output recorded.")
        return

    with st.container(border=True):
        if heading:
            title, status = st.columns([2.2, 1])
            with title:
                st.markdown(f"**{heading}**")
                if subtitle:
                    st.caption(subtitle)
            with status:
                pills = ""
                if released is not None:
                    pills += theme.pill("released" if released else "WITHHELD",
                                        "ok" if released else "bad")
                if model:
                    pills += theme.pill(model, "info")
                if pills:
                    st.markdown(pills, unsafe_allow_html=True)

        if banner:
            st.markdown(theme.caveat(f"<b>{banner}</b>"), unsafe_allow_html=True)

        st.markdown(
            f'<div class="ai-body">{_markdown_body(body)}</div>', unsafe_allow_html=True
        )

        if released is not None:
            st.markdown(
                theme.disclaimer_note(
                    "Recommendation, not a decision.",
                    "Generated by an LLM from model output and loan data supplied to it. "
                    "It contains no independent prediction and no credit decision. A human "
                    "reviewer is responsible for any action taken."
                    + (f" Guardrails: <b>{verdict}</b>." if verdict else ""),
                ),
                unsafe_allow_html=True,
            )


def _markdown_body(text: str) -> str:
    """
    Render the model's markdown to HTML for the styled card.

    Streamlit's own `st.markdown` cannot be used here: it writes its own
    element, so it cannot sit inside a styled wrapper. The same `markdown`
    library the reports already depend on does the conversion, and falls back
    to escaped preformatted text if it is unavailable, because a reviewer note
    that renders as raw asterisks is still readable and a stack trace is not.
    """
    body = (text or "").strip()
    try:
        import markdown as md_lib

        return md_lib.markdown(body, extensions=["tables", "sane_lists"])
    except Exception:
        escaped = body.replace("&", "&amp;").replace("<", "&lt;")
        return f"<pre>{escaped}</pre>"


# --------------------------------------------------------------------------
# 11. Submission & reproducibility
# --------------------------------------------------------------------------
def submission_page() -> None:
    st.markdown(
        theme.hero(
            "Phase 9 · ML engineering",
            "Submission & reproducibility",
            "The graded file, and the checks run before it was written. The template is "
            "read at run time and treated as the binding contract, so a change the "
            "organiser makes surfaces as a validation failure rather than a silently "
            "wrong file.",
        ),
        unsafe_allow_html=True,
    )

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
        "make test         # the regression suite",
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
    st.markdown(
        theme.hero(
            "Task 8 · 5 points · section 11",
            "Model card & development log",
            "The model card generated from measured tables, and the AI development log "
            "kept per session rather than reconstructed at the end.",
        ),
        unsafe_allow_html=True,
    )

    tab_card, tab_log = st.tabs(["Model card", "AI development log"])
    with tab_card:
        card = data.read_text("reports/model_card.md")
        if not card:
            _missing("The model card", "python main.py --model-card")
        else:
            # The card's tables are the record; the charts beside them are the
            # same numbers read from the same CSVs the card was generated from,
            # so the two cannot disagree. A calibration table is precise and
            # slow to read -- three bars answer "are these probabilities honest"
            # at a glance, and the table is still there for the exact figure.
            document, charts = st.columns([1.45, 1], gap="large")
            with document:
                st.markdown(card)
            with charts:
                _model_card_charts()
    with tab_log:
        log = data.read_text("ai_dev_log/log.md")
        if not log:
            st.info("No development log found at `ai_dev_log/log.md`.")
        else:
            document, charts = st.columns([1.45, 1], gap="large")
            with document:
                st.markdown(log)
            with charts:
                _dev_log_charts(log)


def _model_card_charts() -> None:
    """The card's own tables, drawn -- read from the CSVs that generated it."""
    st.markdown("#### Results at a glance")
    st.caption(
        "The same numbers as the tables beside this, read from the same files. "
        "Charts for comparison; the tables for the exact figure."
    )

    calibration = data.read_csv("reports/explainability_report/calibration_summary.csv")
    if not calibration.empty and {"mean_predicted", "observed_rate"}.issubset(calibration.columns):
        st.markdown("**Calibration — do the probabilities mean what they say?**")
        long = calibration.melt(
            id_vars="model", value_vars=["mean_predicted", "observed_rate"],
            var_name="series", value_name="rate",
        )
        _grouped_bars(long, "model", "rate", "series",
                      title="rate", domain=["mean_predicted", "observed_rate"], height=230)
        st.caption(
            "Equal pairs mean a calibrated model: the average probability it assigns "
            "matches how often the event actually happened."
        )

    results = data.read_csv("reports/task2_model_results.csv")
    if not results.empty:
        st.markdown("**Baseline vs improved**")
        options = [c for c in ["pr_auc", "roc_auc", "f1", "brier_calibrated", "macro_f1"]
                   if c in results.columns]
        metric = st.selectbox("Metric", options, key="card_metric")
        _grouped_bars(results[["target", "model", metric]].dropna(), "target", metric,
                      "model", title=metric, domain=["baseline", "improved"], height=260)
        if metric == "brier_calibrated":
            st.caption("Brier is lower-is-better — the one metric here where a shorter "
                       "bar is the good outcome.")

    importance = data.read_csv("reports/explainability_report/global_importance.csv")
    if not importance.empty:
        st.markdown("**Top drivers**")
        model = st.selectbox("Model", sorted(importance.model.unique()), key="card_driver")
        _ranked_bars(importance[importance.model == model].head(8), "feature", "share",
                     title="share of |SHAP|", height=230)


def _dev_log_charts(log: str) -> None:
    """
    Task 8 asks for the AI-generated code share. The log states it per session;
    plotted, the trend is legible in a glance rather than by scrolling twelve
    sessions and holding the numbers in your head.

    Parsed rather than maintained separately, so the chart cannot drift away
    from the prose. A session whose line is phrased differently is skipped
    rather than guessed at.
    """
    import re

    sessions = re.findall(r"^##\s+Session\s+(\d+)\s+[—-]\s*(.+)$", log, flags=re.M)
    shares = re.findall(
        r"Approximate AI-generated code share this session:\*\*\s*~?(\d+)%", log
    )

    st.markdown("#### The log, in numbers")
    st.caption(
        "Read from `ai_dev_log/log.md` itself, so these cannot drift away from the "
        "prose beside them."
    )

    _tiles([
        {"label": "Sessions logged", "value": f"{len(sessions)}",
         "note": "kept per session, not reconstructed"},
        {"label": "Median AI share", "value":
            f"{int(np.median([int(s) for s in shares]))}%" if shares else "—",
         "note": "of lines drafted, 100% human-reviewed", "tone": "good"},
    ])

    if shares:
        frame = pd.DataFrame({
            "session": [f"S{i + 1}" for i in range(len(shares))],
            "ai_share": [int(s) for s in shares],
        })
        st.markdown("**AI-generated share by session**")
        chart = (
            alt.Chart(frame)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=theme.BLUE)
            .encode(
                x=alt.X("session:N", title=None, sort=list(frame.session), axis=_AXIS_LABEL),
                y=alt.Y("ai_share:Q", title="% of lines drafted by AI",
                        scale=alt.Scale(domain=[0, 100]), axis=_AXIS_LABEL),
                tooltip=["session", "ai_share"],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Every session also records 100% human review, the outputs that were "
            "rejected, and why — the share alone is not the evidence."
        )
