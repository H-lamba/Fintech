# Testing

> **Docs:** [Setup](setup.md) · [Architecture](architecture.md) · [Pipeline](pipeline.md) · [Module reference](api.md) · [Data](data.md) · [Outputs](outputs.md) · [Dashboard](dashboard.md) · [Results](results.md) · [Design decisions](design-decisions.md) · [Testing](testing.md)
>
> [← Back to the README](../README.md)


---

```bash
make test                          # or: python -m pytest tests/ -q
python -m pytest tests/test_leakage_controls.py -v
```

**140 tests, run on every push by [CI](../.github/workflows/ci.yml).**

These are not here for coverage. They exist because this project makes a small number of
claims whose silent failure would invalidate every number in every report without raising
an error anywhere: that no split leaks, that no feature reads forward, that censored loans
are kept, that the LLM cannot smuggle in a prediction. A test suite is the only form in
which those claims stay true after the next refactor.

---

## The files

| File | What it pins |
| :--- | :--- |
| `test_leakage_controls.py` | **The disqualification conditions.** That the split is time-ordered and horizon-purged; that `audit_split` raises on overlapping windows; that a deliberately shuffled split *fails*; that no label-derived column reaches the feature matrix; that rebuilding features on a panel truncated after month *t* leaves every rolling feature at *t* unchanged. |
| `test_survival_censoring.py` | That censored loans contribute exposure and no event and are never dropped; that prepayment is treated as a **competing risk** rather than as censoring; that the Aalen-Johansen CIF stays below the naive `1 − KM`; that left truncation is handled on the loan-age axis. |
| `test_anomaly.py` | That each detector fires on the defect it targets; that the noisy-OR combination lets a high-severity rule set a floor the model cannot argue down; that the "no sequence information" ablation really drops `rule_score` rather than just its prefix. |
| `test_scenario.py` | That the stress channels apply in the right direction; that banded columns are rebuilt when the value underneath them moves; that stressed features are clipped to plausible ranges; that credit-channel saturation is detected rather than clamped and reported as an answer. |
| `test_explainability.py` | That SHAP sampling is stratified on the outcome; that error rates are computed on the *calibrated* probability at the deployed threshold; that the disparity screen applies its significance test and minimum group size. |
| `test_copilot.py` | That prediction and decision language are blocked; that ungrounded numbers are caught; that the audit trail is written on failure too; that every probe declares its trap and its correction; that a NaN `triggered_rules` cannot become a rule named `nan`; that `ACTION_TASK` never asks for a decision. |
| `test_submission.py` | That the submission matches the template's columns and order; that rows align by `(loan_id, reporting_month)` and never by position; that probabilities stay in [0, 1]; that values survive a CSV round trip; that **nothing is refitted on test**. |
| `test_dashboard.py` | That all twelve pages render through Streamlit's own test harness; that navigation follows the section 14 demo order; that a missing phase is reported rather than crashed; that report figures are inlined as data URIs; that the copilot status never leaks key material; that generated notes render as content rather than as a raw file. |

---

## Why `test_dashboard.py` exists

Nothing else in the suite imports the dashboard modules. Without this file the app could
break silently on a Streamlit upgrade and nobody would find out until the demo — which
nearly happened: a Streamlit release made `height=None` invalid and deprecated
`use_container_width`, and **ten of twelve pages raised** while the app still served
HTTP 200, because Streamlit renders an exception as a red box inside a successful
response.

Rendering every page through `streamlit.testing.v1.AppTest` found in one command what
clicking around would have found slowly and incompletely.

---

## Conventions

- **Tests are named for the property, not the function.**
  `test_a_record_with_no_triggered_rules_grounds_no_rules` says what must be true;
  `test_build_context` would not.
- **Each test's docstring says why the property matters** — usually the specific failure
  it prevents, often one that actually occurred. Several docstrings here are the incident
  report for a bug found in the browser or in a live LLM response.
- **Fixtures build small frames in memory** rather than reading `data/`, so the suite runs
  in ~15 seconds and does not need the pipeline to have been run.
- **A test that would pass on broken code is worse than no test.** The leakage tests
  assert the *failure* path too: shuffle the split and the audit must raise.
