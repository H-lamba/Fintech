# Outputs & deliverables

> **Docs:** [Setup](setup.md) · [Architecture](architecture.md) · [Pipeline](pipeline.md) · [Module reference](api.md) · [Data](data.md) · [Outputs](outputs.md) · [Dashboard](dashboard.md) · [Results](results.md) · [Design decisions](design-decisions.md) · [Testing](testing.md)
>
> [← Back to the README](../README.md)


---

Everything the pipeline writes. All of it is generated -- nothing in these folders
should be edited by hand.

---

## `reports/` — the graded deliverables

Everything a judge reads. **All of it is generated** — every file here is rewritten by
`python main.py`, so nothing in this folder should ever be edited by hand. If a number
looks wrong, the fix belongs in [`src/`](api.md), not here.

Each report is written twice: **markdown** (readable in the repo and on GitHub) and
**standalone HTML** (better for the demo, and for judges who would rather scroll than
clone). The dashboard serves the HTML at its own URL and can also render it inline.

---

### Top-level reports

| File | Task | What it contains |
| :--- | :--- | :--- |
| `data_intelligence_report.{md,html}` | Task 1 | The graded profiling report: 10 sections, 20 tables — distributions, missingness, outliers, rule breaks, correlations, drift, source conflicts, quality scoring, figures, and the feature dictionary folded in. |
| `task2_model_results.{md,csv}` | Task 2 | Baseline vs improved, every target, every metric, with thresholds, calibration method and fit times. |
| `survival_report.{md,html}` | Task 3 | Competing-risk survival, the censoring treatment, model comparison and event curves. |
| `anomaly_report.{md,html}` | Task 4 | Detector ablation, signal coverage, exception-type classification, driver analysis and the curated queue. |
| `scenario_report.{md,html,csv}` | Task 5 | Base / adverse / high-prepayment projections, method, saturation, segment impacts and driver attribution. |
| `explainability_report.{md,html}` | Task 6 | SHAP globals and locals, confusion, error rates by segment, calibration, the disparity screen. |
| `copilot_report.{md,html}` | Task 7 | Grounded notes, adversarial probes, the guardrail failures, the log summary. |
| `model_card.md` | Section 11 | Objective, data, features, model types, validation, metrics, limitations, leakage controls, known failure modes. Generated from the measured tables so it cannot drift from them. |
| `llm_prompt_log.jsonl` | Task 7 | **The mandatory audit trail.** Append-only, one JSON object per line: prompt, system prompt, model, timestamp, latency, tokens, output, guardrail verdict and human decision. Two records per call, joined on `call_id`. |
| `feature_dictionary.{md,csv}` | — | Every feature, its family, its **information window** and its definition. Generated from the matrix that was actually built. |
| `anomaly_examples.csv` | Task 4 | The 25-row reviewer queue — the explicit "20+ reviewer-ready examples" deliverable. |
| `submission_validation.csv` | Phase 9 | Every check run before `submission.csv` was written. |
| `batch_quality_summary.json` | Task 1 | The batch-level data-quality aggregate. |

### Large intermediate outputs

| File | Contents |
| :--- | :--- |
| `dq_scores_train.csv` | Record-level quality score plus a human-readable reason string, one row per panel row. Consumed as a **feature** by Phase 2. |
| `anomaly_scores.csv` | Record-level rule score, ML score, hybrid score, predicted exception type and triggered rules. |
| `task2_test_predictions.csv` | Calibrated probabilities for every target on the held-out window. |
| `task2_split_audit.csv` | Evidence the split is time-aware and horizon-purged: which months landed in each window, which were purged, and how many rows survived. |

### Sub-directories

| Directory | Contents |
| :--- | :--- |
| `profiling/` | Every intermediate profiling table (~20 CSVs) plus `charts/` — missingness, distributions, drift, quality scores, rule violations. |
| `task2/` | Per-target reliability curves, lift tables, feature importances, per-class F1 and the `next_state` confusion matrix. |
| `survival/` | Model comparison, hazard ratios, cumulative-incidence tables, calibration deciles, `censoring_summary.json`, and the event-curve figures. |
| `anomaly/` | Detector ablation, signal coverage, driver importances, and the precision-at-queue-size and driver-layer figures. |
| `scenario/` | Portfolio projection, record-level projection, per-segment tables (vintage, credit band, state, servicer), credit calibration and saturation, driver attribution, figures. |
| `explainability_report/` | SHAP beeswarms, nine local waterfalls, reliability and error-rate charts, global importance, disparity summary, fairness groups, confusion and confidence profiles. |
| `copilot/` | `reviewer_notes.csv`, `adversarial_probes.csv`, `control_failures.csv`, `log_summary.csv`. |

---

### Reading these as a judge

Start with `model_card.md` — it is the shortest complete account, and it leads with what
does **not** work. Then the report for whichever task you are scoring.

Three things are stated in the reports rather than buried, because they change how every
number should be read:

1. **The data is synthetic.** Every metric measures pipeline wiring, not forecasting skill.
2. **Task 4's near-perfect scores are a property of injected defects**, which carry
   near-deterministic fingerprints. What transfers is the layering, not the numbers.
3. **The prepayment head does not work** — ROC-AUC 0.52. Three phases reach that
   conclusion independently, and it is reported as-is rather than tuned until it looks
   better.

---

### Regenerating

```bash
python main.py                 # everything
make explain                   # just Task 6 + the model card
python main.py --model-card    # just the model card, from existing reports
```

**One caveat.** `llm_prompt_log.jsonl` is append-only *by design* — it is evidence, and a
crashed run must not be able to truncate it retroactively. But deleting `reports/` deletes
it along with everything else, and the live API call records in it cannot be regenerated
without spending credits again. If you clear this folder, preserve that file first.

---

## `models/` — fitted artifacts

Written by [`scripts/run_prediction.py`](pipeline.md) (Phase 3). **Generated, not
source** — this folder is gitignored, because a fitted model is reproducible output rather
than something to review in a diff.

If it is empty, run:

```bash
python scripts/run_prediction.py --score-test    # or: make predict
```

---

### The files

| File | What it holds |
| :--- | :--- |
| `<target>__baseline.joblib` | The logistic-regression baseline for that target, with its preprocessing pipeline. |
| `<target>__improved.joblib` | The LightGBM model, its calibrator and its tuned threshold, as one bundle. |
| `manifest.json` | What was fitted, from what, and how it should be used. |

One pair per target:

```
next_3m_delinquency_flag__{baseline,improved}.joblib
next_6m_delinquency_flag__{baseline,improved}.joblib
next_12m_default_flag__{baseline,improved}.joblib
next_12m_prepayment_flag__{baseline,improved}.joblib
next_state__{baseline,improved}.joblib          ← multiclass
```

**Both models are kept, not just the winner.** The baseline-vs-improved comparison is
graded, and a comparison you cannot re-run is an assertion.

---

### `manifest.json`

The manifest is what makes these files usable without re-reading Phase 3's source. For
each target it records the backend, the feature list, the calibration method, the tuned
threshold, the split boundaries the model was fitted under, and the seed.

**The threshold travels with the model on purpose.** 0.5 is meaningless on a reweighted
model with an 8% base rate — the exception head's probabilities stay compressed below
0.52, and a hardcoded 0.5 produced a `exception_required` rate of 0.00% on a book with a
2.6% base rate. Each threshold maximises F1 on the validation window and is frozen before
the test window is scored.

---

### How they are loaded

[`src/submission/inference.py`](api.md#submission--phase-9) loads these to score
the unlabelled panel, and [`src/scenario/`](api.md#scenario--phase-6-task-5)
loads them to re-score stressed portfolios. **Nothing refits on test** — a regression test
in [`tests/test_submission.py`](testing.md) asserts it.

The calibrator is part of the bundle, so a caller gets calibrated probabilities without
having to know that isotonic regression sits on top of the booster.

---

### Reproducibility

One seed (`src.config.RANDOM_SEED`) is used everywhere, so re-running Phase 3 on the same
data pack produces the same models. **Regenerating `data/` invalidates these** — the
models were fitted on the previous panel. Re-run `python main.py` after any regeneration.

---

## `submission/` — the graded output

One file: **`submission.csv`**, 78,409 rows × 13 columns, matching
[`data/submission_template.csv`](data.md) exactly.

```bash
python main.py                 # every phase, ending here
python main.py --submission    # just inference + submission (needs trained models)
```

---

### The columns

| Column | What it holds |
| :--- | :--- |
| `loan_id`, `reporting_month` | The key. Rows are aligned to the template by this pair, **never by position**. |
| `prob_next_3m_delinquency` | Calibrated probability, [0, 1] |
| `prob_next_6m_delinquency` | Calibrated probability, [0, 1] |
| `prob_next_12m_default` | Calibrated probability, [0, 1] |
| `prob_next_12m_prepayment` | Calibrated probability, [0, 1] |
| `next_state` | Predicted performance state at *t+1* |
| `exception_required` | 0/1 — from the supervised exception head at its tuned threshold |
| `exception_type` | Which defect class, or `No exception` |
| `anomaly_score` | The hybrid rule + ML score, [0, 1] |
| `top_drivers` | Why this record was flagged |
| `action` | The deterministic, rule-derived reviewer action |
| `confidence` | Model confidence in the predicted next state |

---

### Validation runs before the file is written

A structural failure **refuses the write**. Every check exists because its failure would
be invisible in a spot check:

- **Columns in the wrong order** — correct on inspection, wrong to a scorer that joins on
  position.
- **A probability of 1.4** — nothing in a spreadsheet flags it.
- **A short row set** — missing exactly the loans whose history was incomplete.
- **A CSV round trip** — a cell holding the string `"None"` passes every in-memory null
  test and comes back as `NaN` the moment anyone opens the file.

The full check list lands in
[`reports/submission_validation.csv`](outputs.md) on every run.

---

### Two corrections worth recording

**`exception_required` fired on 13.8% of the book** against a 2.6% base rate, because it
was the raw "any rule fired" flag — almost all of it one low-severity document check.
Replaced with the supervised head's judgement.

**Then it fired on 0.00%.** The head's probabilities stay compressed below 0.52: the
sequence detectors are near-perfect indicators, so average precision saturates within ten
boosting rounds and early stopping correctly halts. The model ranked fine — the hardcoded
0.5 threshold was wrong. Thresholds are now tuned on a held-back window and travel with
the model.

Both produced a **structurally perfect file that passed every validation check**. Neither
was findable by reading code or running tests; comparing the output's own summary
statistics against a base rate found both in seconds.

---

### Detection against ground truth

Predicted exception counts on the unlabelled panel match the generator's injection ledger
exactly — zero error on all four defect classes. Read it as an **end-to-end wiring
check**, not a performance claim: each class carries a near-deterministic fingerprint
because it was injected. Real servicing errors are not this separable.

---

## `ai_dev_log/` — the AI Development Log

**Task 8, Agentic Coding Evidence — 5 points.** A required deliverable.

| File | What it is |
| :--- | :--- |
| `log.md` | The log itself. Fourteen dated sessions, kept incrementally as the work happened rather than reconstructed at the end. |

---

### What Task 8 asks for, and where it is

| Required | Where |
| :--- | :--- |
| AI tools used | The **Tools used** table at the top — tool, model, and what each was used for. |
| Representative prompts | Per session, under **Goal** and **Accepted** — the illustrative ones, not all of them. |
| Accepted / rejected outputs | Every session has an **Accepted** and a **Rejected / corrected** block, with the reason. |
| Human review process | A **Human review** line per session describing what was checked and how. |
| Approximate AI-generated code share | Stated per session (~90–95% drafted, 100% human-reviewed). Charted in the dashboard. |
| Lessons learned | The **Lessons so far** list at the end — 22 of them, each tied to a specific incident. |

---

### How to read it

The **Rejected / corrected** blocks are the substance. An AI development log that only
records what was accepted is a changelog; the value is in the outputs that were wrong and
how they were caught. Several entries are incident reports for bugs that no test would
have found:

- A metric that was wrong in a plausible-looking direction (multiclass log-loss of 8.72 —
  scikit-learn binding probability columns to sorted labels while the pipeline held them
  in severity order).
- Survival curves that ran flat to month 59 because the risk set had emptied, saying the
  opposite of the truth while nothing errored.
- An LLM audit trail cut from 208 records to 20 by a successful pipeline run.
- A copilot faithfully reporting that a validation rule named `nan` had fired — a
  fabricated finding produced by correct, grounded behaviour.
- A sidebar at 1.03:1 contrast, invisible to every reader whose system was in dark mode
  and to none whose system was not.

**Lessons 1, 17, 18 and 19 are the ones worth reading first** if you only read four
lines: check the AI's numbers rather than its code; evidence can be destroyed by a
successful run; "nothing is untrue" is not "nothing is fabricated"; and test the
environment you are graded in, not the one CI is cheapest in.

---

### Keeping it current

Add a session entry whenever a meaningful piece of work lands. The format each session
follows:

```markdown
## Session N — <what it was about>

**Date:** YYYY-MM-DD
**Goal:** <what was being attempted>

**Accepted:** <what was kept, and why it is right>
**Rejected / corrected:** <what was wrong, how it was caught, what replaced it>
**Verification performed:** <tests added, runs made, things checked by hand>
**Human review:** <what a human actually checked, as opposed to what was generated>
**Approximate AI-generated code share this session:** ~N% drafted by AI, 100% reviewed.
```

The dashboard's **Model card & log** page parses the per-session share out of this file to
chart it, so keeping the wording of that last line consistent keeps the chart accurate. A
session phrased differently is skipped rather than guessed at.
