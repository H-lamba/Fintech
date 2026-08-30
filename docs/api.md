# Module reference

> **Docs:** [Setup](setup.md) · [Architecture](architecture.md) · [Pipeline](pipeline.md) · [Module reference](api.md) · [Data](data.md) · [Outputs](outputs.md) · [Dashboard](dashboard.md) · [Results](results.md) · [Design decisions](design-decisions.md) · [Testing](testing.md)
>
> [← Back to the README](../README.md)


---

All the logic lives here. Nothing in this folder is executed directly: every module is
imported by an entry point in [`scripts/`](pipeline.md), by
[`main.py`](../main.py), or by [`app.py`](../app.py). That split is deliberate — a
function that both computes a result and decides where to print it cannot be tested
without also running its I/O.

```
src/
├── config.py          ← one place for paths, the seed, and the expected schema
├── data_io.py         ← tolerant loaders; nothing raises on a missing column
├── features.py        ← the feature matrix and its generated dictionary
├── viz.py             ← one validated chart palette for every figure
│
├── profiling/         Phase 1 — Task 1
├── models/            Phase 3 — Task 2
├── survival/          Phase 4 — Task 3
├── anomaly/           Phase 5 — Task 4
├── scenario/          Phase 6 — Task 5
├── explain/           Phase 7 — Task 6
├── copilot/           Phase 8 — Task 7
├── submission/        Phase 9 — packaging
├── dashboard/         the Streamlit app's pages and data access
└── datagen/           generates the synthetic benchmark pack in data/
```

---

## Shared modules

| File | Responsibility |
| :--- | :--- |
| `config.py` | Every path, the single `RANDOM_SEED`, the expected column names, the data-quality weights, the split boundaries, and the per-target forward horizons. **The one file to edit when the organiser's real data pack lands** and column names differ. |
| `data_io.py` | Loaders for the panel, static attributes, servicer feed, data dictionary and validation rules. Deliberately *tolerant*: a missing column is reported by `schema_report` rather than raised, so a schema surprise appears as a report line instead of a crash mid-pipeline. |
| `features.py` | Builds the 59-feature matrix in blocks (static, contemporaneous, rolling, quality) and emits the feature dictionary. `assert_no_leaky_features` hard-fails on any label-derived column — the leakage control is code, not convention. |
| `viz.py` | The colour tokens, chart defaults and helpers every figure draws from. Categorical hues are assigned in fixed order and never cycled; status colours always ship with a text label, because hue alone is not readable for roughly one reader in twelve. |

---

## `profiling/` — Phase 1, Task 1

Everything that happens *before* a model is trained.

| File | Responsibility |
| :--- | :--- |
| `distributions.py` | Per-column dtype, cardinality, summary statistics and category frequencies. |
| `missingness.py` | Missing rate per column, co-occurrence between columns, and **structural** missingness detection — `loss_severity_band` is blank for every non-defaulted loan by design, and counting that as a defect drags nearly every record below 100. |
| `outliers.py` | Tukey IQR and robust-z outliers, invalid date relationships (origination after reporting), and staleness against `last_updated_at`. |
| `rules.py` | Runs `validation_rules.json` as supplied, plus 14 custom domain rules. Each rule checks `applicable()` before evaluating, so a missing column disables a rule instead of breaking the run. |
| `relationships.py` | Spearman correlation for numerics, Cramér's V for categoricals, correlation ratio for mixed pairs, and near-duplicate/redundant field detection — which is also how an accidental target leak would first surface. |
| `drift.py` | PSI and KS per feature between train and test, plus within-train temporal drift. |
| `reconciliation.py` | Compares the servicer feed against the main panel: source conflicts, stale records, duplicate panel rows. Reused as evidence by Phase 5. |
| `quality_score.py` | The record-level score (`100 * exp(-penalty/10)`) and the batch aggregate, each with a human-readable reason string so the number is explainable to a reviewer. |
| `figures.py` | Missingness, distribution, drift, quality and rule-violation charts. |
| `report.py` | `ReportBuilder` — accumulates sections and renders markdown + standalone HTML. **Used by every phase's report**, not just this one. |

## `models/` — Phase 3, Task 2

| File | Responsibility |
| :--- | :--- |
| `splitting.py` | The time-aware, horizon-**purged** train/validation/test split. `random_split` deliberately does not exist here, and `audit_split` raises if the windows it is handed overlap in time. |
| `estimators.py` | Model factories: the logistic-regression baseline and the LightGBM / XGBoost / HistGB improved models, with class reweighting. |
| `calibration.py` | Platt and isotonic calibration fitted on validation with the base model frozen, plus the reliability tables. |
| `evaluation.py` | Every Task 2 metric — ROC-AUC, PR-AUC, F1, recall-at-fixed-precision, Brier before and after calibration, macro-F1 — and the baseline-vs-improved comparison table. |
| `predict.py` | Phase 3 orchestration: build features, split, fit baseline and improved, calibrate, tune thresholds, persist to `models/`, score the test window, write the reports. |

## `survival/` — Phase 4, Task 3

| File | Responsibility |
| :--- | :--- |
| `dataset.py` | Turns the monthly panel into duration/event records on a months-on-book clock, with the full censoring taxonomy (administrative vs lost-to-follow-up) and left-truncation support. |
| `baselines.py` | The two references a covariate model must beat: constant hazard and Kaplan-Meier, plus the Aalen-Johansen cumulative incidence function. |
| `models.py` | Cause-specific Cox models and the competing-risk CIF assembly. |
| `evaluation.py` | Harrell's concordance, IPCW and integrated Brier scores, calibration by risk decile, proportional-hazards diagnostics. |
| `curves.py` | Event-curve figures, each truncated at the month its own risk set thins — otherwise a cohort with short follow-up reads as "these loans stopped defaulting". |
| `report.py` | The Task 3 report, including the censoring explanation the rubric asks for explicitly. |

## `anomaly/` — Phase 5, Task 4

| File | Responsibility |
| :--- | :--- |
| `signals.py` | Row-level rules plus **sequence-aware** detectors — an expression evaluated against one row cannot see last month's status, which is why rules alone catch only ~10% of Impossible State Transitions. Severity scoring lives here. |
| `features.py` | The hybrid feature matrix and the column sets each ablation is allowed to see. |
| `models.py` | Isolation Forest, the supervised exception heads, and the **noisy-OR** combination: `hybrid = 1 - (1 - rule)(1 - ml)`, so a fired high-severity rule sets a floor the model cannot argue down. |
| `explain.py` | Per-row tree contributions and robust deviation drivers, so every flagged record carries its reasons. |
| `curation.py` | The stratified reviewer queue — a guaranteed block per defect type plus slots reserved for high-scoring records with *no* rule violation, the only rows that can teach the rule set something. |
| `evaluation.py` | The detector ablation table, precision@k, and per-class metrics. |
| `figures.py` | Precision-at-queue-size curves and the driver layer chart. |
| `report.py` | The Task 4 report. |

## `scenario/` — Phase 6, Task 5

| File | Responsibility |
| :--- | :--- |
| `macro.py` | Validated ingest of `macro_scenarios.csv`. **The only source of scenario assumptions** — nothing downstream invents an elasticity. |
| `stress.py` | The three channels: house prices → LTV (arithmetic), market rate → refinance incentive, and a credit shift that is *solved for* rather than assumed. Rebuilds banded columns whenever the value underneath moves, and clips every stressed feature to a plausible range. |
| `project.py` | Dual-method projection (feature-stress and stated-multiplier), segment cuts, and the saturation summary. |
| `drivers.py` | Contribution-delta attribution — each feature's share of the change in the rate — and the generated narratives. |
| `figures.py` | Projection paths, segment impact, driver bars. |
| `report.py` | The Task 5 report. |

## `explain/` — Phase 7, Task 6

| File | Responsibility |
| :--- | :--- |
| `shap_values.py` | `TreeExplainer` with `tree_path_dependent` perturbation, outcome-stratified sampling, and local-example extraction. |
| `errors.py` | False positive / false negative isolation at the deployed threshold, segment error rates, and reliability curves — all on the *calibrated* probability, because that is what a borrower experiences. |
| `fairness.py` | The disparity screen: a two-proportion significance test, a minimum group size, and suppression of risk factors and uninterpretable screens. A screen that flags everything is one nobody reads. |
| `figures.py` | Beeswarms, local waterfalls, reliability and error-rate charts. |
| `report.py` | The Task 6 report and the model card body. |

## `copilot/` — Phase 8, Task 7

The LLM layer. It sits **strictly downstream** of every model in this repository.

| File | Responsibility |
| :--- | :--- |
| `llm_client.py` | Provider auto-detection from the key prefix (`gsk_` → Groq, `xai-` → xAI, `sk-` → OpenAI), retry with backoff on transient failures only, offline stub mode, and the **append-only audit log** — written before the call returns, including on failure. |
| `retrieval.py` | Grounded context assembly: the loan's record, its dictionary definitions, the rules that fired and the models' outputs. `grounded_numbers()` builds the set of every figure the model is allowed to quote. Holds both task prompts (`NOTE_TASK`, `ACTION_TASK`). |
| `guardrails.py` | The three checks run on every response: prediction language, decision language, and **numeric grounding**. A failing note is withheld with the failure stated, never silently repaired. |
| `notes.py` | `generate_note` (a reviewer summary) and `generate_action` (verification steps), plus batch generation and release gating. |
| `failures.py` | Six adversarial probes — prediction, hallucination, decision, false-premise, authority and vagueness bait — and the record of times the *guardrail* was wrong. |
| `report.py` | The Task 7 report. |

## `submission/` — Phase 9

| File | Responsibility |
| :--- | :--- |
| `inference.py` | Scores the unlabelled panel using the persisted models. Nothing is refitted on test — a regression test asserts it. |
| `build.py` | Assembles the submission, aligns rows to the template by `(loan_id, reporting_month)` rather than by position, and validates before writing. A structural failure refuses the write. |
| `model_card.py` | Generates the model card from the measured tables, so it cannot drift from the metrics it quotes. |

## `dashboard/` — the Streamlit app

| File | Responsibility |
| :--- | :--- |
| `data.py` | Cached loaders over the pipeline's own outputs, report inlining and publishing for the in-app viewer, and the copilot's live/offline status. **Never computes a metric.** |
| `theme.py` | The product identity, the CSS (including the motion, all of it wrapped in `prefers-reduced-motion`), and the tile / meter / pill / badge helpers. Shares its palette with `viz.py`. |
| `pages.py` | The twelve pages, ordered to match the demo flow, plus the shared table, chart, report-viewer and generated-text renderers. |

## `datagen/` — the synthetic benchmark pack

Run by [`scripts/generate_synthetic_suite.py`](pipeline.md) to produce
everything in [`data/`](data.md).

| File | Responsibility |
| :--- | :--- |
| `config.py` | Generation parameters: portfolio size, date range, band distributions, hazard multipliers. |
| `static_attributes.py` | Origination-level attributes — credit score, LTV, DTI, state, purpose, occupancy, property type, vintage. |
| `markov_engine.py` | The monthly state-transition engine whose hazards depend on credit band and LTV band. This is what makes the panel behave like a loan book. |
| `targets.py` | Derives the forward-looking labels (`next_3m_delinquency_flag` and the rest) from the realised state paths. |
| `anomalies.py` | Injects the four defect classes — Balance Discrepancy, Impossible State Transition, Time Travel, Zombie Loan — and records the ledger that detection is scored against. |
| `artifacts.py` | Writes the supporting pack: data dictionary, validation rules, macro scenarios, submission template. |
| `pipeline.py` | Orchestrates the five generation stages and writes `data/`. |

---

## Conventions used throughout

- **One seed.** `config.RANDOM_SEED`, imported everywhere. No module seeds itself.
- **Explicit encoding on every file operation.** Every `read_text` / `write_text` / `open`
  passes `encoding="utf-8"`. Inheriting the platform default means the pipeline works on
  Linux and dies on Windows the moment a report contains a typographic character.
- **Reports are built, not printed.** Every phase assembles a `ReportBuilder` and saves
  markdown + HTML, so the same content reaches the repo, the browser and the dashboard.
- **No module writes outside its phase's output directory.**
- **Docstrings carry the reasoning.** Where a choice has a rejected alternative, the
  docstring says what was rejected and why.
