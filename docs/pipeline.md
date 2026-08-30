# The pipeline

> **Docs:** [Setup](setup.md) · [Architecture](architecture.md) · [Pipeline](pipeline.md) · [Module reference](api.md) · [Data](data.md) · [Outputs](outputs.md) · [Dashboard](dashboard.md) · [Results](results.md) · [Design decisions](design-decisions.md) · [Testing](testing.md)
>
> [← Back to the README](../README.md)

---

## What each phase writes

| Command | Output | Contents |
| :--- | :--- | :--- |
| `make profile` | `reports/data_intelligence_report.md` / `.html` | The graded Data Intelligence Report, with figures |
| | `reports/dq_scores_train.csv` | Record-level quality score + reason string |
| | `reports/profiling/*.csv` | Every intermediate table (20 of them) |
| | `reports/profiling/charts/*.png` | Missingness, distributions, drift, quality, rule violations |
| `make predict` | `reports/task2_model_results.csv` / `.md` | Baseline vs. improved, every target, every metric |
| | `reports/task2_split_audit.csv` | Evidence the split is time-aware and horizon-purged |
| | `reports/feature_dictionary.md` / `.csv` | Every feature, its family and its information window |
| | `reports/task2/*.csv` | Reliability, lift, per-class F1, importances |
| | `models/*.joblib` + `manifest.json` | Fitted models with thresholds and calibrators |
| `make survival` | `reports/survival_report.md` / `.html` | The graded Task 3 report, including the censoring note |
| | `reports/survival/*.png` / `*.csv` | Event curves and every table behind them |
| `make anomaly` | `reports/anomaly_examples.csv` | The curated reviewer queue (25 rows, all defect classes) |
| | `reports/anomaly_report.md` / `.html` | The graded Task 4 report |
| | `reports/anomaly_scores.csv` | Record-level hybrid score, predicted type and triggered rules |
| | `reports/anomaly/*.png` / `*.csv` | Ablation, signal coverage, per-class, driver importances |
| `make scenario` | `reports/scenario_report.csv` | Portfolio projection, one row per scenario-horizon |
| | `reports/scenario_report.md` / `.html` | The graded Task 5 report |
| | `reports/scenario/*.png` / `*.csv` | Segment tables, drivers, calibration, saturation |
| `make explain` | `reports/explainability_report.md` / `.html` | The graded Task 6 report |
| | `reports/model_card.md` | The section 11 model card, generated from measured tables |
| | `reports/explainability_report/*.png` | SHAP beeswarms, 9 local waterfalls, reliability, error rates |
| | `reports/explainability_report/*.csv` | Importances, errors by segment, disparity screen |
| `python main.py` | `submission/submission.csv` | The graded submission, validated against the template |
| | `reports/model_card.md` | The section 11 model card, generated from measured tables |
| | `reports/submission_validation.csv` | Every check run before the file was written |
| `make copilot` | `reports/llm_prompt_log.jsonl` | The mandatory audit trail, append-only |
| | `reports/copilot_report.md` / `.html` | The graded Task 7 report |
| | `reports/copilot/*.csv` | Notes, adversarial probe results, control failures |

Leakage and censoring controls are covered by regression tests rather than
convention — `make test` runs them, and CI runs them on every push:

```bash
make test
```

---


---

## The entry points

One script per phase. Each is a thin CLI wrapper: it parses arguments, calls into
[`src/`](api.md), prints progress, and writes that phase's outputs. **No
modelling logic lives here** — a function that both computes a result and decides where
to print it cannot be tested without also running its I/O.

Every script supports `--help`, and every one can be run on its own if the phases it
depends on have already produced their outputs.

```bash
python scripts/run_profiling.py --help
```

Normally you do not call these individually — [`main.py`](../main.py) runs them in the
right order as subprocesses, so a crash names the phase that failed.

---

### The phase scripts

| Script | Phase / Task | Needs first | Writes |
| :--- | :--- | :--- | :--- |
| `run_profiling.py` | Phase 1 · Task 1 | `data/` | `reports/data_intelligence_report.{md,html}`, `reports/dq_scores_train.csv`, `reports/profiling/**` |
| `run_prediction.py` | Phases 2–3 · Task 2 | Phase 1 | `reports/task2_*`, `reports/feature_dictionary.*`, `models/*.joblib`, `models/manifest.json` |
| `run_survival.py` | Phase 4 · Task 3 | `data/` | `reports/survival_report.{md,html}`, `reports/survival/**` |
| `run_anomaly.py` | Phase 5 · Task 4 | Phase 1 | `reports/anomaly_report.{md,html}`, `reports/anomaly_examples.csv`, `reports/anomaly_scores.csv`, `reports/anomaly/**` |
| `run_scenario.py` | Phase 6 · Task 5 | Phase 3 | `reports/scenario_report.{md,html,csv}`, `reports/scenario/**` |
| `run_explainability.py` | Phase 7 · Task 6 | Phase 3 | `reports/explainability_report.{md,html}`, `reports/model_card.md`, `reports/explainability_report/**` |
| `run_copilot.py` | Phase 8 · Task 7 | Phase 5 | `reports/copilot_report.{md,html}`, `reports/llm_prompt_log.jsonl`, `reports/copilot/**` |

#### Common options

| Flag | Effect |
| :--- | :--- |
| `--sample N` | Run on N loans, for a fast end-to-end check. |
| `--no-figures` | Skip chart rendering — much faster, and what CI uses. |
| `--help` | The full option set for that script. |

#### Script-specific options worth knowing

```bash
# Choose the gradient-boosting backend
python scripts/run_prediction.py --backend xgboost --score-test

# Override the split boundaries (defaults come from src/config.py)
python scripts/run_prediction.py --train-end 2021-06-01 --valid-end 2022-12-01

# Survival: change the vintage cutoff, or disable left-truncation handling
python scripts/run_survival.py --vintage-cutoff 2021-06-01 --no-left-truncation

# Scenarios: restrict the projection horizons
python scripts/run_scenario.py --horizons 12 24

# Copilot: OFFLINE is the default. --live spends API credits.
python scripts/run_copilot.py --offline --notes 4
python scripts/run_copilot.py --live --notes 6
python scripts/run_copilot.py --live --no-probes
```

**`run_copilot.py` defaults to `--offline` on purpose.** It sends loan data to a
third-party API and spends the account's credits, so a live run has to be asked for
explicitly rather than being what happens if you forget a flag. Offline mode exercises
the full assembly, guardrail and logging path with deterministic stubs, each marked
`provider: offline-stub` in the audit trail and never presented as a real generation.

---

### The supporting scripts

| Script | Responsibility |
| :--- | :--- |
| `generate_synthetic_suite.py` | Produces the entire input pack in [`data/`](data.md) — the panel, static attributes, servicer feed, data dictionary, validation rules, macro scenarios and submission template, with data defects injected and their ledger recorded. Wraps [`src/datagen/`](api.md#datagen--the-synthetic-benchmark-pack). |
| `capture_screenshots.py` | Regenerates the dashboard screenshots embedded in the root README. Optional extra — needs `pip install playwright && playwright install chromium`, which is deliberately *not* in `requirements.txt` so a judge reproducing the pipeline does not have to download a browser engine. |

```bash
# Regenerate the data pack (10,000 loans is the default)
python scripts/generate_synthetic_suite.py --loans 10000

# Regenerate the README screenshots
python scripts/capture_screenshots.py --launch
```

---

### Dependency order

If you run scripts individually rather than through `main.py`, this is the order that
resolves their dependencies:

```
generate_synthetic_suite.py
        ↓
run_profiling.py ──────────────┐   (writes the DQ score the feature matrix consumes)
        ↓                      │
run_prediction.py              │   (writes models/ and the feature dictionary)
        ↓                      │
   ┌────┼────┬─────────┬───────┘
   ↓    ↓    ↓         ↓
survival anomaly scenario explainability
             ↓
        run_copilot.py         (summarises the anomaly queue)
             ↓
   run_profiling.py again      (folds in the feature dictionary)
             ↓
      main.py --submission     (scores the unlabelled panel)
```

`run_survival.py` reads the panel directly and does not need Phase 3, so it can run at
any point after the data exists.
