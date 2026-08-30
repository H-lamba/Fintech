# Architecture

> **Docs:** [Setup](setup.md) · [Architecture](architecture.md) · [Pipeline](pipeline.md) · [Module reference](api.md) · [Data](data.md) · [Outputs](outputs.md) · [Dashboard](dashboard.md) · [Results](results.md) · [Design decisions](design-decisions.md) · [Testing](testing.md)
>
> [← Back to the README](../README.md)

---

Nine phases, run in the order that resolves their one circular dependency: profiling
produces the record-level data-quality score the feature matrix consumes, and the
prediction run emits the feature dictionary the Data Intelligence Report folds in — so
profiling runs at both ends.

```mermaid
flowchart TD
    subgraph INPUT["data/ — the input pack"]
        RAW["loan_monthly_performance_train.csv<br/>loan_monthly_performance_test.csv<br/>loan_static_attributes.csv<br/>servicer_updates.csv"]
        REF["data_dictionary.md<br/>validation_rules.json<br/>macro_scenarios.csv<br/>submission_template.csv"]
    end

    P1["<b>Phase 1 · Profiling</b><br/>distributions · missingness · outliers<br/>rules · drift · reconciliation"]
    P2["<b>Phase 2 · Features</b><br/>59 features, each with<br/>a declared information window"]
    P3["<b>Phase 3 · Prediction</b><br/>baseline vs improved<br/>purged time-aware split · calibration"]
    P4["<b>Phase 4 · Survival</b><br/>cause-specific Cox<br/>competing risks · censoring"]
    P5["<b>Phase 5 · Anomaly</b><br/>rules + sequence detectors<br/>+ Isolation Forest — noisy-OR"]
    P6["<b>Phase 6 · Scenarios</b><br/>base · adverse-credit<br/>· high-prepayment"]
    P7["<b>Phase 7 · Explainability</b><br/>SHAP · error analysis<br/>calibration · disparity screen"]
    P8["<b>Phase 8 · LLM copilot</b><br/>grounded notes · guardrails<br/>· adversarial probes"]
    P9["<b>Phase 9 · Submission</b><br/>score the unlabelled panel<br/>validate against the template"]

    RAW --> P1
    REF --> P1
    P1 -->|"dq_scores_train.csv"| P2
    P2 --> P3
    P3 -->|"models/*.joblib"| P4
    P3 --> P5
    P3 -->|"fitted models"| P6
    P3 --> P7
    REF -->|"macro_scenarios.csv"| P6
    P5 -->|"reviewer queue"| P8
    P3 -->|"probabilities"| P8
    REF -->|"dictionary + rules<br/>(grounding)"| P8
    P2 -.->|"feature dictionary"| P1
    P3 --> P9
    P5 --> P9

    SUB["<b>submission/submission.csv</b><br/>78,409 × 13, validated"]
    REP["<b>reports/</b><br/>the graded deliverables<br/>md · html · csv · png"]
    UI["<b>app.py</b><br/>Streamlit dashboard<br/>reads files, recomputes nothing"]

    P9 --> SUB
    P1 --> REP
    P3 --> REP
    P4 --> REP
    P5 --> REP
    P6 --> REP
    P7 --> REP
    P8 --> REP
    REP --> UI
    SUB --> UI
```

**The one rule that shapes everything above:** the LLM sits strictly *downstream*. Phase 8
consumes Phase 3's probabilities and Phase 5's anomaly scores as inputs and restates them.
No arrow runs from Phase 8 back into a prediction.

### Module map

```mermaid
flowchart LR
    subgraph SHARED["shared"]
        CFG["config.py<br/><i>paths · seed · schema</i>"]
        IO["data_io.py<br/><i>tolerant loaders</i>"]
        VIZ["viz.py<br/><i>one chart palette</i>"]
    end

    subgraph PIPE["pipeline packages"]
        PROF["profiling/"]
        FEAT["features.py"]
        MOD["models/"]
        SURV["survival/"]
        ANOM["anomaly/"]
        SCEN["scenario/"]
        EXPL["explain/"]
        COP["copilot/"]
        SUB["submission/"]
    end

    subgraph SURFACE["surfaces"]
        DASH["dashboard/"]
        GEN["datagen/"]
    end

    CFG --> PIPE
    IO --> PIPE
    VIZ --> PIPE
    VIZ --> DASH
    GEN -->|"writes data/"| IO
    PIPE -->|"writes reports/"| DASH
```

Each package has its responsibilities documented in **[the module reference](api.md)**.


---

## Phase status

| Phase | Task | Points | Status |
| :--- | :--- | ---: | :--- |
| 0 — Repo & environment | — | 5 (ML Eng) | Done (`make all`, CI) |
| 1 — Data intelligence & profiling | Task 1 | 15 | Done (+ figures) |
| 2 — Feature engineering | — | (feeds Task 2) | Done (+ dictionary) |
| 3 — Loan performance prediction | Task 2 | 20 | Done |
| 4 — Survival / transition modeling | Task 3 | 15 | Done |
| 5 — Anomaly & exception detection | Task 4 | 10 | Done |
| 6 — Scenario & stress simulation | Task 5 | 10 | Done |
| 7 — Explainability | Task 6 | 10 | Done |
| 8 — LLM reviewer copilot | Task 7 | 10 | Done |
| 9 — Packaging & submission | — | — | Done |
| 10 — AI development log | Task 8 | 5 | Ongoing |

---

