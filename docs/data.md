# The data pack

> **Docs:** [Setup](setup.md) · [Architecture](architecture.md) · [Pipeline](pipeline.md) · [Module reference](api.md) · [Data](data.md) · [Outputs](outputs.md) · [Dashboard](dashboard.md) · [Results](results.md) · [Design decisions](design-decisions.md) · [Testing](testing.md)
>
> [← Back to the README](../README.md)


---

Everything the pipeline **reads**. Nothing here is written by a phase; the only thing that
writes to this folder is
[`scripts/generate_synthetic_suite.py`](pipeline.md#the-supporting-scripts).

**The pack is committed**, so a fresh clone can run the pipeline immediately without
generating anything first. It is also fully reproducible — see
[Regenerating](#regenerating) below.

---

## Why this data is synthetic

The challenge expects an organiser-provided pack. Rather than block on it, the pack is
generated here from a Markov transition engine, and that turned out to buy something a
real feed cannot give: **ground truth for the anomaly task**. Each data defect is injected
deliberately and its count recorded, so Phase 5's detection can be scored against what was
actually put there instead of against a human's guess at what looks wrong.

The cost is stated everywhere it matters: every metric in this repository measures whether
the pipeline is wired correctly, not how it would perform on a real servicer feed. Real
defects arrive partially and mixed with legitimate rarities; these were injected with
near-deterministic fingerprints.

**When the organiser's real pack arrives**, drop it in here and edit
[`src/config.py`](api.md#shared-modules) if the column names differ. The loaders
are deliberately tolerant — a missing column is reported by `data_io.schema_report` rather
than raised.

---

## The files

| File | Rows | Cols | What it is |
| :--- | ---: | ---: | :--- |
| `loan_monthly_performance_train.csv` | 268,125 | 34 | **The labelled panel.** One row per loan per month, 2017-01 → 2023-12, across 10,000 loans. Carries the monthly performance fields *and* the seven target columns. |
| `loan_monthly_performance_test.csv` | 78,409 | 27 | **The unlabelled panel**, 2024-01 → 2025-08. Same fields, no targets — this is what the submission scores. |
| `loan_static_attributes.csv` | 10,000 | 20 | Origination-level attributes, one row per loan: original balance, credit-score band, LTV band, DTI band, state, purpose, occupancy, property type, vintage. Joined onto the panel by `loan_id`. |
| `servicer_updates.csv` | 86,633 | 8 | **A second, partially conflicting source.** Used for source-conflict detection, stale-record logic and reconciliation in Phase 1, and reused as anomaly evidence in Phase 5. |
| `macro_scenarios.csv` | 144 | 8 | Base / adverse-credit / high-prepayment assumptions by projection month. **The only source of scenario assumptions** — Phase 6 invents no elasticities of its own. |
| `submission_template.csv` | 78,409 | 13 | The required output shape. Read at run time and treated as the **binding contract**, so a change the organiser makes surfaces as a validation failure rather than a silently wrong file. |
| `data_dictionary.md` | — | — | Plain-English definition of every field. Used for documentation *and* as retrieval grounding for the Phase 8 copilot. |
| `validation_rules.json` | 3 keys | — | Starter deterministic checks (`schema_version`, `standard`, `rules`). Phase 1 runs these as supplied and adds 14 domain rules of its own. |
| `_injected_anomalies.csv` | 4 | 2 | **The ground-truth ledger.** How many rows of each defect class the generator injected. Leading underscore because it would not exist in a real pack — it is the answer key, not an input. |

---

## The target columns

Present in the train panel only:

| Column | Meaning |
| :--- | :--- |
| `next_3m_delinquency_flag` | 30+ days past due within 3 months |
| `next_6m_delinquency_flag` | 30+ days past due within 6 months |
| `next_12m_default_flag` | Default within 12 months |
| `next_12m_prepayment_flag` | Prepaid within 12 months |
| `next_state` | Performance state at *t+1* (multiclass) |
| `exception_required` | Whether this record needs reviewer attention |
| `exception_type` | Which defect class, if any |

The forward window of each target drives the **purge gap** between time-ordered splits —
see [`src/models/splitting.py`](api.md#models--phase-3-task-2).

---

## The injected defect classes

| Class | Rows | Signature |
| :--- | ---: | :--- |
| **Balance Discrepancy** | 3,017 | `current_balance` exceeds origination, or moves without a recorded modification. Caught by row-level rules. |
| **Impossible State Transition** | 2,155 | A delinquency bucket skipped with no intermediate month. **Invisible to a single-row rule** — needs a sequence-aware detector. |
| **Time Travel** | 1,724 | `origination_month` after `reporting_month`, or other date-order violations. Caught by row-level rules. |
| **Zombie Loan** | 1,724 | Activity reported after a terminal state (Default / Prepaid). Also needs sequence awareness. |

That rules alone catch two of these four classes and only ~10% of the other two is the
finding Phase 5 is built around — it is why the detector layering exists, and it is
reported rather than smoothed over.

---

## Regenerating

```bash
python scripts/generate_synthetic_suite.py --loans 10000    # the default
python scripts/generate_synthetic_suite.py --loans 2000     # a smaller, faster pack
```

Generation is seeded from `src.config.RANDOM_SEED`, so the same command produces the same
pack. **Regenerating invalidates everything downstream** — models are fitted on the old
panel and reports quote its numbers. Re-run `python main.py` afterwards.

The pack **is committed to the repository** (~100 MB) so that a judge can clone and run
`python main.py` without a generation step, and so the numbers in the reports are checkable
against the exact bytes that produced them. Regenerating with the same `--loans` value
reproduces it.
