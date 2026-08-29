# Loan Performance Intelligence Engine — Execution Plan
### Intain Campus FinTech Challenge 2026 (AI Track)

Scoring is out of 100. Every phase below is tagged with the criterion it earns points on, so nothing gets built that doesn't move the score. Order matters: later phases (survival modeling, explainability, LLM copilot) all consume outputs from earlier phases, so skipping ahead just creates rework.

Total points map:
Data Intelligence 15 | Predictive Modeling 20 | Time-to-Event 15 | Anomaly/Exception 10 | Scenario 10 | Explainability/RAI 10 | LLM Usage 10 | ML Engineering 5 | Agentic Evidence 5

---

## Phase 0 — Repo Skeleton & Environment (supports ML Engineering, 5 pts)

- Create the GitHub repo now, not at the end. Structure: `/data`, `/notebooks`, `/src` (pipeline, features, models, eval), `/reports`, `/ai_dev_log`, `/submission`.
- Pin a `requirements.txt` / `environment.yml`. Decide once: pandas/polars, scikit-learn, lightgbm/xgboost, lifelines or scikit-survival (for Task 3), shap, optuna (optional), mlflow (optional, advanced feature).
- Set a fixed `RANDOM_SEED` used everywhere.
- Start the **AI Development Log** file today (`ai_dev_log/log.md`) and add an entry every time you use an AI tool from here on — this is graded (5 pts) and is much cheaper to do incrementally than reconstruct later.
- Start the README with a "how to reproduce" section stub; fill it in as each phase lands.

Deliverable check-off: empty-but-runnable pipeline skeleton, README stub, AI dev log started.

---

## Phase 1 — Data Intelligence & Profiling (15 pts)

Work only on `loan_monthly_performance_train.csv`, `loan_static_attributes.csv`, `servicer_updates.csv`, guided by `data_dictionary.md` and `validation_rules.json`.

1. **Schema & distribution profiling**: per-column dtype, cardinality, distribution plots for numeric fields, category frequency for categoricals.
2. **Missingness**: missing-value rate per column, and whether missingness is random vs. structured (e.g., missing `days_past_due` correlated with `current_status`). A missingness heatmap by column and by loan_age_months bucket is a strong, cheap visual.
3. **Outlier & invalid-date detection**: e.g. `origination_month` after `reporting_month`, negative balances, `current_balance > original_balance` without a valid modification, `remaining_term_months` inconsistent with `loan_age_months` + original term.
4. **Cross-column relationship breaks**: run `validation_rules.json` as-is, then add 5–10 of your own consistency rules (e.g. `default_flag=1` but `days_past_due` low; `prepayment_flag=1` but `current_balance>0`).
5. **Correlation / dependency analysis**: correlation matrix for numeric features, association measures (Cramér's V or similar) for categoricals, and flag near-duplicate/redundant fields.
6. **Train vs. test drift**: population stability index (PSI) or KS-test per feature between train and test; also check `servicer_updates.csv` against the main file for conflicting values (this doubles as reconciliation logic reused in Task 4).
7. **Data-quality scoring**: produce both a **record-level** score (weighted rule violations + missingness + outlier flags) and a **batch-level** score (aggregate). This score becomes an input feature later and also feeds the anomaly task.

Deliverable: `reports/data_intelligence_report.{md,html}` with the profiling, missingness, outliers, drift, and top issues — this is graded output #4 directly, and demo-flow steps 2–3.

---

## Phase 2 — Feature Engineering (feeds Predictive Modeling, 20 pts)

- Loan-level static features (credit band, LTV band, DTI band, state, purpose, occupancy, property type, vintage) one-hot/ordinal encoded per your model choice.
- Time-varying/rolling features per loan: trend in `days_past_due` over last 3/6 months, balance paydown rate, number of status changes, months since last delinquency, servicer-transfer flags, document-status changes.
- The Phase-1 data-quality score as a feature (and separately kept out of leakage-sensitive targets — see below).
- **Leakage guardrails, written down explicitly**: no feature computed using information only available after the label's future window (e.g. don't use month t+1's `current_status` to predict a t+3 label). Document this in the model card — judges explicitly penalize leakage.

Deliverable: `src/features.py` + a short feature dictionary appended to the data intelligence report.

---

## Phase 3 — Loan Performance Prediction (20 pts — the single biggest line item)

1. **Time-aware split**: split by `reporting_month`/`origination_month` (e.g. train on earlier vintages/months, validate/test on later ones), never a random row split — call this out explicitly in the model card, since random splits are an explicit disqualification/low-score trigger.
2. **Targets**: `next_3m_delinquency_flag`, `next_6m_delinquency_flag`, `next_12m_default_flag`, `next_12m_prepayment_flag`, `next_state` (multiclass).
3. **Baseline model** (logistic regression or simple gradient boosting with minimal features) then an **improved model** (LightGBM/XGBoost with full feature set) — judges want to see the comparison, so keep both, don't just ship the final one.
4. **Class imbalance**: class weights, or threshold tuning, or SMOTE-style resampling — pick one, justify it.
5. **Calibration**: Platt scaling or isotonic regression on top of the improved model; report Brier score before/after.
6. **Metrics**: ROC-AUC, PR-AUC, F1, recall-at-fixed-precision, Brier score, macro-F1 for `next_state`. Report all of them, per target, baseline vs improved, in a single results table.

Deliverable: `src/models/predict.py`, results table in README or model card, demo-flow steps 6–7.

---

## Phase 4 — Time-to-Event / Survival / Transition Modeling (15 pts)

- Pick one: Cox proportional hazards or Kaplan-Meier survival curves for time-to-default/prepayment (via `lifelines`), or a discrete-time monthly transition/Markov model between `current_status` states.
- Handle censoring explicitly (loans still active/unresolved at data cutoff) — write a paragraph on how you treated it, since judges explicitly want this explained.
- Produce event/cumulative-incidence curves, segment them (e.g. by credit band or vintage) since that reuse feeds Phase 6 scenario segmentation too.
- Compare against a naive baseline (e.g. constant hazard rate) to show the model adds value.

Deliverable: `reports/survival_report` with curves, demo-flow step 8.

---

## Phase 5 — Anomaly & Exception Detection (10 pts)

- Combine your Phase-1 rule violations with an unsupervised ML anomaly score (Isolation Forest, or reconstruction error from a simple autoencoder, or a one-class model) — judges specifically want "rule/ML combination."
- Predict `exception_required` / `exception_type` as a supervised or semi-supervised head where labels exist.
- Explain anomaly drivers per flagged record (top contributing features/rules).
- Curate **20+ reviewer-ready examples**: loan_id, anomaly score, rule(s) triggered, top drivers, suggested action — this is an explicit minimum deliverable, don't skip the count.

Deliverable: `reports/anomaly_examples.csv` + write-up, demo-flow step 9.

---

## Phase 6 — Scenario & Stress Simulation (10 pts)

- Apply `macro_scenarios.csv` assumptions for base / adverse-credit / high-prepayment cases against your Phase-3 models (e.g. shift credit-score/DTI distributions or reweight hazard rates per scenario).
- Produce projected delinquency/default/prepayment rates under each scenario.
- Segment outputs by vintage, credit band, state, or servicer (reuse Phase-4 segmentation code).
- Write up the top 3–5 drivers behind why each scenario moves the numbers the way it does.

Deliverable: `reports/scenario_report`, demo-flow step 10.

---

## Phase 7 — Explainability & Responsible AI (10 pts)

- Global importances (SHAP summary or gain-based) for each Phase-3 model.
- Local explanation for individual loans (SHAP waterfall) — prepare at least one for the demo (demo-flow step 11).
- False positive / false negative analysis: pull actual misclassified examples and characterize them (which segments, which feature ranges).
- Report model confidence/uncertainty (predicted-probability calibration curves, or prediction intervals if you have time).
- Optional but cheap points: a short bias/fairness check across state or credit-band segments.

Deliverable: `reports/explainability_report`.

---

## Phase 8 — LLM-Assisted Reviewer Copilot (10 pts)

- Build a thin, **grounded** layer: LLM reads retrieved context (the loan's own record + relevant `data_dictionary.md` entries + relevant `validation_rules.json` entries + your model's anomaly/prediction outputs) and produces a reviewer note — never let it invent numbers.
- Log every call: prompt, model, timestamp, output — required, not optional (Task 7).
- Explicitly label all LLM output as "recommendation, not decision" in the UI/output format.
- **Deliberately include failure examples**: cases where the LLM was vague, overconfident, or wrong, and show the correction/rejection — this is graded directly ("hallucination controls," demo-flow step 13). Don't cherry-pick only good examples.
- Keep this LLM layer strictly downstream of the ML outputs — never let it replace the Task 2/3/5 predictions themselves, or you risk the disqualification clause ("only uses an LLM API for prediction").

Deliverable: `src/copilot/`, `reports/llm_prompt_log.jsonl`, demo-flow steps 12–13.

---

## Phase 9 — Packaging, Submission & Reproducibility (5 pts, plus gates the Minimum Acceptable Solution)

- Generate `submission.csv` matching `submission_template.csv` exactly (column names, order, value ranges) for the unlabeled test set — probabilities, anomaly scores, reviewer actions, confidence.
- Write the **model card**: objective, data, features, model type(s), validation method, metrics, limitations, leakage controls, known failure modes.
- Finalize README with exact reproduction steps (one command or a short numbered sequence) from raw CSVs to `submission.csv`.
- Sanity-check: no target leakage, no random split anywhere, all required files present per the Minimum Acceptable Solution checklist in the problem statement (§9).

---

## Phase 10 — Agentic Coding Evidence (5 pts)

- Finalize `ai_dev_log/log.md`: AI tools used, representative prompts (not all of them — the illustrative ones), accepted vs rejected AI outputs with reasons, approximate AI-generated code share, lessons learned.
- This should already be 80% done if you logged as you went in Phase 0 onward — don't leave it for the last night.

---

## Phase 11 — Demo Video (5-minute flow, maps 1:1 to problem statement §14)

Rehearse in this exact order since it mirrors both the demo-flow spec and the judging rubric, so nothing you built goes unseen:
dataset/targets → profiling report → top data-quality issues → feature engineering → time-aware split → baseline model → improved model → survival/transition output → anomaly examples → scenario output → one loan's local explanation → one LLM reviewer note → one rejected/corrected LLM output → final submission file → AI Development Log.

---

## Cross-cutting things that silently cost points if missed

- Random or loan-leaking splits anywhere (explicit disqualification trigger) — audit every split call.
- Any feature that's actually a disguised future label (explicit disqualification trigger).
- A model card that doesn't state limitations/failure modes — judges look for this explicitly, its absence reads as overconfidence.
- LLM copilot with zero grounding or zero "we rejected this LLM output" examples — both are explicitly graded for.
- Missing reproducibility — if a judge can't re-run your pipeline, that's a direct hit on the 5-point ML Engineering line and undermines credibility everywhere else.

## Suggested time allocation (if this is a timeboxed hackathon)

Rough split by points-per-effort: Phase 1 (profiling) and Phase 3 (prediction) are the most point-dense and should get the most hours; Phase 4 (survival) is conceptually the newest ground for most teams — budget extra research time there; Phases 5/6/7/8 can be built in parallel by different team members once Phase 3's model outputs exist, since they all consume it.
