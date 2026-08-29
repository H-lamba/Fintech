# Intain Campus FinTech Challenge 2026 | AI Track
## Problem Statement: Loan Performance Intelligence Engine

Build an ML-first system for loan-data profiling, performance prediction, anomaly detection, scenario simulation, explainability, and grounded LLM-assisted review.

### Benchmarking lens
The problem is designed to move beyond LLM wrappers. It reflects current market direction in loan analytics: data profiling, prepayment/default prediction, survival or transition modeling, anomaly detection, scenario simulation, explainability, and governed AI copilots.

### 1. Challenge Title
**Loan Performance Intelligence Engine**
Build an AI system that profiles messy loan-level data, predicts loan performance, detects anomalies, runs simple scenarios, and explains outputs to a human reviewer.

### 2. Participant Hook
Can you build a small but serious AI engine for loan-level data?
This challenge is not about wrapping an LLM API. Participants must build real data-science and ML capabilities, including:
* Data profiling
* Feature engineering
* Supervised prediction
* Time-aware validation
* Anomaly detection
* Explainability
* Model calibration
* Scenario simulation
* LLM-assisted reviewer explanations
* Agentic coding evidence

The LLM can help explain, summarize, retrieve definitions, generate reviewer notes, and assist development, but the core predictive work must come from data science and machine learning.

### 3. Business-Adjacent Context
Loan-level data is the raw material of many financial workflows. A single loan record may describe loan amount, interest rate, origination date, loan term, geography, credit score band, loan-to-value band, payment status, delinquency status, prepayment status, balance, servicer updates, and document status.

**Core question**
Given messy loan-level data and historical performance, can we identify which records are unreliable, which loans are likely to deteriorate, and what the portfolio may look like under different future scenarios?

Participants do not need structured-finance knowledge. They only need to understand tabular data, model development, time-based outcomes, and explainable AI.

### 4. Benchmarking Takeaways
The AI track is benchmarked against the current direction of analytics platforms serving loans, private credit, and structured finance.

| Benchmark theme | Implication for the problem statement |
| :--- | :--- |
| **Data intelligence before modeling** | Require profiling of distributions, missingness, outliers, relationships, association rules, and drift before model training. |
| **Prediction is multi-outcome** | Move beyond one binary risk label to delinquency, default, prepayment, and next-state prediction. |
| **Time-aware modeling matters** | Require a time-aware split and at least one survival, hazard, or transition-style model. |
| **Scenario analytics are table stakes** | Require base, adverse-credit, and high-prepayment scenario projections. |
| **LLM copilots need governance** | Require grounded explanations, prompt logs, human decision control, and examples of rejected LLM output. |

### 5. Source Data
The organizer should provide a curated synthetic or preprocessed dataset for judging. It should be inspired by public loan-level sources but should not require students to register with data portals or understand raw mortgage-performance schemas during the hackathon.

* **Fannie Mae Single-Family Loan Performance Data:** https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data
* **Fannie Mae Data Dynamics:** https://datadynamics.fanniemae.com/data-dynamics/#/reportMenu:category=Loan_Performance
* **Freddie Mac Single-Family Loan-Level Dataset:** https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
* **Freddie Mac Clarity Data Intelligence Download Portal:** https://claritydownload.fmapps.freddiemac.com/
* **HMDA Data Publication:** https://ffiec.cfpb.gov/data-publication/
* **HMDA Public LAR Data Fields:** https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields

### 6. Organizer-Provided Data Pack

| File | Purpose / Expected contents |
| :--- | :--- |
| `loan_monthly_performance_train.csv` | Panel dataset with one row per loan per month; suggested 250,000 to 1,000,000 rows; includes static and monthly performance features plus target labels. |
| `loan_monthly_performance_test.csv` | Unlabeled test dataset for final scoring. Participants submit probabilities, anomaly scores, and reviewer actions. |
| `loan_static_attributes.csv` | Origination-level information such as original balance, credit-score band, LTV band, DTI band, state, loan purpose, property type, and vintage. |
| `servicer_updates.csv` | Second-source file with partial or conflicting updates used for source conflict detection, stale record logic, and reconciliation. |
| `data_dictionary.md` | Plain-English field definitions used for feature understanding, LLM grounding, and documentation. |
| `validation_rules.json` | Starter deterministic checks for balance consistency, date validity, delinquency consistency, closed/prepaid status, and document gaps. |
| `macro_scenarios.csv` | Simple scenario assumptions for base, adverse-credit, and high-prepayment cases. |
| `submission_template.csv` | Required output format for probabilities, next state, exception type, anomaly score, top drivers, action, and confidence. |

### 7. Example Training Fields and Targets
The main training file should contain monthly loan performance records. Example fields include:
`loan_id`, `month_index`, `reporting_month`, `origination_month`, `loan_age_months`, `remaining_term_months`, `original_balance`, `current_balance`, `interest_rate`, `credit_score_band`, `ltv_band`, `dti_band`, `state`, `loan_purpose`, `occupancy_type`, `property_type`, `servicer_name`, `current_status`, `days_past_due`, `modification_flag`, `prepayment_flag`, `default_flag`, `loss_severity_band`, `last_updated_at`, `source_system`, `document_status`.

Target variables should include `next_3m_delinquency_flag`, `next_6m_delinquency_flag`, `next_12m_default_flag`, `next_12m_prepayment_flag`, `next_state`, `exception_required`, and `exception_type`.

### 8. Required Tasks

**Task 1: Data Intelligence and Profiling**
* Profile column distributions.
* Identify missing-value patterns.
* Detect outliers and invalid date relationships.
* Identify correlations and highly dependent fields.
* Detect cross-column relationship breaks.
* Compare train versus test drift.
* Generate record-level and batch-level data-quality scores.

**Task 2: Loan Performance Prediction**
* Train non-LLM models for delinquency, default, prepayment, and next-state prediction.
* Use a time-aware split rather than random row-level splitting.
* Compare baseline and improved models.
* Handle class imbalance and calibration.
* Use metrics such as ROC-AUC, PR-AUC, F1, recall at fixed precision, Brier score, and macro-F1.

**Task 3: Time-to-Event or Survival Modeling**
* Implement a survival, hazard, competing-risk approximation, or monthly transition model.
* Show event curves or cumulative probabilities.
* Explain treatment of censoring or state transitions.
* Compare against a simpler baseline.

**Task 4: Anomaly and Exception Detection**
* Generate a record-level anomaly score.
* Predict exception probability and exception type.
* Explain anomaly drivers.
* Provide at least 20 reviewer-ready anomaly examples.

**Task 5: Scenario and Stress Simulation**
* Apply base, adverse-credit, and high-prepayment scenarios.
* Produce projected delinquency, default, and prepayment rates.
* Show segment-level impacts by vintage, credit band, state, or servicer.
* Explain top scenario drivers.

**Task 6: Explainability Layer**
* Provide global feature importance and local explanations.
* Explain drivers of default, delinquency, prepayment, and anomaly scores.
* Show model confidence or uncertainty.
* Analyze false positives and false negatives.

**Task 7: LLM-Assisted Reviewer Copilot**
* Use an LLM for grounded summaries, reviewer notes, data-dictionary retrieval, rule suggestions, scenario summaries, or natural-language analysis.
* Log prompt, model, timestamp, and output.
* Label LLM output as a recommendation, not a decision.
* Include examples where the LLM was wrong, vague, or overconfident.

**Task 8: Agentic ML Development Evidence**
* Submit an AI Development Log.
* Document AI tools used, representative prompts, accepted/rejected outputs, human review process, approximate AI-generated code share, and lessons learned.

### 9. Minimum Acceptable Solution
* Reproducible data pipeline
* Data profiling report
* Feature engineering
* Non-LLM supervised model
* Time-aware train / validation split
* Delinquency or default prediction
* Prepayment or next-state prediction
* Anomaly or exception detection
* Explainability output
* LLM reviewer summary
* Model card
* AI Development Log
* `submission.csv`

**Qualification rule**
A solution that only sends records to an LLM API for classification should not qualify.

### 10. Advanced Features
* Competing-risk survival model
* Monte Carlo portfolio simulation
* Drift monitoring dashboard
* Segment-level scenario curves
* Model calibration by vintage or credit band
* MLflow or Weights & Biases experiment tracking
* RAG over data dictionary and validation rules
* Agentic experiment runner
* Automated feature-store style pipeline
* Bias / fairness analysis
* Counterfactual explanations
* Stress sensitivity by feature cluster
* Model confidence intervals
* Human-in-the-loop active learning
* Synthetic-data stress testing

### 11. Expected Deliverables

| Deliverable | Description |
| :--- | :--- |
| **GitHub repository** | Complete source code. |
| **Reproducible notebook or scripts** | End-to-end model development and scoring workflow. |
| **submission.csv** | Predictions in the required format. |
| **Model card** | Objective, data, features, model type, validation method, metrics, limitations, leakage controls, and known failure modes. |
| **Data intelligence report** | Profiling, missingness, outliers, drift, relationship checks, and top anomalies. |
| **Explainability report** | Global feature importance, local examples, false positives, false negatives, and model uncertainty. |
| **Scenario report** | Base, adverse, and high-prepayment scenario outputs. |
| **LLM copilot demo** | Grounded reviewer explanation or natural-language analysis. |
| **AI Development Log** | Required. |
| **Five-minute demo video** | End-to-end flow. |

### 12. Judging Criteria

| Criterion | Points | What judges should look for |
| :--- | :--- | :--- |
| **Data Intelligence and Profiling** | 15 | Missingness, outliers, relationship checks, train/test drift, and data-quality score. |
| **Predictive Modeling** | 20 | Supervised models, time-aware split, class imbalance handling, default/delinquency/prepayment prediction, calibration. |
| **Time-to-Event / Transition Modeling** | 15 | Survival, hazard, or transition model with sensible curves/projections and baseline comparison. |
| **Anomaly and Exception Intelligence** | 10 | Suspicious record detection, anomaly drivers, rule/ML combination, reviewer-ready examples. |
| **Scenario and Stress Simulation** | 10 | Base/adverse/high-prepayment scenarios, segment outputs, impact explanation. |
| **Explainability and Responsible AI** | 10 | Global/local explanations, model card, error analysis, calibration, uncertainty, limitations. |
| **Smart LLM Usage** | 10 | Grounded LLM output, useful reviewer summaries, prompt logs, hallucination controls, ML not replaced by LLM. |
| **ML Engineering and Reproducibility** | 5 | Clean code, runnable pipeline, reproducible submission, README. |
| **Agentic Coding Evidence** | 5 | AI Development Log, useful prompts, human review process, rejected AI output examples. |

### 13. Low-Score or Disqualification Conditions
* Only uses an LLM API for prediction.
* Does not train a non-LLM model.
* Uses random splits that leak the same loan across train and validation without justification.
* Leaks target labels into features.
* Provides no reproducible code.
* Provides no evaluation metrics.
* Fabricates results.
* Uses public data in violation of source terms.
* Cannot explain model behavior.
* Presents LLM-generated narratives without grounding.

### 14. Five-Minute Demo Flow
1. Dataset and targets
2. Data profiling report
3. Top data-quality issues
4. Feature-engineering approach
5. Time-aware split
6. Baseline model performance
7. Improved model performance
8. Survival or transition model output
9. Anomaly examples
10. Scenario output
11. Local explanation for one loan
12. LLM-generated reviewer note
13. Example of LLM output rejected or corrected
14. Final submission file
15. AI Development Log
