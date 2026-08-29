"""
Central configuration: paths, seed, and the expected schema.

Everything downstream imports from here so there is exactly one place to change
when the organiser's real data pack lands and column names differ slightly.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility (ML Engineering criterion -- one seed, used everywhere)
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

TRAIN_PATH = DATA_DIR / "loan_monthly_performance_train.csv"
TEST_PATH = DATA_DIR / "loan_monthly_performance_test.csv"
STATIC_PATH = DATA_DIR / "loan_static_attributes.csv"
SERVICER_PATH = DATA_DIR / "servicer_updates.csv"
DATA_DICTIONARY_PATH = DATA_DIR / "data_dictionary.md"
VALIDATION_RULES_PATH = DATA_DIR / "validation_rules.json"
MACRO_SCENARIOS_PATH = DATA_DIR / "macro_scenarios.csv"
SUBMISSION_TEMPLATE_PATH = DATA_DIR / "submission_template.csv"

# --------------------------------------------------------------------------
# Schema expectations (from the problem statement, section 7)
# Nothing here is enforced -- it is used to *check* what actually arrived,
# so a schema surprise shows up as a report line instead of a crash.
# --------------------------------------------------------------------------
ID_COL = "loan_id"
TIME_COL = "reporting_month"
MONTH_INDEX_COL = "month_index"

EXPECTED_NUMERIC = [
    "month_index",
    "loan_age_months",
    "remaining_term_months",
    "original_balance",
    "current_balance",
    "interest_rate",
    "days_past_due",
]

EXPECTED_CATEGORICAL = [
    "credit_score_band",
    "ltv_band",
    "dti_band",
    "state",
    "loan_purpose",
    "occupancy_type",
    "property_type",
    "servicer_name",
    "current_status",
    "loss_severity_band",
    "source_system",
    "document_status",
]

EXPECTED_FLAGS = [
    "modification_flag",
    "prepayment_flag",
    "default_flag",
]

EXPECTED_DATES = [
    "reporting_month",
    "origination_month",
    "last_updated_at",
]

TARGET_COLS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
    "next_state",
    "exception_required",
    "exception_type",
]

# --------------------------------------------------------------------------
# Data-quality scoring weights (Task 1: record-level and batch-level scores)
# Higher weight == a more serious defect. Tune once you see the real data.
# --------------------------------------------------------------------------
DQ_WEIGHTS = {
    "rule_violation": 3.0,
    "missing_critical": 2.0,
    "missing_non_critical": 0.5,
    "outlier": 1.0,
    "stale_record": 1.5,
    "source_conflict": 2.5,
}

CRITICAL_COLUMNS = [
    "loan_id",
    "reporting_month",
    "current_balance",
    "current_status",
    "days_past_due",
]

# Drift thresholds (industry-conventional PSI bands)
PSI_MINOR = 0.10
PSI_MAJOR = 0.25

# A record whose servicer update is older than this many days is "stale"
STALE_DAYS = 90

# --------------------------------------------------------------------------
# Phase 2/3: feature engineering and predictive modelling (Task 2)
# --------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
DQ_SCORES_PATH = REPORTS_DIR / "dq_scores_train.csv"

# Binary targets and the forward window each one looks over, in months.
# The horizon drives the purge gap between time-ordered splits: a row labelled
# with a 12-month window must not sit in TRAIN while the months it peeks at sit
# in VALIDATION, or the split leaks the future backwards.
TARGET_HORIZONS = {
    "next_3m_delinquency_flag": 3,
    "next_6m_delinquency_flag": 6,
    "next_12m_default_flag": 12,
    "next_12m_prepayment_flag": 12,
    "next_state": 1,
}

BINARY_TARGETS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]
MULTICLASS_TARGET = "next_state"

# Time-aware split boundaries (inclusive month ends). Chosen so the three
# windows are contiguous, ordered and non-overlapping; see models/splitting.py.
# The organiser's own test file (2024-01 onward) is unlabelled, so the graded
# holdout below is carved out of the labelled train panel.
#
# The validation window is 18 months wide on purpose: each split is purged by
# the target's forward horizon, and a window only as long as the horizon (12
# months, for the default/prepayment targets) would be purged out of existence.
SPLIT_TRAIN_END = "2021-06-01"
SPLIT_VALID_END = "2022-12-01"
SPLIT_TEST_END = "2023-12-01"

# Absorbing states carry no forward-looking decision: the outcome is already
# realised. They are held out of training/evaluation and handled by a
# deterministic override at scoring time.
ABSORBING_STATES = ("Default", "Prepaid")

# Ordinal encoding of the performance state, worst-last. Used for the
# "worst status to date" feature and for multiclass label ordering.
STATUS_ORDER = ["Prepaid", "Current", "30-DPD", "60-DPD", "90-DPD", "Default"]
DOC_STATUS_ORDER = ["Complete", "Pending", "Missing"]

# Columns that must never reach a feature matrix: they are labels, they are
# derived from labels, or they are populated only on an absorbing row.
FORBIDDEN_FEATURES = [
    *TARGET_COLS,
    "loss_severity_band",
    "default_flag",
    "prepayment_flag",
]

# Operating point for the recall-at-fixed-precision metric.
PRECISION_FLOOR = 0.50

# --------------------------------------------------------------------------
# Phase 4: time-to-event / survival modelling (Task 3)
# --------------------------------------------------------------------------
# Competing-risk event codes. 0 is reserved for "still at risk when observation
# ended" -- see survival/dataset.py for the censoring taxonomy.
EVENT_CENSORED = 0
EVENT_DEFAULT = 1
EVENT_PREPAID = 2

EVENT_LABELS = {
    EVENT_CENSORED: "censored",
    EVENT_DEFAULT: "default",
    EVENT_PREPAID: "prepaid",
}

# Map from the absorbing performance state to its competing-risk code.
ABSORBING_TO_EVENT = {
    "Default": EVENT_DEFAULT,
    "Prepaid": EVENT_PREPAID,
}

# Time-aware holdout for survival: train on older vintages, test on newer ones.
# Splitting by origination (not by reporting month) is the right axis here --
# the clock a survival model runs on is loan age, so a vintage is the natural
# unit of "what the model had already seen".
SURVIVAL_TRAIN_VINTAGE_END = "2021-06-01"

# Horizons, in months on book, at which cumulative incidence is tabulated.
SURVIVAL_HORIZONS = (12, 24, 36, 48)

# Origination-time covariates for the Cox models. Deliberately excludes every
# time-varying panel field: a survival model's covariates are measured at t=0,
# and a value observed at month 18 cannot inform the hazard at month 6.
SURVIVAL_NUMERIC_COVARIATES = [
    "credit_score",
    "ltv",
    "dti",
    "interest_rate",
    "log_original_balance",
    "original_term_months",
]
SURVIVAL_CATEGORICAL_COVARIATES = [
    "loan_purpose",
    "occupancy_type",
    "property_type",
]

# Segments the event curves are cut by.
SURVIVAL_SEGMENTS = ["credit_score_band", "vintage_year", "ltv_band"]


# --------------------------------------------------------------------------
# Phase 6: scenario and stress simulation (Task 5)
# --------------------------------------------------------------------------
# Band edges, used to *recompute* the banded columns after a stress shifts the
# continuous value underneath them. Shifting `ltv` while leaving `ltv_band` at
# its original level hands the model a contradictory record.
CREDIT_SCORE_BANDS = ("<620", "620-659", "660-699", "700-739", "740-799", "800+")
CREDIT_SCORE_EDGES = (-1e18, 620, 660, 700, 740, 800, 1e18)

LTV_BANDS = ("<60%", "60-75%", "75-80%", "80-90%", "90-97%")
LTV_EDGES = (-1e18, 60, 75, 80, 90, 1e18)

DTI_BANDS = ("<30%", "30-36%", "36-43%", ">43%")
DTI_EDGES = (-1e18, 30, 36, 43, 1e18)

# Plausible ranges a stressed feature may not leave. A scenario that pushes a
# credit score to 380 or an LTV to 400% is describing a world the model was
# never fitted on, and the resulting prediction is extrapolation dressed as a
# projection.
STRESS_BOUNDS = {
    "credit_score": (500.0, 850.0),
    "ltv": (5.0, 150.0),
    "dti": (5.0, 75.0),
    "interest_rate": (1.0, 20.0),
}

# The Phase 3 targets projected under each scenario, and the label each carries
# in the scenario report.
SCENARIO_TARGETS = {
    "next_3m_delinquency_flag": "delinquency_3m",
    "next_12m_default_flag": "default_12m",
    "next_12m_prepayment_flag": "prepayment_12m",
}

# Segments the projections are cut by.
SCENARIO_SEGMENTS = ["vintage_year", "credit_score_band", "state", "servicer_name"]

# Projection months reported in the summary tables (months from the scenario start).
SCENARIO_HORIZONS = (1, 12, 24, 36, 48)


# --------------------------------------------------------------------------
# Phase 7: explainability and responsible AI (Task 6)
# --------------------------------------------------------------------------
EXPLAINABILITY_DIR = REPORTS_DIR / "explainability_report"

# Models explained in Phase 7, and the name each carries in the report.
EXPLAIN_TARGETS = {
    "next_3m_delinquency_flag": "delinquency",
    "next_12m_default_flag": "default",
    "next_12m_prepayment_flag": "prepayment",
}

# Rows sampled for SHAP. Not a memory workaround at this data size -- the full
# 58k-row test set computes in ~3s with tree_path_dependent perturbation, which
# needs no background dataset at all. It is a guard for the organiser's real
# pack (potentially 10x larger, and 6x again for the multiclass head) and a
# legibility limit: a beeswarm of 58,000 points is a solid block of ink.
SHAP_SAMPLE_ROWS = 20000
SHAP_PLOT_ROWS = 4000

# Segments the error and disparity analysis is cut by.
EXPLAIN_SEGMENTS = ["credit_score_band", "ltv_band", "vintage_year", "state", "servicer_name"]

# A group smaller than this is not reported as a disparity finding: a false
# positive rate computed on nine loans is noise with a decimal point.
MIN_GROUP_SIZE = 200

# Conventional adverse-impact threshold. A selection-rate ratio below this
# between the most and least selected group is the flag used in US employment
# law (the "four-fifths rule"); it is borrowed here as a *screening* device,
# not as a legal test. See src/explain/fairness.py.
DISPARITY_RATIO_FLOOR = 0.80


# --------------------------------------------------------------------------
# Phase 9: packaging and submission
# --------------------------------------------------------------------------
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

# The template is the binding contract for the submission's shape. It is read
# at run time rather than hardcoded, so a change the organiser makes to the
# template surfaces as a validation failure instead of a silently wrong file.
SUBMISSION_PROBABILITY_COLUMNS = {
    "next_3m_delinquency_flag": "prob_next_3m_delinquency",
    "next_6m_delinquency_flag": "prob_next_6m_delinquency",
    "next_12m_default_flag": "prob_next_12m_default",
    "next_12m_prepayment_flag": "prob_next_12m_prepayment",
}

# Columns that must be a probability in [0, 1] if present.
SUBMISSION_UNIT_INTERVAL_COLUMNS = [
    *SUBMISSION_PROBABILITY_COLUMNS.values(),
    "anomaly_score",
    "confidence",
]

# How many driver names go into the submission's `top_drivers` cell.
SUBMISSION_TOP_DRIVERS = 3
