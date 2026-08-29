"""
Phase 5 -- supporting artifacts.

macro_scenarios.csv, validation_rules.json, data_dictionary.md and
submission_template.csv. The validation rules are emitted in the schema the
profiling layer's rule engine already consumes (column/min/max, allowed,
not_null, expression), so the two halves of the project stay wired together.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GenerationConfig, STATES


class MacroScenarioBuilder:
    """
    Monthly macro paths for Baseline, Adverse-Credit and High-Prepayment.

    Paths are deterministic given the seed and shaped by a simple mean-reverting
    ramp rather than a random walk, so scenario deltas stay interpretable: any
    difference in projected performance is attributable to the scenario, not to
    noise in the scenario itself.
    """

    def __init__(self, config: GenerationConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng

    def build(self, start: str = "2024-01", horizon_months: int = 48) -> pd.DataFrame:
        months = pd.period_range(start=start, periods=horizon_months, freq="M")
        t = np.arange(horizon_months)
        ramp = t / max(horizon_months - 1, 1)

        specs = {
            "Baseline": {
                "mortgage_rate": 6.60 + (-0.85) * ramp,
                "unemployment_rate": 3.90 + 0.30 * ramp,
                "hpi_index": 100.0 * np.power(1.030, t / 12.0),
                "default_multiplier": np.full(horizon_months, 1.00),
                "prepayment_multiplier": np.full(horizon_months, 1.00),
            },
            "Adverse-Credit": {
                "mortgage_rate": 6.60 + 1.10 * ramp,
                "unemployment_rate": 3.90 + 4.20 * np.sqrt(ramp),
                "hpi_index": 100.0 * np.power(0.920, t / 12.0),
                "default_multiplier": 1.00 + 1.60 * np.sqrt(ramp),
                "prepayment_multiplier": 1.00 - 0.45 * ramp,
            },
            "High-Prepayment": {
                "mortgage_rate": 6.60 - 2.40 * ramp,
                "unemployment_rate": 3.90 - 0.45 * ramp,
                "hpi_index": 100.0 * np.power(1.055, t / 12.0),
                "default_multiplier": 1.00 - 0.25 * ramp,
                "prepayment_multiplier": 1.00 + 1.90 * ramp,
            },
        }

        frames = []
        for scenario, series in specs.items():
            frame = pd.DataFrame(series)
            frame.insert(0, "scenario", scenario)
            frame.insert(1, "projection_month", months.to_timestamp())
            frame.insert(2, "horizon_month", t + 1)
            frames.append(frame)

        out = pd.concat(frames, ignore_index=True)
        numeric = out.select_dtypes(include=[np.number]).columns
        out[numeric] = out[numeric].round(4)
        return out


class ValidationRuleBuilder:
    """Deterministic checks in the FR Y-14M / CCAR idiom."""

    @staticmethod
    def build() -> dict:
        return {
            "schema_version": "1.0",
            "standard": "FR Y-14M / CCAR-aligned loan-level edit checks",
            "rules": [
                {
                    "name": "LOAN_ID_PRESENT", "column": "loan_id", "not_null": True,
                    "severity": "high",
                    "description": "Every remittance record must carry a loan identifier.",
                },
                {
                    "name": "REPORTING_MONTH_PRESENT", "column": "reporting_month",
                    "not_null": True, "severity": "high",
                    "description": "Every record must carry a reporting period.",
                },
                {
                    "name": "BALANCE_SIGN", "column": "current_balance", "min": 0,
                    "severity": "high",
                    "description": "Unpaid principal balance may not be negative.",
                },
                {
                    "name": "BALANCE_CEILING",
                    "expression": "current_balance <= original_balance * 1.001",
                    "severity": "high",
                    "description": "Balance may not exceed the original balance absent a capitalising modification.",
                },
                {
                    "name": "DPD_RANGE", "column": "days_past_due", "min": 0, "max": 1080,
                    "severity": "medium",
                    "description": "Days past due must fall within a plausible reporting range.",
                },
                {
                    "name": "STATUS_DOMAIN", "column": "current_status",
                    "allowed": list(STATES), "severity": "high",
                    "description": "Performance status must be a recognised state.",
                },
                {
                    "name": "SEQUENTIAL_DELINQUENCY",
                    "expression": "not (current_status == '90-DPD' and days_past_due < 90)",
                    "severity": "high",
                    "description": "Delinquency status and days past due must agree; buckets may not be skipped.",
                },
                {
                    "name": "ABSORBING_STATE_FINALITY",
                    "expression": "not (prepayment_flag == 1 and current_balance > 1.0)",
                    "severity": "high",
                    "description": "A prepaid or defaulted loan is terminal and must carry a zero balance.",
                },
                {
                    "name": "DEFAULT_DELINQUENCY_CONSISTENCY",
                    "expression": "not (default_flag == 1 and days_past_due < 90)",
                    "severity": "high",
                    "description": "A defaulted loan must be at least 90 days delinquent.",
                },
                {
                    "name": "TEMPORAL_ORDERING",
                    "expression": "reporting_month >= origination_month",
                    "severity": "high",
                    "description": "A performance record may not predate origination.",
                },
                {
                    "name": "RATE_RANGE", "column": "interest_rate", "min": 0, "max": 25,
                    "severity": "medium",
                    "description": "Note rate must fall within a plausible range.",
                },
                {
                    "name": "TERM_NON_NEGATIVE", "column": "remaining_term_months", "min": 0,
                    "severity": "medium",
                    "description": "Remaining term may not be negative.",
                },
                {
                    "name": "MUTUALLY_EXCLUSIVE_TERMINATION",
                    "expression": "not (default_flag == 1 and prepayment_flag == 1)",
                    "severity": "high",
                    "description": "A loan cannot both default and prepay in the same period.",
                },
                {
                    "name": "DOCUMENT_STATUS_DOMAIN", "column": "document_status",
                    "allowed": ["Complete", "Pending", "Missing"], "severity": "low",
                    "description": "Document status must be a recognised value.",
                },
            ],
        }


class DataDictionaryBuilder:
    """Field-level documentation, emitted as a markdown table."""

    FIELDS: list[tuple[str, str]] = [
        ("loan_id", "Unique 12-character alphanumeric loan identifier."),
        ("month_index", "Zero-based months-on-book counter within the loan's observed history."),
        ("reporting_month", "Calendar month the performance record describes."),
        ("origination_month", "Month the loan was originated."),
        ("loan_age_months", "Months elapsed between origination and the reporting month."),
        ("remaining_term_months", "Contractual months remaining at the reporting month."),
        ("original_balance", "Unpaid principal balance at origination, USD."),
        ("current_balance", "Unpaid principal balance at the reporting month, USD. Zero once terminal."),
        ("interest_rate", "Annual fixed note rate, percent. Vintage base rate plus a credit-risk premium."),
        ("credit_score", "Borrower credit score at origination (500-850)."),
        ("credit_score_band", "Credit score bucket: <620, 620-659, 660-699, 700-739, 740-799, 800+."),
        ("ltv", "Loan-to-value ratio at origination, percent."),
        ("ltv_band", "LTV bucket: <60%, 60-75%, 75-80%, 80-90%, 90-97%."),
        ("dti", "Debt-to-income ratio at origination, percent."),
        ("dti_band", "DTI bucket: <30%, 30-36%, 36-43%, >43%."),
        ("state", "US state of the mortgaged property."),
        ("loan_purpose", "Home Purchase, Rate/Term Refinance, or Cash-Out Refinance."),
        ("property_type", "Single-Family Detached, Condominium, or PUD."),
        ("occupancy_type", "Primary Residence, Investment Property, or Second Home."),
        ("servicer_name", "Institution servicing the loan."),
        ("original_term_months", "Contractual term at origination (180, 240, or 360 months)."),
        ("vintage_year", "Calendar year of origination."),
        ("current_status", "Performance state: Current, 30-DPD, 60-DPD, 90-DPD, Default, Prepaid."),
        ("days_past_due", "Days delinquent at the reporting month, consistent with current_status."),
        ("modification_flag", "True if the loan has received a loss-mitigation modification."),
        ("prepayment_flag", "1 if the loan prepaid in full in this period."),
        ("default_flag", "1 if the loan is in default in this period."),
        ("loss_severity_band", "Realised loss severity (Low/Medium/High). Populated only on Default rows; null by design elsewhere."),
        ("last_updated_at", "Timestamp the servicing record was last refreshed."),
        ("source_system", "System of record that produced the row."),
        ("document_status", "Completeness of the loan's document file: Complete, Pending, or Missing."),
        ("next_3m_delinquency_flag", "TARGET. 1 if the loan reaches 30+ DPD in any of months t+1 to t+3. Null where the window is censored."),
        ("next_6m_delinquency_flag", "TARGET. 1 if the loan reaches 60+ DPD in any of months t+1 to t+6. Null where the window is censored."),
        ("next_12m_default_flag", "TARGET. 1 if the loan defaults within months t+1 to t+12. Null where the window is censored."),
        ("next_12m_prepayment_flag", "TARGET. 1 if the loan prepays within months t+1 to t+12. Null where the window is censored."),
        ("next_state", "TARGET. Exact performance state at month t+1. Null where t+1 is unobserved."),
        ("exception_required", "TARGET. True if the record carries an injected data-quality defect."),
        ("exception_type", "TARGET. Defect taxonomy: Balance Discrepancy, Time Travel, Impossible State Transition, Zombie Loan, or None."),
    ]

    @classmethod
    def build(cls) -> str:
        header = (
            "# Data Dictionary\n\n"
            "Loan Performance Intelligence Engine -- synthetic benchmark suite.\n\n"
            "## Panel fields\n\n"
            "| Field | Definition |\n| :--- | :--- |\n"
        )
        body = "".join(f"| {name} | {desc} |\n" for name, desc in cls.FIELDS)

        notes = (
            "\n## Censoring convention\n\n"
            "Forward-looking targets are null where the outcome window extends past the "
            "observation cutoff **and** the loan had not already reached an absorbing state. "
            "Loans that defaulted or prepaid inside the window retain their labels, because "
            "an absorbing state resolves the outcome with certainty -- censoring those rows "
            "would bias the observed event rate downward.\n\n"
            "## Absorbing states\n\n"
            "`Default` and `Prepaid` are terminal. A loan emits exactly one row in its "
            "absorbing state and none afterwards. Any subsequent active row is, by "
            "construction, a `Zombie Loan` exception.\n\n"
            "## Structural nulls\n\n"
            "`loss_severity_band` is populated only for `Default` rows. Its absence "
            "elsewhere is a business rule, not a data-quality defect, and should be "
            "excluded from missingness penalties.\n"
        )
        return header + body + notes


class SubmissionTemplateBuilder:
    """Blank prediction frame keyed to the test set."""

    PREDICTION_COLUMNS: dict[str, object] = {
        "prob_next_3m_delinquency": np.nan,
        "prob_next_6m_delinquency": np.nan,
        "prob_next_12m_default": np.nan,
        "prob_next_12m_prepayment": np.nan,
        "next_state": "",
        "exception_required": "",
        "exception_type": "",
        "anomaly_score": np.nan,
        "top_drivers": "",
        "action": "",
        "confidence": np.nan,
    }

    @classmethod
    def build(cls, test: pd.DataFrame) -> pd.DataFrame:
        template = test[["loan_id", "reporting_month"]].copy()
        for column, default in cls.PREDICTION_COLUMNS.items():
            template[column] = default
        return template.reset_index(drop=True)


class ArtifactWriter:
    """Writes the non-panel artifacts to disk."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_all(self, scenarios: pd.DataFrame, test: pd.DataFrame) -> list[Path]:
        written: list[Path] = []

        scenario_path = self.output_dir / "macro_scenarios.csv"
        scenarios.to_csv(scenario_path, index=False)
        written.append(scenario_path)

        rules_path = self.output_dir / "validation_rules.json"
        rules_path.write_text(json.dumps(ValidationRuleBuilder.build(), indent=2))
        written.append(rules_path)

        dictionary_path = self.output_dir / "data_dictionary.md"
        dictionary_path.write_text(DataDictionaryBuilder.build())
        written.append(dictionary_path)

        template_path = self.output_dir / "submission_template.csv"
        SubmissionTemplateBuilder.build(test).to_csv(template_path, index=False)
        written.append(template_path)

        return written
