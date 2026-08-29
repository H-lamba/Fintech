"""
Generation configuration for the synthetic loan-performance suite.

Every tunable lives here so the generator's behaviour is auditable from one
file and reproducible from one seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# State space (order is load-bearing: indices are used throughout the engine)
# ---------------------------------------------------------------------------
STATES: tuple[str, ...] = ("Current", "30-DPD", "60-DPD", "90-DPD", "Default", "Prepaid")
S_CURRENT, S_30, S_60, S_90, S_DEFAULT, S_PREPAID = range(6)
ABSORBING_STATES: tuple[int, ...] = (S_DEFAULT, S_PREPAID)
DELINQUENT_STATES: tuple[int, ...] = (S_30, S_60, S_90, S_DEFAULT)

# Baseline one-month transition matrix. Rows sum to 1.0 (asserted at build time).
BASE_TRANSITION_MATRIX: np.ndarray = np.array(
    [
        # Current  30-DPD  60-DPD  90-DPD  Default  Prepaid
        [0.975, 0.018, 0.000, 0.000, 0.000, 0.007],  # from Current
        [0.600, 0.200, 0.180, 0.000, 0.000, 0.020],  # from 30-DPD
        [0.250, 0.150, 0.200, 0.400, 0.000, 0.000],  # from 60-DPD
        [0.100, 0.000, 0.100, 0.350, 0.450, 0.000],  # from 90-DPD
        [0.000, 0.000, 0.000, 0.000, 1.000, 0.000],  # Default (absorbing)
        [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # Prepaid (absorbing)
    ],
    dtype=np.float64,
)

# Transitions toward worse performance -- amplified by borrower risk.
DOWNWARD_TRANSITIONS: tuple[tuple[int, int], ...] = (
    (S_CURRENT, S_30),
    (S_30, S_60),
    (S_60, S_90),
    (S_90, S_DEFAULT),
)

# Transitions toward better performance -- suppressed by borrower risk.
CURE_TRANSITIONS: tuple[tuple[int, int], ...] = (
    (S_30, S_CURRENT),
    (S_60, S_CURRENT),
    (S_60, S_30),
    (S_90, S_CURRENT),
    (S_90, S_60),
)

PREPAY_TRANSITIONS: tuple[tuple[int, int], ...] = (
    (S_CURRENT, S_PREPAID),
    (S_30, S_PREPAID),
)

# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------
CREDIT_BANDS: tuple[str, ...] = ("<620", "620-659", "660-699", "700-739", "740-799", "800+")
CREDIT_EDGES: tuple[float, ...] = (-np.inf, 620, 660, 700, 740, 800, np.inf)

LTV_BANDS: tuple[str, ...] = ("<60%", "60-75%", "75-80%", "80-90%", "90-97%")
LTV_EDGES: tuple[float, ...] = (-np.inf, 60, 75, 80, 90, np.inf)

DTI_BANDS: tuple[str, ...] = ("<30%", "30-36%", "36-43%", ">43%")
DTI_EDGES: tuple[float, ...] = (-np.inf, 30, 36, 43, np.inf)

# Risk multipliers applied to downward-transition hazard.
CREDIT_RISK_MULTIPLIER: dict[str, float] = {
    "<620": 4.00,
    "620-659": 2.80,
    "660-699": 1.75,
    "700-739": 1.15,
    "740-799": 0.75,
    "800+": 0.50,
}

LTV_RISK_MULTIPLIER: dict[str, float] = {
    "<60%": 0.70,
    "60-75%": 0.90,
    "75-80%": 1.00,
    "80-90%": 1.30,
    "90-97%": 1.70,
}

# Higher-credit borrowers refinance more readily.
CREDIT_PREPAY_MULTIPLIER: dict[str, float] = {
    "<620": 0.55,
    "620-659": 0.75,
    "660-699": 0.95,
    "700-739": 1.10,
    "740-799": 1.30,
    "800+": 1.45,
}

# Risk premium added to the vintage base rate, in percentage points.
CREDIT_RATE_PREMIUM: dict[str, float] = {
    "<620": 1.60,
    "620-659": 1.05,
    "660-699": 0.62,
    "700-739": 0.32,
    "740-799": 0.12,
    "800+": 0.00,
}

# Approximate US 30-year fixed averages by origination vintage.
VINTAGE_BASE_RATE: dict[int, float] = {
    2018: 4.55,
    2019: 3.95,
    2020: 3.10,
    2021: 2.98,
    2022: 5.35,
    2023: 6.85,
}

# ---------------------------------------------------------------------------
# Categorical distributions
# ---------------------------------------------------------------------------
STATE_WEIGHTS: dict[str, float] = {
    "CA": 0.135, "TX": 0.105, "FL": 0.095, "NY": 0.062, "PA": 0.042,
    "IL": 0.041, "OH": 0.038, "GA": 0.037, "NC": 0.035, "MI": 0.031,
    "NJ": 0.030, "VA": 0.029, "WA": 0.028, "AZ": 0.027, "MA": 0.026,
    "TN": 0.024, "IN": 0.022, "MO": 0.021, "MD": 0.021, "WI": 0.019,
    "CO": 0.019, "MN": 0.018, "SC": 0.017, "AL": 0.015, "LA": 0.014,
    "KY": 0.013, "OR": 0.013, "OK": 0.012, "CT": 0.011, "UT": 0.010,
}

LOAN_PURPOSES: dict[str, float] = {
    "Home Purchase": 0.60,
    "Rate/Term Refinance": 0.25,
    "Cash-Out Refinance": 0.15,
}

PROPERTY_TYPES: dict[str, float] = {
    "Single-Family Detached": 0.75,
    "Condominium": 0.15,
    "PUD": 0.10,
}

OCCUPANCY_TYPES: dict[str, float] = {
    "Primary Residence": 0.85,
    "Investment Property": 0.10,
    "Second Home": 0.05,
}

SERVICERS: tuple[str, ...] = (
    "Cornerstone Loan Servicing",
    "Meridian Residential Capital",
    "Atlas Mortgage Services",
    "Beacon Home Loans",
    "Northgate Financial Servicing",
)

SOURCE_SYSTEMS: dict[str, float] = {
    "CoreServicing": 0.72,
    "VendorAPI": 0.19,
    "LegacySFTP": 0.09,
}

DOCUMENT_STATUSES: dict[str, float] = {
    "Complete": 0.88,
    "Pending": 0.09,
    "Missing": 0.03,
}

LOSS_SEVERITY_BANDS: dict[str, float] = {"Low": 0.35, "Medium": 0.45, "High": 0.20}

ORIGINAL_TERMS: dict[int, float] = {360: 0.78, 240: 0.09, 180: 0.13}

# ---------------------------------------------------------------------------
# Anomaly taxonomy (Phase 4)
# ---------------------------------------------------------------------------
ANOMALY_TYPES: tuple[str, ...] = (
    "Balance Discrepancy",
    "Time Travel",
    "Impossible State Transition",
    "Zombie Loan",
)

ANOMALY_MIX: dict[str, float] = {
    "Balance Discrepancy": 0.35,
    "Time Travel": 0.20,
    "Impossible State Transition": 0.25,
    "Zombie Loan": 0.20,
}


@dataclass(frozen=True)
class GenerationConfig:
    """All knobs for one generation run."""

    # Cohort
    n_loans: int = 50_000
    seed: int = 42

    # Calendar
    origination_start: str = "2018-01"
    origination_end: str = "2023-12"
    observation_end: str = "2025-06"
    max_months_on_book: int = 60

    # Out-of-time split
    split_date: str = "2024-01-01"

    # Copula marginals: (mean, std, lower, upper)
    credit_score_params: tuple[float, float, float, float] = (700.0, 50.0, 500.0, 850.0)
    ltv_params: tuple[float, float, float, float] = (75.0, 10.0, 40.0, 100.0)
    dti_params: tuple[float, float, float, float] = (36.0, 8.0, 15.0, 55.0)

    # Copula correlation structure
    rho_credit_ltv: float = -0.45
    rho_credit_dti: float = -0.45
    rho_ltv_dti: float = 0.30

    # Origination balance (bounded log-normal)
    balance_log_median: float = 330_000.0
    balance_log_sigma: float = 0.34
    balance_min: float = 50_000.0
    balance_max: float = 1_000_000.0
    balance_round_to: int = 1_000

    # Risk scaling bounds
    risk_multiplier_bounds: tuple[float, float] = (0.40, 4.00)

    # Target horizons
    horizon_3m: int = 3
    horizon_6m: int = 6
    horizon_12m: int = 12

    # Anomalies
    anomaly_rate: float = 0.025

    # Servicer feed
    servicer_sample_rate: float = 0.25
    servicer_balance_conflict_rate: float = 0.18
    servicer_status_conflict_rate: float = 0.10

    # Output
    output_dir: Path = field(default=Path("data"))
    write_parquet: bool = False

    def correlation_matrix(self) -> np.ndarray:
        """Correlation matrix for (credit_score, ltv, dti)."""
        return np.array(
            [
                [1.0, self.rho_credit_ltv, self.rho_credit_dti],
                [self.rho_credit_ltv, 1.0, self.rho_ltv_dti],
                [self.rho_credit_dti, self.rho_ltv_dti, 1.0],
            ],
            dtype=np.float64,
        )
