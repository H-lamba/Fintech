"""
Phase 1 -- construction of the static attribute space.

Correlated continuous attributes are drawn through a Gaussian copula: sample a
multivariate normal with the target correlation structure, map each margin to
uniform via the normal CDF, then invert through a *truncated* normal for that
margin. This preserves the dependence structure while respecting hard bounds,
which naive sample-then-clip does not -- clipping piles mass on the boundaries
and attenuates the correlation you asked for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    CREDIT_BANDS,
    CREDIT_EDGES,
    CREDIT_RATE_PREMIUM,
    DOCUMENT_STATUSES,
    DTI_BANDS,
    DTI_EDGES,
    GenerationConfig,
    LOAN_PURPOSES,
    LTV_BANDS,
    LTV_EDGES,
    OCCUPANCY_TYPES,
    ORIGINAL_TERMS,
    PROPERTY_TYPES,
    SERVICERS,
    SOURCE_SYSTEMS,
    STATE_WEIGHTS,
    VINTAGE_BASE_RATE,
)


class StaticAttributeGenerator:
    """Builds the origination-level attribute table."""

    def __init__(self, config: GenerationConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng

    # -- identifiers --------------------------------------------------------

    def _loan_ids(self) -> np.ndarray:
        """
        12-character alphanumeric identifiers, drawn without replacement.

        Sampling integers from a space vastly larger than the cohort and
        base-36 encoding them is far cheaper than uuid4() per row, and the
        collision check is exact rather than probabilistic.
        """
        n = self.cfg.n_loans
        alphabet = np.array(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        space = len(alphabet) ** 12

        draws = self.rng.choice(space, size=n, replace=False)
        digits = np.empty((n, 12), dtype=np.int64)
        remainder = draws.copy()
        for position in range(11, -1, -1):
            digits[:, position] = remainder % 36
            remainder //= 36

        chars = alphabet[digits]
        return np.array(["".join(row) for row in chars], dtype=object)

    # -- copula -------------------------------------------------------------

    def _gaussian_copula(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw correlated (credit_score, ltv, dti) respecting each margin's bounds."""
        n = self.cfg.n_loans
        corr = self.cfg.correlation_matrix()

        # Cholesky requires positive-definiteness; fail loudly rather than
        # silently producing a different dependence structure.
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:  # pragma: no cover
            raise ValueError(
                f"Correlation matrix is not positive definite:\n{corr}"
            ) from exc

        latent = self.rng.standard_normal((n, 3)) @ chol.T
        uniforms = stats.norm.cdf(latent)

        def _truncated(u: np.ndarray, params: tuple[float, float, float, float]) -> np.ndarray:
            mean, std, low, high = params
            a, b = (low - mean) / std, (high - mean) / std
            return stats.truncnorm.ppf(u, a, b, loc=mean, scale=std)

        credit = _truncated(uniforms[:, 0], self.cfg.credit_score_params)
        ltv = _truncated(uniforms[:, 1], self.cfg.ltv_params)
        dti = _truncated(uniforms[:, 2], self.cfg.dti_params)
        return credit, ltv, dti

    # -- helpers ------------------------------------------------------------

    def _weighted_choice(self, mapping: dict, n: int) -> np.ndarray:
        keys = list(mapping.keys())
        probs = np.array(list(mapping.values()), dtype=np.float64)
        probs = probs / probs.sum()
        return self.rng.choice(keys, size=n, p=probs)

    def _origination_months(self) -> pd.Series:
        start = pd.Period(self.cfg.origination_start, freq="M")
        end = pd.Period(self.cfg.origination_end, freq="M")
        span = (end - start).n + 1
        offsets = self.rng.integers(0, span, size=self.cfg.n_loans)
        periods = pd.PeriodIndex([start + int(o) for o in offsets], freq="M")
        return pd.Series(periods.to_timestamp())

    def _original_balance(self) -> np.ndarray:
        cfg = self.cfg
        mu = np.log(cfg.balance_log_median)
        raw = self.rng.lognormal(mean=mu, sigma=cfg.balance_log_sigma, size=cfg.n_loans)
        clipped = np.clip(raw, cfg.balance_min, cfg.balance_max)
        rounded = np.round(clipped / cfg.balance_round_to) * cfg.balance_round_to
        return np.clip(rounded, cfg.balance_min, cfg.balance_max)

    def _interest_rate(self, vintage_year: np.ndarray, credit_band: np.ndarray) -> np.ndarray:
        base = np.array([VINTAGE_BASE_RATE[int(y)] for y in vintage_year])
        premium = np.array([CREDIT_RATE_PREMIUM[b] for b in credit_band])
        noise = self.rng.normal(0.0, 0.18, size=len(base))
        return np.round(np.clip(base + premium + noise, 2.25, 11.0), 3)

    # -- public API ---------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        n = self.cfg.n_loans

        credit_score, ltv, dti = self._gaussian_copula()

        credit_band = pd.cut(credit_score, bins=CREDIT_EDGES, labels=CREDIT_BANDS, right=False)
        ltv_band = pd.cut(ltv, bins=LTV_EDGES, labels=LTV_BANDS, right=False)
        dti_band = pd.cut(dti, bins=DTI_EDGES, labels=DTI_BANDS, right=False)

        origination_month = self._origination_months()
        vintage_year = origination_month.dt.year.to_numpy()

        static = pd.DataFrame(
            {
                "loan_id": self._loan_ids(),
                "origination_month": origination_month,
                "vintage_year": vintage_year,
                "vintage_quarter": origination_month.dt.to_period("Q").astype(str),
                "credit_score": np.round(credit_score).astype(int),
                "credit_score_band": credit_band.astype(str),
                "ltv": np.round(ltv, 2),
                "ltv_band": ltv_band.astype(str),
                "dti": np.round(dti, 2),
                "dti_band": dti_band.astype(str),
                "original_balance": self._original_balance(),
                "original_term_months": self._weighted_choice(ORIGINAL_TERMS, n).astype(int),
                "state": self._weighted_choice(STATE_WEIGHTS, n),
                "loan_purpose": self._weighted_choice(LOAN_PURPOSES, n),
                "property_type": self._weighted_choice(PROPERTY_TYPES, n),
                "occupancy_type": self._weighted_choice(OCCUPANCY_TYPES, n),
                "servicer_name": self.rng.choice(list(SERVICERS), size=n),
                "source_system": self._weighted_choice(SOURCE_SYSTEMS, n),
                "document_status": self._weighted_choice(DOCUMENT_STATUSES, n),
            }
        )

        static["interest_rate"] = self._interest_rate(
            vintage_year, static["credit_score_band"].to_numpy()
        )
        return static

    # -- diagnostics --------------------------------------------------------

    @staticmethod
    def achieved_correlations(static: pd.DataFrame) -> pd.DataFrame:
        """Realised Pearson correlations, for verifying the copula did its job."""
        cols = ["credit_score", "ltv", "dti"]
        return static[cols].corr().round(4)
