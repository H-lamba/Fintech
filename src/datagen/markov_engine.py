"""
Phase 2 -- longitudinal simulation via a discrete-time Markov chain.

Performance note
----------------
A Markov chain is sequential in *time* but embarrassingly parallel across
*loans*. The engine therefore holds one state vector of length n_loans and
advances every loan through one month per iteration, so the Python-level loop
runs `max_months_on_book` times (60) rather than n_loans times (50,000).

Sampling a categorical draw per loan is done with the inverse-CDF trick on a
pre-computed cumulative transition tensor: one `cumsum`, one uniform draw, one
comparison-and-sum per timestep. This is why multiprocessing is not used here --
the vectorised form already runs the whole cohort in a few seconds, and process
pools would spend more on pickling the state than they would save.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ABSORBING_STATES,
    BASE_TRANSITION_MATRIX,
    CREDIT_PREPAY_MULTIPLIER,
    CREDIT_RISK_MULTIPLIER,
    CURE_TRANSITIONS,
    DOWNWARD_TRANSITIONS,
    GenerationConfig,
    LOSS_SEVERITY_BANDS,
    LTV_RISK_MULTIPLIER,
    PREPAY_TRANSITIONS,
    S_60,
    S_90,
    S_CURRENT,
    S_DEFAULT,
    S_PREPAID,
    STATES,
)


class TransitionMatrixBuilder:
    """
    Builds a per-loan (n_loans, 6, 6) transition tensor from the baseline
    matrix, perturbed by borrower risk.

    Scaling is multiplicative on the hazard entries followed by row
    renormalisation, which keeps every row a valid probability distribution
    without needing the multipliers to be chosen so that they happen to sum
    to one.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.cfg = config
        self._validate_base()

    @staticmethod
    def _validate_base() -> None:
        sums = BASE_TRANSITION_MATRIX.sum(axis=1)
        if not np.allclose(sums, 1.0):
            raise ValueError(f"Baseline transition rows must sum to 1.0, got {sums}")

    def build(self, static: pd.DataFrame) -> np.ndarray:
        n = len(static)
        lo, hi = self.cfg.risk_multiplier_bounds

        credit_mult = static["credit_score_band"].map(CREDIT_RISK_MULTIPLIER).to_numpy(float)
        ltv_mult = static["ltv_band"].map(LTV_RISK_MULTIPLIER).to_numpy(float)
        risk = np.clip(credit_mult * ltv_mult, lo, hi)

        # Cure suppression is the inverse square root of the risk multiplier:
        # a 4x deterioration hazard halves the cure rate rather than quartering
        # it, which matches observed servicer cure behaviour more closely than
        # a symmetric inverse.
        cure_mult = np.clip(1.0 / np.sqrt(risk), 0.35, 1.60)

        prepay_mult = static["credit_score_band"].map(CREDIT_PREPAY_MULTIPLIER).to_numpy(float)

        tensor = np.broadcast_to(BASE_TRANSITION_MATRIX, (n, 6, 6)).copy()

        for src, dst in DOWNWARD_TRANSITIONS:
            tensor[:, src, dst] *= risk
        for src, dst in CURE_TRANSITIONS:
            tensor[:, src, dst] *= cure_mult
        for src, dst in PREPAY_TRANSITIONS:
            tensor[:, src, dst] *= prepay_mult

        # Absorbing rows must stay exactly absorbing.
        for absorbing in ABSORBING_STATES:
            tensor[:, absorbing, :] = 0.0
            tensor[:, absorbing, absorbing] = 1.0

        row_sums = tensor.sum(axis=2, keepdims=True)
        tensor /= row_sums

        return np.cumsum(tensor, axis=2)


class PanelSimulator:
    """Simulates the monthly performance panel for the whole cohort."""

    def __init__(self, config: GenerationConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng

    # -- amortisation -------------------------------------------------------

    @staticmethod
    def _scheduled_balance(
        principal: np.ndarray,
        annual_rate: np.ndarray,
        term_months: np.ndarray,
        payments_made: np.ndarray,
    ) -> np.ndarray:
        """
        Closed-form fixed-rate amortisation.

            B_t = P * ((1+r)^N - (1+r)^t) / ((1+r)^N - 1)

        Evaluated directly rather than accumulated month by month: it is exact,
        avoids drift from repeated floating-point subtraction, and vectorises.

        `payments_made` counts only months the loan was Current, so a delinquent
        loan's balance correctly stops amortising while it is not paying.
        """
        r = annual_rate / 100.0 / 12.0
        t = np.minimum(payments_made, term_months).astype(np.float64)

        growth_n = np.power(1.0 + r, term_months)
        growth_t = np.power(1.0 + r, t)

        denom = growth_n - 1.0
        # Guard the degenerate zero-rate case with a straight-line fallback.
        safe = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        balance = principal * (growth_n - growth_t) / safe
        straight_line = principal * (1.0 - t / np.maximum(term_months, 1))
        balance = np.where(np.abs(denom) < 1e-12, straight_line, balance)

        return np.clip(balance, 0.0, None)

    # -- horizon ------------------------------------------------------------

    def _max_months(self, static: pd.DataFrame) -> np.ndarray:
        """Months observable per loan: capped by the portfolio cutoff and the book cap."""
        obs_end = pd.Period(self.cfg.observation_end, freq="M")
        orig = static["origination_month"].dt.to_period("M")
        available = np.array([(obs_end - p).n + 1 for p in orig], dtype=np.int64)
        return np.clip(
            np.minimum(available, self.cfg.max_months_on_book),
            1,
            self.cfg.max_months_on_book,
        )

    # -- simulation ---------------------------------------------------------

    def simulate(self, static: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        n = len(static)

        cum_trans = TransitionMatrixBuilder(cfg).build(static)

        principal = static["original_balance"].to_numpy(np.float64)
        rate = static["interest_rate"].to_numpy(np.float64)
        term = static["original_term_months"].to_numpy(np.int64)
        max_months = self._max_months(static)

        loan_index = np.arange(n)
        state = np.zeros(n, dtype=np.int8)          # everyone starts Current
        payments_made = np.zeros(n, dtype=np.int64)
        modified = np.zeros(n, dtype=bool)
        active = np.ones(n, dtype=bool)

        chunks: list[dict[str, np.ndarray]] = []

        for mob in range(cfg.max_months_on_book):
            # A loan is observable this month if it hasn't been absorbed and
            # hasn't run past its own observation horizon.
            observable = active & (mob < max_months)
            if not observable.any():
                break

            idx = loan_index[observable]
            cur_state = state[idx]

            balance = self._scheduled_balance(
                principal[idx], rate[idx], term[idx], payments_made[idx]
            )
            # Absorbing terminal rows carry a zero balance by definition.
            terminal = np.isin(cur_state, ABSORBING_STATES)
            balance = np.where(terminal, 0.0, balance)

            remaining_term = np.maximum(term[idx] - mob, 0)

            chunks.append(
                {
                    "loan_row": idx,
                    "month_index": np.full(idx.size, mob, dtype=np.int16),
                    "state_code": cur_state.copy(),
                    "current_balance": np.round(balance, 2),
                    "remaining_term_months": remaining_term.astype(np.int16),
                    "modification_flag": modified[idx].copy(),
                }
            )

            # --- advance one month -----------------------------------------
            draws = self.rng.random(idx.size)
            cdf = cum_trans[idx, cur_state]                      # (m, 6)
            nxt = (draws[:, None] > cdf).sum(axis=1).astype(np.int8)
            np.clip(nxt, 0, len(STATES) - 1, out=nxt)

            # A loan that was Current this month made its payment.
            payments_made[idx] += (cur_state == S_CURRENT).astype(np.int64)

            # Curing from serious delinquency is often a modification.
            deep = np.isin(cur_state, (S_60, S_90))
            cured = deep & (nxt == S_CURRENT)
            newly_modified = cured & (self.rng.random(idx.size) < 0.45)
            modified[idx[newly_modified]] = True

            state[idx] = nxt
            # Loans absorbed *this* month emit one final row next iteration,
            # then stop: mark them inactive only after that row is written.
            state_is_absorbing = np.isin(cur_state, ABSORBING_STATES)
            active[idx[state_is_absorbing]] = False

        panel = self._assemble(chunks, static)
        return panel

    # -- assembly -----------------------------------------------------------

    def _assemble(self, chunks: list[dict[str, np.ndarray]], static: pd.DataFrame) -> pd.DataFrame:
        if not chunks:
            return pd.DataFrame()

        stacked = {
            key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]
        }
        loan_row = stacked.pop("loan_row")

        panel = pd.DataFrame(stacked)
        panel["loan_id"] = static["loan_id"].to_numpy()[loan_row]

        origination = static["origination_month"].to_numpy()[loan_row]
        panel["origination_month"] = origination
        panel["reporting_month"] = (
            pd.PeriodIndex(pd.DatetimeIndex(origination).to_period("M"))
            + panel["month_index"].to_numpy()
        ).to_timestamp()

        panel["loan_age_months"] = panel["month_index"].astype(np.int16)
        panel["current_status"] = np.array(STATES, dtype=object)[panel["state_code"].to_numpy()]

        panel["days_past_due"] = self._days_past_due(panel["state_code"].to_numpy())
        panel["default_flag"] = (panel["state_code"] == S_DEFAULT).astype(np.int8)
        panel["prepayment_flag"] = (panel["state_code"] == S_PREPAID).astype(np.int8)

        panel["loss_severity_band"] = self._loss_severity(panel["state_code"].to_numpy())

        # Carry through the origination-level attributes the panel needs.
        carry = [
            "original_balance", "interest_rate", "credit_score_band", "ltv_band",
            "dti_band", "state", "loan_purpose", "property_type", "occupancy_type",
            "servicer_name", "source_system", "document_status", "vintage_year",
        ]
        for col in carry:
            panel[col] = static[col].to_numpy()[loan_row]

        panel["last_updated_at"] = panel["reporting_month"] + pd.to_timedelta(
            self.rng.integers(2, 26, size=len(panel)), unit="D"
        )

        panel = panel.sort_values(["loan_id", "month_index"], kind="stable").reset_index(drop=True)
        return panel

    def _days_past_due(self, state_code: np.ndarray) -> np.ndarray:
        """
        Days past due, jittered within the band implied by the state.

        A deterministic state->DPD map would make `days_past_due` a perfect
        surrogate for `current_status` and hand any model a trivially
        redundant feature; the jitter keeps them consistent but not identical.
        """
        n = state_code.size
        dpd = np.zeros(n, dtype=np.int16)
        lookup = {1: (30, 59), 2: (60, 89), 3: (90, 119), 4: (120, 210)}
        for code, (low, high) in lookup.items():
            mask = state_code == code
            count = int(mask.sum())
            if count:
                dpd[mask] = self.rng.integers(low, high + 1, size=count)
        return dpd

    def _loss_severity(self, state_code: np.ndarray) -> np.ndarray:
        """Populated only on Default rows -- null elsewhere by design."""
        out = np.full(state_code.size, None, dtype=object)
        mask = state_code == S_DEFAULT
        count = int(mask.sum())
        if count:
            bands = list(LOSS_SEVERITY_BANDS.keys())
            probs = np.array(list(LOSS_SEVERITY_BANDS.values()))
            out[mask] = self.rng.choice(bands, size=count, p=probs / probs.sum())
        return out
