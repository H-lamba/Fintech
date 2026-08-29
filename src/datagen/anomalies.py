"""
Phase 4 -- strategic anomaly injection and servicer conflict simulation.

Ordering matters: anomalies are injected *after* target engineering. Two of the
four corrupt the panel's temporal structure (Time Travel reorders reporting
months, Zombie Loan appends rows past an absorbing state), so computing targets
afterwards would propagate the corruption into the labels instead of leaving it
as a detectable defect.

Every injected row is labelled, giving a ground-truth set to score an anomaly
detector against -- which is the whole point of injecting them deliberately
rather than hoping the generator produced some by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ANOMALY_MIX,
    GenerationConfig,
    S_90,
    S_CURRENT,
    STATES,
)


class AnomalyInjector:
    """Injects labelled data-quality defects into the clean panel."""

    def __init__(self, config: GenerationConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        self.ledger: dict[str, int] = {}

    # -- helpers ------------------------------------------------------------

    def _allocate(self, n_rows: int) -> dict[str, int]:
        total = int(round(n_rows * self.cfg.anomaly_rate))
        mix = {k: v for k, v in ANOMALY_MIX.items()}
        weights = np.array(list(mix.values()), dtype=float)
        weights /= weights.sum()
        counts = (weights * total).astype(int)
        return dict(zip(mix.keys(), counts))

    @staticmethod
    def _init_labels(panel: pd.DataFrame) -> pd.DataFrame:
        panel["exception_required"] = False
        panel["exception_type"] = "None"
        return panel

    def _mark(self, panel: pd.DataFrame, positions: np.ndarray, label: str) -> None:
        panel.loc[positions, "exception_required"] = True
        panel.loc[positions, "exception_type"] = label
        self.ledger[label] = self.ledger.get(label, 0) + int(positions.size)

    def _available(self, panel: pd.DataFrame, mask: np.ndarray, count: int) -> np.ndarray:
        """Pick `count` unlabelled row positions satisfying `mask`."""
        eligible = np.flatnonzero(mask & (~panel["exception_required"].to_numpy()))
        if eligible.size == 0 or count <= 0:
            return np.array([], dtype=np.int64)
        chosen = self.rng.choice(eligible, size=min(count, eligible.size), replace=False)
        return panel.index.to_numpy()[chosen]

    # -- individual anomalies -----------------------------------------------

    def _balance_discrepancy(self, panel: pd.DataFrame, count: int) -> None:
        """Balance inflated above origination with no modification to justify it."""
        mask = (panel["current_balance"].to_numpy() > 0)
        rows = self._available(panel, mask, count)
        if rows.size == 0:
            return
        inflation = self.rng.uniform(1.05, 1.35, size=rows.size)
        panel.loc[rows, "current_balance"] = np.round(
            panel.loc[rows, "original_balance"].to_numpy() * inflation, 2
        )
        panel.loc[rows, "modification_flag"] = False
        self._mark(panel, rows, "Balance Discrepancy")

    def _time_travel(self, panel: pd.DataFrame, count: int) -> None:
        """Reporting month precedes origination -- physically impossible."""
        rows = self._available(panel, np.ones(len(panel), dtype=bool), count)
        if rows.size == 0:
            return
        back = self.rng.integers(1, 13, size=rows.size)
        origination = panel.loc[rows, "origination_month"]
        shifted = (
            origination.dt.to_period("M").to_numpy()
            - back
        )
        panel.loc[rows, "reporting_month"] = pd.PeriodIndex(shifted, freq="M").to_timestamp()
        self._mark(panel, rows, "Time Travel")

    def _impossible_transition(self, panel: pd.DataFrame, count: int) -> None:
        """
        A jump straight from Current to 90-DPD in a single month, skipping the
        30 and 60 buckets the delinquency ladder requires.
        """
        state = panel["state_code"].to_numpy()
        loan = panel["loan_id"].to_numpy()

        prev_state = np.roll(state, 1)
        prev_loan = np.roll(loan, 1)
        same_loan = np.zeros(len(panel), dtype=bool)
        same_loan[1:] = loan[1:] == prev_loan[1:]

        mask = same_loan & (prev_state == S_CURRENT) & (state == S_CURRENT)
        rows = self._available(panel, mask, count)
        if rows.size == 0:
            return

        panel.loc[rows, "state_code"] = np.int8(S_90)
        panel.loc[rows, "current_status"] = STATES[S_90]
        panel.loc[rows, "days_past_due"] = self.rng.integers(
            90, 120, size=rows.size
        ).astype(panel["days_past_due"].dtype)
        self._mark(panel, rows, "Impossible State Transition")

    def _zombie_loans(self, panel: pd.DataFrame, count: int) -> pd.DataFrame:
        """
        Active payment rows appearing after an absorbing event -- the classic
        symptom of a servicer feed replaying stale records.
        """
        terminal = panel[panel["current_status"].isin(("Default", "Prepaid"))]
        if terminal.empty or count <= 0:
            return panel

        n_loans = max(1, count // 2)
        picked = terminal.sample(
            n=min(n_loans, len(terminal)),
            random_state=int(self.rng.integers(0, 2**31 - 1)),
        )

        zombies = []
        for extra in (1, 2):
            block = picked.copy()
            block["month_index"] = block["month_index"] + extra
            block["loan_age_months"] = block["loan_age_months"] + extra
            block["reporting_month"] = block["reporting_month"] + pd.DateOffset(months=extra)
            block["last_updated_at"] = block["last_updated_at"] + pd.DateOffset(months=extra)
            block["state_code"] = np.int8(S_CURRENT)
            block["current_status"] = STATES[S_CURRENT]
            block["days_past_due"] = 0
            block["default_flag"] = 0
            block["prepayment_flag"] = 0
            block["loss_severity_band"] = None
            block["current_balance"] = np.round(
                block["original_balance"].to_numpy()
                * self.rng.uniform(0.55, 0.92, size=len(block)),
                2,
            )
            block["exception_required"] = True
            block["exception_type"] = "Zombie Loan"
            zombies.append(block)

        zombie_frame = pd.concat(zombies, ignore_index=True)
        self.ledger["Zombie Loan"] = self.ledger.get("Zombie Loan", 0) + len(zombie_frame)

        combined = pd.concat([panel, zombie_frame], ignore_index=True)
        return combined.sort_values(["loan_id", "month_index"], kind="stable").reset_index(drop=True)

    # -- public API ---------------------------------------------------------

    def inject(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        panel = self._init_labels(panel.copy())
        allocation = self._allocate(len(panel))

        self._balance_discrepancy(panel, allocation.get("Balance Discrepancy", 0))
        self._time_travel(panel, allocation.get("Time Travel", 0))
        self._impossible_transition(panel, allocation.get("Impossible State Transition", 0))
        panel = self._zombie_loans(panel, allocation.get("Zombie Loan", 0))

        ledger = pd.DataFrame(
            [{"anomaly_type": k, "n_rows": v} for k, v in sorted(self.ledger.items())]
        )
        return panel, ledger


class ServicerFeedSimulator:
    """
    Builds `servicer_updates.csv`: a partial, unscrubbed second source.

    Column names deliberately mirror the panel's, so a reconciliation routine
    can join on (loan_id, reporting_month) and diff like-for-like fields.
    """

    def __init__(self, config: GenerationConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng

    def build(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        cfg = self.cfg
        sample_size = int(len(panel) * cfg.servicer_sample_rate)
        feed = panel.sample(
            n=min(sample_size, len(panel)),
            random_state=int(self.rng.integers(0, 2**31 - 1)),
        )[
            ["loan_id", "reporting_month", "current_balance", "current_status",
             "days_past_due", "servicer_name"]
        ].copy()

        n = len(feed)

        balance_conflict = self.rng.random(n) < cfg.servicer_balance_conflict_rate
        drift = self.rng.uniform(1.03, 1.18, size=n)
        feed.loc[balance_conflict, "current_balance"] = np.round(
            feed.loc[balance_conflict, "current_balance"].to_numpy()
            * drift[balance_conflict],
            2,
        )

        status_conflict = self.rng.random(n) < cfg.servicer_status_conflict_rate
        alt_status = self.rng.choice(["Current", "30-DPD", "60-DPD"], size=n)
        feed.loc[status_conflict, "current_status"] = alt_status[status_conflict]

        # A quarter of the feed lags badly -- the stale-record scenario.
        stale = self.rng.random(n) < 0.22
        lag_days = np.where(stale, self.rng.integers(120, 400, size=n), self.rng.integers(1, 25, size=n))
        feed["last_updated_at"] = feed["reporting_month"] + pd.to_timedelta(lag_days, unit="D")
        feed["source_system"] = "SubServicerRemittance"

        stats = {
            "servicer_rows": n,
            "balance_conflicts": int(balance_conflict.sum()),
            "status_conflicts": int(status_conflict.sum()),
            "stale_rows": int(stale.sum()),
        }
        return feed.reset_index(drop=True), stats
