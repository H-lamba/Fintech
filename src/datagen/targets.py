"""
Phase 3 -- vectorised forward-looking target engineering.

No Python iteration over rows or groups. The panel is sorted by
(loan_id, month_index), so each loan occupies a contiguous block; a
forward-looking window is therefore a plain array shift, and the only care
needed is invalidating shifts that would run past a loan's own last row into
the next loan's first row.

Censoring
---------
A target is only knowable if its whole forward window was observed. Two cases:

  * The loan reached an absorbing state (Default/Prepaid) inside the window.
    Nothing can happen afterwards, so the label is definitively known even
    though there are no further rows -- do NOT censor.
  * The loan was still active when the observation window closed. The forward
    window is genuinely unobserved -- censor to NaN.

Treating both cases as censored (the naive reading) would throw away every
resolved outcome near the cutoff and bias the observed default rate downward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ABSORBING_STATES,
    GenerationConfig,
    S_30,
    S_60,
    S_90,
    S_DEFAULT,
    S_PREPAID,
    STATES,
)


class TargetEngineer:
    """Computes the multi-outcome label set on a sorted panel."""

    def __init__(self, config: GenerationConfig) -> None:
        self.cfg = config

    # -- block geometry -----------------------------------------------------

    @staticmethod
    def _block_geometry(loan_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Position within each loan's block, and each block's length, per row.

        Computed from run-length boundaries rather than a groupby, so cost is
        one pass over the array regardless of how many loans there are.
        """
        n = loan_ids.size
        if n == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        boundary = np.empty(n, dtype=bool)
        boundary[0] = True
        boundary[1:] = loan_ids[1:] != loan_ids[:-1]

        start_positions = np.flatnonzero(boundary)
        counts = np.diff(np.append(start_positions, n))

        starts = np.repeat(start_positions, counts)
        position = np.arange(n) - starts
        block_len = np.repeat(counts, counts)
        return position, block_len

    @staticmethod
    def _shift_forward(values: np.ndarray, k: int, fill) -> np.ndarray:
        """values[i + k], with `fill` past the end of the array."""
        out = np.full(values.shape, fill, dtype=values.dtype)
        if 0 < k < values.size:
            out[:-k] = values[k:]
        elif k == 0:
            out = values.copy()
        return out

    def _any_within_horizon(
        self,
        indicator: np.ndarray,
        position: np.ndarray,
        block_len: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        True where `indicator` is set at any offset 1..horizon ahead, staying
        inside the loan's own block.
        """
        result = np.zeros(indicator.size, dtype=bool)
        for k in range(1, horizon + 1):
            shifted = self._shift_forward(indicator, k, False)
            in_block = (position + k) < block_len
            result |= shifted & in_block
        return result

    # -- public API ---------------------------------------------------------

    def engineer(self, panel: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        panel = panel.sort_values(["loan_id", "month_index"], kind="stable").reset_index(drop=True)

        loan_ids = panel["loan_id"].to_numpy()
        state_code = panel["state_code"].to_numpy()
        position, block_len = self._block_geometry(loan_ids)

        # Terminal condition per loan: did its final row sit in an absorbing state?
        last_row = position == (block_len - 1)
        # Broadcast each block's final state back across all its rows.
        resolved_per_row = np.repeat(state_code[last_row], block_len[last_row])
        loan_resolved = np.isin(resolved_per_row, ABSORBING_STATES)

        is_30_plus = np.isin(state_code, (S_30, S_60, S_90, S_DEFAULT))
        is_60_plus = np.isin(state_code, (S_60, S_90, S_DEFAULT))
        is_default = state_code == S_DEFAULT
        is_prepaid = state_code == S_PREPAID

        targets = {
            "next_3m_delinquency_flag": (
                self._any_within_horizon(is_30_plus, position, block_len, cfg.horizon_3m),
                cfg.horizon_3m,
            ),
            "next_6m_delinquency_flag": (
                self._any_within_horizon(is_60_plus, position, block_len, cfg.horizon_6m),
                cfg.horizon_6m,
            ),
            "next_12m_default_flag": (
                self._any_within_horizon(is_default, position, block_len, cfg.horizon_12m),
                cfg.horizon_12m,
            ),
            "next_12m_prepayment_flag": (
                self._any_within_horizon(is_prepaid, position, block_len, cfg.horizon_12m),
                cfg.horizon_12m,
            ),
        }

        for name, (values, horizon) in targets.items():
            window_complete = (position + horizon) < block_len
            censored = ~window_complete & ~loan_resolved
            column = values.astype("float64")
            column[censored] = np.nan
            panel[name] = column

        panel["next_state"] = self._next_state(state_code, position, block_len, loan_resolved)

        return panel

    def _next_state(
        self,
        state_code: np.ndarray,
        position: np.ndarray,
        block_len: np.ndarray,
        loan_resolved: np.ndarray,
    ) -> np.ndarray:
        """
        Exact state at t+1.

        On the final row of an absorbed loan the chain stays in its absorbing
        state forever, so t+1 is that same state -- a real label, not a gap.
        On the final row of a censored loan, t+1 is genuinely unobserved.
        """
        shifted = self._shift_forward(state_code, 1, np.int8(0))
        in_block = (position + 1) < block_len

        out = np.full(state_code.size, None, dtype=object)
        state_names = np.array(STATES, dtype=object)

        out[in_block] = state_names[shifted[in_block]]

        tail_resolved = ~in_block & loan_resolved
        out[tail_resolved] = state_names[state_code[tail_resolved]]

        return out

    # -- diagnostics --------------------------------------------------------

    @staticmethod
    def summarise(panel: pd.DataFrame) -> pd.DataFrame:
        """Base rate and censored share per target -- a sanity check worth printing."""
        rows = []
        for col in (
            "next_3m_delinquency_flag",
            "next_6m_delinquency_flag",
            "next_12m_default_flag",
            "next_12m_prepayment_flag",
        ):
            if col not in panel.columns:
                continue
            series = panel[col]
            observed = series.dropna()
            rows.append(
                {
                    "target": col,
                    "n_observed": int(observed.size),
                    "n_censored": int(series.isna().sum()),
                    "pct_censored": round(100.0 * series.isna().mean(), 2),
                    "positive_rate_pct": round(100.0 * observed.mean(), 3) if observed.size else np.nan,
                }
            )
        return pd.DataFrame(rows)
