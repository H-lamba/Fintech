"""
Time-aware splitting for the loan performance panel.

The rule this module exists to enforce
--------------------------------------
**No random row-level split, anywhere.** A monthly panel repeats each loan
dozens of times; a random split puts month t of a loan in TRAIN and month t+1
of the *same* loan in TEST, so the model is scored on loans whose behaviour it
has already memorised. Reported metrics then measure recall of the training
set, not forecasting skill. :func:`random_split` does not exist here, and
:func:`audit_split` fails loudly if the windows it is handed overlap in time.

Purging
-------
Splitting on the reporting month alone is still not enough. A row labelled
``next_12m_default_flag`` at 2021-12 describes what happens through 2022-12 --
which is the validation window. Left in TRAIN, that row hands the model the
validation period's outcomes. So the last ``horizon`` months of each split are
**purged**: dropped from the earlier split rather than trusted. The cost is a
few months of training rows; the alternative is an optimistic, unusable score.

The graded holdout is carved out of the labelled panel, because the organiser's
own ``loan_monthly_performance_test.csv`` (2024-01 onward) ships without labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import config


@dataclass(frozen=True)
class SplitBoundaries:
    """Inclusive last reporting month of each window."""

    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    test_end: pd.Timestamp

    @classmethod
    def from_config(cls) -> "SplitBoundaries":
        return cls(
            train_end=pd.Timestamp(config.SPLIT_TRAIN_END),
            valid_end=pd.Timestamp(config.SPLIT_VALID_END),
            test_end=pd.Timestamp(config.SPLIT_TEST_END),
        )

    def __post_init__(self) -> None:
        if not (self.train_end < self.valid_end < self.test_end):
            raise ValueError(
                "Split boundaries must be strictly increasing in time: "
                f"train_end={self.train_end.date()}, valid_end={self.valid_end.date()}, "
                f"test_end={self.test_end.date()}"
            )


@dataclass
class TimeSplit:
    """One target's purged, time-ordered train / validation / test partition."""

    target: str
    horizon_months: int
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    audit: dict

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"TimeSplit(target={self.target!r}, horizon={self.horizon_months}m, "
            f"train={len(self.train):,}, valid={len(self.valid):,}, test={len(self.test):,})"
        )


def _month_offset(month: pd.Timestamp, months: int) -> pd.Timestamp:
    return month - pd.DateOffset(months=months)


def make_time_split(
    df: pd.DataFrame,
    target: str,
    boundaries: SplitBoundaries | None = None,
    horizon_months: int | None = None,
    drop_absorbing: bool = True,
    time_col: str = config.TIME_COL,
) -> TimeSplit:
    """
    Partition ``df`` into three time-ordered windows for one target.

    Parameters
    ----------
    horizon_months:
        Length of the target's forward window. The last ``horizon_months`` of
        TRAIN and of VALIDATION are purged so neither can see into the window
        that follows it. Defaults to ``config.TARGET_HORIZONS[target]``.
    drop_absorbing:
        Drop rows already in an absorbing state (Default / Prepaid). Their
        outcome is realised, not forecast; scoring them inflates every metric
        with rows a one-line rule already answers. They are handled by a
        deterministic override at prediction time instead.
    """
    boundaries = boundaries or SplitBoundaries.from_config()
    if horizon_months is None:
        horizon_months = config.TARGET_HORIZONS.get(target, 1)

    frame = df.copy()
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")

    n_input = len(frame)
    frame = frame[frame[time_col].notna()]

    n_absorbing = 0
    if drop_absorbing and "current_status" in frame.columns:
        absorbing = frame["current_status"].isin(config.ABSORBING_STATES)
        n_absorbing = int(absorbing.sum())
        frame = frame[~absorbing]

    labelled = frame[frame[target].notna()]
    n_censored = len(frame) - len(labelled)

    months = labelled[time_col]
    train_cut = _month_offset(boundaries.train_end, horizon_months)
    valid_cut = _month_offset(boundaries.valid_end, horizon_months)

    train = labelled[months <= train_cut]
    valid = labelled[(months > boundaries.train_end) & (months <= valid_cut)]
    test = labelled[(months > boundaries.valid_end) & (months <= boundaries.test_end)]

    audit = {
        "target": target,
        "horizon_months": horizon_months,
        "split_type": "time-aware (purged, forward-chaining)",
        "rows_input": n_input,
        "rows_absorbing_dropped": n_absorbing,
        "rows_censored_dropped": n_censored,
        "train_months": _describe_months(train[time_col]),
        "valid_months": _describe_months(valid[time_col]),
        "test_months": _describe_months(test[time_col]),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "purged_train_months": _describe_months(
            labelled.loc[(months > train_cut) & (months <= boundaries.train_end), time_col]
        ),
        "purged_valid_months": _describe_months(
            labelled.loc[(months > valid_cut) & (months <= boundaries.valid_end), time_col]
        ),
    }
    _assert_windows_populated(target, horizon_months, boundaries, train, valid, test)
    audit.update(audit_split(train, valid, test, time_col=time_col))

    return TimeSplit(
        target=target,
        horizon_months=horizon_months,
        train=train,
        valid=valid,
        test=test,
        audit=audit,
    )


def _assert_windows_populated(
    target: str,
    horizon_months: int,
    boundaries: SplitBoundaries,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Fail with a diagnosis rather than an empty-array error deep in sklearn.

    The usual cause is a window no longer than the target's forward horizon:
    purging then removes every row in it. The fix is to widen the boundaries,
    not to drop the purge.
    """
    empty = [name for name, frame in
             (("train", train), ("validation", valid), ("test", test)) if frame.empty]
    if not empty:
        return
    raise ValueError(
        f"Target {target!r} (horizon {horizon_months}m): the {', '.join(empty)} "
        f"window(s) are empty after purging. Each window must be longer than the "
        f"target's forward horizon, since its last {horizon_months} month(s) are "
        f"purged. Current boundaries: train<={boundaries.train_end:%Y-%m}, "
        f"valid<={boundaries.valid_end:%Y-%m}, test<={boundaries.test_end:%Y-%m}."
    )


def _describe_months(months: pd.Series) -> str:
    if months.empty:
        return "(empty)"
    return f"{months.min():%Y-%m} .. {months.max():%Y-%m}"


def audit_split(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    time_col: str = config.TIME_COL,
) -> dict:
    """
    Verify the partition is genuinely time-ordered, and raise if it is not.

    Loans are expected to appear in more than one window -- a panel that spans
    the boundary legitimately does -- so loan overlap is *reported*, not
    treated as an error. What is checked is that no reporting month appears in
    two windows and that the windows run strictly forward in time. Those are
    the properties a random split would violate.
    """
    windows = {"train": train, "valid": valid, "test": test}
    month_sets = {
        name: set(pd.to_datetime(frame[time_col]).unique())
        for name, frame in windows.items()
        if not frame.empty
    }

    for left, right in (("train", "valid"), ("valid", "test"), ("train", "test")):
        if left not in month_sets or right not in month_sets:
            continue
        shared = month_sets[left] & month_sets[right]
        if shared:
            raise ValueError(
                f"Time-aware split violated: {len(shared)} reporting month(s) appear in "
                f"both {left} and {right}. A month may belong to exactly one window."
            )
        if max(month_sets[left]) >= min(month_sets[right]):
            raise ValueError(
                f"Time-aware split violated: {left} extends past the start of {right}."
            )

    loans = {name: set(frame[config.ID_COL]) for name, frame in windows.items() if not frame.empty}
    return {
        "months_disjoint": True,
        "windows_ordered": True,
        "loans_train_and_test": len(loans.get("train", set()) & loans.get("test", set())),
        "note": (
            "Loan overlap across windows is expected and correct for a panel: the "
            "same loan is observed in successive months. Leakage is prevented by "
            "the time ordering plus the horizon purge, not by separating loans."
        ),
    }


def split_summary_frame(splits: dict[str, TimeSplit]) -> pd.DataFrame:
    """One row per target describing how its partition was drawn."""
    rows = []
    for target, split in splits.items():
        audit = split.audit
        rows.append(
            {
                "target": target,
                "horizon_months": audit["horizon_months"],
                "train_window": audit["train_months"],
                "train_rows": audit["train_rows"],
                "purged_from_train": audit["purged_train_months"],
                "valid_window": audit["valid_months"],
                "valid_rows": audit["valid_rows"],
                "purged_from_valid": audit["purged_valid_months"],
                "test_window": audit["test_months"],
                "test_rows": audit["test_rows"],
                "absorbing_rows_dropped": audit["rows_absorbing_dropped"],
                "censored_rows_dropped": audit["rows_censored_dropped"],
            }
        )
    return pd.DataFrame(rows)
