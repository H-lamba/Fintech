"""
End-to-end orchestration of the five-phase generation blueprint.

Concurrency note
----------------
The simulation itself is not parallelised across processes: it is already
vectorised across the cohort, so a process pool would spend more time pickling
50,000-row state arrays than it would recover. Where concurrency *does* pay is
the final write -- three large, independent CSV serialisations that release the
GIL inside pandas' C writer. Those run on a thread pool.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .anomalies import AnomalyInjector, ServicerFeedSimulator
from .artifacts import ArtifactWriter, MacroScenarioBuilder
from .config import GenerationConfig
from .markov_engine import PanelSimulator
from .static_attributes import StaticAttributeGenerator
from .targets import TargetEngineer

TARGET_COLUMNS: tuple[str, ...] = (
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
    "next_state",
    "exception_required",
    "exception_type",
)

PANEL_COLUMN_ORDER: tuple[str, ...] = (
    "loan_id", "month_index", "reporting_month", "origination_month",
    "loan_age_months", "remaining_term_months", "original_balance",
    "current_balance", "interest_rate", "credit_score_band", "ltv_band",
    "dti_band", "state", "loan_purpose", "property_type", "occupancy_type",
    "servicer_name", "vintage_year", "current_status", "days_past_due",
    "modification_flag", "prepayment_flag", "default_flag",
    "loss_severity_band", "last_updated_at", "source_system", "document_status",
    *TARGET_COLUMNS,
)


@dataclass
class GenerationResult:
    """Everything one run produced, plus timing and diagnostics."""

    static: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    servicer: pd.DataFrame
    scenarios: pd.DataFrame
    anomaly_ledger: pd.DataFrame
    target_summary: pd.DataFrame
    correlations: pd.DataFrame
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict[str, object] = field(default_factory=dict)


class _Stopwatch:
    """Records wall-clock per phase."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._label: str | None = None
        self._start: float = 0.0

    def __call__(self, label: str) -> "_Stopwatch":
        self._label = label
        return self

    def __enter__(self) -> "_Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self._label:
            self.timings[self._label] = time.perf_counter() - self._start


class SyntheticDataPipeline:
    """Runs Phases 1-5 and writes the full suite to disk."""

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.cfg = config or GenerationConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.clock = _Stopwatch()

    # -- phases -------------------------------------------------------------

    def _phase1_static(self) -> pd.DataFrame:
        generator = StaticAttributeGenerator(self.cfg, self.rng)
        with self.clock("phase_1_static_attributes"):
            static = generator.generate()
        return static

    def _phase2_panel(self, static: pd.DataFrame) -> pd.DataFrame:
        simulator = PanelSimulator(self.cfg, self.rng)
        with self.clock("phase_2_markov_simulation"):
            panel = simulator.simulate(static)
        return panel

    def _phase3_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        engineer = TargetEngineer(self.cfg)
        with self.clock("phase_3_target_engineering"):
            panel = engineer.engineer(panel)
        return panel

    def _phase4_anomalies(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        injector = AnomalyInjector(self.cfg, self.rng)
        feed_simulator = ServicerFeedSimulator(self.cfg, self.rng)
        with self.clock("phase_4_anomaly_injection"):
            panel, ledger = injector.inject(panel)
            servicer, feed_stats = feed_simulator.build(panel)
        return panel, ledger, servicer, feed_stats

    def _phase5_split(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = pd.Timestamp(self.cfg.split_date)
        with self.clock("phase_5_partitioning"):
            ordered = [c for c in PANEL_COLUMN_ORDER if c in panel.columns]
            panel = panel[ordered]

            train = panel[panel["reporting_month"] < cutoff].copy()
            test = panel[panel["reporting_month"] >= cutoff].copy()
            test = test.drop(columns=[c for c in TARGET_COLUMNS if c in test.columns])
        return train.reset_index(drop=True), test.reset_index(drop=True)

    # -- io -----------------------------------------------------------------

    def _write(self, result: GenerationResult) -> list[Path]:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        jobs = [
            (result.train, out / "loan_monthly_performance_train.csv"),
            (result.test, out / "loan_monthly_performance_test.csv"),
            (result.static, out / "loan_static_attributes.csv"),
            (result.servicer, out / "servicer_updates.csv"),
        ]

        with self.clock("write_csv"):
            with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
                list(pool.map(lambda job: job[0].to_csv(job[1], index=False), jobs))

        written = [path for _, path in jobs]
        written += ArtifactWriter(out).write_all(result.scenarios, result.test)

        result.anomaly_ledger.to_csv(out / "_injected_anomalies.csv", index=False)
        written.append(out / "_injected_anomalies.csv")

        if self.cfg.write_parquet:
            with self.clock("write_parquet"):
                result.train.to_parquet(out / "loan_monthly_performance_train.parquet", index=False)
                result.test.to_parquet(out / "loan_monthly_performance_test.parquet", index=False)

        return written

    # -- public API ---------------------------------------------------------

    def run(self, write: bool = True) -> GenerationResult:
        static = self._phase1_static()
        panel = self._phase2_panel(static)
        panel = self._phase3_targets(panel)
        panel, ledger, servicer, feed_stats = self._phase4_anomalies(panel)
        train, test = self._phase5_split(panel)

        scenarios = MacroScenarioBuilder(self.cfg, self.rng).build()

        result = GenerationResult(
            static=static,
            train=train,
            test=test,
            servicer=servicer,
            scenarios=scenarios,
            anomaly_ledger=ledger,
            target_summary=TargetEngineer.summarise(panel),
            correlations=StaticAttributeGenerator.achieved_correlations(static),
            timings=self.clock.timings,
        )

        result.stats = {
            "n_loans": len(static),
            "n_panel_rows": len(panel),
            "n_train_rows": len(train),
            "n_test_rows": len(test),
            "mean_months_on_book": round(len(panel) / max(len(static), 1), 2),
            "terminal_default_loans": int(
                panel.groupby("loan_id")["current_status"].last().eq("Default").sum()
            ),
            "terminal_prepaid_loans": int(
                panel.groupby("loan_id")["current_status"].last().eq("Prepaid").sum()
            ),
            "anomaly_rows": int(panel["exception_required"].sum()),
            "anomaly_rate_pct": round(100.0 * panel["exception_required"].mean(), 3),
            **feed_stats,
        }

        if write:
            written = self._write(result)
            result.stats["files_written"] = [p.name for p in written]

        result.timings["total"] = sum(
            v for k, v in self.clock.timings.items() if k != "total"
        )
        return result
