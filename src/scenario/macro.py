"""
Ingesting the macro scenario assumptions.

``data/macro_scenarios.csv`` is the **single source of truth** for every stress
variable. Nothing in this package invents an economic assumption: no assumed
unemployment path, no assumed rate shock, no assumed elasticity. Where a
transmission channel needs a parameter the file does not state -- how far
credit scores move for a given rise in unemployment, for instance -- that
parameter is *solved for* from the file's own stated multipliers rather than
guessed. See :mod:`src.scenario.stress`.

The file supplies, per scenario and projection month:

===================== ==========================================================
``mortgage_rate``     prevailing market rate; drives the refinance incentive
``unemployment_rate`` labour-market stress; the narrative driver of credit
``hpi_index``         house price index, rebased to 100 at the scenario start
``default_multiplier`` the scenario's own stated view of the default hazard
``prepayment_multiplier`` likewise for prepayment
===================== ==========================================================

The multipliers and the macro levels are two different views of the same
scenario, and the pipeline uses both: the levels drive the feature-space
stress, and the multipliers calibrate it and provide an independent check.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .. import config

REQUIRED_COLUMNS = (
    "scenario",
    "projection_month",
    "horizon_month",
    "mortgage_rate",
    "unemployment_rate",
    "hpi_index",
    "default_multiplier",
    "prepayment_multiplier",
)


@dataclass(frozen=True)
class ScenarioMonth:
    """One scenario's assumptions at one projection month."""

    scenario: str
    projection_month: pd.Timestamp
    horizon_month: int
    mortgage_rate: float
    unemployment_rate: float
    hpi_index: float
    default_multiplier: float
    prepayment_multiplier: float

    def multiplier_for(self, target: str) -> float | None:
        """
        The scenario's stated hazard multiplier for a projected measure.

        ``None`` where the file states none. The file covers default and
        prepayment; it says nothing about delinquency, and quietly reusing the
        default multiplier there would put a number in the report and attribute
        it to a source that never gave it.
        """
        if "prepayment" in target:
            return self.prepayment_multiplier
        if "default" in target:
            return self.default_multiplier
        return None


class MacroScenarios:
    """The scenario table, validated and indexed for lookup."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table

    @classmethod
    def load(cls, path=None) -> "MacroScenarios":
        """
        Read and validate the scenario file.

        Validation is not decoration here. A missing column or a scenario with
        no baseline month would otherwise surface as a silently wrong
        projection, and a stress projection that is quietly wrong is worse than
        one that fails.
        """
        path = path or config.MACRO_SCENARIOS_PATH
        table = pd.read_csv(path)

        missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {missing}. "
                "The scenario file is the only source of stress assumptions, so a "
                "partial file cannot be silently substituted for."
            )

        table["projection_month"] = pd.to_datetime(table["projection_month"], errors="coerce")
        if table["projection_month"].isna().any():
            raise ValueError(f"{path} contains unparseable projection_month values")

        table = table.sort_values(["scenario", "horizon_month"]).reset_index(drop=True)
        return cls(table)

    # -- access ------------------------------------------------------------
    @property
    def scenarios(self) -> list[str]:
        """Scenario names, with the baseline first where one is identifiable."""
        names = list(self.table["scenario"].unique())
        baseline = self.baseline_name
        if baseline in names:
            names = [baseline, *[n for n in names if n != baseline]]
        return names

    @property
    def baseline_name(self) -> str:
        """
        The scenario that acts as the reference.

        Identified by its multipliers being flat at 1.0 rather than by matching
        the string "Baseline", so a differently-named reference scenario still
        works.
        """
        flat = self.table.groupby("scenario")[["default_multiplier", "prepayment_multiplier"]].std()
        candidates = flat[(flat.fillna(0.0) < 1e-9).all(axis=1)].index.tolist()
        if candidates:
            return candidates[0]
        return self.table["scenario"].iloc[0]

    def horizons(self, scenario: str | None = None) -> list[int]:
        subset = self.table if scenario is None else self.table[self.table["scenario"] == scenario]
        return sorted(subset["horizon_month"].unique().tolist())

    def month(self, scenario: str, horizon: int) -> ScenarioMonth:
        rows = self.table[
            (self.table["scenario"] == scenario) & (self.table["horizon_month"] == horizon)
        ]
        if rows.empty:
            raise KeyError(f"No assumptions for scenario {scenario!r} at horizon {horizon}")
        row = rows.iloc[0]
        return ScenarioMonth(
            scenario=str(row["scenario"]),
            projection_month=pd.Timestamp(row["projection_month"]),
            horizon_month=int(row["horizon_month"]),
            mortgage_rate=float(row["mortgage_rate"]),
            unemployment_rate=float(row["unemployment_rate"]),
            hpi_index=float(row["hpi_index"]),
            default_multiplier=float(row["default_multiplier"]),
            prepayment_multiplier=float(row["prepayment_multiplier"]),
        )

    def anchor(self, scenario: str) -> ScenarioMonth:
        """The scenario's first projection month -- the point everything is relative to."""
        return self.month(scenario, min(self.horizons(scenario)))

    def summary(self) -> pd.DataFrame:
        """What each scenario assumes, at the reported horizons."""
        rows = []
        for scenario in self.scenarios:
            available = set(self.horizons(scenario))
            for horizon in config.SCENARIO_HORIZONS:
                if horizon not in available:
                    continue
                month = self.month(scenario, horizon)
                rows.append(
                    {
                        "scenario": scenario,
                        "horizon_month": horizon,
                        "projection_month": f"{month.projection_month:%Y-%m}",
                        "mortgage_rate": month.mortgage_rate,
                        "unemployment_rate": month.unemployment_rate,
                        "hpi_index": month.hpi_index,
                        "default_multiplier": month.default_multiplier,
                        "prepayment_multiplier": month.prepayment_multiplier,
                    }
                )
        return pd.DataFrame(rows)
