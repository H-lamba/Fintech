"""
Loading what the dashboard shows.

Every number on the page is read from a file the pipeline wrote. Nothing is
recomputed here and nothing is hardcoded: if a phase has not been run, its
loader returns an empty frame and the page says so, rather than showing a stale
figure from a previous run. That is the failure a dashboard makes easiest and
most damaging -- a demo that confidently displays last week's numbers.

Loaders are cached by Streamlit on the file's modification time, so re-running
a phase in another terminal refreshes the app without a restart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config

REPORTS = config.REPORTS_DIR
ROOT = config.PROJECT_ROOT


def _stamp(path: Path) -> float:
    """Modification time, so the cache invalidates when a phase re-runs."""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def read_csv(relative: str) -> pd.DataFrame:
    """Read a pipeline output, returning an empty frame if the phase has not run."""
    path = ROOT / relative
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_text(relative: str) -> str:
    try:
        return (ROOT / relative).read_text()
    except Exception:
        return ""


def read_jsonl(relative: str, limit: int | None = None) -> list[dict]:
    import json

    path = ROOT / relative
    if not path.exists():
        return []
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:] if limit else records


def figures(relative_dir: str) -> list[tuple[str, Path]]:
    """(title, path) for every figure in a directory, sorted."""
    directory = ROOT / relative_dir
    if not directory.exists():
        return []
    return [
        (path.stem.replace("_", " ").replace("cif ", "cumulative incidence ").title(), path)
        for path in sorted(directory.glob("*.png"))
    ]


# --------------------------------------------------------------------------
# The deliverables checklist
# --------------------------------------------------------------------------
DELIVERABLES = [
    ("Data intelligence report", "reports/data_intelligence_report.md", "Task 1"),
    ("Prediction results, baseline vs improved", "reports/task2_model_results.md", "Task 2"),
    ("Survival / competing-risk report", "reports/survival_report.md", "Task 3"),
    ("Anomaly examples (20+ reviewer-ready)", "reports/anomaly_examples.csv", "Task 4"),
    ("Scenario & stress report", "reports/scenario_report.md", "Task 5"),
    ("Explainability report", "reports/explainability_report.md", "Task 6"),
    ("LLM prompt audit log", "reports/llm_prompt_log.jsonl", "Task 7"),
    ("AI development log", "ai_dev_log/log.md", "Task 8"),
    ("Model card", "reports/model_card.md", "Section 11"),
    ("submission.csv", "submission/submission.csv", "Section 11"),
]


def deliverables() -> pd.DataFrame:
    """Which required outputs exist, resolved against the filesystem right now."""
    rows = []
    for name, path, task in DELIVERABLES:
        full = ROOT / path
        rows.append(
            {
                "Deliverable": name,
                "Task": task,
                "Present": full.exists(),
                "Path": path,
                "Size": f"{full.stat().st_size / 1024:,.0f} KB" if full.exists() else "-",
            }
        )
    return pd.DataFrame(rows)


def pipeline_state() -> dict:
    """
    Which phases have produced output, for the "run this first" hints.

    A dashboard that renders an empty panel with no explanation sends the user
    to read the code. One that says "run `make survival`" does not.
    """
    checks = {
        "profiling": "reports/data_intelligence_report.md",
        "prediction": "reports/task2_model_results.csv",
        "survival": "reports/survival/model_comparison.csv",
        "anomaly": "reports/anomaly_examples.csv",
        "scenario": "reports/scenario_report.csv",
        "explainability": "reports/explainability_report/global_importance.csv",
        "copilot": "reports/llm_prompt_log.jsonl",
        "submission": "submission/submission.csv",
    }
    return {phase: (ROOT / path).exists() for phase, path in checks.items()}
