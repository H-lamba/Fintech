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
        return (ROOT / relative).read_text(encoding="utf-8")
    except Exception:
        return ""


def read_jsonl(relative: str, limit: int | None = None) -> list[dict]:
    import json

    path = ROOT / relative
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:] if limit else records


# --------------------------------------------------------------------------
# Serving the generated HTML reports inside the app
# --------------------------------------------------------------------------
# Keyed on (path, mtime) so a re-run of a phase invalidates the entry without a
# restart. A plain dict rather than st.cache_data because this module is
# imported directly by the tests, and a loader that only works inside a
# Streamlit script run is a loader that cannot be tested.
_CACHE: dict = {}


def _cached(key: str, path: Path, build):
    stamp = _stamp(path)
    hit = _CACHE.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    value = build()
    _CACHE[key] = (stamp, value)
    return value


def read_bytes(relative: str) -> bytes:
    """Raw bytes, for a download button. Empty if the phase has not run."""
    try:
        return (ROOT / relative).read_bytes()
    except OSError:
        return b""


# Streamlit serves `./static/` at `/app/static/` when `enableStaticServing` is
# on. That directory is a publishing target, not a source: everything in it is
# regenerated from `reports/`, so it is gitignored.
STATIC_DIR = ROOT / "static"
STATIC_URL_PREFIX = "app/static"


def publish_report(relative: str) -> str:
    """
    Publish a generated report to a real URL and return it.

    The self-contained copy -- figures already inlined as data URIs -- is
    written into the served static directory, so opening it in a new tab yields
    a complete standalone page rather than a wall of broken images. One file
    with no asset directory beside it is also what makes the tab survive being
    bookmarked or sent to someone.

    Republished whenever the source report is newer, so re-running a phase is
    picked up without restarting the app. Returns "" if the phase has not run.
    """
    source = ROOT / relative
    if not source.exists():
        return ""

    name = relative.rsplit("/", 1)[-1]
    target = STATIC_DIR / "reports" / name
    try:
        stale = (not target.exists()) or target.stat().st_mtime < source.stat().st_mtime
        if stale:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(report_html(relative), encoding="utf-8")
    except OSError:
        return ""
    return f"{STATIC_URL_PREFIX}/reports/{name}"


def report_html(relative: str, max_image_mb: float = 12.0) -> str:
    """
    A generated report, rewritten so it renders standalone inside an iframe.

    The reports reference their figures by *relative* path
    (``profiling/charts/drift.png``). That resolves from the reports directory
    on disk, and resolves to nothing inside a sandboxed iframe with no base
    URL -- which is why the in-app viewer showed text and broken images. Each
    referenced figure is inlined as a data URI instead.

    Inlining is capped: the explainability report carries seventeen figures and
    an uncapped inline would push several megabytes into a single component.
    Past the cap the remaining images are dropped rather than half-encoded, and
    the reader is pointed at the download, because a report that half-renders is
    worse than one that says it did not.
    """
    import base64
    import mimetypes
    import re

    path = ROOT / relative
    if not path.exists():
        return ""

    def build() -> str:
        html = path.read_text(encoding="utf-8")
        budget = int(max_image_mb * 1024 * 1024)
        spent = 0
        dropped = 0

        def replace(match: "re.Match") -> str:
            nonlocal spent, dropped
            src = match.group(2)
            if src.startswith(("data:", "http://", "https://")):
                return match.group(0)
            asset = (path.parent / src).resolve()
            try:
                raw = asset.read_bytes()
            except OSError:
                dropped += 1
                return ""
            if spent + len(raw) > budget:
                dropped += 1
                return ""
            spent += len(raw)
            mime = mimetypes.guess_type(asset.name)[0] or "image/png"
            encoded = base64.b64encode(raw).decode("ascii")
            return f'{match.group(1)}data:{mime};base64,{encoded}"'

        html = re.sub(r'(<img[^>]*\ssrc=")([^"]+)"', replace, html)
        if dropped:
            html = html.replace(
                "<body>",
                "<body><p style='background:#fdf6e8;border:1px solid #f0dcae;"
                "padding:.6rem .9rem;border-radius:8px;font:14px system-ui'>"
                f"{dropped} figure(s) omitted from this inline view to keep it "
                "loadable. Download the report for the complete version.</p>",
                1,
            )
        return html

    return _cached(f"html::{relative}", path, build)


# --------------------------------------------------------------------------
# Grounding sources for the in-app copilot
# --------------------------------------------------------------------------
def panel_row(loan_id: str, reporting_month: str) -> "pd.Series | None":
    """
    One loan-month from the unlabelled test panel, for grounding an LLM call.

    The copilot may only restate what it is given, so it needs the loan's own
    record -- not the submission row, which holds the pipeline's conclusions
    rather than the facts they were drawn from.
    """
    path = ROOT / "data/loan_monthly_performance_test.csv"
    if not path.exists():
        return None

    def build() -> pd.DataFrame:
        frame = pd.read_csv(path)
        frame["_month"] = frame["reporting_month"].astype(str).str[:7]
        return frame

    frame = _cached("panel::test", path, build)
    match = frame[
        (frame["loan_id"] == loan_id) & (frame["_month"] == str(reporting_month)[:7])
    ]
    if match.empty:
        return None
    return match.drop(columns=["_month"]).iloc[0]


def grounding_sources() -> tuple[dict, list]:
    """The data-dictionary definitions and validation rules, for context assembly."""
    from .. import data_io

    def build():
        try:
            return data_io.load_data_dictionary(), data_io.load_validation_rules()
        except Exception:
            return {}, []

    return _cached("grounding", config.DATA_DICTIONARY_PATH, build)


def copilot_status() -> tuple[bool, str]:
    """
    ``(live_available, model)`` -- whether a real provider is reachable.

    Read through the copilot's own client so the app cannot disagree with the
    pipeline about which provider is configured, and so no key material is
    handled here.
    """
    try:
        from ..copilot import llm_client

        return llm_client.has_credentials(), llm_client.LLM_MODEL
    except Exception:
        return False, ""


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
# Mirrors section 11 of the problem statement, then the per-task outputs that
# back it. Only artefacts the pipeline itself produces are listed: the demo
# video is a section 11 deliverable too, but it is recorded by hand and lives
# outside the repository, so a filesystem check can only ever report it missing.
# The "outstanding" callout on the Overview page still fires for anything here
# that genuinely fails to generate.
DELIVERABLES = [
    ("GitHub repository — complete source", "README.md", "Section 11"),
    ("Reproducible scripts — end-to-end workflow", "main.py", "Section 11"),
    ("submission.csv", "submission/submission.csv", "Section 11"),
    ("Model card", "reports/model_card.md", "Section 11"),
    ("Data intelligence report", "reports/data_intelligence_report.md", "Section 11"),
    ("Explainability report", "reports/explainability_report.md", "Section 11"),
    ("Scenario report", "reports/scenario_report.md", "Section 11"),
    ("LLM copilot demo", "reports/copilot_report.md", "Section 11"),
    ("AI Development Log", "ai_dev_log/log.md", "Section 11"),
    ("Prediction results, baseline vs improved", "reports/task2_model_results.md", "Task 2"),
    ("Survival / competing-risk report", "reports/survival_report.md", "Task 3"),
    ("Anomaly examples (20+ reviewer-ready)", "reports/anomaly_examples.csv", "Task 4"),
    ("LLM prompt audit log", "reports/llm_prompt_log.jsonl", "Task 7"),
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
