"""
Regression tests for the dashboard.

A Streamlit page that raises is a broken deliverable, and nothing else in the
suite imports these modules -- so without this file the app can break silently
on a Streamlit upgrade or a renamed pipeline output and nobody finds out until
the demo. Every page is rendered through Streamlit's own test harness, which is
the only way to catch an API change like ``height=None`` becoming invalid.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.dashboard import data, theme

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

# AppTest resolves a relative path against the *test file's* directory, not the
# working directory, so the entrypoint is addressed absolutely.
APP = str(config.PROJECT_ROOT / "app.py")

PAGES = [
    "Overview", "Data intelligence", "Features & split", "Prediction", "Time to event",
    "Anomalies", "Scenarios", "Explainability", "LLM copilot", "Loan explorer",
    "Submission", "Model card & log",
]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_raising(page):
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    app.sidebar.radio[0].set_value(page).run()

    assert not app.exception, f"{page}: {[e.value for e in app.exception]}"


def test_navigation_follows_the_demo_flow():
    """
    Section 14 specifies the order a demo should follow. The navigation *is*
    that order, so the app can be walked top to bottom on camera without
    deciding what comes next -- and a reordering should be a deliberate act.
    """
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()

    # The widget reports its *formatted* labels ("Data intelligence   ·  Task 1"),
    # not the raw values, because the sidebar uses a format_func.
    options = [str(o).split("·")[0].strip() for o in app.sidebar.radio[0].options]

    assert options[0] == "Overview"
    assert options.index("Data intelligence") < options.index("Prediction")
    assert options.index("Prediction") < options.index("Time to event")
    assert options.index("Anomalies") < options.index("Scenarios")
    assert options.index("Explainability") < options.index("LLM copilot")
    assert options[-1] == "Model card & log"


def test_a_missing_phase_is_reported_not_crashed(monkeypatch):
    """
    A dashboard that renders an empty panel with no explanation sends the user
    to read the code. One that names the command to run does not.
    """
    monkeypatch.setattr(data, "read_csv", lambda relative: pd.DataFrame())

    from src.dashboard import pages

    monkeypatch.setattr(pages.data, "read_csv", lambda relative: pd.DataFrame())
    monkeypatch.setattr(pages.data, "read_jsonl", lambda relative, limit=None: [])

    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception


def test_deliverables_resolve_against_the_filesystem():
    frame = data.deliverables()
    assert set(frame.columns) >= {"Deliverable", "Task", "Present", "Path"}
    assert len(frame) == len(data.DELIVERABLES)
    assert frame["Present"].dtype == bool


def test_missing_output_returns_an_empty_frame_not_an_error():
    """A phase that has not run must not take the page down with it."""
    assert data.read_csv("reports/does_not_exist.csv").empty
    assert data.read_text("reports/does_not_exist.md") == ""
    assert data.read_jsonl("reports/does_not_exist.jsonl") == []


def test_the_dashboard_shares_the_figures_palette():
    """
    The app embeds the matplotlib figures directly. A page in a different
    colour language than the charts inside it reads as two products stapled
    together.
    """
    from src import viz

    assert theme.BLUE == viz.CATEGORICAL[0]
    assert theme.ORANGE == viz.CATEGORICAL[1]
    assert theme.SURFACE == viz.SURFACE
    assert theme.STATUS == viz.STATUS
