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


def test_report_viewer_inlines_every_figure(tmp_path, monkeypatch):
    """
    The in-app report viewer must not depend on relative asset paths.

    Streamlit serves the app from its own origin and never exposes the working
    directory over HTTP, so a report rendered into an iframe with a relative
    `<img src="charts/x.png">` shows a broken image -- which is exactly what the
    old "the report is at reports/x.html" caption led to. Every referenced
    figure has to survive as a data URI.
    """
    reports = tmp_path / "reports"
    (reports / "charts").mkdir(parents=True)
    # A one-pixel PNG is enough; what is under test is the rewriting, not the image.
    (reports / "charts" / "fig.png").write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )
    )
    (reports / "r.html").write_text(
        '<html><body><img alt="f" src="charts/fig.png"></body></html>', encoding="utf-8"
    )
    monkeypatch.setattr(data, "ROOT", tmp_path)
    data._CACHE.clear()

    html = data.report_html("reports/r.html")

    assert "data:image/png;base64," in html
    assert 'src="charts/fig.png"' not in html


def test_report_viewer_is_silent_when_the_phase_has_not_run():
    assert data.report_html("reports/does_not_exist.html") == ""
    assert data.read_bytes("reports/does_not_exist.html") == b""
    assert data.publish_report("reports/does_not_exist.html") == ""


def test_a_published_report_is_self_contained_at_its_own_url(tmp_path, monkeypatch):
    """
    "Open as a page" has to yield a document that stands on its own.

    It is served out of Streamlit's static directory with no asset folder
    beside it, so a report still carrying relative image paths would open as a
    wall of broken figures -- and would stay broken if the tab were bookmarked
    or forwarded.
    """
    reports = tmp_path / "reports"
    (reports / "charts").mkdir(parents=True)
    (reports / "charts" / "fig.png").write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )
    )
    (reports / "r.html").write_text(
        '<html><body><img alt="f" src="charts/fig.png"></body></html>', encoding="utf-8"
    )
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "STATIC_DIR", tmp_path / "static")
    data._CACHE.clear()

    url = data.publish_report("reports/r.html")

    assert url == "app/static/reports/r.html"
    published = (tmp_path / "static" / "reports" / "r.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in published
    assert 'src="charts/fig.png"' not in published


def test_generated_notes_render_as_content_not_as_a_file():
    """
    Stored notes carry the disclaimer top *and* bottom so the label survives a
    copy-paste into a case file. On screen that prints one long sentence twice
    around three of content, so the display strips the wrapper and states the
    label once -- and the markdown has to survive, or the reader sees asterisks
    where the emphasis should be.
    """
    from src.copilot.guardrails import GuardrailVerdict, wrap
    from src.dashboard import pages

    note = wrap("Loan **X** is Current.\n\n- step one\n- step two",
                GuardrailVerdict(passed=True))
    body, banner = pages._unwrap_note(note)

    assert "RECOMMENDATION, NOT A DECISION" not in body
    assert banner == ""
    assert body.startswith("Loan **X**")

    rendered = pages._markdown_body(body)
    assert "<strong>X</strong>" in rendered
    assert "<li>" in rendered


def test_a_withheld_note_surfaces_its_guardrail_banner():
    """The reason a note was withheld must not be stripped along with the wrapper."""
    from src.copilot.guardrails import GuardrailVerdict, wrap
    from src.dashboard import pages

    note = wrap("Some output.", GuardrailVerdict(passed=False, ungrounded_numbers=["42"]))
    body, banner = pages._unwrap_note(note)

    assert "GUARDRAIL FAILED" in banner
    assert "42" in banner
    assert body == "Some output."


def test_the_offline_stubs_own_bracketed_opener_is_not_stripped():
    """
    The stub announces itself as `[OFFLINE STUB -- not a model response] Loan...`.
    That is content, not a wrapper, and removing it would present a stub as a
    real model generation -- the one thing offline mode exists to prevent.
    """
    from src.copilot.guardrails import GuardrailVerdict, wrap
    from src.dashboard import pages

    note = wrap("[OFFLINE STUB -- not a model response] Loan Z.", GuardrailVerdict(passed=True))
    body, _ = pages._unwrap_note(note)

    assert body.startswith("[OFFLINE STUB")


def test_the_deliverables_checklist_mirrors_section_11():
    """
    Every section 11 deliverable the *pipeline produces* is tracked, so the
    landing page reports what is actually on disk. The demo video is excluded
    deliberately: it is recorded by hand and lives outside the repository, so a
    filesystem check could only ever report it missing.
    """
    frame = data.deliverables()
    required = frame[frame.Task == "Section 11"]

    assert len(required) == 9
    for name in ("submission.csv", "Model card", "AI Development Log"):
        assert name in set(required.Deliverable), f"{name} is not tracked"
    assert "Five-minute demo video" not in set(required.Deliverable)


def test_copilot_status_never_leaks_key_material():
    """
    The explorer shows whether the copilot is live. It must report *state*, not
    the credential -- an app that renders a key into the page has published it.
    """
    live, model = data.copilot_status()

    assert isinstance(live, bool)
    assert isinstance(model, str)
    for prefix in ("gsk_", "xai-", "sk-"):
        assert not model.startswith(prefix)


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
