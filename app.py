"""
Interactive dashboard for the Loan Performance Intelligence Engine.

    streamlit run app.py

Reads only what the pipeline has already written to `reports/` and
`submission/`. Nothing is recomputed here: a dashboard that recalculates its own
numbers can disagree with the report beside it, and the version a viewer trusts
is whichever they saw last.

Run the pipeline first:

    python main.py

The navigation follows section 14 of the problem statement -- the demo flow --
so the app can be walked top to bottom on camera without deciding what comes
next: dataset and targets, profiling, features and the time-aware split,
baseline then improved model, survival output, anomaly examples, scenario
output, a local explanation, an LLM note and a rejected one, the submission,
and the development log.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _bridge_secrets_to_env() -> None:
    """
    Make Streamlit Cloud's secrets visible to the copilot, which reads `os.environ`.

    Locally the key comes from `.env`, which is gitignored and therefore absent
    on a deployment. Streamlit's own store is `st.secrets`, and while top-level
    secrets are also exported as environment variables, secrets nested under a
    section are not -- so a key pasted as `[llm]\\napi_key = ...` would leave the
    copilot silently serving offline stubs on a hosted app with a perfectly
    valid key configured.

    Copying them across here keeps the pipeline modules free of any Streamlit
    import: `src/copilot/llm_client.py` reads the environment and does not care
    who populated it. Existing environment variables win, so a local `.env`
    still takes precedence.
    """
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 -- no secrets file locally is the normal case
        return

    for key in ("LLM_API_KEY", "XAI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY",
                "LLM_BASE_URL", "LLM_MODEL"):
        try:
            value = secrets.get(key)
        except Exception:  # noqa: BLE001
            continue
        if value and not os.environ.get(key):
            os.environ[key] = str(value)


# Must run before the copilot client is imported: it resolves the key at import time.
_bridge_secrets_to_env()

from src.dashboard import data, pages, theme  # noqa: E402

st.set_page_config(
    page_title=f"{theme.PRODUCT_NAME} · {theme.PRODUCT_TAGLINE}",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.CSS, unsafe_allow_html=True)

# Ordered to match the demo flow, with the task each page answers. The task tag
# is the one piece of scaffolding worth keeping in the navigation: it is how a
# judge maps a page to the criterion it is scored under.
PAGES = {
    "Overview": (pages.overview, ""),
    "Data intelligence": (pages.data_intelligence, "Task 1"),
    "Features & split": (pages.features, "Phase 2"),
    "Prediction": (pages.prediction, "Task 2"),
    "Time to event": (pages.survival, "Task 3"),
    "Anomalies": (pages.anomalies, "Task 4"),
    "Scenarios": (pages.scenarios, "Task 5"),
    "Explainability": (pages.explainability, "Task 6"),
    "LLM copilot": (pages.copilot, "Task 7"),
    "Loan explorer": (pages.explorer, ""),
    "Submission": (pages.submission_page, "Phase 9"),
    "Model card & log": (pages.documents, "Task 8"),
}


def main() -> None:
    with st.sidebar:
        st.markdown(theme.brand(), unsafe_allow_html=True)

        choice = st.radio(
            "Section",
            list(PAGES),
            format_func=lambda name: f"{name}   ·  {PAGES[name][1]}" if PAGES[name][1] else name,
            label_visibility="collapsed",
        )

        st.markdown("---")
        state = data.pipeline_state()
        missing = [name for name, ok in state.items() if not ok]
        if missing:
            st.warning(
                "Not yet run: " + ", ".join(missing) + "\n\nRun `python main.py` to populate."
            )
        else:
            st.success(f"All {len(state)} phases have output on disk.")

        st.caption(
            "Every number is read from a file the pipeline wrote. Nothing is recomputed in "
            "this app, so it cannot disagree with the reports beside it."
        )

    render, _ = PAGES[choice]
    render()


if __name__ == "__main__":
    main()
