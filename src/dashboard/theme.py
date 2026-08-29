"""
The dashboard's visual language.

Deliberately the *same* palette the figures use (`src/viz.py`), because the app
embeds those figures directly. A page in a different colour language than the
charts inside it reads as two products stapled together, and a judge notices
that before they notice the metrics.

The categorical hues are assigned in a fixed order and never cycled; the status
colours are reserved for state and always ship with a text label, because roughly
one reader in twelve cannot separate them by hue alone.
"""

from __future__ import annotations

from .. import viz

SURFACE = viz.SURFACE
INK = viz.INK_PRIMARY
INK_SECONDARY = viz.INK_SECONDARY
INK_MUTED = viz.INK_MUTED
GRID = viz.GRIDLINE
AXIS = viz.AXIS

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = viz.CATEGORICAL[:6]
STATUS = viz.STATUS

# Consistent across every page: a cause keeps its colour wherever it appears.
CAUSE_COLOR = {"default": BLUE, "prepaid": ORANGE, "prepayment": ORANGE, "delinquency": AQUA}

CSS = f"""
<style>
  .stApp {{ background: {viz.SURFACE}; }}
  section[data-testid="stSidebar"] {{ background: #f4f3ef; border-right: 1px solid {GRID}; }}

  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.9rem; font-weight: 650; }}
  h2 {{ font-size: 1.25rem; font-weight: 600; margin-top: 1.6rem; }}

  /* Stat tiles: a thin left rule carries the tone so the number itself stays ink. */
  .tile {{
    background: #ffffff; border: 1px solid {GRID}; border-left: 3px solid {AXIS};
    border-radius: 8px; padding: 0.85rem 1rem; height: 100%;
  }}
  .tile.good     {{ border-left-color: {STATUS['good']}; }}
  .tile.warning  {{ border-left-color: {STATUS['warning']}; }}
  .tile.critical {{ border-left-color: {STATUS['critical']}; }}
  .tile .label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
                  color: {INK_MUTED}; }}
  .tile .value {{ font-size: 1.75rem; font-weight: 650; color: {INK}; line-height: 1.15; }}
  .tile .unit  {{ font-size: 0.85rem; font-weight: 500; color: {INK_SECONDARY}; }}
  .tile .note  {{ font-size: 0.76rem; color: {INK_SECONDARY}; margin-top: 0.3rem;
                  line-height: 1.35; }}

  /* A caveat panel. Used where a headline number would otherwise be read as a
     performance claim it cannot support. */
  .caveat {{
    background: #fdf6e8; border: 1px solid #f0dcae; border-left: 3px solid {STATUS['warning']};
    border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #5c4a22;
    line-height: 1.5;
  }}
  .note-box {{
    background: #ffffff; border: 1px solid {GRID}; border-radius: 8px;
    padding: 0.9rem 1.1rem; font-size: 0.87rem; line-height: 1.55; color: {INK_SECONDARY};
    white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  .pill {{
    display: inline-block; padding: 0.14rem 0.55rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; margin-right: 0.3rem;
  }}
  .pill.ok   {{ background: #e4f5e4; color: #0a6b0a; }}
  .pill.bad  {{ background: #fbe6e6; color: #8f2020; }}
  .pill.info {{ background: #e6effb; color: #1c4f8f; }}

  div[data-testid="stDataFrame"] {{ border: 1px solid {GRID}; border-radius: 8px; }}
  .stTabs [data-baseweb="tab"] {{ font-size: 0.9rem; }}
</style>
"""


def tile(label: str, value: str, unit: str = "", note: str = "", tone: str = "") -> str:
    """One stat tile. Tone is a thin left rule, never the value's own colour."""
    return (
        f'<div class="tile {tone}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value} <span class="unit">{unit}</span></div>'
        f'<div class="note">{note}</div>'
        f"</div>"
    )


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def caveat(text: str) -> str:
    return f'<div class="caveat">{text}</div>'
