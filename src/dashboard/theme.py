"""
The dashboard's visual language.

Deliberately the *same* palette the figures use (`src/viz.py`), because the app
embeds those figures directly. A page in a different colour language than the
charts inside it reads as two products stapled together, and a judge notices
that before they notice the metrics.

The categorical hues are assigned in a fixed order and never cycled; the status
colours are reserved for state and always ship with a text label, because roughly
one reader in twelve cannot separate them by hue alone.

Motion
------
Animation here is **orientation, not decoration**. Every effect answers "what
just changed and where do I look": tiles rise in sequence so the eye lands on
the first one, meters grow from zero so a short bar reads as small rather than
as a rendering artefact, and a freshly generated LLM note gets a one-shot
highlight because it appeared after the page did. Nothing loops forever except
the live-status dot, which is reporting a real state.

Every animation is wrapped in `prefers-reduced-motion`, which switches the
entire sheet to static. Motion that a reader cannot turn off is an
accessibility defect, and the reader who needs it off is the one most likely to
be reviewing this on a projector.
"""

from __future__ import annotations

from .. import viz

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
# One place to change the product's name. "Bellwether" is the sheep that leads
# the flock and, in markets, the indicator that moves first -- which is what
# this system is for: reading a book of loans early enough to act on it. It is
# deliberately not any of the servicer names the generator invents (Atlas,
# Northgate, Cornerstone, Beacon, Meridian), so a name on the page can never be
# mistaken for a name in the data.
PRODUCT_NAME = "Bellwether"
PRODUCT_TAGLINE = "Loan Performance Intelligence"
PRODUCT_CONTEXT = "Intain Campus FinTech Challenge · AI Track"

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
  /* ---------------------------------------------------------------- motion */
  @keyframes riseIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
  }}
  @keyframes growWidth {{
    from {{ width: 0%; }}
  }}
  @keyframes sweep {{
    0%   {{ background-position: -220% 0; }}
    100% {{ background-position: 220% 0; }}
  }}
  @keyframes livePulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%      {{ opacity: .45; transform: scale(.82); }}
  }}
  @keyframes flashOnce {{
    0%   {{ background: #eaf2fd; border-color: {BLUE}; }}
    100% {{ background: #ffffff; border-color: {GRID}; }}
  }}

  .stApp {{ background: {SURFACE}; color: {INK}; }}

  /* Text colour is set alongside every background, never left to Streamlit's
     base theme. Overriding only the background is what put near-white sidebar
     text on a near-white sidebar for anyone whose OS was set to dark mode --
     a 1.03:1 contrast ratio that was invisible on a light-mode dev machine.
     `.streamlit/config.toml` pins the base theme; this is the second lock. */
  section[data-testid="stSidebar"] {{
    background: #f4f3ef; border-right: 1px solid {GRID};
  }}
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] * {{ color: {INK}; }}
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stCaption,
  section[data-testid="stSidebar"] small {{ color: {INK_SECONDARY}; }}

  h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
  h1 {{ font-size: 1.9rem; font-weight: 650; }}
  h2 {{ font-size: 1.25rem; font-weight: 600; margin-top: 1.6rem; }}

  /* Streamlit re-renders the whole block on every interaction, so a blanket
     entry animation would re-run on every click. It is scoped to the elements
     where "this is new" is the message: tiles, the hero, generated output. */
  .hero {{
    background: linear-gradient(135deg, #ffffff 0%, #f6f5f1 55%, #eef2f8 100%);
    border: 1px solid {GRID}; border-radius: 14px;
    padding: 1.35rem 1.6rem; margin-bottom: 1.1rem;
    animation: riseIn .45s cubic-bezier(.2,.7,.3,1) both;
    position: relative; overflow: hidden;
  }}
  .hero::after {{
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(100deg, transparent 38%, rgba(42,120,214,.07) 50%, transparent 62%);
    background-size: 220% 100%;
    animation: sweep 5.5s ease-in-out infinite;
    pointer-events: none;
  }}
  .hero .eyebrow {{
    font-size: .7rem; text-transform: uppercase; letter-spacing: .1em;
    color: {INK_MUTED}; font-weight: 600;
  }}
  .hero .headline {{
    font-size: 1.55rem; font-weight: 650; color: {INK};
    line-height: 1.2; margin: .3rem 0 .35rem;
  }}
  .hero .sub {{ font-size: .9rem; color: {INK_SECONDARY}; line-height: 1.55; max-width: 78ch; }}

  /* Stat tiles: a thin left rule carries the tone so the number itself stays ink. */
  .tile {{
    background: #ffffff; border: 1px solid {GRID}; border-left: 3px solid {AXIS};
    border-radius: 10px; padding: 0.85rem 1rem; height: 100%;
    animation: riseIn .4s cubic-bezier(.2,.7,.3,1) both;
    transition: transform .16s ease, box-shadow .16s ease;
  }}
  .tile:hover {{ transform: translateY(-2px); box-shadow: 0 6px 18px rgba(11,11,11,.07); }}
  /* Staggered so the row resolves left to right instead of snapping in as a
     block -- it is what makes the first tile read as the headline number. */
  .tile.d0 {{ animation-delay: 0ms; }}
  .tile.d1 {{ animation-delay: 55ms; }}
  .tile.d2 {{ animation-delay: 110ms; }}
  .tile.d3 {{ animation-delay: 165ms; }}
  .tile.d4 {{ animation-delay: 220ms; }}
  .tile.d5 {{ animation-delay: 275ms; }}
  .tile.good     {{ border-left-color: {STATUS['good']}; }}
  .tile.warning  {{ border-left-color: {STATUS['warning']}; }}
  .tile.critical {{ border-left-color: {STATUS['critical']}; }}
  .tile .label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
                  color: {INK_MUTED}; }}
  .tile .value {{ font-size: 1.75rem; font-weight: 650; color: {INK}; line-height: 1.15; }}
  .tile .unit  {{ font-size: 0.85rem; font-weight: 500; color: {INK_SECONDARY}; }}
  .tile .note  {{ font-size: 0.76rem; color: {INK_SECONDARY}; margin-top: 0.3rem;
                  line-height: 1.35; }}

  /* A meter. Grown from zero on render so length is read as magnitude. */
  .meter {{ margin: .45rem 0 .7rem; }}
  .meter .meter-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: .78rem; color: {INK_SECONDARY}; margin-bottom: .25rem;
  }}
  .meter .meter-head b {{ color: {INK}; font-weight: 620; font-size: .9rem; }}
  .meter .track {{
    background: #efeee9; border-radius: 999px; height: 8px; overflow: hidden;
  }}
  .meter .fill {{
    height: 100%; border-radius: 999px;
    animation: growWidth .7s cubic-bezier(.2,.7,.3,1) both;
  }}

  /* A caveat panel. Used where a headline number would otherwise be read as a
     performance claim it cannot support. */
  .caveat {{
    background: #fdf6e8; border: 1px solid #f0dcae; border-left: 3px solid {STATUS['warning']};
    border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; color: #5c4a22;
    line-height: 1.5; animation: fadeIn .5s ease both;
  }}
  .note-box {{
    background: #ffffff; border: 1px solid {GRID}; border-radius: 8px;
    padding: 0.9rem 1.1rem; font-size: 0.87rem; line-height: 1.55; color: {INK_SECONDARY};
    white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  /* Applied only to output that was generated after the page loaded, so the
     highlight means "this is the thing you just asked for". */
  .note-box.fresh {{ animation: flashOnce 1.6s ease-out both; }}

  /* Generated LLM content. Set in the reading face, not the monospace one: the
     model answers in prose and bullets, and a checklist rendered as
     preformatted text is a checklist nobody works through. */
  .ai-body {{
    font-size: .93rem; line-height: 1.68; color: {INK};
    margin: .35rem 0 .2rem; animation: fadeIn .45s ease both;
  }}
  .ai-body p {{ margin: 0 0 .7rem; }}
  .ai-body p:last-child {{ margin-bottom: 0; }}
  .ai-body ul, .ai-body ol {{ margin: .1rem 0 .7rem; padding-left: 0; list-style: none; }}
  .ai-body li {{
    position: relative; padding: .3rem 0 .3rem 1.5rem; margin: 0;
    border-bottom: 1px solid {GRID};
  }}
  .ai-body li:last-child {{ border-bottom: 0; }}
  /* A verification step is something a reviewer works through, so each one gets
     a checkbox glyph rather than a bullet. */
  .ai-body li::before {{
    content: "\\2610"; position: absolute; left: 0; top: .28rem;
    color: {BLUE}; font-size: 1rem; line-height: 1.5;
  }}
  .ai-body strong {{ color: {INK}; font-weight: 620; }}
  .ai-body code {{
    background: #f1f0ec; border: 1px solid {GRID}; border-radius: 4px;
    padding: .04rem .3rem; font-size: .84em;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  .ai-body table {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
  .ai-body th, .ai-body td {{ border: 1px solid {GRID}; padding: .3rem .5rem; text-align: left; }}

  /* The mandatory label, stated once and in its own register -- present and
     unmissable, but not competing with the content for the reader's attention. */
  .ai-disclaimer {{
    display: flex; gap: .55rem; align-items: flex-start;
    background: #f7f6f2; border: 1px solid {GRID}; border-left: 3px solid {AXIS};
    border-radius: 8px; padding: .55rem .8rem; margin-top: .9rem;
    font-size: .76rem; line-height: 1.5; color: {INK_SECONDARY};
  }}
  .ai-disclaimer .mark {{ font-size: .9rem; line-height: 1.3; }}
  .ai-disclaimer b {{ color: {INK}; font-weight: 620; }}

  .pill {{
    display: inline-block; padding: 0.14rem 0.55rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; margin-right: 0.3rem;
  }}
  /* Sidebar identity */
  /* align-items: flex-start, not center: centred against a two-line block the
     mark floats between the lines and reads as belonging to neither. */
  .brand {{ display: flex; align-items: flex-start; gap: .5rem; margin: .1rem 0 .15rem; }}
  .brand-mark {{
    font-size: 1.15rem; line-height: 1; color: {BLUE}; margin-top: .18rem;
    animation: riseIn .5s cubic-bezier(.2,.7,.3,1) both;
  }}
  .brand-name {{
    font-size: 1.32rem; font-weight: 680; color: {INK};
    letter-spacing: -0.02em; line-height: 1.1;
  }}
  .brand-tagline {{
    font-size: .74rem; color: {INK_SECONDARY}; letter-spacing: .02em; margin-top: .1rem;
  }}
  .brand-context {{
    font-size: .68rem; color: {INK_MUTED}; letter-spacing: .04em;
    text-transform: uppercase; margin: .5rem 0 .9rem;
    padding-bottom: .7rem; border-bottom: 1px solid {GRID};
  }}

  .status-line {{
    display: flex; align-items: baseline; gap: .1rem; flex-wrap: wrap;
    margin: -0.2rem 0 .9rem;
  }}
  .status-line .status-text {{
    font-size: .82rem; color: {INK_SECONDARY}; line-height: 1.55; max-width: 88ch;
  }}

  .pill.ok   {{ background: #e4f5e4; color: #0a6b0a; }}
  .pill.bad  {{ background: #fbe6e6; color: #8f2020; }}
  .pill.info {{ background: #e6effb; color: #1c4f8f; }}
  .pill.warn {{ background: #fdf1dc; color: #7a5410; }}

  /* The live dot loops because it reports a continuing state, not an event. */
  .livedot {{
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: {STATUS['good']}; margin-right: .4rem; vertical-align: middle;
    animation: livePulse 1.7s ease-in-out infinite;
  }}
  .offdot {{
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: {INK_MUTED}; margin-right: .4rem; vertical-align: middle;
  }}

  div[data-testid="stDataFrame"] {{ border: 1px solid {GRID}; border-radius: 8px; }}
  .stTabs [data-baseweb="tab"] {{ font-size: 0.9rem; }}
  div[data-testid="stExpander"] {{ border-radius: 10px; }}

  .stButton > button {{
    border-radius: 8px; font-weight: 560;
    transition: transform .14s ease, box-shadow .14s ease;
  }}
  .stButton > button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(11,11,11,.09); }}

  @media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
      animation: none !important;
      transition: none !important;
    }}
  }}
</style>
"""


def tile(label: str, value: str, unit: str = "", note: str = "", tone: str = "",
         delay: int = 0) -> str:
    """One stat tile. Tone is a thin left rule, never the value's own colour."""
    return (
        f'<div class="tile {tone} d{min(delay, 5)}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value} <span class="unit">{unit}</span></div>'
        f'<div class="note">{note}</div>'
        f"</div>"
    )


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def caveat(text: str) -> str:
    return f'<div class="caveat">{text}</div>'


def hero(eyebrow: str, headline: str, sub: str = "") -> str:
    """The page's opening block: what this page is, in one screen."""
    return (
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<div class="headline">{headline}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def meter(label: str, value: float, display: str = "", color: str = BLUE,
          scale: float = 1.0) -> str:
    """
    A labelled bar, grown from zero on render.

    ``value`` is the magnitude and ``scale`` the value that means "full",
    so a set of meters sharing a scale stays comparable across rows.
    """
    try:
        pct = max(0.0, min(100.0, (float(value) / float(scale)) * 100.0)) if scale else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        pct = 0.0
    return (
        f'<div class="meter"><div class="meter-head"><span>{label}</span>'
        f"<b>{display or value}</b></div>"
        f'<div class="track"><div class="fill" style="width:{pct:.1f}%;background:{color};">'
        f"</div></div></div>"
    )


def brand() -> str:
    """The sidebar identity block: a name, what it is, and where it came from."""
    return (
        '<div class="brand">'
        f'<div class="brand-mark">&#9672;</div>'
        f'<div><div class="brand-name">{PRODUCT_NAME}</div>'
        f'<div class="brand-tagline">{PRODUCT_TAGLINE}</div></div>'
        "</div>"
        f'<div class="brand-context">{PRODUCT_CONTEXT}</div>'
    )


def disclaimer_note(title: str, body: str) -> str:
    """The 'recommendation, not a decision' label, stated once, below the content."""
    return (
        f'<div class="ai-disclaimer"><span class="mark">&#9878;</span>'
        f"<span><b>{title}</b> {body}</span></div>"
    )


def status_line(badge: str, text: str) -> str:
    """A status pill reading as part of the sentence it qualifies, not floated away."""
    return (
        f'<div class="status-line">{badge}<span class="status-text">{text}</span></div>'
    )


def live_badge(is_live: bool) -> str:
    """
    Whether the copilot is wired to a real provider, said plainly.

    Deliberately reports the *state* and not the model identifier. Which
    provider and model served a response is governance evidence and belongs in
    the audit trail and the Task 7 page, where it is recorded per call; on the
    explorer it would be a vendor name printed beside a reviewer's loan and
    nothing a reviewer can act on.
    """
    if is_live:
        return '<span class="pill ok"><span class="livedot"></span>live</span>'
    return '<span class="pill warn"><span class="offdot"></span>offline stub</span>'
