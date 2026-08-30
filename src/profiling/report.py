"""
Assembles the Data Intelligence Report -- one of the graded deliverables.

Writes both a markdown file (readable in the repo, good for the README link)
and a standalone HTML file (good for the demo video and for judges who'd
rather scroll than clone).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def _df_to_md(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_No findings._\n"
    shown = df.head(max_rows)
    try:
        body = shown.to_markdown(index=False)
    except Exception:
        body = "```\n" + shown.to_string(index=False) + "\n```"
    note = ""
    if len(df) > max_rows:
        note = f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return body + note + "\n"


class ReportBuilder:
    """Accumulate sections, then render to markdown and HTML."""

    def __init__(self, title: str = "Data Intelligence Report") -> None:
        self.title = title
        self.sections: list[tuple[int, str, str]] = []  # (level, heading, body)
        self.generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_text(self, heading: str, text: str, level: int = 2) -> "ReportBuilder":
        self.sections.append((level, heading, text.strip() + "\n"))
        return self

    def add_table(
        self, heading: str, df: pd.DataFrame, note: str = "", level: int = 2, max_rows: int = 40
    ) -> "ReportBuilder":
        body = (note.strip() + "\n\n" if note else "") + _df_to_md(df, max_rows)
        self.sections.append((level, heading, body))
        return self

    def add_kv(self, heading: str, mapping: dict, level: int = 2) -> "ReportBuilder":
        if not mapping:
            self.sections.append((level, heading, "_No data._\n"))
            return self
        lines = ["| Metric | Value |", "| :--- | ---: |"]
        for k, v in mapping.items():
            lines.append(f"| {k} | {v} |")
        self.sections.append((level, heading, "\n".join(lines) + "\n"))
        return self

    def to_markdown(self) -> str:
        out = [f"# {self.title}", "", f"_Generated {self.generated_at}_", ""]
        for level, heading, body in self.sections:
            out.append(f"{'#' * level} {heading}")
            out.append("")
            out.append(body)
            out.append("")
        return "\n".join(out)

    def to_html(self) -> str:
        md = self.to_markdown()
        try:
            import markdown as md_lib

            content = md_lib.markdown(md, extensions=["tables", "fenced_code"])
        except Exception:
            # No markdown lib installed -- fall back to preformatted text so the
            # pipeline never fails on a reporting dependency.
            content = "<pre>" + md.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{self.title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 2rem 1.25rem; line-height: 1.6; }}
  h1 {{ border-bottom: 2px solid currentColor; padding-bottom: .4rem; }}
  h2 {{ margin-top: 2.5rem; border-bottom: 1px solid rgba(128,128,128,.35); padding-bottom: .25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem;
           display: block; overflow-x: auto; }}
  th, td {{ border: 1px solid rgba(128,128,128,.35); padding: .4rem .6rem; text-align: left; }}
  th {{ background: rgba(128,128,128,.12); }}
  tr:nth-child(even) td {{ background: rgba(128,128,128,.06); }}
  /* Figures are rendered on a light chart surface, so they get that surface
     behind them regardless of the reader's theme -- otherwise a dark-mode page
     puts a light chart in a dark frame with no border. */
  img {{ max-width: 100%; height: auto; display: block; margin: 1rem auto;
         background: #fcfcfb; padding: .5rem; border-radius: 6px;
         border: 1px solid rgba(128,128,128,.25); }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem; }}
  pre {{ overflow-x: auto; padding: .75rem; background: rgba(128,128,128,.1); border-radius: 6px; }}
</style>
</head>
<body>
{content}
</body>
</html>"""

    def save(self, md_path: Path | str, html_path: Path | str | None = None) -> None:
        # encoding is explicit because it is not a default we can inherit: on
        # Windows `write_text` falls back to cp1252, which cannot encode the
        # typographic characters these reports legitimately carry (the copilot's
        # own control-failure table documents a non-breaking hyphen). The HTML
        # already declares utf-8 in its meta tag, so writing it as anything else
        # produces a file that lies about its own encoding.
        md_path = Path(md_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        if html_path:
            html_path = Path(html_path)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(self.to_html(), encoding="utf-8")
