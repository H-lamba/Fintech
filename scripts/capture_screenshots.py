"""
Capture the dashboard screenshots embedded in the README.

    python scripts/capture_screenshots.py            # assumes the app is already up
    python scripts/capture_screenshots.py --launch   # start Streamlit, shoot, stop

Screenshots in a README rot faster than any other documentation: the UI moves,
nobody re-shoots, and the page ends up advertising a product that no longer
exists. Keeping the capture as a script means regenerating them is one command
rather than twelve manual window-drags, so it actually gets done.

Playwright is an optional extra -- it is not in `requirements.txt`, because a
judge reproducing the pipeline should not have to download a browser engine to
do it:

    pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "docs" / "screenshots"

# (filename, sidebar label, extra scroll in px, settle seconds). The scroll
# offsets frame the part of each page worth showing rather than its heading.
SHOTS = [
    ("overview.png", "Overview", 0, 2.5),
    ("data-intelligence.png", "Data intelligence", 500, 3.0),
    ("prediction.png", "Prediction", 250, 3.0),
    ("survival.png", "Time to event", 300, 3.0),
    ("anomalies.png", "Anomalies", 350, 3.0),
    ("scenarios.png", "Scenarios", 300, 3.5),
    ("explainability.png", "Explainability", 400, 3.5),
    ("copilot.png", "LLM copilot", 400, 3.0),
    ("loan-explorer.png", "Loan explorer", 700, 3.5),
    ("submission.png", "Submission", 200, 3.0),
    ("model-card.png", "Model card & log", 300, 3.0),
]

VIEWPORT = {"width": 1600, "height": 1000}


def capture(url: str) -> None:
    from playwright.sync_api import sync_playwright

    OUTDIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # color_scheme is pinned light to match `.streamlit/config.toml`. Left to
        # the machine's own preference the shots would come out in whichever
        # theme the capturing laptop happened to be in.
        #
        # device_scale_factor stays at 1: a 2x capture of this UI is four times
        # the bytes for detail no one reads at README width, and eleven retina
        # PNGs is most of a ten-megabyte clone.
        page = browser.new_page(viewport=VIEWPORT, color_scheme="light",
                                device_scale_factor=1)
        page.goto(url, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(4000)

        for name, label, scroll, settle in SHOTS:
            try:
                page.get_by_text(label, exact=False).first.click(timeout=15_000)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {name}: could not reach '{label}' ({exc})")
                continue
            page.wait_for_timeout(int(settle * 1000))
            if scroll:
                page.mouse.wheel(0, scroll)
                page.wait_for_timeout(1200)
            page.screenshot(path=str(OUTDIR / name))
            print(f"  [ok]   {name}")

        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--launch", action="store_true",
                        help="Start Streamlit for the capture, then stop it.")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    url = args.url if not args.launch else f"http://localhost:{args.port}"
    server = None
    if args.launch:
        print(f"Starting Streamlit on :{args.port} ...")
        server = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
             "--server.port", str(args.port), "--server.headless", "true"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(12)

    try:
        print(f"Capturing {len(SHOTS)} screenshots from {url} ...")
        capture(url)
        print(f"\nWrote to {OUTDIR.relative_to(ROOT)}")
    finally:
        if server:
            server.terminate()
            print("Stopped Streamlit.")


if __name__ == "__main__":
    main()
