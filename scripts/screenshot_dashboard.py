#!/usr/bin/env python3
"""Take screenshots of the Orbit dashboard using Playwright.

Launches Streamlit headless, waits for it to be ready, then captures:
  - Session Overview page  -> docs/dashboard-overview.png
  - Prescriptions page     -> docs/dashboard-prescriptions.png

Usage:
    python scripts/screenshot_dashboard.py [--data-dir ./test_deployments]
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_streamlit(port: int, timeout: float = 30.0) -> bool:
    """Poll the Streamlit health endpoint until ready."""
    url = f"http://localhost:{port}/_stcore/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Screenshot the Orbit dashboard.")
    parser.add_argument("--data-dir", default="./test_deployments")
    args = parser.parse_args()

    port = find_free_port()
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    print(f"Launching Streamlit on port {port}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            "orbit/dashboard/app.py",
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--", f"--data-dir={args.data_dir}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        print("Waiting for Streamlit to be ready...")
        if not wait_for_streamlit(port):
            print("ERROR: Streamlit failed to start within 30 seconds.")
            sys.exit(1)

        # Give Streamlit a moment to fully render its first page
        time.sleep(3)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 800})

            # Screenshot 1: Session Overview (default page)
            base_url = f"http://localhost:{port}"
            page.goto(base_url, wait_until="networkidle")
            # Wait for Streamlit content to render
            page.wait_for_timeout(5000)
            page.screenshot(path=str(docs_dir / "dashboard-overview.png"))
            print("Saved docs/dashboard-overview.png")

            # Screenshot 2: Prescriptions page
            # st.Page navigation — click the sidebar nav link
            prescriptions_link = page.locator('a:has-text("Prescriptions")').first
            if prescriptions_link.is_visible():
                prescriptions_link.click()
            else:
                # Fallback: try URL-based navigation
                page.goto(f"{base_url}/Prescriptions", wait_until="networkidle")

            page.wait_for_timeout(5000)
            page.screenshot(path=str(docs_dir / "dashboard-prescriptions.png"))
            print("Saved docs/dashboard-prescriptions.png")

            browser.close()

    finally:
        proc.terminate()
        proc.wait(timeout=5)
        print("Streamlit shut down.")


if __name__ == "__main__":
    main()
