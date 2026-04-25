"""
QA tests for the Jarvis web UI using Playwright.

Install dependencies:
    pip install pytest playwright pytest-playwright
    playwright install chromium

Run:
    pytest tests/test_qa.py -v

Notes:
    - Set ANTHROPIC_API_KEY in your environment (any non-placeholder value suffices
      because these tests use Tier-2 device commands that never reach the Claude API).
    - Screenshots are written to tests/screenshots/ after each state transition.
"""
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

# ── Config ────────────────────────────────────────────────────
PORT = 5002
BASE_URL = f"http://localhost:{PORT}"
DEMO_DIR = Path(__file__).parent.parent / "demo"
SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"

# Tier-2 command: handled by the local intent classifier; never calls Claude.
TIER2_COMMAND = "Turn on the bedroom lights"


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def flask_server():
    """Start the Flask server on PORT and wait until /health responds."""
    SCREENSHOTS_DIR.mkdir(exist_ok=True)

    env = os.environ.copy()
    if not env.get("ANTHROPIC_API_KEY"):
        # Any non-placeholder string passes the startup guard in jarvis_web.py.
        # Tier-2 commands never reach the Claude API, so no real key is required.
        env["ANTHROPIC_API_KEY"] = "sk-ant-placeholder-for-qa-tests"

    proc = subprocess.Popen(
        [
            sys.executable, "-c",
            (
                "import sys; sys.path.insert(0, '.');"
                " from jarvis_web import app;"
                f" app.run(port={PORT}, debug=False, use_reloader=False)"
            ),
        ],
        cwd=str(DEMO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1):
                break
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    else:
        proc.kill()
        raise RuntimeError("Flask server failed to start within 20 s")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(autouse=True)
def reset_between_tests(flask_server):
    """Reset conversation history and device states before each test."""
    urllib.request.urlopen(
        urllib.request.Request(f"{BASE_URL}/reset", method="POST"), timeout=3
    )


# ── Helpers ───────────────────────────────────────────────────

def dismiss_screensaver(page: Page) -> None:
    """Click the JARVIS ring and wait for the screensaver to leave the DOM."""
    page.locator(".ss-ring").click()
    # dismissScreensaver() adds .dismissed (opacity: 0) then calls screensaver.remove()
    # after a 600 ms CSS transition.  Playwright's to_be_hidden() treats opacity-0 as
    # hidden, so this assertion passes as soon as the fade begins.
    expect(page.locator("#screensaver")).to_be_hidden(timeout=3000)


# ── Tests ─────────────────────────────────────────────────────

def test_screensaver_loads(flask_server, page: Page):
    """Screensaver is visible and shows the JARVIS branding on first load."""
    page.goto(BASE_URL)

    screensaver = page.locator("#screensaver")
    expect(screensaver).to_be_visible()
    expect(screensaver).to_contain_text("JARVIS")
    expect(page.locator(".ss-hint")).to_contain_text("click")

    page.screenshot(path=str(SCREENSHOTS_DIR / "01_screensaver.png"))


def test_screensaver_dismisses(flask_server, page: Page):
    """Clicking the screensaver reveals the main chat UI."""
    page.goto(BASE_URL)

    screensaver = page.locator("#screensaver")
    expect(screensaver).to_be_visible()
    page.screenshot(path=str(SCREENSHOTS_DIR / "02_before_dismiss.png"))

    page.locator(".ss-ring").click()
    expect(screensaver).to_be_hidden(timeout=3000)
    page.screenshot(path=str(SCREENSHOTS_DIR / "03_after_dismiss.png"))

    # Core chat chrome must be present after dismissal
    expect(page.locator("#input")).to_be_visible()
    expect(page.locator("#send")).to_be_visible()
    expect(page.locator("#chat")).to_be_visible()


def test_chat_response(flask_server, page: Page):
    """Sending a device command produces a Jarvis reply bubble in the chat area."""
    page.goto(BASE_URL)
    dismiss_screensaver(page)
    page.screenshot(path=str(SCREENSHOTS_DIR / "04_chat_ready.png"))

    # Count pre-existing Jarvis bubbles (the hardcoded greeting)
    initial_count = page.locator(".message.jarvis").count()

    page.fill("#input", TIER2_COMMAND)
    page.screenshot(path=str(SCREENSHOTS_DIR / "05_message_typed.png"))

    page.keyboard.press("Enter")

    # A new Jarvis bubble should appear after the local classifier responds
    expect(page.locator(".message.jarvis")).to_have_count(
        initial_count + 1, timeout=8000
    )

    latest = page.locator(".message.jarvis .text").last
    expect(latest).to_be_visible()
    expect(latest).not_to_be_empty()
    page.screenshot(path=str(SCREENSHOTS_DIR / "06_response_received.png"))
