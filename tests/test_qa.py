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
import re
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
                f" app.run(port={PORT}, debug=False, use_reloader=False, threaded=True)"
            ),
        ],
        cwd=str(DEMO_DIR),
        env=env,
        # DEVNULL prevents the pipe buffer from filling up and blocking Flask's
        # access-log writes, which would stall request processing under load.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
        urllib.request.Request(f"{BASE_URL}/reset", method="POST"), timeout=8
    )


# ── Helpers ───────────────────────────────────────────────────

def dismiss_screensaver(page: Page) -> None:
    """Click the JARVIS ring and wait for the screensaver to leave the DOM."""
    page.locator(".ss-ring").click()
    # dismissScreensaver() adds .dismissed (opacity: 0) then calls screensaver.remove()
    # after a 600 ms CSS transition.  Playwright's to_be_hidden() treats opacity-0 as
    # hidden, so this assertion passes as soon as the fade begins.
    expect(page.locator("#screensaver")).to_be_hidden(timeout=3000)
    # Wait for the element to be fully removed from the DOM.  screensaver.remove() fires
    # inside the same 600 ms setTimeout that also dispatches POST /greeting.  Waiting
    # for detach ensures that greeting request has been dispatched (and processed by the
    # server) before the test sends its own requests.
    page.locator("#screensaver").wait_for(state="detached", timeout=3000)


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


def test_device_chip_turns_on(flask_server, page: Page):
    """#dev-light-bedroom gains class 'on' after 'turn on the bedroom lights'."""
    page.goto(BASE_URL)
    dismiss_screensaver(page)

    chip = page.locator("#dev-light-bedroom")
    # Chip starts off — class attribute must not contain the word 'on'
    expect(chip).not_to_have_attribute("class", re.compile(r"\bon\b"))
    page.screenshot(path=str(SCREENSHOTS_DIR / "07_chip_before.png"))

    initial_count = page.locator(".message.jarvis").count()
    page.fill("#input", TIER2_COMMAND)
    page.keyboard.press("Enter")
    # Wait for Jarvis reply (guarantees the /chat round-trip completed)
    expect(page.locator(".message.jarvis")).to_have_count(
        initial_count + 1, timeout=8000
    )

    # syncDeviceChips() runs after addMessage(), so allow a short retry window
    expect(chip).to_have_attribute("class", re.compile(r"\bon\b"), timeout=3000)
    expect(chip).to_contain_text("on")
    page.screenshot(path=str(SCREENSHOTS_DIR / "08_chip_on.png"))


def test_voice_mode_canvas_renders(flask_server, page: Page):
    """Switching to voice mode shows the canvas with non-zero pixel dimensions."""
    page.goto(BASE_URL)
    dismiss_screensaver(page)

    page.locator("#mode-voice").click()
    expect(page.locator("#voice-mode")).to_have_attribute(
        "class", re.compile(r"\bactive\b"), timeout=3000
    )
    expect(page.locator("#chat-mode")).to_have_attribute(
        "class", re.compile(r"\bhidden\b"), timeout=3000
    )
    expect(page.locator("#voice-canvas")).to_be_visible()

    # resizeVoiceCanvas() sets canvas.width/height from clientWidth * devicePixelRatio
    w = page.evaluate("document.getElementById('voice-canvas').width")
    h = page.evaluate("document.getElementById('voice-canvas').height")
    assert w > 0, f"canvas width unexpectedly {w}"
    assert h > 0, f"canvas height unexpectedly {h}"

    page.screenshot(path=str(SCREENSHOTS_DIR / "09_voice_canvas.png"))


# All six waveform style names, matching the onclick= values in the HTML
_VOICE_STYLES = ["waveform", "bars", "orb", "ring", "pulse", "spiral"]


def test_voice_style_chips_cycle(flask_server, page: Page):
    """Clicking each voice-style chip activates it and deactivates the rest."""
    page.goto(BASE_URL)
    dismiss_screensaver(page)
    page.locator("#mode-voice").click()
    expect(page.locator("#voice-mode")).to_have_attribute(
        "class", re.compile(r"\bactive\b"), timeout=3000
    )

    chips = page.locator(".voice-style-chip")
    for idx, style in enumerate(_VOICE_STYLES):
        chips.nth(idx).click()
        # Clicked chip must be active
        expect(chips.nth(idx)).to_have_attribute(
            "class", re.compile(r"\bactive\b"), timeout=2000
        )
        # All other chips must not be active
        for other in range(len(_VOICE_STYLES)):
            if other != idx:
                expect(chips.nth(other)).not_to_have_attribute(
                    "class", re.compile(r"\bactive\b")
                )
        page.screenshot(
            path=str(SCREENSHOTS_DIR / f"1{idx}_voice_style_{style}.png")
        )


# (hex, rgb-string) pairs matching the six theme-dot onclick= calls in the HTML
_THEMES = [
    ("#4a9eff", "74,158,255"),
    ("#00e5c8", "0,229,200"),
    ("#7c3aed", "124,58,237"),
    ("#f59e0b", "245,158,11"),
    ("#ef4444", "239,68,68"),
    ("#22c55e", "34,197,94"),
]


def test_themes_cycle_css_variables(flask_server, page: Page):
    """Clicking each theme dot updates --accent and --accent-rgb on :root."""
    page.goto(BASE_URL)
    dismiss_screensaver(page)

    dots = page.locator(".theme-dot")
    for idx, (hex_color, rgb) in enumerate(_THEMES):
        # Hover the picker to expose the tray, then click the dot
        page.hover(".theme-picker")
        dots.nth(idx).click()

        accent = page.evaluate(
            "document.documentElement.style.getPropertyValue('--accent').trim()"
        )
        assert accent == hex_color, f"theme {idx}: expected --accent={hex_color!r}, got {accent!r}"

        accent_rgb = page.evaluate(
            "document.documentElement.style.getPropertyValue('--accent-rgb').trim()"
        )
        assert accent_rgb == rgb, (
            f"theme {idx}: expected --accent-rgb={rgb!r}, got {accent_rgb!r}"
        )

        page.screenshot(
            path=str(SCREENSHOTS_DIR / f"{16 + idx:02d}_theme_{hex_color[1:]}.png")
        )


def test_tts_toggle(flask_server, page: Page):
    """TTS button cycles on→off→on, updating text and 'off' class accordingly."""
    page.goto(BASE_URL)
    dismiss_screensaver(page)

    btn = page.locator("#tts-toggle")
    # Initial state: TTS on, no 'off' class
    expect(btn).to_have_text("TTS: on")
    expect(btn).not_to_have_attribute("class", re.compile(r"\boff\b"))
    page.screenshot(path=str(SCREENSHOTS_DIR / "22_tts_initially_on.png"))

    btn.click()
    expect(btn).to_have_text("TTS: off")
    expect(btn).to_have_attribute("class", re.compile(r"\boff\b"), timeout=2000)
    page.screenshot(path=str(SCREENSHOTS_DIR / "23_tts_off.png"))

    btn.click()
    expect(btn).to_have_text("TTS: on")
    expect(btn).not_to_have_attribute("class", re.compile(r"\boff\b"))
    page.screenshot(path=str(SCREENSHOTS_DIR / "24_tts_on_again.png"))
