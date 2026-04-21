"""
Jarvis Phase 1 — Demo v2 (Streaming + Browser TTS + Voice Mode + Intent Classifier)
=====================================================================================
Claude's response streams to the browser in real-time.
Browser speaks each sentence INSTANTLY via Web Speech API.
No server-side TTS, no file generation, no latency.

Intent classifier routes simple commands locally (~0.01ms) instead of
calling Claude (~1-2s). Only complex/conversational messages go to Claude.

Two modes:
  - Chat mode: text bubbles, type or speak
  - Voice mode: waveform visualization, pure voice interaction

Open in Microsoft Edge for the best neural voices.

Setup:
  pip install anthropic flask python-dotenv
  python jarvis_web.py
  Open http://localhost:5000 in Microsoft Edge
"""

import os
import json
import re
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

try:
    from anthropic import Anthropic
    from flask import Flask, request, jsonify, render_template
except ImportError:
    print("\n❌ Missing packages. Run:")
    print("   pip install anthropic flask python-dotenv")
    print("   Then try again.\n")
    exit(1)

from intent_classifier import classify, get_filler_phrase  # noqa: E402

# ─── CONFIG ────────────────────────────────────────────────────
API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
FILLER_ENABLED = True   # dev flag — set False to disable filler phrases globally
TTS_ENABLED = True      # dev flag — set False to disable browser TTS globally

app = Flask(__name__)
client = Anthropic(api_key=API_KEY)
conversation_history = []

# ─── Device State Tracking ─────────────────────────────────────
device_states = {
    "light.bedroom": {"state": "off", "friendly_name": "Bedroom main light", "brightness": 0},
    "light.bedroom_lamp": {"state": "off", "friendly_name": "Bedroom lamp", "brightness": 0},
    "switch.bedroom_fan": {"state": "off", "friendly_name": "Bedroom fan"},
    "cover.bedroom_blinds": {"state": "closed", "friendly_name": "Bedroom blinds"},
}


def update_device_state(action):
    """Update local device state based on a parsed action."""
    entity_id = action.get("entity_id", "")
    service = action.get("service", "")
    data = action.get("data", {})

    if entity_id not in device_states:
        return

    if "turn_on" in service or "open" in service:
        device_states[entity_id]["state"] = "open" if "cover" in entity_id else "on"
        if "brightness" in data:
            device_states[entity_id]["brightness"] = data["brightness"]
        elif "brightness" in device_states[entity_id] and "turn_on" in service:
            device_states[entity_id]["brightness"] = 255
    elif "turn_off" in service or "close" in service:
        device_states[entity_id]["state"] = "closed" if "cover" in entity_id else "off"
        if "brightness" in device_states[entity_id]:
            device_states[entity_id]["brightness"] = 0


def get_device_status_text():
    """Generate a human-readable summary of all device states."""
    lines = []
    for entity_id, info in device_states.items():
        name = info["friendly_name"]
        state = info["state"].upper()
        extra = ""
        if "brightness" in info and info["state"] == "on":
            pct = round(info["brightness"] / 255 * 100)
            extra = f" (brightness: {pct}%)"
        lines.append(f"  - {name} ({entity_id}): {state}{extra}")
    return "\n".join(lines)


SYSTEM_STATIC = """You are Jarvis, a personal AI assistant for a smart home system.
Your personality is helpful, witty, and concise — like the Jarvis from Iron Man but
more casual and friendly. You're talking to Michael in his bedroom.

RULES:
- Keep responses SHORT (1-3 sentences max) since they'll eventually be spoken via TTS.
- Be natural and conversational, not robotic.
- You can be playful — you're Jarvis after all.

DEVICE CONTROL:
When the user asks to control a device, respond with a natural confirmation AND include
an action block on its own line in this exact format:
[ACTION: {"service": "light.turn_on", "entity_id": "light.bedroom", "data": {}}]

Available services:
- light.turn_on / light.turn_off (data: {"brightness": 0-255})
- switch.turn_on / switch.turn_off
- cover.open_cover / cover.close_cover

If the user says something conversational (not device-related), just respond naturally.
If you're unsure which device, ask for clarification."""


def build_system_prompt():
    """
    Returns system prompt as a list of content blocks for prompt caching.
    The static block (personality, rules, action format) is marked cacheable —
    only the dynamic device status block changes per request.
    """
    return [
        {
            "type": "text",
            "text": SYSTEM_STATIC,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"CURRENT DEVICE STATUS:\n{get_device_status_text()}\n\n"
                "IMPORTANT: When asked about device status, use the CURRENT STATUS above "
                "to give accurate answers. If a light is already on and the user asks to "
                "turn it on, let them know it's already on. If they ask 'what's on?' or "
                "'status?', report the current state of all devices."
            ),
        },
    ]


def parse_actions(text):
    actions = []
    for match in re.finditer(r'\[ACTION:\s*({.*?})\s*\]', text):
        try:
            actions.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    return actions


def clean_response(text):
    return re.sub(r'\[ACTION:.*?\]', '', text).strip()


def log_interaction(command, response, actions, tier=3, latency_ms=0):
    """Log every interaction to a JSONL file. Now includes tier and latency."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "response": response,
        "actions": actions,
        "intent": "home_control" if actions else "conversation",
        "tier": tier,
        "latency_ms": round(latency_ms, 2),
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/interactions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", tts_enabled=TTS_ENABLED)


GREETING = "Good evening, Michael. Systems are online. What can I do for you?"


@app.route("/greeting", methods=["POST"])
def greeting():
    """Return greeting text — browser handles TTS."""
    return jsonify({"text": GREETING})


@app.route("/filler", methods=["POST"])
def filler():
    """
    Return a filler phrase if this will be a Tier 3 (Claude) request, else null.
    Called by the frontend simultaneously with /chat so the phrase can be
    spoken immediately while Claude generates.
    """
    if not FILLER_ENABLED:
        return jsonify({"filler": None})

    data = request.json
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"filler": None})

    # Run the classifier (< 5ms) — skip filler for instant Tier 1/2 responses
    result = classify(user_message, device_states)
    if result["tier"] < 3:
        return jsonify({"filler": None})

    return jsonify({"filler": get_filler_phrase(user_message)})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Intent classifier routes simple commands locally (~0.01ms).
    Only complex/conversational messages go to Claude (~1-2s).
    """
    global conversation_history

    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "I didn't catch that.", "actions": []})

    start_time = time.perf_counter()

    # ─── Intent Classifier ─────────────────────
    # Try to handle locally BEFORE calling Claude
    result = classify(user_message, device_states)

    print(
        f"[classifier] layer={result.get('matched_layer','?')} tier={result['tier']}"
        f" intent={result['intent']} | \"{user_message}\""
    )

    if result["tier"] < 3:
        # Handled locally — no Claude needed!
        for action in result["actions"]:
            update_device_state(action)

        latency = (time.perf_counter() - start_time) * 1000
        log_interaction(user_message, result["response"], result["actions"],
                        tier=result["tier"], latency_ms=latency)

        return jsonify({
            "response": result["response"],
            "actions": result["actions"],
            "device_states": device_states,
            "tier": result["tier"],
            "latency_ms": round(latency, 2),
        })

    # ─── Tier 3: Claude API ────────────────────
    # Complex/conversational — needs real intelligence
    conversation_history.append({"role": "user", "content": user_message})
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    try:
        current_prompt = build_system_prompt()

        full_response = ""
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=current_prompt,
            messages=conversation_history,
        ) as stream:
            for text_chunk in stream.text_stream:
                full_response += text_chunk

        actions = parse_actions(full_response)
        spoken_response = clean_response(full_response)

        for action in actions:
            update_device_state(action)

        conversation_history.append({"role": "assistant", "content": full_response})

        latency = (time.perf_counter() - start_time) * 1000
        log_interaction(user_message, spoken_response, actions,
                        tier=3, latency_ms=latency)

        return jsonify({
            "response": spoken_response,
            "actions": actions,
            "device_states": device_states,
            "tier": 3,
            "latency_ms": round(latency, 2),
        })

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}", "actions": []})


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "service": "jarvis_orchestrator",
        "version": "0.4.0",
        "intent_classifier": "active",
    })


@app.route("/devices")
def devices():
    return jsonify(device_states)


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history
    conversation_history = []
    for entity_id in device_states:
        device_states[entity_id]["state"] = "closed" if "cover" in entity_id else "off"
        if "brightness" in device_states[entity_id]:
            device_states[entity_id]["brightness"] = 0
    return jsonify({"status": "reset", "device_states": device_states})


if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n❌ Set your API key first!")
        print("   Option 1: Edit this file and replace YOUR_API_KEY_HERE")
        print("   Option 2: Set environment variable ANTHROPIC_API_KEY\n")
        exit(1)

    print()
    print("=" * 56)
    print("  J.A.R.V.I.S. — Streaming + Browser TTS + Voice Mode")
    print("  Intent Classifier: ACTIVE (Tier 1-2 handled locally)")
    print("  Open http://localhost:5000 in Microsoft Edge")
    print("  (Edge has the best neural voices)")
    print("=" * 56)
    print()
    app.run(debug=True, port=5000)
