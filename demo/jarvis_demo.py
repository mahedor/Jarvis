"""
Jarvis Phase 1 — Demo v1 (Terminal)
====================================
The simplest possible starting point.
Talk to Jarvis in your terminal. No hardware needed.

Setup:
  1. pip install anthropic
  2. Set your API key (see below)
  3. python jarvis_demo.py
  4. Start talking to Jarvis!

This is your FOUNDATION. Everything else builds on top of this.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file (if using Option B for API key)
import json
import re
from datetime import datetime

# ─── CHANGE THIS ──────────────────────────────────────────────
# Option A: Paste your key directly (fine for local dev, NEVER commit this)
# API_KEY = "YOUR_API_KEY_HERE"

# Option B (better): Set an environment variable instead
# In PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
# Then uncomment:
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# ──────────────────────────────────────────────────────────────

try:
    from anthropic import Anthropic
except ImportError:
    print("\n❌ The 'anthropic' package is not installed.")
    print("   Run: pip install anthropic")
    print("   Then try again.\n")
    exit(1)

# ─── Device State Tracking ────────────────────────────────────
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
        elif "brightness" in device_states[entity_id]:
            device_states[entity_id]["brightness"] = 255
    elif "turn_off" in service or "close" in service:
        device_states[entity_id]["state"] = "closed" if "cover" in entity_id else "off"
        if "brightness" in device_states[entity_id]:
            device_states[entity_id]["brightness"] = 0


def build_system_prompt():
    """Build the system prompt with live device states."""
    status_lines = []
    for eid, info in device_states.items():
        extra = ""
        if "brightness" in info and info["state"] == "on":
            extra = f" (brightness: {round(info['brightness']/255*100)}%)"
        status_lines.append(f"  - {info['friendly_name']} ({eid}): {info['state'].upper()}{extra}")
    status_text = "\n".join(status_lines)

    return f"""You are Jarvis, a personal AI assistant for a smart home system.
Your personality is helpful, witty, and concise — like the Jarvis from Iron Man but
more casual and friendly. You're talking to Michael in his bedroom.

RULES:
- Keep responses SHORT (1-3 sentences max) since they'll eventually be spoken via TTS.
- Be natural and conversational, not robotic.
- You can be playful — you're Jarvis after all.

DEVICE CONTROL:
When the user asks to control a device, respond with a natural confirmation AND include
an action block on its own line in this exact format:
[ACTION: {{"service": "light.turn_on", "entity_id": "light.bedroom", "data": {{}}}}]

Available devices and their CURRENT STATUS:
{status_text}

Available services:
- light.turn_on / light.turn_off (data: {{"brightness": 0-255}})
- switch.turn_on / switch.turn_off
- cover.open_cover / cover.close_cover

IMPORTANT: When asked about device status, use the CURRENT STATUS above to give accurate answers.
If a light is already on and the user asks to turn it on, let them know it's already on.

If the user says something conversational (not device-related), just respond naturally.
If you're unsure which device, ask for clarification.
"""


# ─── Action Parser ─────────────────────────────────────────────
def parse_actions(response_text):
    """Extract ACTION blocks from Jarvis's response."""
    actions = []
    pattern = r'\[ACTION:\s*({.*?})\s*\]'
    for match in re.finditer(pattern, response_text):
        try:
            action = json.loads(match.group(1))
            actions.append(action)
        except json.JSONDecodeError:
            pass
    return actions


def clean_response(response_text):
    """Remove ACTION blocks — the user only hears the natural language part."""
    cleaned = re.sub(r'\[ACTION:.*?\]', '', response_text).strip()
    return cleaned


# ─── Interaction Logger ────────────────────────────────────────
def log_interaction(command, response, actions):
    """
    Log every interaction to a JSONL file.
    This is the start of your data warehouse.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "response": response,
        "actions": actions,
        "intent": "home_control" if actions else "conversation",
    }
    
    os.makedirs("logs", exist_ok=True)
    with open("logs/interactions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# ─── Main Chat Loop ───────────────────────────────────────────
def main():
    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY:
        print("\n❌ You need to set your Anthropic API key!")
        print("   Open this file and replace YOUR_API_KEY_HERE with your key.")
        print("   Get a key at: https://console.anthropic.com\n")
        return

    client = Anthropic(api_key=API_KEY)
    conversation_history = []

    print()
    print("=" * 56)
    print("  J.A.R.V.I.S. — Phase 1 Demo")
    print("  Just A Rather Very Intelligent System")
    print("=" * 56)
    print()
    print("  Type a command or have a conversation.")
    print("  Try: 'turn on the bedroom lights'")
    print("  Try: 'dim the lamp to 50%'")
    print("  Try: 'what time is it?'")
    print("  Try: 'open the blinds'")
    print()
    print("  Type 'quit' to exit.")
    print("  Type 'log' to see interaction history.")
    print("  Type 'status' to see device states.")
    print("-" * 56)

    while True:
        try:
            user_input = input("\n  You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Jarvis > Goodbye, Michael. Try not to break anything.\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Jarvis > Shutting down. Don't stay up too late.\n")
            break

        if user_input.lower() == "status":
            print("\n  [Device Status]")
            for eid, info in device_states.items():
                extra = ""
                if "brightness" in info and info["state"] == "on":
                    extra = f" ({round(info['brightness']/255*100)}%)"
                icon = "🟢" if info["state"] in ("on", "open") else "⚫"
                print(f"  {icon} {info['friendly_name']}: {info['state'].upper()}{extra}")
            continue

        if user_input.lower() == "log":
            log_path = "logs/interactions.jsonl"
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    lines = f.readlines()
                print(f"\n  [{len(lines)} interactions logged]")
                for line in lines[-5:]:  # Show last 5
                    entry = json.loads(line)
                    intent_tag = "🏠" if entry["intent"] == "home_control" else "💬"
                    print(f"  {intent_tag} {entry['timestamp'][:19]} | {entry['command'][:50]}")
            else:
                print("\n  [No interactions logged yet]")
            continue

        # Add to conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Keep conversation history manageable (last 20 messages)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        try:
            # Call Claude with CURRENT device states injected into the prompt
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=build_system_prompt(),
                messages=conversation_history,
            )

            full_response = response.content[0].text
            actions = parse_actions(full_response)
            spoken_response = clean_response(full_response)

            # Update device states based on parsed actions
            for action in actions:
                update_device_state(action)

            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": full_response})

            # Display response
            print(f"\n  Jarvis > {spoken_response}")

            # Display any actions that were parsed
            if actions:
                for action in actions:
                    service = action.get("service", "unknown")
                    entity = action.get("entity_id", "unknown")
                    data = action.get("data", {})
                    print(f"  ⚡ Action: {service} → {entity}", end="")
                    if data:
                        print(f" (data: {data})", end="")
                    print()

            # Log the interaction
            log_interaction(user_input, spoken_response, actions)

        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                print("\n  ❌ API key error. Check that your key is correct.")
            elif "rate" in error_msg.lower():
                print("\n  ⏳ Rate limited. Wait a moment and try again.")
            else:
                print(f"\n  ❌ Error: {error_msg}")


if __name__ == "__main__":
    main()
