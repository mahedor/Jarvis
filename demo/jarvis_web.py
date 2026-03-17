"""
Jarvis Phase 1 — Demo v2 (Streaming + Browser TTS + Notion)
==============================================================
Claude's response streams to the browser in real-time.
Browser speaks each sentence INSTANTLY via Web Speech API.
No server-side TTS, no file generation, no latency.
Notion integration reads your databases for time tracking queries.

Open in Microsoft Edge for the best neural voices.

Setup:
  pip install anthropic flask python-dotenv notion-client
  python jarvis_web.py
  Open http://localhost:5000 in Microsoft Edge
"""

import os
import json
import re
import time as _time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

try:
    from anthropic import Anthropic
    from flask import Flask, request, jsonify, render_template_string
except ImportError:
    print("\n❌ Missing packages. Run:")
    print("   pip install anthropic flask python-dotenv notion-client")
    print("   Then try again.\n")
    exit(1)

# ─── Notion Setup ──────────────────────────────────────────────
NOTION_AVAILABLE = False
try:
    from notion_client import Client as NotionClient
    NOTION_AVAILABLE = True
except ImportError:
    print("  📝 Notion not available. Run: pip install notion-client")

# ─── CONFIG ────────────────────────────────────────────────────
API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
# Or set it directly: API_KEY = "sk-ant-..."

app = Flask(__name__)
client = Anthropic(api_key=API_KEY)
conversation_history = []

# ─── Device State Tracking ─────────────────────────────────────
# This simulates what Home Assistant would provide in the real setup.
# Each device has a state and optional attributes (like brightness).
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
            device_states[entity_id]["brightness"] = 255  # Default to full brightness
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


def build_system_prompt():
    """Build the system prompt with LIVE device states + Notion data."""
    notion_data = get_notion_summary()
    notion_section = ""
    if notion_data:
        notion_section = f"""

NOTION DATA (Michael's workspace):
{notion_data}

UNDERSTANDING THE DATA:
- The Work Table tracks 30-minute focused work blocks.
- Productivity is rated 1-10. A 7 means natural, focused pace (like a 70% passing grade).
  Above 7 means pushing faster than normal. Below 7 means distracted.
- "Distracted? No" + Productivity 7+ = a good focused block.
- "Distracted? Yes" + low Productivity = a rough block.
- Difference (Hours) shows how long the actual block was.
- The Start Table likely tracks when work sessions begin.

When Michael asks about his time tracking, productivity, focus, or work habits,
use this data to give accurate, insightful answers. You can summarize trends,
calculate averages, spot patterns (e.g. "you're most productive in the evening"),
count focused vs distracted blocks, or give encouragement. Keep it concise.
If the data doesn't have what he's asking about, say what you CAN see."""

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
{get_device_status_text()}

Available services:
- light.turn_on / light.turn_off (data: {{"brightness": 0-255}})
- switch.turn_on / switch.turn_off
- cover.open_cover / cover.close_cover

IMPORTANT: When asked about device status, use the CURRENT STATUS above to give accurate answers.
If a light is already on and the user asks to turn it on, let them know it's already on.
If they ask "what's on?" or "status?", report the current state of all devices.
{notion_section}
If the user says something conversational (not device-related), just respond naturally.
If you're unsure which device, ask for clarification.
"""


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


def log_interaction(command, response, actions):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "response": response,
        "actions": actions,
        "intent": "home_control" if actions else "conversation",
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/interactions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Notion Integration ────────────────────────────────────────
# Reads databases shared with the Jarvis integration.
# Data is fetched on each request (with caching) and injected
# into Claude's system prompt so Jarvis can answer questions.

notion = NotionClient(auth=NOTION_API_KEY) if (NOTION_AVAILABLE and NOTION_API_KEY) else None
_notion_cache = {"data": "", "ts": 0}
NOTION_CACHE_TTL = 60  # seconds

# Direct database IDs — bypasses flaky search endpoint
NOTION_DATABASES = {
    "Start Table": "d951c0d353b7498897e1a5c2ccf47599",
    "Work Table": "daedb7b1888f46149d26211c82dbe13b",
}


def fetch_notion_databases():
    """Discover all databases shared with the Jarvis integration."""
    if not notion:
        return []
    try:
        resp = notion.search(filter={"property": "object", "value": "database"})
        return resp.get("results", [])
    except Exception as e:
        print(f"  Notion error: {e}")
        return []


def format_notion_prop(prop):
    """Extract a readable value from a Notion property."""
    t = prop.get("type", "")
    if t == "title":
        return "".join(x.get("plain_text", "") for x in prop.get("title", []))
    elif t == "rich_text":
        return "".join(x.get("plain_text", "") for x in prop.get("rich_text", []))
    elif t == "number":
        v = prop.get("number")
        return str(v) if v is not None else ""
    elif t == "select":
        s = prop.get("select")
        return s.get("name", "") if s else ""
    elif t == "multi_select":
        return ", ".join(s.get("name", "") for s in prop.get("multi_select", []))
    elif t == "date":
        d = prop.get("date")
        if d:
            start = d.get("start", "")
            end = d.get("end", "")
            return f"{start} to {end}" if end else start
        return ""
    elif t == "checkbox":
        return "Yes" if prop.get("checkbox") else "No"
    elif t == "status":
        s = prop.get("status")
        return s.get("name", "") if s else ""
    elif t == "formula":
        f = prop.get("formula", {})
        ft = f.get("type", "")
        return str(f.get(ft, ""))
    elif t == "rollup":
        r = prop.get("rollup", {})
        rt = r.get("type", "")
        if rt == "number":
            return str(r.get("number", ""))
        return ""
    elif t == "relation":
        return str(len(prop.get("relation", []))) + " linked"
    elif t == "people":
        return ", ".join(p.get("name", "") for p in prop.get("people", []))
    elif t == "url":
        return prop.get("url", "") or ""
    elif t == "email":
        return prop.get("email", "") or ""
    elif t == "phone_number":
        return prop.get("phone_number", "") or ""
    elif t == "created_time":
        return prop.get("created_time", "")[:10]
    elif t == "last_edited_time":
        return prop.get("last_edited_time", "")[:10]
    else:
        return ""


def query_notion_db(database_id, limit=20):
    """Query a Notion database and return formatted rows."""
    if not NOTION_API_KEY:
        return []
    try:
        import requests
        resp = requests.post(
            f"https://api.notion.com/v1/databases/{database_id}/query",
            headers={
                "Authorization": f"Bearer {NOTION_API_KEY}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            },
            json={
                "page_size": limit,
                "sorts": [{"timestamp": "created_time", "direction": "descending"}],
            },
        )
        data = resp.json()
        if "results" not in data:
            print(f"  Notion API error: {data.get('message', data)}")
            return []
        rows = []
        for page in data["results"]:
            props = page.get("properties", {})
            row = {}
            for name, prop in props.items():
                val = format_notion_prop(prop)
                if val:
                    row[name] = val
            if row:
                rows.append(row)
        return rows
    except Exception as e:
        print(f"  Notion query error: {e}")
        return []


def get_notion_summary():
    """Build a formatted summary of all shared Notion databases."""
    now = _time.time()
    if _notion_cache["data"] and (now - _notion_cache["ts"]) < NOTION_CACHE_TTL:
        return _notion_cache["data"]

    if not notion:
        return ""

    sections = []
    for db_name, db_id in NOTION_DATABASES.items():
        rows = query_notion_db(db_id, limit=30)
        if not rows:
            continue

        lines = [f"  {db_name} ({len(rows)} entries):"]
        for i, row in enumerate(rows, 1):
            parts = [f"{k}: {v}" for k, v in row.items()]
            lines.append(f"    {i}. " + " | ".join(parts))
        sections.append("\n".join(lines))

    summary = "\n\n".join(sections)
    _notion_cache["data"] = summary
    _notion_cache["ts"] = now

    if summary:
        print(f"  📝 Notion: loaded {len(sections)} database(s)")

    return summary


# ─── Web UI ────────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>J.A.R.V.I.S.</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --accent: #7c3aed;
    --accent-rgb: 124, 58, 237;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Style all scrollbars to match the aesthetic */
  ::-webkit-scrollbar { width: 4px; height: 0; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

  .header {
    padding: 24px 32px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
  }

  .header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 300;
    font-size: 14px;
    letter-spacing: 6px;
    text-transform: uppercase;
    color: var(--accent);
  }

  .header p {
    font-size: 12px;
    color: #555;
    margin-top: 4px;
  }

  .theme-picker {
    display: flex;
    align-items: center;
    gap: 0;
    padding-top: 6px;
    position: relative;
  }

  .theme-trigger {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.2);
    background: var(--accent);
    cursor: pointer;
    transition: all 0.2s;
    flex-shrink: 0;
  }

  .theme-trigger:hover { border-color: rgba(255,255,255,0.5); }

  .theme-tray {
    display: flex;
    gap: 6px;
    align-items: center;
    max-width: 0;
    opacity: 0;
    transition: max-width 0.3s ease, opacity 0.2s ease, padding 0.3s ease;
    padding-left: 0;
    overflow: visible;
    clip-path: inset(0 100% 0 0);
  }

  .theme-picker:hover .theme-tray {
    max-width: 200px;
    opacity: 1;
    padding-left: 10px;
    clip-path: inset(-4px 0 -4px 0);
  }

  .theme-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.08);
    cursor: pointer;
    transition: all 0.15s;
    opacity: 0.6;
    flex-shrink: 0;
  }

  .theme-dot:hover { opacity: 1; transform: scale(1.4); }
  .theme-dot.active { opacity: 1; border-color: rgba(255,255,255,0.5); }

  .status-bar {
    display: flex;
    gap: 16px;
    padding: 12px 32px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: #444;
  }

  .status-item { display: flex; align-items: center; gap: 6px; }
  .status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #2d5a2d;
    animation: pulse 2s ease-in-out infinite;
  }
  .status-dot.active { background: #4ade80; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

  .devices-bar {
    display: flex;
    gap: 10px;
    padding: 12px 32px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    flex-wrap: wrap;
  }

  .device-chip {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    color: #555;
    transition: all 0.3s ease;
  }

  .device-chip.on {
    border-color: rgba(var(--accent-rgb), 0.4);
    color: var(--accent);
    background: rgba(var(--accent-rgb), 0.08);
  }

  .chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px 32px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .message {
    max-width: 600px;
    animation: fadeIn 0.3s ease;
  }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  .message.user {
    align-self: flex-end;
    background: rgba(var(--accent-rgb), 0.12);
    border: 1px solid rgba(var(--accent-rgb), 0.2);
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    font-size: 14px;
    line-height: 1.5;
  }

  .message.jarvis {
    align-self: flex-start;
    padding: 12px 0;
  }

  .message.jarvis .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 6px;
  }

  .message.jarvis .text {
    font-size: 14px;
    line-height: 1.6;
    color: #ccc;
  }

  .action-card {
    margin-top: 8px;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid rgba(74, 222, 128, 0.2);
    background: rgba(74, 222, 128, 0.05);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4ade80;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .action-card .bolt { font-size: 14px; }

  .input-area {
    padding: 16px 32px 24px;
    border-top: 1px solid rgba(255,255,255,0.06);
  }

  .input-row {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .input-row input {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 14px 20px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: #e0e0e0;
    outline: none;
    transition: border-color 0.2s;
  }

  .input-row input:focus {
    border-color: rgba(var(--accent-rgb), 0.5);
  }

  .input-row input::placeholder { color: #333; }

  .input-row button {
    background: transparent;
    border: 1px solid rgba(var(--accent-rgb), 0.4);
    border-radius: 12px;
    padding: 14px 24px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 400;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .input-row button:hover { background: rgba(var(--accent-rgb), 0.1); border-color: rgba(var(--accent-rgb), 0.6); }
  .input-row button:active { transform: scale(0.97); }
  .input-row button:disabled { background: transparent; border-color: rgba(255,255,255,0.08); color: #444; cursor: not-allowed; }

  .mic-btn {
    background: transparent;
    border: 1px solid rgba(var(--accent-rgb), 0.4);
    border-radius: 12px;
    padding: 12px;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .mic-btn:hover { background: rgba(var(--accent-rgb), 0.1); border-color: rgba(var(--accent-rgb), 0.6); }

  .mic-btn.recording {
    border-color: #e24b4a;
    background: rgba(226, 75, 74, 0.1);
    animation: mic-pulse 1.2s ease-in-out infinite;
  }

  @keyframes mic-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(226, 75, 74, 0.3); }
    50% { box-shadow: 0 0 0 8px rgba(226, 75, 74, 0); }
  }

  .mic-icon { width: 20px; height: 20px; }

  .input-row input { min-width: 0; }

  .typing-indicator {
    display: none;
    align-self: flex-start;
    padding: 12px 0;
  }

  .typing-indicator .dots {
    display: flex; gap: 4px;
  }

  .typing-indicator .dots span {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    animation: typing 1.2s ease-in-out infinite;
  }

  .typing-indicator .dots span:nth-child(2) { animation-delay: 0.15s; }
  .typing-indicator .dots span:nth-child(3) { animation-delay: 0.3s; }

  @keyframes typing {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.3; }
    30% { transform: translateY(-6px); opacity: 1; }
  }

  .suggestions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
  }

  .suggestion {
    font-size: 12px;
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    color: #555;
    cursor: pointer;
    transition: all 0.15s;
    background: none;
    font-family: 'Inter', sans-serif;
  }

  .suggestion:hover {
    border-color: rgba(var(--accent-rgb), 0.3);
    color: var(--accent);
  }

  .tts-toggle {
    margin-left: auto;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(74, 222, 128, 0.3);
    background: rgba(74, 222, 128, 0.08);
    color: #4ade80;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    cursor: pointer;
    transition: all 0.15s;
  }

  .tts-toggle:hover { background: rgba(74, 222, 128, 0.15); }

  .tts-toggle.off {
    border-color: rgba(255,255,255,0.08);
    background: none;
    color: #555;
  }

  /* ─── Screensaver ─────────────────────────────────── */
  .screensaver {
    position: fixed;
    inset: 0;
    background: #06060a;
    z-index: 100;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: none;
    transition: opacity 0.6s ease;
  }

  .screensaver.dismissed {
    opacity: 0;
    pointer-events: none;
  }

  .ss-cursor-core {
    position: fixed;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #666;
    box-shadow: 0 0 6px var(--accent), 0 0 15px rgba(var(--accent-rgb), 0.4);
    pointer-events: none;
    z-index: 9999;
    transform: translate(-50%, -50%);
    transition: opacity 0.15s;
  }

  .ss-cursor-ring {
    position: fixed;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1px solid rgba(var(--accent-rgb), 0.4);
    pointer-events: none;
    z-index: 9998;
    transform: translate(-50%, -50%);
    transition: width 0.25s ease-out, height 0.25s ease-out, border-color 0.25s, background 0.25s;
  }

  .ss-cursor-ring.hover {
    width: 44px;
    height: 44px;
    border-color: rgba(var(--accent-rgb), 0.6);
    background: rgba(var(--accent-rgb), 0.06);
  }

  .ss-cursor-core.hidden, .ss-cursor-ring.hidden { opacity: 0; }

  .ss-ring {
    width: 240px;
    height: 240px;
    border-radius: 50%;
    border: 1px solid rgba(var(--accent-rgb), 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    animation: ss-breathe 4s ease-in-out infinite;
  }

  .ss-ring::before {
    content: '';
    position: absolute;
    inset: -6px;
    border-radius: 50%;
    border: 1px solid transparent;
    border-top-color: rgba(var(--accent-rgb), 0.4);
    animation: ss-spin 3s linear infinite;
  }

  .ss-ring::after {
    content: '';
    position: absolute;
    inset: -14px;
    border-radius: 50%;
    border: 1px solid transparent;
    border-bottom-color: rgba(var(--accent-rgb), 0.2);
    animation: ss-spin 5s linear infinite reverse;
  }

  @keyframes ss-spin { to { transform: rotate(360deg); } }
  @keyframes ss-breathe {
    0%, 100% { box-shadow: 0 0 30px rgba(var(--accent-rgb), 0.05); }
    50% { box-shadow: 0 0 50px rgba(var(--accent-rgb), 0.12); }
  }

  .ss-initial {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 300;
    font-size: 22px;
    letter-spacing: 10px;
    color: rgba(255,255,255,0.9);
    padding-left: 10px;
  }

  .ss-time {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 300;
    font-size: 48px;
    color: rgba(255,255,255,0.6);
    margin-top: 40px;
    letter-spacing: 4px;
  }

  .ss-date {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 300;
    font-size: 13px;
    color: rgba(255,255,255,0.2);
    margin-top: 8px;
    letter-spacing: 3px;
    text-transform: uppercase;
  }

  .ss-hint {
    position: absolute;
    bottom: 40px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.25);
    letter-spacing: 2px;
    animation: ss-hint-fade 3s ease-in-out infinite;
  }

  @keyframes ss-hint-fade {
    0%, 100% { opacity: 0.25; }
    50% { opacity: 0.5; }
  }

  /* Background effects */
  .ss-canvas {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }

  .ss-grid {
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(var(--accent-rgb), 0.08) 1px, transparent 1px),
      linear-gradient(90deg, rgba(var(--accent-rgb), 0.08) 1px, transparent 1px);
    background-size: 60px 60px;
    mask-image: radial-gradient(ellipse at center, transparent 15%, black 70%);
    -webkit-mask-image: radial-gradient(ellipse at center, transparent 15%, black 70%);
    display: none;
  }

  .ss-grid.active { display: block; }

  .ss-pulse {
    position: absolute;
    width: 240px;
    height: 240px;
    border-radius: 50%;
    top: 0;
    left: 0;
    pointer-events: none;
    display: none;
  }

  .ss-pulse.active { display: block; }

  .ss-pulse-wave {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 1px solid rgba(var(--accent-rgb), 0.3);
    animation: ss-pulse-expand 4s ease-out infinite;
  }

  .ss-pulse-wave:nth-child(2) { animation-delay: 1.3s; }
  .ss-pulse-wave:nth-child(3) { animation-delay: 2.6s; }

  @keyframes ss-pulse-expand {
    0% { transform: scale(1); opacity: 0.4; }
    100% { transform: scale(4); opacity: 0; }
  }

  /* Effect toggle panel */
  .ss-controls {
    position: absolute;
    bottom: 40px;
    left: 24px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    z-index: 101;
    cursor: none;
  }

  .ss-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    color: rgba(255,255,255,0.2);
    cursor: none;
    transition: color 0.15s;
    user-select: none;
  }

  .ss-toggle:hover { color: rgba(255,255,255,0.4); }

  .ss-toggle .ss-check {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    border: 1px solid rgba(255,255,255,0.15);
    transition: all 0.15s;
    flex-shrink: 0;
  }

  .ss-toggle.on .ss-check {
    background: var(--accent);
    border-color: var(--accent);
  }

  .ss-toggle.on { color: rgba(255,255,255,0.4); }
</style>
</head>
<body>

<div class="screensaver" id="screensaver">
  <div class="ss-cursor-core" id="ss-cursor-core"></div>
  <div class="ss-cursor-ring" id="ss-cursor-ring"></div>
  <div class="ss-grid" id="ss-grid"></div>
  <canvas class="ss-canvas" id="ss-canvas"></canvas>
  <div style="position:relative">
    <div class="ss-pulse" id="ss-pulse">
      <div class="ss-pulse-wave"></div>
      <div class="ss-pulse-wave"></div>
      <div class="ss-pulse-wave"></div>
    </div>
    <div class="ss-ring">
      <span class="ss-initial">JARVIS</span>
    </div>
  </div>
  <div class="ss-time" id="ss-time"></div>
  <div class="ss-date" id="ss-date"></div>
  <div class="ss-hint">click or press any key</div>
  <div class="ss-controls" id="ss-controls">
    <div class="ss-toggle" onclick="toggleEffect('particles', this)"><div class="ss-check"></div>particles</div>
    <div class="ss-toggle" onclick="toggleEffect('grid', this)"><div class="ss-check"></div>grid</div>
    <div class="ss-toggle" onclick="toggleEffect('pulse', this)"><div class="ss-check"></div>pulse</div>
    <div class="ss-toggle" onclick="toggleEffect('orbits', this)"><div class="ss-check"></div>orbits</div>
    <div class="ss-toggle" onclick="toggleEffect('warpgrid', this)"><div class="ss-check"></div>warp</div>
    <div class="ss-toggle" onclick="toggleEffect('neural', this)"><div class="ss-check"></div>neural</div>
    <div class="ss-toggle" onclick="toggleEffect('tendrils', this)"><div class="ss-check"></div>tendrils</div>
    <div class="ss-toggle" onclick="toggleEffect('starfield', this)"><div class="ss-check"></div>starfield</div>
    <div class="ss-toggle" onclick="toggleEffect('circuits', this)"><div class="ss-check"></div>circuits</div>
    <div class="ss-toggle" onclick="toggleEffect('helix', this)"><div class="ss-check"></div>helix</div>
  </div>
</div>

<div class="header">
  <div>
    <h1>J.A.R.V.I.S.</h1>
    <p>Just A Rather Very Intelligent System — Phase 1 Demo</p>
  </div>
  <div class="theme-picker">
    <div class="theme-trigger" id="theme-trigger"></div>
    <div class="theme-tray">
      <div class="theme-dot" style="background:#4a9eff" onclick="setTheme('#4a9eff','74,158,255',this)"></div>
      <div class="theme-dot" style="background:#00e5c8" onclick="setTheme('#00e5c8','0,229,200',this)"></div>
      <div class="theme-dot active" style="background:#7c3aed" onclick="setTheme('#7c3aed','124,58,237',this)"></div>
      <div class="theme-dot" style="background:#f59e0b" onclick="setTheme('#f59e0b','245,158,11',this)"></div>
      <div class="theme-dot" style="background:#ef4444" onclick="setTheme('#ef4444','239,68,68',this)"></div>
      <div class="theme-dot" style="background:#22c55e" onclick="setTheme('#22c55e','34,197,94',this)"></div>
    </div>
  </div>
</div>

<div class="status-bar">
  <div class="status-item"><div class="status-dot active"></div> Orchestrator online</div>
  <div class="status-item"><div class="status-dot"></div> Whisper (not connected)</div>
  <div class="status-item"><div class="status-dot active"></div> Claude API</div>
  <button class="tts-toggle" id="tts-toggle" onclick="toggleTTS()">TTS: on</button>
</div>

<div class="devices-bar" id="devices">
  <div class="device-chip" id="dev-light-bedroom">bedroom light: off</div>
  <div class="device-chip" id="dev-light-lamp">bedroom lamp: off</div>
  <div class="device-chip" id="dev-switch-fan">bedroom fan: off</div>
  <div class="device-chip" id="dev-cover-blinds">blinds: closed</div>
</div>

<div class="chat-area" id="chat">
  <div class="message jarvis">
    <div class="label">Jarvis</div>
    <div class="text">Good evening, Michael. Systems are online. What can I do for you?</div>
  </div>
</div>

<div class="typing-indicator" id="typing">
  <div class="dots"><span></span><span></span><span></span></div>
</div>

<div class="input-area">
  <div class="input-row">
    <button class="mic-btn" id="mic-btn" onclick="toggleMic()">
      <svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="1" width="6" height="12" rx="3"/>
        <path d="M19 10v1a7 7 0 01-14 0v-1"/>
        <line x1="12" y1="19" x2="12" y2="23"/>
        <line x1="8" y1="23" x2="16" y2="23"/>
      </svg>
    </button>
    <input type="text" id="input" placeholder="Talk to Jarvis or click the mic..." autocomplete="off" />
    <button id="send" onclick="sendMessage()">Send</button>
  </div>
  <div class="suggestions" id="suggestions">
    <button class="suggestion" onclick="useSuggestion(this)">Turn on the bedroom lights</button>
    <button class="suggestion" onclick="useSuggestion(this)">Dim the lamp to 50%</button>
    <button class="suggestion" onclick="useSuggestion(this)">Open the blinds</button>
    <button class="suggestion" onclick="useSuggestion(this)">What's the status of everything?</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const typing = document.getElementById('typing');
const sendBtn = document.getElementById('send');
const suggestions = document.getElementById('suggestions');

// ─── Theme Switching ───────────────────────────────────
let currentAccent = '#7c3aed';

function setTheme(hex, rgb, dot) {
  currentAccent = hex;
  document.documentElement.style.setProperty('--accent', hex);
  document.documentElement.style.setProperty('--accent-rgb', rgb);
  document.getElementById('mic-btn').querySelector('svg').setAttribute('stroke', hex);
  document.getElementById('theme-trigger').style.background = hex;
  document.querySelectorAll('.theme-dot').forEach(d => d.classList.remove('active'));
  if (dot) dot.classList.add('active');
}

const deviceChipMap = {
  'light.bedroom': 'dev-light-bedroom',
  'light.bedroom_lamp': 'dev-light-lamp',
  'switch.bedroom_fan': 'dev-switch-fan',
  'cover.bedroom_blinds': 'dev-cover-blinds',
};

const deviceLabels = {
  'light.bedroom': 'bedroom light',
  'light.bedroom_lamp': 'bedroom lamp',
  'switch.bedroom_fan': 'bedroom fan',
  'cover.bedroom_blinds': 'blinds',
};

function syncDeviceChips(states) {
  if (!states) return;
  for (const [entityId, info] of Object.entries(states)) {
    const chipId = deviceChipMap[entityId];
    const chip = document.getElementById(chipId);
    if (!chip) continue;
    const label = deviceLabels[entityId] || entityId;
    const isOn = info.state === 'on' || info.state === 'open';
    let stateText = info.state;
    if (info.brightness && info.state === 'on') {
      const pct = Math.round(info.brightness / 255 * 100);
      stateText = 'on (' + pct + '%)';
    }
    chip.textContent = label + ': ' + stateText;
    chip.classList.toggle('on', isOn);
  }
}

function addMessage(role, text, actions) {
  const div = document.createElement('div');
  div.className = 'message ' + role;

  if (role === 'user') {
    div.textContent = text;
  } else {
    div.innerHTML = '<div class="label">Jarvis</div><div class="text">' + text + '</div>';
    if (actions && actions.length > 0) {
      actions.forEach(a => {
        const card = document.createElement('div');
        card.className = 'action-card';
        card.innerHTML = '<span class="bolt">&#9889;</span> ' + (a.service||'') + ' &rarr; ' + (a.entity_id||'');
        div.appendChild(card);
      });
    }
  }

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

// ─── Browser TTS (Web Speech API) ──────────────────────
// This is INSTANT — no file generation, no network call.
// Open in Microsoft Edge for the best neural voices.
let ttsOn = true;
let ttsVoice = null;

function initTTS() {
  const voices = speechSynthesis.getVoices();
  // Prefer: Microsoft Ryan (Edge), then any British male, then default
  const prefs = ['ryan', 'george', 'ryanneural'];
  for (const p of prefs) {
    const v = voices.find(v => v.name.toLowerCase().includes(p));
    if (v) { ttsVoice = v; break; }
  }
  if (!ttsVoice) {
    // Fallback to first English voice
    ttsVoice = voices.find(v => v.lang.startsWith('en')) || voices[0];
  }
}

speechSynthesis.onvoiceschanged = initTTS;
initTTS();

function speakText(text) {
  if (!ttsOn || !text.trim()) return;
  speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  if (ttsVoice) utter.voice = ttsVoice;
  utter.rate = 1.05;
  utter.pitch = 0.95;
  speechSynthesis.speak(utter);
}

function toggleTTS() {
  ttsOn = !ttsOn;
  if (!ttsOn) speechSynthesis.cancel();
  const btn = document.getElementById('tts-toggle');
  btn.textContent = 'TTS: ' + (ttsOn ? 'on' : 'off');
  btn.classList.toggle('off', !ttsOn);
}

// ─── Speech Recognition (mic input) ───────────────────
// Uses the browser's built-in SpeechRecognition API.
// Click the mic → speak → text appears → auto-sends to Jarvis.
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isRecording = false;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onresult = (event) => {
    let transcript = '';
    let isFinal = false;
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
      if (event.results[i].isFinal) isFinal = true;
    }
    input.value = transcript;
    if (isFinal) {
      stopMic();
      sendMessage();
    }
  };

  recognition.onend = () => { stopMic(); };
  recognition.onerror = (e) => {
    console.log('Speech recognition error:', e.error);
    stopMic();
  };
}

function toggleMic() {
  if (isRecording) { stopMic(); }
  else { startMic(); }
}

function startMic() {
  if (!recognition) {
    alert('Speech recognition not supported in this browser. Try Edge or Chrome.');
    return;
  }
  // Stop any TTS that's playing so mic doesn't pick it up
  speechSynthesis.cancel();
  isRecording = true;
  document.getElementById('mic-btn').classList.add('recording');
  document.getElementById('mic-btn').querySelector('svg').setAttribute('stroke', '#e24b4a');
  input.placeholder = 'Listening...';
  recognition.start();
}

function stopMic() {
  isRecording = false;
  document.getElementById('mic-btn').classList.remove('recording');
  document.getElementById('mic-btn').querySelector('svg').setAttribute('stroke', currentAccent);
  input.placeholder = 'Talk to Jarvis or click the mic...';
  try { recognition.stop(); } catch(e) {}
}

// ─── Chat ──────────────────────────────────────────────
// Response appears, then browser speaks it INSTANTLY.
async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  addMessage('user', text);
  input.value = '';
  sendBtn.disabled = true;
  typing.style.display = 'block';
  suggestions.style.display = 'none';
  chat.scrollTop = chat.scrollHeight;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    typing.style.display = 'none';
    addMessage('jarvis', data.response, data.actions);
    syncDeviceChips(data.device_states);

    // Speak via browser — INSTANT, no file generation
    speakText(data.response);
  } catch (err) {
    typing.style.display = 'none';
    addMessage('jarvis', 'Something went wrong. Check that the server is running.');
  }

  sendBtn.disabled = false;
  input.focus();
}

function useSuggestion(btn) {
  input.value = btn.textContent;
  sendMessage();
}

input.addEventListener('keydown', e => {
  if (e.key === 'Enter') sendMessage();
});

input.focus();

// ─── Screensaver ───────────────────────────────────
const screensaver = document.getElementById('screensaver');
const ssCore = document.getElementById('ss-cursor-core');
const ssRing = document.getElementById('ss-cursor-ring');
let ssActive = true;
let ssClockInterval;
let ssAnimFrame;

// Cursor: core follows mouse exactly, ring trails behind with smooth lerp
let mouseX = -100, mouseY = -100;
let ringX = -100, ringY = -100;

screensaver.addEventListener('mousemove', e => {
  mouseX = e.clientX;
  mouseY = e.clientY;
  ssCore.style.left = mouseX + 'px';
  ssCore.style.top = mouseY + 'px';
});

screensaver.addEventListener('mouseenter', () => {
  ssCore.classList.remove('hidden');
  ssRing.classList.remove('hidden');
});
screensaver.addEventListener('mouseleave', () => {
  ssCore.classList.add('hidden');
  ssRing.classList.add('hidden');
});

function animateRing() {
  ringX += (mouseX - ringX) * 0.12;
  ringY += (mouseY - ringY) * 0.12;
  ssRing.style.left = ringX + 'px';
  ssRing.style.top = ringY + 'px';
  if (ssActive) requestAnimationFrame(animateRing);
}
animateRing();

// Expand ring on hoverable elements
document.querySelectorAll('.ss-toggle, .ss-check, .theme-dot, .theme-trigger').forEach(el => {
  el.addEventListener('mouseenter', () => ssRing.classList.add('hover'));
  el.addEventListener('mouseleave', () => ssRing.classList.remove('hover'));
});

function updateSSClock() {
  const now = new Date();
  const h = now.getHours().toString().padStart(2, '0');
  const m = now.getMinutes().toString().padStart(2, '0');
  document.getElementById('ss-time').textContent = h + ':' + m;
  const days = ['sunday','monday','tuesday','wednesday','thursday','friday','saturday'];
  const months = ['january','february','march','april','may','june','july','august','september','october','november','december'];
  document.getElementById('ss-date').textContent = days[now.getDay()] + '  \u00b7  ' + months[now.getMonth()] + ' ' + now.getDate();
}

updateSSClock();
ssClockInterval = setInterval(updateSSClock, 10000);

// ─── Canvas Effects (particles + orbits) ───────────
const ssCanvas = document.getElementById('ss-canvas');
const ctx = ssCanvas.getContext('2d');
let particles = [];
let orbitDots = [];
let effectsOn = { particles: false, grid: false, pulse: false, orbits: false, warpgrid: false, neural: false, tendrils: false, starfield: false, circuits: false, helix: false };
let warpTime = 0;
let helixTime = 0;
let neuralTime = 0;

// Tendrils — organic branching dendrites growing from the ring
let tendrilBranches = [];
let tendrilTimer = 0;

function spawnTendril(cx, cy) {
  const angle = Math.random() * Math.PI * 2;
  const startR = 125;
  tendrilBranches.push({
    points: [{ x: cx + Math.cos(angle) * startR, y: cy + Math.sin(angle) * startR }],
    angle: angle,
    speed: 0.8 + Math.random() * 1.2,
    curl: (Math.random() - 0.5) * 0.06,
    life: 1,
    fade: 0.002 + Math.random() * 0.003,
    thickness: 0.3 + Math.random() * 0.8,
    branched: false,
    depth: 0,
  });
}

function branchTendril(parent, cx, cy) {
  if (parent.depth > 2) return;
  const last = parent.points[parent.points.length - 1];
  for (let i = 0; i < 2; i++) {
    const spread = (Math.random() - 0.5) * 0.8;
    tendrilBranches.push({
      points: [{ x: last.x, y: last.y }],
      angle: parent.angle + spread,
      speed: parent.speed * (0.6 + Math.random() * 0.3),
      curl: (Math.random() - 0.5) * 0.08,
      life: parent.life * 0.7,
      fade: parent.fade * 1.3,
      thickness: parent.thickness * 0.6,
      branched: false,
      depth: parent.depth + 1,
    });
  }
}

// Neural network nodes
let neurons = [];
let synapses = [];
let neuralPulses = [];

function initNeural() {
  neurons = [];
  synapses = [];
  const w = window.innerWidth;
  const h = window.innerHeight;
  const count = 35;
  for (let i = 0; i < count; i++) {
    neurons.push({
      x: Math.random() * w,
      y: Math.random() * h,
      r: 1 + Math.random() * 1.5,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      energy: Math.random(),
      pulsePhase: Math.random() * Math.PI * 2,
    });
  }
  // Connect nearby neurons
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      const dx = neurons[i].x - neurons[j].x;
      const dy = neurons[i].y - neurons[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 250) {
        synapses.push({ from: i, to: j, dist: dist });
      }
    }
  }
}
initNeural();

// Starfield
let stars = [];
for (let i = 0; i < 200; i++) {
  stars.push({
    x: (Math.random() - 0.5) * 2,
    y: (Math.random() - 0.5) * 2,
    z: Math.random() * 1,
    pz: 0,
  });
}

// Circuit traces
let circuits = [];
let circuitTimer = 0;

function spawnCircuit(cx, cy) {
  const angle = Math.random() * Math.PI * 2;
  const dist = 130 + Math.random() * 40;
  circuits.push({
    points: [{ x: cx + Math.cos(angle) * dist, y: cy + Math.sin(angle) * dist }],
    dir: Math.floor(Math.random() * 4),
    steps: 0,
    maxSteps: 15 + Math.floor(Math.random() * 30),
    speed: 2 + Math.random() * 2,
    life: 1,
    fade: 0.003 + Math.random() * 0.005,
  });
}

function resizeCanvas() {
  ssCanvas.width = window.innerWidth;
  ssCanvas.height = window.innerHeight;
  initNeural();
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Create particles
for (let i = 0; i < 60; i++) {
  particles.push({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    r: Math.random() * 1.5 + 0.5,
    o: Math.random() * 0.3 + 0.05,
  });
}

// Create orbit dots
for (let i = 0; i < 12; i++) {
  orbitDots.push({
    angle: (Math.PI * 2 / 12) * i + Math.random() * 0.5,
    radius: 150 + Math.random() * 80,
    speed: 0.002 + Math.random() * 0.003,
    dir: Math.random() > 0.5 ? 1 : -1,
    size: Math.random() * 2 + 1,
    o: Math.random() * 0.3 + 0.1,
  });
}

function getAccentRGB() {
  const s = getComputedStyle(document.documentElement).getPropertyValue('--accent-rgb').trim();
  const parts = s.split(',').map(Number);
  return parts.length === 3 ? parts : [124, 58, 237];
}

function drawEffects() {
  if (!ssActive) return;
  ctx.clearRect(0, 0, ssCanvas.width, ssCanvas.height);
  const ring = document.querySelector('.ss-ring');
  const rRect = ring.getBoundingClientRect();
  const cx = rRect.left + rRect.width / 2;
  const cy = rRect.top + rRect.height / 2;
  const rgb = getAccentRGB();

  if (effectsOn.particles) {
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = ssCanvas.width;
      if (p.x > ssCanvas.width) p.x = 0;
      if (p.y < 0) p.y = ssCanvas.height;
      if (p.y > ssCanvas.height) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + p.o + ')';
      ctx.fill();
    });
  }

  if (effectsOn.orbits) {
    orbitDots.forEach(d => {
      d.angle += d.speed * d.dir;
      const x = cx + Math.cos(d.angle) * d.radius;
      const y = cy + Math.sin(d.angle) * d.radius;
      ctx.beginPath();
      ctx.arc(x, y, d.size, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + d.o + ')';
      ctx.fill();
    });
  }

  if (effectsOn.warpgrid) {
    warpTime += 0.008;
    const w = ssCanvas.width;
    const h = ssCanvas.height;
    const spacing = 50;
    const warpStrength = 40;
    ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',0.07)';
    ctx.lineWidth = 0.5;

    // Horizontal lines that warp toward center
    for (let row = -2; row <= h / spacing + 2; row++) {
      ctx.beginPath();
      for (let col = 0; col <= w; col += 4) {
        const baseY = row * spacing;
        const dx = col - cx;
        const dy = baseY - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const warp = warpStrength * Math.sin(dist * 0.01 - warpTime * 3) / (1 + dist * 0.005);
        const fy = baseY + warp;
        if (col === 0) ctx.moveTo(col, fy);
        else ctx.lineTo(col, fy);
      }
      ctx.stroke();
    }

    // Vertical lines that warp toward center
    for (let col = -2; col <= w / spacing + 2; col++) {
      ctx.beginPath();
      for (let row = 0; row <= h; row += 4) {
        const baseX = col * spacing;
        const dx = baseX - cx;
        const dy = row - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const warp = warpStrength * Math.sin(dist * 0.01 - warpTime * 3) / (1 + dist * 0.005);
        const fx = baseX + warp;
        if (row === 0) ctx.moveTo(fx, row);
        else ctx.lineTo(fx, row);
      }
      ctx.stroke();
    }
  }

  // ── Neural Network ──
  if (effectsOn.neural) {
    neuralTime += 0.01;
    const w = ssCanvas.width;
    const h = ssCanvas.height;

    // Move neurons slowly (drifting, organic)
    neurons.forEach(n => {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < -20) n.x = w + 20;
      if (n.x > w + 20) n.x = -20;
      if (n.y < -20) n.y = h + 20;
      if (n.y > h + 20) n.y = -20;
      n.energy = 0.3 + Math.sin(n.pulsePhase + neuralTime * 2) * 0.3;
    });

    // Draw synapses (connections between neurons)
    synapses.forEach(s => {
      const a = neurons[s.from];
      const b = neurons[s.to];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 280) return;
      const alpha = (1 - dist / 280) * 0.25;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      const mx = (a.x + b.x) / 2 + (a.y - b.y) * 0.08;
      const my = (a.y + b.y) / 2 + (b.x - a.x) * 0.08;
      ctx.quadraticCurveTo(mx, my, b.x, b.y);
      ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',' + alpha + ')';
      ctx.lineWidth = 0.8;
      ctx.stroke();
    });

    // Draw neurons (nodes)
    neurons.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + (0.2 + n.energy * 0.25) + ')';
      ctx.fill();
    });
  }

  // ── Tendrils (organic dendrites) ──
  if (effectsOn.tendrils) {
    tendrilTimer++;
    if (tendrilTimer % 40 === 0 && tendrilBranches.filter(t => t.depth === 0).length < 8) {
      spawnTendril(cx, cy);
    }

    tendrilBranches.forEach(t => {
      if (t.life > 0.2) {
        const last = t.points[t.points.length - 1];
        t.angle += t.curl;
        const nx = last.x + Math.cos(t.angle) * t.speed;
        const ny = last.y + Math.sin(t.angle) * t.speed;
        t.points.push({ x: nx, y: ny });

        // Branch occasionally
        if (!t.branched && t.points.length > 15 && Math.random() < 0.02 && t.depth < 3) {
          t.branched = true;
          branchTendril(t, cx, cy);
        }
      }

      t.life -= t.fade;

      if (t.points.length > 1 && t.life > 0) {
        ctx.beginPath();
        ctx.moveTo(t.points[0].x, t.points[0].y);
        for (let i = 1; i < t.points.length; i++) {
          ctx.lineTo(t.points[i].x, t.points[i].y);
        }
        ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',' + Math.max(0, t.life * 0.3) + ')';
        ctx.lineWidth = t.thickness;
        ctx.stroke();

        // Glowing tip
        if (t.life > 0.2) {
          const tip = t.points[t.points.length - 1];
          ctx.beginPath();
          ctx.arc(tip.x, tip.y, t.thickness + 1, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + Math.max(0, t.life * 0.4) + ')';
          ctx.fill();
        }
      }
    });
    tendrilBranches = tendrilBranches.filter(t => t.life > 0);
  }

  // ── Starfield ──
  if (effectsOn.starfield) {
    const hw = ssCanvas.width / 2;
    const hh = ssCanvas.height / 2;
    stars.forEach(s => {
      s.pz = s.z;
      s.z -= 0.005;
      if (s.z <= 0) { s.z = 1; s.pz = 1; s.x = (Math.random() - 0.5) * 2; s.y = (Math.random() - 0.5) * 2; }
      const sx = (s.x / s.z) * hw + hw;
      const sy = (s.y / s.z) * hh + hh;
      const px = (s.x / s.pz) * hw + hw;
      const py = (s.y / s.pz) * hh + hh;
      const size = (1 - s.z) * 2;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(sx, sy);
      ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',' + (1 - s.z) * 0.5 + ')';
      ctx.lineWidth = size;
      ctx.stroke();
    });
  }

  // ── Circuit Traces ──
  if (effectsOn.circuits) {
    circuitTimer++;
    if (circuitTimer % 20 === 0 && circuits.length < 30) spawnCircuit(cx, cy);
    const dirs = [[1,0],[0,1],[-1,0],[0,-1]];
    circuits.forEach(c => {
      if (c.steps < c.maxSteps) {
        const last = c.points[c.points.length - 1];
        if (Math.random() < 0.2) c.dir = Math.floor(Math.random() * 4);
        const d = dirs[c.dir];
        c.points.push({ x: last.x + d[0] * c.speed * 3, y: last.y + d[1] * c.speed * 3 });
        c.steps++;
      }
      c.life -= c.fade;
      if (c.points.length > 1) {
        ctx.beginPath();
        ctx.moveTo(c.points[0].x, c.points[0].y);
        for (let i = 1; i < c.points.length; i++) ctx.lineTo(c.points[i].x, c.points[i].y);
        ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',' + Math.max(0, c.life * 0.4) + ')';
        ctx.lineWidth = 0.8;
        ctx.stroke();
        const tip = c.points[c.points.length - 1];
        ctx.beginPath();
        ctx.arc(tip.x, tip.y, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + Math.max(0, c.life * 0.6) + ')';
        ctx.fill();
      }
    });
    circuits = circuits.filter(c => c.life > 0);
  }

  // ── DNA Helix ──
  if (effectsOn.helix) {
    helixTime += 0.015;
    const helixX = ssCanvas.width - 60;
    const helixH = ssCanvas.height;
    const nodes = 40;
    const amp = 20;
    const spacing = helixH / nodes;
    for (let i = 0; i < nodes; i++) {
      const yy = i * spacing;
      const phase = (i * 0.3) + helixTime;
      const x1 = helixX + Math.sin(phase) * amp;
      const x2 = helixX + Math.sin(phase + Math.PI) * amp;
      const depth1 = (Math.sin(phase) + 1) / 2;
      const depth2 = (Math.sin(phase + Math.PI) + 1) / 2;
      // Connecting rung
      ctx.beginPath();
      ctx.moveTo(x1, yy);
      ctx.lineTo(x2, yy);
      ctx.strokeStyle = 'rgba(' + rgb.join(',') + ',0.06)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
      // Strand 1
      ctx.beginPath();
      ctx.arc(x1, yy, 2 + depth1, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + (0.1 + depth1 * 0.25) + ')';
      ctx.fill();
      // Strand 2
      ctx.beginPath();
      ctx.arc(x2, yy, 2 + depth2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + rgb.join(',') + ',' + (0.1 + depth2 * 0.25) + ')';
      ctx.fill();
    }
  }

  ssAnimFrame = requestAnimationFrame(drawEffects);
}

drawEffects();

// ─── Effect toggles ────────────────────────────────
function toggleEffect(name, el) {
  effectsOn[name] = !effectsOn[name];
  el.classList.toggle('on', effectsOn[name]);

  if (name === 'grid') document.getElementById('ss-grid').classList.toggle('active', effectsOn[name]);
  if (name === 'pulse') document.getElementById('ss-pulse').classList.toggle('active', effectsOn[name]);
}

// Prevent control clicks from dismissing screensaver
const ssControls = document.getElementById('ss-controls');
ssControls.addEventListener('click', e => e.stopPropagation());
ssControls.addEventListener('touchstart', e => e.stopPropagation());
ssControls.addEventListener('touchend', e => e.stopPropagation());

function dismissScreensaver(e) {
  if (!ssActive) return;
  if (ssControls.contains(e.target)) return;
  ssActive = false;
  screensaver.classList.add('dismissed');
  clearInterval(ssClockInterval);
  cancelAnimationFrame(ssAnimFrame);
  setTimeout(() => {
    fetch('/greeting', { method: 'POST' }).then(r => r.json()).then(d => speakText(d.text));
    screensaver.remove();
  }, 600);
}

document.addEventListener('click', dismissScreensaver);
document.addEventListener('keydown', dismissScreensaver);
document.addEventListener('touchstart', dismissScreensaver);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


GREETING = "Good evening, Michael. Systems are online. What can I do for you?"


@app.route("/greeting", methods=["POST"])
def greeting():
    """Return greeting text — browser handles TTS."""
    return jsonify({"text": GREETING})


@app.route("/chat", methods=["POST"])
def chat():
    """Stream Claude internally (faster), return JSON, browser handles TTS."""
    global conversation_history

    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "I didn't catch that.", "actions": []})

    conversation_history.append({"role": "user", "content": user_message})
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    try:
        current_prompt = build_system_prompt()

        # Stream Claude internally — faster than blocking call
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
        log_interaction(user_message, spoken_response, actions)

        return jsonify({
            "response": spoken_response,
            "actions": actions,
            "device_states": device_states,
        })

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}", "actions": []})


@app.route("/health")
def health():
    return jsonify({"status": "online", "service": "jarvis_orchestrator", "version": "0.3.0"})


@app.route("/devices")
def devices():
    return jsonify(device_states)


@app.route("/notion")
def notion_debug():
    """Debug endpoint — see what Notion data Jarvis can access."""
    summary = get_notion_summary()
    return jsonify({
        "available": NOTION_AVAILABLE,
        "connected": notion is not None,
        "databases": list(NOTION_DATABASES.keys()),
        "summary": summary,
    })


@app.route("/notion/raw")
def notion_raw():
    """Raw debug — see exactly what Notion API returns."""
    if not notion:
        return jsonify({"error": "Notion not connected"})
    results = {}
    for db_name, db_id in NOTION_DATABASES.items():
        try:
            import requests as req
            resp = req.post(
                f"https://api.notion.com/v1/databases/{db_id}/query",
                headers={
                    "Authorization": f"Bearer {NOTION_API_KEY}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json",
                },
                json={"page_size": 3},
            )
            data = resp.json()
            pages = data.get("results", [])
            results[db_name] = {
                "num_results": len(pages),
                "first_page_properties": pages[0].get("properties", {}) if pages else data,
            }
        except Exception as e:
            results[db_name] = {"error": str(e)}
    return jsonify(results)


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
    print("  J.A.R.V.I.S. — Streaming + Browser TTS + Notion")
    print("  Open http://localhost:5000 in Microsoft Edge")
    print("  (Edge has the best neural voices)")
    if notion:
        print("  📝 Notion connected")
    else:
        print("  📝 Notion not configured (set NOTION_API_KEY in .env)")
    print("=" * 56)
    print()
    app.run(debug=True, port=5000)