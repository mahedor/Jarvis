"""
Jarvis Phase 1 — Demo v2 (Streaming + Browser TTS)
=====================================================
Claude's response streams to the browser in real-time.
Browser speaks each sentence INSTANTLY via Web Speech API.
No server-side TTS, no file generation, no latency.

Open in Microsoft Edge for the best neural voices.

Setup:
  pip install anthropic flask python-dotenv
  python jarvis_web.py
  Open http://localhost:5000 in Microsoft Edge
"""

import os
import json
import re
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

try:
    from anthropic import Anthropic
    from flask import Flask, request, jsonify, render_template_string
except ImportError:
    print("\n❌ Missing packages. Run:")
    print("   pip install anthropic flask python-dotenv")
    print("   Then try again.\n")
    exit(1)

# ─── CONFIG ────────────────────────────────────────────────────
API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
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
    """Build the system prompt with LIVE device states injected."""
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
</style>
</head>
<body>

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

// Greet on page load
fetch('/greeting', { method: 'POST' }).then(r => r.json()).then(d => speakText(d.text));
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
    return jsonify({"status": "online", "service": "jarvis_orchestrator", "version": "0.2.0"})


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
    print("  J.A.R.V.I.S. — Streaming + Browser TTS")
    print("  Open http://localhost:5000 in Microsoft Edge")
    print("  (Edge has the best neural voices)")
    print("=" * 56)
    print()
    app.run(debug=True, port=5000)