# J.A.R.V.I.S. — Just A Rather Very Intelligent System

A personal AI assistant that controls my smart home, tracks my routines, and coaches me toward my goals. Built from scratch as a hands-on AI engineering project.

> **Status:** Phase 1 — Voice + Home Control (Bedroom) + Phase 3 Notion Integration
>
> Working web demo with Claude-powered intelligence, browser-native voice I/O, Notion time tracking integration, device state tracking, futuristic UI with screensaver and 10 visual effects. Hardware deployment coming next.

---

## Demo

Run the Jarvis web interface locally — no hardware needed:

```bash
pip install anthropic flask python-dotenv notion-client requests
```

```powershell
# Set your API key (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
$env:NOTION_API_KEY = "ntn_your-key-here"  # Optional — for Notion integration

# Run it
python demo/jarvis_web.py
```

Open http://localhost:5000 in **Microsoft Edge** (best voice quality) and start talking to Jarvis. Try:
- "Turn on the bedroom lights"
- "Dim the lamp to 50%"
- "What's the status of everything?"
- "How productive was I today?" *(requires Notion integration)*

Or click the **mic button** and speak — Jarvis listens, processes, and speaks back.

### Debug Endpoints
- `/health` — service status
- `/devices` — current device states
- `/notion` — Notion connection status and data preview
- `/notion/raw` — raw API response for debugging

---

## Features

### Voice & Intelligence
- **Claude API (Sonnet)** — conversational AI with Jarvis personality
- **Browser-native TTS** — instant speech via Web Speech API (no server-side audio generation)
- **Speech input** — click the mic, speak, auto-sends when you stop talking
- **Device control** — structured `[ACTION: {...}]` parsing for smart home commands
- **Device state tracking** — Jarvis knows what's on/off and gives accurate status reports
- **Notion integration** — reads Work Table and Start Table for productivity/time tracking queries
- **Interaction logging** — every command logged to `logs/interactions.jsonl`

### UI
- **Dark futuristic theme** — JetBrains Mono + Inter fonts, #0a0a0f background
- **6-color theme picker** — purple (default), blue, cyan, amber, red, green
- **Screensaver** — JARVIS logo inside orbital ring, clock, 10 toggleable effects
- **Custom cursor** — gray dot core with accent-colored trailing ring (screensaver only)
- **Styled scrollbars** — thin 4px translucent bars matching the aesthetic

### Screensaver Effects
Particles, grid, pulse, orbits, warp, neural, tendrils, starfield, circuits, helix — all in accent color, toggled independently via bottom-left controls.

---

## How It Works

```
You (voice/text)  →  Flask Orchestrator  →  Claude API  →  Orchestrator  →  You
                      (injects system        (thinks,       (parses out
                       prompt + device        responds)      actions +
                       state + Notion)                       clean text)
                                                                │
                                                    ┌───────────┴───────────┐
                                                    │                       │
                                              Action JSON              Spoken text
                                              → Device state           → Browser TTS
                                              → (future: Home          → Instant audio
                                                 Assistant)
```

1. Your message (typed or spoken) goes to the **Flask orchestrator**
2. The orchestrator injects the **system prompt** with Jarvis's personality, device states, and Notion data
3. **Claude** responds with natural text + structured `[ACTION: {...}]` blocks
4. The orchestrator **parses** the response — actions update device state, clean text goes to the browser
5. Browser **speaks** the response instantly via Web Speech API — zero latency TTS

---

## Tech Stack

| Layer | Technology | Status |
|-------|-----------|--------|
| LLM | Claude API (Sonnet) | ✅ Working |
| Orchestrator | Python, Flask | ✅ Working |
| TTS | Browser Web Speech API (Edge recommended) | ✅ Working |
| STT | Browser SpeechRecognition API | ✅ Working |
| Data | Notion API (time tracking) | ✅ Working |
| Home Control | Home Assistant + Zigbee | ⬜ Needs hardware |
| STT (production) | Whisper | ⬜ Needs hardware |
| TTS (production) | TBD (Kokoro-82M / ElevenLabs) | ⬜ Evaluating |
| Wake Word | OpenWakeWord — "Hey Jarvis" | ⬜ Needs hardware |

---

## Architecture Decisions

| ADR | Decision |
|-----|----------|
| [001](docs/adr/001-mini-pc-over-raspberry-pi.md) | Mini PC over Raspberry Pi for the server |
| [002](docs/adr/002-browser-tts-over-server-tts.md) | Browser-native TTS over server-side edge-tts |

---

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full plan.

| Phase | What | Status |
|-------|------|--------|
| **1 — Voice + Home Control** | Voice pipeline, device control, orchestrator | 🟡 In progress |
| **3 — Notion Integration** | Read time tracking databases, query via voice | 🟢 Working |
| 2 — Identity + Vision | Face recognition, activity detection, speaker ID | ⬜ Planned |
| 3 — Calendar + Intelligence | Google Calendar, habits, web search | ⬜ Planned |
| 4 — Data Warehouse | Sleep, screen time, location, spending tracking | ⬜ Planned |
| 5 — Coaching | Daily check-ins, pattern recognition, RAG | ⬜ Planned |
| 6 — Companion App | Phone app, wall tablet dashboard | ⬜ Planned |

---

## Repo Structure

```
jarvis/
├── demo/
│   ├── jarvis_demo.py    # Terminal interface (v1)
│   └── jarvis_web.py     # Web UI (v2 — browser TTS, mic, screensaver, Notion)
├── docs/
│   ├── ROADMAP.md
│   └── adr/
│       ├── 001-mini-pc-over-raspberry-pi.md
│       └── 002-browser-tts-over-server-tts.md
├── .env.example
├── .gitignore
└── README.md
```

### Branches
- `main` — stable demo
- `browser-tts` — browser-native TTS, mic input, screensaver, theme system
- `notion-integration` — Notion API for time tracking data

---

## Version History

| Version | What changed |
|---------|-------------|
| v0.1.0 | Terminal demo with Claude + device state tracking |
| v0.2.0 | Web UI, streaming, server-side edge-tts |
| v0.3.0 | Browser-native TTS/STT, screensaver, themes, Notion integration |