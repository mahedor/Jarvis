# J.A.R.V.I.S. — Just A Rather Very Intelligent System

A personal AI assistant that controls my smart home, tracks my routines, and coaches me toward my goals. Built from scratch as a hands-on AI engineering project.

> **Status:** Phase 1 — Voice + Home Control (Bedroom)
>
> Currently a working demo with Claude-powered command processing, device state tracking, and action parsing. Hardware deployment coming next.

---

## Demo

Run the Jarvis web interface locally — no hardware needed:

```bash
pip install anthropic flask
```

```powershell
# Set your API key (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"

# Run it
python demo/jarvis_web.py
```

Open http://localhost:5000 and start talking to Jarvis. Try:
- "Turn on the bedroom lights"
- "Dim the lamp to 50%"
- "What's the status of everything?"
- "Turn off the fan and close the blinds"

Jarvis responds in character, parses device control actions from the LLM response, tracks device state across the conversation, and logs every interaction.

---

## How it works

```
You  →  Orchestrator  →  Claude API  →  Orchestrator  →  You
         (adds system      (thinks,       (parses out
          prompt +          responds)      actions +
          device state)                    clean text)
                                              │
                                    ┌─────────┴─────────┐
                                    │                     │
                              Action JSON            Spoken text
                              → Home Assistant        → Speaker
                              → controls devices      → TTS audio
```

1. Your message goes to the **orchestrator** (a Python FastAPI server)
2. The orchestrator injects the **system prompt** (Jarvis's personality, available devices, current device states) and sends everything to **Claude**
3. Claude responds with natural text + structured `[ACTION: {...}]` blocks
4. The orchestrator **parses** the response — separating the spoken text from device actions
5. Actions get sent to **Home Assistant** (when hardware is connected), spoken text goes to **TTS**

The orchestrator is the central nervous system — Claude is the brain, Home Assistant is the hands.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude API (Sonnet) |
| Orchestrator | Python, FastAPI, Flask (demo) |
| Home Control | Home Assistant + Zigbee (Phase 1 hardware) |
| Speech-to-Text | Whisper (Phase 1 hardware) |
| Text-to-Speech | Piper (Phase 1 hardware) |
| Wake Word | OpenWakeWord — "Hey Jarvis" (Phase 1 hardware) |

---

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full plan. High-level:

| Phase | What | Status |
|-------|------|--------|
| **1 — Voice + Home Control** | Voice pipeline, device control, orchestrator | 🟡 In progress |
| 2 — Identity + Vision | Face recognition, activity detection, speaker ID | ⬜ Planned |
| 3 — Calendar + Intelligence | Google Calendar, Notion, habits, web search | ⬜ Planned |
| 4 — Data Warehouse | Sleep, screen time, location, spending tracking | ⬜ Planned |
| 5 — Coaching | Daily check-ins, pattern recognition, RAG | ⬜ Planned |
| 6 — Companion App | Phone app, wall tablet dashboard | ⬜ Planned |

---

## Architecture Decisions

| ADR | Decision |
|-----|----------|
| [001](docs/adr/001-mini-pc-over-raspberry-pi.md) | Mini PC over Raspberry Pi for the server |

More ADRs will be added as the project evolves.

---

## Repo Structure

```
jarvis/
├── demo/                 # Working demo (runs on any PC)
│   ├── jarvis_demo.py    # Terminal interface
│   └── jarvis_web.py     # Web UI interface
├── docs/
│   ├── ROADMAP.md        # Full phase breakdown
│   └── adr/              # Architecture Decision Records
├── .env.example
├── .gitignore
└── README.md
```

The repo grows with the project. No scaffolding for things that haven't been built yet.
