# J.A.R.V.I.S. — Just A Rather Very Intelligent System

A personal AI assistant that controls my smart home, tracks my routines, and coaches me toward my goals. Built from scratch as a hands-on AI engineering project.

> **Status:** Phase 1 — Voice + Home Control (Bedroom)
>
> Working web demo with Claude-powered intelligence, browser-native voice I/O, device state tracking, futuristic UI with screensaver and 10 visual effects. Hardware deployment coming next.

---

## Running it locally

No hardware needed. From the **repo root**:

```bash
# 1. Install dependencies
pip install anthropic flask python-dotenv

# 2. Configure your keys (see "Environment" below)
cp .env.example .env     # then edit .env and paste your key in

# 3. Start the server
python demo/jarvis_web.py
```

That's it — the server prints both URLs on startup.

| | URL | What it is |
|---|---|---|
| **Assistant** | http://localhost:5000/ | The Jarvis UI — chat, voice, device control |
| **Engineering log** | http://localhost:5000/engineering | How the face pipeline was designed and measured |

Open in **Microsoft Edge** for the best neural voices. Try "Turn on the bedroom
lights", "Dim the lamp to 50%", "What's the status of everything?", or click the
mic button and speak.

> Run it from the repo root, not from inside `demo/`. The engineering log reads
> `results/*.json` relative to the repo root, and the terminal demo
> (`python demo/jarvis_demo.py`) is the simpler no-Flask alternative.

### Environment

Values are read from a `.env` file in the repo root (loaded automatically via
`python-dotenv`), or from real environment variables — either works, and env
vars win.

| Variable | Required | Default | What it does |
|----------|----------|---------|--------------|
| `ANTHROPIC_API_KEY` | **yes** | — | Claude API key. The server refuses to start without it. |
| `JARVIS_DEBUG` | no | `1` (on) | Flask debug mode. Set to `0` to disable. |
| `JARVIS_PORT` | no | `5000` | Port to serve on. |

Copy `.env.example` to `.env` and fill it in:

```ini
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Or set them per-shell instead:

```powershell
# PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
python demo/jarvis_web.py
```

```bash
# bash / Git Bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
python demo/jarvis_web.py
```

### Debug mode

**Debug mode is already ON by default** — you don't have to enable it. That
gives you auto-reload on save and full tracebacks in the browser.

```bash
python demo/jarvis_web.py                 # debug on (default)
JARVIS_DEBUG=0 python demo/jarvis_web.py  # debug off
```

```powershell
$env:JARVIS_DEBUG = "0"; python demo/jarvis_web.py   # debug off, PowerShell
```

Turn it **off** when timing startup or attaching a debugger — the reloader
executes the module twice, which doubles the model-loading output and is
confusing to read. The startup banner prints `debug=on` / `debug=off` so you
always know which you're in.

### Debug endpoints

- `/health` — service status
- `/devices` — current in-memory device states
- `POST /reset` — clear conversation history and reset device states

---

## Engineering log

`/engineering` is a "how this was built" record of the face pipeline — the
decisions and the reasoning first, the benchmark tables underneath. It is not
part of the assistant: no Claude calls, no device state, no API key needed to
render it.

| Route | Contents |
|-------|----------|
| `/engineering` | The decisions — why the dev samples were rejected, why max-F1 ships and TAR@FAR only ranks, why MediaPipe was eliminated, plus the known limits |
| `/engineering/routing` | The intent cascade, the prompt-caching experiment that was measured and rejected, and what the eval corpus defines as correct |
| `/engineering/detection` | Detector AP / ROC AUC / F1 / latency across WIDER FACE, MAFA and FDDB, with a confidence threshold explorer |
| `/engineering/recognition` | Encoder comparison and a score-distribution threshold explorer |

Every number is read from `results/*.json` when the page loads, so rerunning a
benchmark updates the pages on the next refresh — nothing is hand-transcribed
and there is no build step.

---

## Features

### Voice & Intelligence
- **Claude API (Sonnet)** — conversational AI with Jarvis personality
- **Browser-native TTS** — instant speech via Web Speech API (no server-side audio generation)
- **Speech input** — click the mic, speak, auto-sends when you stop talking
- **Device control** — structured `[ACTION: {...}]` parsing for smart home commands
- **Device state tracking** — Jarvis knows what's on/off and gives accurate status reports
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
                       state)                                clean text)
                                                                │
                                                    ┌───────────┴───────────┐
                                                    │                       │
                                              Action JSON              Spoken text
                                              → Device state           → Browser TTS
                                              → (future: Home          → Instant audio
                                                 Assistant)
```

1. Your message (typed or spoken) goes to the **Flask orchestrator**
2. The orchestrator injects the **system prompt** with Jarvis's personality and device states
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
| Data | Notion API (time tracking) | ⬜ Planned |
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
| 2 — Identity + Vision | Face recognition, activity detection, speaker ID | ⬜ Planned |
| 3 — Calendar + Intelligence | Google Calendar, habits, Notion, web search | ⬜ Planned |
| 4 — Data Warehouse | Sleep, screen time, location, spending tracking | ⬜ Planned |
| 5 — Coaching | Daily check-ins, pattern recognition, RAG | ⬜ Planned |
| 6 — Companion App | Phone app, wall tablet dashboard | ⬜ Planned |

---

## Repo Structure

```
jarvis/
├── demo/
│   ├── jarvis_demo.py         # Terminal interface (v1)
│   ├── jarvis_web.py          # Web UI (v2 — browser TTS, mic, screensaver)
│   ├── engineering.py         # /engineering blueprint (the build log)
│   ├── engineering_data.py    # Reshapes results/*.json for those pages
│   ├── intent_classifier.py   # Local Tier 1-2 intent routing
│   ├── static/                # css/, js/ — no bundler, no build step
│   └── templates/             # index.html + engineering/
├── tools/                     # Face pipeline (not needed to run the demo)
│   ├── face_utils.py          # Detector + encoder backends, one interface each
│   ├── collect_faces.py       # Photos -> clustered, unnamed reference crops
│   ├── benchmark_detection.py # Detector benchmark (WIDER FACE / MAFA / FDDB)
│   ├── benchmark_recognition.py # Encoder benchmark (open-set identification)
│   ├── build_gallery.py       # Reference crops -> data/gallery.npz
│   ├── pipeline_config.py     # Locked operating points + their provenance
│   └── *_metrics.py           # Pure, unit-tested scoring logic
├── results/                   # Benchmark artifacts (committed — the engineering log reads these)
├── tests/
├── docs/
│   ├── ROADMAP.md
│   └── adr/
├── .env.example
├── .gitignore
└── README.md
```

The face pipeline in `tools/` has heavier dependencies (OpenCV, InsightFace,
PyTorch, TensorFlow) and is **not** needed to run the web demo — that stays on
`anthropic`, `flask` and `python-dotenv`.

### Branches
- `main` — stable demo
- `browser-tts` — browser-native TTS, mic input, screensaver, theme system
- `notion-integration` — unmerged prototype of the Notion time-tracking read; not on `main`

---

## Version History

| Version | What changed |
|---------|-------------|
| v0.1.0 | Terminal demo with Claude + device state tracking |
| v0.2.0 | Web UI, streaming, server-side edge-tts |
| v0.3.0 | Browser-native TTS/STT, screensaver, themes |