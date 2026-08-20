# J.A.R.V.I.S. — Just A Rather Very Intelligent System

A personal AI assistant that controls my smart home, tracks my routines, and coaches me toward my goals. Built from scratch as a hands-on AI engineering project.

> **Status:** Phase 1 — Voice + Home Control (Bedroom)
>
> Working web demo with Claude-powered intelligence, browser-native voice I/O, a
> local intent cascade that answers most commands without calling Claude, and
> device state tracking. The face pipeline for Phase 2 has been **benchmarked and
> its operating points locked**, but the live presence service is not built yet.
> Hardware deployment is the next step.

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

### Optional: the full intent cascade

The classifier runs without these — the keyword layer works alone and the other
two disable themselves at import — but its spaCy and embedding layers stay dark
until you install them:

```bash
pip install spacy sentence-transformers numpy
python -m spacy download en_core_web_sm
```

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

### HTTP endpoints

Everything the assistant serves, from `demo/jarvis_web.py`:

| Method | Route | What it does |
|--------|-------|--------------|
| GET | `/` | The Jarvis UI |
| POST | `/chat` | The one that matters — classify, then answer locally or via Claude |
| POST | `/greeting` | Greeting text for the browser to speak |
| POST | `/filler` | A tier-aware filler phrase ("one sec…") to speak while Tier 3 runs |
| GET | `/health` | Service status |
| GET | `/devices` | Current in-memory device states |
| POST | `/reset` | Clear conversation history and reset device states |

The `/engineering` blueprint adds its own routes — see below. There are no other
endpoints; anything else you may have read about in older docs does not exist.

---

## Engineering log

`/engineering` is a "how this was built" record of the face pipeline — the
decisions and the reasoning first, the benchmark tables underneath. It is not
part of the assistant: no Claude calls, no device state, no API key needed to
render it.

| Route | Contents |
|-------|----------|
| `/engineering` | The decisions — why the dev samples were rejected, why max-F1 ships and TAR@FAR only ranks, why MediaPipe was eliminated, plus the known limits |
| `/engineering/detection` | Detector AP / ROC AUC / F1 / latency across WIDER FACE, MAFA and FDDB, with a confidence threshold explorer |
| `/engineering/recognition` | Encoder comparison and a score-distribution threshold explorer |
| `/engineering/enrollment` | How an unlabelled photo bucket became a gallery — clustering, noise, and the curation calls |
| `/engineering/routing` | The intent cascade, the prompt-caching experiment that was measured and rejected, and what the eval corpus defines as correct |
| `/engineering/architecture` | The system all of this is being built toward |
| `/engineering/figure/<name>` | Serves one generated benchmark figure out of `results/` |

`/engineering/decisions` and `/engineering/direction` are 301 redirects to the
index and architecture pages, kept so older links still resolve.

Every number is read from `results/*.json` when the page loads, so rerunning a
benchmark updates the pages on the next refresh — nothing is hand-transcribed
and there is no build step.

### Publishing it as a static site

`tools/freeze_engineering.py` renders the blueprint to flat HTML that any static
host can serve — no Python, no Flask, no API key at the other end. It builds its
own bare Flask app rather than importing `jarvis_web`, and asserts afterwards
that neither `jarvis_web` nor `anthropic` reached `sys.modules`, so the log
cannot quietly grow a dependency on the assistant.

```bash
python tools/freeze_engineering.py                        # -> _site/
python tools/freeze_engineering.py --out dist --strict --clean
```

| Flag | What it does |
|------|--------------|
| `--out OUT` | Directory to write into (default `_site`) |
| `--strict` | Fail the build on a bad response, a missing `results/` artifact, or data that would render blank |
| `--clean` | Remove the output directory before writing |

Use `--strict` for anything you actually publish: the pages degrade to empty
states by design when an artifact is missing, and a silently blank benchmark
table is a worse published artifact than a failed build.

Before publishing, scan the output for anything that should not go public:

```bash
python tools/scan_public_output.py --site _site
```

The frozen log is deployed to <https://jarvis-mahedor.onrender.com>. The deploy
itself is configured in the host's dashboard — there is no deploy config
committed to this repo, so treat the build commands above as the source of
truth and the hosting as external setup.

---

## Face pipeline (Phase 2 groundwork)

The detectors and encoders have been benchmarked and the operating points are
locked. **The live presence service that would consume them is not built** — what
exists today is the measurement work, the gallery builder, and the pinned config.

### Locked operating points

| Stage | Shipped | Value | Lives in |
|-------|---------|-------|----------|
| Detection | YOLOv8n-face | confidence ≥ **0.57** | `tools/pipeline_config.py` |
| Recognition | ArcFace, multi-reference | cosine ≥ **0.342** | `tools/build_gallery.py` (`DEFAULT_THRESHOLD`), and stored inside `data/gallery.npz` |

The two thresholds live apart on purpose. A detection confidence means the same
thing on any frame, so it belongs to the pipeline. A cosine threshold is only
meaningful next to the encoder that produced the vectors it compares against, so
it travels inside the gallery artifact — ship a gallery, ship its threshold.
`tools/pipeline_config.py` explains the split in full.

Neither number is a magic constant. Each is pinned to the benchmark run that
produced it by timestamp, and `tests/test_pipeline_config.py` re-reads
`results/benchmarks_detection.json` to fail the build if a shipped value drifts
outside the evidence supporting it.

**Detection — 0.57.** It sits between the F1-max thresholds measured
independently on the two hard datasets, MAFA (0.5812) and WIDER FACE val
(0.5727), which agree to within 0.009 despite testing different failure modes.
FDDB's 0.694 is deliberately excluded as the easy set. YOLOv8n-face won AP on all
three: 0.8644 / 0.7788 / 0.9488 on WIDER FACE / MAFA / FDDB, against RetinaFace's
0.8202 / 0.6997 / 0.9338. MediaPipe was eliminated — AP 0.0719 on WIDER FACE.

**Recognition — 0.342.** ArcFace with multi-reference aggregation scored F1
0.9844 and rank-1 0.9846 over 6 enrolled people against 500 strangers, beating
FaceNet512 (F1 0.9365) and VGG-Face (F1 0.7350). The shipped gallery holds 133
512-dimensional vectors. Cosine thresholds do not transfer between encoders, so
`build_gallery.py` warns when the default is carried onto a pair it was never
measured on.

### Benchmark tooling

All of it is in `tools/`, and none of it is needed to run the web demo:

| Script | What it does |
|--------|--------------|
| `benchmark_detection.py` | Detector benchmark — AP, ROC AUC, F1, latency across WIDER FACE, MAFA, FDDB |
| `benchmark_recognition.py` | Encoder benchmark — open-set identification, TAR@FAR and max-F1 |
| `benchmark.py` | Intent classifier timing, per-tier aggregates |
| `benchmark_claude.py` | Claude prompt-caching latency, cache on vs off |
| `collect_faces.py` | Photos → clustered, unnamed reference crops |
| `build_gallery.py` | Reference crops → `data/gallery.npz` |
| `face_utils.py` | Detector and encoder backends behind one interface each |
| `detection_metrics.py`, `recognition_metrics.py` | Pure scoring logic, unit-tested |
| `plot_recognition_curves.py` | Generates the ROC / PR figures in `results/` |
| `sanity_check_detectors.py`, `rerun_detection.sh` | Spot-check and re-run helpers |
| `freeze_engineering.py`, `scan_public_output.py` | Freeze the log; scan it before publishing |

Results land in `results/*.json`, which is committed — it is what the engineering
log reads. The datasets themselves and `data/gallery.npz` are not committed.

---

## Development

```bash
python -m pytest tests/      # 265 tests
./run_lint.sh                # ruff + ESLint (configs: ruff.toml, eslint.config.mjs)
```

---

## Features

### Voice & Intelligence
- **Local intent cascade** — most commands never reach Claude. Tier 1 (time/date) and Tier 2 (device commands) are answered locally; only Tier 3 calls the API
- **Four-layer parsing** — filler-prefix normalization → keyword match → spaCy dependency parse → embedding similarity, each a fallback for the one above
- **Claude API (Sonnet)** — Tier 3 only, for conversational and ambiguous input
- **Prompt caching** — the static half of the system prompt is marked `cache_control: ephemeral`; the device-status half is rebuilt per request
- **Browser-native TTS** — instant speech via Web Speech API (no server-side audio generation)
- **Speech input** — click the mic, speak, auto-sends when you stop talking
- **Voice mode** — waveform UI for voice-only interaction, separate from chat mode
- **Tier-aware filler phrases** — spoken while a Tier 3 request is in flight
- **Device control** — structured `[ACTION: {...}]` parsing for smart home commands
- **Device state tracking** — Jarvis knows what's on/off and gives accurate status reports
- **Interaction logging** — every turn appended to `logs/interactions.jsonl` with tier and latency

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
You (voice/text)
      │
      ▼
Flask orchestrator  ──►  Intent classifier
                              │
              ┌───────────────┴───────────────┐
              │                               │
        Tier 1 / Tier 2                    Tier 3
        answered locally               Claude API call
        (no API call)                  (system prompt +
              │                         device state)
              └───────────────┬───────────────┘
                              ▼
                     Parse [ACTION: {...}]
                              │
                  ┌───────────┴───────────┐
                  │                       │
            Action JSON              Spoken text
            → Device state           → Browser TTS
            → (future: Home          → Instant audio
               Assistant)
```

1. Your message (typed or spoken) goes to the **Flask orchestrator**
2. The **intent classifier** runs first. Tier 1 (time/date) and Tier 2 (device commands) are answered locally and return immediately — Claude is never called
3. Only **Tier 3** reaches Claude, with a system prompt carrying Jarvis's personality and the live device states
4. Either path is **parsed** for `[ACTION: {...}]` blocks — actions update device state, clean text goes to the browser
5. Browser **speaks** the response instantly via Web Speech API — zero latency TTS

---

## Tech Stack

| Layer | Technology | Status |
|-------|-----------|--------|
| LLM | Claude API (Sonnet) | ✅ Working |
| Orchestrator | Python, Flask | ✅ Working |
| Intent routing | Rules + spaCy + sentence-transformers | ✅ Working |
| TTS | Browser Web Speech API (Edge recommended) | ✅ Working |
| STT | Browser SpeechRecognition API | ✅ Working |
| Face detection | YOLOv8n-face @ 0.57 | 🟡 Benchmarked + locked, not deployed |
| Face recognition | ArcFace, multi-reference @ 0.342 | 🟡 Benchmarked + locked, not deployed |
| Presence service | Consumes the above | ⬜ Planned |
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
| **1 — Voice + Home Control** | Voice pipeline, device control, orchestrator | 🟡 Demo done, needs hardware |
| 2 — Identity + Vision | Face recognition, activity detection, speaker ID | 🟡 Pipeline benchmarked + locked; presence service not built |
| 3 — Calendar + Intelligence | Google Calendar, habits, Notion, web search | ⬜ Planned |
| 4 — Data Warehouse | Sleep, screen time, location, spending tracking | ⬜ Planned |
| 5 — Coaching | Daily check-ins, pattern recognition, RAG | ⬜ Planned |
| 6 — Companion App | Phone app, wall tablet dashboard | ⬜ Planned |

---

## Repo Structure

```
jarvis/
├── demo/
│   ├── jarvis_web.py          # Web UI + orchestrator (the primary interface)
│   ├── jarvis_demo.py         # Terminal interface (v1, no Flask)
│   ├── intent_classifier.py   # Tier 1-2 routing + filler phrases
│   ├── engineering.py         # /engineering blueprint (the build log)
│   ├── engineering_data.py    # Reshapes results/*.json for those pages
│   ├── static/
│   │   ├── css/               # styles.css (assistant), engineering.css (log)
│   │   └── js/                # chat, tts, voice-engine, screensaver, theme,
│   │                          #   state + 3 explorer widgets for the log
│   └── templates/
│       ├── index.html         # The assistant
│       └── engineering/       # base + 6 log pages
├── tools/                     # Face pipeline + benchmarks (not needed for the demo)
│   ├── face_utils.py          # Detector + encoder backends, one interface each
│   ├── collect_faces.py       # Photos -> clustered, unnamed reference crops
│   ├── build_gallery.py       # Reference crops -> data/gallery.npz (+ threshold)
│   ├── pipeline_config.py     # Locked detection operating point + provenance
│   ├── benchmark_detection.py # Detector benchmark (WIDER FACE / MAFA / FDDB)
│   ├── benchmark_recognition.py # Encoder benchmark (open-set identification)
│   ├── benchmark.py           # Intent classifier timing
│   ├── benchmark_claude.py    # Claude prompt-caching latency
│   ├── detection_metrics.py   # Pure, unit-tested scoring logic
│   ├── recognition_metrics.py #   "
│   ├── plot_recognition_curves.py  # ROC / PR figures -> results/
│   ├── sanity_check_detectors.py   # Spot-check helper
│   ├── rerun_detection.sh     # Re-run helper
│   ├── freeze_engineering.py  # /engineering -> static site
│   └── scan_public_output.py  # Leak scan before publishing
├── results/                   # Benchmark artifacts, committed — the log reads these
│   ├── benchmarks_detection.json, benchmarks_recognition.json
│   ├── benchmarks_intent.json, benchmarks_caching.json
│   └── recognition_pr.svg, recognition_roc.svg
├── tests/                     # 265 tests
│   └── eval/                  # Routing + conversation eval corpus, 4 scenario suites
├── docs/
│   ├── ROADMAP.md
│   └── adr/                   # 001 mini PC, 002 browser TTS
├── .env.example
├── ruff.toml, eslint.config.mjs, run_lint.sh, package.json
├── constraints-venv.txt
├── .gitignore
├── CLAUDE.md
└── README.md
```

The face pipeline in `tools/` has heavier dependencies (OpenCV, InsightFace,
PyTorch, TensorFlow) and is **not** needed to run the web demo — that stays on
`anthropic`, `flask` and `python-dotenv`. `constraints-venv.txt` pins the
versions that pipeline needs to coexist.

Not committed, by design: everything under `data/` (the WIDER FACE / MAFA / FDDB
/ LFW benchmark sets, the enrollment photos, and the built `gallery.npz`), model
weights (`tools/weights/`, `*.pt`), `logs/`, and QA screenshots. Download the
datasets and weights separately. The `.svg` figures in `results/` are committed
because the log serves them; the `.png` equivalents are not.

### Branches

Work happens on numbered issue branches that merge back into `main` — the voice
work, the linter, the eval suite, face detection, face recognition and the
engineering log all landed that way and are merged.

- `main` — stable; everything described above is here
- `notion-integration` — the one **unmerged** branch that matters. It holds a
  Notion time-tracking prototype that was never merged and is not on `main`.
  The docs used to describe it as a shipped feature; it isn't.

---

## Version History

These are informal milestones — the repo carries no git tags.

| Version | What changed |
|---------|-------------|
| v0.1.0 | Terminal demo with Claude + device state tracking |
| v0.2.0 | Web UI, streaming, server-side edge-tts |
| v0.3.0 | Browser-native TTS/STT, screensaver, themes |
| Unreleased | Local intent cascade + voice mode + filler phrases; ruff/ESLint; 265-test suite; routing eval corpus; intent and prompt-caching benchmarks; the face pipeline (detection + recognition benchmarks, gallery builder, locked operating points); the `/engineering` log and its static-site freezer |