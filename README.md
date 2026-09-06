# J.A.R.V.I.S. — Just A Rather Very Intelligent System

A personal AI assistant that controls my smart home, tracks my routines, and coaches me toward my goals. Built from scratch as a hands-on AI engineering project.

> **Status:** Phase 1 — Voice + Home Control (Bedroom)
>
> Working web demo with Claude-powered intelligence, browser-native voice I/O, a
> local intent cascade that answers most commands without calling Claude, and
> device state tracking. The face pipeline for Phase 2 has been **benchmarked and
> its operating points locked**, and `tools/presence_service.py` runs that
> pipeline live onto an MQTT bus — though it is standalone, not yet consumed by
> the assistant. Hardware deployment is the next step.

---

## Running it locally

No hardware needed. From the **repo root**:

```bash
# 1. Install dependencies
pip install anthropic flask python-dotenv paho-mqtt

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
| `/engineering/presence` | **Preliminary.** The presence pipeline diagram and topic layout — a design that runs, not a measurement. Debounce values are rendered live from `PresenceConfig` |
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

**Every route declares what it needs.** `@requires(artifacts=..., data_checks=...)`
on each view in `demo/engineering.py` says which `results/` artifacts and which
shaped-data conditions that page depends on, and `preflight()` walks every
registered route to enforce them. A route carrying no declaration fails the
build. This replaced a hand-maintained checklist that only covered the pages
someone remembered to add — which is how `/engineering/enrollment` published an
empty state for months while every build reported success.

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
locked. `tools/presence_service.py` consumes them live and publishes presence
over MQTT (documented below); **it is standalone and unintegrated**, and has only
been run from a video file, never a live camera. What is settled today is the
measurement work, the gallery builder, and the pinned config.

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

## MQTT hello-world (presence groundwork)

Throwaway spike for Phase 2 item 16 (REST -> MQTT message bus), written before
the presence service to pin down the semantics it depends on. **Nothing in
`demo/` uses MQTT** — the assistant's request path is unchanged. The real
consumer is `tools/presence_service.py`, documented in the next section; these
two scripts are kept because they demonstrate each MQTT behaviour in isolation,
which the service cannot.

### Install and start the broker (Windows)

```powershell
winget install --source winget --id EclipseFoundation.Mosquitto
```

Installs to `C:\Program Files\mosquitto\` and registers a Windows service named
`mosquitto` that **starts automatically** — after install there is usually
nothing left to do. To check and control it:

```powershell
sc query mosquitto              # state, no admin needed
Start-Service mosquitto         # admin
Stop-Service mosquitto          # admin
netstat -ano | findstr 1883     # confirm it is listening
```

To watch traffic while debugging, stop the service and run it in the foreground
with verbose logging instead (the service already owns port 1883, so it has to
be stopped first):

```powershell
Stop-Service mosquitto
& "C:\Program Files\mosquitto\mosquitto.exe" -v
```

The shipped `mosquitto.conf` is **entirely comments**, so mosquitto 2.x falls
back to its defaults: listen on `127.0.0.1:1883` only, anonymous access allowed.
That is what makes the hello-world work with zero configuration — and it is also
why a camera node on another machine will not be able to connect. Publishing
presence from a Pi later means adding a real `listener` plus auth; treat this
setup as loopback-only.

### The two scripts

`tools/mqtt_hello_pub.py` and `tools/mqtt_hello_sub.py`, both on topic
`jarvis/test/hello`. Throwaway diagnostics, not imported by anything. They use
paho-mqtt 2.x (`CallbackAPIVersion.VERSION2`) over MQTT v3.1.1.

They live on `jarvis/test/` rather than under `jarvis/presence/` on purpose:
wildcards do not honour naming conventions, so a spike parked at
`jarvis/presence/test` would be delivered to any subscriber on
`jarvis/presence/+` as though it were a person named "test".

**QoS 0 vs QoS 1 — does a message survive the subscriber being offline?**

```bash
python tools/mqtt_hello_sub.py --scenario qos --phase register   # claim session, disconnect
python tools/mqtt_hello_pub.py --scenario qos                    # send both, nobody listening
python tools/mqtt_hello_sub.py --scenario qos --phase collect    # come back
```

Only the QoS 1 message is waiting. The QoS 0 one was dropped the instant it
arrived at a broker with no connected subscriber. The queueing depends on a
**persistent session** (`clean_session=False` plus a fixed client id) — on
reconnect the broker reports `session_present=True`. With the default clean
session, QoS 1 buys nothing across a disconnect, because the subscription itself
is gone.

**RETAIN — does a late subscriber learn the current state?**

```bash
python tools/mqtt_hello_pub.py --scenario retained    # publish, then exit
python tools/mqtt_hello_sub.py --scenario retained    # start afterwards, still gets it
```

The message arrives immediately on SUBACK with `retain=True` — that flag is how
you tell a stored state snapshot from a live event. The broker holds exactly one
retained message per topic. This is the mechanism for "who is home right now?":
a subscriber that boots at any time gets the answer without waiting for the next
detection. `--scenario clear` (empty payload, `retain=True`) wipes it.

**LAST WILL — does the broker announce a publisher that died?**

```bash
python tools/mqtt_hello_sub.py --scenario will --seconds 12 &   # subscriber first
python tools/mqtt_hello_pub.py --scenario will --die-after 3    # hard-exits, no DISCONNECT
```

The subscriber sees the publisher's `hello`, then the will fires the moment the
socket dies. The will payload is composed at **connect** time and handed to the
broker inside the CONNECT packet, so its `sent_at` is the connect timestamp, not
the death timestamp. `--die-after` uses `os._exit`, which looks identical to a
`kill` from the broker's side; omit it to sit until you kill the process
yourself. A clean `disconnect()` suppresses the will, which is the point — this
is how a presence node will report "camera offline" instead of going silently
stale.

### Cleaning up

The QoS demo leaves a persistent session on the broker that keeps queueing QoS 1
messages for a subscriber that never returns:

```bash
python tools/mqtt_hello_sub.py --scenario wipe    # drop the session
python tools/mqtt_hello_pub.py --scenario clear   # drop the retained message
```

---

## Presence service (`tools/presence_service.py`)

Reads a camera, decides who is in front of it, and publishes presence
transitions onto the MQTT bus. This is the live consumer of the Phase 2 face
pipeline — the first thing in the repo that actually *runs* the locked
detection and recognition operating points rather than benchmarking them.

**Status:** works end to end and has been run against a real broker, but only
from a video file. The live-camera path (`--source 0`) is written and
unexercised. It is not wired into `demo/` and the assistant does not consume it
yet.

```bash
python tools/presence_service.py --stats            # default camera
python tools/presence_service.py --dry-run --source clip.mp4 --stats
python tools/presence_service.py --duration 60      # bounded run
python tools/presence_service.py --preview --dry-run # watch it work
```

Needs the broker above. `--dry-run` runs the full pipeline and prints the
transitions it *would* publish, connecting to nothing — it does not even import
paho, so it works on a machine with no broker installed.

`--preview` opens a window with the detection boxes labelled by matched
identity and score (or `unknown` below threshold), plus the current presence
state and the dropped-frame counter. `q` or ESC quits. It is off by default and
costs nothing when off — no annotations are collected. When on it costs worker
time, so it lowers throughput and raises the drop count; `--stats` prints a
warning saying exactly that, because preview numbers are not headless numbers.
A headless OpenCV build downgrades to running without the window rather than
failing.

### Topic layout

| Topic | Retained | Payload |
|---|---|---|
| `jarvis/presence/<name>` | yes | `{state, name, confidence, timestamp, last_seen}` |
| `jarvis/status/presence_service` | yes | `online` / `offline` |

`jarvis/presence/+` means **people, and nothing else** — that is the whole
reason the service's own liveness lives on `jarvis/status/` instead of at
`jarvis/presence/_service`. A broker has no notion of a leading underscore
meaning "internal": a wildcard subscriber would receive the service's status as
though it were another person. Same reason the hello-world spike was moved to
`jarvis/test/hello`.

The status topic is not optional. Without it, a retained `absent` is ambiguous —
it means either "nobody is there" or "this service died and what you are reading
is a fossil". `offline` is registered as the MQTT **last will**, so the broker
publishes it if the process is killed; a clean shutdown suppresses the will, so
the service also publishes `offline` itself before disconnecting.

Everything is retained, so a subscriber that starts at any time immediately
learns the current state without waiting for the next detection.

### What it reads, and from where

Nothing about the pipeline is hardcoded — the service prints the provenance of
every value at startup:

| Value | Source |
|---|---|
| detection confidence `0.57` | `pipeline_config.DETECTION_CONFIDENCE_THRESHOLD` |
| detection min box `20px` | `pipeline_config.DETECTION_THRESHOLD_PROVENANCE` |
| detector + weights | `pipeline_config.DETECTION_DETECTOR` / `_WEIGHTS` |
| recognition threshold `0.342` | `data/gallery.npz` metadata |
| encoder + aggregation | `data/gallery.npz` metadata |

The gallery also *chooses* the encoder, so a cosine threshold can never be
applied to embeddings from a different model than the one that produced it.

### Frame handling

A capture thread reads at the camera's native rate into a **single-slot buffer
that overwrites**. The worker takes whatever is in the slot when it is free;
frames arriving while it is busy are dropped, not queued.

This is deliberately not "process every Nth frame". Processing time varies with
how many faces are in shot and drifts with load, so any fixed N is wrong
somewhere — and a queue would let the service fall progressively further behind
real time while still looking healthy. A presence answer computed from a frame
twelve seconds old is not a late answer, it is a wrong one. Dropping bounds
staleness at one frame no matter how slow the pipeline gets, and the drop count
is reported rather than hidden. Video files are paced to their recorded FPS so
they behave like a camera instead of arriving as one enormous burst.

### State machine

Per identity, independently:

- **absent → present:** `N=3` hits within the last `M=5` processed frames
- **present → absent:** `T=10.0` seconds with no hit

Only transitions are published — somebody who stays put generates no traffic.
A fourth constant, `max_observation_age=5.0`, caps how old an observation may be
and still count toward arrival: `M` counts *frames*, and frames are not a unit
of time, so if throughput collapses the same five frames can span a minute and
the rule quietly degrades into "3 hits some time recently". All four live in
`PresenceConfig` and are deliberately **not** CLI flags — they are a locked
operating point, pinned by `tests/test_presence_service.py`.

### Instrumentation

`--stats` prints counters and per-stage mean/p95 on exit. This service is the
only source of real end-to-end latency numbers in the project. Measured on CPU
at 960x540:

| stage | unit | mean | p95 |
|---|---|---|---|
| capture_decode | per frame | 2.0ms | 2.6ms |
| slot_wait | per frame | 32.0ms | 62.4ms |
| detect | per frame | 109.8ms | 156.0ms |
| crop_align | per face | 0.1ms | 0.1ms |
| embed | per face | 164.5ms | 195.8ms |
| match | per face | 0.3ms | 0.6ms |
| **END-TO-END** | per frame | **165.5ms** | **295.0ms** |

Capture ran at 13.7 fps, the worker kept up with 7.0 fps, and ~45% of frames
were dropped. Detection and embedding are essentially all of the cost; matching
a probe against 133 gallery vectors is free.

These are one run, not a benchmark — they move a lot with machine load. A second
run on a busier machine gave detect 149ms / embed 284ms and only 5.4 fps
processed, with 64% of frames dropped. That spread is the reason the arrival
window has a wall-clock ceiling as well as a frame count.

**The rows do not sum.** Per-frame stages are sampled once per processed frame,
per-face stages once per face found, and most frames contain no face — so
`embed`'s mean is an average over a much smaller population and adding it to a
per-frame mean can exceed the end-to-end figure. The table prints `n` and the
unit per row for exactly this reason.

---

## Development

```bash
python -m pytest tests/      # 340 tests
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
| 2 — Identity + Vision | Face recognition, activity detection, speaker ID | 🟡 Pipeline benchmarked + locked; presence service built but unintegrated |
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
│   ├── mqtt_hello_pub.py      # MQTT spike - publisher (throwaway)
│   ├── mqtt_hello_sub.py      # MQTT spike - subscriber (throwaway)
│   ├── presence_service.py    # Live face presence -> MQTT (Phase 2)
│   ├── freeze_engineering.py  # /engineering -> static site
│   └── scan_public_output.py  # Leak scan before publishing
├── results/                   # Benchmark artifacts, committed — the log reads these
│   ├── enrollment_summary.json  # Anonymised curation summary (no names/paths)
│   ├── benchmarks_detection.json, benchmarks_recognition.json
│   ├── benchmarks_intent.json, benchmarks_caching.json
│   └── recognition_pr.svg, recognition_roc.svg
├── tests/                     # 340 tests
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
| Unreleased | Local intent cascade + voice mode + filler phrases; ruff/ESLint; 340-test suite; routing eval corpus; intent and prompt-caching benchmarks; the face pipeline (detection + recognition benchmarks, gallery builder, locked operating points); the `/engineering` log and its static-site freezer |