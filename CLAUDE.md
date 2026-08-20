# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
# Install dependencies
pip install anthropic flask python-dotenv

# Set API keys (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Web UI (primary interface)
python demo/jarvis_web.py
# Open http://localhost:5000 in Microsoft Edge (best voice quality)

# Terminal interface (simpler, no Flask)
python demo/jarvis_demo.py
```

Optional extras for the intent classifier — without them the spaCy and embedding
layers disable themselves at import and only the keyword layer runs:

```bash
pip install spacy sentence-transformers numpy
python -m spacy download en_core_web_sm
```

Venv is at `.venv/`. `constraints-venv.txt` pins the versions the face pipeline
needs to coexist (numpy 2.5.1, TF 2.21, protobuf 7.35, mediapipe 0.10.35,
insightface 1.0.1, opencv 4.13).

## Tests and linting

```bash
python -m pytest tests/    # 265 tests, ~2.5 min
./run_lint.sh              # ruff + ESLint
```

Configs: `ruff.toml`, `eslint.config.mjs`. `tests/test_pipeline_config.py` is
load-bearing — it re-reads `results/benchmarks_detection.json` and fails if a
locked operating point drifts away from the evidence that justifies it. Changing
a threshold without re-measuring breaks the build on purpose.

## HTTP endpoints

From `demo/jarvis_web.py`:

- `GET /` — the assistant UI
- `POST /chat` — classify, then answer locally or via Claude
- `POST /greeting` — greeting text for the browser to speak
- `POST /filler` — tier-aware filler phrase, spoken while Tier 3 runs
- `GET /health` — service status
- `GET /devices` — current in-memory device states
- `POST /reset` — clear conversation history and reset device states

From the `/engineering` blueprint (`demo/engineering.py`): `/engineering`,
`/detection`, `/recognition`, `/enrollment`, `/routing`, `/architecture`,
`/figure/<name>`, plus 301 redirects from `/decisions` and `/direction`.

There are no other routes. In particular there is no `/notion` or `/notion/raw`.

## Architecture

Four modules, not one file:

| File | Role |
|------|------|
| `demo/jarvis_web.py` | Flask orchestrator — routes, device state, Claude call, action parsing |
| `demo/intent_classifier.py` | The Tier 1-2 cascade + filler phrases. Runs *before* Claude |
| `demo/engineering.py` | `/engineering` blueprint — self-contained, registered defensively |
| `demo/engineering_data.py` | Reshapes `results/*.json` into what those templates render |

The request path through `POST /chat`:

1. **User input** (typed or via browser SpeechRecognition API) hits `POST /chat`
2. **The intent classifier runs first** — `classify(user_message, device_states)`.
   Tier 1 is time/date, Tier 2 is device commands. Anything under Tier 3 returns
   immediately with its own response and actions; **Claude is never called**.
   This is the single most important thing to know about the flow, and it is
   easy to miss when reading `jarvis_web.py` alone.
3. **Tier 3 only** — `build_system_prompt()` returns a *list of two content
   blocks*: a static block (personality, rules, action format) marked
   `cache_control: {"type": "ephemeral"}` for prompt caching, and a dynamic block
   holding the live `device_states` text. Only the second changes per request.
4. **Claude** (`claude-sonnet-4-20250514`, `max_tokens=300`, streamed internally
   and accumulated server-side) responds with natural language + optional
   `[ACTION: {...}]` blocks inline. History is capped at the last 20 messages.
5. **Orchestrator parses** the response: regex extracts action JSON → updates
   `device_states` in memory → strips action blocks from the text returned
6. **Browser speaks** the clean text via Web Speech API — no server-side audio

The classifier's own parsing pipeline is four layers, each a fallback for the one
above: normalize filler prefixes → keyword match → spaCy dependency parse →
embedding cosine similarity. spaCy and sentence-transformers are optional; the
module sets `SPACY_AVAILABLE` / `EMBEDDINGS_AVAILABLE` at import and degrades.

### Action format

Claude emits structured actions inline:
```
[ACTION: {"service": "light.turn_on", "entity_id": "light.bedroom", "data": {"brightness": 128}}]
```
Supported services: `light.turn_on/off`, `switch.turn_on/off`, `cover.open_cover/close_cover`

Device state is **in-memory only** — resets on server restart. It simulates what Home Assistant will provide once hardware is deployed.

### Key design constraint

Claude responses are always spoken aloud. The system prompt enforces 1-3 sentences max. Any new prompt changes or features should respect this.

### Notion integration (planned — not in this codebase)

There is no Notion code in `demo/`. A prototype that queried a Work Table and a
Start Table and injected the result into the system prompt lives on the unmerged
`notion-integration` branch; it was never merged and is not wired into
`jarvis_web.py`. Treat Notion as a planned Phase 3 feature.

### Interaction logging

Every turn is appended to `logs/interactions.jsonl`:
`{timestamp, command, response, actions, intent, tier, latency_ms}`. `intent` is
derived, not classified — it is `"home_control"` when actions were produced and
`"conversation"` otherwise.

## Face pipeline (Phase 2 groundwork, `tools/`)

Benchmarked and locked; **the live presence service does not exist yet**. Do not
describe this as deployed.

- **Detection** — YOLOv8n-face, confidence ≥ **0.57**, in `tools/pipeline_config.py`.
  Chosen between the F1-max thresholds of the two hard datasets (MAFA 0.5812,
  WIDER FACE 0.5727); FDDB's 0.694 excluded as the easy set. YOLO won AP on all
  three (0.8644 / 0.7788 / 0.9488). MediaPipe was eliminated (WIDER FACE AP 0.0719).
- **Recognition** — ArcFace + multi-reference, cosine ≥ **0.342**. This lives in
  `tools/build_gallery.py` (`DEFAULT_THRESHOLD`) and travels inside
  `data/gallery.npz`, **not** in `pipeline_config.py` — a cosine threshold is
  only meaningful beside the encoder that produced the vectors. The shipped
  gallery holds 133 512-d vectors; the benchmark scored 6 people vs 500
  strangers (F1 0.9844, rank-1 0.9846).

`results/*.json` is committed and is what the engineering log reads. The datasets
under `data/` and the model weights are not committed.

## Engineering log

`/engineering` is a build-log blueprint that shares nothing with the assistant —
no Claude calls, no device state, no API key. `jarvis_web.py` registers it inside
a `try/except` so a failure there can never stop the assistant booting.

`tools/freeze_engineering.py` renders it to a static site (`--out`, `--strict`,
`--clean`; default `_site/`). It deliberately does **not** import `jarvis_web`,
and asserts afterwards that neither `jarvis_web` nor `anthropic` reached
`sys.modules`. Keep it that way. `tools/scan_public_output.py --site _site` scans
the build for secrets before it goes public. The frozen log is published at
jarvis-mahedor.onrender.com; no deploy config lives in this repo.

## Eval corpus

`tests/eval/routing_eval.jsonl` holds **156 cases** across 24 route types, plus 5
compound multi-route cases. `conversation_eval.jsonl` holds 25. The files open
with `#` comment lines that the loader skips — count records, not lines, or you
will get 159. Scenario suites live in `tests/eval/scenarios/`. Route names there
(including `notion_api`) are *planned* targets, which is correct and intended;
the runner and per-suite results writer are not built yet.

## Roadmap context

See `docs/ROADMAP.md`.

**Real today:** browser TTS/STT, voice mode, the local intent cascade, device
state tracking, the screensaver UI, prompt caching, the linter, the 265-test
suite, the routing eval corpus, the intent and prompt-caching benchmarks, the
face detection/recognition benchmarks with locked operating points, and the
`/engineering` log plus its static freezer.

**Planned, not built:** Notion (Phase 3), the presence service that would consume
the face pipeline, the eval runner, and everything needing hardware — Home
Assistant, Zigbee, Whisper STT, wake word, the mic array.

When updating docs, verify against code rather than against other docs. This file
and the README both drifted badly once by copying each other's claims.
