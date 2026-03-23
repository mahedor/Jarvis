# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
# Install dependencies
pip install anthropic flask python-dotenv

# Set API keys (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:NOTION_API_KEY = "ntn_..."   # optional

# Web UI (primary interface)
python demo/jarvis_web.py
# Open http://localhost:5000 in Microsoft Edge (best voice quality)

# Terminal interface (simpler, no Flask)
python demo/jarvis_demo.py
```

No tests or linter configured. Venv is at `.venv/`.

## Debug endpoints (web UI)

- `GET /health` — service status
- `GET /devices` — current in-memory device states
- `POST /reset` — clear conversation history and reset device states

## Architecture

Everything lives in `demo/jarvis_web.py` — a single Flask orchestrator:

1. **User input** (typed or via browser SpeechRecognition API) hits `POST /chat`
2. **System prompt is rebuilt per request** — `build_system_prompt()` injects the live `device_states` dict so Claude always sees current device status
3. **Claude** (Sonnet, streamed internally) responds with natural language + optional `[ACTION: {...}]` blocks inline
4. **Orchestrator parses** the response: regex extracts action JSON → updates `device_states` in memory → strips action blocks from text returned to the browser
5. **Browser speaks** the clean text via Web Speech API — no server-side audio generation

### Action format

Claude emits structured actions inline:
```
[ACTION: {"service": "light.turn_on", "entity_id": "light.bedroom", "data": {"brightness": 128}}]
```
Supported services: `light.turn_on/off`, `switch.turn_on/off`, `cover.open_cover/close_cover`

Device state is **in-memory only** — resets on server restart. It simulates what Home Assistant will provide once hardware is deployed.

### Key design constraint

Claude responses are always spoken aloud. The system prompt enforces 1-3 sentences max. Any new prompt changes or features should respect this.

### Notion integration

Notion data (Work Table — 30-min focus blocks with productivity ratings; Start Table — session starts) is fetched and injected into the system prompt before each Claude call.

### Interaction logging

Every turn is appended to `logs/interactions.jsonl`: `{timestamp, command, response, actions, intent}`.

## Roadmap context

See `docs/ROADMAP.md`. Phase 1 web demo is complete (browser TTS/STT, streaming, device state tracking, screensaver UI). Hardware deployment (Home Assistant, Zigbee, Whisper STT, wake word) is the next Phase 1 step. Notion integration (Phase 3) is working. Phases 2 and 4-6 are planned.
