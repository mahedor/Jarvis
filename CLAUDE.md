# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
# Install dependencies
pip install anthropic flask python-dotenv paho-mqtt

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
python -m pytest tests/    # 340 tests, ~2.5 min
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
`/detection`, `/recognition`, `/enrollment`, `/routing`, `/presence`,
`/architecture`, `/followme`, `/figure/<name>`, plus 301 redirects from `/decisions` and
`/direction`.

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

Benchmarked and locked. `tools/presence_service.py` now consumes these
operating points live (see the MQTT section below), but it is a standalone tool:
**not deployed, not wired into `demo/`, and verified only from a video file** -
the live-camera path is written and unexercised. Do not describe any of this as
deployed.

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

`results/enrollment_summary.json` is the derived, anonymised half of a curation
run, written by `collect_faces.py` beside the private `manifest.json`. The
manifest holds filenames and Google Photos folder names (i.e. people's names) and
stays in gitignored `data/`; the summary holds aggregates plus `folder A/B/C`
labels and is committed, because the enrollment page is published and used to
render an empty state in production for months while reading the private file.

## MQTT (`tools/mqtt_hello_*.py`, `tools/presence_service.py`)

**Nothing in `demo/` imports either of these** and MQTT is not part of the
assistant's request path. Both need a local mosquitto on 127.0.0.1:1883 —
install/start commands are in the README.

`mqtt_hello_pub.py` / `mqtt_hello_sub.py` are throwaway scripts on topic
`jarvis/test/hello` demonstrating QoS 0 vs QoS 1 across a disconnect, retained
messages, and last-will. Groundwork for the Phase 2 REST -> MQTT migration.

`presence_service.py` is the real one: camera -> YOLO -> ArcFace -> gallery
match -> per-identity state machine -> retained MQTT transitions. It reads both
operating points from the artifacts (detection from `pipeline_config`,
recognition from inside `gallery.npz`) and hardcodes neither. Verified end to
end against a real broker from a video file; **the live-camera path is
unexercised**.

TOPIC BRANCHES. `jarvis/presence/<name>` is people and only people, so that
`jarvis/presence/+` is safe to subscribe to. Service liveness is deliberately on
a different branch at `jarvis/status/presence_service` (retained, `offline`
registered as the last will) — a wildcard would otherwise deliver it as if it
were a person, since the broker has no notion of a name being "internal". The
hello-world spike is on `jarvis/test/hello` for the same reason. Do not put
anything under `jarvis/presence/` that is not a person — per-sensor events
(camera, mmWave) go under `jarvis/sensor/`, which nothing publishes yet. Only
direct children collide. The rule is a convention, not an invariant: `topic_for()`
is the only thing that builds a presence topic, nothing in the string marks it as
a person, and `tests/test_presence_service.py` only catches a non-person topic
added as a module constant.

`PresenceConfig` (N=3 hits, M=5 frames, T=10.0s, max_observation_age=5.0s) lives
in **`tools/presence_config.py`**, not in `presence_service.py`, and the split is
load-bearing: `presence_service` imports cv2/numpy at module level, while `demo/`
is stdlib-only by design, so `/engineering/presence` renders those values live
without the assistant growing a dependency on OpenCV. It is deliberately NOT in
`pipeline_config.py` either — that file's claim is that every number is pinned to
a benchmark run, and these are unvalidated placeholders.
`tests/test_presence_service.py` pins the values and the state-machine behaviour
with an injected clock.

## Follow-me drone (`followme/`)

The live follow-me loop for a Parrot ANAFI 4K: YOLOv8n detector (INT8 OpenVINO) → BoT-SORT (+ camera-motion
compensation, settings tuned 11-27) → lock-on with OSNet re-ID (`lockon.py`) → Olympe commands. `lookhere.py` adds the two-drone "look here / go there"
(share a clicked ground spot over a file or MQTT; built 2026-11-22, not yet flown). Runs in WSL from its own
uv venv (`~/jarvis-follow`), not the repo `.venv`; see `followme/README.md`. **Nothing in `demo/`
imports it.** Flown once (2026-11-15): follow, re-ID re-locks and the two-way spin search ran on the
real drone; map position (`geo.py`) logged numbers but is unchecked against ground truth.

`followme/weights/` (detector + OSNet `.pt`) is gitignored. Flight logs (per-run video + `follow.csv`,
300+ MB each) go to `~/follow_logs/` in the WSL home, outside the repo and OneDrive. `flight_log.py` (stdlib only) rewrites
`~/follow_logs/flight_log.csv` + `flight_log.md` (one row per real flight, newest first, settings from each run's
`drone.json`, stats from `follow.csv`, "not logged" where a run didn't record a field, typed csv notes kept) at the end of
every `live_follow.py` run, inside a try so a failure only warns; `python flight_log.py [--root DIR] [--copy-to DIR]` backfills
(`--copy-to` also copies both files there, e.g. `/mnt/c/Users/mahed/Downloads`). It is the only flight list: the 11-22
quick table (commit 0af1cfb) was folded in and deleted 11-27; its spin-search rule (transitions into a state starting
`SEARCH (spinning`), `boxes.jsonl` flag and folder `path` are columns. `followme/flight_log_annotations.json` (committed,
Michael 11-27) is merged on every regenerate: drone corrections ("droneB (corrected 11-27: ...; program logged: droneA)")
and chat-history notes for what a run did not record (model, cutoffs, tracker, re-ID, fake test vs real flight, who was
followed), always marked "from chat history:" and citing the chat time + seq or a commit; never overwrites a logged
value; a key matching no run folder is ignored with a warning. Michael's WSL clone is
`/mnt/c/Users/mahed/OneDrive/Documents/Jarvis/jarvis-lean` (branch `followme-folder`). The folder's 235 tests are run from inside it
(`python -m pytest -q test_live_follow.py test_lockon.py test_geo.py test_lookhere.py test_drone_settings.py test_cutoffs_evidence.py test_reid_evidence.py test_replay.py test_flight_log.py test_time_trackers.py test_side_by_side.py test_flight_vs_bench.py`) and are **not** part of the
`tests/` suite counted above. `run_lint.sh` does not lint it.

`time_trackers.py` times the 15 tuned tracker setups on the laptop CPU (Michael 11-27: talk speed numbers come only
from his Ryzen 7 7735HS, CPU only): the flight detector (live_follow's `resolve_detection` + `Detector`) runs once per
frame of the newest real flight's clean video and its boxes are cached; each setup is built as `tracker_tuning_all.py`
built it (all six trackers are Ultralytics built-ins, not boxmot; re-ID = `yolo26n-cls.pt` crop encoder on the CPU) with
its `chosen.settings_changed` read from `results/followme_tracker_tuning_all_summary.json` (byte-exact from Drive, sha256
pinned in `test_time_trackers.py`), plus a 16th row `BoT-SORT + CMC (flies)` = `botsort_tuned.yaml` via
`live_follow.resolve_tracker` (match 0.90) after the summary's BoT-SORT + CMC row; only `tracker.update` is timed (GMC and re-ID included), after 20 warm-up frames.
Writes `~/follow_logs/tracker_speed_<date>.json` (+ `--save` copy). No laptop numbers are committed yet.

Detection cutoffs are read from the Colab F1-max sweep's own output, per model by exact name: the INT8 export that
flies (`vd_mav_cl_s42_int8_openvino_model`) from entry `vd_mav_cl_s42_int8` of `results/followme_cutoffs_balanced_int8.json`
(person 0.23, vehicle 0.29; swept on the INT8 export itself 11-26, curves in `results/followme_thresholds_balanced_int8.json`),
a `.pt` from `results/followme_cutoffs_balanced.json` (Oct 28-29 sweep, `vd_mav_cl_s42` 0.24 / 0.28, curves in
`results/followme_thresholds_balanced.json`). An INT8 model never falls back to its `.pt` entry: missing INT8 file or
entry stops the program (the Oct 29 note that INT8 kept the `.pt`'s cutoffs was wrong). `test_cutoffs_evidence.py`
recomputes both from their curves, the way `tests/test_pipeline_config.py` guards the face threshold. The engineering
log does not read these four files. The re-ID re-lock cutoff `lockon.MATCH_SIM` (0.70) is read from
`results/followme_reid_cutoff.json`, a hand-written decision record: Michael chose 0.70 by judgment among measured
options, nothing computes it. Its evidence is committed byte-exact beside it (`followme_reid_bench_v3.json` +
`followme_reid_scores_osnet_x0_25.npz`, the two `followme_reid_replay_1029*.json` replays), and
`test_reid_evidence.py` checks every number in the record against them. At startup `live_follow.py` compares what is
about to run with `followme/flight_decisions.json` (only settings with a recorded decision, now including the tracker
`botsort_tuned.yaml`, the detector precision `INT8`, `--imgsz 640`, and `lost_s` 1.0) and stops on any
difference unless `--off-decisions`; the result is written into each flight's `drone.json`.

The tracker is BoT-SORT (camera-motion compensation on, tracker re-ID off: the winner of
`results/followme_tracker_scores.jsonl` at default settings) run from `followme/botsort_tuned.yaml`: the installed
ultralytics' stock `botsort.yaml` with only five values changed (Michael 11-27: track_high / new_track 0.40,
track_low 0.05, match 0.90, buffer 60). The decision record `results/followme_tracker_settings_choice.json` holds the
values, four measured options and the reason (0.90, not the val-best 0.975: +0.0026 on val, within his seed-noise figure,
and below stock on MOT20); its evidence is the 11-27 Colab sweep, byte-exact (`results/followme_tracker_settings_sweep.jsonl`
+ `_summary.json`). `live_follow.resolve_tracker` stops if the yaml differs from the record; the tests check the yaml
against the record and the installed ultralytics' keys, and every number in the record against the sweep rows. Stock
`botsort.yaml` is a decisions deviation. The detector's precision is read from the model files
(`live_follow.detector_precision`: the OpenVINO export's `metadata.yaml` int8 / half / quantize, dynamic, imgsz,
calibration, cross-checked with the FakeQuantize layers and NNCF parameters in its `.xml`; "not recorded" or "conflict"
otherwise, never guessed), printed at startup (`detector: INT8 OpenVINO, static 384x640`), checked as the
`detector_precision` decision, saved in `drone.json` as `detector`, and shown in the flight log's `precision` column.

Flight vs bench (11-27, for the 11-28 comparison flights): every flight writes `detector.csv` (one row per detector
cycle: box age, cycle / busy / idle ms, model.track ms with Ultralytics' pre / inference / post split, `track_ms` =
tracker.update alone via a pass-through timer on the tracker model.track built, embed ms, boxes); follow.csv gains
`det_n` / `track_ms`, drone.json `tracker.bench_setup` (its row in time_trackers' bench JSON), `tracker_runtime`,
`label`, `comparison`; the flight log gains `comparison` and `timing` columns (cache schema 4). Comparison trackers
`bytetrack_tuned.yaml` and `botsort_reid_cmc_tuned.yaml` (`live_follow.COMPARISON_TRACKERS`) = installed stock yaml +
grid switches + the tuning summary's chosen settings, exactly time_trackers' construction (`resolve_tracker` stops
on a difference, and stops if the re-ID one's `yolo26n-cls.pt` is not on disk); flying one needs `--off-decisions`
and `--label` naming it (`--label` is now a live_follow flag too, replay's `label_problem` rule). TrackTrack was
left out on purpose: model.track adds a loose-NMS pass the bench does not time. `flight_vs_bench.py` (stdlib only)
matches flights to bench rows, pools per config (after 20 warm-up cycles, airborne only when the run was airborne)
and fits flight ms / loop FPS = a + b x bench ms with residuals (2 points exact, 3 rough). Plan + commands: README
"Flight vs bench" and `followme/flight_plan_1128.txt`. Nothing flown with it yet.

`replay.py` (replay a flight or video without flying) has no settings of its own (Michael 11-26: replay always
matches what flies unless he says otherwise): its flags, detector pass, `FollowLogic` and decisions check all come
from `live_follow.py`, so it runs the same check, stops on a mismatch, and with `--off-decisions` prints a DEVIATION
banner and saves it in `replay_info.json`. Locks come only from the flight log's recorded click, `--lock-id` or a
window click (the old nearest-box auto re-lock is deleted). Its unavoidable offline differences (no drone, search
driven by recorded yaw, synchronous detection so `STALE_S` never fires) are `replay.OFFLINE_DEVIATIONS`, printed every
run. `test_replay.py` fails if replay grows a setting live_follow doesn't share. Comparison runs (11-27): `--label
'cutoff 0.79'` stamps a neutral caption on every frame and, with `--off-decisions`, replaces the red per-frame banner only
if it names every deviating value (else STOP); console banners, `replay_info.json` (`label`, `frame_caption`) and
commands.csv keep the deviation. `--lock-id only@18s` locks the only person in view at that time (several people: no
lock, their IDs printed) for pre-11-22 flights with no `boxes.jsonl`. `side_by_side.py` stacks two run folders (aligned
by wall clock: folder-name start, refined by follow.csv mtime minus last t_s when within 2 s), two replay folders (by
commands.csv t_s) or two videos, with label bars; ffmpeg (H.264) if on PATH, else OpenCV mp4v.

## Engineering log

`/engineering` is a build-log blueprint that shares nothing with the assistant —
no Claude calls, no device state, no API key. `jarvis_web.py` registers it inside
a `try/except` so a failure there can never stop the assistant booting.

Every route declares what it needs via `@requires(artifacts=..., data_checks=...)`
in `demo/engineering.py`, and `freeze_engineering.preflight()` walks every
registered route to enforce it. **A route with no declaration is a build
error** — the previous hand-maintained checklist is what let the enrollment page
ship hollow. Add the decorator when you add a page.

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
state tracking, the screensaver UI, prompt caching, the linter, the 340-test
suite, the routing eval corpus, the intent and prompt-caching benchmarks, the
face detection/recognition benchmarks with locked operating points, the
`/engineering` log plus its static freezer, the MQTT spike, and the standalone
presence service (video-file verified; live camera unexercised).

**Planned, not built:** Notion (Phase 3), the eval runner, integrating the
presence service into the assistant, and everything needing hardware — Home
Assistant, Zigbee, Whisper STT, wake word, the mic array.

When updating docs, verify against code rather than against other docs. This file
and the README both drifted badly once by copying each other's claims.
