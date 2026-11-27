# followme/ — follow-me drone (Parrot ANAFI 4K)

The live follow-me loop: drone video → YOLOv8n person/vehicle detector (INT8 OpenVINO) → BoT-SORT tracker
(with camera-motion compensation, five settings tuned 11-27) → lock-on with OSNet re-ID → yaw/forward-back commands back to
the drone over Olympe. Laptop = brain, drone = body. Moved here from the laptop's `Downloads`
folder on 2026-11-19 so it lives with the rest of Jarvis.

**Status: flown once (2026-11-15).** Follow, re-ID re-locks and the two-way spin search all ran on
the real drone; map position logged numbers but is not yet checked against reality. Nothing in
`demo/` imports this folder.

## Files

| File | What it does |
|---|---|
| `live_follow.py` | The program you fly with: video, detector thread, keys, follow logic, logging |
| `lockon.py` | Lock-on: pick one person, keep a re-ID gallery (OSNet), re-lock only on a confident match |
| `replay.py` | Replay a recorded flight or any video through the flight program without flying (same settings, same decisions check); also the shared controller, target drawing and class names |
| `record.py` | Olympe connection, telemetry (incl. GPS), and the fake drone used by tests |
| `geo.py` | Map position: target box → ray → flat ground → meters north/east of takeoff |
| `drone_settings.py` | Same max tilt / rotation / climb speed on both drones: read, `--save` (drone A), `--apply`; live_follow applies `drone_profile.json` on connect |
| `botsort_tuned.yaml` | The tracker settings that fly (Michael 11-27): Ultralytics' `botsort.yaml` with only five values changed (high / new 0.40, low 0.05, match 0.90, buffer 60). Decision record `results/followme_tracker_settings_choice.json`; `live_follow.py` stops if the file differs from it |
| `flight_log.py` | The flight log: one row per flight in `~/follow_logs`, newest first, with the settings it flew under (drone, model + INT8, detector precision, cutoffs + source, tracker (+ tuned values), imgsz, re-ID + cutoff, pre-flight result) and what happened (duration, FOLLOW time, re-locks, spin searches, battery, video files, boxes.jsonl) → `flight_log.csv` + `flight_log.md`. The only flight list. Runs by itself at the end of every `live_follow.py` run; `python flight_log.py [--root DIR] [--copy-to DIR]` backfills |
| `flight_log_annotations.json` | What the flight log's runs did NOT record, each note with its source (Michael 11-27): drone corrections and what the chat history says about a run (model, cutoffs, tracker, re-ID, fake test or real flight, who was followed). Merged into the log on every regenerate, always marked "from chat history:" / "corrected ...; program logged: ..." |
| `lookhere.py` | Two drones: share a clicked ground spot (GPS); the other drone looks at it (V) or flies there (X) |
| `calibrate_camera.py` | Checkerboard video → `camera_calib.json` (optional; on standby) |
| `flight_decisions.json` | The settings that were actually decided (value, source, date). `live_follow.py` and `replay.py` check against it at startup and stop on any difference |
| `test_*.py` | 184 tests: logic on made-up inputs, full runs against the fake drone, the cutoffs recomputed from the sweeps' curves, `.pt` and INT8 (`test_cutoffs_evidence.py`), the re-lock cutoff checked against its evidence (`test_reid_evidence.py`), replay running the flight program's settings (`test_replay.py`), the tuned tracker settings checked against their record and the sweep, and the detector precision read from the export (`test_live_follow.py`), and the flight log + its annotations (`test_flight_log.py`) |

## Folders

- `weights/` — put `vd_mav_cl_s42_int8_openvino_model/` (the detector for flying AND replays, INT8 OpenVINO),
  `vd_mav_cl_s42.pt` (same detector unquantized; a replay with it is a deviation, `--off-decisions`) and
  `osnet_x0_25_msmt17.pt` (re-ID) here.
- `drone_profile.json` — the shared max tilt / rotation / climb / altitude / distance / gimbal settings, saved
  from drone A (11-22) and pushed to whichever drone connects. Committed so every clone flies the same.
  `lockon.py` looks in `weights/` first.
- Flight logs are **not** kept here: every run writes `~/follow_logs/<YYYYMMDD_HHMMSS>_<droneA|droneB>/` (with `drone.json`: name, serial, firmware; a `drone` column in follow.csv; `boxes.jsonl`: every box the program saw each frame; and `overlay_*.mp4`: the on-screen view with boxes and labels exactly as shown live — `--no-overlay-video` turns it off) (video chunks +
  `follow.csv`, 300+ MB per flight) in the WSL home, outside the repo and OneDrive. `--out` changes it.
- `~/follow_logs/flight_log.md` / `flight_log.csv` — the flight log (see Flight log below), rewritten at the end of
  every run.

## Setup (WSL, uv-managed venv `~/jarvis-follow`)

Michael's clone, as seen from WSL: `/mnt/c/Users/mahed/OneDrive/Documents/Jarvis/jarvis-lean` (branch
`followme-folder`); every command below runs from its `followme/` folder. (That clone also holds untracked `AGENTS.md`
and `followme/reid_speed.json`; leave them alone.)

```bash
source ~/jarvis-follow/bin/activate
# boxmot without its OpenCV downgrade, then the helpers it needs
uv pip install --no-deps boxmot==26.0.0
uv pip install gdown timm filterpy "rich>=15" click lapcpp pandas platformdirs pyarrow \
  scikit-learn threadpoolctl huggingface-hub tqdm typing-extensions
```

Check: `python -c "import lockon; e, n = lockon.make_embedder('osnet'); print(n)"` →
`re-ID: osnet_x0_25_msmt17`.

## Run

```bash
cd /mnt/c/Users/mahed/OneDrive/Documents/Jarvis/jarvis-lean/followme
source ~/jarvis-follow/bin/activate
python -m pytest -q test_live_follow.py test_lockon.py test_geo.py test_lookhere.py test_drone_settings.py test_cutoffs_evidence.py test_reid_evidence.py test_replay.py test_flight_log.py   # expect 184 passed
python live_follow.py                                                     # laptop on the drone's Wi-Fi (INT8 + OSNet by default)
python live_follow.py --fake --fake-detector                              # no drone: simulated drone + person (still needs OSNet)
python3 flight_log.py --root ~/follow_logs --copy-to /mnt/c/Users/mahed/Downloads   # backfill the flight log, copy it where Windows sees it
```

## Replay (test without flying)

```bash
python replay.py --source ~/follow_logs/<YYYYMMDD_HHMMSS>_droneA       # a flight: replays its recorded lock, N and M
python replay.py --source clip.mp4 --lock-id 3                          # any video: lock onto replay track 3 (3@120 = from frame 120)
python replay.py --source clip.mp4 --window                             # click a box to lock, SPACE pause, Q quit
```

Replay runs **what we fly** (Michael 11-26: "replay always matches whatever we're flying, unless I state explicitly
otherwise"): it has no settings of its own. Flags and defaults, detector, cutoffs, tracker, input size, re-ID,
match_sim, click-to-lock, `LOST_S` / `STALE_S` and the two-way 360° search all come from `live_follow.py` / `lockon.py`,
and the same check against `flight_decisions.json` runs first: any difference **stops**; `--off-decisions` runs it with
a DEVIATION banner (printed at the start and the end, saved in `replay_info.json`, stamped on every frame of
`replay.mp4` and in a `deviation` column of `commands.csv`). Nothing locks by itself: the lock is the flight's
recorded click (placed on the replay track that overlaps the recorded target box, from `boxes.jsonl`), `--lock-id`, or
a window click. Re-locks after a loss are the replay's own re-ID. Output in `<source>/replay/` (`--out`):
`replay.mp4`, `commands.csv` (the commands it would send next to the flight's `flight_*` columns), `replay_info.json`.

Known offline deviations, printed on every run and saved in `replay_info.json`: no drone profile (no drone); commands
are not sent, so the picture keeps showing what the flight did; the spin search counts turn from the recorded
`drone_yaw_deg`, not from replay's own yaw (no yaw telemetry → ends on the `SEARCH_S` timeout); detection runs on
every frame one at a time (live: background thread, newest frame), so `STALE_S` never fires and track IDs / re-ID
timing can differ; frames are the re-encoded mp4; only lock / N / mode are replayed from the operator, not look/go
or sticks. Flights before 11-22 have no `boxes.jsonl` and need `--lock-id` or `--window`.

Keys (window focused): T take off · L land · SPACE hover + manual · Q quit (lands first) ·
W/S forward/back · A/D left/right · R/F up/down · Z/C turn · G/B camera up/down · 1/2/3 speed ·
M follow on/off · N new lock (forget target + re-ID gallery) · click a box to lock onto it ·
P point mode (overseer: next click shares a ground spot) · O share the tracked target continuously (overseer) ·
V look at the shared spot (keeps watching a shared target) · X go to it.

## Lock-on behavior (Michael 2026-11-20)

- **Nothing locks until you click a box** (person or vehicle). Lock-on runs in both modes, so clicking
  yourself in MANUAL starts your re-ID gallery right away; pressing **M** keeps the target and gallery.
  **N** forgets them. With no target, FOLLOW hovers and shows "NO TARGET (click someone)".
- If the target's track ID is lost, it searches and **only** takes a person back on a confident re-ID
  match against their own gallery. It never locks onto anyone new: no nearest-box guess, no automatic
  "biggest person" (`--auto-lock` exists only for the fake-drone test).
- **Search (Michael 2026-11-21):** only clean boxes are compared (confident, not tiny, not cut off by the
  frame edge, not overlapping another person — the same rule as gallery samples), and a refused track is
  looked at again every 2 s instead of being dropped after 3 quick checks. Cutoff stays 0.70.
- **Vehicles:** OSNet only knows people, so a lost vehicle is not re-acquired automatically; click it again.
  A vehicle re-ID model is a planned add-on. Follow distance is tuned for people.

## Two drones: look here / go there (Michael 2026-11-22, not yet flown)

Overseer (high drone) and worker (lower drone) each run `live_follow.py`, sharing clicked spots:

```bash
# one laptop (two programs):  --share-file /tmp/jarvis_spot.json on both
python live_follow.py --name droneA --share-file /tmp/jarvis_spot.json
python live_follow.py --name droneB --listen droneA --share-file /tmp/jarvis_spot.json --ip 192.168.53.1
# two laptops: --mqtt <broker host> instead of --share-file (topic jarvis/sensor/<name>/target)
```

On the overseer press **P** and click the ground spot. On the worker press **V** (hover, turn + tilt the camera to look
at it) or **X** (turn to face it, fly slowly at a fixed height, stop `STOP_M` = 6 m short; refuses spots more than
40 m away). Any movement key, SPACE or L takes control back. Needs a GPS fix and a true compass heading on both drones
(heading unverified on the ANAFI 4K); the ground is assumed flat.

## Flight log (Michael 11-27)

One place listing every real flight with the context it was filmed under, so a flight video can be watched knowing the
settings and situation. Nothing is filled in by hand: at the end of every `live_follow.py` run (after the logs are
closed and the drone has landed) `flight_log.update()` rescans the logs root and rewrites `flight_log.csv` and
`flight_log.md` there, newest first. If that fails (e.g. Excel has the csv open) it prints a warning with the backfill
command and the program exits normally. Backfill or refresh by hand: `python flight_log.py [--root DIR] [--rebuild]
[--copy-to DIR]`; on Michael's laptop:

```bash
cd /mnt/c/Users/mahed/OneDrive/Documents/Jarvis/jarvis-lean/followme
python3 flight_log.py --root ~/follow_logs --copy-to /mnt/c/Users/mahed/Downloads
```

`--copy-to DIR` also copies `flight_log.csv` and `flight_log.md` into DIR (which must already exist; a mistyped
folder stops with an error), so Windows can open them from Downloads. Edit typed notes in the copy under
`~/follow_logs`, not the Downloads copy: only that one is read back on regenerate.

- `precision`: the detector's precision as flown, from drone.json's `detector` record (since 11-27, e.g. `INT8 OpenVINO,
  static 384x640`); older runs say `INT8 (derived from the model path; precision not logged)` only when the recorded
  model path ends in `_int8_openvino_model`, otherwise **not logged**.
- Every field is read from the run's own `drone.json` and `follow.csv`; a field that version of `live_follow.py` did not
  record shows **not logged** (never guessed: e.g. the 11-15 / 11-22 flights show no model, because drone.json had no
  model field until aae2f0f). The md lists which drone.json fields exist since which commit.
- Times are measured from `t_s`, not frames / an assumed 30 fps. FOLLOW time = time in FOLLOW mode (and, in brackets,
  on target); re-locks = re-ID re-locks (`reacquired_reid` onsets); clicks / N use `replay.recorded_locks`' rule.
- spin searches = spin searches started: transitions into a `state` starting `SEARCH (spinning` (a run that starts
  mid-spin does not count it). boxes.jsonl = yes / no (every box per frame, logged since 0af1cfb). path = the run
  folder's full path (csv only).
- Fake-drone runs (`drone_nickname` "fake") are left out (`--include-fake` lists them).
- Notes (optional): a `notes.txt` in the run folder (column `notes.txt`), or type into the `notes` column of
  `flight_log.csv`; typed notes are kept on every regenerate, matched by run folder (also if the folder is deleted).
- Unchanged run folders come from a cache (`.flight_log_cache.json`), so the call at the end of a flight is fast.

### Annotations: what the program did not record (`flight_log_annotations.json`, Michael 11-27)

Runs from before a field was logged (all 31 runs up to 11-22 show model / cutoffs / tracker / preflight **not logged**)
or with a wrong recorded value get their context from `followme/flight_log_annotations.json`, committed next to
`flight_log.py` and merged on every regenerate (applied after the cache, so an edit shows on the next update). Michael's
rule: never guess; anything not recorded by the program says where it came from. Keyed by run folder:

```json
"20261122_153153_droneA": {
  "drone":   {"value": "droneB", "reason": "corrected 11-27: serial ... and firmware 1.8.2 are drone B's; ..."},
  "history": {"run": "...", "followed": "...", "model": "vd_mav_cl_s42.pt (not INT8): the command Claude gave (11-22 15:23 PST, seq 26900 ...)"}
}
```

- `drone` (a correction): the column shows `droneB (corrected 11-27: ...; program logged: droneA)`.
- `history` (what the chat history says): a note for a column the run left empty (`drone`, `model`, `precision`,
  `cutoffs`, `tracker`, `imgsz`, `re-ID`) shows there as `not logged; from chat history: <note>`; any other key (`run`,
  `followed`, `code edits`, ...), or a column the program did fill, goes in the `history` column (csv and md), which
  always starts `from chat history:`. A value the program logged is never overwritten.
- Every note cites its source: the chat date / time (Michael's local time: PDT on 10-29, PST from 11-01) and message
  number (`seq`), or a commit. A field the chat does not clearly cover for a run is left out, so it stays **not
  logged**. A key that matches no run folder, an unknown entry key, or a `drone` override without a reason is ignored
  with a `flight log WARNING` line; a missing or broken file never stops the log.

## Change log

- **2026-11-27 (Michael): flight log annotations.** New committed `flight_log_annotations.json`, merged by
  `flight_log.py` on every regenerate (`load_annotations`, `annotate`; new `history` column in csv and md, `update(...,
  annotations=PATH)`): `drone` corrections shown with their reason and the program's value, and `history` notes from
  the chat transcript, each marked "from chat history:" and citing chat date / time + seq or a commit (see Flight log,
  Annotations). Filled for the 31 runs in Michael's 11-27 log: `20261122_153153_droneA` corrected to droneB (serial
  PI040416BA8J102372 / firmware 1.8.2 are drone B's; "droneA" was the program's fallback when the name wasn't read in
  time, fixed 553032f); `20261122_151520_droneA` / `_151758_droneA` corrected to droneB too (Michael 11-22 15:22: "drone
  B has flown on my home wifi" through the SkyController; no serial was logged). Fake-drone tests (10-29 x 9, from the
  pasted / given `--fake` commands) are marked as such but still listed (they predate drone.json, so the program can't
  tell). 6 new tests (override with reason, history fills only empty columns and never overwrites, unknown keys / bad
  entries warn and are ignored, missing / broken file never stops the log, edits apply to cached rows, every committed
  note cites a seq or commit and names a real run folder): 184 in all.

- **2026-11-27 (Michael): one flight log.** `flight_index.py` (11-22, commit 0af1cfb, wrote `flights.csv`) is folded
  into `flight_log.py` and deleted. Carried over: its spin-search count, by exactly its rule (`flight_log.spin_searches`,
  transitions into a state starting `SEARCH (spinning`; the existing `spin searches` column now uses it, so a run that
  starts mid-spin no longer counts one), its `boxes_logged` flag as a new `boxes.jsonl` column (yes / no), its folder
  path as a new `path` column (csv only, set on every update so it follows a moved root), and `--copy-to DIR` (copies
  `flight_log.csv` + `flight_log.md` there, e.g. `/mnt/c/Users/mahed/Downloads`; DIR must exist). Its other fields
  were already in the log (firmware, duration, airborne, FOLLOW / on-target time, re-ID re-locks, battery, LOOK/GO,
  video files). Cache schema 3. README Setup / Run now give the WSL path of Michael's clone and the backfill command.
  `test_live_follow.py`'s boxes test now checks `flight_log.summarize`; 10 new tests in `test_flight_log.py` (spin
  rule identical to the old expression on five state sequences, spin column, boxes.jsonl and path columns, `--copy-to`
  copies byte-identical files and refuses a missing folder, no copy without it, the old file and every reference gone):
  178 in all.

- **2026-11-27 (Michael): fly the tuned BoT-SORT settings; the detector's precision can't be missed.**
  - Tracker settings sweep (Colab today, `tracker_settings_sweep.py`; Ultralytics `botsort.yaml`, camera-motion
    compensation sparseOptFlow, re-ID off; chosen on VisDrone-MOT val class-average HOTA with the benchmark's own scoring
    and cached detections, checked on test-dev, MOT17 / MOT20 train and the flight-like INT8 detector). Committed
    byte-exact from Drive: `results/followme_tracker_settings_sweep.jsonl` (79 rows) and
    `results/followme_tracker_settings_sweep_summary.json`.
  - Michael chose **track_high_thresh 0.40, new_track_thresh 0.40, track_low_thresh 0.05, match_thresh 0.90,
    track_buffer 60**, fuse_score True, everything else stock. Not the val-best match_thresh 0.975: only +0.0026 on val
    (0.2802 vs 0.2776), within his ±0.003 seed-noise figure, and below stock on MOT20 (0.3166 vs 0.3173). 0.90 beats
    stock on every set (class-average HOTA, stock → tuned): val 0.2658 → 0.2776, test-dev 0.2927 → 0.2996, MOT17
    0.3946 → 0.4040, MOT20 0.3173 → 0.3326, INT8 val 0.2508 → 0.2574, INT8 test-dev 0.2823 → 0.2925.
  - New `botsort_tuned.yaml` (the installed ultralytics' stock file, five values edited, header citing the evidence)
    is the `--tracker` default for `live_follow.py` and so for `replay.py`; `flight_decisions.json` `tracker` is now
    `botsort_tuned.yaml`, so stock `botsort.yaml` **stops** the pre-flight / replay check unless `--off-decisions`.
    Decision record `results/followme_tracker_settings_choice.json`: the chosen values, four options (stock, 0.90,
    0.975, the sweep's earlier 0.35 / 0.90 / 30 pick) with class-average and person HOTA on all eight sets, the reason.
    `live_follow.resolve_tracker` stops if the yaml's five values differ from the record or the file is missing.
    In flight: live_follow hands the tracker only boxes >= 0.23, so a box needs >= 0.40 to start a track and 0.23-0.40
    only keeps one alive; `track_buffer` counts detector passes, not video frames.
  - Precision: `live_follow.detector_precision` reads it from the model files, never the folder name: the OpenVINO
    export's `metadata.yaml` (`args.int8` / `args.half`, or `args.quantize` on newer Ultralytics; `dynamic`, `imgsz`,
    calibration `data` / `fraction`) cross-checked against the FakeQuantize layers in its `.xml` (plus the NNCF
    parameters saved there), and the input shape from the `.xml`. Metadata and model disagreeing → `conflict`;
    nothing recorded → `not recorded`. A `.pt` is FP32 PyTorch (no half / quantize is passed). The flight export reads
    as INT8 (`args.int8 = true`, 67 FakeQuantize layers), static 384x640, Ultralytics 8.4.60, calibration
    `/content/data/int8_calib/calib.yaml` (fraction 1.0), NNCF 3.4.0 preset mixed, subset_size 300.
  - Startup (live and replay) now prints `detector: INT8 OpenVINO, static 384x640  (weights/…)`, where it came from,
    the export details, and a `tracker:` line with the five values. New decision `detector_precision` = `INT8`
    (Michael 11-22; skipped under `--fake-detector`). drone.json adds `detector` (the whole record),
    `running.detector_precision` and `tracker` (file + settings); replay_info.json the same.
  - `flight_log.py`: new `precision` column (csv + md; older runs: INT8 only from an `_int8_openvino_model` path, else
    not logged) and the tuned values in the `tracker` column; cache schema 2.
  - 13 new tests: yaml = record and stock otherwise (keys checked against the installed ultralytics), every number in
    the record found in the sweep rows, the stated reason re-checked, live and replay resolve to `botsort_tuned.yaml`,
    stock `botsort.yaml` stops (functions and an end-to-end start), edited / missing yaml or record stops, precision of
    the real export's metadata and of FP32 / FP16 / new-style / missing / conflicting / dynamic exports, precision as a
    pre-flight decision, drone.json records, flight-log column: 168 in all.

- **2026-11-27 (Michael): automatic flight log.** New `flight_log.py` (stdlib only) writes `flight_log.csv` +
  `flight_log.md` in the logs root, one row per flight, newest first: date/time, run folder, drone (+ serial,
  firmware), duration, frames, FPS, airborne time, battery, model (+INT8), cutoffs (+source), tracker (+settings file),
  imgsz, re-ID + re-lock cutoff, pre-flight pass / deviations, drone profile, FOLLOW time, re-locks, clicks / N, spin
  searches, LOOK/GO time, video files, any other `running` setting, notes. Missing fields read "not logged".
  `live_follow.py` calls it at the very end of its `finally` (after landing and closing the logs) inside a try: a
  failure prints a warning, never breaks shutdown. The fake-drone tests now pick run folders only (the logs root also
  holds the log files). `test_flight_log.py`: 17 new tests (new and old drone.json formats, missing / broken files,
  typed notes kept, newest first, fake runs left out, cache, the hook at exit and its failure path): 155 in all.


- **2026-11-26 (Michael): the INT8 model flies at its own measured cutoffs, person 0.23 / vehicle 0.29.** Rerun of the
  INT8 sweep, then "fly the measured INT8 cutoffs". Until now the INT8 export that flies took the `.pt`'s cutoffs
  (0.24 / 0.28) on the strength of an Oct 29 note that INT8 kept the same best cutoffs. **That note was wrong.**
  - Measured tonight (Colab, `thresholds_balanced.py`, same method; only change: one image per predict call at
    `imgsz=[384,640]`, because the export is static 384x640, the way `quantize_int8.py` scored it): **person 0.23**
    (mean F1 0.5364), **vehicle 0.29** (mean F1 0.5989) for `vd_mav_cl_s42_int8`. A `.pt` rerun in the same session
    reproduced 0.24 / 0.28 with identical curves; the older Drive file `jarvis_det/int8/int8_report.json` also says
    0.23 / 0.29.
  - Committed byte-exact from Drive: `results/followme_cutoffs_balanced_int8.json` and
    `results/followme_thresholds_balanced_int8.json` (the INT8 curves on the three sets).
  - `live_follow.resolve_cutoffs`: an INT8 export (`<name>_int8_openvino_model`) reads entry `<name>_int8` from the INT8
    file; anything else reads `<name>` / `<name>.pt` from `results/followme_cutoffs_balanced.json`. Exact names only and
    **no fallback** from INT8 to the `.pt` entry: missing INT8 file or entry → **stops**. Startup now prints
    `cutoffs: person >= 0.23, vehicle >= 0.29  (F1-max sweep for vd_mav_cl_s42_int8, results/followme_cutoffs_balanced_int8.json)`.
  - `flight_decisions.json` `detection_cutoffs` is now `results/followme_cutoffs_balanced_int8.json` (still "cutoffs read
    from the evidence file"); a `.pt` `--model` deviates on it as well as on the detector.
  - `test_cutoffs_evidence.py` recomputes the INT8 argmax from the INT8 curves and checks INT8 → 0.23 / 0.29,
    `.pt` → 0.24 / 0.28, INT8 entry or file missing → stop; `test_replay.py` checks a replay of the INT8 model gets
    0.23 / 0.29 and passes its settings check. 6 new tests: 138 in all.

- **2026-11-26 (Michael): replay runs what we fly.** "Replay lets you test things without flying", so it "always
  matches whatever we're flying, unless I state explicitly otherwise", and any deviation must be brought to his
  attention. Until now `replay.py` ran stock `yolov8n.pt` at one cutoff of 0.10 and grabbed the nearest/biggest person
  by itself (`pick_target`), none of which is what flies.
  - `replay.py` now builds its flags with `live_follow.add_flight_args` and runs `resolve_detection`,
    `Detector.process` (the live detector pass, now one method both use), `make_follow_logic` (the same `FollowLogic`
    + `lockon.TargetLock`) and `draw_follow`. It defines no setting of its own; `test_replay.py` fails if a replay-only
    flag, a different default or a shadow copy of a decided constant appears.
  - Same check as the pre-flight: `running_settings` + `check_decisions` against `flight_decisions.json`; mismatch →
    **stops**; `--off-decisions` → DEVIATION banner, saved in `replay_info.json`, on every frame and in `commands.csv`.
    The `replay` mode skips only `drone_profile_applied` (no drone).
  - Click-to-lock in replay: `pick_target` (nearest-box re-lock) is deleted. The lock comes from the flight log's
    recorded lock event, `--lock-id` or a window click; the recorded N and MANUAL/FOLLOW mode are replayed.
  - `flight_decisions.json` adds `lost_s` = 1.0 (Michael 11-26: "We can keep the 1 second spin": hover 1.0 s after
    the target is lost, then start the spin search; unchanged since the first commit, ce5675a). `LOST_S` moved from
    `replay.py` to `live_follow.py` next to `STALE_S`.
  - Unavoidable offline differences are listed in `replay.OFFLINE_DEVIATIONS` and printed on every run (see Replay).
  - 10 new tests (`test_replay.py`): 132 in all.

- **2026-11-26: re-lock cutoff read from its evidence; tracker and input size enforced.**
  - `lockon.MATCH_SIM` (0.70) is now read from `results/followme_reid_cutoff.json`; missing file → **stops**. That file
    is a hand-written decision record, because 0.70 was Michael's judgment (11-21) among measured options, not a
    computed number. The evidence is committed byte-exact next to it: `followme_reid_bench_v3.json` and
    `followme_reid_scores_osnet_x0_25.npz` (re-ID benchmark, Colab 11-22: OSNet x0_25 ROC-AUC 0.965; at 0.70, 8.1% of
    single stranger looks pass and 87% of returning-target looks; 1% of strangers pass at 0.789), and the 10-29 flight
    replays (`followme_reid_replay_1029_before_fix.json`, 7 cutoffs 0.65-0.85, the table he chose from;
    `followme_reid_replay_1029.json`, 0.70 / 0.725 / 0.75 with the search fixes). No wrong re-lock at any cutoff; at
    0.79 he came back at 0.779 and stayed lost ~2½ min. `test_reid_evidence.py` recomputes the benchmark summary from
    the raw scores, reads each option's stranger / target rate from them, checks the replay numbers, and requires the
    chosen value to be a replayed option with no wrong re-lock and to equal what runs. `MARGIN` (0.05) is still unmeasured.
  - `flight_decisions.json` adds `tracker` = `botsort.yaml` and `imgsz` = 640 (Michael 11-26: keep both; skipped under
    `--fake-detector`). Tracker benchmark (`results/followme_tracker_scores.jsonl`, byte-exact from Drive): with
    `vd_mav_cl_s42`, BoT-SORT + camera-motion compensation, tracker re-ID off (Ultralytics' default `botsort.yaml`) has
    the top class-average HOTA on VisDrone-MOT val 0.266, test-dev 0.293, MOT17 0.395, MOT20 0.317. Person-only HOTA
    on VisDrone is a little higher with tracker re-ID (Deep OC-SORT + re-ID 0.328 vs 0.316 on val), at ~270 vs ~25 ms
    per frame and more ID switches. 640 has been the default since the first version and no flight command changed it.
    A test re-derives the winner from the scores file; `drone.json` now records `tracker` and `imgsz` under `running`.

- **2026-11-26: cutoffs read from the evidence; automatic pre-flight check instead of reading the printout.**
  - The sweep's own output now lives in the repo, byte-exact from Drive: `results/followme_cutoffs_balanced.json`
    (one person + one vehicle cutoff per model) and `results/followme_thresholds_balanced.json` (every model's F1 curve
    on each of the three sets). `live_follow.py` reads the cutoffs from the first file; the hardcoded `MODEL_CUTOFFS`
    is gone. Missing file, or a model not in it → **stops**, unless `--conf-person` and `--conf-vehicle` are both given.
    Model names must match an entry exactly (`<key>` or `<key>.pt`), no prefix guessing.
    `test_cutoffs_evidence.py` recomputes the argmax of the equal-weight mean F1 from the curves for all 9 models ×
    2 classes and fails if it no longer equals the cutoffs file. (At this point the INT8 model used its `.pt` entry on
    the strength of an Oct 29 note that INT8 kept the same best cutoffs. Wrong: INT8 measures 0.23 / 0.29 and now has
    its own file; see the later 11-26 entry.)
  - `flight_decisions.json` lists the decided settings: INT8 detector, cutoffs from the results file, OSNet re-ID,
    match_sim 0.70, drone profile applied, click-to-lock, 360° search both ways. At startup the program compares what
    is about to run with it and **stops** on any difference; `--off-decisions` flies the deviations on purpose. Either
    way the result (pass, or the deviations) is saved in `drone.json` as `preflight`, next to `running`.
    `--fake-detector` skips the two detector checks, `--fake` skips drone profile and click-to-lock. A profile push that
    reports a FAILED setting now counts as not applied (it used to print and fly on). `--no-profile`, a `.pt`
    `--model` or hand-set cutoffs now also need `--off-decisions`.

- **2026-11-26 (Michael): fly at the F1-max cutoffs, per class.** The program used one cutoff, 0.25, for every
  class: Ultralytics' default, never chosen for this model. The F1-max sweep (`thresholds_balanced.py`, Oct 28-29:
  confidence 0.05-0.95 in 0.01 steps, IoU >= 0.5, equal-weight average of F1 on VisDrone val, MAVREC val (aerial) and a
  COCO holdout) picked **0.24 for people and 0.28 for vehicles** for `vd_mav_cl_s42` (the `.pt`; the INT8 export's own sweep
  picks 0.23 / 0.29, see the later 11-26 entry; the old note that INT8 kept the same cutoffs was wrong).
  These are now built in (`MODEL_CUTOFFS` in `live_follow.py`), printed at startup and saved in `drone.json`. `--conf`
  is gone; `--conf-person` / `--conf-vehicle` override on purpose. A model with no known cutoffs **stops** unless both are
  given. The Oct 29-30 flights passed `--conf 0.24` (one cutoff, so vehicles also ran at 0.24); from Nov 15 the flag
  was missing from the run commands, so flights from Nov 15 until this fix ran at 0.25 for every class.

- **2026-11-22 (Michael): fly the INT8 detector from now on, and no silent fallbacks.**
  - Why: the INT8 decision (10-29: ~3x faster, -0.01 F1) was lost from the copied run command on 11-15, and the
    11-15 and 11-22 flights ran the `.pt` without anyone noticing. Commands are now plain `python live_follow.py`.
  - Detector: default is `weights/vd_mav_cl_s42_int8_openvino_model`. If it's missing the program **stops**
    (it used to quietly use the `.pt`). A `.pt` still runs if passed with `--model` on purpose, with a warning.
  - Re-ID: default is `--reid osnet`. If OSNet can't load the program **stops** (it used to quietly switch to the
    colour histogram, which lets strangers score 0.72-0.88, over the 0.70 cutoff). `auto` is gone; `hist` only by name.
  - Drone settings: if `drone_profile.json` can't be applied the program **stops** (it used to print and fly on
    with mismatched max speeds). `--no-profile` skips it on purpose. The profile file is now committed.
  - First flight on INT8 after this: the next one. Flights before it (11-15, 11-22) are `.pt` in the flight log.

## Notes

- Lint: `run_lint.sh` only checks `demo/`; this folder has style-only ruff findings (one-line
  statements, long lines) and no F-class errors.
- Design history and decisions (search behavior, re-ID cutoff, one- vs two-laptop setup) live in the
  project notes, not here.
