# followme/ — follow-me drone (Parrot ANAFI 4K)

The live follow-me loop: drone video → YOLOv8n person/vehicle detector → BoT-SORT tracker
(with camera-motion compensation) → lock-on with OSNet re-ID → yaw/forward-back commands back to
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
| `flight_index.py` | One table of every flight in `~/follow_logs` (drone, time, follow minutes, re-locks, spins, battery, folder) → `flights.csv` |
| `lookhere.py` | Two drones: share a clicked ground spot (GPS); the other drone looks at it (V) or flies there (X) |
| `calibrate_camera.py` | Checkerboard video → `camera_calib.json` (optional; on standby) |
| `flight_decisions.json` | The settings that were actually decided (value, source, date). `live_follow.py` and `replay.py` check against it at startup and stop on any difference |
| `test_*.py` | 132 tests: logic on made-up inputs, full runs against the fake drone, the cutoffs recomputed from the sweep's curves (`test_cutoffs_evidence.py`), the re-lock cutoff checked against its evidence (`test_reid_evidence.py`), and replay running the flight program's settings (`test_replay.py`) |

## Folders

- `weights/` — put `vd_mav_cl_s42_int8_openvino_model/` (the detector for flying AND replays, INT8 OpenVINO),
  `vd_mav_cl_s42.pt` (same detector unquantized; a replay with it is a deviation, `--off-decisions`) and
  `osnet_x0_25_msmt17.pt` (re-ID) here.
- `drone_profile.json` — the shared max tilt / rotation / climb / altitude / distance / gimbal settings, saved
  from drone A (11-22) and pushed to whichever drone connects. Committed so every clone flies the same.
  `lockon.py` looks in `weights/` first.
- Flight logs are **not** kept here: every run writes `~/follow_logs/<YYYYMMDD_HHMMSS>_<droneA|droneB>/` (with `drone.json`: name, serial, firmware; a `drone` column in follow.csv; `boxes.jsonl`: every box the program saw each frame; and `overlay_*.mp4`: the on-screen view with boxes and labels exactly as shown live — `--no-overlay-video` turns it off) (video chunks +
  `follow.csv`, 300+ MB per flight) in the WSL home, outside the repo and OneDrive. `--out` changes it.

## Setup (WSL, uv-managed venv `~/jarvis-follow`)

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
cd followme
python -m pytest -q test_live_follow.py test_lockon.py test_geo.py test_lookhere.py test_drone_settings.py test_cutoffs_evidence.py test_reid_evidence.py test_replay.py   # expect 132 passed
python live_follow.py                                                     # laptop on the drone's Wi-Fi (INT8 + OSNet by default)
python live_follow.py --fake --fake-detector                              # no drone: simulated drone + person (still needs OSNet)
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

## Change log

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
    Model names must match an entry exactly (`<key>`, `<key>.pt` or `<key>_int8_openvino_model`), no prefix guessing.
    `test_cutoffs_evidence.py` recomputes the argmax of the equal-weight mean F1 from the curves for all 9 models ×
    2 classes and fails if it no longer equals the cutoffs file. The INT8 model uses its `.pt` entry (Oct 29 note: INT8
    kept the same best cutoffs); that INT8 sweep itself is not in the repo.
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
  COCO holdout) picked **0.24 for people and 0.28 for vehicles** for `vd_mav_cl_s42`; INT8 kept the same best cutoffs.
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
