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
| `replay.py` | Shared pieces: follow controller, target drawing, class names |
| `record.py` | Olympe connection, telemetry (incl. GPS), and the fake drone used by tests |
| `geo.py` | Map position: target box → ray → flat ground → meters north/east of takeoff |
| `drone_settings.py` | Same max tilt / rotation / climb speed on both drones: read, `--save` (drone A), `--apply`; live_follow applies `drone_profile.json` on connect |
| `lookhere.py` | Two drones: share a clicked ground spot (GPS); the other drone looks at it (V) or flies there (X) |
| `calibrate_camera.py` | Checkerboard video → `camera_calib.json` (optional; on standby) |
| `test_*.py` | 61 tests: logic on made-up inputs + one full run against the fake drone |

## Folders

- `weights/` — put `vd_mav_cl_s42.pt` (detector) and `osnet_x0_25_msmt17.pt` (re-ID) here.
  `lockon.py` looks in `weights/` first.
- Flight logs are **not** kept here: every run writes `~/follow_logs/<YYYYMMDD_HHMMSS>_<droneA|droneB>/` (with `drone.json`: name, serial, firmware; and a `drone` column in follow.csv) (video chunks +
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
python -m pytest -q test_live_follow.py test_lockon.py test_geo.py test_lookhere.py test_drone_settings.py   # expect 61 passed
python live_follow.py --model weights/vd_mav_cl_s42.pt --reid osnet       # laptop on the drone's Wi-Fi
python live_follow.py --fake --fake-detector                              # no drone: simulated drone + person
```

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
python live_follow.py --model weights/vd_mav_cl_s42.pt --reid osnet --name droneA --share-file /tmp/jarvis_spot.json
python live_follow.py --model weights/vd_mav_cl_s42.pt --reid osnet --name droneB --listen droneA --share-file /tmp/jarvis_spot.json --ip 192.168.53.1
# two laptops: --mqtt <broker host> instead of --share-file (topic jarvis/sensor/<name>/target)
```

On the overseer press **P** and click the ground spot. On the worker press **V** (hover, turn + tilt the camera to look
at it) or **X** (turn to face it, fly slowly at a fixed height, stop `STOP_M` = 6 m short; refuses spots more than
40 m away). Any movement key, SPACE or L takes control back. Needs a GPS fix and a true compass heading on both drones
(heading unverified on the ANAFI 4K); the ground is assumed flat.

## Notes

- Lint: `run_lint.sh` only checks `demo/`; this folder has style-only ruff findings (one-line
  statements, long lines) and no F-class errors.
- Design history and decisions (search behavior, re-ID cutoff, one- vs two-laptop setup) live in the
  project notes, not here.
