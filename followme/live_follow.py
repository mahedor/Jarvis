#!/usr/bin/env python3
"""LIVE follow-me on the ANAFI:  video -> detector + tracker -> controller -> drone, with you as the safety pilot.

Run in WSL (Olympe env + ultralytics):
  python live_follow.py                                          # real drone (INT8 detector + OSNet re-ID; stops if either is missing)
  python live_follow.py --fake --fake-detector --seconds 20      # no drone, no model: tests the loop + safety logic

Pre-flight self-check (11-26): before anything flies, the settings about to run (detector, INT8, cutoffs source,
re-ID, match_sim, drone profile, click-to-lock, search) are compared with flight_decisions.json. Any difference STOPS
the program unless --off-decisions is passed; the result (pass, or the deliberate deviations) goes into drone.json.
--fake-detector skips the two detector checks, --fake skips the drone-profile and click-to-lock checks.

It starts in MANUAL (same keys as record.py). Follow mode only runs when YOU turn it on, and any movement key
takes control back instantly.
  T takeoff   L land   Q quit (lands first)   SPACE hover + MANUAL
  W/S A/D R/F Z/C (or arrows) fly by hand -> always switches to MANUAL
  CLICK a box to lock onto it (person or vehicle); nothing locks without a click (11-20).
  M  toggle FOLLOW. Lock-on + re-ID also run in MANUAL: the clicked target and its re-ID gallery carry over when
     FOLLOW turns on.   N  new lock (forget target + gallery)
  1/2/3 manual speed 20/35/50 %    G/B gimbal up/down 5 deg
  Two drones (lookhere.py, 11-22; needs --share-file or --mqtt):
  P  (overseer) point mode: the next click shares that ground spot instead of locking a box
  V  (worker) LOOK at the shared spot: hover in place, turn + tilt the camera only
  O  (overseer) share the TRACKED target: while the clicked/followed person is in view, their ground spot is
     re-shared twice a second, so a worker in LOOK (V) keeps turning to watch them (11-22)
  X  (worker) GO to the shared spot: turn to face it, fly slowly, stop lookhere.STOP_M short, keep height
     Any movement key, SPACE or L takes back control (MANUAL).

Safety rules in FOLLOW (all of them make the drone HOVER, i.e. send zero sticks):
  - target lost for more than LOST_S seconds           - detections older than STALE_S (detector stalled)
  - no video for more than 1 s (also drops to MANUAL)  - battery below --min-battery (also drops to MANUAL)
  - forward/back and yaw are capped at --max-speed %;  height is never changed (gaz = 0)
Every frame + command is logged to ~/follow_logs/<timestamp>_<drone>/ (kept out of the repo and OneDrive): clean video
chunks, overlay_*.mp4 (the on-screen view with boxes, exactly as shown live), follow.csv, boxes.jsonl, drone.json.
Detection runs in a background thread on the newest frame, so keys and stick commands never wait for the model.
"""
import argparse
import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import record                       # connect_real / connect_fake (Olympe I/O, tested 2026-10-25)
import replay                       # Controller, pick_target, draw (the same logic the replays use)
import geo                          # one-drone map position: target box -> metres N/E of takeoff (2026-11-08)
import lockon                       # lock-on: gallery of re-ID vectors, verified re-acquisition (2026-11-10)
import lookhere                     # two drones: share a clicked ground spot, look at it / go there (2026-11-22)

HOLD = 0.3
STALE_S = 0.6
SPEEDS = {ord("1"): 20, ord("2"): 35, ord("3"): 50}
ARROWS = {65362: "up", 65364: "down", 65361: "yaw_l", 65363: "yaw_r"}
MOVES = {"w": dict(pitch=1), "s": dict(pitch=-1), "a": dict(roll=-1), "d": dict(roll=1),
         "r": dict(gaz=1), "f": dict(gaz=-1), "z": dict(yaw=-1), "c": dict(yaw=1)}
ARROW_MOVES = {"up": dict(gaz=1), "down": dict(gaz=-1), "yaw_l": dict(yaw=-1), "yaw_r": dict(yaw=1)}
GIMBAL_PERIOD_S = 0.3
GIMBAL_MIN, GIMBAL_MAX = -90.0, 0.0


class Detector(threading.Thread):
    """Runs model.track on the newest frame it is given; publishes (tracks, names, frame_time, ms)."""
    def __init__(self, model_path, tracker, imgsz, cutoffs=(0.24, 0.28), fake=False, embedder=None, want_fn=None):
        super().__init__(daemon=True)
        self.fake = fake
        self.embedder, self.want_fn = embedder, want_fn     # re-ID: only the crops lock-on asks for (usually 0-1)
        self.vecs, self.embed_ms = {}, 0.0
        self.lock, self.new = threading.Lock(), threading.Event()
        self.frame = self.frame_t = None
        self.result = ([], {0: "person"}, 0.0, 0.0)
        self.stop_flag = False
        self.person_cut, self.vehicle_cut = cutoffs
        if not fake:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.names = self.model.names
            # the model keeps everything above the lower cutoff; keep_box then applies each class's own cutoff
            self.args = dict(persist=True, tracker=tracker, imgsz=imgsz, conf=min(cutoffs), verbose=False)
        else:
            self.names = {0: "person"}

    def submit(self, frame, t):
        with self.lock:
            self.frame, self.frame_t = frame, t
        self.new.set()

    def latest(self):
        with self.lock:
            return self.result

    def _fake_tracks(self, img):
        # finds the orange ball record.FakeDrone draws and calls it person #1
        m = cv2.inRange(img, (0, 150, 200), (60, 255, 255))
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return []
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        return [(float(x1), float(y1 - 60), float(x2), float(y2 + 60), 1, 0, 0.9)]

    def run(self):
        while not self.stop_flag:
            if not self.new.wait(0.2):
                continue
            self.new.clear()
            with self.lock:
                img, t = self.frame, self.frame_t
            t0 = time.time()
            if self.fake:
                tracks = self._fake_tracks(img)
            else:
                r = self.model.track(img, **self.args)[0]
                tracks = []
                if r.boxes is not None and r.boxes.id is not None:
                    for b, tid, cls, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int),
                                               r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
                        if keep_box(self.names.get(int(cls), ""), float(cf), self.person_cut, self.vehicle_cut):
                            tracks.append((*b.tolist(), int(tid), int(cls), float(cf)))
            det_ms = (time.time() - t0) * 1000
            vecs, t1 = {}, time.time()
            if self.embedder is not None and self.want_fn is not None and tracks:
                ids = set(self.want_fn(tracks, t, img.shape[1], img.shape[0]))
                sel = [tr for tr in tracks if tr[4] in ids]
                if sel:
                    try:
                        V = self.embedder(img, [tr[:4] for tr in sel])
                        vecs = {tr[4]: V[i] for i, tr in enumerate(sel)}
                    except Exception as ex:                  # noqa: BLE001 - re-ID must never kill the detector
                        print("re-ID embed failed:", ex)
            with self.lock:
                self.result = (tracks, self.names, t, det_ms)
                self.vecs, self.embed_ms = vecs, (time.time() - t1) * 1000

    def latest_vecs(self):
        with self.lock:
            return self.vecs, self.embed_ms


SEARCH_YAW = 50.0   # % of max yaw while searching. Michael 2026-11-14: "yaw pretty quickly... a 360 relatively quickly
                    # but it SHOULDN'T change its coordinates" -> yaw only (roll = pitch = gaz = 0), NOT capped by
                    # --max-speed (that cap is for following, which moves the drone; this only spins it)
SEARCH_DEG = 360.0  # one full turn per direction, measured from drone_yaw_deg telemetry
SEARCH_BOTH = True  # Michael 2026-11-14: spin one way (toward the side last seen), then the other way, then hover
SEARCH_S = 40.0     # safety timeout for the whole search if telemetry is missing or turns are slow; then hover


class FollowLogic:
    """Decides the stick command in FOLLOW mode. Pure logic (no I/O) so the safety rules can be tested.

    Target choice and re-acquisition live in lockon.TargetLock (2026-11-10): a lost target is only re-locked onto a
    person whose re-ID vector matches the target's gallery; without a gallery it falls back to the old nearest-box rule.
    Lost for < lost_s: hover (the tracker often gets them back under the same ID). Then spin IN PLACE toward the side
    they were last seen, quickly, for one full 360 (or search_s at most), re-ID checking everyone who comes into view
    (Michael 2026-11-14). Then hover until picked again. Only yaw changes while searching: no roll/pitch/gaz.
    """
    def __init__(self, person_ids, max_speed, lost_s=replay.LOST_S, stale_s=STALE_S, search_s=SEARCH_S,
                 search_yaw=SEARCH_YAW, match_sim=lockon.MATCH_SIM, search_deg=SEARCH_DEG, search_both=SEARCH_BOTH,
                 auto_lock=False):
        self.ctrl = replay.Controller()
        self.person_ids, self.max_speed, self.lost_s, self.stale_s = person_ids, max_speed, lost_s, stale_s
        self.search_s, self.search_yaw, self.search_deg, self.search_both = search_s, search_yaw, search_deg, search_both
        self._yaw_prev, self.turned_deg = None, 0.0
        self.lock = lockon.TargetLock(person_ids, match_sim=match_sim, auto_lock=auto_lock)
        self.last_seen = None
        self.event = ""
        self._fed_det_t = None

    # the rest of live_follow (drawing, logging) reads these
    target_id = property(lambda self: self.lock.target_id)
    picked_by = property(lambda self: self.lock.picked_by)
    last_box = property(lambda self: self.lock.last_box)

    def reset(self):
        self.ctrl = replay.Controller(); self.lock.reset(); self.last_seen = None; self.event = ""; self._fed_det_t = None

    def pick(self, tid, tracks=None):
        self.lock.pick(tid, tracks, by="you"); self.last_seen = time.time()

    def resume(self):
        """FOLLOW switched on: fresh controller and search, but KEEP the target + re-ID gallery built so far
        (Michael 2026-11-20: lock-on should already know you before follow mode starts)."""
        self.ctrl = replay.Controller(); self._yaw_prev, self.turned_deg = None, 0.0
        if self.lock.target_id is not None and self.lock.lost_since is None:
            self.last_seen = time.time()

    def observe(self, tracks, det_t, now, W, H, vecs=None):
        """MANUAL mode: run lock-on (choose the target, build the gallery, re-lock) but send no commands."""
        if det_t is None or now - det_t > self.stale_s:
            return ""
        if det_t == self._fed_det_t:
            vecs = {}
        self._fed_det_t = det_t
        tgt, self.event = self.lock.update(tracks, vecs or {}, det_t, W, H)
        if tgt is not None:
            self.last_seen = now
        return self.event

    def wants_embedding(self, tracks, now, W=None, H=None):
        return self.lock.wants_embedding(tracks, now, W, H)

    def step(self, tracks, det_t, now, W, H, tel, vecs=None):
        """Returns (pcmd dict, info dict). Hover = all zeros. vecs = {track_id: re-ID vector} from the detector."""
        hover = dict(roll=0, pitch=0, yaw=0, gaz=0)
        if det_t is None or now - det_t > self.stale_s:
            return hover, dict(state="STALE -> HOVER", gimbal_delta=0.0, dist=None, method="stale", event="")
        if det_t == self._fed_det_t:                    # the loop runs faster than the detector: use each
            vecs = {}                                  # frame's re-ID vectors once, not every loop iteration
        self._fed_det_t = det_t
        tgt, self.event = self.lock.update(tracks, vecs or {}, det_t, W, H)
        if tgt is None and self.lock.target_id is None:
            return hover, dict(state="NO TARGET (click someone)", gimbal_delta=0.0, dist=None, method="no_target",
                               event=self.event)
        if tgt is None:
            gone = 0.0 if self.last_seen is None else now - self.last_seen
            if self.last_seen is None or gone <= self.lost_s:
                return hover, dict(state="SEARCHING (hover)", gimbal_delta=0.0, dist=None, method="searching",
                                   event=self.event)
            cur = tel.get("drone_yaw_deg") if tel else None
            if cur is not None:                            # accumulate how far we've actually turned
                if self._yaw_prev is not None:
                    self.turned_deg += abs((cur - self._yaw_prev + 180.0) % 360.0 - 180.0)
                self._yaw_prev = cur
            total = self.search_deg * (2 if self.search_both else 1)
            if gone <= self.lost_s + self.search_s and self.search_yaw > 0 and self.turned_deg < total:
                self.ctrl.step(None, W, H, tel)
                first = self.lock.last_side(W)                 # first turn: toward where the target was last seen
                direction = first if self.turned_deg < self.search_deg else -first   # second turn: the other way
                yaw = direction * min(self.search_yaw, 100.0)
                return dict(roll=0, pitch=0, yaw=yaw, gaz=0), dict(state=f"SEARCH (spinning {self.turned_deg:.0f}/{total:.0f} deg)",
                                                                   gimbal_delta=0.0, dist=None, method="searching",
                                                                   event=self.event)
            self.ctrl.step(None, W, H, tel)
            return hover, dict(state="LOST -> HOVER", gimbal_delta=0.0, dist=None, method="lost", event=self.event)
        self.last_seen = now
        self._yaw_prev, self.turned_deg = None, 0.0        # found: the next loss starts a fresh 360
        c = self.ctrl.step(tgt[:4], W, H, tel)
        cap = lambda v: float(np.clip(v, -self.max_speed, self.max_speed))
        cmd = dict(roll=0, pitch=cap(c["pitch"]), yaw=cap(c["yaw"]), gaz=0)
        return cmd, dict(state="FOLLOWING", gimbal_delta=c["gimbal_delta"], dist=c["dist"], method=c["method"],
                         event=self.event)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
INT8_SUFFIX = "_int8_openvino_model"
INT8_MODEL = HERE / "weights" / ("vd_mav_cl_s42" + INT8_SUFFIX)   # live detector (Michael 11-22)

# Detection cutoffs per model, from the F1-max sweep (thresholds_balanced.py, Oct 28-29, Colab): confidence swept
# 0.05-0.95 in 0.01 steps, IoU >= 0.5, equal-weight average of F1 on VisDrone val, MAVREC val (aerial) and a COCO
# holdout; one cutoff for people, one for all vehicle classes. Michael 11-26: fly at these instead of the old single
# --conf 0.25 (Ultralytics' default, never chosen for this model). The numbers are READ from the sweep's own output,
# committed byte-exact from Drive; test_cutoffs_evidence.py recomputes them from the full curves
# (results/followme_thresholds_balanced.json) so the two can't drift. A model's INT8 export uses its .pt's entry
# (the INT8 run kept the same best cutoffs, Oct 29; that INT8 sweep is not in the repo). No silent fallback: missing
# file or a model not in it -> stop, unless --conf-person and --conf-vehicle are both given on purpose.
CUTOFFS_FILE = REPO / "results" / "followme_cutoffs_balanced.json"
DECISIONS_FILE = HERE / "flight_decisions.json"


def _rel(p, base=REPO):
    p = Path(p).resolve()
    return p.relative_to(base).as_posix() if p.is_relative_to(base) else str(p)


def load_cutoffs(path=CUTOFFS_FILE):
    """{model_key: {"person": cut, "vehicle": cut, ...}} from the F1-max sweep's cutoffs file; missing -> stop."""
    p = Path(path)
    if not p.is_file():
        sys.exit(f"STOP: F1-max cutoffs file not found: {p}\n"
                 f"  It is committed in results/; restore it, or pass --conf-person and --conf-vehicle on purpose.")
    return json.loads(p.read_text())


def cutoff_key(model_path, table):
    """The cutoffs-file entry for this model: exact name, name.pt, or name + INT8_SUFFIX. No prefix guessing."""
    stem = Path(str(model_path)).name
    return next((k for k in table if stem in (k, k + ".pt", k + INT8_SUFFIX)), None)


def resolve_cutoffs(model_path, conf_person=None, conf_vehicle=None, path=CUTOFFS_FILE):
    """-> (person, vehicle, source). Explicit values win; else the model's entry in the cutoffs file; else stop."""
    if conf_person is not None and conf_vehicle is not None:
        return conf_person, conf_vehicle, "set on the command line"
    table = load_cutoffs(path)
    key = cutoff_key(model_path, table)
    if key is None:
        sys.exit(f"STOP: no F1-max cutoffs for {Path(str(model_path)).name} in {_rel(path)}. "
                 f"Pass --conf-person and --conf-vehicle on purpose.")
    src = f"F1-max sweep for {key}, {_rel(path)}"
    if conf_person is not None or conf_vehicle is not None:
        src += " (one value overridden on the command line)"
    return (conf_person if conf_person is not None else table[key]["person"],
            conf_vehicle if conf_vehicle is not None else table[key]["vehicle"], src)


def load_decisions(path=DECISIONS_FILE):
    p = Path(path)
    if not p.is_file():
        sys.exit(f"STOP: {p} not found: the pre-flight check has nothing to compare against.")
    return json.loads(p.read_text())["decisions"]


def check_decisions(actual, decisions, modes):
    """Compare what is about to run with flight_decisions.json.
    actual: {decision name: running value}; modes: active test-mode flags, e.g. {"--fake-detector"}.
    A decision is skipped only if one of its skip_with flags is active. A decision live_follow has no reading for
    counts as a deviation (never silently passes).
    -> {"passed": bool, "deviations": [{name, decided, actual, source}], "skipped": [{name, because}], "checked": [...]}"""
    res = dict(passed=True, deviations=[], skipped=[], checked=[])
    for name, d in decisions.items():
        because = [f for f in d.get("skip_with", []) if f in modes]
        if because:
            res["skipped"].append(dict(name=name, because=because))
            continue
        res["checked"].append(name)
        got = actual.get(name, "<no reading in live_follow.py>")
        if got != d["value"]:
            res["deviations"].append(dict(name=name, decided=d["value"], actual=got, source=d["source"]))
    res["passed"] = not res["deviations"]
    return res


def enforce_decisions(result, off_decisions):
    """Stop on any deviation unless --off-decisions; -> the record saved in drone.json."""
    if result["passed"]:
        print(f"pre-flight check: PASS ({len(result['checked'])} decisions"
              + (f"; skipped in this mode: {', '.join(s['name'] for s in result['skipped'])})" if result["skipped"] else ")"))
        return dict(result, off_decisions=off_decisions)
    lines = "\n".join(f"  {x['name']}: decided {x['decided']!r}, about to run {x['actual']!r}" for x in result["deviations"])
    if not off_decisions:
        sys.exit(f"STOP: not flying what was decided ({_rel(DECISIONS_FILE)}):\n{lines}\n"
                 f"  Fix it, or pass --off-decisions to fly these deviations on purpose (they are logged in drone.json).")
    print(f"pre-flight check: DELIBERATE DEVIATIONS (--off-decisions), logged in drone.json:\n{lines}")
    return dict(result, off_decisions=True)


def profile_ok(note):
    """record.apply_profile's answer -> was drone_profile.json really applied? (missing file or a failed setting = no)"""
    return not str(note).startswith("no drone_profile") and "FAILED" not in str(note)


def keep_box(cls_name, conf, person_cut, vehicle_cut):
    """Per-class cutoff: people use the person cutoff, every other class the vehicle cutoff."""
    return conf >= (person_cut if str(cls_name).lower() in replay.PERSON_NAMES else vehicle_cut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.42.1")
    ap.add_argument("--model", default=str(INT8_MODEL),
                    help="detector (default: the INT8 OpenVINO model, ~3x faster than .pt; Michael 11-22). No silent "
                         "fallback: if it's missing the program stops. Pass a .pt here only on purpose.")
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-person", type=float, default=None,
                    help="person cutoff (default: this model's F1-max from results/followme_cutoffs_balanced.json)")
    ap.add_argument("--conf-vehicle", type=float, default=None,
                    help="vehicle cutoff (default: this model's F1-max from results/followme_cutoffs_balanced.json)")
    ap.add_argument("--max-speed", type=float, default=20.0, help="cap on FOLLOW yaw and forward/back, %% of max")
    ap.add_argument("--follow-dist", type=float, default=replay.FOLLOW_DIST_M, help="metres to keep from the target")
    ap.add_argument("--min-battery", type=float, default=25.0)
    ap.add_argument("--calib", default="", help="camera_calib.json from calibrate_camera.py (default: next to this file)")
    ap.add_argument("--reid", default="osnet", choices=["osnet", "hist", "none"],
                    help="lock-on appearance model: osnet (default; stops if boxmot/OSNet can't load), hist = colour "
                         "histogram baseline (tests only), none = no re-ID (a lost target is never re-acquired)")
    ap.add_argument("--match-sim", type=float, default=lockon.MATCH_SIM, help="re-ID similarity needed to re-lock (0.70, Michael 11-21)")
    ap.add_argument("--search-s", type=float, default=SEARCH_S, help="max seconds spinning to find a lost target (0 = just hover)")
    ap.add_argument("--search-yaw", type=float, default=SEARCH_YAW, help="%% yaw while spinning in place to search")
    ap.add_argument("--search-deg", type=float, default=SEARCH_DEG, help="degrees per spin direction (360 = one full turn)")
    ap.add_argument("--search-one-way", action="store_true", help="only spin one way (default: one way, then the other)")
    ap.add_argument("--out", default=str(Path.home() / "follow_logs"))   # outside the repo/OneDrive: 300+ MB per flight
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fake", action="store_true", help="no drone (synthetic video)")
    ap.add_argument("--fake-detector", action="store_true", help="no model (tracks the fake video's ball)")
    ap.add_argument("--fake-stall", type=float, nargs=2, default=(0, 0))
    ap.add_argument("--headless", action="store_true", help="(testing) no window; --script drives the keys")
    ap.add_argument("--script", default="", help="(testing) 't@1,m@2,w@5,m@6,q@9' = key@seconds")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--auto-lock", action="store_true",
                    help="lock onto the biggest person without a click (fake-drone tests only; default: click to lock)")
    ap.add_argument("--name", default="", help="this drone's name in shared messages (default: its nickname, droneA / droneB)")
    ap.add_argument("--listen", default="", help="name of the drone whose clicked spots this one follows (worker)")
    ap.add_argument("--share-file", default="", help="JSON file used to share clicked spots (both programs on ONE laptop)")
    ap.add_argument("--mqtt", default="", help="MQTT broker host used to share clicked spots (TWO laptops)")
    ap.add_argument("--no-overlay-video", action="store_true",
                    help="don't also record the on-screen view (boxes + labels exactly as shown live) to overlay_*.mp4")
    ap.add_argument("--no-profile", action="store_true", help="don't push drone_profile.json to the drone on connect")
    ap.add_argument("--off-decisions", action="store_true",
                    help="fly even though settings differ from flight_decisions.json (on purpose; logged in drone.json)")
    args = ap.parse_args()
    if not args.fake_detector and not Path(args.model).exists():   # no silent fallback to the .pt (Michael 11-22)
        sys.exit(f"STOP: detector not found: {args.model}\n"
                 f"  Put vd_mav_cl_s42_int8_openvino_model/ in followme/weights/, or pass --model on purpose.")
    cut_p, cut_v, cut_src = resolve_cutoffs(args.model, args.conf_person, args.conf_vehicle)
    replay.FOLLOW_DIST_M = args.follow_dist
    lookgo, point_mode, shared_px = lookhere.LookGo(), False, None
    share_target, last_share_t, last_rx_print = False, 0.0, 0.0

    d = record.connect_fake(tuple(args.fake_stall)) if args.fake else record.connect_real(args.ip)
    ident = d["identity"]() if "identity" in d else {}
    print(f"detector: {args.model}" + ("" if "int8" in str(args.model).lower() else "   <-- NOT the INT8 model (slower)"))
    print(f"cutoffs: person >= {cut_p:.2f}, vehicle >= {cut_v:.2f}  ({cut_src})")
    print(f"drone: {ident.get('drone_nickname')} ({ident.get('drone_product_name')}, firmware {ident.get('drone_firmware')})")
    nick = ident.get("drone_nickname")
    args.name = args.name or (nick if nick not in (None, "unknown", "fake") else
                              "droneA" if args.fake else f"unknown_{args.ip.replace('.', '_')}")   # never guess a drone (11-22)
    link = (lookhere.TargetLink(args.share_file or None, args.mqtt or None, drone=args.name,
                                listen_to=args.listen or ("droneA" if args.name != "droneA" else "droneB"))
            if (args.share_file or args.mqtt) else None)
    profile_note = "not applied (--no-profile)" if args.no_profile else "not applied (fake drone)"
    if not args.no_profile and "apply_profile" in d:
        try:
            profile_note = d["apply_profile"]()                # same max tilt / rotation / climb on both drones (11-22)
            print("settings:", profile_note)
        except Exception as e:                                 # no silent fallback (Michael 11-22)
            sys.exit(f"STOP: could not apply drone_profile.json ({e}). Fix it, or pass --no-profile on purpose.")
    embedder, reid_note = lockon.make_embedder(args.reid)
    print(reid_note)
    # ---- pre-flight self-check: what is about to run vs. what was decided (replaces reading the printout) ----
    modes = {f for f, on in (("--fake", args.fake), ("--fake-detector", args.fake_detector)) if on}
    actual = dict(detector_model=_rel(args.model, HERE), detector_int8=Path(args.model).name.endswith(INT8_SUFFIX),
                  detection_cutoffs=(_rel(CUTOFFS_FILE) if args.conf_person is None and args.conf_vehicle is None
                                     else f"command line: person {cut_p}, vehicle {cut_v}"),
                  reid=args.reid, match_sim=args.match_sim,
                  drone_profile_applied="apply_profile" in d and not args.no_profile and profile_ok(profile_note),
                  click_to_lock=not args.auto_lock, search_deg=args.search_deg, search_both_ways=not args.search_one_way)
    try:
        preflight = enforce_decisions(check_decisions(actual, load_decisions(), modes), args.off_decisions)
    except SystemExit:
        d["close"]()                                           # still on the ground: disconnect cleanly, then stop
        raise
    det = Detector(args.model, args.tracker, args.imgsz, (cut_p, cut_v), fake=args.fake_detector, embedder=embedder)
    person_ids = {k for k, v in det.names.items() if str(v).lower() in replay.PERSON_NAMES}
    logic = FollowLogic(person_ids, args.max_speed, search_s=args.search_s, search_yaw=args.search_yaw, search_deg=args.search_deg, search_both=not args.search_one_way,
                        match_sim=args.match_sim, auto_lock=args.auto_lock)
    det.want_fn = lambda tr, t, W, H: logic.wants_embedding(tr, t, W, H)   # lock-on runs in MANUAL too (11-20)
    calib = geo.load_calib(args.calib or None)
    mapf = geo.MapFrame(calib)
    print('map position:', 'camera_calib.json' if calib else f'ASSUMED FOV {geo.HFOV_DEG}x{geo.VFOV_DEG} deg (run calibrate_camera.py)')
    det.start()

    out = Path(args.out) / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.name}'   # which drone, and two programs never share a folder
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "follow.csv", "w", newline=""); log_w = None
    box_f = open(out / "boxes.jsonl", "w")    # every box the program saw, per frame: replays show exactly what it saw (11-22)
    (out / "drone.json").write_text(json.dumps(dict(ident, name=args.name, ip=args.ip, model=str(args.model),
                                                    conf_person=cut_p, conf_vehicle=cut_v, conf_source=cut_src,
                                                    reid=reid_note, profile=profile_note, running=actual,
                                                    preflight=preflight), indent=1))
    writer, chunk, chunk_frames = None, 0, 0
    script = sorted((float(s.split("@")[1]), s.split("@")[0]) for s in args.script.split(",") if "@" in s)
    mode, speed, gimbal_pitch, flying = "MANUAL", 20, 0.0, False
    held, last_key_t, last_gimbal_t = None, 0.0, 0.0
    img, last_frame_t, t0, n = None, time.time(), time.time(), 0
    click = {"pt": None}
    if not args.headless:
        cv2.namedWindow("follow", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("follow", lambda ev, x, y, fl, p: click.__setitem__("pt", (x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    print(f"Logging to {out}. MANUAL mode. T = takeoff, M = follow on/off, SPACE = hover, L = land, Q = quit.")
    info = dict(state="MANUAL")
    # per-step timing of the main loop (ms). The row for a frame carries this iteration's wait/record times and the
    # PREVIOUS iteration's draw/show/key times (the row is written before this iteration draws).
    tm = dict(wait_ms=0.0, rec_ms=0.0, draw_ms=0.0, show_ms=0.0, key_ms=0.0, ov_ms=0.0, loops_since_frame=0)
    ov_writer, ov_chunk = None, -1          # the live on-screen view, chunked in step with video_*.mp4 (11-22)
    pc = time.perf_counter
    try:
        while True:
            _a = pc()
            new = d["next_frame"](timeout=0.03)                       # never block
            tm["wait_ms"] = (pc() - _a) * 1000
            tm["loops_since_frame"] += 1
            now = time.time(); t = now - t0
            if new is not None:
                img, last_frame_t = new, now
                det.submit(new, now)
            elif img is None:
                img = np.zeros((720, 1280, 3), np.uint8)
            H, W = img.shape[:2]
            video_ok = now - last_frame_t < 1.0
            tel = d["telemetry"]()
            tracks, names, det_t, det_ms = det.latest()
            vecs, embed_ms = det.latest_vecs()

            # ---- keys (window or test script) ----
            ch, k = "", -1
            if script and t >= script[0][0]:
                ch = script.pop(0)[1]
            elif not args.headless:
                _a = pc()
                k = cv2.waitKeyEx(1)
                tm["key_ms"] = (pc() - _a) * 1000
                if k != -1:
                    kc = k & 0xFF if k < 256 else k
                    ch = chr(kc).lower() if kc < 256 else ""
            if ch == "q":
                break
            if ch == "t":
                d["takeoff"](); flying = True
            elif ch == "l":
                mode = "MANUAL"; lookgo.stop(); d["pcmd"](); d["land"](); flying = False
            elif ch == " ":
                mode, held = "MANUAL", None; lookgo.stop()
            elif ch == "p":
                point_mode = not point_mode and link is not None
                print("POINT mode: click a spot to share it" if point_mode else
                      ("point mode off" if link else "no --share-file / --mqtt: nothing to share with"))
            elif ch == "o":
                share_target = not share_target and link is not None
                print("SHARING the tracked target's ground spot (2/s)" if share_target else
                      ("stopped sharing the target" if link else "no --share-file / --mqtt: nothing to share with"))
            elif ch in ("v", "x"):
                if lookgo.start("look" if ch == "v" else "go"):
                    mode, held = "LOOKGO", None; print("LOOK at shared spot" if ch == "v" else "GO to shared spot")
                else:
                    print("no shared spot yet (press P on the overseer and click the ground)")
            elif ch == "m" and mode == "MANUAL" and tel.get("battery_pct") is not None and tel["battery_pct"] < args.min_battery:
                print(f"battery {tel['battery_pct']}% < {args.min_battery:.0f}%: FOLLOW refused - land and swap the battery")
            elif ch == "m":
                mode = "FOLLOW" if mode == "MANUAL" else "MANUAL"; held = None
                if mode == "FOLLOW":
                    logic.resume()                     # keeps the target + gallery collected in MANUAL
                print("mode", mode, f"(target {logic.target_id}, gallery {len(logic.lock.gallery)})")
            elif ch == "n":
                logic.reset(); print("NEW LOCK: target + gallery cleared (next = biggest person, or click a box)")
            elif ch == "g":
                gimbal_pitch = min(GIMBAL_MAX, gimbal_pitch + 5); d["gimbal"](gimbal_pitch)
            elif ch == "b":
                gimbal_pitch = max(GIMBAL_MIN, gimbal_pitch - 5); d["gimbal"](gimbal_pitch)
            elif k in SPEEDS:
                speed = SPEEDS[k]
            else:
                mv = MOVES.get(ch) or (ARROW_MOVES.get(ARROWS.get(k)) if k in ARROWS else None)
                if mv:
                    if mode in ("FOLLOW", "LOOKGO"):
                        print("manual key -> MANUAL")
                    mode = "MANUAL"; lookgo.stop(); held = {a: v * speed for a, v in mv.items()}; last_key_t = now
            if click["pt"] is not None:
                if point_mode:
                    g = lookhere.click_to_gps(*click["pt"], W, H, tel, calib)
                    if g is None:
                        print("can't place that spot (no GPS fix / heading, or clicked above the horizon)")
                    else:
                        link.publish(*g); shared_px = click["pt"]; point_mode = False
                        print(f"shared spot {g[0]:.6f}, {g[1]:.6f}")
                else:
                    hit = replay.box_at(tracks, *click["pt"])
                    if hit is not None:
                        logic.pick(hit[4], tracks)
                click["pt"] = None
            if share_target and logic.target_id is not None and now - last_share_t > 0.5:
                seen = [t for t in tracks if t[4] == logic.target_id]
                if seen and seen[0][3] < H - 3:                      # feet visible: the box bottom is on the ground
                    g = lookhere.click_to_gps((seen[0][0] + seen[0][2]) / 2, seen[0][3], W, H, tel, calib)
                    if g is not None:
                        link.publish(*g); last_share_t = now
            if link is not None:
                got = link.poll()
                if got is not None:
                    lookgo.set_target(*got, now)
                    if now - last_rx_print > 10:
                        print(f"received spot {got[0]:.6f}, {got[1]:.6f}: V = look, X = go"); last_rx_print = now

            # ---- safety drops out of FOLLOW ----
            bat = tel.get("battery_pct")
            if mode in ("FOLLOW", "LOOKGO") and not video_ok:
                mode = "MANUAL"; lookgo.stop(); print("NO VIDEO -> MANUAL (hover)")
            if mode in ("FOLLOW", "LOOKGO") and bat is not None and bat < args.min_battery:
                mode = "MANUAL"; lookgo.stop(); print(f"battery {bat}% -> MANUAL (hover)")

            # ---- command ----
            if mode == "FOLLOW":
                cmd, info = logic.step(tracks, det_t, now, W, H, tel, vecs)
                if info["gimbal_delta"] and now - last_gimbal_t > GIMBAL_PERIOD_S:
                    gp = tel.get("gimbal_pitch_deg")
                    base = gp if gp is not None else gimbal_pitch
                    gimbal_pitch = float(np.clip(base + info["gimbal_delta"], GIMBAL_MIN, GIMBAL_MAX))
                    d["gimbal"](gimbal_pitch); last_gimbal_t = now
            elif mode == "LOOKGO":
                cmd, gp, lg = lookgo.step(tel, now)
                if gp is not None and abs(gp - gimbal_pitch) > 1.0 and now - last_gimbal_t > GIMBAL_PERIOD_S:
                    gimbal_pitch = float(np.clip(gp, GIMBAL_MIN, GIMBAL_MAX)); d["gimbal"](gimbal_pitch); last_gimbal_t = now
                ev = logic.observe(tracks, det_t, now, W, H, vecs)
                info = dict(state=lg["state"], dist=lg.get("dist"), method="lookgo", event=lg.get("event") or ev, lg=lg)
                if lookgo.mode is None:
                    mode = "MANUAL"
            else:
                if held and now - last_key_t > HOLD:
                    held = None
                cmd = dict(roll=0, pitch=0, yaw=0, gaz=0, **{}) if not held else {**dict(roll=0, pitch=0, yaw=0, gaz=0), **held}
                ev = logic.observe(tracks, det_t, now, W, H, vecs)
                info = dict(state="MANUAL", dist=None, method="manual", event=ev)
            d["pcmd"](**cmd)

            # ---- log + record ----
            if new is not None:
                if writer is None or chunk_frames >= int(60 * args.fps):
                    if writer: writer.release()
                    name = f"video_{chunk:03d}.mp4"; chunk += 1; chunk_frames = 0
                    writer = cv2.VideoWriter(str(out / name), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
                _a = pc()
                writer.write(img); chunk_frames += 1
                tm["rec_ms"] = (pc() - _a) * 1000
                row = dict(frame=n, t_s=round(t, 3), video_file=name, frame_in_file=chunk_frames - 1, mode=mode,
                           state=info["state"], target_id=logic.target_id,
                           dist_m=info.get("dist"), method=info.get("method"), roll=cmd["roll"], pitch=round(cmd["pitch"], 1),
                           yaw=round(cmd["yaw"], 1), gaz=cmd["gaz"], gimbal_cmd=round(gimbal_pitch, 1),
                           det_age_ms=None if det_t is None else round((now - det_t) * 1000), det_ms=round(det_ms, 1),
                           n_tracks=len(tracks), lock_event=info.get("event", ""),
                           gallery_n=len(logic.lock.gallery), n_embedded=len(vecs), embed_ms=round(embed_ms, 1),
                           lg_state=info.get("lg", {}).get("state", ""), lg_bearing=info.get("lg", {}).get("bearing"),
                           lg_yaw_err=info.get("lg", {}).get("yaw_err"),
                           shared_lat=lookgo.target[0] if lookgo.target else None,
                           shared_lon=lookgo.target[1] if lookgo.target else None,
                           best_sim=(logic.lock.last_sim if info.get("event") == "reacquired_reid" else
                                     max((c[2] for c in logic.lock.checked.values() if c[2] is not None), default=None)),
                           drone=args.name, **{k: round(v, 1) for k, v in tm.items()}, **tel,
                           **mapf.update(logic.last_box if mode == "FOLLOW" and info["state"] == "FOLLOWING" else None,
                                         W, H, tel))
                tm["loops_since_frame"] = 0
                if log_w is None:
                    log_w = csv.DictWriter(log_f, fieldnames=list(row)); log_w.writeheader()
                log_w.writerow(row); log_f.flush()
                box_f.write(__import__("json").dumps(dict(
                    frame=n, video_file=name, frame_in_file=chunk_frames - 1, target_id=logic.target_id,
                    boxes=[[round(float(v), 1) for v in t[:4]] + [int(t[4]), int(t[5]), round(float(t[6]), 3)] for t in tracks],
                    names={int(k): v for k, v in names.items()} if n == 0 else None)) + "\n")
                n += 1

            # ---- display ----
            if not args.headless:
                _a = pc()
                c = dict(yaw=cmd["yaw"], pitch=cmd["pitch"], gimbal_delta=0.0, dist=info.get("dist") or 0.0,
                         method=info.get("method") if mode == "FOLLOW" and info["state"] == "FOLLOWING" else "searching")
                vis = replay.draw(img.copy(), tracks, names, logic.target_id, c, 1000 / max(det_ms, 1), tel, n, 0, logic.picked_by)
                col = (0, 255, 0) if mode == "FOLLOW" and info["state"] == "FOLLOWING" else (0, 200, 255) if mode == "FOLLOW" else (200, 200, 200)
                cv2.putText(vis, f"{mode}: {info['state']}   (M follow on/off, N new lock, SPACE hover, L land)", (10, H - 45), 0, 0.8, col, 2)
                if share_target:
                    cv2.putText(vis, "SHARING TARGET with the other drone (O to stop)", (10, 150), 0, 0.8, (255, 0, 255), 2)
                if point_mode:
                    cv2.putText(vis, "POINT MODE: click the ground spot to share (P to cancel)", (10, 80), 0, 0.9, (255, 0, 255), 2)
                if shared_px is not None:
                    cv2.drawMarker(vis, shared_px, (255, 0, 255), cv2.MARKER_CROSS, 40, 3)
                if lookgo.target and link is not None:
                    cv2.putText(vis, f"shared spot ready: V look, X go  {info.get('lg', {}).get('dist', '')}", (10, 115), 0, 0.7, (255, 0, 255), 2)
                if not video_ok:
                    cv2.putText(vis, "NO VIDEO - L = land, SPACE = hover", (30, H // 2), 0, 1.0, (0, 0, 255), 3)
                if bat is not None and bat < args.min_battery:
                    cv2.putText(vis, f"BATTERY {bat}% < {args.min_battery:.0f}%: FOLLOW/LOOK/GO OFF - land (L)", (30, 160), 0, 1.0, (0, 0, 255), 3)
                tm["draw_ms"] = (pc() - _a) * 1000
                _a = pc()
                cv2.imshow("follow", vis)
                tm["show_ms"] = (pc() - _a) * 1000
                if new is not None and not args.no_overlay_video:
                    _a = pc()
                    if ov_chunk != chunk:                    # new video chunk started -> new overlay chunk
                        if ov_writer: ov_writer.release()
                        ov_chunk = chunk
                        ov_writer = cv2.VideoWriter(str(out / f"overlay_{chunk - 1:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                    args.fps, (vis.shape[1], vis.shape[0]))
                    ov_writer.write(vis)
                    tm["ov_ms"] = (pc() - _a) * 1000
            if args.seconds and t > args.seconds:
                break
    finally:
        det.stop_flag = True
        det.join(timeout=3)              # let a running YOLO call finish: killing it mid-call aborts Python at exit
        try:
            d["pcmd"]()
        except Exception:
            pass
        if writer: writer.release()
        if ov_writer: ov_writer.release()
        log_f.close(); box_f.close()
        if flying:
            print("Landing..."); d["land"]()
        d["close"]()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"Saved {n} frames + follow.csv to {out}")


if __name__ == "__main__":
    main()
