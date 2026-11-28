#!/usr/bin/env python3
"""Replay a recorded flight (or any footage) through the FLIGHT PROGRAM's detector -> tracker -> lock-on -> follow
logic, WITHOUT flying. Saves replay.mp4 (the same on-screen view live_follow draws), commands.csv (the commands it
WOULD send, next to what the flight recorded) and replay_info.json (settings check + known offline deviations).

  python replay.py --source ~/follow_logs/20261129_150554_droneA     # a live_follow.py flight (follow.csv + boxes.jsonl)
  python replay.py --source ~/recordings/20261025_150000 --lock-id 3  # a record.py folder (telemetry.csv)
  python replay.py --source clip.mp4 --window                         # any video / image folder: click to lock

Michael 11-26: "replay always matches whatever we're flying, unless I state explicitly otherwise", and any deviation
must be brought to his attention. So this file has NO settings of its own: the flags, the detector and its precision
(read from the model files, 11-27), the cutoffs, the tracker (botsort_tuned.yaml, 11-27), the input size, re-ID, MATCH_SIM, MARGIN, click-to-lock, LOST_S / STALE_S and the 360-degree two-way search all come
from live_follow.py / lockon.py (add_flight_args, resolve_detection, Detector.process, make_follow_logic). Before
anything runs, the same check as live_follow's pre-flight compares the settings with flight_decisions.json and STOPS on
any difference unless --off-decisions; with it, a DEVIATION banner is printed, saved in replay_info.json and stamped on
every frame. (Until 11-26 replay ran stock yolov8n at conf 0.10 with the old nearest-box auto re-lock.)

Lock (click-to-lock, Michael 11-20): nothing locks by itself. The lock comes from the flight log's recorded lock event
(the operator's click, or first_lock in a --fake --auto-lock test flight), placed on the replay's own track with the
highest IoU to the recorded target box (boxes.jsonl); or from --lock-id TID[@FRAME | @SECONDSs]; or from a click with
--window. For flights with no boxes.jsonl (before 11-22), --lock-id only@SECONDSs locks the ONLY person in view at that
time (a stand-in for clicking them; if several people are in view it locks nobody and prints their IDs). A recorded
N (new lock) clears it.

Comparison runs (11-27): --label 'cutoff 0.79' stamps a neutral caption on every frame. With --off-decisions it replaces
the red per-frame DEVIATION banner, but only if it names every deviating value (else STOP); the console banners,
replay_info.json (preflight + label + frame_caption) and commands.csv's deviation column still carry the deviation. Re-locks after a loss are the replay's own re-ID decisions, never copied from the
log. The recorded MANUAL / FOLLOW mode is replayed too (MANUAL: lock-on runs, no commands; M -> FOLLOW: resume).
Sources with no flight log (video, image folder, record.py) replay as FOLLOW throughout.

Known offline deviations (unavoidable; printed every run and saved in replay_info.json): see OFFLINE_DEVIATIONS.

What it measured (11-28, talk videos: no number without a file): replay_info.json's summary adds target_lost_pct,
target_ids_used, person_track_ids and det_ms (definitions in STATS_DEFINITIONS, saved beside them), plus lock_events
(every lock / re-ID re-lock with its similarity / N, with frame, t_s and the target box) and target_segments (how long
each target ID was followed); lock_events.txt is the same list as text; commands.csv adds the target box per frame.

The shared pieces live_follow imports from here: Controller (P-control with deadbands, clamps and smoothing; gains
are first guesses), estimate_distance, draw, box_at, PERSON_NAMES / VEHICLE_NAMES.
Distance:
  'angle' (needs telemetry): ground distance = height_above_takeoff / tan(angle down to the target's FEET)
  'box'   (fallback): pinhole estimate from box height, assuming a 1.7 m person
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

# ---- controller tunables (ASSUMED first guesses; live_follow's controller is this one) ----
VFOV_DEG = 42.0          # ANAFI live stream vertical field of view, from ~69 deg HFOV at 16:9 (ASSUMED, verify)
PERSON_H_M = 1.7
FOLLOW_DIST_M = 6.0
K_YAW, K_PITCH, K_GIMBAL = 60.0, 12.0, 0.25    # % per unit offset, % per metre, deg per deg of offset
MAX_YAW, MAX_PITCH = 40, 30                     # % stick
DEADBAND_X, DEADBAND_D = 0.05, 0.5              # fraction of half-width, metres
MIN_ANGLE_DEG = 3.0                             # below this the angle method is too unstable -> box method
SMOOTH = 0.5                                    # EMA on commands (0 = no smoothing)


def _log_file(p):
    """The per-frame CSV of a recording folder: record.py writes telemetry.csv, live_follow.py writes follow.csv.
    Both have a video_file column and the same telemetry columns, so either drives the replay."""
    for name in ("telemetry.csv", "follow.csv"):
        if p.is_dir() and (p / name).exists():
            return p / name
    return None


def count_frames(source):
    p = Path(source).expanduser()
    if _log_file(p):
        return sum(1 for _ in open(_log_file(p))) - 1
    if p.is_dir():
        return sum(1 for x in p.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_COUNT))


def frames_from(source):
    """Yields (frame_bgr, telemetry_row_or_None). Handles record.py folders, video files, image folders."""
    p = Path(source).expanduser()
    if _log_file(p):
        rows = list(csv.DictReader(open(_log_file(p))))
        caps = {}
        for r in rows:
            f = r["video_file"]
            if f not in caps:
                caps = {f: cv2.VideoCapture(str(p / f))}
            ok, img = caps[f].read()
            if not ok:
                continue
            yield img, r
    elif p.is_dir():
        for f in sorted(x for x in p.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png")):
            yield cv2.imread(str(f)), None
    else:
        cap = cv2.VideoCapture(str(p))
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img, None


def _f(row, key):
    try:
        v = row.get(key) if row else None
        return float(v) if v not in (None, "", "None") else None
    except ValueError:
        return None


def estimate_distance(box, H, tel):
    """Returns (metres or None, method)."""
    x1, y1, x2, y2 = box
    alt, gp = _f(tel, "altitude_takeoff_m"), _f(tel, "gimbal_pitch_deg")
    feet_visible = y2 < H - 3                                # feet cut off by the frame edge -> angle method is wrong
    if alt is not None and gp is not None and alt > 0.3 and feet_visible:
        off = math.degrees(math.atan(((y2 - H / 2) / (H / 2)) * math.tan(math.radians(VFOV_DEG / 2))))
        down = -gp + off                                  # gimbal pitch is negative when looking down
        if down > MIN_ANGLE_DEG:
            return alt / math.tan(math.radians(down)), "angle"
    h_px = max(y2 - y1, 1.0)
    f_px = (H / 2) / math.tan(math.radians(VFOV_DEG / 2))
    return PERSON_H_M * f_px / h_px, ("box" if feet_visible else "box, feet cut off")


class Controller:
    def __init__(self):
        self.cmd = dict(yaw=0.0, pitch=0.0, gaz=0.0, gimbal_delta=0.0)

    def step(self, box, W, H, tel):
        if box is None:
            self.cmd = dict(yaw=0.0, pitch=0.0, gaz=0.0, gimbal_delta=0.0)
            return dict(self.cmd, dist=None, method="lost")
        x1, y1, x2, y2 = box
        ex = ((x1 + x2) / 2 - W / 2) / (W / 2)
        ey = ((y1 + y2) / 2 - H / 2) / (H / 2)
        dist, method = estimate_distance(box, H, tel)
        yaw = 0.0 if abs(ex) < DEADBAND_X else np.clip(K_YAW * ex, -MAX_YAW, MAX_YAW)
        derr = dist - FOLLOW_DIST_M
        pitch = 0.0 if abs(derr) < DEADBAND_D else np.clip(K_PITCH * derr, -MAX_PITCH, MAX_PITCH)
        gim = -K_GIMBAL * ey * (VFOV_DEG / 2)              # tilt down if target is low in the frame
        new = dict(yaw=float(yaw), pitch=float(pitch), gaz=0.0, gimbal_delta=float(gim))
        self.cmd = {k: SMOOTH * self.cmd[k] + (1 - SMOOTH) * new[k] for k in new}
        return dict(self.cmd, dist=dist, method=method)


VEHICLE_NAMES = {"car", "van", "truck", "bus", "motor", "motorcycle", "bicycle", "tricycle", "awning-tricycle",
                 "three_wheeler", "threewheeler"}
PERSON_NAMES = {"person", "pedestrian", "people"}
DRAW_CONF = 0.35            # other people / vehicles are drawn only above this confidence (target always drawn)
METHOD_TEXT = {"angle": "camera angle", "box": "box size", "box, feet cut off": "box size, feet cut off"}


def draw(img, tracks, names, target_id, c, fps, tel, frame_i, total, picked_by, paused=False):
    """tracks: (x1,y1,x2,y2,id,cls,conf). Decided 2026-10-25: target + confident people + vehicles; status, distance
    and method, commands, telemetry."""
    H, W = img.shape[:2]
    for x1, y1, x2, y2, tid, cls, conf in tracks:
        nm = str(names.get(cls, cls)).lower()
        is_t = tid == target_id
        if not is_t and (conf < DRAW_CONF or nm not in PERSON_NAMES | VEHICLE_NAMES):
            continue
        col = (0, 255, 0) if is_t else (0, 165, 255) if nm in PERSON_NAMES else (255, 140, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 3 if is_t else 1)
        label = f"TARGET {tid}" if is_t else f"{nm} {tid} {conf:.2f}"
        cv2.putText(img, label, (int(x1), max(int(y1) - 5, 46)), 0, 0.5, col, 2 if is_t else 1)
    cx, cy = W // 2, H // 2
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    status = {"lost": "LOST -> HOVER", "searching": "SEARCHING"}.get(c["method"], "LOCKED")
    if status == "LOCKED":
        cv2.arrowedLine(img, (cx, cy), (int(cx + c["yaw"] * 4), cy), (0, 200, 255), 3, tipLength=0.3)
        cv2.arrowedLine(img, (cx, cy), (cx, int(cy - c["pitch"] * 4)), (255, 120, 0), 3, tipLength=0.3)
    # top bar: status + target + distance + commands
    cv2.rectangle(img, (0, 0), (W, 40), (0, 0, 0), -1)
    scol = (0, 255, 0) if status == "LOCKED" else (0, 200, 255) if status == "SEARCHING" else (0, 0, 255)
    cv2.putText(img, status, (8, 28), 0, 0.8, scol, 2)
    if status == "LOCKED":
        txt = (f"target {target_id} ({picked_by}) | {c['dist']:.1f} m by {METHOD_TEXT.get(c['method'], c['method'])} | "
               f"turn {c['yaw']:+.0f}%  fwd {c['pitch']:+.0f}%  tilt {c['gimbal_delta']:+.1f} deg")
    else:
        txt = f"last target {target_id} ({picked_by})" if target_id is not None else "no target yet"
    cv2.putText(img, txt, (230, 27), 0, 0.55, (255, 255, 255), 1)
    # bottom bar: telemetry
    def f(k, fmt):
        v = _f(tel, k)
        return "n/a" if v is None else fmt.format(v)
    cv2.rectangle(img, (0, H - 30), (W, H), (0, 0, 0), -1)
    bot = (f"height {f('altitude_takeoff_m', '{:.1f} m')} | camera {f('gimbal_pitch_deg', '{:.0f} deg')} | "
           f"battery {f('battery_pct', '{:.0f}%')} | frame {frame_i}/{total} | {fps:.1f} fps"
           + ("   [PAUSED - space to play]" if paused else ""))
    cv2.putText(img, bot, (8, H - 9), 0, 0.5, (255, 255, 255), 1)
    return img


def box_at(tracks, x, y):
    """Smallest tracked box containing the click."""
    hits = [t for t in tracks if t[0] <= x <= t[2] and t[1] <= y <= t[3]]
    return min(hits, key=lambda t: (t[2] - t[0]) * (t[3] - t[1])) if hits else None


# ---- the replay program: every setting comes from live_follow.py / lockon.py ----

# What replay cannot do the way the flight does (11-26). Printed on every run and saved in replay_info.json.
OFFLINE_DEVIATIONS = [
    "drone profile: not applied. Replay talks to no drone (drone_profile_applied is skipped in replay mode).",
    "commands are computed, never sent: the camera, gimbal and drone did what the FLIGHT did, so after the first "
    "difference the picture no longer follows the replay's commands.",
    "search spin: the 360-degree two-way search counts its turn from the RECORDED drone_yaw_deg (what the drone did in "
    "the flight), not from the yaw the replay commands. With no yaw telemetry (video, image folder) it ends on the "
    "SEARCH_S timeout.",
    "timing: detection runs on every recorded frame, one at a time. Live it runs in a background thread on the newest "
    "frame and skips frames when slower than the video. So STALE_S (detector stall) never triggers here, the follow "
    "logic runs once per frame (live: every loop), and tracker IDs and re-ID sample times can differ from the flight. "
    "Clock = the recorded t_s (else frame / --fps).",
    "frames: the recorded mp4 (re-encoded), not the raw stream the live detector saw.",
    "operator input: only the recorded lock / N (new lock) / MANUAL-FOLLOW mode are replayed (or --lock-id / --window "
    "clicks); look/go and manual sticks are not. Track IDs differ from the flight, so a recorded lock is placed on the "
    "replay track with the highest box overlap (IoU >= LOCK_IOU).",
]
LOCK_IOU = 0.3      # a recorded lock is placed on the replay track overlapping the recorded target box at least this much


def _flight():
    """live_follow, imported here and not at the top: live_follow imports this file (Controller, draw, box_at), so a
    top-level import would be circular. Everything decided comes from it."""
    import live_follow
    return live_follow


def load_flight_log(src):
    """A live_follow.py flight folder -> dict(rows, boxes, running, note); anything else -> None.
    boxes: {frame: [[x1, y1, x2, y2, id, cls, conf], ...]} from boxes.jsonl (flights from 11-22 on), else None."""
    src = Path(src)
    if not (src.is_dir() and (src / "follow.csv").exists()):
        return None
    rows = list(csv.DictReader(open(src / "follow.csv")))
    boxes = None
    if (src / "boxes.jsonl").exists():
        boxes = {}
        for line in (src / "boxes.jsonl").read_text().splitlines():
            if line.strip():
                b = json.loads(line)
                boxes[int(b["frame"])] = b["boxes"]
    running = None
    if (src / "drone.json").exists():
        running = json.loads((src / "drone.json").read_text()).get("running")
    return dict(rows=rows, boxes=boxes, running=running)


def recorded_locks(rows):
    """The operator's lock decisions in a flight's follow.csv -> [(frame, "pick", recorded_target_id) |
    (frame, "reset", None)], or None if the log can't tell clicks from re-ID re-locks (no lock_event column).
    target_id is the lock's target after each frame. A change to a new id that is NOT a re-ID re-lock
    (lock_event reacquired_reid) is a click (or first_lock in an --auto-lock fake-drone flight); a change to empty
    is N (new lock). Re-locks are left to the replay's own re-ID, never copied."""
    if not rows or "lock_event" not in rows[0]:
        return None
    ev, prev = [], ""
    for r in rows:
        tid = r.get("target_id") or ""
        if tid != prev:
            if tid == "":
                ev.append((int(r["frame"]), "reset", None))
            elif r.get("lock_event") != "reacquired_reid":
                ev.append((int(r["frame"]), "pick", tid))
        prev = tid
    return ev


def place_lock(box, tracks):
    """The replay track with the highest IoU to a recorded target box, if at least LOCK_IOU; else None."""
    import lockon
    best = max(tracks, key=lambda t: lockon.iou(t[:4], box), default=None)
    return best if best is not None and lockon.iou(best[:4], box) >= LOCK_IOU else None


ONLY = "only"       # --lock-id only@T: the one person in view (a stand-in for a click on them; never a choice among people)


def parse_lock_id(s):
    """-> (tid, frame, t_s): when to place the lock = the first frame at or after `frame` (or, with an 's' suffix, the
    first frame whose recorded time is at or after t_s seconds).
      '3' -> (3, 0, None); '3@120' -> (3, 120, None); '3@18s' -> (3, None, 18.0)
      'only@18s' -> ('only', None, 18.0): lock onto the ONLY person track in view at that time. If several people are
      in view nothing is locked (no guessing: their IDs are printed and saved, use TID@... instead); if nobody is,
      it waits for the first frame with a person."""
    if not s:
        return None, 0, None
    tid, _, at = str(s).partition("@")
    tid = ONLY if tid.strip().lower() == ONLY else int(tid)
    at = at.strip().lower()
    if at.endswith("s"):
        return tid, None, float(at[:-1])
    return tid, int(at or 0), None


def lock_target(lock_tid, tracks, person_ids):
    """--lock-id on this frame's tracks -> (track to lock or None, the person tracks in view).
    A TID locks that track if present. ONLY locks the person track only if it is the single one: with several people
    it returns None (never a choice among people: no biggest / nearest / most confident)."""
    people = [t for t in tracks if t[5] in person_ids]
    if lock_tid == ONLY:
        return (people[0] if len(people) == 1 else None), people
    return next((t for t in tracks if t[4] == lock_tid), None), people


def label_problem(label, deviations):
    """--label replaces the red DEVIATION banner on the frames only if it names every deviation's running value
    (e.g. 'cutoff 0.79' for match_sim 0.79), so the picture still says what differs. -> None if OK, else why not."""
    missing = [d for d in deviations if str(d["actual"]) not in label]
    if not missing:
        return None
    return ("--label must name every deviating value it stands in for, so the frames still say what differs; "
            "missing: " + ", ".join(f"{d['name']} {d['actual']!r}" for d in missing))


# ---- what a replay measured, saved in replay_info.json (11-28, talk videos: no number without a file) ----
STATS_WARMUP_FRAMES = 20    # detector timings are also given without the first 20 frames (model load / first calls),
                            # the same warm-up time_trackers.py and flight_vs_bench.py skip
STATS_DEFINITIONS = {
    "target_lost_pct": "of the FOLLOW-mode frames from the first lock on, % whose state is not FOLLOWING (searching, "
                       "spinning, lost -> hover). Frames before the first lock are not counted.",
    "target_ids_used": "distinct track IDs the lock was on (the lock / click, then every re-ID re-lock). The pre-11-26 "
                       "replay auto re-locked onto the nearest person, so its 'target IDs used' counted something else.",
    "person_track_ids": "distinct tracker IDs given to person boxes over the whole replay (after the per-class cutoffs): "
                        "fewer = the tracker kept people under one ID for longer",
    "det_ms": "Detector.process's model.track wall time per frame (pre-process + inference + NMS + tracker), on this "
              "computer; 'after_warmup' leaves out the first STATS_WARMUP_FRAMES frames. Not the re-ID embedding (embed_ms).",
    "lock_events": "every change of the replay's target (lock = --lock-id / recorded click / window click; re-ID re-lock "
                   "with its similarity; cleared = N) with frame, flight time t_s and the target box",
    "target_segments": "consecutive frames on one target ID: start / end, how it started, seconds and frames FOLLOWING",
}


def _no_target(tid):
    return tid is None or tid == ""


def _ms_stats(xs):
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return None
    return dict(n=len(xs), mean=round(sum(xs) / len(xs), 1), median=round(xs[len(xs) // 2], 1),
                p95=round(xs[min(int(0.95 * len(xs)), len(xs) - 1)], 1))


def lock_events(rows):
    """commands.csv rows -> every change of target: [dict(frame, t_s, kind, from_id, to_id, sim, box)].
    kind: 'lock' (a pick: --lock-id, recorded click or window click), 're-ID re-lock', 'cleared' (N)."""
    out, prev = [], None
    for r in rows:
        tid = None if _no_target(r["target_id"]) else r["target_id"]
        relock = r.get("lock_event") == "reacquired_reid"
        if tid != prev or relock:
            kind = "re-ID re-lock" if relock else "cleared" if tid is None else "lock"
            box = [r.get(k) for k in ("target_x1", "target_y1", "target_x2", "target_y2")]
            out.append(dict(frame=r["frame"], t_s=r["t_s"], kind=kind, from_id=prev, to_id=tid,
                            sim=r.get("best_sim") if relock else None, box=box if box[0] not in (None, "") else None))
        prev = tid
    return out


def target_segments(rows):
    """commands.csv rows -> [dict(target_id, start_frame, start_t_s, end_frame, end_t_s, started_by, sim, frames,
    following_frames, following_s)]: one per run of consecutive frames on the same target ID. following_s sums each
    FOLLOWING frame's time to the next frame (the last frame of the replay counts 0)."""
    segs, cur = [], None
    for i, r in enumerate(rows):
        tid = None if _no_target(r["target_id"]) else r["target_id"]
        relock = r.get("lock_event") == "reacquired_reid"
        if cur is not None and (tid != cur["target_id"] or relock):
            segs.append(cur); cur = None
        if tid is not None and cur is None:
            cur = dict(target_id=tid, start_frame=r["frame"], start_t_s=r["t_s"], end_frame=r["frame"],
                       end_t_s=r["t_s"], started_by="re-ID re-lock" if relock else "lock",
                       sim=r.get("best_sim") if relock else None, frames=0, following_frames=0, following_s=0.0)
        if cur is not None:
            cur["end_frame"], cur["end_t_s"], cur["frames"] = r["frame"], r["t_s"], cur["frames"] + 1
            if r["state"] == "FOLLOWING":
                cur["following_frames"] += 1
                if i + 1 < len(rows):
                    cur["following_s"] += max(float(rows[i + 1]["t_s"]) - float(r["t_s"]), 0.0)
    if cur is not None:
        segs.append(cur)
    for s in segs:
        s["following_s"] = round(s["following_s"], 2)
    return segs


def replay_stats(rows, person_ids_seen=(), track_ids_seen=()):
    """The numbers a comparison replay is judged on (STATS_DEFINITIONS), from commands.csv rows + the track IDs seen."""
    first = next((i for i, r in enumerate(rows) if not _no_target(r["target_id"])), None)
    after = [r for r in rows[first:] if r["mode"] == "FOLLOW"] if first is not None else []
    lost = sum(r["state"] != "FOLLOWING" for r in after)
    used = []
    for r in rows:
        if not _no_target(r["target_id"]) and r["target_id"] not in used:
            used.append(r["target_id"])
    det = [r.get("det_ms") for r in rows]
    return dict(first_lock_frame=rows[first]["frame"] if first is not None else None,
                first_lock_t_s=rows[first]["t_s"] if first is not None else None,
                follow_frames_after_first_lock=len(after), not_following_after_first_lock=lost,
                target_lost_pct=round(100.0 * lost / len(after), 1) if after else None,
                target_ids_used=len(used), target_ids=used,
                person_track_ids=len(set(person_ids_seen)), all_track_ids=len(set(track_ids_seen)),
                det_ms=_ms_stats(det), det_ms_after_warmup=_ms_stats(det[STATS_WARMUP_FRAMES:]),
                embed_ms=_ms_stats([r.get("embed_ms") for r in rows]))


def events_text(events):
    """lock_events -> lines like the 11-22 replays' *_events.txt ('1:01.6  RE-LOCK by re-ID: track 78, score 0.70')."""
    def mmss(t):
        return f"{int(float(t) // 60)}:{float(t) % 60:04.1f}"
    out = []
    for e in events:
        if e["kind"] == "lock":
            out.append(f"{mmss(e['t_s'])}  LOCK on track {e['to_id']}  (frame {e['frame']})")
        elif e["kind"] == "re-ID re-lock":
            sim = f", score {float(e['sim']):.3f}" if e["sim"] not in (None, "") else ""
            out.append(f"{mmss(e['t_s'])}  RE-LOCK by re-ID: track {e['from_id']} -> {e['to_id']}{sim}  (frame {e['frame']})")
        else:
            out.append(f"{mmss(e['t_s'])}  CLEARED (N), was track {e['from_id']}  (frame {e['frame']})")
    return "\n".join(out) + ("\n" if out else "")


def build_parser():
    lf = _flight()
    ap = argparse.ArgumentParser(description="Replay footage through the flight program without flying.")
    ap.add_argument("--source", required=True)
    lf.add_flight_args(ap)             # the SAME flags and defaults as live_follow.py; no replay-only settings
    ap.add_argument("--lock-id", default="",
                    help="TID, TID@FRAME or TID@SECONDSs: lock onto replay track TID (replaces the flight log's recorded "
                         "locks); only@SECONDSs = the only person in view then (several people: no lock, IDs printed)")
    ap.add_argument("--label", default="",
                    help="caption stamped on every frame (e.g. 'cutoff 0.79') for a deliberate comparison run. With "
                         "--off-decisions it replaces the red DEVIATION banner on the frames, and only if it names every "
                         "deviating value; the console banners, replay_info.json and commands.csv keep the deviation")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate for sources without recorded timestamps (t_s)")
    ap.add_argument("--out", default=None, help="output folder (default: <source>/replay)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--window", action="store_true",
                    help="live window: CLICK a box to lock (like live), SPACE pause, Q quit (runs at detection speed)")
    return ap


def main(argv=None):
    import sys
    import lockon
    lf = _flight()
    args = build_parser().parse_args(argv)
    src = Path(args.source).expanduser()
    if not src.exists():
        sys.exit(f"STOP: no such source: {src}")
    cut_p, cut_v, cut_src = lf.resolve_detection(args)
    prec = lf.detector_precision(args.model, args.imgsz, fake=args.fake_detector)   # same reading as live (11-27)
    trk = lf.resolve_tracker(args.tracker)
    # ---- the same check as live_follow's pre-flight: what replay is about to run vs. flight_decisions.json ----
    modes = {"replay"} | ({"--fake-detector"} if args.fake_detector else set())
    actual = lf.running_settings(args, drone_profile_applied=False, cutoffs=(cut_p, cut_v), precision=prec["precision"])
    preflight = lf.enforce_decisions(lf.check_decisions(actual, lf.load_decisions(), modes), args.off_decisions,
                                     program="replay")
    if args.label and label_problem(args.label, preflight["deviations"]):
        sys.exit(f"STOP: {label_problem(args.label, preflight['deviations'])}")
    print("KNOWN OFFLINE DEVIATIONS (replay cannot do these the way the flight does):")
    for x in OFFLINE_DEVIATIONS:
        print("  -", x)
    for line in lf.precision_lines(prec, args.model):
        print(line)
    print(f"cutoffs: person >= {cut_p:.2f}, vehicle >= {cut_v:.2f}  ({cut_src})")
    print(lf.tracker_line(trk))
    embedder, reid_note = lockon.make_embedder(args.reid)
    print(reid_note)
    det = lf.Detector(args.model, trk["path"], args.imgsz, (cut_p, cut_v), fake=args.fake_detector, embedder=embedder)
    names = det.names
    person_ids = {k for k, v in names.items() if str(v).lower() in PERSON_NAMES}
    logic = lf.make_follow_logic(person_ids, args)
    det.want_fn = lambda tr, t, W, H: logic.wants_embedding(tr, t, W, H)

    # ---- where the lock comes from: never automatic (click-to-lock, Michael 11-20) ----
    flight = load_flight_log(src)
    lock_tid, lock_frame, lock_t = parse_lock_id(args.lock_id)
    events, notes, lock_placed = [], [], None
    if lock_tid is not None:
        lock_source = f"--lock-id {args.lock_id}"
    elif flight is None:
        lock_source = "none recorded (not a live_follow flight): use --lock-id or --window"
    else:
        events = recorded_locks(flight["rows"])
        if events is None:
            lock_source = "none: this follow.csv has no lock_event column (before 11-10); use --lock-id or --window"
            events = []
        elif flight["boxes"] is None:
            lock_source = "none: no boxes.jsonl (flights before 11-22), the recorded lock can't be placed; use --lock-id or --window"
            events = []
        else:
            lock_source = f"flight log: {sum(e[1] == 'pick' for e in events)} recorded lock(s), {sum(e[1] == 'reset' for e in events)} N"
    print(f"lock: {lock_source}" + ("   + clicks in the window" if args.window else ""))
    flight_differs = []
    if flight and flight["running"]:
        flight_differs = [dict(name=k, flown=v, replay=actual[k]) for k, v in flight["running"].items()
                          if k in actual and k != "drone_profile_applied" and v != actual[k]]
        if flight_differs:
            notes.append("this flight was flown with other settings than what flies now (replay runs what flies now): "
                         + "; ".join(f"{d['name']} {d['flown']!r} -> {d['replay']!r}" for d in flight_differs))
    for n_ in notes:
        print("NOTE:", n_)
    notes = list(notes)

    out = Path(args.out) if args.out else (src if src.is_dir() else src.parent) / "replay"
    out.mkdir(parents=True, exist_ok=True)
    banner = ("DEVIATION (--off-decisions): " + ", ".join(f"{x['name']} {x['actual']!r} (flown {x['decided']!r})"
                                                         for x in preflight["deviations"])) if preflight["deviations"] else ""
    # what the frames show on top of the flight view: the label (a deliberate comparison run), else the red banner
    frame_caption = ("label" if args.label else "deviation banner" if banner else "none")
    info_doc = dict(program="replay.py", source=str(src), running=actual, preflight=preflight,
                    offline_deviations=OFFLINE_DEVIATIONS, cutoffs=dict(person=cut_p, vehicle=cut_v, source=cut_src),
                    reid=reid_note, lock_source=lock_source, flight_differs=flight_differs, notes=notes,
                    detector=prec, tracker=dict(file=trk["label"], settings=trk["settings"]),
                    label=args.label or None, frame_caption=frame_caption)
    (out / "replay_info.json").write_text(json.dumps(info_doc, indent=1, default=str))

    writer, rows, t_start = None, [], time.time()
    total = count_frames(src)
    print(f"{total} frames to process...")
    show, click, paused = args.window, {"pt": None}, False
    if show:
        cv2.namedWindow("replay", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("replay", lambda ev, x, y, fl, p: click.__setitem__("pt", (x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    ei, pending, lock_done, prev_mode, placed, n = 0, None, False, None, 0, 0
    person_ids_seen, track_ids_seen = set(), set()
    for i, (img, row) in enumerate(frames_from(src)):
        H, W = img.shape[:2]
        f = int(row["frame"]) if row and row.get("frame") not in (None, "") else i
        now = _f(row, "t_s")
        now = i / args.fps if now is None else now
        tel = {k: _f(row, k) for k in row} if row else {}
        tracks, det_ms, vecs, embed_ms = det.process(img, now)      # live_follow's own detector pass
        track_ids_seen.update(int(t[4]) for t in tracks)
        person_ids_seen.update(int(t[4]) for t in tracks if t[5] in person_ids)
        # ---- operator input, as live_follow handles it before the logic runs: N, clicks, M ----
        while ei < len(events) and events[ei][0] <= f:
            _, kind, rec_tid = events[ei]; ei += 1
            if kind == "reset":
                logic.reset(); pending = None
            else:
                pending = rec_tid
        if pending is not None:                                        # recorded lock: place it by box overlap
            rec_box = next((b[:4] for b in flight["boxes"].get(f, []) if str(b[4]) == pending), None)
            hit = place_lock(rec_box, tracks) if rec_box is not None else None
            if hit is not None:
                logic.pick(hit[4], tracks, now=now); pending = None; placed += 1
        if lock_tid is not None and not lock_done and (f >= lock_frame if lock_t is None else now >= lock_t):
            hit, people = lock_target(lock_tid, tracks, person_ids)
            if lock_tid == ONLY and len(people) > 1:                   # several people: never choose between them
                lock_done = True
                lock_placed = dict(frame=f, t_s=round(now, 3), track=None, people_in_view=len(people),
                                   ids=[int(t[4]) for t in people])
                notes.append(f"--lock-id {args.lock_id}: {len(people)} people in view at frame {f} "
                             f"(t {now:.1f} s), so NOTHING was locked. Pick one: --lock-id TID@{f} with TID one of "
                             + ", ".join(f"{int(t[4])} (box x {t[0]:.0f}-{t[2]:.0f}, conf {t[6]:.2f})" for t in people))
                print("\nNOTE:", notes[-1])
            if hit is not None:
                logic.pick(hit[4], tracks, now=now); lock_done = True; placed += 1
                lock_placed = dict(frame=f, t_s=round(now, 3), track=int(hit[4]),
                                   people_in_view=sum(t[5] in person_ids for t in tracks))
                print(f"\nlock placed: replay track {int(hit[4])} at frame {f} (t {now:.1f} s)"
                      + (" - the only person in view" if lock_tid == ONLY else ""))
        clicked = False
        while True:                                                     # window clicks (also while paused)
            if click["pt"] is not None:
                hit = box_at(tracks, *click["pt"]); click["pt"] = None
                if hit is not None:
                    logic.pick(hit[4], tracks, now=now); clicked = True
            if not (show and paused):
                break
            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                paused = False
            elif k == ord("q"):
                total = n; break
        mode = (row.get("mode") or "MANUAL") if flight else "FOLLOW"
        if mode == "FOLLOW" and prev_mode is not None and prev_mode != "FOLLOW":
            logic.resume(now=now)                                       # M: keep target + gallery
        prev_mode = mode
        if mode == "FOLLOW":
            cmd, info = logic.step(tracks, now, now, W, H, tel, vecs)
        else:                                                           # MANUAL / LOOKGO: lock-on runs, no follow commands
            ev = logic.observe(tracks, now, now, W, H, vecs)
            cmd, info = dict(roll=0, pitch=0, yaw=0, gaz=0), dict(state=mode, dist=None, method=mode.lower(), event=ev)
        vis = lf.draw_follow(img, tracks, names, logic, cmd, info, mode, 1000 / max(det_ms, 1), tel, n + 1, total)
        if args.label:                                                  # comparison run: a neutral caption
            cv2.rectangle(vis, (0, 42), (W, 72), (40, 40, 40), -1)
            cv2.putText(vis, args.label[:150], (8, 64), 0, 0.7, (255, 255, 255), 2)
        elif banner:
            cv2.rectangle(vis, (0, 42), (W, 72), (0, 0, 160), -1)
            cv2.putText(vis, banner[:150], (8, 64), 0, 0.6, (255, 255, 255), 2)
        if writer is None:
            writer = cv2.VideoWriter(str(out / "replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        writer.write(vis)
        if show:
            cv2.imshow("replay", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                paused = True
            elif k == ord("q"):
                break
        g = lambda key: (row.get(key, "") if row else "")
        tb = next((t[:4] for t in tracks if logic.target_id is not None and t[4] == logic.target_id), None)
        rows.append(dict(frame=f, t_s=round(now, 3), mode=mode, state=info["state"], target_id=logic.target_id,
                         picked_by=logic.picked_by, clicked=int(clicked), lock_event=info.get("event", ""),
                         dist_m=info.get("dist"), method=info.get("method"), roll=cmd["roll"],
                         pitch=round(cmd["pitch"], 1), yaw=round(cmd["yaw"], 1), gaz=cmd["gaz"],
                         gimbal_delta=round(info.get("gimbal_delta") or 0.0, 2), n_tracks=len(tracks),
                         gallery_n=len(logic.lock.gallery), n_embedded=len(vecs),
                         best_sim=logic.lock.last_sim if info.get("event") == "reacquired_reid" else None,
                         det_ms=round(det_ms, 1), embed_ms=round(embed_ms, 1),
                         **{k: (round(float(v), 1) if tb is not None else None)
                            for k, v in zip(("target_x1", "target_y1", "target_x2", "target_y2"), tb or (0, 0, 0, 0))},
                         flight_mode=g("mode"), flight_state=g("state"), flight_target_id=g("target_id"),
                         flight_lock_event=g("lock_event"), flight_yaw=g("yaw"), flight_pitch=g("pitch"),
                         deviation=banner))
        n += 1
        if n % 10 == 0 or n == total:
            el = time.time() - t_start; rate = n / max(el, 1e-6)
            eta = (total - n) / max(rate, 1e-6) if total else 0
            bar = "#" * int(30 * n / max(total, 1))
            print(f"\r[{bar:<30}] {n}/{total} ({100*n/max(total,1):.0f}%) | {rate:.1f} fps | ETA {eta/60:.1f} min ", end="", flush=True)
        if args.max_frames and n >= args.max_frames:
            break
    print()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    if rows:
        with open(out / "commands.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = dict(frames=n, following=sum(r_["state"] == "FOLLOWING" for r_ in rows),
                   locks_placed=placed, window_clicks=sum(r_["clicked"] for r_ in rows),
                   reid_relocks=sum(r_["lock_event"] == "reacquired_reid" for r_ in rows),
                   lost_or_searching=sum(r_["method"] in ("lost", "searching") for r_ in rows),
                   no_target=sum(r_["method"] == "no_target" for r_ in rows))
    summary.update(replay_stats(rows, person_ids_seen, track_ids_seen))      # 11-28: the comparison numbers, saved
    events, segments = lock_events(rows), target_segments(rows)
    (out / "lock_events.txt").write_text(events_text(events))
    if pending is not None:
        notes.append(f"a recorded lock (flight target {pending}) never matched a replay track (IoU >= {LOCK_IOU})")
    if lock_tid is not None and lock_placed is None:
        notes.append(f"--lock-id {args.lock_id}: never placed (no matching track at or after that point)")
    (out / "replay_info.json").write_text(json.dumps(dict(info_doc, notes=notes, summary=summary, lock=lock_placed,
                                                          summary_definitions=STATS_DEFINITIONS, lock_events=events,
                                                          target_segments=segments), indent=1, default=str))
    short = {k: v for k, v in summary.items() if k not in ("target_ids", "det_ms", "det_ms_after_warmup", "embed_ms")}
    dm = summary["det_ms_after_warmup"] or summary["det_ms"]
    print(" | ".join(f"{k} {v}" for k, v in short.items())
          + (f" | det_ms mean {dm['mean']} median {dm['median']}" if dm else "")
          + f"\nsaved {out / 'replay.mp4'} + commands.csv + replay_info.json + lock_events.txt")
    print(events_text(events), end="")
    for n_ in notes:
        print("NOTE:", n_)
    if banner:
        print("#" * 78 + f"\n{banner}\n  This replay did NOT run what is flown.\n" + "#" * 78)
    return out


if __name__ == "__main__":
    import replay as _replay     # run as the module live_follow imports, so there is ONE copy of Controller/FOLLOW_DIST_M
    _replay.main()
