#!/usr/bin/env python3
"""Replay recorded footage through  detector -> tracker -> follow-me controller,  WITHOUT flying.

Draws the commands the controller WOULD send (yaw / forward-back / up-down / gimbal) on every frame,
and saves an annotated video + a CSV of commands so we can check them before real flights.

  python replay.py --source ~/recordings/20261025_150000            # a record.py folder (uses telemetry.csv)
  python replay.py --source ~/follow_logs/20261029_150554           # a live_follow.py folder (uses follow.csv)
  python replay.py --source clip.mp4                                 # any video (distance from box size)
  python replay.py --source VisDrone2019-MOT-val/sequences/uav0000086_00000_v   # an image folder

Needs: pip install ultralytics  (separate env from Olympe is fine; this never talks to the drone).
Default: no window, just a progress bar, then a normal-speed replay.mp4 to watch.
--window: live window where you can CLICK a box to make it the target (plays at detection speed), SPACE pause, Q quit.

Controller (P-control with deadbands, clamps and smoothing; all gains are first guesses, to be tuned):
  yaw    <- horizontal offset of the target from image center
  gimbal <- vertical offset (camera tilts to keep the target centered)
  pitch  <- distance error (forward if too far, back if too close)
  gaz    <- 0 for now (hold altitude)
  lost target > LOST_S seconds -> hover (all zero)
Distance:
  'angle' (needs telemetry): ground distance = height_above_takeoff / tan(angle down to the target's FEET)
  'box'   (fallback): pinhole estimate from box height, assuming a 1.7 m person
"""
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

# ---- tunables (ASSUMED first guesses) ----
VFOV_DEG = 42.0          # ANAFI live stream vertical field of view, from ~69 deg HFOV at 16:9 (ASSUMED, verify)
PERSON_H_M = 1.7
FOLLOW_DIST_M = 6.0
K_YAW, K_PITCH, K_GIMBAL = 60.0, 12.0, 0.25    # % per unit offset, % per metre, deg per deg of offset
MAX_YAW, MAX_PITCH = 40, 30                     # % stick
DEADBAND_X, DEADBAND_D = 0.05, 0.5              # fraction of half-width, metres
MIN_ANGLE_DEG = 3.0                             # below this the angle method is too unstable -> box method
SMOOTH = 0.5                                    # EMA on commands (0 = no smoothing)
LOST_S = 1.0


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


def pick_target(tracks, person_ids, last_box, W, H):
    """First lock: biggest person. Re-lock after a loss: person closest to where we last saw the target."""
    cands = [t for t in tracks if t[5] in person_ids]
    if not cands:
        return None
    if last_box is None:
        return max(cands, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
    lx, ly = (last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2
    return min(cands, key=lambda t: ((t[0] + t[2]) / 2 - lx) ** 2 + ((t[1] + t[3]) / 2 - ly) ** 2)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="yolov8n.pt", help=".pt or .onnx (ultralytics)")
    ap.add_argument("--tracker", default="botsort.yaml", help="ultralytics tracker yaml (botsort has camera-motion comp. on)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--fps", type=float, default=30.0, help="source frame rate (for the lost-target timer)")
    ap.add_argument("--out", default=None, help="output folder (default: next to source)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--window", action="store_true", help="show a live window you can click in (runs at detection speed)")
    args = ap.parse_args()

    from ultralytics import YOLO
    import time
    model = YOLO(args.model)
    names = model.names
    person_ids = {k for k, v in names.items() if str(v).lower() in PERSON_NAMES}
    src = Path(args.source).expanduser()
    out = Path(args.out) if args.out else (src if src.is_dir() else src.parent) / "replay"
    out.mkdir(parents=True, exist_ok=True)
    writer, ctrl = None, Controller()
    target_id, last_box, lost_frames, n, picked_by = None, None, 0, 0, "auto"
    rows, t_start = [], time.time()
    total = count_frames(src)
    print(f"{total} frames to process...")
    show = args.window
    click = {"pt": None}
    paused = False
    if show:
        cv2.namedWindow("replay", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("replay", lambda ev, x, y, fl, p: click.__setitem__("pt", (x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    for img, tel in frames_from(src):
        H, W = img.shape[:2]
        t0 = time.time()
        r = model.track(img, persist=True, tracker=args.tracker, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        tracks = []
        if r.boxes is not None and r.boxes.id is not None:
            for b, tid, cls, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int),
                                       r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
                tracks.append((*b.tolist(), int(tid), int(cls), float(cf)))
        clicked = False
        while True:                                   # handle clicks (also while paused)
            if click["pt"] is not None:
                hit = box_at(tracks, *click["pt"]); click["pt"] = None
                if hit is not None:
                    target_id, picked_by, lost_frames, clicked = hit[4], "you", 0, True
            if not (show and paused):
                break
            vis = draw(img.copy(), tracks, names, target_id, dict(ctrl.cmd, dist=None, method="searching"), 0, tel, n, total, picked_by, True)
            cv2.imshow("replay", vis)
            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                paused = False
            elif k == ord("q"):
                total = n; break
        tgt = next((t for t in tracks if t[4] == target_id), None) if target_id is not None else None
        if tgt is None:
            lost_frames += 1
            if target_id is None or lost_frames > 3:              # (re)lock automatically
                cand = pick_target([t[:6] for t in tracks], person_ids, last_box, W, H)
                if cand is not None:
                    target_id, tgt, lost_frames, picked_by = cand[4], cand, 0, "auto"
        else:
            lost_frames = 0
        box = tgt[:4] if tgt is not None else None
        if box is not None:
            last_box = box
        if box is not None:
            c = ctrl.step(box, W, H, tel)
        elif lost_frames / args.fps > LOST_S:
            c = ctrl.step(None, W, H, tel)                          # lost too long -> hover
        else:
            c = dict(ctrl.cmd, dist=None, method="searching")       # brief gap: keep last command
        fps = 1.0 / max(time.time() - t0, 1e-6)
        vis = draw(img.copy(), tracks, names, target_id, c, fps, tel, n + 1, total, picked_by)
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
        rows.append(dict(frame=n, target_id=target_id if box is not None else None, picked_by=picked_by, clicked=int(clicked),
                         dist_m=c["dist"], method=c["method"], yaw=round(c["yaw"], 1), pitch=round(c["pitch"], 1), gaz=0,
                         gimbal_delta=round(c["gimbal_delta"], 2), n_tracks=len(tracks), ms=round(1000 / fps, 1)))
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
    with open(out / "commands.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lost = sum(1 for r_ in rows if r_["method"] == "lost")
    ids = {r_["target_id"] for r_ in rows if r_["target_id"] is not None}
    print(f"{n} frames | target lost on {lost} ({100*lost/max(n,1):.0f}%) | target IDs used: {len(ids)} "
          f"| your clicks: {sum(r_['clicked'] for r_ in rows)} | avg {np.mean([r_['ms'] for r_ in rows]):.0f} ms/frame "
          f"| saved {out/'replay.mp4'} + commands.csv")


if __name__ == "__main__":
    main()
