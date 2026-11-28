#!/usr/bin/env python3
"""Two videos side by side, aligned in time, each with a label bar (name + time). For the talk (Michael 11-27):

  python side_by_side.py ~/follow_logs/20261122_162719_droneA ~/follow_logs/20261122_162740_droneB --out both.mp4
  python side_by_side.py ~/talk_videos/reid_cutoff_070 ~/talk_videos/reid_cutoff_079 --out cutoffs.mp4

Each input is one of:
  * a live_follow.py run folder (follow.csv + video_*.mp4 chunks): every follow.csv row is one recorded frame
    (video_file, frame_in_file, t_s). Its clock: the run's start time + t_s. The start time is the folder name
    (<YYYYMMDD_HHMMSS>, live_follow names the folder a few ms before t_s = 0, so it is up to 1 s early), refined to
    follow.csv's modification time minus its last t_s when that lands within 2 s after the folder time (live_follow
    flushes follow.csv on every row, so its mtime is the last row's wall clock; a copied folder loses it, then the
    folder time is used). Which one was used is printed. Label: the folder's drone suffix ("drone A").
  * a replay.py output folder (replay.mp4 + commands.csv), or that replay.mp4: commands.csv row i is frame i, at the
    flight's recorded t_s, so two replays of the same flight line up frame for frame. Label: replay_info.json's
    --label if it has one, else the folder name.
  * any other video file: frame i at i / its fps.
Two run folders are aligned by wall clock (two drones flown from one laptop share its clock); anything else by its own
t (flight time for replays, 0 = first frame for plain videos). --offset shifts the RIGHT input by that many seconds.

Output: one frame every 1/--fps s over the time both inputs cover (--span overlap, default) or either covers
(--span union: a side with nothing yet / any more shows "not recording"). Each side shows its latest frame at or
before that time (no interpolation). Encoder: ffmpeg (H.264, yuv420p: plays in PowerPoint / browsers) when the
`ffmpeg` command is on PATH, else OpenCV's mp4v (plays in VLC); --encoder forces one.
"""
import argparse
import bisect
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BAR_H = 44                    # label bar above each panel
GAP = 8                       # black gap between the panels
MTIME_WINDOW_S = 2.0          # follow.csv mtime - last t_s is trusted only this close after the folder-name time
_RUN_NAME = re.compile(r"^(\d{8})_(\d{6})(?:_(.+))?$")


def drone_label(suffix):
    """'droneA' -> 'drone A'; anything else as is."""
    m = re.fullmatch(r"drone([A-Za-z0-9])", suffix or "")
    return f"drone {m.group(1).upper()}" if m else (suffix or "")


def folder_time(name):
    """Run folder name -> local epoch seconds of <YYYYMMDD_HHMMSS>, or None."""
    m = _RUN_NAME.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return None


class Source:
    """frames: [(t, video_path, index_in_file)] sorted by t. anchor: epoch seconds of t = 0, or None (no wall clock)."""
    def __init__(self, path, kind, frames, anchor=None, anchor_note="", label=""):
        self.path, self.kind, self.frames, self.anchor, self.anchor_note, self.label = (
            path, kind, frames, anchor, anchor_note, label)
        self.ts = [f[0] for f in frames]
        if not frames:
            raise SystemExit(f"STOP: no frames in {path}")
        gaps = [b - a for a, b in zip(self.ts, self.ts[1:]) if b > a]
        self.frame_dt = float(np.median(gaps)) if gaps else 1 / 30


def load_run(p, label=None):
    rows = list(csv.DictReader(open(p / "follow.csv", newline="")))
    frames, count = [], {}
    for r in rows:
        try:
            t = float(r["t_s"])
        except (KeyError, TypeError, ValueError):
            continue
        vf = r.get("video_file") or ""
        if not vf:
            continue
        idx = r.get("frame_in_file")
        idx = int(idx) if idx not in (None, "") else count.get(vf, 0)   # older logs: count rows per chunk
        count[vf] = idx + 1
        frames.append((t, p / vf, idx))
    frames.sort(key=lambda f: f[0])
    ft = folder_time(p.name)
    anchor, note = ft, "start = folder name"
    if frames:
        refined = (p / "follow.csv").stat().st_mtime - max(f[0] for f in frames)
        if ft is not None and 0.0 <= refined - ft < MTIME_WINDOW_S:
            anchor, note = refined, f"start = follow.csv mtime - last t_s ({refined - ft:+.3f} s from the folder name)"
        elif ft is None:
            note = "no start time in the folder name: aligned by t_s from 0"
    m = _RUN_NAME.match(p.name)
    return Source(p, "run", frames, anchor, note, label or drone_label(m.group(3) if m else "") or p.name)


def load_replay(mp4, label=None):
    folder = mp4.parent
    rows = list(csv.DictReader(open(folder / "commands.csv", newline="")))
    frames = [(float(r["t_s"]), mp4, i) for i, r in enumerate(rows)]
    if not label and (folder / "replay_info.json").exists():
        try:
            label = json.loads((folder / "replay_info.json").read_text()).get("label")
        except ValueError:
            label = None
    return Source(mp4, "replay", frames, None, "replay: aligned by the flight's recorded t_s", label or folder.name)


def load_video(p, label=None):
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return Source(p, "video", [(i / fps, p, i) for i in range(n)], None, f"video: frame / {fps:g} fps", label or p.stem)


def load_source(path, label=None):
    p = Path(path).expanduser()
    if p.is_dir() and (p / "follow.csv").exists():
        return load_run(p, label)
    if p.is_dir() and (p / "replay.mp4").exists() and (p / "commands.csv").exists():
        return load_replay(p / "replay.mp4", label)
    if p.is_file() and (p.parent / "commands.csv").exists() and p.suffix.lower() == ".mp4":
        return load_replay(p, label)
    if p.is_file():
        return load_video(p, label)
    raise SystemExit(f"STOP: {p}: not a run folder (follow.csv), a replay folder (replay.mp4 + commands.csv) or a video")


def clocks(a, b, offset=0.0):
    """-> (base_a, base_b, wall): add base_x to source x's t to put both on one clock. wall: True if that clock is the
    wall clock (both are run folders with a start time)."""
    if a.anchor is not None and b.anchor is not None:
        return a.anchor, b.anchor + offset, True
    return 0.0, offset, False


def plan(a, b, fps=30.0, span="overlap", start=0.0, duration=0.0, offset=0.0):
    """The output timeline. -> (times, picks_a, picks_b, wall): times on the common clock; picks_x[i] = index into
    x.frames shown at times[i] (the latest frame at or before it), or None when x is not recording then."""
    ba, bb, wall = clocks(a, b, offset)
    lo_a, hi_a = a.ts[0] + ba, a.ts[-1] + ba + a.frame_dt
    lo_b, hi_b = b.ts[0] + bb, b.ts[-1] + bb + b.frame_dt
    lo, hi = (max(lo_a, lo_b), min(hi_a, hi_b)) if span == "overlap" else (min(lo_a, lo_b), max(hi_a, hi_b))
    if hi <= lo:
        raise SystemExit("STOP: the two inputs never record at the same time (try --span union or --offset)")
    lo += start
    if duration:
        hi = min(hi, lo + duration)
    n = max(int((hi - lo) * fps + 1e-9), 0)
    times = [lo + i / fps for i in range(n)]

    def picks(src, base, lo_x, hi_x):
        out = []
        for t in times:
            if t < lo_x or t >= hi_x:
                out.append(None)
            else:
                out.append(max(bisect.bisect_right(src.ts, t - base + 1e-6) - 1, 0))   # 1e-6: float noise
        return out
    return times, picks(a, ba, lo_a, hi_a), picks(b, bb, lo_b, hi_b), wall


class Reader:
    """Sequential frame access across a source's video files (forward reads; reopens to go back)."""
    def __init__(self):
        self.cap, self.file, self.pos, self.last = None, None, 0, None

    def get(self, path, idx):
        if path != self.file or idx < self.pos - 1:
            if self.cap is not None:
                self.cap.release()
            self.cap, self.file, self.pos, self.last = cv2.VideoCapture(str(path)), path, 0, None
        if idx == self.pos - 1 and self.last is not None:
            return self.last
        while self.pos < idx:                                  # skip forward
            if not self.cap.grab():
                return self.last
            self.pos += 1
        ok, img = self.cap.read()
        if ok:
            self.pos += 1
            self.last = img
        return self.last                                       # an unreadable frame holds the previous one

    def close(self):
        if self.cap is not None:
            self.cap.release()


def fmt_t(s):
    sign, s = ("-" if s < 0 else ""), abs(s)
    return f"{sign}{int(s // 60)}:{s % 60:04.1f}"


def panel(img, height, width, label, right_text, missing_text=""):
    """One side: label bar + the frame scaled to `height` (letterboxed to `width`)."""
    out = np.zeros((BAR_H + height, width, 3), np.uint8)
    out[:BAR_H] = (30, 30, 30)
    cv2.putText(out, label, (10, BAR_H - 14), 0, 0.8, (255, 255, 255), 2)
    (tw, _), _ = cv2.getTextSize(right_text, 0, 0.6, 1)
    cv2.putText(out, right_text, (width - tw - 10, BAR_H - 15), 0, 0.6, (220, 220, 220), 1)
    if img is None:
        (tw, _), _ = cv2.getTextSize(missing_text, 0, 0.9, 2)
        cv2.putText(out, missing_text, (max((width - tw) // 2, 5), BAR_H + height // 2), 0, 0.9, (160, 160, 160), 2)
        return out
    h, w = img.shape[:2]
    nw = min(int(round(w * height / h)), width)
    out[BAR_H:, (width - nw) // 2:(width - nw) // 2 + nw] = cv2.resize(img, (nw, height))
    return out


def ffmpeg_cmd(out, W, H, fps):
    return ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
            "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out)]


def pick_encoder(choice):
    if choice == "auto":
        return "ffmpeg" if shutil.which("ffmpeg") else "opencv"
    if choice == "ffmpeg" and not shutil.which("ffmpeg"):
        raise SystemExit("STOP: --encoder ffmpeg, but no `ffmpeg` on PATH (sudo apt install ffmpeg, or --encoder opencv)")
    return choice


def render(a, b, out, fps=30.0, span="overlap", start=0.0, duration=0.0, offset=0.0, height=720, encoder="auto"):
    times, pa, pb, wall = plan(a, b, fps, span, start, duration, offset)
    ba, bb, _ = clocks(a, b, offset)
    h0 = [cv2.VideoCapture(str(s.frames[0][1])) for s in (a, b)]
    dims = [(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for c in h0]
    for c in h0:
        c.release()
    height = min(height, *(d[1] for d in dims if d[1])) // 2 * 2 or height
    widths = [max(int(round(w * height / h)), 2) if h else height * 16 // 9 for w, h in dims]
    pw = max(widths) // 2 * 2                                   # both panels the same width
    W, H = pw * 2 + GAP, BAR_H + height
    enc = pick_encoder(encoder)
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    if enc == "ffmpeg":
        proc = subprocess.Popen(ffmpeg_cmd(out, W, H, fps), stdin=subprocess.PIPE)
        write = proc.stdin.write
    else:
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        write = lambda buf: vw.write(np.frombuffer(buf, np.uint8).reshape(H, W, 3))  # noqa: E731
    readers = (Reader(), Reader())
    try:
        for i, T in enumerate(times):
            sides = []
            for src, picks, base, rd in ((a, pa, ba, readers[0]), (b, pb, bb, readers[1])):
                k = picks[i]
                if k is None:
                    before = T < src.ts[0] + base
                    sides.append(panel(None, height, pw, src.label, "", "not recording yet" if before else "recording ended"))
                    continue
                t, path, idx = src.frames[k]
                img = rd.get(path, idx)
                txt = (f"{datetime.fromtimestamp(T).strftime('%H:%M:%S.%f')[:10]}  (t {fmt_t(t)})" if wall
                       else f"t {fmt_t(t)}")
                sides.append(panel(img, height, pw, src.label, txt, "unreadable frame"))
            frame = np.zeros((H, W, 3), np.uint8)
            frame[:, :pw], frame[:, pw + GAP:] = sides[0], sides[1]
            write(frame.tobytes())
            if (i + 1) % 300 == 0 or i + 1 == len(times):
                print(f"\r{i + 1}/{len(times)} frames ({100 * (i + 1) / max(len(times), 1):.0f}%)", end="", flush=True)
    finally:
        for rd in readers:
            rd.close()
        if enc == "ffmpeg":
            proc.stdin.close()
            if proc.wait() != 0:
                raise SystemExit("STOP: ffmpeg failed (see above); retry with --encoder opencv")
        else:
            vw.release()
    print()
    return dict(out=str(out), frames=len(times), encoder=enc, wall=wall, size=(W, H),
                start=times[0] if times else None, seconds=len(times) / fps)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Two videos side by side, aligned in time, with labels.")
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--labels", nargs=2, metavar=("LEFT", "RIGHT"), default=None,
                    help="labels (default: drone A / drone B from a run folder name, a replay's --label, else the name)")
    ap.add_argument("--span", choices=["overlap", "union"], default="overlap",
                    help="overlap (default): only while both record; union: from the first start to the last end")
    ap.add_argument("--start", type=float, default=0.0, help="skip this many seconds from the start of the span")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to keep (0 = to the end)")
    ap.add_argument("--offset", type=float, default=0.0, help="shift the RIGHT input by this many seconds (nudge by eye)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--height", type=int, default=720, help="panel height in pixels (never upscaled)")
    ap.add_argument("--encoder", choices=["auto", "ffmpeg", "opencv"], default="auto")
    args = ap.parse_args(argv)
    labels = args.labels or (None, None)
    a, b = load_source(args.left, labels[0]), load_source(args.right, labels[1])
    _, _, wall = clocks(a, b, args.offset)
    for s in (a, b):
        when = (f", starts {datetime.fromtimestamp(s.anchor).strftime('%Y-%m-%d %H:%M:%S.%f')[:23]}"
                if s.anchor is not None and wall else "")
        print(f"{s.label}: {s.path} ({s.kind}, {len(s.frames)} frames, t {fmt_t(s.ts[0])}-{fmt_t(s.ts[-1])}{when}; {s.anchor_note})")
    if not wall:
        print("aligned by each input's own t (not wall clock)" + (f", right shifted {args.offset:+g} s" if args.offset else ""))
    elif args.offset:
        print(f"right input shifted {args.offset:+g} s")
    r = render(a, b, args.out, args.fps, args.span, args.start, args.duration, args.offset, args.height, args.encoder)
    print(f"saved {r['out']}: {r['frames']} frames, {r['seconds']:.1f} s at {args.fps:g} fps, {r['size'][0]}x{r['size'][1]}, "
          f"encoder {r['encoder']}" + ("" if r["encoder"] == "ffmpeg" else " (mp4v: plays in VLC; for H.264 install ffmpeg in WSL: sudo apt install ffmpeg)"))
    return r


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
