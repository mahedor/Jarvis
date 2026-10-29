"""
JARVIS detector comparison - several YOLO models on the SAME frames, side by side
==================================================================================
Answers "does the fine-tuned model actually beat stock YOLO on MY drone?" without
labels. Every frame is run through every model; the window shows one tile per
model, and a per-frame CSV plus an end-of-run summary record what each model saw.

    # live, laptop on the Anafi's Wi-Fi (or SkyController USB: rtsp://192.168.53.1/live)
    python tools/compare_detectors.py --source rtsp://192.168.42.1/live \
        --models stock=yolov8n.pt visdrone=vd_gray_s42.pt cocoM=vd_coco_m_s42.pt

    # a clip you recorded with view_stream.py --record-raw (every frame, no drops)
    python tools/compare_detectors.py --source raw.mp4 --models ... --csv cmp.csv

Label-free metrics (there is no ground truth here, so these are proxies, not mAP):
  person_rate   share of frames with >= 1 person box. For follow-me footage where
                you are always in frame this is close to recall of "you".
  person_conf   mean confidence of the best person box when there is one.
  vehicles/fr   vehicle boxes per frame.
  infer_ms      model time per frame on this machine (each model in turn).
Compare models on the same footage; mark angle/height changes with the k key
(a 'segment' column in the CSV) so results can be split by viewing angle.

Stock COCO weights are mapped to our 8 classes by name (person, bicycle, car,
motorcycle, bus, truck); our weights already use them. A standalone tool:
nothing in demo/ imports it.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from view_stream import (  # noqa: E402  - reuse the tested capture path (timeouts, drop-to-latest)
    DEFAULT_TIMEOUT, SOURCE_FILE, FrameReader, RawRecorder, StreamError, StreamTimeout,
    is_rtsp, open_capture, parse_source,
)

OUR_CLASSES = ("person", "bicycle", "car", "van", "truck", "bus", "motorcycle", "three_wheeler")
VEHICLES = {"bicycle", "car", "van", "truck", "bus", "motorcycle", "three_wheeler"}
WINDOW = "JARVIS detector comparison"
COLORS = {"person": (0, 200, 255)}
VEHICLE_COLOR = (255, 160, 0)
HUD_H = 48                   # height of the name/stats strip above each tile
LABEL_CONF = 0.35            # boxes below this are drawn without a text label


def parse_models(items):
    """['stock=yolov8n.pt', 'ft=best.pt'] -> [('stock', Path), ('ft', Path)]."""
    out = []
    for item in items:
        name, sep, path = item.partition("=")
        if not sep:
            name, path = Path(item).stem, item
        out.append((name, Path(path)))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique: {names}")
    return out


def class_map(model_names):
    """Model class id -> our class name (None = ignore). Works for COCO and ours."""
    return {i: (n if n in OUR_CLASSES else None) for i, n in model_names.items()}


class Model:
    def __init__(self, name, weights, conf, imgsz, device):
        from ultralytics import YOLO
        self.name, self.conf, self.imgsz, self.device = name, conf, imgsz, device
        self.yolo = YOLO(str(weights))
        self.map = class_map(self.yolo.names)
        self.kind = "coco" if len(self.yolo.names) == 80 else f"{len(self.yolo.names)}-class"

    def detect(self, frame):
        t0 = time.perf_counter()
        r = self.yolo.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
                              verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000.0
        dets = []
        if r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                             r.boxes.cls.tolist()):
                label = self.map.get(int(k))
                if label:
                    dets.append((label, float(c), (int(x1), int(y1), int(x2), int(y2))))
        return dets, ms


def frame_stats(dets):
    persons = [c for lab, c, _ in dets if lab == "person"]
    return {
        "n_person": len(persons),
        "best_person_conf": max(persons) if persons else 0.0,
        "n_vehicle": sum(1 for lab, _, _ in dets if lab in VEHICLES),
    }


def draw_tile(frame, name, dets, ms, width):
    """Shrink the frame to tile size FIRST, then draw, so text stays readable at any tile width
    (drawing on the full 1280-px frame and shrinking afterwards made labels unreadably small)."""
    scale = width / frame.shape[1]
    tile = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for lab, c, (x1, y1, x2, y2) in dets:
        col = COLORS.get(lab, VEHICLE_COLOR)
        x1, y1, x2, y2 = (int(v * scale) for v in (x1, y1, x2, y2))
        cv2.rectangle(tile, (x1, y1), (x2, y2), col, 2)
        if c >= LABEL_CONF:                       # only label confident boxes; the rest would be clutter
            txt = f"{'P' if lab == 'person' else lab[:3]} {c:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, font, 0.45, 1)
            ty = max(th + 2, y1 - 3)
            cv2.rectangle(tile, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(tile, txt, (x1 + 1, ty), font, 0.45, col, 1, cv2.LINE_AA)
    s = frame_stats(dets)
    bar = np.zeros((HUD_H, tile.shape[1], 3), np.uint8)       # own strip ABOVE the video, so it hides nothing
    cv2.putText(bar, name, (6, 20), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, f"persons {s['n_person']}  vehicles {s['n_vehicle']}  {ms:.0f} ms", (6, 40), font, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def grid(tiles, cols):
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return canvas


class Summary:
    def __init__(self, names):
        self.rows = {n: [] for n in names}
        self.ms = {n: [] for n in names}

    def add(self, name, stats, ms):
        self.rows[name].append(stats)
        self.ms[name].append(ms)

    def table(self):
        lines = [f"{'model':14s} {'frames':>6s} {'person_rate':>11s} {'person_conf':>11s} "
                 f"{'persons/fr':>10s} {'vehicles/fr':>11s} {'infer_ms':>8s}"]
        for n, rows in self.rows.items():
            if not rows:
                continue
            with_p = [r["best_person_conf"] for r in rows if r["n_person"]]
            lines.append(
                f"{n:14s} {len(rows):6d} {len(with_p) / len(rows):11.3f} "
                f"{(statistics.mean(with_p) if with_p else 0):11.3f} "
                f"{statistics.mean(r['n_person'] for r in rows):10.2f} "
                f"{statistics.mean(r['n_vehicle'] for r in rows):11.2f} "
                f"{statistics.median(self.ms[n]):8.1f}")
        return "\n".join(lines)


def run(models, reader, timeout, every=1, max_frames=0, csv_path=None, record_path=None,
        display=True, tile_width=640, cols=2, window_height=850):
    summary = Summary([m.name for m in models])
    writer = csv_file = video = None
    if csv_path:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "t_s", "segment", "model", "n_person", "best_person_conf",
                         "n_vehicle", "infer_ms"])
    seen = processed = segment = 0
    shown = False
    t_start = time.perf_counter()
    try:
        while True:
            frame = reader.next_frame(timeout)
            if frame is None:
                break
            seen += 1
            if (seen - 1) % every:
                continue
            t = time.perf_counter() - t_start
            tiles = []
            for m in models:
                dets, ms = m.detect(frame)
                st = frame_stats(dets)
                summary.add(m.name, st, ms)
                if writer:
                    writer.writerow([seen - 1, f"{t:.3f}", segment, m.name, st["n_person"],
                                     f"{st['best_person_conf']:.4f}", st["n_vehicle"], f"{ms:.1f}"])
                tiles.append(draw_tile(frame, m.name, dets, ms, tile_width))
            processed += 1
            canvas = grid(tiles, min(cols, len(tiles)))
            foot = np.zeros((24, canvas.shape[1], 3), np.uint8)    # own strip, so it doesn't cover any video
            cv2.putText(foot, f"segment {segment}  (k = next segment, q = quit)",
                        (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            canvas = np.vstack([canvas, foot])
            if record_path:
                if video is None:
                    video = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                            (canvas.shape[1], canvas.shape[0]))
                video.write(canvas)
            if display:
                if not shown:
                    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)      # resizable: many tiles can still fit a screen
                    h, w = canvas.shape[:2]
                    f = min(1.0, window_height / h)                 # open no taller than the screen allows
                    cv2.resizeWindow(WINDOW, int(w * f), int(h * f))
                    cv2.moveWindow(WINDOW, 0, 0)
                    shown = True
                elif cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break                                           # window closed with X: stop, don't reopen it
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("k"):
                    segment += 1
                    print(f"[segment] {segment} starts at frame {seen - 1}")
            if max_frames and processed >= max_frames:
                break
    finally:
        reader.stop()
        reader.join(5.0)
        if csv_file:
            csv_file.close()
        if video is not None:
            video.release()
    return summary, processed


def build_parser():
    p = argparse.ArgumentParser(description="Run several detectors on the same frames, side by side.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--source", default="0", help="camera index, video file, or stream URL")
    p.add_argument("--models", nargs="+", required=True, metavar="NAME=WEIGHTS",
                   help="e.g. stock=yolov8n.pt visdrone=vd_gray_s42.pt cocoM=vd_coco_m_s42.pt")
    p.add_argument("--conf", type=float, default=0.25, help="min confidence for every model (0.25)")
    p.add_argument("--imgsz", type=int, default=640, help="inference size (640 = training size; "
                   "16:9 frames run letterboxed at 384x640)")
    p.add_argument("--device", default="cpu", help="cpu (default) or 0 for a CUDA GPU")
    p.add_argument("--every", type=int, default=1, help="process every Nth frame (files)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames")
    p.add_argument("--csv", type=Path, default=None, help="per-frame, per-model CSV")
    p.add_argument("--record", type=Path, default=None, help="save the comparison grid (.mp4)")
    p.add_argument("--record-raw", type=Path, default=None, help="save the raw stream (.mp4) "
                   "for later re-runs and labeling")
    p.add_argument("--no-display", action="store_true", help="run headless")
    p.add_argument("--cols", type=int, default=2, help="tiles per row (e.g. 4 for many models)")
    p.add_argument("--tile-width", type=int, default=640, help="pixel width of each tile")
    p.add_argument("--window-height", type=int, default=850,
                   help="open the window at most this tall (px) so it fits the screen; drag to resize")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--rtsp-transport", choices=("udp", "tcp"), default="udp")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    models_spec = parse_models(args.models)
    for _, w in models_spec:
        if not w.is_file() and w.name not in ("yolov8n.pt", "yolo11n.pt"):  # stock names auto-download
            print(f"[error] weights not found: {w}")
            return 1
    kind, source = parse_source(args.source)
    if kind == SOURCE_FILE and not Path(source).is_file():
        print(f"[error] video file not found: {source}")
        return 1
    models = [Model(n, w, args.conf, args.imgsz, args.device) for n, w in models_spec]
    for m in models:
        print(f"  model {m.name:12s} {m.kind}")
    reader = FrameReader(lambda: open_capture(kind, source, args.rtsp_transport, args.timeout),
                         live=kind != SOURCE_FILE, description=f"{kind} {source!r}",
                         raw_recorder=RawRecorder(args.record_raw) if args.record_raw else None)
    reader.start()
    try:
        summary, n = run(models, reader, args.timeout, args.every, args.max_frames, args.csv,
                         args.record, display=not args.no_display,
                         tile_width=args.tile_width, cols=args.cols, window_height=args.window_height)
    except (StreamTimeout, StreamError) as exc:
        print(f"[error] {exc}")
        if is_rtsp(source):
            print("        on the drone's Wi-Fi? try --rtsp-transport tcp")
        return 2
    finally:
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    print(f"\n{n} frames compared\n" + summary.table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
