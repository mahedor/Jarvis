#!/usr/bin/env python3
"""Blind spot re-check (Michael 11-28, talk chapter 12 / D4): does TODAY's flight detector see Michael in the 13 s at
dusk on flight 20261029_182029 where he stands between the cars?

The old numbers come from gap_fps.py (11-22, Michael's laptop CPU; copied byte-exact into
results/followme_blind_spot_gap_fps.py.txt with its outputs results/followme_blind_spot_gap_fps_yolo.json /
_rfdetr.json): the fine-tuned nano `vd_mav_cl_s42.pt` (FP32, model.predict at conf 0.25, no tracker) boxed him in 0 of
63 frames, stock YOLO11l in 54, RF-DETR nano (fine-tuned, final) in 59. It wrote no video. Since then the flight
detector changed: the INT8 OpenVINO export at its own F1-max cutoffs (person 0.23 / vehicle 0.29), imgsz 640, and the
tuned tracker botsort_tuned.yaml. This script measures it on the SAME 63 frames with the SAME hit rule:

  frames  3507, 3513, ..., 3879 (every 6th; follow.csv frame numbers = gap_fps.py's running index over the video_*.mp4
          chunks, checked: 1800 rows and 1800 frames per chunk). Frame 3600 is the video_001 -> video_002 boundary; each
          frame is found through follow.csv's video_file / frame_in_file.
  hit     a person box whose CENTRE is inside REGION = [241.5, 135.2, 452.0, 474.5] (where Michael stands, from the
          boxes before / after the gap; gap_fps.py's near()).

Models (each one's per-frame boxes, hit and time are saved):
  flight (default)   live_follow's own Detector.process: the decided model / cutoffs / imgsz / tracker exactly as
                     live_follow.py resolves them (resolve_detection, resolve_tracker; the same settings check as replay
                     must pass). It runs on EVERY frame from PREROLL frames before 3507 to 3879, the way the flight runs
                     it (so the tracker has its tracks, as it would mid-flight); hits are counted on the 63 frames.
  flight, detector only   the same model with no tracker: model.predict at the person cutoff, persons only, on the 63
                     frames (gap_fps.py's method with today's model and cutoff), plus the highest person confidence
                     inside the region at conf 0.01 (how close a miss was).
  --also NAME=PATH   optional reference models, never required: any Ultralytics .pt (e.g. yolo11l=yolo11l.pt) run as
                     gap_fps.py ran them (model.predict, conf 0.25, imgsz 640, persons only), or rfdetr=PATH.pth
                     (RF-DETR nano as gap_fps.py loaded it; `rfdetr` is imported only then; gap_fps ran it in a separate
                     env: there use --no-flight, then add that JSON to the main run with --merge).

Output (--out folder): blind_spot.json (settings, every frame's boxes and hit, per-model summary, the flight's own
recorded state on those frames, the old gap_fps numbers for reference) and blind_spot.mp4: the 63 frames, each held
for 6 frame times so it plays in real time (~12.6 s), one panel per model, the region drawn, boxes whose centre is in
it green, a running hit count. ffmpeg (H.264) if on PATH, else OpenCV mp4v (sudo apt install ffmpeg).
Missing run folder / follow.csv / video file / model -> STOP, never a silent fallback. Times are this computer's
(Michael 11-27: talk speed numbers come only from his laptop, Ryzen 7 7735HS, CPU only).

  python blind_spot_check.py --run /mnt/c/Users/mahed/Downloads/jarvis_videos/20261029_182029 \\
      --also yolo11l=yolo11l.pt --out /mnt/c/Users/mahed/Downloads/talk_videos/blind_spot
"""
import argparse
import csv
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import side_by_side as sbs          # Reader, panel, pick_encoder, ffmpeg_cmd (cv2 + numpy only)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FLIGHT = "20261029_182029"
FIRST, LAST, STEP = 3507, 3879, 6                     # gap_fps.py: 63 frames, ~5 per second across the gap
REGION = (241.5, 135.2, 452.0, 474.5)                 # gap_fps.py REG: where Michael stands
PREROLL = 60                                          # frames the flight pass runs before FIRST (2 s at 30 fps)
ALSO_CONF = 0.25                                      # gap_fps.py's conf for every model (not an F1-max cutoff)
ALSO_IMGSZ = 640
WARMUP = 3                                            # gap_fps.py: 3 untimed calls before timing
EVIDENCE = dict(script=REPO / "results" / "followme_blind_spot_gap_fps.py.txt",
                yolo=REPO / "results" / "followme_blind_spot_gap_fps_yolo.json",
                rfdetr=REPO / "results" / "followme_blind_spot_gap_fps_rfdetr.json")
PERSON = ("person", "pedestrian", "people")           # = replay.PERSON_NAMES (gap_fps.py used the same three)
RULE = "hit = at least one person box whose centre (x, y) is inside REGION (edges included), as gap_fps.py near()"


# ---- pure logic (tested) ----

def sample_frames(first=FIRST, last=LAST, step=STEP):
    return list(range(first, last + 1, step))


def frame_map(rows, frames):
    """follow.csv rows -> {frame: (video_file, frame_in_file, t_s)} for the frames asked for. Older logs without
    frame_in_file: counted per chunk (as side_by_side.load_run). A frame not in follow.csv -> STOP."""
    by_frame, count = {}, {}
    for r in rows:
        vf = r.get("video_file") or ""
        idx = r.get("frame_in_file")
        idx = int(idx) if idx not in (None, "") else count.get(vf, 0)
        count[vf] = idx + 1
        by_frame[int(r["frame"])] = (vf, idx, float(r["t_s"]) if r.get("t_s") not in (None, "") else None)
    missing = [f for f in frames if f not in by_frame or not by_frame[f][0]]
    if missing:
        raise SystemExit(f"STOP: follow.csv has no video frame for frame(s) {missing[:5]}"
                         + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""))
    return {f: by_frame[f] for f in frames}


def check_videos(run, mapping, frame_count=None):
    """Every video file the frames need exists and holds the needed frame index, else STOP."""
    frame_count = frame_count or (lambda p: int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_COUNT)))
    need = {}
    for vf, idx, _ in mapping.values():
        need[vf] = max(need.get(vf, -1), idx)
    for vf, idx in sorted(need.items()):
        p = Path(run) / vf
        if not p.is_file():
            raise SystemExit(f"STOP: video file missing: {p}")
        n = frame_count(p)
        if n <= idx:
            raise SystemExit(f"STOP: {p} has {n} frames, frame_in_file {idx} needed")


def center_in(box, region=REGION):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def is_hit(boxes, region=REGION):
    return any(center_in(b["xyxy"], region) for b in boxes)


def summarize(frames):
    """Per-frame results [{hit, ms, ...}] -> {hits, of, hit_frames, ms_median, fps}."""
    ms = [f["ms"] for f in frames if f.get("ms") is not None]
    med = statistics.median(ms) if ms else None
    return dict(hits=sum(bool(f["hit"]) for f in frames), of=len(frames),
                hit_frames=[f["frame"] for f in frames if f["hit"]],
                ms_median=round(med, 1) if med else None, fps=round(1000 / med, 1) if med else None)


def parse_also(spec):
    """'yolo11l=yolo11l.pt' -> ('yolo11l', 'yolo', Path); 'rfdetr=x.pth' -> ('rfdetr', 'rfdetr', Path). The weights
    must exist (asked for = required; never skipped quietly)."""
    name, sep, path = str(spec).partition("=")
    if not sep or not name.strip() or not path.strip():
        raise SystemExit(f"STOP: --also {spec!r}: use NAME=PATH, e.g. yolo11l=yolo11l.pt or rfdetr=rfdetr_final_best_total.pth")
    p = Path(path.strip()).expanduser()
    if not p.is_file():
        raise SystemExit(f"STOP: --also {name}: weights not found: {p}")
    kind = "rfdetr" if name.strip().lower().startswith("rfdetr") else "yolo"
    return name.strip(), kind, p


def same_frames(a, b):
    """Two blind_spot.json 'frames' blocks describe the same measurement (frames, region, rule)."""
    keys = ("list", "region", "rule")
    return all(a.get(k) == b.get(k) for k in keys)


# ---- models ----

def _yolo_predict(model, conf, imgsz):
    pid = [k for k, v in model.names.items() if str(v).lower() in PERSON]

    def f(img):
        r = model.predict(img, conf=conf, imgsz=imgsz, classes=pid, verbose=False)[0]
        return [dict(xyxy=[round(float(x), 1) for x in b], conf=round(float(c), 3))
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return f


def _rfdetr(path):
    try:
        from PIL import Image
        from rfdetr import RFDETRNano
    except ImportError as ex:
        raise SystemExit(f"STOP: --also rfdetr needs the rfdetr package ({ex}). gap_fps.py ran it in a separate env: "
                         f"run this script there with --no-flight --also rfdetr=..., then pass that JSON with --merge.")
    m = RFDETRNano(pretrain_weights=str(path), num_classes=7, resolution=640)     # as gap_fps.py
    try:
        m.optimize_for_inference()
    except Exception:                                                          # noqa: BLE001 - as gap_fps.py
        pass

    def f(img):
        o = m.predict(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), threshold=ALSO_CONF)
        return [dict(xyxy=[round(float(x), 1) for x in b], conf=round(float(s), 3))
                for b, c, s in zip(o.xyxy, o.class_id, o.confidence) if c == 0]
    return f


def measure(detect, images, mapping, region=REGION, warmup=WARMUP, extra=None):
    """detect(img) -> [{xyxy, conf}] on each sampled frame (after `warmup` untimed calls) -> per-frame results."""
    frames = sorted(images)
    for f in frames[:warmup]:
        detect(images[f])
    out = []
    for f in frames:
        t0 = time.perf_counter()
        boxes = detect(images[f])
        ms = (time.perf_counter() - t0) * 1000
        vf, idx, t = mapping[f]
        row = dict(frame=f, video_file=vf, frame_in_file=idx, t_s=t, hit=is_hit(boxes, region), ms=round(ms, 1),
                   boxes=boxes)
        if extra:
            row.update(extra(images[f]))
        out.append(row)
    return out


def flight_models(run, mapping_all, sampled, images, region=REGION):
    """The flight detector as live_follow resolves it: tracked pass on every frame + detector-only pass."""
    import live_follow as lf
    import replay
    args = lf.build_parser().parse_args([])                  # the defaults = what flies (no flags here to change them)
    cut_p, cut_v, cut_src = lf.resolve_detection(args)
    prec = lf.detector_precision(args.model, args.imgsz)
    trk = lf.resolve_tracker(args.tracker)
    actual = lf.running_settings(args, drone_profile_applied=False, cutoffs=(cut_p, cut_v), precision=prec["precision"])
    check = lf.check_decisions(actual, lf.load_decisions(), {"replay"})
    if not check["passed"]:
        raise SystemExit("STOP: the flight detector here is not what flies:\n  " + "\n  ".join(
            f"{x['name']}: decided {x['decided']!r}, about to run {x['actual']!r}" for x in check["deviations"]))
    for line in lf.precision_lines(prec, args.model):
        print(line)
    print(f"cutoffs: person >= {cut_p:.2f}, vehicle >= {cut_v:.2f}  ({cut_src})")
    print(lf.tracker_line(trk))
    settings = dict(model=lf._rel(args.model, HERE), precision=prec, imgsz=args.imgsz,
                    cutoffs=dict(person=cut_p, vehicle=cut_v, source=cut_src),
                    tracker=dict(file=trk["label"], settings=trk["settings"]), settings_check=check)
    # 1. tracked, every frame (as in flight)
    det = lf.Detector(args.model, trk["path"], args.imgsz, (cut_p, cut_v))
    person_ids = {k for k, v in det.names.items() if str(v).lower() in replay.PERSON_NAMES}
    rd, tracked = sbs.Reader(), []
    try:
        for f in sorted(mapping_all):
            vf, idx, t = mapping_all[f]
            img = rd.get(Path(run) / vf, idx)
            if img is None:
                raise SystemExit(f"STOP: could not read frame {f} ({vf} frame {idx})")
            tracks, det_ms, _, _ = det.process(img, t if t is not None else f / 30.0)
            if f in sampled:
                boxes = [dict(xyxy=[round(float(x), 1) for x in tr[:4]], conf=round(float(tr[6]), 3), id=int(tr[4]))
                         for tr in tracks if tr[5] in person_ids]
                tracked.append(dict(frame=f, video_file=vf, frame_in_file=idx, t_s=t, hit=is_hit(boxes, region),
                                    ms=round(det_ms, 1), boxes=boxes))
            if f % 30 == 0 or f == max(mapping_all):
                print(f"\rflight detector (tracked): frame {f} ({len(tracked)}/{len(sampled)} counted)", end="", flush=True)
    finally:
        rd.close()
    print()
    # 2. detector only (no tracker), the sampled frames
    from ultralytics import YOLO
    m = YOLO(args.model)
    at_cut = _yolo_predict(m, cut_p, args.imgsz)
    low = _yolo_predict(m, 0.01, args.imgsz)

    def best_in_region(img):
        c = [b["conf"] for b in low(img) if center_in(b["xyxy"], region)]
        return dict(max_conf_in_region=max(c) if c else None)
    det_only = measure(at_cut, images, mapping_all, region, extra=best_in_region)   # ms: the cutoff pass only
    name = Path(args.model).name
    return [dict(label=f"flight detector (INT8, {cut_p:.2f}, tracked)", kind="flight_tracked", model=name,
                 how=f"live_follow Detector.process (model.track + {trk['label']} + per-class cutoffs) on every frame "
                     f"from {min(mapping_all)} to {max(mapping_all)}; counted on the sampled frames; ms = det_ms",
                 settings=settings, frames=tracked),
            dict(label=f"flight detector, no tracker ({cut_p:.2f})", kind="flight_detector_only", model=name,
                 how=f"model.predict(conf={cut_p}, imgsz={args.imgsz}, persons only) on the sampled frames, {WARMUP} "
                     f"untimed warm-up calls; max_conf_in_region from a second, untimed pass at conf 0.01",
                 settings=settings, frames=det_only)]


def also_model(name, kind, path, images, mapping, region=REGION):
    if kind == "rfdetr":
        detect, how = _rfdetr(path), f"RFDETRNano(num_classes=7, resolution=640).predict(threshold={ALSO_CONF}), class 0"
    else:
        from ultralytics import YOLO
        detect = _yolo_predict(YOLO(str(path)), ALSO_CONF, ALSO_IMGSZ)
        how = f"model.predict(conf={ALSO_CONF}, imgsz={ALSO_IMGSZ}, persons only), as gap_fps.py"
    print(f"{name}: {path}")
    return dict(label=f"{name} (conf {ALSO_CONF})", kind=kind, model=str(path), how=how + f"; {WARMUP} untimed warm-up calls",
                frames=measure(detect, images, mapping, region))


# ---- video ----

def draw_panel(img, frame_res, label, cum, k, height, width, region=REGION):
    vis = img.copy()
    cv2.rectangle(vis, (int(region[0]), int(region[1])), (int(region[2]), int(region[3])), (255, 255, 255), 1)
    cv2.putText(vis, "region", (int(region[0]) + 4, int(region[1]) + 16), 0, 0.5, (255, 255, 255), 1)
    for b in frame_res["boxes"]:
        x1, y1, x2, y2 = (int(v) for v in b["xyxy"])
        inside = center_in(b["xyxy"], region)
        col = (0, 255, 0) if inside else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3 if inside else 1)
        txt = (f"#{b['id']} " if "id" in b else "") + f"{b['conf']:.2f}"
        cv2.putText(vis, txt, (x1, max(y1 - 5, 14)), 0, 0.5, col, 2 if inside else 1)
    mark = "SEEN" if frame_res["hit"] else "missed"
    t = frame_res.get("t_s")
    right = f"{mark}   seen in {cum} of {k + 1}   frame {frame_res['frame']}" + (f"  t {t:.1f} s" if t is not None else "")
    return sbs.panel(vis, height, width, label, right)


FOOTER_H = 30


def render(models, images, out, fps=30.0, step=STEP, region=REGION, encoder="auto"):
    """One composite per sampled frame (a panel per model), held `step` frame times. -> dict(out, frames, encoder)."""
    frames = sorted(images)
    h0, w0 = images[frames[0]].shape[:2]
    n = len(models)
    height = min(h0, int((1920 - sbs.GAP * (n - 1)) / n * h0 / w0)) // 2 * 2
    pw = int(round(w0 * height / h0)) // 2 * 2
    W, H = pw * n + sbs.GAP * (n - 1), sbs.BAR_H + height + FOOTER_H
    enc = sbs.pick_encoder(encoder)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if enc == "ffmpeg":
        import subprocess
        proc = subprocess.Popen(sbs.ffmpeg_cmd(out, W, H, fps), stdin=subprocess.PIPE)
        write = proc.stdin.write
    else:
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        write = lambda buf: vw.write(np.frombuffer(buf, np.uint8).reshape(H, W, 3))  # noqa: E731
    by = [{r["frame"]: r for r in m["frames"]} for m in models]
    cums = [0] * n
    written = 0
    try:
        for k, f in enumerate(frames):
            canvas = np.zeros((H, W, 3), np.uint8)
            for j, m in enumerate(models):
                r = by[j][f]
                cums[j] += bool(r["hit"])
                x0 = j * (pw + sbs.GAP)
                canvas[:sbs.BAR_H + height, x0:x0 + pw] = draw_panel(images[f], r, m["label"], cums[j], k, height, pw, region)
            cv2.putText(canvas, "SEEN = a person box with its centre inside the white region (where Michael stands)",
                        (10, H - 9), 0, 0.55, (220, 220, 220), 1)
            for _ in range(step):
                write(canvas.tobytes()); written += 1
    finally:
        if enc == "ffmpeg":
            proc.stdin.close()
            if proc.wait() != 0:
                raise SystemExit("STOP: ffmpeg failed (see above); retry with --encoder opencv")
        else:
            vw.release()
    return dict(out=str(out), frames=written, seconds=round(written / fps, 2), encoder=enc, size=[W, H])


# ---- main ----

def _cpu():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "not recorded"


def previous_numbers():
    """The 11-22 gap_fps.py results (committed copies), for the comparison printout; missing file -> None."""
    out = {}
    for k in ("yolo", "rfdetr"):
        p = EVIDENCE[k]
        out[k] = json.loads(p.read_text()) if p.is_file() else None
    return dict(source="gap_fps.py 11-22 on Michael's laptop (results/followme_blind_spot_gap_fps.py.txt): "
                       "model.predict at conf 0.25, imgsz 640, no tracker; 'YOLOv8n fine-tuned (flies today)' was "
                       "vd_mav_cl_s42.pt in FP32, not today's INT8 export",
                yolo=out["yolo"], rfdetr=out["rfdetr"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Blind spot re-check on flight 20261029_182029 frames 3507-3879.")
    ap.add_argument("--run", required=True, help="the flight folder (follow.csv + video_*.mp4)")
    ap.add_argument("--out", required=True, help="output folder (blind_spot.json + blind_spot.mp4)")
    ap.add_argument("--also", action="append", default=[], metavar="NAME=PATH",
                    help="extra reference model: yolo11l=yolo11l.pt (any Ultralytics .pt) or rfdetr=PATH.pth")
    ap.add_argument("--no-flight", action="store_true", help="skip the flight detector (for a separate RF-DETR env)")
    ap.add_argument("--merge", action="append", default=[], metavar="JSON",
                    help="add the models of an earlier blind_spot.json (same frames and region) to this one's JSON and video")
    ap.add_argument("--name", default="blind_spot", help="output file name stem (default blind_spot)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--encoder", choices=["auto", "ffmpeg", "opencv"], default="auto")
    args = ap.parse_args(argv)
    run = Path(args.run).expanduser()
    if not (run / "follow.csv").is_file():
        raise SystemExit(f"STOP: no follow.csv in {run}")
    also = [parse_also(s) for s in args.also]                 # check every asked-for file before any work
    merged = []
    for mp in args.merge:
        p = Path(mp).expanduser()
        if not p.is_file():
            raise SystemExit(f"STOP: --merge: no such file {p}")
        merged.append((p, json.loads(p.read_text())))
    if args.no_flight and not also and not merged:
        raise SystemExit("STOP: --no-flight with nothing else to measure (add --also NAME=PATH)")
    rows = list(csv.DictReader(open(run / "follow.csv", newline="")))
    sampled = sample_frames(FIRST, LAST, STEP)
    mapping = frame_map(rows, sampled)
    first_all = max(FIRST - PREROLL, 0)
    mapping_all = frame_map(rows, list(range(first_all, LAST + 1))) if not args.no_flight else mapping
    check_videos(run, mapping_all)
    frames_doc = dict(list=sampled, n=len(sampled), first=FIRST, last=LAST, step=STEP, region=list(REGION), rule=RULE,
                      preroll_from=first_all if not args.no_flight else None,
                      span=dict(first=dict(zip(("video_file", "frame_in_file", "t_s"), mapping[sampled[0]])),
                                last=dict(zip(("video_file", "frame_in_file", "t_s"), mapping[sampled[-1]]))),
                      boundaries=[dict(frame=f, video_file=mapping_all[f][0]) for f in sorted(mapping_all)
                                  if f - 1 in mapping_all and mapping_all[f][0] != mapping_all[f - 1][0]],
                      frame_source="follow.csv video_file / frame_in_file (frame numbers = gap_fps.py's running "
                                   "index over the chunks)")
    for p, doc in merged:
        if not same_frames(doc.get("frames", {}), frames_doc):
            raise SystemExit(f"STOP: --merge {p}: measured on other frames / region / rule than this run")
    print(f"{run.name}: {len(sampled)} frames {FIRST}-{FIRST + STEP * (len(sampled) - 1)} every {STEP}; "
          f"t_s {mapping[sampled[0]][2]}-{mapping[sampled[-1]][2]} s; "
          f"{mapping[sampled[0]][0]}:{mapping[sampled[0]][1]} -> {mapping[sampled[-1]][0]}:{mapping[sampled[-1]][1]}")
    rd, images = sbs.Reader(), {}
    try:
        for f in sampled:
            img = rd.get(run / mapping[f][0], mapping[f][1])
            if img is None:
                raise SystemExit(f"STOP: could not read frame {f}")
            images[f] = img.copy()
    finally:
        rd.close()
    models = []
    if not args.no_flight:
        models += flight_models(run, mapping_all, set(sampled), images, REGION)
    for name, kind, path in also:
        models.append(also_model(name, kind, path, images, mapping, REGION))
    for p, doc in merged:
        for m in doc.get("models", []):
            models.append(dict(m, merged_from=str(p)))
    for m in models:
        m["summary"] = summarize(m["frames"])
    by_frame = {int(r["frame"]): r for r in rows}
    flight_recorded = [dict(frame=f, state=by_frame[f].get("state"), mode=by_frame[f].get("mode"),
                            target_id=by_frame[f].get("target_id")) for f in sampled]
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(program="blind_spot_check.py", created=datetime.now().isoformat(timespec="seconds"), computer=_cpu(),
               run=str(run), flight=run.name, frames=frames_doc,
               models=models, flight_recorded=flight_recorded, previous=previous_numbers())
    video = None
    if not args.no_video and models:
        shown = [m for m in models if m["kind"] != "flight_detector_only"]     # the video: tracked flight + references
        video = render(shown, images, out / f"{args.name}.mp4", step=STEP, region=REGION, encoder=args.encoder)
        doc["video"] = dict(video, panels=[m["label"] for m in shown],
                            note=f"each of the {len(sampled)} frames held {STEP} frame times at 30 fps (real time)")
    (out / f"{args.name}.json").write_text(json.dumps(doc, indent=1, default=str))
    print(f"\n{'model':48s} {'seen':>8s} {'ms/frame':>9s} {'FPS':>6s}")
    for m in models:
        s = m["summary"]
        print(f"{m['label'][:48]:48s} {s['hits']:3d}/{s['of']:<3d}  {s['ms_median'] or 0:9.1f} {s['fps'] or 0:6.1f}")
    prev = doc["previous"]
    if prev["yolo"] or prev["rfdetr"]:
        print("for reference, gap_fps.py 11-22 (.pt FP32, conf 0.25, no tracker):")
        for k in ("yolo", "rfdetr"):
            for name, r in (prev[k] or {}).items():
                print(f"  {name[:46]:46s} {r['hits']:3d}/{r['of']:<3d}  {r['ms']:9.1f} {r['fps']:6.1f}")
    print(f"saved {out / (args.name + '.json')}" + (f" + {video['out']} ({video['seconds']} s, encoder {video['encoder']}"
          + ("" if video["encoder"] == "ffmpeg" else "; for H.264 install ffmpeg in WSL: sudo apt install ffmpeg") + ")"
          if video else ""))
    return doc


if __name__ == "__main__":
    main()
