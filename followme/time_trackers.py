#!/usr/bin/env python3
"""Laptop-CPU speed of the 15 tuned tracker setups (Michael 11-27: talk speed numbers come only from his laptop,
Ryzen 7 7735HS, CPU only), to show next to their tuned HOTA.

  cd /mnt/c/Users/mahed/OneDrive/Documents/Jarvis/jarvis-lean/followme && python time_trackers.py
  python time_trackers.py --video ~/follow_logs/<run>/video_000.mp4 --frames 600 --save ../results/followme_tracker_speed.json

What it does:
  1. Frames: the newest real flight's clean video (video_*.mp4, never overlay_*.mp4, never a fake-drone run) under
     ~/follow_logs, or --video (a file or a run folder). The first --frames frames (default 600) after --start.
  2. Detector: the flight detector exactly as live_follow.py resolves it (resolve_detection -> the INT8 export and its
     F1-max cutoffs, detector_precision read from the model files, Detector's own YOLO model and track arguments:
     imgsz, conf = the lower cutoff). Run ONCE per frame; the boxes are kept in memory and every setup gets the same
     boxes on the same frames. The model gets every box above the lower cutoff, as model.track hands the tracker in
     flight (keep_box's per-class cutoff is applied after tracking).
  3. Setups: the 15 labels of results/followme_tracker_tuning_all_summary.json (byte-exact from Drive jarvis_det/,
     written by tracker_tuning_all.py, Colab 11-27). Each is built the way that script built it: jarvis_track_v5's
     make_tracker (the Ultralytics built-in YAML of the tracker + the grid's overrides from tracker_configs(): with_reid,
     gmc_method sparseOptFlow / none, re-ID model yolo26n-cls.pt on box crops) + that setup's `chosen.settings_changed`.
     All six trackers (ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, FastTrack, TrackTrack) are Ultralytics built-ins
     there (8.4.184; its tracker YAMLs and tracker code are identical to 8.4.216's). Settings are read, never typed here.
     The re-ID encoder runs on the CPU (device "cpu"), whatever torch can see.
  4. Timing: time.perf_counter around tracker.update(boxes, frame) only, which includes camera-motion compensation and
     the re-ID encoder where the setup has them (Ultralytics runs both inside update). Not timed: decoding, the
     detector, building the Boxes. Each setup starts from a fresh tracker; the first --warmup frames (default 20) run
     but are not counted. Per setup: mean ms/frame, p95 ms, FPS = 1000 / mean. The detector's own ms is reported for
     reference (same warm-up rule).
  3b. Plus a 16th row, "BoT-SORT + CMC (flies)": followme/botsort_tuned.yaml exactly as live_follow.resolve_tracker
     resolves it (the decided file, checked against its decision record; match 0.90), placed right after the summary's
     botsort|reid-|gmc+ row. Its val HOTA is read from results/followme_tracker_settings_choice.json (the option whose
     settings equal the record's chosen ones).
  5. Output: a 16-row table, ~/follow_logs/tracker_speed_<YYYYMMDD_HHMMSS>.json (--out to change the folder) and a copy
     at --save PATH if given. Recorded: CPU model (/proc/cpuinfo), thread counts, library versions.
A setup that cannot be built or fails mid-run gets its error in its row and the run continues.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SUMMARY_FILE = REPO / "results" / "followme_tracker_tuning_all_summary.json"
LOGS = Path.home() / "follow_logs"
FRAMES = 600
WARMUP = 20
# The grid's construction constants (jarvis_det/jarvis_track_v5.py, tracker_configs(); used unchanged by
# tracker_tuning_all.py). They say HOW a label is built; every tuned VALUE comes from the summary file.
REID_MODEL = "yolo26n-cls.pt"          # external appearance encoder on box crops
GMC_ON, GMC_OFF = "sparseOptFlow", "none"
MICHAELS_CPU = "7735HS"                # Michael 11-27: talk speed numbers come only from his laptop
LIBS = ("ultralytics", "torch", "torchvision", "openvino", "opencv-python", "opencv-python-headless",
        "opencv-contrib-python", "numpy", "scipy", "lap", "boxmot")
_RUN_NAME = re.compile(r"^\d{8}_\d{6}")


# ---------------------------------------------------------------- the 15 setups, from the summary file

def load_summary(path=SUMMARY_FILE):
    p = Path(path)
    if not p.is_file():
        sys.exit(f"STOP: tuning summary not found: {p}\n  It is committed in results/; restore it.")
    raw = p.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def parse_label(label):
    """'botsort|reid+|gmc-' -> (tracker, reid, gmc, overrides), as jarvis_track_v5.tracker_configs() builds it.
    A plain label ('bytetrack') has no switches: no overrides, reid/gmc None."""
    parts = label.split("|")
    if len(parts) == 1:
        return parts[0], None, None, {}
    if len(parts) != 3 or parts[1] not in ("reid+", "reid-") or parts[2] not in ("gmc+", "gmc-"):
        raise ValueError(f"label {label!r} is not <tracker> or <tracker>|reid+/-|gmc+/-")
    reid, gmc = parts[1] == "reid+", parts[2] == "gmc+"
    ov = {"with_reid": reid, "gmc_method": GMC_ON if gmc else GMC_OFF}
    if reid:
        ov["model"] = REID_MODEL
    return parts[0], reid, gmc, ov


def setup_specs(summary):
    """One spec per label of summary['per_setup'], in the file's order. A label without a 'chosen' block gets an
    'error' instead of settings (its row reports it; the run goes on)."""
    specs = []
    for label, rec in summary["per_setup"].items():
        spec = dict(label=label, tracker=None, reid=None, gmc=None, overrides={}, settings=None,
                    val_HOTA=None, chosen_run=None, error=None)
        try:
            spec["tracker"], spec["reid"], spec["gmc"], spec["overrides"] = parse_label(label)
            ch = rec.get("chosen")
            if not ch or "settings_changed" not in ch:
                raise ValueError("no chosen.settings_changed in the summary")
            spec.update(settings=dict(ch["settings_changed"]), val_HOTA=ch.get("val_HOTA"), chosen_run=ch.get("run"))
        except Exception as e:                         # noqa: BLE001 - one bad entry must not stop the other rows
            spec["error"] = f"spec: {e}"
        specs.append(spec)
    return specs


FLIES_LABEL = "BoT-SORT + CMC (flies)"
FLIES_AFTER = "botsort|reid-|gmc+"


def flight_spec():
    """The tracker that flies: botsort_tuned.yaml as live_follow.resolve_tracker resolves it (it STOPS if the file
    differs from its decision record). Every key of the file goes to the tracker, as Ultralytics builds it from the
    file. val HOTA: the decision record's option with the chosen settings (class average, val)."""
    import live_follow as lf
    spec = dict(label=FLIES_LABEL, tracker=None, reid=False, gmc=None, overrides={}, settings=None, val_HOTA=None,
                chosen_run=None, error=None)
    try:
        trk = lf.resolve_tracker(lf.TRACKER)
        cfg = dict(trk["settings"])
        spec.update(tracker=cfg["tracker_type"], settings=cfg, gmc=cfg.get("gmc_method", GMC_OFF) != GMC_OFF,
                    reid=bool(cfg.get("with_reid")), chosen_run=f"followme/{trk['label']} (live_follow.resolve_tracker)")
        rec = lf.load_tracker_choice()
        opt = [o for o in rec.get("options", []) if o.get("settings") == rec["chosen"]]
        if len(opt) == 1:
            spec["val_HOTA"] = opt[0]["class_avg_HOTA"].get("val")
    except SystemExit as e:                            # resolve_tracker stops on a missing / changed file
        spec["error"] = f"spec: {e}"
    except Exception as e:                             # noqa: BLE001
        spec["error"] = f"spec: {type(e).__name__}: {e}"
    return spec


def all_specs(summary):
    """The summary's 15 setups with the flying tracker inserted after the summary's BoT-SORT + CMC row (16 rows)."""
    specs = setup_specs(summary)
    k = next((i + 1 for i, x in enumerate(specs) if x["label"] == FLIES_AFTER), len(specs))
    return specs[:k] + [flight_spec()] + specs[k:]


def make_tracker(name, **overrides):
    """jarvis_track_v5.make_tracker: the tracker's Ultralytics YAML + overrides. Added here: a key the YAML does not
    have is an error (Ultralytics would ignore it silently), and the re-ID encoder is pinned to the CPU."""
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    cfg = YAML.load(check_yaml(f"{name}.yaml"))
    unknown = sorted(k for k in overrides if k not in cfg and k != "device")
    if unknown:
        raise KeyError(f"{name}.yaml has no key(s) {unknown}")
    cfg.update(overrides)
    return TRACKER_MAP[cfg["tracker_type"]](args=IterableSimpleNamespace(**cfg))


def reid_weights(dirs=None):
    """-> (path to give Ultralytics, note). Looks in followme/weights/, the current folder and Ultralytics' weights_dir;
    if absent, Ultralytics downloads yolo26n-cls.pt into followme/weights/ (gitignored) on first use."""
    if dirs is None:
        dirs = [HERE / "weights", Path.cwd()]
        try:
            from ultralytics.utils import SETTINGS
            dirs.append(Path(SETTINGS["weights_dir"]))
        except Exception:                              # noqa: BLE001
            pass
    for d in dirs:
        if (Path(d) / REID_MODEL).is_file():
            return str(Path(d) / REID_MODEL), f"re-ID encoder: {Path(d) / REID_MODEL}"
    target = Path(dirs[0]) / REID_MODEL
    return str(target), (f"re-ID encoder: {REID_MODEL} NOT found locally -> Ultralytics will download it (a few MB, "
                         f"needs internet) into {target} when the first re-ID setup is built")


# ---------------------------------------------------------------- frames

def newest_flight_video(root=LOGS):
    """Newest run folder (by its <YYYYMMDD_HHMMSS> name) that is not a fake-drone run and has clean video_*.mp4
    chunks. -> (folder, [videos]) or (None, [])."""
    root = Path(root)
    runs = sorted((p for p in root.iterdir() if p.is_dir() and _RUN_NAME.match(p.name)), key=lambda p: p.name,
                  reverse=True) if root.is_dir() else []
    for run in runs:
        try:
            info = json.loads((run / "drone.json").read_text()) if (run / "drone.json").exists() else {}
        except (OSError, ValueError):
            info = {}
        if info.get("drone_nickname") == "fake":
            continue
        vids = sorted(run.glob("video_*.mp4"))
        if vids:
            return run, vids
    return None, []


def resolve_videos(video=None, root=LOGS):
    if video:
        p = Path(video).expanduser()
        vids = sorted(p.glob("video_*.mp4")) if p.is_dir() else [p] if p.is_file() else []
        if not vids:
            sys.exit(f"STOP: no video at {p} (a file, or a run folder with video_*.mp4)")
    else:
        run, vids = newest_flight_video(root)
        if not vids:
            sys.exit(f"STOP: no flight video (video_*.mp4 in a non-fake run folder) under {root}; pass --video PATH")
    if any(v.name.startswith("overlay_") for v in vids):
        sys.exit("STOP: overlay_*.mp4 has boxes and labels drawn on it; use the clean video_*.mp4")
    return vids


def iter_frames(videos, start=0, n=None):
    """BGR frames of the chunks in order, skipping `start`, at most n. Decoding is deterministic, so every pass over
    the same files gives the same frames."""
    skip, out = start, 0
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        try:
            while n is None or out < n:
                ok, img = cap.read()
                if not ok:
                    break
                if skip:
                    skip -= 1
                    continue
                out += 1
                yield img
        finally:
            cap.release()
        if n is not None and out >= n:
            return


# ---------------------------------------------------------------- detector (live_follow's, run once per frame)

def make_flight_detector(args):
    """-> (detect(img) -> (N,6) float32 [x1,y1,x2,y2,conf,cls], info). live_follow.resolve_detection picks the model
    and cutoffs (stops if the INT8 export is missing), detector_precision reads the model files, and Detector loads the
    model with its own track arguments; predict() gets those minus the tracker (persist / tracker)."""
    import live_follow as lf
    args.fake_detector = False
    cut_p, cut_v, cut_src = lf.resolve_detection(args)
    prec = lf.detector_precision(args.model, args.imgsz)
    det = lf.Detector(args.model, lf.TRACKER, args.imgsz, (cut_p, cut_v))
    kw = {k: v for k, v in det.args.items() if k not in ("persist", "tracker")}

    def detect(img):
        r = det.model.predict(img, **kw)[0]
        return r.boxes.data.cpu().numpy().astype(np.float32)[:, :6] if r.boxes is not None else np.zeros((0, 6), np.float32)

    info = dict(model=lf._rel(args.model, HERE), precision=prec.get("summary"), precision_source=prec.get("source"),
                cutoffs=dict(person=cut_p, vehicle=cut_v, source=cut_src), predict_args=kw)
    return detect, info


def detect_all(videos, start, n, detect, warmup=WARMUP, progress=print):
    """The detector pass: once per frame. -> (dets, shapes, det_ms list incl. warm-up)."""
    dets, shapes, ms = [], [], []
    for i, img in enumerate(iter_frames(videos, start, n)):
        t0 = time.perf_counter()
        d = detect(img)
        ms.append((time.perf_counter() - t0) * 1000)
        dets.append(np.asarray(d, np.float32).reshape(-1, 6))
        shapes.append(img.shape[:2])
        if progress and (i + 1) % 100 == 0:
            progress(f"  detector: {i + 1}/{n} frames")
    return dets, shapes, ms


# ---------------------------------------------------------------- timing

def stats(ms, warmup):
    t = np.asarray(ms[warmup:], float)
    if not len(t):
        return dict(n_timed=0, mean_ms=None, p95_ms=None, fps=None)
    mean = float(t.mean())
    return dict(n_timed=int(len(t)), mean_ms=round(mean, 3), p95_ms=round(float(np.percentile(t, 95)), 3),
                fps=round(1000.0 / mean, 1) if mean > 0 else None)


def time_setup(spec, videos, start, dets, shapes, warmup=WARMUP, make=make_tracker, reid_model=None):
    """One setup over the cached boxes and the same frames; only tracker.update is timed."""
    from ultralytics.engine.results import Boxes
    row = dict(setup=spec["label"], tracker=spec["tracker"], reid=spec["reid"], gmc=spec["gmc"],
               settings_changed=spec["settings"], val_HOTA=spec["val_HOTA"], chosen_run=spec["chosen_run"],
               n_timed=0, mean_ms=None, p95_ms=None, fps=None, error=spec["error"])
    if row["error"]:
        return row
    ov = dict(spec["overrides"])
    if ov.get("with_reid") and reid_model:
        ov["model"] = reid_model                       # same file the grid used, found or downloaded locally
    row["overrides"] = ov
    try:
        tracker = make(spec["tracker"], **{**ov, **spec["settings"], "device": "cpu"})
    except Exception as e:                             # noqa: BLE001 - print it in the row, go on with the rest
        row["error"] = f"build: {type(e).__name__}: {e}"
        return row
    ms, n_tracks, i = [], [], -1
    try:
        for i, img in enumerate(iter_frames(videos, start, len(dets))):
            boxes = Boxes(dets[i], shapes[i])
            t0 = time.perf_counter()
            out = tracker.update(boxes, img)
            ms.append((time.perf_counter() - t0) * 1000)
            n_tracks.append(len(out))
    except Exception as e:                             # noqa: BLE001
        row["error"] = f"update (frame {i}): {type(e).__name__}: {e}"
        return row
    if len(ms) != len(dets):
        row["error"] = f"video gave {len(ms)} frames on this pass, the detector pass had {len(dets)}"
        return row
    row.update(stats(ms, warmup), mean_tracks=round(float(np.mean(n_tracks[warmup:])), 2) if len(ms) > warmup else None)
    return row


# ---------------------------------------------------------------- machine record

def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def versions():
    out = {"python": platform.python_version()}
    for lib in LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    return out


def machine():
    m = dict(cpu_model=cpu_model(), logical_cpus=os.cpu_count(), platform=platform.platform(),
             wsl="microsoft" in platform.release().lower(), cv2_threads=cv2.getNumThreads(),
             omp_num_threads=os.environ.get("OMP_NUM_THREADS"), versions=versions())
    try:
        m["usable_cpus"] = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    try:
        import torch
        m.update(torch_threads=torch.get_num_threads(), torch_cuda_available=torch.cuda.is_available())
    except ImportError:
        pass
    m["michaels_laptop"] = MICHAELS_CPU in m["cpu_model"]
    return m


# ---------------------------------------------------------------- report

def _f(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def table(rows, det):
    head = f"{'#':>2}  {'setup':24s} {'mean ms':>8s} {'p95 ms':>8s} {'FPS':>7s} {'tracks':>6s} {'val HOTA':>8s}  note"
    lines = [head, "-" * len(head)]
    for k, r in enumerate(rows, 1):
        lines.append(f"{k:>2}  {r['setup']:24s} {_f(r['mean_ms'], 2):>8s} {_f(r['p95_ms'], 2):>8s} {_f(r['fps']):>7s} "
                     f"{_f(r.get('mean_tracks')):>6s} {_f(r['val_HOTA'], 4):>8s}  {('ERROR ' + r['error']) if r['error'] else ''}")
    lines.append("-" * len(head))
    lines.append(f"    {'detector (reference)':24s} {_f(det['mean_ms'], 2):>8s} {_f(det['p95_ms'], 2):>8s} "
                 f"{_f(det['fps']):>7s} {_f(det.get('mean_boxes')):>6s}           boxes/frame in the tracks column")
    return "\n".join(lines)


def run(videos, start, frames, detect, specs, warmup=WARMUP, make=make_tracker, reid_model=None, progress=print):
    """Detector once, then every setup. -> (rows, detector record, n frames)."""
    if frames <= warmup:
        raise ValueError(f"--frames {frames} must be more than the warm-up ({warmup})")
    progress(f"detector pass over {frames} frames (once; every setup reuses these boxes) ...")
    dets, shapes, det_ms = detect_all(videos, start, frames, detect, warmup, progress)
    if len(dets) < frames:
        progress(f"  NOTE: the video has only {len(dets)} frames after --start {start}; using those")
    if len(dets) <= warmup:
        raise ValueError(f"only {len(dets)} frames: need more than the warm-up ({warmup})")
    det = dict(stats(det_ms, warmup), mean_boxes=round(float(np.mean([len(d) for d in dets[warmup:]])), 2))
    rows = []
    for k, spec in enumerate(specs, 1):
        progress(f"[{k:>2}/{len(specs)}] {spec['label']} ...")
        r = time_setup(spec, videos, start, dets, shapes, warmup, make, reid_model)
        progress(f"         ERROR {r['error']}" if r["error"] else
                 f"         {r['mean_ms']:.2f} ms/frame (p95 {r['p95_ms']:.2f}), {r['fps']} FPS")
        rows.append(r)
    return rows, det, len(dets)


def build_parser():
    import live_follow as lf
    ap = argparse.ArgumentParser(description="Laptop-CPU speed of the 15 tuned tracker setups + the flying tracker (tracker update only).")
    ap.add_argument("--video", default=None, help="a video file or a run folder (default: newest real flight's "
                                                  "video_*.mp4 under ~/follow_logs)")
    ap.add_argument("--frames", type=int, default=FRAMES, help="frames used, warm-up included (default 600)")
    ap.add_argument("--start", type=int, default=0, help="skip this many frames first (default 0)")
    ap.add_argument("--warmup", type=int, default=WARMUP, help="frames run but not timed, per setup (default 20)")
    ap.add_argument("--model", default=str(lf.INT8_MODEL), help="detector (default: the INT8 flight export, as live_follow)")
    ap.add_argument("--imgsz", type=int, default=lf.IMGSZ)
    ap.add_argument("--conf-person", type=float, default=None, help="as live_follow (default: the model's F1-max file)")
    ap.add_argument("--conf-vehicle", type=float, default=None, help="as live_follow")
    ap.add_argument("--summary", default=str(SUMMARY_FILE), help="tuning summary (default: results/followme_tracker_tuning_all_summary.json)")
    ap.add_argument("--logs", default=str(LOGS), help="where to look for flights (default ~/follow_logs)")
    ap.add_argument("--out", default=str(LOGS), help="folder for tracker_speed_<date>.json (default ~/follow_logs)")
    ap.add_argument("--save", default=None, help="also copy the JSON here (e.g. ../results/followme_tracker_speed.json)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary, sha = load_summary(args.summary)
    specs = all_specs(summary)
    videos = resolve_videos(args.video, args.logs)
    m = machine()
    print(f"CPU: {m['cpu_model']}  ({m['logical_cpus']} logical CPUs, torch threads {m.get('torch_threads')}, "
          f"cv2 threads {m['cv2_threads']}; CUDA visible: {m.get('torch_cuda_available')}, re-ID forced to CPU)")
    if not m["michaels_laptop"]:
        print(f"  NOTE: not Michael's laptop ({MICHAELS_CPU}): these are NOT talk numbers (his rule, 11-27)")
    print("versions: " + ", ".join(f"{k} {v}" for k, v in m["versions"].items()))
    print(f"video: {videos[0]}" + (f" .. {videos[-1].name} ({len(videos)} files)" if len(videos) > 1 else "")
          + f"; frames {args.start}..{args.start + args.frames - 1}, first {args.warmup} per setup not timed")
    print(f"settings: {Path(args.summary).name} (sha256 {sha[:12]}, made {summary.get('made')}), "
          f"{len(specs) - 1} setups at their chosen.settings_changed + {FLIES_LABEL} from botsort_tuned.yaml")
    reid_model, reid_note = reid_weights() if any(s["reid"] for s in specs) else (None, "re-ID encoder: not needed")
    print(reid_note)
    detect, det_info = make_flight_detector(args)
    print(f"detector: {det_info['precision']}  ({det_info['model']}); conf to the model {det_info['predict_args'].get('conf')}"
          f", cutoffs person {det_info['cutoffs']['person']} / vehicle {det_info['cutoffs']['vehicle']}")
    t0 = time.time()
    rows, det, n = run(videos, args.start, args.frames, detect, specs, args.warmup, reid_model=reid_model)
    det.update(det_info)
    print()
    print(table(rows, det))
    print(f"\nms = time.perf_counter around tracker.update only (includes GMC and the re-ID encoder where used); "
          f"FPS = 1000 / mean. {n} frames, {n - args.warmup} timed per setup. Wall time {time.time() - t0:.0f} s.")
    now = datetime.now()
    rec = dict(
        what="laptop-CPU speed of the 15 tuned follow-me tracker setups + the flying botsort_tuned.yaml: tracker.update only, on cached flight-detector "
             "boxes (time_trackers.py)",
        made=now.isoformat(timespec="seconds"), machine=m,
        video=dict(files=[str(v) for v in videos], start=args.start, frames=n, warmup=args.warmup,
                   frame_size=None),
        detector=det,
        settings_source=dict(file=str(Path(args.summary).resolve()), sha256=sha, made=summary.get("made"),
                             trackers=summary.get("trackers"), which="per_setup[label].chosen.settings_changed; the flies row: followme/botsort_tuned.yaml via live_follow.resolve_tracker",
                             reid_model=reid_model, reid_device="cpu"),
        timing=("time.perf_counter around tracker.update(boxes, frame); includes camera-motion compensation and the "
                "re-ID encoder where the setup has them; excludes decoding, the detector and building Boxes. Fresh "
                "tracker per setup; the first `warmup` frames run untimed. FPS = 1000 / mean_ms."),
        rows=rows)
    first = next(iter_frames(videos, args.start, 1), None)
    if first is not None:
        rec["video"]["frame_size"] = [int(first.shape[1]), int(first.shape[0])]
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"tracker_speed_{now.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"saved {path}")
    if args.save:
        dst = Path(args.save).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
        print(f"copied to {dst}")
    return rec


if __name__ == "__main__":
    main()
