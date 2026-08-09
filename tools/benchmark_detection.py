"""
JARVIS Face Recognition — Stage 1 Detection Benchmark
=====================================================
Compares face DETECTORS (the ones defined in face_utils) on a face dataset and
reports which one to ship. It loads a dataset via a pluggable per-dataset loader,
runs every detector over the same sample of images, scores each with the pure
metrics in detection_metrics, and prints a side-by-side table. Modelled on
tools/benchmark.py (iterations, JSON persistence, --compare).

Usage:
  # WIDER FACE (the default dataset)
  python tools/benchmark_detection.py \
      --data-dir   path/to/WIDER_val/images \
      --annotations path/to/wider_face_val_bbx_gt.txt \
      --limit 100

  python tools/benchmark_detection.py ... --compare         # diff vs last run
  python tools/benchmark_detection.py ... --save-annotated   # dump sample images

This script does NOT reimplement metrics or detectors — it only orchestrates
face_utils (load_detector, draw_detections) and detection_metrics
(match_detections, accumulate_pr, average_precision, roc_points, roc_auc).

----------------------------------------------------------------------
DATASET-LOADER CONTRACT (the whole point of this file's design):
    A loader is ANY function returning a list of (image_path, gt_boxes), where
        image_path : str | Path to a readable image, and
        gt_boxes   : list of [x1, y1, x2, y2] in ABSOLUTE PIXELS
                     (the face_utils / detection_metrics box convention).
    The benchmark never inspects which dataset produced the list — add a loader,
    register it, and it slots straight in. load_widerface is implemented;
    load_fddb and load_mafa are stubbed with the same contract.
----------------------------------------------------------------------

CPU-only. Each detector is isolated: if one backend fails to construct or run,
it is reported as an error and the others still complete.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import scipy.io

# ── Path setup — this file lives in tools/ next to the modules it imports ──
TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from face_utils import load_detector, draw_detections
from detection_metrics import (
    match_detections,
    accumulate_pr,
    average_precision,
    best_f1_threshold,
    roc_points,
    roc_auc,
)

RESULTS_DIR = REPO_ROOT / "results"
BENCH_FILE = RESULTS_DIR / "benchmarks_detection.json"
WEIGHTS_DIR = TOOLS_DIR / "weights"

DEFAULT_DETECTORS = ["mediapipe", "mtcnn", "retinaface", "yolo"]
DEFAULT_LIMIT = None
DEFAULT_IOU = 0.5
DEFAULT_WEIGHTS = WEIGHTS_DIR / "yolov8n-face.pt"
MEDIAPIPE_MODEL = WEIGHTS_DIR / "blaze_face_short_range.tflite"
N_ANNOTATE = 10  # sample images dumped per detector with --save-annotated


# ════════════════════════════════════════════════════════════════════
# Dataset loaders — every one returns list[(image_path, gt_boxes)].
# ════════════════════════════════════════════════════════════════════

def load_widerface(data_dir, annotations, limit=None, min_box_size=0):
    """Load WIDER FACE validation annotations.

    Parses the official ``wider_face_val_bbx_gt.txt``, whose records are:
        <relative/image/path.jpg>
        <N>                       (face count for that image)
        x y w h blur expr illum invalid occlusion pose   (N such lines)
    Native boxes are [x, y, w, h] in absolute pixels; this CONVERTS each to
    [x1, y1, x2, y2] = [x, y, x + w, y + h] explicitly.

    WIDER FACE quirk: an image with zero faces is still followed by one filler
    line of zeros — that line is read and discarded so the parser stays aligned.

    Inputs:
        data_dir (str | Path): images root the relative paths are joined onto
            (e.g. ``WIDER_val/images``).
        annotations (str | Path): path to ``wider_face_val_bbx_gt.txt``.
        limit (int | None): stop after this many images (None = all).
        min_box_size (int): drop GT boxes whose width OR height is smaller than
            this many pixels (0 = keep everything). Useful to exclude the tiny,
            barely-labelled faces WIDER FACE is notorious for.

    Returns:
        list[tuple[str, list[list[int]]]]: (image_path, gt_boxes) per image,
        gt_boxes as [x1, y1, x2, y2] absolute pixels.
    """
    data_dir = Path(data_dir)
    samples = []

    with open(annotations, "r", encoding="utf-8") as f:
        lines = iter(line.rstrip("\n") for line in f)

        for rel_path in lines:
            rel_path = rel_path.strip()
            if not rel_path:
                continue  # skip stray blank lines between records

            try:
                count = int(next(lines).strip())
            except StopIteration:
                break  # truncated file — stop cleanly

            gt_boxes = []
            # count==0 still has ONE filler line of zeros to consume.
            for _ in range(max(count, 1) if count == 0 else count):
                fields = next(lines).split()
                if count == 0:
                    break  # discard the single filler line, no real face
                x, y, w, h = (int(float(v)) for v in fields[:4])
                if w <= 0 or h <= 0:
                    continue
                if min_box_size and (w < min_box_size or h < min_box_size):
                    continue
                gt_boxes.append([x, y, x + w, y + h])  # [x,y,w,h] -> [x1,y1,x2,y2]

            samples.append((str(data_dir / rel_path), gt_boxes))
            if limit is not None and len(samples) >= limit:
                break

    return samples


def load_fddb(data_dir, annotations, limit=None, min_box_size=0):
    """Load FDDB ellipse annotations, converting each ellipse to a bounding box.

    Parses one FDDB ellipse-list fold file (e.g. ``FDDB-fold-01-ellipseList.txt``),
    whose records are:
        <relative/image/path>     (NO extension — FDDB images are .jpg; appended)
        <N>                        (face count for that image)
        <major_axis_radius minor_axis_radius angle center_x center_y score>  (N lines)
    repeating per image.

    Each face is an ELLIPSE rotated by ``angle`` (RADIANS, measured from the
    x-axis). This converts it to a TIGHT axis-aligned bounding box whose
    half-extents are:
        half_w = sqrt((major*cos(angle))^2 + (minor*sin(angle))^2)
        half_h = sqrt((major*sin(angle))^2 + (minor*cos(angle))^2)
    giving box = [cx - half_w, cy - half_h, cx + half_w, cy + half_h], rounded
    to ints (the score field is ground truth — ignored here).

    Inputs:
        data_dir (str | Path): images root the relative paths join onto; ".jpg"
            is appended to each listed path since FDDB lists them without one.
        annotations (str | Path): path to ONE ellipse-list fold .txt file. To
            evaluate over all 10 folds, concatenate the fold files (they share
            this format) into one file and pass that — multi-file loading can be
            added later without changing this contract.
        limit (int | None): stop after this many images (None = all).
        min_box_size (int): drop boxes whose width OR height is smaller than this
            many pixels (0 = keep everything).

    Returns:
        list[tuple[str, list[list[int]]]]: (image_path, gt_boxes) per image,
        gt_boxes as [x1, y1, x2, y2] absolute pixels — same contract as
        load_widerface.
    """
    data_dir = Path(data_dir)
    samples = []

    with open(annotations, "r", encoding="utf-8") as f:
        lines = iter(line.rstrip("\n") for line in f)

        for rel_path in lines:
            rel_path = rel_path.strip()
            if not rel_path:
                continue  # skip stray blank lines between records

            try:
                count = int(next(lines).strip())
            except StopIteration:
                break  # truncated file — stop cleanly

            gt_boxes = []
            for _ in range(count):
                fields = next(lines).split()
                major, minor, angle, cx, cy = (float(v) for v in fields[:5])

                cos_a, sin_a = math.cos(angle), math.sin(angle)
                half_w = math.sqrt((major * cos_a) ** 2 + (minor * sin_a) ** 2)
                half_h = math.sqrt((major * sin_a) ** 2 + (minor * cos_a) ** 2)

                x1 = int(round(cx - half_w))
                y1 = int(round(cy - half_h))
                x2 = int(round(cx + half_w))
                y2 = int(round(cy + half_h))

                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                if min_box_size and (w < min_box_size or h < min_box_size):
                    continue
                gt_boxes.append([x1, y1, x2, y2])  # ellipse -> [x1,y1,x2,y2]

            # FDDB lists paths without an extension; the images are .jpg.
            samples.append((str(data_dir / (rel_path + ".jpg")), gt_boxes))
            if limit is not None and len(samples) >= limit:
                break

    return samples


def load_mafa(data_dir, annotations, limit=None, min_box_size=0):
    """Load MAFA (masked faces) TEST-set annotations from ``LabelTestAll.mat``.

    MAFA ships its labels as MATLAB ``.mat`` files. This loader targets the TEST
    set only — the train and test files have DIFFERENT internal struct layouts,
    and ``LabelTestAll.mat`` is the one parsed here.

    Loaded via ``scipy.io.loadmat``, the test labels live under the key
    ``LabelTest`` (a 1xN MATLAB struct array). After unwrapping the leading
    singleton dimension, each record is read POSITIONALLY (matching the
    reference parser data/MAFA/extract_face_from_MAFA.py):
        record[0][0] -> image filename, e.g. 'test_00000001.jpg'
        record[1]    -> an (num_faces x 18) label matrix; per face row the
                        first four values are the face box [x, y, w, h]. The
                        remaining columns (face_type, occluder box, occ
                        type/degree, gender, race, orientation, glasses box)
                        are mask/occlusion metadata, ignored for plain
                        detection scoring.
    Each face box is CONVERTED [x, y, w, h] -> [x1, y1, x2, y2] =
    [x, y, x + w, y + h], exactly like load_widerface.

    The .mat lists bare filenames (e.g. 'test_00000001.jpg'); each is joined
    onto ``data_dir``. MAFA sometimes encodes occluder/garbage rows with
    non-positive width/height, so any box with w <= 0 or h <= 0 after
    conversion is skipped.

    Inputs:
        data_dir (str | Path): directory directly containing the test .jpg
            files (typically ``test-images/images``); each listed filename is
            joined onto it.
        annotations (str | Path): path to ``LabelTestAll.mat``.
        limit (int | None): stop after this many images (None = all).
        min_box_size (int): drop boxes whose width OR height is smaller than
            this many pixels (0 = keep everything).

    Returns:
        list[tuple[str, list[list[int]]]]: (image_path, gt_boxes) per image,
        gt_boxes as [x1, y1, x2, y2] absolute pixels — same contract as
        load_widerface and load_fddb.
    """
    data_dir = Path(data_dir)
    mat = scipy.io.loadmat(annotations)

    # Test labels live under 'LabelTest'; fall back to the first non-metadata
    # key (scipy injects '__header__' / '__version__' / '__globals__').
    if "LabelTest" in mat:
        records = mat["LabelTest"]
    else:
        keys = [k for k in mat if not k.startswith("__")]
        if not keys:
            return []
        records = mat[keys[0]]
    records = records[0]  # unwrap the leading singleton dimension -> (N,)

    samples = []
    for record in records:
        image_name = str(record[0][0])  # bare filename, e.g. 'test_00000001.jpg'
        raw_labels = record[1]          # (num_faces x 18) label matrix

        gt_boxes = []
        for row in raw_labels:
            x, y, w, h = (int(v) for v in row[:4])
            if w <= 0 or h <= 0:
                continue  # occluder/garbage flag, not a real face box
            if min_box_size and (w < min_box_size or h < min_box_size):
                continue
            gt_boxes.append([x, y, x + w, y + h])  # [x,y,w,h] -> [x1,y1,x2,y2]

        samples.append((str(data_dir / image_name), gt_boxes))
        if limit is not None and len(samples) >= limit:
            break

    return samples


# Loader registry — --dataset selects one. Add a function above, register here.
LOADERS = {
    "widerface": load_widerface,
    "fddb": load_fddb,
    "mafa": load_mafa,
}


def load_dataset(args):
    """Dispatch to the loader chosen by --dataset, returning the sample list."""
    loader = LOADERS[args.dataset]
    return loader(
        args.data_dir,
        args.annotations,
        limit=args.limit,
        min_box_size=args.min_box_size,
    )


# ════════════════════════════════════════════════════════════════════
# Detector construction — built ONCE, reused across every image.
# ════════════════════════════════════════════════════════════════════

def detector_kwargs(name, weights):
    """Per-backend constructor kwargs, all pinned to min_confidence=0.0.

    A 0.0 floor keeps every candidate box so accumulate_pr / roc_points can
    sweep the full confidence range. Backends that need a model file get its
    path here so calling code stays backend-agnostic everywhere else.
    """
    kwargs = {"min_confidence": 0.0}
    if name in ("yolo", "yolo-face"):
        kwargs["weights"] = str(weights)
    elif name == "mediapipe":
        kwargs["model_asset_path"] = str(MEDIAPIPE_MODEL)
    return kwargs


# ════════════════════════════════════════════════════════════════════
# Running + scoring one detector over the sample.
# ════════════════════════════════════════════════════════════════════

def evaluate_detector(detector, samples, iou_threshold, n_annotate=0):
    """Run one detector over every sample, collecting labels + latencies.

    The image is loaded with cv2 OUTSIDE the timed region; only the
    ``detector.detect(image)`` call is timed with time.perf_counter(), so the
    latency reflects inference alone, not disk I/O or decoding.

    Inputs:
        detector: object exposing .detect(image) -> list[dict] (from face_utils).
        samples (list[tuple[str, list]]): (image_path, gt_boxes) from a loader.
        iou_threshold (float): IoU for a detection to count as a TP.
        n_annotate (int): keep (path, detections) for the first N images so
            --save-annotated can render them later.

    Returns:
        dict with all_labeled, total_gt, total_fp, latencies_ms, num_images,
        num_detections, and annotated (the kept samples).
    """
    all_labeled = []
    total_gt = 0
    total_fp = 0
    latencies_ms = []
    annotated = []

    for idx, (image_path, gt_boxes) in enumerate(samples):
        image = cv2.imread(str(image_path))
        if image is None:
            continue  # unreadable/missing file — skip, don't crash the run

        t0 = time.perf_counter()
        detections = detector.detect(image)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        labeled, _fn = match_detections(detections, gt_boxes, iou_threshold)
        all_labeled.extend(labeled)
        total_gt += len(gt_boxes)
        total_fp += sum(1 for _, is_tp in labeled if not is_tp)

        if idx < n_annotate:
            annotated.append((image_path, detections))

    return {
        "all_labeled": all_labeled,
        "total_gt": total_gt,
        "total_fp": total_fp,
        "latencies_ms": latencies_ms,
        "num_images": len(latencies_ms),
        "num_detections": len(all_labeled),
        "annotated": annotated,
    }


def _percentile(sorted_vals, pct):
    """Nearest-rank percentile of an already-sorted list (0.0 if empty)."""
    if not sorted_vals:
        return 0.0
    k = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def latency_stats(latencies_ms):
    """Mean / p50 / p95 / min / max (ms) and FPS (1000/mean) for a run."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                "min_ms": 0.0, "max_ms": 0.0, "fps": 0.0}
    ordered = sorted(latencies_ms)
    mean_ms = sum(ordered) / len(ordered)
    return {
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(_percentile(ordered, 50), 3),
        "p95_ms": round(_percentile(ordered, 95), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


def score_run(run):
    """Turn a raw evaluate_detector() result into AP / ROC-AUC + latency stats.

    AP (accumulate_pr -> average_precision) is the metric of record. ROC-AUC
    (roc_points with total_negatives = total FP candidates, -> roc_auc) is
    illustrative only — see the caveat in detection_metrics. The F1-max point
    (best_f1_threshold) is the recommended DEPLOYMENT confidence threshold —
    the single cut-off to actually ship, as opposed to AP's whole-curve summary.
    """
    precisions, recalls, thresholds = accumulate_pr(run["all_labeled"], run["total_gt"])
    ap = average_precision(precisions, recalls)
    f1_threshold, f1_precision, f1_recall, f1 = best_f1_threshold(
        precisions, recalls, thresholds
    )

    fpr, tpr = roc_points(
        run["all_labeled"], run["total_gt"], total_negatives=run["total_fp"]
    )
    auc = roc_auc(fpr, tpr)

    stats = latency_stats(run["latencies_ms"])
    return {
        "ap": round(ap, 4),
        "roc_auc": round(auc, 4),
        "f1_threshold": round(f1_threshold, 4),
        "f1_precision": round(f1_precision, 4),
        "f1_recall": round(f1_recall, 4),
        "f1": round(f1, 4),
        "num_images": run["num_images"],
        "num_detections": run["num_detections"],
        "total_gt": run["total_gt"],
        "total_fp": run["total_fp"],
        **stats,
    }


def run_stamp():
    """Filesystem-safe LOCAL timestamp for naming a per-run output folder.

    Local time (not the UTC used in the JSON log) so the folder name reads as
    the wall-clock time you actually ran it; colons are dropped because Windows
    forbids them in paths.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_annotated_images(detector_name, annotated, outdir, dataset):
    """Draw + write a few sample detections for one detector (best-effort).

    ``outdir`` is the per-run folder; it is created on demand so a run that
    saves nothing leaves no empty directory behind.
    """
    # One subfolder per detector inside the run folder, so all of a given
    # model's images live together (run_<stamp>/<detector>/...).
    detector_dir = Path(outdir) / detector_name
    detector_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (image_path, detections) in enumerate(annotated):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        out_path = detector_dir / f"detect_{dataset}_{i}.jpg"
        cv2.imwrite(str(out_path), draw_detections(image, detections))
        written.append(out_path)
    return written


# ════════════════════════════════════════════════════════════════════
# Orchestration — build each detector, evaluate, isolate failures.
# ════════════════════════════════════════════════════════════════════

def run_benchmark(samples, detector_names, weights, iou_threshold,
                  save_annotated=False, outdir=None, dataset="widerface"):
    """Evaluate every requested detector over the same sample list.

    Each detector is fully isolated: construction and the whole evaluation run
    are wrapped so a single failing backend records an ``error`` and the others
    still complete. Detectors are constructed ONCE and reused across all images.

    Returns:
        list[dict]: one result per detector, each either scored (ap, roc_auc,
        latency...) or carrying an ``error`` string.
    """
    results = []
    n_annotate = N_ANNOTATE if save_annotated else 0

    for name in detector_names:
        print(f"  • {name}: building...", flush=True)
        try:
            detector = load_detector(name, **detector_kwargs(name, weights))
        except Exception as exc:  # backend import / model-file / construction
            print(f"    ! {name} failed to construct: {exc}")
            results.append({"detector": name, "error": f"construct: {exc}"})
            continue

        print(f"  • {name}: running over {len(samples)} images...", flush=True)
        try:
            run = evaluate_detector(detector, samples, iou_threshold, n_annotate)
        except Exception as exc:  # an inference blow-up mid-dataset
            print(f"    ! {name} failed during detection: {exc}")
            results.append({"detector": name, "error": f"detect: {exc}"})
            continue

        scored = {"detector": name, **score_run(run)}

        if save_annotated and outdir is not None:
            written = save_annotated_images(name, run["annotated"], outdir, dataset)
            if written:
                print(f"    ↳ wrote {len(written)} annotated image(s) to {outdir}")

        results.append(scored)
        print(f"    AP={scored['ap']:.4f}  mean={scored['mean_ms']:.1f}ms  "
              f"FPS={scored['fps']:.1f}")

    return results


# ════════════════════════════════════════════════════════════════════
# Reporting.
# ════════════════════════════════════════════════════════════════════

def print_table(results, dataset, num_images, iou_threshold):
    """Print the detector comparison table, sorted by AP (the metric of record)."""
    ok = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    header = (f"  {'Detector':<12}  {'AP':>7}  {'ROC-AUC':>8}  "
              f"{'mean ms':>8}  {'p95 ms':>8}  {'FPS':>7}  {'#imgs':>6}")
    print("=" * len(header))
    print("  JARVIS Stage 1 — Face Detection Benchmark")
    print(f"  dataset: {dataset}  |  images: {num_images}  |  IoU: {iou_threshold}")
    print("=" * len(header))
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in sorted(ok, key=lambda x: x["ap"], reverse=True):
        print(f"  {r['detector']:<12}  {r['ap']:>7.4f}  {r['roc_auc']:>8.4f}  "
              f"{r['mean_ms']:>8.1f}  {r['p95_ms']:>8.1f}  {r['fps']:>7.1f}  "
              f"{r['num_images']:>6}")

    for r in errored:
        print(f"  {r['detector']:<12}  {'ERR':>7}  (skipped: {r['error']})")

    print()
    print("  AP is the metric of RECORD (standard for face detection — "
          "Pascal VOC / COCO / WIDER FACE).")
    print("  ROC-AUC is ILLUSTRATIVE ONLY: its FPR denominator is an "
          "approximation, so it is")
    print("  not comparable across detectors or runs (see detection_metrics).")
    print()


# ════════════════════════════════════════════════════════════════════
# Persistence + comparison (mirrors benchmark.py's shape).
# ════════════════════════════════════════════════════════════════════

def save_results(ts, dataset, iou_threshold, limit, num_images, min_box_size, results):
    """Append this run to results/benchmarks_detection.json.

    Records every setting that changes what the numbers MEAN, so a stored run
    can be interpreted without the command line that produced it. min_box_size
    matters as much as iou_threshold here: it drops ground-truth boxes below a
    pixel size, so raising it quietly removes the hardest faces and lifts AP
    without any detector improving.

    NOTE: entries written before min_box_size was recorded do not have the
    field — treat a missing key as "unknown", not as 0. (load_previous_run
    applies the same rule to legacy entries missing `dataset`.)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = []
    if BENCH_FILE.exists():
        with open(BENCH_FILE) as f:
            data = json.load(f)
    data.append({
        "timestamp": ts,
        "dataset": dataset,
        "iou_threshold": iou_threshold,
        "limit": limit,
        "num_images": num_images,
        "min_box_size": min_box_size,
        "results": results,
    })
    with open(BENCH_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_previous_run(current_ts, dataset):
    """Most recent saved run of the SAME dataset that isn't the current one.

    Deltas are only meaningful within one dataset — AP on WIDER FACE and AP on
    MAFA measure different things — so runs from other datasets are never
    candidates. Legacy entries written before ``dataset`` was recorded have no
    dataset to match and are skipped. Returns None if there is nothing to
    compare against.
    """
    if not BENCH_FILE.exists():
        return None
    with open(BENCH_FILE) as f:
        data = json.load(f)
    runs = [e for e in data
            if e["timestamp"] != current_ts and e.get("dataset") == dataset]
    if not runs:
        return None
    return max(runs, key=lambda e: e["timestamp"])


def print_comparison(prev, curr_results):
    """Print per-detector AP and mean-latency deltas vs a previous run."""
    prev_by = {r["detector"]: r for r in prev["results"] if "error" not in r}
    curr_by = {r["detector"]: r for r in curr_results if "error" not in r}
    names = list(dict.fromkeys(list(curr_by) + list(prev_by)))

    header = (f"  {'Detector':<12}  {'AP old':>8}  {'AP new':>8}  {'ΔAP':>8}  "
              f"{'ms old':>8}  {'ms new':>8}  {'Δms':>8}")
    print("=" * len(header))
    print(f"  Comparison vs run at {prev['timestamp']} "
          f"(dataset: {prev.get('dataset', '?')})")
    print("=" * len(header))
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in names:
        if name not in prev_by:
            print(f"  {name:<12}  {'(new)':>8}")
            continue
        if name not in curr_by:
            print(f"  {name:<12}  {'(removed)':>8}")
            continue
        o, n = prev_by[name], curr_by[name]
        d_ap = n["ap"] - o["ap"]
        d_ms = n["mean_ms"] - o["mean_ms"]
        print(f"  {name:<12}  {o['ap']:>8.4f}  {n['ap']:>8.4f}  {d_ap:>+8.4f}  "
              f"{o['mean_ms']:>8.1f}  {n['mean_ms']:>8.1f}  {d_ms:>+8.1f}")
    print()


# ════════════════════════════════════════════════════════════════════
# Entry point.
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 face-detection benchmark (dataset-loader agnostic).")
    parser.add_argument("--dataset", default="widerface", choices=sorted(LOADERS),
                        help="Which dataset loader to run (default: widerface).")
    parser.add_argument("--data-dir", required=True,
                        help="Images root the loader's relative paths join onto.")
    parser.add_argument("--annotations", required=True,
                        help="Annotation file for the chosen dataset "
                             "(e.g. wider_face_val_bbx_gt.txt).")
    parser.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS,
                        help=f"Detectors to compare (default: {' '.join(DEFAULT_DETECTORS)}).")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Max images to sample (default: {DEFAULT_LIMIT}).")
    parser.add_argument("--min-box-size", type=int, default=0,
                        help="Drop GT boxes smaller than this many px (0 = keep all).")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="YOLO face weights path.")
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU,
                        help=f"IoU for a TP match (default: {DEFAULT_IOU}).")
    parser.add_argument("--outdir", default=str(TOOLS_DIR / "output"),
                        help="Directory for --save-annotated images.")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Dump a few annotated sample images per detector.")
    parser.add_argument("--compare", action="store_true",
                        help="Diff AP/latency against the previous saved run.")
    args = parser.parse_args()

    print()
    print(f"  Loading dataset '{args.dataset}'...")
    samples = load_dataset(args)
    if not samples:
        print("  No images loaded — check --data-dir / --annotations. Aborting.\n")
        return
    total_gt = sum(len(gt) for _, gt in samples)
    print(f"  {len(samples)} images, {total_gt} ground-truth faces.\n")

    # Each run gets its own timestamped subfolder under --outdir so annotated
    # images from different runs never mix (only created if --save-annotated).
    run_dir = Path(args.outdir) / f"run_{run_stamp()}"
    if args.save_annotated:
        print(f"  Annotated images -> {run_dir}\n")

    t_start = time.perf_counter()
    results = run_benchmark(
        samples, args.detectors, Path(args.weights), args.iou_threshold,
        save_annotated=args.save_annotated, outdir=run_dir, dataset=args.dataset,
    )
    elapsed = time.perf_counter() - t_start

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print()
    print_table(results, args.dataset, len(samples), args.iou_threshold)
    print(f"  Total wall time: {elapsed:.1f}s\n")

    if args.compare:
        prev = load_previous_run(ts, args.dataset)
        if prev:
            print_comparison(prev, results)
        else:
            print(f"  --compare: no previous '{args.dataset}' run to compare "
                  f"against — skipping (runs on other datasets are not "
                  f"comparable).\n")

    save_results(ts, args.dataset, args.iou_threshold, args.limit,
                 len(samples), args.min_box_size, results)
    print(f"  Saved -> {BENCH_FILE.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
