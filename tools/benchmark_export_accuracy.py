"""
JARVIS — Export Accuracy Check (VisDrone val mAP per format)
============================================================
Companion to benchmark_speed.py. A faster export is only worth having if it
detects the same things, so every format gets scored on the full VisDrone val
split (548 images) and compared against the PyTorch weights it came from.

WHY NOT JUST `yolo val`
  Ultralytics' validator refuses non-square input for non-PyTorch formats, and
  the exports are static 384x640 (the shape the PyTorch baseline actually runs
  at on 16:9 drone frames). So this script drives every format through ONE
  loop instead: benchmark_speed's backend preprocess + inference at the export
  shape, then the validator's own NMS (multi_label=True), matching
  (DetectionValidator.match_predictions over IoU 0.50:0.95) and AP code
  (DetMetrics -> ap_per_class). Same images, same letterbox, same NMS, same
  scorer — the only thing that varies between rows is the runtime.

  As a check on the loop itself, the .pt is ALSO scored by stock
  `YOLO.val()` at imgsz 640 (rect batches, which on 16:9 also gives 384x640).
  The two PyTorch numbers should be close but need not be identical: the val
  dataloader resizes with INTER_AREA when shrinking, predict() with
  INTER_LINEAR. Compare exports to the loop's PyTorch row, never to the
  reference row.

  Settings are Ultralytics' val defaults — conf 0.001, NMS IoU 0.7, max_det
  300 — not the deployment conf (0.172). mAP integrates over all confidences;
  thresholding at 0.172 first would truncate the PR curve.

LABELS
  The val split ships raw VisDrone annotations. They are converted exactly the
  way Ultralytics' VisDrone.yaml converts them when the model was trained in
  Colab: rows with score 0 (ignored regions, and every category-11 "others"
  row) are dropped, class = category - 1. They are written to
  VisDrone2019-DET-val/labels/, where Ultralytics looks for them, and the
  images stay where benchmark_speed.py reads them.

Usage:
    python tools/benchmark_export_accuracy.py            # the three VisDrone formats
    python tools/benchmark_export_accuracy.py --models a.pt a.onnx --no-reference

Results are appended to results/benchmarks_export_accuracy.json.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import benchmark_speed as bs  # noqa: E402

RESULTS_FILE = bs.RESULTS_DIR / "benchmarks_export_accuracy.json"
VAL_DIR = bs.VISDRONE_VAL_IMAGES.parent
LABELS_DIR = VAL_DIR / "labels"
DATA_YAML = bs.VISDRONE_DIR / "visdrone_val.yaml"

VISDRONE_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
                  "tricycle", "awning-tricycle", "bus", "motor"]

MODEL_STEM = bs.REPO_ROOT / "data" / "models" / "yolov8n_visdrone_640_e50_s42"
DEFAULT_MODELS = [MODEL_STEM.with_suffix(".pt"),
                  MODEL_STEM.with_suffix(".onnx"),
                  MODEL_STEM.parent / f"{MODEL_STEM.name}_openvino_model"]
DEFAULT_IMGSZ = [384, 640]
VAL_CONF, VAL_IOU, VAL_MAX_DET = 0.001, 0.7, 300
# FP32 exports should reproduce PyTorch to within this much mAP.
FP32_TOLERANCE = 0.005


# ════════════════════════════════════════════════════════════════════
# Pure helpers — unit-tested (tests/test_benchmark_export_accuracy.py).
# ════════════════════════════════════════════════════════════════════

def visdrone_to_yolo(annotation_text, width, height):
    """VisDrone annotation rows -> YOLO label lines, as Ultralytics converts them.

    Row: x,y,w,h,score,category,truncation,occlusion (pixels, top-left origin).
    score 0 marks an ignored region and is dropped; class = category - 1.
    """
    dw, dh = 1.0 / width, 1.0 / height
    lines = []
    for row in (r.split(",") for r in annotation_text.strip().splitlines() if r.strip()):
        if row[4] == "0":
            continue
        x, y, w, h = map(int, row[:4])
        cls = int(row[5]) - 1
        lines.append(f"{cls} {(x + w / 2) * dw:.6f} {(y + h / 2) * dh:.6f} {w * dw:.6f} {h * dh:.6f}\n")
    return lines


def yolo_to_xyxy(label_text, width, height):
    """YOLO label lines -> (classes, [[x1, y1, x2, y2], ...]) in pixels."""
    classes, boxes = [], []
    for line in label_text.strip().splitlines():
        if not line.strip():
            continue
        c, xc, yc, w, h = line.split()
        xc, yc, w, h = float(xc) * width, float(yc) * height, float(w) * width, float(h) * height
        classes.append(int(c))
        boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return classes, boxes


def compare_to_baseline(rows, baseline="pytorch", tolerance=FP32_TOLERANCE):
    """Add delta_map50 / delta_map50_95 / within_tolerance to each row.

    Deltas are row - baseline. A row is within tolerance when BOTH |deltas|
    are <= tolerance. Rows are copied, not mutated.
    """
    base = next((r for r in rows if r["backend"] == baseline), None)
    if base is None:
        raise ValueError(f"no {baseline!r} row to compare against")
    out = []
    for r in rows:
        d50 = round(r["map50"] - base["map50"], 5)
        d = round(r["map50_95"] - base["map50_95"], 5)
        out.append({**r, "delta_map50": d50, "delta_map50_95": d,
                    "within_tolerance": abs(d50) <= tolerance and abs(d) <= tolerance})
    return out


def backend_for(model_path):
    """The benchmark_speed backend name whose loader accepts this path."""
    for name, cls in bs.BACKENDS.items():
        if cls.accepts(model_path):
            return name
    raise ValueError(f"no backend accepts {model_path}")


# ════════════════════════════════════════════════════════════════════
# Dataset prep
# ════════════════════════════════════════════════════════════════════

def ensure_labels(images_dir=bs.VISDRONE_VAL_IMAGES, labels_dir=LABELS_DIR):
    """Convert VisDrone annotations to YOLO labels if any are missing."""
    from PIL import Image

    ann_dir = images_dir.parent / "annotations"
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in bs.IMAGE_EXTS)
    if labels_dir.is_dir() and all((labels_dir / f"{p.stem}.txt").exists() for p in images):
        return labels_dir
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Converting {len(images)} VisDrone annotations -> {bs.display_path(labels_dir)}")
    for p in images:
        width, height = Image.open(p).size
        text = (ann_dir / f"{p.stem}.txt").read_text(encoding="utf-8")
        (labels_dir / f"{p.stem}.txt").write_text("".join(visdrone_to_yolo(text, width, height)),
                                                 encoding="utf-8")
    return labels_dir


def write_data_yaml(path=DATA_YAML):
    """Minimal dataset yaml for YOLO.val() over the val split only."""
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(VISDRONE_NAMES))
    path.write_text(f"path: {VAL_DIR.as_posix()}\ntrain: images\nval: images\nnames:\n{names}\n",
                    encoding="utf-8")
    return path


# ════════════════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════════════════

def score_model(model_path, image_paths, imgsz):
    """mAP of one model over image_paths via the shared loop.

    Preprocess and inference are benchmark_speed's backend stages (same load
    path, same CPU-only guard, same letterbox). Post-processing is NOT
    predict()'s: it is the validator's NMS, with multi_label=True, so one box
    may be scored under several classes. predict() keeps only the top class
    per box, and scoring that cost ~0.03 mAP50 against stock val on VisDrone
    (pedestrian/people and car/van are routinely confused).
    """
    import cv2
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import nms, ops
    from ultralytics.utils.metrics import DetMetrics, box_iou

    backend = bs.BACKENDS[backend_for(model_path)]()
    backend.load(model_path, imgsz, VAL_CONF)
    validator = DetectionValidator()          # only for match_predictions / iouv
    metrics = DetMetrics(names=dict(enumerate(VISDRONE_NAMES)))
    niou = validator.niou
    shapes = {}

    for i, p in enumerate(image_paths, 1):
        img = cv2.imread(str(p))
        with torch.inference_mode():
            x = backend.preprocess(img)
            raw = backend.infer(x)
            det = nms.non_max_suppression(raw, VAL_CONF, VAL_IOU, nc=0, multi_label=True,
                                          max_det=VAL_MAX_DET)[0]
        key = "x".join(str(d) for d in backend.input_shape(x))
        shapes[key] = shapes.get(key, 0) + 1
        p_box = ops.scale_boxes(x.shape[2:], det[:, :4].clone(), img.shape[:2]).cpu()
        p_conf, p_cls = det[:, 4].cpu(), det[:, 5].cpu()

        height, width = img.shape[:2]
        classes, boxes = yolo_to_xyxy((LABELS_DIR / f"{p.stem}.txt").read_text(encoding="utf-8"),
                                      width, height)
        t_cls = torch.tensor(classes, dtype=torch.float32)
        t_box = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if len(t_cls) == 0 or len(p_cls) == 0:
            tp = np.zeros((len(p_cls), niou), dtype=bool)
        else:
            tp = validator.match_predictions(p_cls, t_cls, box_iou(t_box, p_box)).cpu().numpy()
        metrics.update_stats({
            "tp": tp,
            "conf": p_conf.numpy(),
            "pred_cls": p_cls.numpy(),
            "target_cls": t_cls.numpy(),
            "target_img": np.unique(t_cls.numpy()),
            "im_name": p.name,
        })
        if i % 100 == 0:
            print(f"    {i}/{len(image_paths)}")

    metrics.process()
    return {
        "backend": backend.name,
        "model": {"path": bs.display_path(model_path), **bs.model_fingerprint(model_path)},
        "map50": round(float(metrics.box.map50), 5),
        "map50_95": round(float(metrics.box.map), 5),
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "input_shapes": dict(sorted(shapes.items())),
    }


def reference_val(pt_path):
    """Stock Ultralytics YOLO.val() on the .pt — a check on score_model."""
    from ultralytics import YOLO

    m = YOLO(str(pt_path), task="detect").val(
        data=str(write_data_yaml()), imgsz=640, batch=16, conf=VAL_CONF, iou=VAL_IOU,
        max_det=VAL_MAX_DET, device="cpu", plots=False, verbose=False, workers=0)
    return {
        "method": "ultralytics YOLO.val(imgsz=640, batch=16, rect)",
        "model": bs.display_path(pt_path),
        "map50": round(float(m.box.map50), 5),
        "map50_95": round(float(m.box.map), 5),
    }


# ════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════

def format_table(entry):
    lines = [
        "",
        f"VisDrone val accuracy — {entry['dataset']['num_images']} images, imgsz {entry['imgsz']}, "
        f"conf {entry['conf']}, NMS IoU {entry['iou']}",
        f"  {'backend':<10} {'input':<13} {'mAP50':>8} {'mAP50-95':>9} {'d mAP50':>9} {'d 50-95':>9}  ok?",
        "  " + "-" * 70,
    ]
    for r in entry["results"]:
        flag = "yes" if r["within_tolerance"] else f"NO (>{entry['tolerance']})"
        shape = ",".join(r.get("input_shapes", {})) or "?"
        lines.append(f"  {r['backend']:<10} {shape:<13} {r['map50']:>8.4f} {r['map50_95']:>9.4f} "
                     f"{r['delta_map50']:>+9.4f} {r['delta_map50_95']:>+9.4f}  {flag}")
    ref = entry.get("reference")
    if ref:
        lines += ["  " + "-" * 70,
                  f"  reference: {ref['method']}: mAP50 {ref['map50']:.4f}, mAP50-95 {ref['map50_95']:.4f}"]
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VisDrone val mAP for each exported format.")
    p.add_argument("--models", nargs="+", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--imgsz", type=bs.parse_imgsz, default=DEFAULT_IMGSZ)
    p.add_argument("--images", type=int, default=None, help="first N val images (default: all)")
    p.add_argument("--no-reference", action="store_true", help="skip stock YOLO.val() on the .pt")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for m in args.models:
        if not m.exists():
            sys.exit(f"model not found: {m}")
    bs.ensure_visdrone_val()
    ensure_labels()
    all_images = sorted(p for p in bs.VISDRONE_VAL_IMAGES.iterdir() if p.suffix.lower() in bs.IMAGE_EXTS)
    images = all_images[:args.images] if args.images else all_images

    rows = []
    for m in args.models:
        print(f"  Scoring {bs.display_path(m)} on {len(images)} images ...")
        rows.append(score_model(m, images, args.imgsz))
    reference = None
    pt = next((m for m in args.models if m.suffix == ".pt"), None)
    if pt is not None and not args.no_reference and args.images is None:
        print("  Reference: stock YOLO.val() on the .pt ...")
        reference = reference_val(pt)

    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {"name": "visdrone-val", "num_images": len(images),
                    "names_sha256": bs.names_digest([p.name for p in images])},
        "imgsz": args.imgsz,
        "conf": VAL_CONF,
        "iou": VAL_IOU,
        "max_det": VAL_MAX_DET,
        "tolerance": FP32_TOLERANCE,
        "results": compare_to_baseline(rows),
        "reference": reference,
    }
    print(format_table(entry))
    if not args.no_save:
        bs.append_entry(entry, RESULTS_FILE)
        print(f"  Appended to {bs.display_path(RESULTS_FILE)}")


if __name__ == "__main__":
    main()
