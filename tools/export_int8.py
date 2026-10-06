"""
JARVIS — OpenVINO INT8 Export (post-training quantization)
==========================================================
Quantizes the VisDrone detector to INT8 with NNCF, calibrated on a fixed,
recorded subset of VisDrone TRAIN, at the same static 384x640 shape as the
FP32 exports. benchmark_speed.py / benchmark_export_accuracy.py then measure it.

    VisDrone train zip  ->  every k-th image by sorted name (default 300)
        ->  extract just those to data/visdrone/VisDrone2019-DET-train-calib/
        ->  YOLO.export(format="openvino", int8=True, imgsz=[384, 640])
        ->  results/int8_calibration.json  (which images, settings, versions)
            results/int8_export_requirements.txt  (pip freeze of the export env)

CALIBRATION SET
  * TRAIN only, never val. Calibrating on the images that are then scored
    would leak the test set into the quantization ranges. The script asserts
    the two name sets are disjoint before exporting.
  * Strided, not a prefix. VisDrone names are <sequence>_<frame>, so the first
    300 by name are consecutive frames from a handful of flights; every k-th
    image (k = total // n) spans the whole split. Still fully deterministic.
  * All n are used: Ultralytics feeds them through a batch-1 dataloader and
    nncf.quantize's default subset_size is 300. Preset MIXED (symmetric
    weights, asymmetric activations); the Detect head's Add/Sub/Mul/Div and
    Sigmoid stay in floating point (Ultralytics' ignored scope).

ENVIRONMENT
  NNCF requires numpy < 2.5 and the project .venv pins numpy 2.5.1 for the face
  pipeline, so this script runs in a SEPARATE throwaway venv (torch, ultralytics
  and openvino at the same versions as .venv, plus nncf). Its exact package list
  is written to results/int8_export_requirements.txt so the export can be
  recreated:

      python -m venv export-venv
      export-venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
          torch==2.13.0 torchvision==0.28.0
      export-venv/Scripts/python -m pip install -r results/int8_export_requirements.txt
      export-venv/Scripts/python tools/export_int8.py --train-zip path/to/VisDrone2019-DET-train.zip

  The resulting IR needs only openvino at inference time, so benchmarking runs
  in the normal .venv.

Usage:
    python tools/export_int8.py --train-zip VisDrone2019-DET-train.zip [--calib-images 300]
"""

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import benchmark_export_accuracy as ea  # noqa: E402
import benchmark_speed as bs  # noqa: E402

TRAIN_ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip"
ZIP_ROOT = "VisDrone2019-DET-train"
CALIB_DIR = bs.VISDRONE_DIR / "VisDrone2019-DET-train-calib"
CALIB_YAML = bs.VISDRONE_DIR / "visdrone_calib.yaml"
RECORD_FILE = bs.RESULTS_DIR / "int8_calibration.json"
FREEZE_FILE = bs.RESULTS_DIR / "int8_export_requirements.txt"

DEFAULT_MODEL = ea.MODEL_STEM.with_suffix(".pt")
DEFAULT_CALIB_IMAGES = 300
DEFAULT_IMGSZ = [384, 640]


# ════════════════════════════════════════════════════════════════════
# Pure helpers — unit-tested (tests/test_export_int8.py).
# ════════════════════════════════════════════════════════════════════

def strided_selection(names, n):
    """Every k-th name of the sorted list, k = len // n; exactly n names.

    Deterministic for a given name set. Raises if fewer than n names exist.
    """
    ordered = sorted(names)
    if n < 1 or len(ordered) < n:
        raise ValueError(f"cannot select {n} from {len(ordered)} names")
    step = len(ordered) // n
    return ordered[::step][:n]


def sequence_of(name):
    """VisDrone '<sequence>_<frame>_<...>.jpg' -> '<sequence>'."""
    return name.split("_", 1)[0]


def build_record(*, timestamp, model, output, imgsz, names, total_train, step,
                 val_overlap, versions, freeze_file):
    """The calibration record written to results/int8_calibration.json."""
    return {
        "timestamp": timestamp,
        "model": model,
        "output": output,
        "imgsz": imgsz,
        "quantization": {
            "tool": "nncf.quantize via ultralytics YOLO.export(format='openvino', int8=True)",
            "preset": "MIXED",
            "subset_size": len(names),
            "ignored_scope": "Detect head Add/Sub/Mul/Div + Sigmoid (ultralytics default)",
        },
        "calibration": {
            "dataset": "visdrone-train",
            "selection": f"1 in {step} of {total_train} train images sorted by filename (indices 0, {step}, ...)",
            "num_images": len(names),
            "num_sequences": len({sequence_of(n) for n in names}),
            "first": names[0],
            "last": names[-1],
            "names_sha256": bs.names_digest(names),
            "val_overlap": val_overlap,
            "names": names,
        },
        "export_environment": {"versions": versions, "freeze_file": freeze_file},
    }


# ════════════════════════════════════════════════════════════════════
# Calibration set
# ════════════════════════════════════════════════════════════════════

def prepare_calibration(train_zip, n):
    """Extract the n selected train images (+ YOLO labels) into CALIB_DIR."""
    with zipfile.ZipFile(train_zip) as zf:
        members = {Path(m).name: m for m in zf.namelist()
                   if m.startswith(f"{ZIP_ROOT}/images/") and m.lower().endswith(".jpg")}
        names = strided_selection(members, n)
        step = len(members) // n
        images_dir, ann_dir = CALIB_DIR / "images", CALIB_DIR / "annotations"
        # Start clean so a changed --calib-images cannot leave stale extras.
        for d in (images_dir, ann_dir, CALIB_DIR / "labels"):
            if d.is_dir():
                for f in d.iterdir():
                    f.unlink()
            d.mkdir(parents=True, exist_ok=True)
        for name in names:
            (images_dir / name).write_bytes(zf.read(members[name]))
            ann = f"{ZIP_ROOT}/annotations/{Path(name).stem}.txt"
            (ann_dir / f"{Path(name).stem}.txt").write_bytes(zf.read(ann))
    ea.ensure_labels(images_dir, CALIB_DIR / "labels")
    return names, len(members), step


def write_calib_yaml(path=CALIB_YAML):
    """Dataset yaml whose `val` is the calibration set: Ultralytics calibrates on `val`."""
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(ea.VISDRONE_NAMES))
    path.write_text(f"path: {CALIB_DIR.as_posix()}\ntrain: images\nval: images\nnames:\n{names}\n",
                    encoding="utf-8")
    return path


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main(argv=None):
    p = argparse.ArgumentParser(description="OpenVINO INT8 export calibrated on VisDrone train.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--train-zip", type=Path, required=True,
                   help=f"VisDrone2019-DET-train.zip ({TRAIN_ZIP_URL})")
    p.add_argument("--calib-images", type=int, default=DEFAULT_CALIB_IMAGES)
    p.add_argument("--imgsz", type=bs.parse_imgsz, default=DEFAULT_IMGSZ)
    args = p.parse_args(argv)

    try:
        import nncf
    except ImportError:
        sys.exit("nncf is not installed - run this in the export venv (see module docstring)")
    import numpy
    import openvino
    import torch
    import ultralytics
    from ultralytics import YOLO

    names, total, step = prepare_calibration(args.train_zip, args.calib_images)
    val_names = {q.name for q in bs.VISDRONE_VAL_IMAGES.iterdir()} if bs.VISDRONE_VAL_IMAGES.is_dir() else set()
    overlap = sorted(set(names) & val_names)
    if overlap:
        sys.exit(f"calibration set overlaps val: {overlap[:5]}")
    print(f"  Calibration: {len(names)} train images (1 in {step} of {total}), "
          f"{len({sequence_of(n) for n in names})} sequences, 0 in val")

    out = YOLO(str(args.model), task="detect").export(
        format="openvino", int8=True, data=str(write_calib_yaml()), imgsz=args.imgsz,
        batch=1, fraction=1.0, dynamic=False, half=False, device="cpu")
    out = Path(out)

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True, check=True).stdout
    FREEZE_FILE.write_text(freeze, encoding="utf-8")

    record = build_record(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        model={"path": bs.display_path(args.model), **bs.model_fingerprint(args.model)},
        output={"path": bs.display_path(out), **bs.model_fingerprint(out)},
        imgsz=args.imgsz,
        names=names,
        total_train=total,
        step=step,
        val_overlap=len(overlap),
        versions={"nncf": nncf.__version__, "openvino": openvino.__version__,
                  "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                  "numpy": numpy.__version__, "python": sys.version.split()[0]},
        freeze_file=bs.display_path(FREEZE_FILE),
    )
    RECORD_FILE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"  Exported {bs.display_path(out)}")
    print(f"  Wrote {bs.display_path(RECORD_FILE)} and {bs.display_path(FREEZE_FILE)}")


if __name__ == "__main__":
    main()
