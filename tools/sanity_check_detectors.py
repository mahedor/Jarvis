"""
Sanity check for the face_utils detector backends.
=================================================
THROWAWAY DIAGNOSTIC — not production code. Run this on real images to
confirm each detector returns sane boxes BEFORE trusting any benchmark
numbers. For every backend it prints the raw boxes/confidences and saves an
annotated sanity_<image>_<detector>.jpg so you can eyeball the result.

Usage:
    python tools/sanity_check_detectors.py path/to/image.jpg
    python tools/sanity_check_detectors.py tools/images/*.jpg     # many at once

Drop test images in tools/images/. Annotated results are written to
tools/output/ by default (override with --outdir). Both folders are
git-ignored so test photos and outputs never get pushed.

CPU-only environment. min_confidence=0.0 so every detection is shown.
"""

import argparse
import os

import cv2

import face_utils  # same directory

DETECTORS = ["mediapipe", "mtcnn", "retinaface", "yolo"]

# Default output dir is tools/output/ relative to THIS file, so results always
# land in the same place no matter what directory you run the script from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTDIR = os.path.join(_HERE, "output")

# Backends that need an explicit model file, kept in tools/weights/ relative
# to this script:
#   - YOLO: a face-weights file (the stock COCO model detects "person").
#   - MediaPipe (Tasks API): the BlazeFace short-range .tflite model.
_YOLO_WEIGHTS = os.path.join(_HERE, "weights", "yolov8n-face.pt")
_MEDIAPIPE_MODEL = os.path.join(_HERE, "weights", "blaze_face_short_range.tflite")

# Per-backend constructor kwargs beyond the shared min_confidence. Backends
# not listed here use the shared call as-is.
_EXTRA_KWARGS = {
    "yolo": {"weights": _YOLO_WEIGHTS},
    "mediapipe": {"model_asset_path": _MEDIAPIPE_MODEL},
}


def main():
    parser = argparse.ArgumentParser(description="Visually sanity-check the face detectors.")
    parser.add_argument(
        "images", nargs="+", help="one or more image files to run detection on"
    )
    parser.add_argument(
        "--outdir",
        default=_DEFAULT_OUTDIR,
        help="directory for annotated output images (default: tools/output/)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="minimum detection confidence to keep (default: 0.0 = show every "
        "detection). NOTE: at 0.0 mediapipe and yolo flood low-confidence boxes "
        "and counts are NOT comparable across backends; use e.g. 0.5 to compare.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load each detector ONCE and reuse it across every image — model weights
    # are expensive to load, so we never want to rebuild them per image.
    detectors = {}
    for name in DETECTORS:
        try:
            detectors[name] = face_utils.load_detector(
                name, min_confidence=args.min_confidence, **_EXTRA_KWARGS.get(name, {})
            )
        except Exception as exc:  # missing backend, bad weights, etc.
            print(f"WARNING: detector {name!r} unavailable ({type(exc).__name__}: {exc})")
            detectors[name] = None

    for image_path in args.images:
        print(f"\n######## {image_path} ########")
        if not os.path.isfile(image_path):
            print(f"  Error: image not found; skipping")
            continue
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error: cv2 could not read image (unsupported/corrupt?); skipping")
            continue

        # Stem of the input file keeps each image's outputs distinct so runs on
        # multiple images never overwrite one another.
        stem = os.path.splitext(os.path.basename(image_path))[0]

        counts = {}
        for name in DETECTORS:
            print(f"\n=== {name} ===")
            detector = detectors[name]
            if detector is None:
                print("  (skipped: backend unavailable)")
                counts[name] = None
                continue
            try:
                detections = detector.detect(image)
            except Exception as exc:
                print(f"  WARNING: detect failed ({type(exc).__name__}: {exc})")
                counts[name] = None
                continue

            for det in detections:
                print(f"  box={det['box']}  confidence={det['confidence']:.3f}")
            counts[name] = len(detections)
            print(f"  -> {len(detections)} face(s)")

            out_path = os.path.join(args.outdir, f"sanity_{stem}_{name}.jpg")
            cv2.imwrite(out_path, face_utils.draw_detections(image, detections))
            print(f"  saved {out_path}")

        print(f"\n=== summary: {stem} ===")
        for name in DETECTORS:
            c = counts.get(name)
            print(f"  {name:12s} {'(skipped)' if c is None else f'{c} face(s)'}")


if __name__ == "__main__":
    main()
