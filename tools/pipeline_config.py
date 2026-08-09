"""
JARVIS Face Pipeline — Locked Operating Points
==============================================
The deployment settings the live presence service reads, together with the
measurements that justify them. Import from here; never retype a number at a
call site.

WHY A MODULE AND NOT A JSON ARTIFACT. The face gallery is a generated artifact
(data/gallery.npz) because it holds data no human could write — 133 embedding
vectors produced by a model. The values in THIS file are the opposite: human
decisions informed by measurement. Nothing generates them, so config-as-code is
the honest shape. It gets reviewed in a diff, explained in a commit message,
imported without a parse-failure path, and it never sits in results/ pretending
to be regenerated output.

WHY THE RECOGNITION THRESHOLD IS NOT HERE. It travels inside the gallery file
instead, because it is only meaningful next to the encoder that produced the
vectors it will be compared against — ship a gallery, ship its threshold. See
build_gallery.py. Detection has no such coupling: a confidence threshold means
the same thing on any frame, so it belongs to the pipeline, not to an artifact.

ANTI-DRIFT. Every number below is pinned to the benchmark run that produced it,
by timestamp. tests/test_pipeline_config.py re-reads
results/benchmarks_detection.json, finds those exact runs, and fails if the
recorded measurements no longer match or if the shipped value drifts outside
the range its evidence supports. Changing a value without re-measuring breaks
the build — which is the point.
"""

__all__ = [
    "DETECTION_DETECTOR",
    "DETECTION_WEIGHTS",
    "DETECTION_CONFIDENCE_THRESHOLD",
    "DETECTION_THRESHOLD_PROVENANCE",
]


# ══════════════════════════════════════════════════════════════════
# Face detection — which detector runs on every frame, and at what
# confidence a detection is believed.
# ══════════════════════════════════════════════════════════════════

DETECTION_DETECTOR = "yolo"
DETECTION_WEIGHTS = "yolov8n-face.pt"

# Accept a detection when its confidence is >= this value.
#
# 0.57 sits between the F1-max thresholds independently measured on the two
# HARD datasets — MAFA 0.5812 (occluded/masked faces) and WIDER FACE val 0.5727
# (crowds, scale extremes, blur). Those two agree to within 0.009 despite
# testing different failure modes, which is what makes the value trustworthy
# rather than a single dataset's quirk.
#
# FDDB's 0.694 is deliberately NOT averaged in. FDDB is the easy set (F1 0.937
# vs ~0.80 on the other two): large, mostly frontal, unoccluded faces. Its
# optimum sits high because almost everything it contains is easy to be
# confident about, so folding it in would bias the threshold upward and start
# dropping the hard faces the household camera will actually produce at a
# distance, in profile, or backlit.
#
# Erring low is also the right direction for this pipeline: a missed face is a
# person JARVIS fails to greet, while a false detection costs one wasted
# embedding that the recognition stage rejects anyway at its own threshold.
DETECTION_CONFIDENCE_THRESHOLD = 0.57

# The evidence, pinned by run timestamp so the value can never become a magic
# number. Cross-checked against results/benchmarks_detection.json by
# tests/test_pipeline_config.py.
DETECTION_THRESHOLD_PROVENANCE = {
    "value": 0.57,
    "decided_on": "2026-08-08",
    "decided_from": "F1-max confidence, full-dataset YOLOv8n-face runs",
    "detector": "yolo",
    "iou_threshold": 0.5,
    "measurements": [
        {
            "dataset": "mafa",
            "run_timestamp": "2026-08-08T19:08:59Z",
            "num_images": 4935,
            "f1_threshold": 0.5812,
            "f1": 0.8094,
            "ap": 0.7788,
            "role": "hard - occlusion and masks; supports the chosen value",
        },
        {
            "dataset": "widerface",
            "run_timestamp": "2026-08-08T20:10:27Z",
            "num_images": 3226,
            "f1_threshold": 0.5727,
            "f1": 0.8036,
            "ap": 0.8644,
            "role": "hard - crowds, scale, blur; supports the chosen value",
        },
        {
            "dataset": "fddb",
            "run_timestamp": "2026-08-08T19:31:06Z",
            "num_images": 2845,
            "f1_threshold": 0.6940,
            "f1": 0.9372,
            "ap": 0.9488,
            "role": "easy - reference only, EXCLUDED from the decision",
        },
    ],
    # The datasets whose F1-max thresholds bracket the shipped value. The test
    # asserts DETECTION_CONFIDENCE_THRESHOLD lies within their span.
    "supported_by": ["mafa", "widerface"],
    # UNVERIFIABLE FROM THE ARTIFACT. These three runs pre-date min_box_size
    # being recorded in the run metadata (added in this repo's history just
    # before this file), so their stored entries carry no min_box_size field.
    # The flag defaults to 0 (keep every ground-truth box) and there is no
    # reason to think it was passed, but that cannot be confirmed from the
    # stored run — and min_box_size moves these numbers, since filtering out
    # small faces removes the hardest cases and lifts both AP and the F1-max
    # threshold. Treat as unknown-but-probably-0 until a re-measure records it.
    "min_box_size": None,
    "min_box_size_recorded_in_artifact": False,
}
