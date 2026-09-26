"""
JARVIS Face Recognition — Grouping Threshold Derivation
=======================================================
Derives the GROUPING threshold: the cosine above which two crops of UNKNOWN
faces are treated as the same stranger. It answers "are these two crops the
same person?" for people who are not in the gallery, so that repeat visits by
one stranger can be grouped instead of counted as N different strangers.

    LFW (identities with >= 2 images)
        -> YOLO detect @ pipeline_config conf -> most central box
        -> collect_faces quality gate -> crop @ margin 0.35
        -> ArcFace -> L2 normalize
        -> genuine pairs (within identity, capped) + impostor pairs
        -> cosine per pair -> FMR/FNMR sweep -> operating points

----------------------------------------------------------------------
THIS IS NOT THE RECOGNITION THRESHOLD. The recognition threshold (0.342, in
build_gallery.py and inside data/gallery.npz) answers a different question:
"does this probe match one of the enrolled household members?" It was chosen
against a multi-reference gallery of curated enrollment crops, scored by max
over each person's references. Grouping compares ONE crop against ONE crop,
with no curation on either side, so its score distributions are different
and so is its threshold. The two must never be substituted for each other.

This script only PRODUCES THE ARTIFACT. It writes nothing into any config,
never reads or writes data/gallery.npz, and does not import build_gallery. A
value chosen from results/grouping_threshold.json has to be adopted by hand,
in a diff, like every other locked operating point.
----------------------------------------------------------------------

WHY FNMR 10% IS THE PRIMARY POINT. A grouping false non-match splits one
stranger into two; a false match merges two strangers into one. The primary
point pins the split rate at 10% and reports what that costs in merges. The
other points (FNMR 5%/1%, FMR 1%/0.1%, best-F1) are there so the choice can be
revisited without re-running.

EMBEDDING PATH = DEPLOYMENT PATH. Every image goes through the same detector,
confidence, crop margin, quality gate and encoder the live presence service and
collect_faces.py use — not through InsightFace's own detect-and-embed. The
detector confidence is read from pipeline_config; the gate is collect_faces'
own quality_check, imported, not re-typed.

ONE FACE PER IMAGE, AND IT IS THE CENTRAL ONE. LFW images are labelled by the
person in the middle of the frame; some contain bystanders. When YOLO returns
more than one box, the MOST CENTRAL box is taken (not the largest — a bystander
nearer the camera can be larger). If that central box then fails the quality
gate the image is DROPPED rather than falling back to another box, because any
other box is by construction not the labelled person.

PAIRS. Genuine pairs are capped at a flat number per identity, sampled without
replacement, so every identity carries (at most) equal weight: uncapped, C(n,2)
lets the few heavily photographed identities (George W Bush alone has 530
images -> 140,185 pairs) swamp everyone else. Impostor pairs sample two
DISTINCT identities uniformly, then one image from each — again weighting by
identity, not by image count.

Usage:
    python tools/derive_grouping_threshold.py
    python tools/derive_grouping_threshold.py --min-blur 60 --seed 7

The first run downloads LFW (~200 MB) into data/lfw, shared with
benchmark_recognition.py.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import sklearn

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import face_utils  # noqa: E402  - after sys.path so tools/ is importable standalone
import pipeline_config  # noqa: E402
from collect_faces import (  # noqa: E402
    REJECT_LOW_CONFIDENCE,
    REJECT_TOO_BLURRY,
    REJECT_TOO_SMALL,
    quality_check,
    variance_of_laplacian,
)
from recognition_metrics import (  # noqa: E402
    best_f1_threshold,
    dprime,
    histogram,
    l2_normalize,
    score_sweep,
    summarize_scores,
    tar_at_far,
)

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "grouping_threshold.json"
DEFAULT_LFW_DIR = REPO_ROOT / "data" / "lfw"
DEFAULT_WEIGHTS = TOOLS_DIR / "weights" / pipeline_config.DETECTION_WEIGHTS

ENCODER = "arcface"
# Same value, same reason as presence_service.CROP_MARGIN and collect_faces'
# --margin default: the encoder re-detects and aligns inside .embed() and needs
# the room it had at enrollment.
CROP_MARGIN = 0.35
# collect_faces.py's --min-box-size default (the enrollment gate), not the 20 px
# the detection benchmark filtered at.
MIN_BOX_SIZE = 40
# The gate's confidence floor. collect_faces defaults to 0.5, but every box here
# already cleared the detector at DETECTION_CONFIDENCE_THRESHOLD, so a lower
# floor would be dead code. Using the deployment value keeps it honest.
GATE_MIN_CONFIDENCE = pipeline_config.DETECTION_CONFIDENCE_THRESHOLD
# collect_faces.py ships with the blur check OFF, and the enrollment run that
# built the gallery ran with it off (results/enrollment_summary.json). Matching
# that is the default; --min-blur turns it on. Blur scores are recorded either
# way.
DEFAULT_MIN_BLUR = None

# fetch_lfw_people kwargs shared with benchmark_recognition.load_strangers.
LFW_KWARGS = {"color": True, "resize": 1.0, "funneled": True}
MIN_FACES_PER_PERSON = 2

# WHICH PART OF EACH 250x250 LFW IMAGE THE DETECTOR SEES. sklearn's default
# slice_ cuts a tight 125x94 face window out of the middle: YOLO then finds a
# face that nearly fills the frame and the 0.35 crop margin is clipped away at
# the image edge, so the encoder never gets the room it gets in deployment.
# "full" hands YOLO the whole image, which is what a camera frame looks like.
# The first run (2026-09-22) used "sklearn-default" and is kept as the
# pre-cropped baseline; the artifact records slice_ so the two are told apart.
LFW_SLICES = {
    "full": (slice(0, 250), slice(0, 250)),
    "sklearn-default": (slice(70, 195), slice(78, 172)),
}
DEFAULT_SLICE = "full"

GENUINE_CAP_PER_IDENTITY = 10
NUM_IMPOSTOR_PAIRS = 100_000
DEFAULT_SEED = 42

PRIMARY_FNMR = 0.10
FNMR_TARGETS = [0.10, 0.05, 0.01]
FMR_TARGETS = [0.01, 0.001]
SWEEP_STEPS = 200
HIST_BINS = 40

REJECT_NO_FACE = "no_face"
REJECT_EMBED_FAILED = "embed_failed"
GATE_REASONS = (REJECT_LOW_CONFIDENCE, REJECT_TOO_SMALL, REJECT_TOO_BLURRY)

CAVEAT = (
    "Pairs come from funneled LFW: press photography of public figures, mostly "
    "frontal, well lit, in focus, and pre-aligned by the funneling step. That is "
    "easier than live webcam crops, which are smaller, blurrier, more off-angle "
    "and worse lit. Low-quality embeddings drift toward each other, so the live "
    "FMR will be higher than reported at the same threshold - and the live FNMR "
    "will be higher too. Treat every rate in this file as a best case."
)


# ══════════════════════════════════════════════════════════════════
# PURE LOGIC — no cv2, no disk, no models. Unit-tested in
# tests/test_derive_grouping_threshold.py.
# ══════════════════════════════════════════════════════════════════

def select_central_detection(detections, image_shape):
    """The detection nearest the image centre, or None if there are none.

    Central, not largest: LFW labels the person in the middle of the frame, and
    a bystander nearer the camera can have the bigger box.

    Inputs:
        detections (list[dict]): face_utils detector output, each with "box".
        image_shape (tuple): the image's .shape, (h, w, ...).
    Returns:
        dict | None: one element of `detections`.
    """
    if not detections:
        return None
    boxes = [d["box"] for d in detections]
    return detections[face_utils._most_central_index(boxes, image_shape)]


def sample_genuine_pairs(labels, cap, rng):
    """Within-identity pairs, at most `cap` per identity, without replacement.

    Inputs:
        labels (array-like): (N,) identity label per embedding.
        cap (int): maximum pairs drawn from any one identity.
        rng (np.random.Generator): seeded generator.
    Returns:
        np.ndarray: (P, 2) int64 embedding-index pairs, i < j within each pair.
        An identity with n embeddings contributes min(cap, n*(n-1)/2) pairs, so
        identities with fewer than 2 embeddings contribute nothing.
    """
    labels = np.asarray(labels)
    chunks = []
    for identity in np.unique(labels):  # sorted, so iteration order is fixed
        members = np.flatnonzero(labels == identity)
        n = members.size
        if n < 2:
            continue
        rows, cols = np.triu_indices(n, k=1)
        total = rows.size
        if total > cap:
            picks = rng.choice(total, size=cap, replace=False)
            rows, cols = rows[picks], cols[picks]
        chunks.append(np.stack([members[rows], members[cols]], axis=1))
    if not chunks:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(chunks).astype(np.int64)


def sample_impostor_pairs(labels, num_pairs, rng):
    """Cross-identity pairs: two distinct identities, then one image from each.

    Identities are drawn uniformly, so a heavily photographed identity is no
    more likely to appear than one with a single image. Pairs are drawn with
    replacement; at 100k draws over ~1,680 identities repeats are rare and
    harmless.

    Inputs:
        labels (array-like): (N,) identity label per embedding.
        num_pairs (int): how many pairs to draw.
        rng (np.random.Generator): seeded generator.
    Returns:
        np.ndarray: (num_pairs, 2) int64 embedding-index pairs whose labels
        always differ.
    Raises:
        ValueError: if fewer than 2 identities are present.
    """
    labels = np.asarray(labels)
    identities, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    m = identities.size
    if m < 2:
        raise ValueError(f"need >= 2 identities to form impostor pairs, got {m}")

    # Members of identity k are order[starts[k] : starts[k] + counts[k]].
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    first = rng.integers(m, size=num_pairs)
    second = rng.integers(m - 1, size=num_pairs)
    second = second + (second >= first)  # skip over `first`: always distinct

    def pick(ids):
        offsets = np.floor(rng.random(ids.size) * counts[ids]).astype(np.int64)
        return order[starts[ids] + offsets]

    return np.stack([pick(first), pick(second)], axis=1).astype(np.int64)


def pair_scores(embeddings, pairs):
    """Cosine similarity per pair. `embeddings` must already be L2-normalized."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    a = embeddings[pairs[:, 0]]
    b = embeddings[pairs[:, 1]]
    return np.einsum("ij,ij->i", a, b).astype(np.float64)


def error_rates(genuine, impostor, threshold):
    """FMR and FNMR at one threshold, "same person if score >= t".

    Returns:
        dict {"threshold", "fmr", "fnmr"}: FMR = impostor pairs scoring >= t,
        FNMR = genuine pairs scoring < t.
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    impostor = np.asarray(impostor, dtype=np.float64)
    return {
        "threshold": round(float(threshold), 6),
        "fmr": round(float(np.mean(impostor >= threshold)), 6) if impostor.size else 0.0,
        "fnmr": round(float(np.mean(genuine < threshold)), 6) if genuine.size else 0.0,
    }


def threshold_at_fnmr(genuine, fnmr_target):
    """The highest threshold whose FNMR does not exceed `fnmr_target`.

    Sets t to the k-th smallest genuine score, k = floor(target * n), so exactly
    k genuine scores fall strictly below it (fewer on ties). The mirror image of
    recognition_metrics.tar_at_far, which pins the other tail.

    Returns:
        float: the threshold. 0.0 if `genuine` is empty.
    """
    genuine = np.sort(np.asarray(genuine, dtype=np.float64))
    if genuine.size == 0:
        return 0.0
    k = int(np.floor(fnmr_target * genuine.size))
    return float(genuine[min(k, genuine.size - 1)])


def compute_operating_points(genuine, impostor):
    """Every operating point the artifact reports, from the two score sets.

    Returns:
        dict with:
            "primary": the FNMR = PRIMARY_FNMR point (threshold + FMR it costs)
            "fnmr_targets": one point per FNMR_TARGETS entry
            "fmr_targets": one point per FMR_TARGETS entry
            "best_f1": the F1-max point plus its precision/recall/F1
            "dprime": (mean_gen - mean_imp) / sqrt((var_gen + var_imp) / 2)
        Every point carries threshold, achieved fmr, achieved fnmr, and the
        target it was solved for.
    """
    fnmr_points = []
    for target in FNMR_TARGETS:
        point = error_rates(genuine, impostor, threshold_at_fnmr(genuine, target))
        fnmr_points.append({"target_fnmr": target, **point})

    fmr_points = []
    for target in FMR_TARGETS:
        threshold = tar_at_far(genuine, impostor, target)["threshold"]
        point = error_rates(genuine, impostor, threshold)
        fmr_points.append({"target_fmr": target, **point})

    f1_threshold, precision, recall, f1 = best_f1_threshold(genuine, impostor)
    best_f1 = {
        **error_rates(genuine, impostor, f1_threshold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "note": "F1 depends on the genuine:impostor pair ratio, which is set by "
                "the cap and the impostor count, not by the world. The least "
                "portable number in this file.",
    }

    primary = next(p for p in fnmr_points if p["target_fnmr"] == PRIMARY_FNMR)
    return {
        "primary": {"criterion": f"FNMR = {PRIMARY_FNMR:g}", **primary},
        "fnmr_targets": fnmr_points,
        "fmr_targets": fmr_points,
        "best_f1": best_f1,
        "dprime": round(dprime(genuine, impostor), 6),
    }


def error_sweep(genuine, impostor, steps=SWEEP_STEPS):
    """FMR/FNMR on a fixed cosine grid over [-1, 1], for plotting later.

    Built on recognition_metrics.score_sweep so the grid matches the
    recognition benchmark's: FMR is its FAR, FNMR is 1 - its TAR.
    """
    return [
        {
            "threshold": row["threshold"],
            "fmr": row["far"],
            "fnmr": round(1.0 - row["tar"], 6),
        }
        for row in score_sweep(genuine, impostor, steps=steps)
    ]


def slice_to_json(slice_):
    """(slice(70, 195), slice(78, 172)) -> [[70, 195], [78, 172]] (rows, cols)."""
    return [[s.start, s.stop] for s in slice_]


def lfw_file_index(data_folder, min_faces_per_person):
    """Enumerate LFW exactly as sklearn's _fetch_lfw_people does, minus decoding.

    Person folders sorted by name, files sorted within each, persons with fewer
    than `min_faces_per_person` files dropped, names with "_" -> " ", labels as
    indices into the sorted unique names, then the same RandomState(42) shuffle.
    Mirroring the order keeps image indices comparable with a fetch_lfw_people
    load of the same subset.

    Returns:
        tuple (paths, labels, names): list[Path], (N,) int64, (M,) str array.
    """
    data_folder = Path(data_folder)
    person_names, paths = [], []
    for folder in sorted(data_folder.iterdir(), key=lambda f: f.name):
        if not folder.is_dir():
            continue
        files = sorted(folder.iterdir(), key=lambda f: f.name)
        if len(files) >= min_faces_per_person:
            person_names.extend([folder.name.replace("_", " ")] * len(files))
            paths.extend(files)
    if not paths:
        raise ValueError(f"no identity in {data_folder} has >= {min_faces_per_person} images")

    names = np.unique(person_names)
    labels = np.searchsorted(names, person_names).astype(np.int64)
    order = np.arange(len(paths))
    np.random.RandomState(42).shuffle(order)
    return [paths[i] for i in order], labels[order], names


def build_payload(*, genuine, impostor, settings, dataset, pipeline_counts, pairs):
    """Assemble the JSON record for one derivation run.

    Pure: every input is passed in, so the shape of the artifact is testable
    without LFW, YOLO or ArcFace.
    """
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # LazyLFW decodes through sklearn's private _load_imgs. The decode test
        # catches a break; this says which version an entry was produced under.
        "sklearn_version": sklearn.__version__,
        "purpose": "GROUPING threshold - are two crops of unknown faces the same "
                   "stranger? Separate from, and never a substitute for, the "
                   "recognition threshold carried in data/gallery.npz.",
        "adopted": False,
        "adoption_note": "Produced by tools/derive_grouping_threshold.py, which writes "
                         "no config. Adopting a value is a separate, reviewed change.",
        "caveat": CAVEAT,
        "settings": settings,
        "dataset": dataset,
        "pipeline_counts": pipeline_counts,
        "pairs": pairs,
        "distributions": {
            "genuine": summarize_scores(genuine),
            "impostor": summarize_scores(impostor),
        },
        "operating_points": compute_operating_points(genuine, impostor),
        "sweep": error_sweep(genuine, impostor),
        "histograms": {
            "genuine": histogram(genuine, bins=HIST_BINS),
            "impostor": histogram(impostor, bins=HIST_BINS),
        },
    }


# ══════════════════════════════════════════════════════════════════
# I/O — LFW, detector, encoder.
# ══════════════════════════════════════════════════════════════════

class LazyLFW:
    """LFW images decoded one at a time, through sklearn's own decoder.

    fetch_lfw_people materialises every image as float32 up front and caches
    the array via joblib under --lfw-dir. At the full 250x250 that is ~6.9 GB
    for this subset, which does not fit. This calls the same per-file decode
    (sklearn.datasets._lfw._load_imgs: PIL crop to slice_, /255 float32) on one
    path at a time, so pixels are identical to fetch_lfw_people's for the same
    kwargs. _load_imgs is private API - sklearn 1.8.0 at time of writing.
    """

    def __init__(self, paths, slice_):
        from sklearn.datasets._lfw import _load_imgs

        self._load = _load_imgs
        self.paths = paths
        self.slice_ = slice_

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rgb = self._load([str(self.paths[index])], self.slice_,
                         LFW_KWARGS["color"], LFW_KWARGS["resize"])[0]
        return to_bgr_uint8(rgb, 255.0)

    def shape(self):
        """(h, w) of every image under this slice_."""
        return tuple(s.stop - s.start for s in self.slice_)


def load_lfw_multi(lfw_dir, slice_):
    """LFW restricted to identities with >= MIN_FACES_PER_PERSON images.

    Returns:
        tuple (images, labels, names): LazyLFW, (N,) identity index, names.
    """
    from sklearn.datasets._lfw import _check_fetch_lfw

    lfw_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Loading LFW (cache: {lfw_dir}; first run downloads ~200 MB)...")
    _, data_folder = _check_fetch_lfw(data_home=str(lfw_dir), funneled=LFW_KWARGS["funneled"])
    paths, labels, names = lfw_file_index(data_folder, MIN_FACES_PER_PERSON)
    images = LazyLFW(paths, slice_)
    h, w = images.shape()
    print(f"  {len(images)} images across {len(names)} identities (image size {w}x{h}).")
    return images, labels, names


def to_bgr_uint8(rgb_image, scale):
    """sklearn's float RGB image -> the BGR uint8 every detector expects."""
    rgb = np.clip(rgb_image * scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def embed_images(images, labels, detector, encoder, min_blur):
    """Run the deployment path over every image; keep one embedding per image.

    Returns:
        tuple (embeddings, kept_labels, counts, blur_scores):
            embeddings (np.ndarray): (K, D) L2-normalized.
            kept_labels (np.ndarray): (K,) identity per embedding.
            counts (dict): per-reason drop counts and pipeline totals.
            blur_scores (list[float]): every crop that was measured.
    """
    counts = Counter()
    blur_scores = []
    embeddings = []
    kept_labels = []
    total = len(images)
    started = time.perf_counter()

    for index in range(total):
        image = images[index]
        counts["images"] += 1

        detections = detector.detect(image)
        if not detections:
            counts[REJECT_NO_FACE] += 1
            continue
        if len(detections) > 1:
            counts["multiple_boxes"] += 1

        detection = select_central_detection(detections, image.shape)
        box = detection["box"]
        confidence = float(detection["confidence"])

        # Cheap checks first, exactly as collect_faces.collect_crops does.
        screened = quality_check(box, confidence, None, MIN_BOX_SIZE, GATE_MIN_CONFIDENCE, min_blur)
        if not screened.keep:
            counts[screened.reason] += 1
            continue

        crop = face_utils.crop_face(image, box, margin=CROP_MARGIN)
        if crop.size == 0:
            counts[REJECT_TOO_SMALL] += 1
            continue

        blur = variance_of_laplacian(crop)
        blur_scores.append(blur)
        result = quality_check(box, confidence, blur, MIN_BOX_SIZE, GATE_MIN_CONFIDENCE, min_blur)
        if not result.keep:
            counts[result.reason] += 1
            continue

        try:
            vector = encoder.embed(crop)
        except Exception as exc:  # one bad crop must not sink a 15-minute run
            print(f"  [warn] embed failed on image {index}: {exc}")
            counts[REJECT_EMBED_FAILED] += 1
            continue

        embeddings.append(vector)
        kept_labels.append(int(labels[index]))

        if (index + 1) % 500 == 0 or index + 1 == total:
            rate = (index + 1) / (time.perf_counter() - started)
            print(f"  [{index + 1}/{total}] kept {len(embeddings)} ({rate:.1f} img/s)")

    counts["embedded"] = len(embeddings)
    counts["alignment_fallbacks"] = int(getattr(encoder, "fallback_count", 0))
    matrix = l2_normalize(np.stack(embeddings)) if embeddings else np.empty((0, 0), np.float32)
    return matrix, np.asarray(kept_labels, dtype=np.int64), counts, blur_scores


# ══════════════════════════════════════════════════════════════════
# Report + persistence.
# ══════════════════════════════════════════════════════════════════

def print_report(payload):
    counts = payload["pipeline_counts"]
    dataset = payload["dataset"]
    pairs = payload["pairs"]
    dist = payload["distributions"]
    ops = payload["operating_points"]

    print("\n== Pipeline ==========================================================")
    print(f"  images               : {counts['images']}")
    print(f"  multiple YOLO boxes  : {counts['multiple_boxes']}  (most central taken)")
    print(f"  no face detected     : {counts[REJECT_NO_FACE]}")
    print(f"  dropped by gate      : {counts['gate_dropped']}  "
          + ", ".join(f"{r} {counts[r]}" for r in GATE_REASONS))
    print(f"  embed failures       : {counts[REJECT_EMBED_FAILED]}")
    print(f"  embedded             : {counts['embedded']}  "
          f"(alignment fallbacks {counts['alignment_fallbacks']})")
    print(f"  identities           : {dataset['identities_loaded']} loaded, "
          f"{dataset['identities_with_genuine_pairs']} with genuine pairs, "
          f"{dataset['identities_in_impostor_pool']} in impostor pool")
    print(f"  pairs                : {pairs['genuine']} genuine (cap {pairs['genuine_cap_per_identity']}), "
          f"{pairs['impostor']} impostor, seed {pairs['seed']}")

    print("\n== Score distributions (cosine) ======================================")
    for name in ("genuine", "impostor"):
        d = dist[name]
        print(f"  {name:<9} n={d['count']:>7}  mean {d['mean']:+.4f}  sd {d['std']:.4f}  "
              f"p5 {d['p5']:+.4f}  p50 {d['p50']:+.4f}  p95 {d['p95']:+.4f}")
    print(f"  d' = {ops['dprime']:.3f}")

    print("\n== Operating points (same stranger if cosine >= t) ===================")
    print(f"  {'criterion':<22} {'threshold':>10} {'FMR':>10} {'FNMR':>10}")
    rows = [(f"FNMR = {p['target_fnmr']:.0%}", p) for p in ops["fnmr_targets"]]
    rows += [(f"FMR = {p['target_fmr']:.1%}", p) for p in ops["fmr_targets"]]
    rows.append((f"best F1 ({ops['best_f1']['f1']:.4f})", ops["best_f1"]))
    for label, p in rows:
        marker = "  <- PRIMARY" if p.get("target_fnmr") == PRIMARY_FNMR else ""
        print(f"  {label:<22} {p['threshold']:>10.4f} {p['fmr']:>10.4%} {p['fnmr']:>10.4%}{marker}")

    primary = ops["primary"]
    print(f"\n  PRIMARY: t = {primary['threshold']:.4f} splits {primary['fnmr']:.2%} of same-stranger "
          f"pairs and merges {primary['fmr']:.4%} of different-stranger pairs.")
    print("  Not written to any config. Recognition threshold (gallery.npz) untouched.")
    print(f"\n  CAVEAT: {payload['caveat']}")


def save_results(payload, path=RESULTS_FILE):
    """Append this run to results/grouping_threshold.json (append-only, like
    the other benchmark artifacts)."""
    path.parent.mkdir(exist_ok=True)
    data = []
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    data.append(payload)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive the stranger-GROUPING cosine threshold from LFW pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lfw-dir", type=Path, default=DEFAULT_LFW_DIR,
                        help="LFW cache dir, shared with benchmark_recognition.py")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="YOLOv8-face weights")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed for pair sampling (recorded in the artifact)")
    parser.add_argument("--slice", choices=sorted(LFW_SLICES), default=DEFAULT_SLICE,
                        help="region of each 250x250 LFW image the detector sees. 'full' "
                        "matches deployment; 'sklearn-default' is the pre-cropped 125x94 "
                        "baseline.")
    parser.add_argument("--min-blur", type=float, default=DEFAULT_MIN_BLUR,
                        help="variance-of-Laplacian floor for the quality gate. OFF by "
                        "default, matching collect_faces.py and the enrollment run.")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()

    slice_ = LFW_SLICES[args.slice]
    images, labels, names = load_lfw_multi(args.lfw_dir.resolve(), slice_)

    detector = face_utils.load_detector(
        pipeline_config.DETECTION_DETECTOR,
        min_confidence=pipeline_config.DETECTION_CONFIDENCE_THRESHOLD,
        weights=str(args.weights),
    )
    encoder = face_utils.load_encoder(ENCODER)

    print(f"  Embedding {len(images)} images (YOLO @ {pipeline_config.DETECTION_CONFIDENCE_THRESHOLD}"
          f" -> gate -> crop {CROP_MARGIN} -> {ENCODER})...")
    embeddings, kept_labels, counts, blur_scores = embed_images(
        images, labels, detector, encoder, args.min_blur)
    if embeddings.shape[0] == 0:
        sys.exit("ERROR: no image survived the pipeline.")

    rng = np.random.default_rng(args.seed)
    genuine_pairs = sample_genuine_pairs(kept_labels, GENUINE_CAP_PER_IDENTITY, rng)
    impostor_pairs = sample_impostor_pairs(kept_labels, NUM_IMPOSTOR_PAIRS, rng)
    genuine = pair_scores(embeddings, genuine_pairs)
    impostor = pair_scores(embeddings, impostor_pairs)

    per_identity = Counter(kept_labels.tolist())
    counts["gate_dropped"] = sum(counts[r] for r in GATE_REASONS)
    pipeline_counts = {
        key: int(counts[key])
        for key in ("images", "multiple_boxes", REJECT_NO_FACE, *GATE_REASONS,
                    "gate_dropped", REJECT_EMBED_FAILED, "embedded", "alignment_fallbacks")
    }
    blur = summarize_scores(blur_scores)
    pipeline_counts["blur_distribution"] = None if blur is None else {
        k: (round(v, 2) if isinstance(v, float) else v) for k, v in blur.items()
    }

    settings = {
        "encoder": ENCODER,
        "detector": pipeline_config.DETECTION_DETECTOR,
        "detector_weights": pipeline_config.DETECTION_WEIGHTS,
        "detector_confidence": pipeline_config.DETECTION_CONFIDENCE_THRESHOLD,
        "box_selection": "most central (face_utils._most_central_index); image dropped "
                         "if that box fails the gate",
        "crop_margin": CROP_MARGIN,
        "gate": {
            "source": "collect_faces.quality_check",
            "min_box_size": MIN_BOX_SIZE,
            "min_confidence": GATE_MIN_CONFIDENCE,
            "min_blur": args.min_blur,
        },
        "embedding_normalization": "L2",
        "similarity": "cosine",
        "decision_rule": "same stranger if score >= threshold",
        "primary_criterion": f"FNMR = {PRIMARY_FNMR:g}",
        "fnmr_targets": FNMR_TARGETS,
        "fmr_targets": FMR_TARGETS,
    }
    dataset = {
        "source": "LFW funneled, enumerated as sklearn's fetch_lfw_people does and "
                  "decoded per image by sklearn.datasets._lfw._load_imgs (see LazyLFW)",
        "kwargs": {"min_faces_per_person": MIN_FACES_PER_PERSON, **LFW_KWARGS},
        "slice_name": args.slice,
        "slice_": slice_to_json(slice_),
        "slice_note": "[[row_start, row_stop], [col_start, col_stop]] of the 250x250 "
                      "funneled image handed to YOLO",
        "image_shape": list(images.shape()),
        "images_loaded": len(images),
        "identities_loaded": int(len(names)),
        "identities_in_impostor_pool": int(len(per_identity)),
        "identities_with_genuine_pairs": int(sum(1 for n in per_identity.values() if n >= 2)),
    }
    pairs = {
        "seed": args.seed,
        "genuine_cap_per_identity": GENUINE_CAP_PER_IDENTITY,
        "genuine_sampling": "within identity, without replacement, min(cap, C(n,2)) per identity",
        "impostor_sampling": "two distinct identities uniformly, then one image from each",
        "genuine": int(genuine.size),
        "impostor": int(impostor.size),
        "impostor_requested": NUM_IMPOSTOR_PAIRS,
    }

    payload = build_payload(genuine=genuine, impostor=impostor, settings=settings,
                            dataset=dataset, pipeline_counts=pipeline_counts, pairs=pairs)
    payload["wall_time_s"] = round(time.perf_counter() - started, 1)

    print_report(payload)
    save_results(payload)
    print(f"\n  Saved -> {RESULTS_FILE.relative_to(REPO_ROOT)}  ({payload['wall_time_s']}s)\n")


if __name__ == "__main__":
    main()
