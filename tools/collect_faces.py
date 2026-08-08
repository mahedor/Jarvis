"""
JARVIS Face Recognition — Stage 2, Step 1: Enrollment crop collection
=====================================================================
Turns a flat bucket of household photos into CURATED, UNNAMED reference
face crops, grouped by identity via clustering. You then name them by hand.

    photos/  ->  detect  ->  quality filter  ->  embed  ->  HDBSCAN
             ->  data/reference_faces/cluster_000/, cluster_001/, _noise/

THIS TOOL NEVER NAMES ANYONE. It has no idea who anybody is, and it never
writes a person's name to disk. It only says "these crops look like the same
person". YOU do the naming, afterwards, by renaming the cluster folders:

    data/reference_faces/cluster_000/  ->  data/reference_faces/michael/
    data/reference_faces/cluster_003/  ->  (delete: a stranger / a guest)

That rename IS the naming step. Each cluster folder holds its crops plus a
_face_card.jpg (the cluster medoid — its most typical face), so you can
identify a cluster at a glance without opening every crop.

Input folder structure is IRRELEVANT to identity. --input-dir is walked
recursively and treated as ONE FLAT BUCKET; subfolder names are carried
along as METADATA ONLY and shown in the report as a naming hint ("this
cluster's crops came mostly from photos/christmas/"). A subfolder is never
treated as a label — Google Photos folders are events, not people, and a
photo of four people contains four identities no matter what folder it is in.

Usage:
    python tools/collect_faces.py
    python tools/collect_faces.py --input-dir data/enrollment_photos --min-cluster-size 4
    python tools/collect_faces.py --min-blur 60 --overwrite

Sibling tool (NOT this one): capture_faces_webcam.py will handle live
webcam capture. This tool only ever reads photos from disk.
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import HDBSCAN

import face_utils  # same directory

# Paths are resolved relative to the REPO ROOT (parent of tools/), so the tool
# behaves identically no matter which directory you run it from.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

DEFAULT_INPUT_DIR = _REPO_ROOT / "data" / "enrollment_photos"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "reference_faces"
DEFAULT_YOLO_WEIGHTS = _HERE / "weights" / "yolov8n-face.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Output folders this tool OWNS and may delete on --overwrite. Anything else in
# the output dir (i.e. a cluster folder you renamed to a person) is left alone.
NOISE_DIR_NAME = "_noise"
# Quality-rejected crops are copied here, under a per-reason subfolder, purely so
# you can eyeball what the filter discarded and confirm no good faces are lost
# (see save_rejected_crop). Tool-owned, so --overwrite clears it. Only too_blurry
# is saved: low_confidence and too_small are rejected BEFORE a crop is ever made
# (see collect_crops), so there is no image on hand to write.
REJECTED_DIR_NAME = "_rejected"
CLUSTER_DIR_PREFIX = "cluster_"
FACE_CARD_NAME = "_face_card.jpg"
MANIFEST_NAME = "manifest.json"

# Quality-filter reject reasons. Every rejected crop is attributed to exactly
# one of these, so the report can count them.
REJECT_LOW_CONFIDENCE = "low_confidence"
REJECT_TOO_SMALL = "too_small"
REJECT_TOO_BLURRY = "too_blurry"


# ══════════════════════════════════════════════════════════════════
# PURE LOGIC — no cv2, no disk, no models, no globals.
# Everything below this line is a function of its arguments alone, which is
# what makes it unit-testable (see tests/test_collect_faces.py).
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class QualityResult:
    """Outcome of the quality filter for a single crop.

    keep (bool): True if the crop should be embedded and clustered.
    reason (str | None): None when keep is True; otherwise exactly one of
        REJECT_LOW_CONFIDENCE / REJECT_TOO_SMALL / REJECT_TOO_BLURRY, so
        rejects are countable by bucket.
    """

    keep: bool
    reason: str | None = None


def quality_check(box, confidence, blur_score, min_box_size, min_confidence, min_blur=None):
    """Decide whether one detected face is good enough to enroll.

    Checks run in ascending order of cost-to-trust: a low-confidence box may
    not be a face at all, so its size and sharpness are meaningless; a tiny
    box's blur score is noise. The FIRST failing check is the reported reason.

    Inputs:
        box (sequence of 4 numbers): [x1, y1, x2, y2] absolute pixels — the
            DETECTION box, not the padded crop.
        confidence (float): detector confidence, 0-1.
        blur_score (float | None): variance of the Laplacian of the crop.
            Higher = sharper. None means "not measured".
        min_box_size (float): minimum acceptable SHORTEST side of the box, in
            pixels. Faces below this carry too few pixels to embed reliably.
        min_confidence (float): minimum acceptable detector confidence.
        min_blur (float | None): minimum acceptable blur score. None (the
            default) disables the blur check entirely — run once with it off,
            read the printed blur distribution, then pick a cutoff.
    Returns:
        QualityResult: .keep plus, on rejection, the single .reason.
    """
    if confidence < min_confidence:
        return QualityResult(False, REJECT_LOW_CONFIDENCE)

    x1, y1, x2, y2 = box
    shortest_side = min(x2 - x1, y2 - y1)
    if shortest_side < min_box_size:
        return QualityResult(False, REJECT_TOO_SMALL)

    if min_blur is not None and blur_score is not None and blur_score < min_blur:
        return QualityResult(False, REJECT_TOO_BLURRY)

    return QualityResult(True, None)


def l2_normalize(embeddings):
    """Scale each embedding to unit length.

    Unit length is what makes euclidean distance monotonic with cosine
    distance (||a-b||^2 == 2 - 2*cos(a,b) when ||a||==||b||==1), which is why
    the clustering step can hand plain euclidean to HDBSCAN and still be
    clustering by the cosine similarity that face embeddings are trained for.

    Inputs:
        embeddings (np.ndarray): shape (N, D). Also accepts a single (D,)
            vector, returned as (D,).
    Returns:
        np.ndarray: float32, same shape, each row of unit length. Zero-length
        rows (a degenerate embedding) are passed through as zeros rather than
        producing NaN — they simply cannot be normalized.
    """
    array = np.asarray(embeddings, dtype=np.float32)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, -1)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # Guard against division by zero: a zero vector normalizes to itself.
    safe = np.where(norms == 0, 1.0, norms)
    normalized = (array / safe).astype(np.float32)

    return normalized[0] if single else normalized


def medoid_index(embeddings):
    """Index of the most typical member of a group — its medoid.

    The medoid is the member with the smallest TOTAL distance to every other
    member: the one closest to everybody else. Unlike a mean it is an ACTUAL
    member of the set, so it corresponds to a real crop we can write to disk
    as the cluster's face card. On an off-pose or half-occluded cluster, the
    medoid is the shot that best represents the whole group.

    Inputs:
        embeddings (np.ndarray): shape (N, D), expected L2-normalized.
    Returns:
        int: index into `embeddings` of the medoid. For N == 1, returns 0.
    Raises:
        ValueError: if there are no embeddings.
    """
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("medoid_index() needs a non-empty (N, D) array")
    if array.shape[0] == 1:
        return 0

    # Full pairwise euclidean distance matrix. Clusters here are tens-to-
    # hundreds of crops, so the O(N^2) matrix is cheap and clarity wins.
    diffs = array[:, None, :] - array[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return int(np.argmin(distances.sum(axis=1)))


def summarize_distribution(values):
    """Percentile summary of a list of numbers, for the blur report.

    Inputs:
        values (sequence of float): e.g. every crop's blur score.
    Returns:
        dict | None: {"count", "min", "p5", "p25", "p50", "p75", "p95", "max"},
        or None if `values` is empty.
    """
    if len(values) == 0:
        return None
    array = np.asarray(values, dtype=np.float64)
    p5, p25, p50, p75, p95 = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(array.max()),
    }


# ══════════════════════════════════════════════════════════════════
# I/O + MODELS — cv2, disk, and the detector/encoder live below here.
# ══════════════════════════════════════════════════════════════════

@dataclass
class Crop:
    """One detected face, carried through the pipeline.

    `image` is the raw generous-margin BGR crop — the SAME array handed to
    whichever encoder is loaded, which does its own alignment/resizing.
    """

    image: np.ndarray
    source_path: Path      # absolute path of the photo it came from
    source_folder: str     # subfolder under --input-dir, "." at the top level.
                           # METADATA ONLY — a naming hint, never an identity.
    face_index: int        # which face within that photo (0-based)
    box: list              # [x1, y1, x2, y2] in the SOURCE photo's pixels
    confidence: float
    blur_score: float
    filename: str          # unique output filename, encodes its provenance


def iter_photos(input_dir):
    """Every image file under input_dir, recursively, sorted for determinism.

    Inputs:
        input_dir (Path): the photo bucket. Subfolders are walked but carry
            no meaning beyond metadata.
    Returns:
        list[Path]: absolute paths, sorted.
    """
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def variance_of_laplacian(bgr_image):
    """Sharpness score: the variance of the image's Laplacian.

    The Laplacian is a second-derivative (edge) operator. A sharp face has
    strong, varied edges -> high variance. A blurred face has soft edges ->
    low variance. It is scale- and exposure-sensitive, which is exactly why
    this tool PRINTS the distribution instead of hard-coding a threshold:
    the right cutoff depends on your photos.

    Inputs:
        bgr_image (np.ndarray): BGR crop.
    Returns:
        float: the variance. Higher is sharper. 0.0 for an empty crop.
    """
    if bgr_image is None or bgr_image.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_crop_filename(photo_path, input_dir, face_index):
    """A unique, provenance-carrying output filename for one crop.

    The photo's path RELATIVE to the input dir is flattened into the name, so
    (a) two photos with the same basename in different subfolders can never
    collide, and (b) while renaming cluster folders you can see at a glance
    which photo (and which event folder) each crop came from.

        photos/christmas/IMG_001.jpg, face 2  ->  christmas_IMG_001__f2.jpg

    Inputs:
        photo_path (Path): the source photo.
        input_dir (Path): the root the walk started from.
        face_index (int): 0-based index of the face within that photo.
    Returns:
        str: a filesystem-safe .jpg filename.
    """
    try:
        relative = photo_path.relative_to(input_dir)
    except ValueError:
        relative = Path(photo_path.name)

    stem = relative.with_suffix("").as_posix().replace("/", "_")
    safe = "".join(char if (char.isalnum() or char in "-_") else "_" for char in stem)
    return f"{safe}__f{face_index}.jpg"


def save_rejected_crop(reject_dir, reason, filename, crop_image):
    """Copy one quality-rejected crop under _rejected/<reason>/ for inspection.

    A debugging aid, nothing more: it lets you eyeball what the quality filter
    threw away and confirm no good faces are being lost. It is deliberately
    decoupled from the report — reject COUNTS are tallied separately in
    collect_crops, and saving (or not saving) a crop here never changes them.

    Only too_blurry crops ever reach this function. low_confidence and too_small
    are rejected BEFORE a crop exists (cropping every sub-threshold box would
    dominate runtime — see collect_crops), so they have no image to save.

    Inputs:
        reject_dir (Path): the _rejected/ directory (output_dir / _rejected).
        reason (str): the reject bucket (e.g. REJECT_TOO_BLURRY); the subfolder.
        filename (str): the crop's provenance-carrying output filename.
        crop_image (np.ndarray): the BGR crop to write.
    """
    folder = reject_dir / reason
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / filename), crop_image)


def collect_crops(photos, detector, input_dir, output_dir, args, stats):
    """Detect faces in every photo, crop them, and apply the quality filter.

    The detector is loaded with min_confidence=0.0 and the confidence floor is
    applied HERE via quality_check, so low-confidence faces are COUNTED as
    rejects instead of vanishing silently inside the backend.

    Inputs:
        photos (list[Path]): photos to process.
        detector: a face_utils detector (.detect(bgr) -> list of dicts).
        input_dir (Path): root of the walk (for provenance filenames).
        output_dir (Path): where clusters get written; too_blurry rejects are
            copied under its _rejected/<reason>/ so you can inspect them.
        args (argparse.Namespace): the CLI knobs.
        stats (dict): mutated in place with counters for the report.
    Returns:
        list[Crop]: only the crops that PASSED the quality filter.
    """
    kept = []

    for index, photo_path in enumerate(photos, start=1):
        image = cv2.imread(str(photo_path))
        if image is None:
            stats["unreadable_photos"] += 1
            print(f"  [{index}/{len(photos)}] {photo_path.name}: UNREADABLE (corrupt/unsupported), skipping")
            continue

        stats["photos_read"] += 1
        detections = detector.detect(image)
        stats["faces_detected"] += len(detections)

        # Folder is metadata only — a hint for naming clusters later.
        relative_parent = photo_path.parent.relative_to(input_dir).as_posix() or "."

        kept_here = 0
        for face_index, detection in enumerate(detections):
            box = detection["box"]
            confidence = float(detection["confidence"])

            # Cheap checks FIRST — confidence and size need no pixels. This
            # matters: at min_confidence=0.0 YOLO returns up to 300 boxes per
            # photo (its max_det cap), and ~95% are junk. Cropping and
            # Laplacian-ing every one of them before rejecting it would
            # dominate the runtime of a real bucket for zero gain. Passing
            # blur_score=None means "not measured", so this call can only
            # reject on confidence or size.
            screened = quality_check(
                box=box,
                confidence=confidence,
                blur_score=None,
                min_box_size=args.min_box_size,
                min_confidence=args.min_confidence,
                min_blur=args.min_blur,
            )
            if not screened.keep:
                stats["rejects"][screened.reason] += 1
                continue

            crop_image = face_utils.crop_face(image, box, margin=args.margin)
            if crop_image.size == 0:
                stats["rejects"][REJECT_TOO_SMALL] += 1
                continue

            # Only now, on a crop we already believe is a real, big-enough
            # face, is it worth measuring sharpness.
            blur_score = variance_of_laplacian(crop_image)
            result = quality_check(
                box=box,
                confidence=confidence,
                blur_score=blur_score,
                min_box_size=args.min_box_size,
                min_confidence=args.min_confidence,
                min_blur=args.min_blur,
            )

            # Blur scores are only meaningful for crops that are actually faces
            # and actually big enough — which is exactly the set that reaches
            # here. So the distribution printed for picking --min-blur is
            # gathered over precisely the crops that cutoff would apply to.
            stats["blur_scores"].append(blur_score)

            if not result.keep:
                stats["rejects"][result.reason] += 1
                # Copy the discarded crop out for a human to eyeball. Only
                # too_blurry can land here (confidence and size already passed,
                # and the crop is non-empty), and its crop already exists — so
                # this is essentially free: no extra detection, no extra crop.
                save_rejected_crop(
                    output_dir / REJECTED_DIR_NAME,
                    result.reason,
                    make_crop_filename(photo_path, input_dir, face_index),
                    crop_image,
                )
                continue

            kept.append(
                Crop(
                    image=crop_image,
                    source_path=photo_path,
                    source_folder=relative_parent,
                    face_index=face_index,
                    box=[int(v) for v in box],
                    confidence=confidence,
                    blur_score=blur_score,
                    filename=make_crop_filename(photo_path, input_dir, face_index),
                )
            )
            kept_here += 1

        print(
            f"  [{index}/{len(photos)}] {photo_path.name}: "
            f"{len(detections)} detected, {kept_here} kept"
        )

    return kept


def embed_crops(crops, encoder, stats):
    """Embed every crop, dropping any the encoder chokes on.

    Each crop is handed to the encoder EXACTLY as cropped — raw, generous
    margin, no alignment, no resize. Geometry is the encoder's business
    (see the encoder contract in face_utils).

    Inputs:
        crops (list[Crop]): quality-filtered crops.
        encoder: a face_utils encoder (.embed(bgr) -> np.ndarray).
        stats (dict): mutated in place ("embed_failures").
    Returns:
        tuple[list[Crop], np.ndarray]: the crops that embedded successfully,
        and their L2-NORMALIZED embeddings as an (N, D) array (empty (0, 0)
        if none survived).
    """
    embedded_crops = []
    vectors = []

    for index, crop in enumerate(crops, start=1):
        try:
            vectors.append(encoder.embed(crop.image))
            embedded_crops.append(crop)
        except Exception as exc:  # a crop the model simply cannot handle
            stats["embed_failures"] += 1
            print(f"  WARNING: embed failed for {crop.filename} ({type(exc).__name__}: {exc})")

        if index % 50 == 0:
            print(f"  ...embedded {index}/{len(crops)}")

    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32)

    return embedded_crops, l2_normalize(np.vstack(vectors))


def cluster_embeddings(embeddings, min_cluster_size, min_samples=None):
    """Group embeddings into identities with HDBSCAN.

    HDBSCAN (rather than k-means) because we do NOT know how many people are
    in the bucket — that is the whole question — and because it has a native
    notion of NOISE: the one-off stranger in the background of a single photo
    is labelled -1 rather than being forced into somebody's cluster and
    poisoning their reference set.

    Euclidean distance on L2-normalized vectors is monotonic with cosine
    distance, so this clusters by the similarity ArcFace was trained on.

    Inputs:
        embeddings (np.ndarray): (N, D), L2-normalized.
        min_cluster_size (int): fewest crops that can form an identity.
        min_samples (int | None): how conservative the algorithm is about
            calling a point noise. None = HDBSCAN's default (min_cluster_size).
    Returns:
        np.ndarray: (N,) int labels; -1 means noise.
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        # copy=True is load-bearing, not cosmetic: at the default (False)
        # HDBSCAN may modify the input array IN PLACE, and the caller reuses
        # these same embeddings afterwards to pick each cluster's medoid.
        copy=True,
    )
    return clusterer.fit_predict(embeddings)


def prepare_output_dir(output_dir, overwrite):
    """Make the output dir safe to write into, WITHOUT destroying your work.

    The manual rename of cluster_XXX/ -> a person's name is the whole point of
    this tool, and it is not reproducible by re-running. So a non-empty output
    dir is refused unless --overwrite, and even then only the folders this tool
    OWNS (cluster_*/, _noise/, manifest.json) are deleted. A folder you have
    renamed to a person is never touched.

    Inputs:
        output_dir (Path): where clusters get written.
        overwrite (bool): whether to clear previous tool-owned output.
    Raises:
        SystemExit: if the dir is non-empty and overwrite is False.
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return

    entries = list(output_dir.iterdir())
    if not entries:
        return

    def is_owned(path):
        return path.name in (NOISE_DIR_NAME, REJECTED_DIR_NAME, MANIFEST_NAME) or path.name.startswith(CLUSTER_DIR_PREFIX)

    if not overwrite:
        # Plain ASCII in this message on purpose: it goes to stderr, which on
        # Windows is cp1252, and an em-dash comes out as a mojibake box.
        sys.exit(
            f"\nERROR: output dir is not empty: {output_dir}\n"
            f"  It holds {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}.\n"
            "  Re-running would clobber previous output, and if you have already renamed\n"
            "  cluster folders to people, that naming is not reproducible.\n"
            "  Pass --overwrite to delete the tool-owned folders (cluster_*/, _noise/,\n"
            "  _rejected/, manifest.json) and regenerate. Renamed folders are always left alone."
        )

    preserved = []
    for entry in entries:
        if is_owned(entry):
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        else:
            preserved.append(entry.name)

    if preserved:
        print(
            "  --overwrite: cleared tool-owned output; left your renamed folders alone: "
            f"{', '.join(sorted(preserved))}"
        )


def write_clusters(crops, labels, embeddings, output_dir):
    """Write every crop to disk under its cluster, with a medoid face card.

    Cluster folders are named cluster_000, cluster_001, ... (zero-padded so
    they sort correctly), and each holds a _face_card.jpg — a COPY of the
    cluster's medoid crop — so you can identify who a cluster is without
    opening its crops one by one. Noise (label -1) goes to _noise/ for you to
    inspect: it is where a stranger, a bad crop, or a person with too few
    photos ends up.

    NO NAME IS WRITTEN ANYWHERE. Naming is the manual rename you do next.

    Inputs:
        crops (list[Crop]): the embedded crops, index-aligned with `labels`.
        labels (np.ndarray): (N,) cluster labels from HDBSCAN, -1 = noise.
        embeddings (np.ndarray): (N, D) L2-normalized, index-aligned too.
        output_dir (Path): data/reference_faces/.
    Returns:
        dict: {cluster_id: [Crop, ...]} including -1 for noise, so the report
        and manifest can describe what was written.
    """
    grouped = defaultdict(list)
    for position, label in enumerate(labels):
        grouped[int(label)].append(position)

    cluster_ids = sorted(label for label in grouped if label >= 0)
    # Zero-pad wide enough that folders sort lexicographically = numerically.
    pad_width = max(3, len(str(max(cluster_ids))) if cluster_ids else 3)

    written = {}
    for label, positions in grouped.items():
        if label < 0:
            folder = output_dir / NOISE_DIR_NAME
        else:
            folder = output_dir / f"{CLUSTER_DIR_PREFIX}{label:0{pad_width}d}"
        folder.mkdir(parents=True, exist_ok=True)

        for position in positions:
            cv2.imwrite(str(folder / crops[position].filename), crops[position].image)

        if label >= 0:
            # The medoid is the cluster's most typical face — the best single
            # thumbnail for "who is this?".
            local_medoid = medoid_index(embeddings[positions])
            medoid_crop = crops[positions[local_medoid]]
            cv2.imwrite(str(folder / FACE_CARD_NAME), medoid_crop.image)

        written[label] = [crops[position] for position in positions]

    return written


def write_manifest(path, written, args, stats, blur_summary):
    """Dump machine-readable provenance for every crop written.

    The next step of enrollment (embedding the NAMED folders into the live
    recognition gallery) wants to know where each reference crop came from,
    and a manifest beats re-deriving it from filenames.

    Inputs:
        path (Path): manifest.json destination.
        written (dict): {cluster_id: [Crop, ...]} from write_clusters.
        args (argparse.Namespace): the run's settings, recorded for repeatability.
        stats (dict): the run's counters.
        blur_summary (dict | None): the blur distribution.
    """
    records = []
    for label, crops in sorted(written.items()):
        for crop in crops:
            records.append(
                {
                    "cluster": int(label),  # -1 = noise
                    "filename": crop.filename,
                    "source_photo": str(crop.source_path),
                    "source_folder": crop.source_folder,  # metadata hint, NOT a label
                    "face_index": crop.face_index,
                    "box": crop.box,
                    "confidence": round(crop.confidence, 4),
                    "blur_score": round(crop.blur_score, 2),
                }
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Cluster ids are NOT identities. Rename cluster folders to name people.",
        "settings": {
            "input_dir": str(args.input_dir),
            "detector": args.detector,
            "encoder": args.encoder,
            "margin": args.margin,
            "min_box_size": args.min_box_size,
            "min_confidence": args.min_confidence,
            "min_blur": args.min_blur,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
        },
        "counts": {
            "photos_read": stats["photos_read"],
            "unreadable_photos": stats["unreadable_photos"],
            "faces_detected": stats["faces_detected"],
            "crops_kept": stats["crops_kept"],
            "rejects": dict(stats["rejects"]),
            "embed_failures": stats["embed_failures"],
            "alignment_fallbacks": stats["alignment_fallbacks"],
        },
        "blur_distribution": blur_summary,
        "crops": records,
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_report(written, stats, blur_summary, args, output_dir):
    """Print the end-of-run report.

    Inputs:
        written (dict): {cluster_id: [Crop, ...]}.
        stats (dict): the run's counters.
        blur_summary (dict | None): percentiles of the blur scores.
        args (argparse.Namespace): the run's settings.
        output_dir (Path): where everything was written.
    """
    print("\n" + "=" * 62)
    print("REPORT")
    print("=" * 62)

    print("\n--- Input ---")
    print(f"  photos found       {stats['photos_found']}")
    print(f"  photos read        {stats['photos_read']}")
    if stats["unreadable_photos"]:
        print(f"  unreadable         {stats['unreadable_photos']}")
    print(f"  faces detected     {stats['faces_detected']}")

    print("\n--- Quality filter ---")
    total_rejected = sum(stats["rejects"].values())
    print(f"  crops kept         {stats['crops_kept']}")
    print(f"  crops rejected     {total_rejected}")
    for reason in (REJECT_LOW_CONFIDENCE, REJECT_TOO_SMALL, REJECT_TOO_BLURRY):
        count = stats["rejects"].get(reason, 0)
        threshold = {
            REJECT_LOW_CONFIDENCE: f"< {args.min_confidence}",
            REJECT_TOO_SMALL: f"shortest side < {args.min_box_size}px",
            REJECT_TOO_BLURRY: "OFF" if args.min_blur is None else f"< {args.min_blur}",
        }[reason]
        print(f"    {reason:16s} {count:6d}   ({threshold})")

    print("\n--- Blur scores (variance of Laplacian; higher = sharper) ---")
    if blur_summary is None:
        print("  no crops measured")
    else:
        print(
            f"  min {blur_summary['min']:8.1f} | p5 {blur_summary['p5']:8.1f} | "
            f"p25 {blur_summary['p25']:8.1f} | p50 {blur_summary['p50']:8.1f}"
        )
        print(
            f"  p75 {blur_summary['p75']:8.1f} | p95 {blur_summary['p95']:8.1f} | "
            f"max {blur_summary['max']:8.1f}"
        )
        if args.min_blur is None:
            print(
                f"  Blur filter is OFF. To drop the softest ~5% of crops next run: "
                f"--min-blur {blur_summary['p5']:.0f}"
            )

    print("\n--- Embedding ---")
    print(f"  encoder            {args.encoder}")
    print(f"  embedded           {stats['crops_embedded']}")
    if stats["embed_failures"]:
        print(f"  embed failures     {stats['embed_failures']}")
    fallbacks = stats["alignment_fallbacks"]
    if fallbacks is None:
        print("  align fallbacks    n/a (this encoder does not report them)")
    else:
        print(f"  align fallbacks    {fallbacks}   (no face re-detected in the crop -> unaligned resize)")

    print("\n--- Clusters ---")
    cluster_ids = sorted(label for label in written if label >= 0)
    noise_crops = written.get(-1, [])
    print(f"  clusters found     {len(cluster_ids)}")
    print(f"  noise crops        {len(noise_crops)}   (label -1, min_cluster_size={args.min_cluster_size})")

    if cluster_ids:
        print("\n  Cluster sizes, with the source folders their crops came from.")
        print("  (Folders are a NAMING HINT ONLY — they were never used to group anyone.)\n")
        pad_width = max(3, len(str(max(cluster_ids))))
        for label in cluster_ids:
            crops = written[label]
            folders = Counter(crop.source_folder for crop in crops)
            breakdown = ", ".join(
                f"{folder} ({count})" for folder, count in folders.most_common()
            )
            print(f"    {CLUSTER_DIR_PREFIX}{label:0{pad_width}d}  {len(crops):4d} crops   from: {breakdown}")

    print("\n--- Next step (this is the naming step) ---")
    print(f"  Open {output_dir}")
    print(f"  Each cluster_XXX/ has a {FACE_CARD_NAME} — the cluster's most typical face.")
    print("    1. Rename the folder to the person, e.g. cluster_000/ -> michael/")
    print("    2. Delete clusters that are strangers, guests, or junk.")
    print(f"    3. Check {NOISE_DIR_NAME}/ for anyone the clustering missed.")
    print("  This tool deliberately wrote no names. Cluster ids mean nothing.\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster a bucket of photos into unnamed reference face crops for enrollment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="photo bucket. Walked RECURSIVELY, but treated as one flat pot: "
        "subfolder names are metadata (a naming hint in the report), never labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="where cluster_XXX/ and _noise/ are written",
    )
    parser.add_argument(
        "--detector",
        default="yolo",
        help="face detector backend (yolo, retinaface, mtcnn, mediapipe)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_YOLO_WEIGHTS,
        help="YOLO face weights (only used when --detector is yolo)",
    )
    parser.add_argument(
        "--encoder",
        default="arcface",
        help="embedding backend (arcface, facenet512). Each encoder does its own "
        "alignment/resizing internally — the pipeline hands them all the same raw crop.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.35,
        help="crop padding as a fraction of the box, per side. Generous on purpose: "
        "it gives the encoder room to re-detect and align the face itself.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=40,
        help="reject a face whose SHORTEST box side is under this many pixels",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="reject a face detected below this confidence",
    )
    parser.add_argument(
        "--min-blur",
        type=float,
        default=None,
        help="reject a crop whose variance-of-Laplacian is below this. OFF by default: "
        "run once with it off, read the printed distribution, then pick a cutoff.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=6,
        help="fewest crops that can form an identity (HDBSCAN). Below this, crops become noise.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN conservativeness about calling a point noise. Default: min_cluster_size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process at most this many photos (0 = all). Handy for a quick first pass.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="clear previous tool-owned output (cluster_*/, _noise/, _rejected/, manifest.json). "
        "Folders you renamed to people are never deleted.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        sys.exit(
            f"ERROR: input dir does not exist: {input_dir}\n"
            "  Drop your household photos there (subfolders are fine: they are walked\n"
            "  recursively and their names are treated as metadata, never as labels)."
        )

    photos = iter_photos(input_dir)
    if args.limit > 0:
        photos = photos[: args.limit]
    if not photos:
        sys.exit(f"ERROR: no images ({', '.join(sorted(IMAGE_EXTENSIONS))}) found under {input_dir}")

    prepare_output_dir(output_dir, args.overwrite)

    stats = {
        "photos_found": len(photos),
        "photos_read": 0,
        "unreadable_photos": 0,
        "faces_detected": 0,
        "crops_kept": 0,
        "crops_embedded": 0,
        "rejects": Counter(),
        "blur_scores": [],
        "embed_failures": 0,
        "alignment_fallbacks": None,
    }

    print(f"\nInput : {input_dir}  ({len(photos)} photos)")
    print(f"Output: {output_dir}")

    # ── 1. Detect + crop + quality filter ────────────────────────
    print(f"\n=== Detecting faces ({args.detector}) ===")
    # min_confidence=0.0: keep EVERY detection so the confidence floor is
    # applied by quality_check and low-confidence rejects get counted.
    detector_kwargs = {"min_confidence": 0.0}
    if args.detector.lower().startswith("yolo"):
        detector_kwargs["weights"] = str(args.weights)
    detector = face_utils.load_detector(args.detector, **detector_kwargs)

    crops = collect_crops(photos, detector, input_dir, output_dir, args, stats)
    stats["crops_kept"] = len(crops)
    if not crops:
        sys.exit(
            "\nERROR: no crops survived the quality filter. Loosen --min-box-size / "
            "--min-confidence, or check that the photos actually contain faces."
        )

    # ── 2. Embed ─────────────────────────────────────────────────
    print(f"\n=== Embedding {len(crops)} crops ({args.encoder}) ===")
    encoder = face_utils.load_encoder(args.encoder)
    crops, embeddings = embed_crops(crops, encoder, stats)
    stats["crops_embedded"] = len(crops)
    # Not every encoder can tell whether it fell back to an unaligned resize.
    stats["alignment_fallbacks"] = getattr(encoder, "fallback_count", None)
    if not crops:
        sys.exit("\nERROR: every crop failed to embed. Check the encoder backend.")

    # ── 3. Cluster ───────────────────────────────────────────────
    print(f"\n=== Clustering (HDBSCAN, min_cluster_size={args.min_cluster_size}) ===")
    labels = cluster_embeddings(embeddings, args.min_cluster_size, args.min_samples)

    # ── 4. Write ─────────────────────────────────────────────────
    print("=== Writing crops ===")
    written = write_clusters(crops, labels, embeddings, output_dir)
    blur_summary = summarize_distribution(stats["blur_scores"])
    write_manifest(output_dir / MANIFEST_NAME, written, args, stats, blur_summary)

    print_report(written, stats, blur_summary, args, output_dir)


if __name__ == "__main__":
    main()
