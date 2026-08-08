"""
JARVIS Face Recognition — Stage 2, Step 3: Gallery builder
==========================================================
Turns the curated reference crops into the ONE FILE the live presence service
loads at startup: every enrolled person's reference vectors, plus the metadata
that says exactly how they were made.

    data/reference_faces/<person>/*.jpg  ->  embed  ->  aggregate per person
                                         ->  data/gallery.npz

    python tools/build_gallery.py
    python tools/build_gallery.py --encoder facenet512 --aggregation medoid --threshold 0.641

WHY THE METADATA IS THE POINT. A face embedding is only comparable to another
embedding from the SAME encoder — an ArcFace vector scored against a FaceNet512
gallery produces a perfectly well-formed cosine similarity that means nothing at
all. That failure is silent: no exception, no NaN, just quietly wrong identities.
So the gallery file carries the encoder name, the aggregation strategy, the
embedding dimension and the similarity threshold that were used to build it, and
load_gallery() will refuse to hand back vectors that do not match what the caller
says it is about to score with.

THRESHOLD. The threshold stored here is the MAX-F1 threshold of the
(encoder, aggregation) pair you are building with — the cut-off that best
balances precision and recall, and the only one meant for deployment.
benchmark_recognition.py also reports a TAR@FAR threshold, but that one is a
YARDSTICK: it pins every encoder to the same false-accept budget so they can be
ranked fairly. It is a measurement instrument. Do not store it here.

--threshold defaults to 0.342, the max-F1 threshold measured for
arcface + multi-reference (the winning cell). It is calibrated for THAT PAIR
ONLY — cosine thresholds are not transferable between encoders (facenet512
wants ~0.64 for the same job). This script cross-checks whatever you pass
against the max-F1 threshold recorded for that pair in
results/benchmarks_recognition.json and warns, naming both numbers, if they
disagree.

AGGREGATION comes from recognition_metrics.AGGREGATIONS — the same pure
functions the benchmark scored, so the gallery that ships is built by the exact
code path that earned the number. 'mean-renormalize' and 'medoid' store one
vector per person; 'multi-reference' stores all of them. Either way the file
holds a stacked (N, D) matrix labelled by person, so the live scorer can always
use max-cosine over a person's rows — which collapses to a plain cosine when
that person has a single row. One code path live, whichever strategy you pick.

Pure logic (aggregation, stacking, metadata assembly) is separated from I/O
(reading crops, embedding, writing the file) exactly as in the benchmark: the
pure half is a function of its arguments and numpy only.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Path setup — this file lives in tools/ next to the modules it imports ──
TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from face_utils import load_encoder, load_reference_crops, embed_crops
from recognition_metrics import AGGREGATIONS

DEFAULT_REFERENCE_DIR = REPO_ROOT / "data" / "reference_faces"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "gallery.npz"
BENCHMARK_RESULTS_FILE = REPO_ROOT / "results" / "benchmarks_recognition.json"

DEFAULT_ENCODER = "arcface"
DEFAULT_AGGREGATION = "multi-reference"
DEFAULT_THRESHOLD = 0.342

# The (encoder, aggregation) pair DEFAULT_THRESHOLD was measured on. Used only
# to warn when the default threshold is carried over to a different pair and
# there is no benchmark result to cross-check against.
THRESHOLD_CALIBRATED_FOR = (DEFAULT_ENCODER, DEFAULT_AGGREGATION)

# The benchmark rounds thresholds to 4 decimals in its JSON, so anything closer
# than this counts as "the same number".
THRESHOLD_TOLERANCE = 1e-3

# Bumped if the on-disk layout changes, so an old gallery cannot be silently
# misread by newer code.
GALLERY_FORMAT_VERSION = 1


# ════════════════════════════════════════════════════════════════════
# Pure logic — numpy only, no disk, no models. Unit-testable as-is.
# ════════════════════════════════════════════════════════════════════

def build_person_vectors(person_embeddings, build_fn):
    """Aggregate each person's embeddings into their gallery vectors.

    Inputs:
        person_embeddings (dict[str, np.ndarray]): person -> (N, D) L2-normalized
            embeddings of that person's reference crops.
        build_fn: an aggregation builder from recognition_metrics.AGGREGATIONS
            — gallery_mean/gallery_medoid return one (D,) vector, gallery_multiref
            returns the full (K, D) matrix.
    Returns:
        dict[str, np.ndarray]: person -> (K, D) float32, K >= 1. Single-vector
        strategies are promoted to (1, D) so every person has the same shape
        rank and the live scorer never branches on the strategy.
    Raises:
        ValueError: if any person has no embeddings, or the aggregated vectors
            do not all share one dimension.
    """
    vectors = {}
    for name in sorted(person_embeddings):
        embeddings = np.asarray(person_embeddings[name], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"{name!r} has no embeddings to aggregate")
        aggregated = np.asarray(build_fn(embeddings), dtype=np.float32)
        vectors[name] = np.atleast_2d(aggregated)

    dims = {v.shape[1] for v in vectors.values()}
    if len(dims) > 1:
        raise ValueError(f"inconsistent embedding dimensions across persons: {sorted(dims)}")
    return vectors


def stack_gallery(person_vectors):
    """Flatten per-person vectors into one labelled matrix.

    Inputs:
        person_vectors (dict[str, np.ndarray]): person -> (K, D), from
            build_person_vectors.
    Returns:
        tuple (vectors, person_ids):
            vectors (np.ndarray): (N, D) float32, all persons stacked in sorted
                name order (N = sum of K).
            person_ids (np.ndarray): (N,) unicode, the owner of each row.
        Returns an empty (0, 0) matrix and an empty id array for empty input.
    """
    names = sorted(person_vectors)
    if not names:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype="<U1")

    blocks = [np.asarray(person_vectors[name], dtype=np.float32) for name in names]
    ids = [name for name, block in zip(names, blocks) for _ in range(block.shape[0])]
    return np.vstack(blocks).astype(np.float32), np.asarray(ids, dtype=np.str_)


def vectors_by_person(vectors, person_ids):
    """Regroup a stacked gallery back into person -> (K, D).

    The inverse of stack_gallery, for callers (the live presence service) that
    want per-person matrices to score against.

    Inputs:
        vectors (np.ndarray): (N, D).
        person_ids (array-like): (N,) person name per row.
    Returns:
        dict[str, np.ndarray]: person -> (K, D) float32, rows in file order.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    ids = [str(p) for p in np.asarray(person_ids).reshape(-1)]
    grouped = {}
    for row, name in zip(matrix, ids):
        grouped.setdefault(name, []).append(row)
    return {name: np.vstack(rows).astype(np.float32) for name, rows in grouped.items()}


def threshold_is_calibrated(encoder, aggregation, threshold):
    """Whether `threshold` is the default AND is being used on the pair it was
    measured for.

    The weak fallback check, used only when no benchmark result is available to
    cross-check against (see f1_threshold_from_runs).

    Returns:
        bool: False only in the dangerous case — the untouched default
        threshold applied to an (encoder, aggregation) pair it was never
        measured on. A threshold the caller passed explicitly is their call,
        and is always treated as calibrated.
    """
    if threshold != DEFAULT_THRESHOLD:
        return True
    return (encoder, aggregation) == THRESHOLD_CALIBRATED_FOR


def f1_threshold_from_runs(runs, encoder, aggregation):
    """The most recently measured MAX-F1 threshold for one (encoder, aggregation).

    This is the deployment threshold for that cell. The TAR@FAR threshold that
    sits beside it in the same record is deliberately NOT read here — that one
    is the ranking yardstick, and storing it in a gallery would ship a cut-off
    chosen to hit a false-accept budget rather than to identify people well.

    Inputs:
        runs (list[dict]): benchmark run payloads, oldest first, as stored in
            results/benchmarks_recognition.json.
        encoder (str): encoder name to look up.
        aggregation (str): aggregation strategy to look up.
    Returns:
        float | None: the max-F1 threshold from the NEWEST run that scored this
        cell, or None if no run contains it (never benchmarked, or the encoder
        errored out in every run).
    """
    for run in reversed(list(runs or [])):
        for result in run.get("results", []):
            if result.get("encoder") != encoder:
                continue
            cell = result.get("aggregations", {}).get(aggregation)
            if not cell:
                continue
            threshold = cell.get("f1_threshold", {}).get("threshold")
            if threshold is not None:
                return float(threshold)
    return None


def threshold_warning(threshold, encoder, aggregation, measured_f1):
    """The warning to print about this threshold, if any.

    Pure so the decision is testable without touching disk: it takes the
    measured value rather than going and finding it.

    Inputs:
        threshold (float): what the caller is about to store.
        encoder (str), aggregation (str): the pair being built.
        measured_f1 (float | None): the max-F1 threshold benchmarked for that
            pair, or None if it has never been benchmarked.
    Returns:
        str | None: a multi-line warning, or None if the threshold checks out.
    """
    if measured_f1 is None:
        # Nothing to compare against. Fall back to the weak check: is the
        # untouched default being applied to a pair it was never measured on?
        if threshold_is_calibrated(encoder, aggregation, threshold):
            return None
        return (
            f"  WARNING: --threshold is still the default {DEFAULT_THRESHOLD}, which was "
            f"measured for\n"
            f"           {THRESHOLD_CALIBRATED_FOR[0]} + {THRESHOLD_CALIBRATED_FOR[1]} - "
            f"NOT for {encoder} + {aggregation}, and there is no\n"
            f"           benchmark result on file for that pair to check against.\n"
            f"           Cosine thresholds do not transfer between encoders. Run\n"
            f"           benchmark_recognition.py and pass the max-F1 threshold it reports."
        )

    if abs(threshold - measured_f1) <= THRESHOLD_TOLERANCE:
        return None

    return (
        f"  WARNING: threshold mismatch for {encoder} + {aggregation}.\n"
        f"           storing        : {threshold:.3f}\n"
        f"           benchmarked max-F1 : {measured_f1:.3f}  <- the deployment threshold\n"
        f"           The max-F1 value is the one to ship. If {threshold:.3f} is the TAR@FAR\n"
        f"           threshold from the benchmark, that is the RANKING yardstick, not a\n"
        f"           setting. Re-run with --threshold {measured_f1:.3f}, or keep yours\n"
        f"           deliberately if you want a stricter/looser operating point."
    )


def build_metadata(encoder, aggregation, threshold, dimension, person_stats,
                   reference_dir, extras=None):
    """Assemble the metadata block stored alongside the vectors.

    Inputs:
        encoder (str): encoder name the vectors were produced with.
        aggregation (str): aggregation strategy name.
        threshold (float): recommended accept threshold (cosine, accept if
            score >= threshold).
        dimension (int): embedding dimension D.
        person_stats (dict[str, dict]): person -> {"crops", "embedded",
            "vectors", "sources"}.
        reference_dir (str): the crop folder the gallery was built from.
        extras (dict | None): extra provenance (embed failures, fallbacks, ...).
    Returns:
        dict: JSON-serializable metadata.
    """
    metadata = {
        "format_version": GALLERY_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "encoder": encoder,
        "aggregation": aggregation,
        "threshold": float(threshold),
        "score": "cosine of L2-normalized embeddings; accept if max over the "
                 "person's vectors >= threshold",
        "dimension": int(dimension),
        "num_persons": len(person_stats),
        "num_vectors": int(sum(s["vectors"] for s in person_stats.values())),
        "persons": {name: person_stats[name] for name in sorted(person_stats)},
        "reference_dir": str(reference_dir),
        "threshold_calibrated_for": {
            "encoder": THRESHOLD_CALIBRATED_FOR[0],
            "aggregation": THRESHOLD_CALIBRATED_FOR[1],
            "is_default_threshold": threshold == DEFAULT_THRESHOLD,
        },
    }
    if extras:
        metadata.update(extras)
    return metadata


# ════════════════════════════════════════════════════════════════════
# I/O — reading crops, embedding, writing and reading the gallery file.
# ════════════════════════════════════════════════════════════════════

def load_benchmark_runs(path=BENCHMARK_RESULTS_FILE):
    """Read the appended benchmark runs, tolerating a missing or broken file.

    A gallery must still be buildable on a machine that has never run the
    benchmark, so every failure here degrades to "no data to check against"
    rather than stopping the build.

    Inputs:
        path (Path): results/benchmarks_recognition.json.
    Returns:
        list[dict]: run payloads oldest first, or [] if unavailable.
    """
    path = Path(path)
    if not path.is_file():
        return []
    try:
        with open(path) as f:
            runs = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return runs if isinstance(runs, list) else []


def save_gallery(path, vectors, person_ids, metadata):
    """Write the gallery to a single compressed .npz.

    Float32 vectors are stored as binary arrays (no JSON text round-trip, so no
    precision loss) and the metadata as one JSON string, which keeps the whole
    file loadable with allow_pickle=False — nothing in it can execute on load.

    Inputs:
        path (Path): destination .npz.
        vectors (np.ndarray): (N, D) float32.
        person_ids (np.ndarray): (N,) person name per row.
        metadata (dict): JSON-serializable, from build_metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=np.asarray(vectors, dtype=np.float32),
        person_ids=np.asarray(person_ids, dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata)),
    )


def load_gallery(path, expected_encoder=None, expected_aggregation=None):
    """Load a gallery, refusing any encoder mismatch.

    This is the guard the whole metadata block exists for. Scoring a probe
    embedding against a gallery built by a DIFFERENT encoder fails silently —
    the cosine is well-formed and wrong — so the caller passes the encoder it is
    about to embed with, and gets an exception instead of bad identities.

    Inputs:
        path (Path | str): gallery .npz written by save_gallery.
        expected_encoder (str | None): encoder the caller will embed probes
            with. None skips the check (only sensible for inspection tools).
        expected_aggregation (str | None): optional same check for the
            aggregation strategy.
    Returns:
        tuple (vectors, person_ids, metadata):
            vectors (np.ndarray): (N, D) float32.
            person_ids (list[str]): (N,) owner of each row.
            metadata (dict): as written by build_metadata.
    Raises:
        ValueError: on an encoder/aggregation mismatch, an unreadable file, or
            an unsupported format version.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        try:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            person_ids = [str(p) for p in data["person_ids"]]
            metadata = json.loads(str(data["metadata"].item()))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path} is not a valid JARVIS gallery file: {exc}") from exc

    version = metadata.get("format_version")
    if version != GALLERY_FORMAT_VERSION:
        raise ValueError(
            f"{path} has gallery format version {version}, but this code reads "
            f"version {GALLERY_FORMAT_VERSION}. Rebuild it with build_gallery.py."
        )

    stored_encoder = metadata.get("encoder")
    if expected_encoder is not None and stored_encoder != expected_encoder:
        raise ValueError(
            f"ENCODER MISMATCH: {path} was built with {stored_encoder!r}, but you "
            f"are scoring with {expected_encoder!r}. Embeddings from different "
            f"encoders are not comparable - the similarities would be meaningless, "
            f"not merely worse. Rebuild the gallery with "
            f"'python tools/build_gallery.py --encoder {expected_encoder}'."
        )

    stored_aggregation = metadata.get("aggregation")
    if expected_aggregation is not None and stored_aggregation != expected_aggregation:
        raise ValueError(
            f"AGGREGATION MISMATCH: {path} was built with {stored_aggregation!r}, "
            f"but you are scoring with {expected_aggregation!r}. The accept "
            f"threshold in this file was calibrated for {stored_aggregation!r}."
        )

    return vectors, person_ids, metadata


def embed_persons(encoder, persons):
    """Embed every person's crops, keeping per-person provenance.

    Inputs:
        encoder: an encoder from load_encoder.
        persons (dict): person -> [(filename, bgr), ...] from load_reference_crops.
    Returns:
        tuple (person_embeddings, person_stats, failures):
            person_embeddings (dict[str, np.ndarray]): person -> (N, D)
                L2-normalized, only for persons with >= 1 embeddable crop.
            person_stats (dict[str, dict]): person -> {"crops", "embedded",
                "sources"} (sources = the crop files that actually embedded).
            failures (int): crops that failed to embed, across all persons.
    """
    person_embeddings, person_stats, failures = {}, {}, 0

    for name in sorted(persons):
        filenames = [f for f, _ in persons[name]]
        images = [im for _, im in persons[name]]

        kept, embeddings, _, fails = embed_crops(encoder, images)
        failures += fails
        if embeddings.shape[0] == 0:
            print(f"    ! {name}: no crop embedded, excluded from the gallery")
            continue

        person_embeddings[name] = embeddings
        person_stats[name] = {
            "crops": len(images),
            "embedded": int(embeddings.shape[0]),
            "sources": [filenames[i] for i in kept],
        }
        if fails:
            print(f"    ! {name}: {fails} crop(s) failed to embed")

    return person_embeddings, person_stats, failures


def print_summary(metadata, output_path):
    """Print the per-person table and the metadata that will ship with it."""
    persons = metadata["persons"]

    print("=" * 66)
    print("  JARVIS face gallery")
    print("=" * 66)
    header = f"    {'Person':<20}  {'crops':>6}  {'embedded':>9}  {'vectors':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name in sorted(persons):
        s = persons[name]
        print(f"    {name:<20}  {s['crops']:>6}  {s['embedded']:>9}  {s['vectors']:>8}")
    print("    " + "-" * (len(header) - 4))
    print(f"    {'TOTAL':<20}  {sum(s['crops'] for s in persons.values()):>6}  "
          f"{sum(s['embedded'] for s in persons.values()):>9}  "
          f"{metadata['num_vectors']:>8}")

    print(f"\n    encoder     : {metadata['encoder']}  ({metadata['dimension']}-D)")
    print(f"    aggregation : {metadata['aggregation']}")
    print(f"    threshold   : {metadata['threshold']:.3f}  "
          f"({metadata.get('threshold_kind', 'max-F1')}; cosine, accept if >=)")
    if metadata.get("embed_failures"):
        print(f"    embed fails : {metadata['embed_failures']}")
    if metadata.get("alignment_fallbacks"):
        print(f"    align falls : {metadata['alignment_fallbacks']} "
              f"(crops the encoder could not re-detect/align internally)")
    print(f"\n  Saved -> {output_path}\n")


# ════════════════════════════════════════════════════════════════════
# Entry point.
# ════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the live face gallery from the curated reference crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR,
                        help="data/reference_faces: one verified folder per person.")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER,
                        help="Encoder to embed with. MUST match what the live "
                             "system will use to embed probes.")
    parser.add_argument("--aggregation", default=DEFAULT_AGGREGATION,
                        choices=sorted(AGGREGATIONS),
                        help="How to aggregate each person's vectors.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Cosine accept threshold stored in the gallery "
                             "metadata. This should be the MAX-F1 threshold of "
                             "the encoder+aggregation you are building with (the "
                             "F1thr column of benchmark_recognition.py) - that is "
                             "the deployment number. Do NOT pass the benchmark's "
                             "TAR@FAR threshold: that one is only the yardstick "
                             "used to rank encoders at a fixed false-accept "
                             "budget. The default is the max-F1 value for "
                             f"{THRESHOLD_CALIBRATED_FOR[0]} + "
                             f"{THRESHOLD_CALIBRATED_FOR[1]} and does NOT carry "
                             "over to other pairs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination .npz gallery file.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output file if it already exists.")
    return parser.parse_args()


def main():
    args = parse_args()

    reference_dir = args.reference_dir.resolve()
    if not reference_dir.is_dir():
        sys.exit(f"ERROR: reference dir does not exist: {reference_dir}")

    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        sys.exit(f"ERROR: {output_path} already exists. Pass --overwrite to replace it.")

    measured_f1 = f1_threshold_from_runs(
        load_benchmark_runs(), args.encoder, args.aggregation
    )
    warning = threshold_warning(args.threshold, args.encoder, args.aggregation, measured_f1)
    if warning:
        print()
        print(warning)

    print(f"\n  Loading reference crops from {reference_dir} ...")
    persons = load_reference_crops(reference_dir)
    if not persons:
        sys.exit(f"ERROR: no person folders with crops found in {reference_dir}.")
    for name in sorted(persons):
        print(f"    {name}: {len(persons[name])} crops")

    print(f"\n  Building encoder {args.encoder!r} ...", flush=True)
    try:
        encoder = load_encoder(args.encoder)
    except Exception as exc:
        sys.exit(f"ERROR: could not load encoder {args.encoder!r}: {exc}")

    print(f"  Embedding with {args.encoder} ...", flush=True)
    person_embeddings, person_stats, failures = embed_persons(encoder, persons)
    if not person_embeddings:
        sys.exit("ERROR: no crop embedded successfully; nothing to build a gallery from.")

    build_fn, _ = AGGREGATIONS[args.aggregation]
    person_vectors = build_person_vectors(person_embeddings, build_fn)
    vectors, person_ids = stack_gallery(person_vectors)

    for name, block in person_vectors.items():
        person_stats[name]["vectors"] = int(block.shape[0])

    metadata = build_metadata(
        encoder=args.encoder,
        aggregation=args.aggregation,
        threshold=args.threshold,
        dimension=int(vectors.shape[1]),
        person_stats=person_stats,
        reference_dir=reference_dir,
        extras={
            # Self-describing, so nobody has to guess later which of the
            # benchmark's two thresholds this number is. "max-F1" exactly when
            # it matches the benchmarked deployment threshold for this pair.
            "threshold_kind": ("max-F1" if warning is None
                               else "custom (does not match the benchmarked max-F1)"),
            "benchmarked_f1_threshold": measured_f1,
            "embed_failures": failures,
            "alignment_fallbacks": getattr(encoder, "fallback_count", None),
        },
    )

    save_gallery(output_path, vectors, person_ids, metadata)
    print()
    print_summary(metadata, output_path)


if __name__ == "__main__":
    main()
