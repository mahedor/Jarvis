"""
JARVIS Face Recognition — Stage 2 Encoder Benchmark
===================================================
Compares face ENCODERS (the ones behind load_encoder in face_utils) on OPEN-SET
identification and recommends which to ship plus a similarity threshold. Modelled
on benchmark_detection.py: pluggable backends, per-backend failure isolation,
CPU-latency timing, a ranked summary table, and append-only JSON persistence.

    reference crops  ->  embed  ->  image-disjoint enroll/probe split
                     ->  build gallery (pluggable aggregation)
                     ->  score probes + LFW strangers by cosine
                     ->  TAR@FAR, rank-1, F1 threshold, d-prime, latency

----------------------------------------------------------------------
TWO THRESHOLDS, TWO JOBS. Every cell in this report has two similarity
thresholds attached to it, and confusing them is the easiest way to ship the
wrong one:

    TAR@FAR threshold  =  the YARDSTICK. Derived by pinning the false-accept
        rate to a fixed budget (1% here) so every encoder is measured at the
        same operating point and the comparison is apples-to-apples. It exists
        to RANK encoders. Do not deploy it.

    max-F1 threshold   =  the SHIP threshold. The cut-off that best balances
        precision and recall on the actual score distributions. This is the
        number that goes into the gallery via
        'build_gallery.py --threshold', and the WINNER line below prints the
        exact command.

Both are printed, always labelled. The F1thr column of each table is the
deployment threshold for that cell.
----------------------------------------------------------------------

GROUND TRUTH: data/reference_faces/<person>/*.jpg — one folder per person,
produced by collect_faces.py and MANUALLY VERIFIED, so the folder name is a
trusted label. Folders and files starting with "_" (e.g. _noise/, _face_card.jpg)
and manifest.json are ignored.

STRANGERS: LFW via sklearn.datasets.fetch_lfw_people, downloaded on first use
into --lfw-dir. None of the household is in LFW, so every stranger score is a
true non-mate (nothing is excluded).

This script owns all the I/O; the scoring math lives in recognition_metrics
(pure, unit-tested). It never reimplements a metric — it orchestrates
load_encoder + recognition_metrics, exactly as benchmark_detection orchestrates
load_detector + detection_metrics.

CPU-only. Each encoder is isolated: if a backend's library is missing or a
construction/inference call blows up, it is reported as skipped/errored and the
others still complete.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ── Path setup — this file lives in tools/ next to the modules it imports ──
TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from face_utils import load_encoder, load_reference_crops, embed_crops
from recognition_metrics import (
    AGGREGATIONS,
    source_image_key,
    split_enroll_probe,
    tar_at_far,
    rank1_accuracy,
    best_f1_threshold,
    dprime,
    summarize_scores,
    histogram,
    roc_auc,
    average_precision,
    score_sweep,
    operating_point,
)

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "benchmarks_recognition.json"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "data" / "reference_faces"
DEFAULT_LFW_DIR = REPO_ROOT / "data" / "lfw"

DEFAULT_ENCODERS = ["arcface", "facenet512", "vgg-face", "adaface", "dlib"]
FAR_TARGETS = [0.01, 0.001]          # TAR is reported at each of these FARs
SWEEP_STEPS = 200                    # grid intervals for the stored ROC/PR sweep
PRIMARY_FAR = "0.01"                 # the table ranks encoders by TAR @ this FAR
PRIMARY_FAR_LABEL = f"{float(PRIMARY_FAR) * 100:g}%"   # "1%", for the report text


# ════════════════════════════════════════════════════════════════════
# I/O — LFW strangers. (Reference crops are loaded by face_utils'
# load_reference_crops, shared with build_gallery.py so the benchmark and
# the shipped gallery always read the same folders by the same rules.)
# ════════════════════════════════════════════════════════════════════

def load_strangers(num_strangers, seed, lfw_dir):
    """Download (once) and sample LFW faces to act as open-set strangers.

    fetch_lfw_people caches into --lfw-dir on first call (~200 MB download). The
    funneled, aligned face images are returned as BGR uint8 so they match the
    reference crops' format; each encoder still re-detects/aligns internally.

    Inputs:
        num_strangers (int): how many stranger faces to sample.
        seed (int): RNG seed for a reproducible sample.
        lfw_dir (Path): cache directory (created if absent).
    Returns:
        list[np.ndarray]: up to num_strangers BGR uint8 images.
    """
    from sklearn.datasets import fetch_lfw_people

    lfw_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Loading LFW strangers (cache: {lfw_dir}; first run downloads ~200 MB)...")
    lfw = fetch_lfw_people(data_home=str(lfw_dir), color=True, resize=1.0, funneled=True)

    images = lfw.images  # (N, H, W, 3), RGB
    total = images.shape[0]
    rng = np.random.default_rng(seed)
    take = min(num_strangers, total)
    picks = rng.choice(total, size=take, replace=False)

    # sklearn returns floats; scale to 0-255 if it handed back a 0-1 range.
    scale = 255.0 if float(images.max()) <= 1.0 else 1.0
    strangers = []
    for i in picks:
        rgb = np.clip(images[i] * scale, 0, 255).astype(np.uint8)
        strangers.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  Sampled {len(strangers)} strangers from {total} LFW faces.")
    return strangers


# ════════════════════════════════════════════════════════════════════
# Latency stats over the timed region (face_utils.embed_crops times each
# encoder.embed() call and nothing else).
# ════════════════════════════════════════════════════════════════════

def latency_stats(latencies_ms):
    """Mean / p50 / p95 ms over a list of per-face embed latencies."""
    if not latencies_ms:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(latencies_ms)
    n = len(ordered)

    def pct(p):
        return ordered[min(n - 1, int(round((p / 100.0) * (n - 1))))]

    return {
        "count": n,
        "mean_ms": round(sum(ordered) / n, 3),
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
    }


# ════════════════════════════════════════════════════════════════════
# Per-encoder evaluation: embed, split image-disjoint, score, measure.
# ════════════════════════════════════════════════════════════════════

def embed_and_split(encoder, persons, enroll_fraction, seed):
    """Embed each person's crops and split them image-disjoint into enroll/probe.

    Inputs:
        encoder: a face_utils encoder.
        persons (dict): name -> [(filename, bgr), ...] from load_reference_crops.
        enroll_fraction (float): fraction of PHOTOS enrolled per person.
        seed (int): seed for the reproducible per-person photo shuffle.
    Returns:
        tuple (person_data, latencies_ms, skipped, failures):
            person_data (dict): name -> {"enroll": (E, D), "probe": (P, D)},
                both L2-normalized, only for persons with >= 2 source photos.
            latencies_ms (list[float]): embed latencies across all persons.
            skipped (list[str]): persons dropped (fewer than 2 source photos or
                nothing embedded), for the report.
            failures (int): total crops that failed to embed.
    """
    rng = np.random.default_rng(seed)
    person_data, latencies, skipped, failures = {}, [], [], 0

    for name in sorted(persons):
        filenames = [f for f, _ in persons[name]]
        images = [im for _, im in persons[name]]

        kept, embeddings, lat, fails = embed_crops(encoder, images)
        latencies.extend(lat)
        failures += fails
        if embeddings.shape[0] == 0:
            skipped.append(name)
            continue

        kept_keys = [source_image_key(filenames[i]) for i in kept]
        enroll_idx, probe_idx = split_enroll_probe(kept_keys, enroll_fraction, rng)
        if not enroll_idx or not probe_idx:
            skipped.append(name)  # only one source photo -> can't split disjoint
            continue

        person_data[name] = {
            "enroll": embeddings[enroll_idx],
            "probe": embeddings[probe_idx],
        }

    return person_data, latencies, skipped, failures


def score_aggregation(person_data, stranger_embs, build_fn, score_fn, hist_bins):
    """Score one aggregation strategy into the full metric set.

    Builds each person's gallery, then produces genuine / impostor / stranger
    score populations and the probe-vs-all-galleries matrix used for rank-1.

    Inputs:
        person_data (dict): name -> {"enroll", "probe"} embeddings.
        stranger_embs (np.ndarray): (S, D) L2-normalized stranger embeddings.
        build_fn: aggregation gallery builder (from AGGREGATIONS).
        score_fn: aggregation scorer (from AGGREGATIONS).
        hist_bins (int): histogram bin count for the stored distributions.
    Returns:
        dict of metrics (rank1, tar_far, f1_threshold, d-prime, distributions,
        counts) ready to drop into the results JSON.
    """
    names = sorted(person_data)
    col_of = {name: j for j, name in enumerate(names)}
    galleries = {name: build_fn(person_data[name]["enroll"]) for name in names}

    genuine, impostor, stranger = [], [], []
    probe_rows, true_idx = [], []

    for name in names:
        probes = person_data[name]["probe"]  # (P, D)
        columns = [score_fn(probes, galleries[other]) for other in names]  # each (P,)
        matrix = np.stack(columns, axis=1)  # (P, num_galleries)
        probe_rows.append(matrix)
        true_idx.extend([col_of[name]] * probes.shape[0])

        own = col_of[name]
        genuine.extend(matrix[:, own].tolist())
        for j in range(matrix.shape[1]):
            if j != own:
                impostor.extend(matrix[:, j].tolist())

    if stranger_embs.shape[0] > 0:
        for name in names:
            stranger.extend(score_fn(stranger_embs, galleries[name]).tolist())

    score_matrix = np.vstack(probe_rows) if probe_rows else np.empty((0, len(names)))
    genuine = np.asarray(genuine)
    impostor = np.asarray(impostor)
    stranger = np.asarray(stranger)
    nonmate = np.concatenate([impostor, stranger]) if (impostor.size or stranger.size) else np.asarray([])

    f1_thr, f1_p, f1_r, f1 = best_f1_threshold(genuine, nonmate)

    def describe(values):
        return {"summary": summarize_scores(values), "histogram": histogram(values, hist_bins)}

    # Keep the unrounded TAR@FAR results: the thresholds feed the curve markers
    # below, and a 4-dp threshold would place the marker off its own curve.
    far_raw = {f"{far}": tar_at_far(genuine, nonmate, far) for far in FAR_TARGETS}
    far_results = {
        key: {k: round(v, 4) for k, v in value.items()} for key, value in far_raw.items()
    }

    # The ROC/PR curves behind the scalars above. The sweep is a fixed grid
    # (comparable across encoders, small enough to store); the scalars and the
    # shipped-threshold markers are computed at full precision off-grid, so the
    # grid resolution never moves a number anyone quotes.
    curves = {
        "roc_auc": round(roc_auc(genuine, nonmate), 4),
        "average_precision": round(average_precision(genuine, nonmate), 4),
        "sweep": score_sweep(genuine, nonmate, SWEEP_STEPS),
        "markers": {
            **{f"tar@far={far}": operating_point(genuine, nonmate, far_raw[f"{far}"]["threshold"])
               for far in FAR_TARGETS},
            "best_f1": operating_point(genuine, nonmate, f1_thr),
        },
    }

    return {
        "rank1_accuracy": round(rank1_accuracy(score_matrix, np.asarray(true_idx)), 4),
        "tar_at_far": far_results,
        "f1_threshold": {
            "threshold": round(f1_thr, 4), "precision": round(f1_p, 4),
            "recall": round(f1_r, 4), "f1": round(f1, 4),
        },
        "curves": curves,
        "separation": {
            "dprime_vs_impostor": round(dprime(genuine, impostor), 4),
            "dprime_vs_stranger": round(dprime(genuine, stranger), 4),
        },
        "distributions": {
            "genuine": describe(genuine),
            "impostor": describe(impostor),
            "stranger": describe(stranger),
        },
        "counts": {
            "genuine": int(genuine.size), "impostor": int(impostor.size),
            "stranger": int(stranger.size),
        },
    }


def evaluate_encoder(name, persons, strangers, aggregations, enroll_fraction, seed, hist_bins):
    """Build one encoder and score it under every requested aggregation.

    Fully isolated: a missing library or a construction/inference error is
    caught and returned as {"encoder", "error"} so the caller can keep going.

    Returns:
        dict: {"encoder", "aggregations": {name: metrics}, "latency", ...} on
        success, or {"encoder", "error"} on skip/failure.
    """
    print(f"  - {name}: building...", flush=True)
    try:
        encoder = load_encoder(name)
    except Exception as exc:  # missing dep, deferred stub, or bad construction
        print(f"    ! {name} skipped: {exc}")
        return {"encoder": name, "error": str(exc).splitlines()[0]}

    print(f"  * {name}: embedding + splitting...", flush=True)
    try:
        person_data, latencies, skipped, failures = embed_and_split(
            encoder, persons, enroll_fraction, seed
        )
        _, stranger_embs, stranger_lat, stranger_fails = embed_crops(encoder, strangers)
    except Exception as exc:  # an inference blow-up mid-run
        print(f"    ! {name} failed during embedding: {exc}")
        return {"encoder": name, "error": f"embed: {exc}"}

    if len(person_data) < 2:
        msg = f"only {len(person_data)} person(s) had >=2 source photos; need >=2 to score impostors"
        print(f"    ! {name} skipped: {msg}")
        return {"encoder": name, "error": msg}

    latencies = latencies + stranger_lat
    scored = {name_agg: score_aggregation(
        person_data, stranger_embs, build_fn, score_fn, hist_bins
    ) for name_agg, (build_fn, score_fn) in aggregations.items()}

    result = {
        "encoder": name,
        "aggregations": scored,
        "latency": latency_stats(latencies),
        "embed_failures": failures + stranger_fails,
        "fallback_count": getattr(encoder, "fallback_count", None),
        "persons_scored": sorted(person_data),
        "persons_skipped": skipped,
        "num_strangers": int(stranger_embs.shape[0]),
    }
    # Report this encoder's BEST aggregation, not whichever one happens to be
    # first — same reason the WINNER line below ranks over every cell.
    best_agg = max(scored, key=lambda a: scored[a]["tar_at_far"][PRIMARY_FAR]["tar"])
    primary = scored[best_agg]["tar_at_far"][PRIMARY_FAR]["tar"]
    print(f"    best TAR@FAR{PRIMARY_FAR}={primary:.3f} ({best_agg})  "
          f"mean={result['latency']['mean_ms']:.1f}ms/face")
    return result


# ════════════════════════════════════════════════════════════════════
# Reporting + persistence.
# ════════════════════════════════════════════════════════════════════

def print_report(results, aggregation_names, num_persons, num_strangers, enroll_fraction):
    """Print one ranked table per aggregation, sorted by TAR @ the primary FAR."""
    ok = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    print("=" * 78)
    print("  JARVIS Stage 2 - Face Encoder Benchmark (open-set identification)")
    print(f"  persons: {num_persons}  |  strangers: {num_strangers}  |  "
          f"enroll fraction: {enroll_fraction}")
    print(f"  rank by: TAR@FAR{PRIMARY_FAR_LABEL} (yardstick)  |  "
          f"ship: max-F1 threshold (the F1thr column)")
    print("=" * 78)

    for agg in aggregation_names:
        print(f"\n  Aggregation: {agg}")
        header = (f"    {'Encoder':<12}  {'TAR@1%':>7}  {'TAR@.1%':>7}  {'Rank-1':>7}  "
                  f"{'F1thr':>6}  {'F1':>5}  {'dp-strg':>7}  {'mean ms':>8}  {'p95 ms':>8}")
        print(header)
        print("    " + "-" * (len(header) - 4))
        rows = [r for r in ok if agg in r["aggregations"]]
        rows.sort(key=lambda r: r["aggregations"][agg]["tar_at_far"][PRIMARY_FAR]["tar"], reverse=True)
        for r in rows:
            m = r["aggregations"][agg]
            tar1 = m["tar_at_far"]["0.01"]["tar"]
            tar01 = m["tar_at_far"]["0.001"]["tar"]
            print(f"    {r['encoder']:<12}  {tar1:>7.3f}  {tar01:>7.3f}  "
                  f"{m['rank1_accuracy']:>7.3f}  {m['f1_threshold']['threshold']:>6.3f}  "
                  f"{m['f1_threshold']['f1']:>5.3f}  {m['separation']['dprime_vs_stranger']:>7.2f}  "
                  f"{r['latency']['mean_ms']:>8.1f}  {r['latency']['p95_ms']:>8.1f}")

    if errored:
        print("\n  Skipped / errored:")
        for r in errored:
            print(f"    {r['encoder']:<12}  {r['error']}")

    if ok:
        # Winner = the single best (encoder, aggregation) cell across EVERY
        # aggregation, ranked by TAR @ the primary FAR — not just the first
        # aggregation.
        #
        # TWO THRESHOLDS, TWO JOBS (see the header of this file):
        #   - the TAR@FAR threshold is the YARDSTICK. It is chosen to pin every
        #     encoder to the same false-accept budget, which is what makes the
        #     comparison fair. It is a measurement instrument, not a setting.
        #   - the max-F1 threshold is what SHIPS. It is the balance of precision
        #     and recall on this household's own score distributions.
        # They are different numbers for the same winning cell, so print both,
        # each labelled with its job, and hand over the exact build_gallery
        # command that stores the deployment one.
        candidates = [
            (r, agg, r["aggregations"][agg])
            for r in ok
            for agg in aggregation_names
            if agg in r["aggregations"]
        ]
        best_r, best_agg, bm = max(
            candidates, key=lambda c: c[2]["tar_at_far"][PRIMARY_FAR]["tar"]
        )
        op = bm["tar_at_far"][PRIMARY_FAR]
        f1 = bm["f1_threshold"]
        print(f"\n  WINNER: {best_r['encoder']} ({best_agg})")
        print(f"    ranked by TAR@FAR{PRIMARY_FAR_LABEL}={op['tar']:.3f}  "
              f"(yardstick threshold {op['threshold']:.3f}, "
              f"FAR achieved {op['far_achieved']:.4f}) - for COMPARING encoders")
        print(f"    recommended DEPLOYMENT threshold (max-F1) = {f1['threshold']:.3f}  "
              f"(F1={f1['f1']:.3f}, precision={f1['precision']:.3f}, "
              f"recall={f1['recall']:.3f}) - the one to SHIP")
        print(f"    -> python tools/build_gallery.py --encoder {best_r['encoder']} "
              f"--aggregation {best_agg} --threshold {f1['threshold']:.3f}")
    print()


def save_results(payload):
    """Append this run to results/benchmarks_recognition.json (mirrors detection)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    data = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            data = json.load(f)
    data.append(payload)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ════════════════════════════════════════════════════════════════════
# Entry point.
# ════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 face-encoder benchmark (open-set identification).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR,
                        help="data/reference_faces: one verified folder per person.")
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS,
                        help="Encoders to compare (missing deps are skipped).")
    parser.add_argument("--aggregation", default="mean-renormalize",
                        choices=sorted(AGGREGATIONS) + ["all"],
                        help="Gallery aggregation; 'all' runs every strategy.")
    parser.add_argument("--num-strangers", type=int, default=500,
                        help="LFW faces to sample as open-set strangers.")
    parser.add_argument("--enroll-fraction", type=float, default=0.5,
                        help="Fraction of each person's PHOTOS used to enroll.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the enroll/probe split and stranger sample.")
    parser.add_argument("--lfw-dir", type=Path, default=DEFAULT_LFW_DIR,
                        help="Cache dir for the LFW download.")
    parser.add_argument("--hist-bins", type=int, default=40,
                        help="Histogram bins for the stored score distributions.")
    return parser.parse_args()


def main():
    args = parse_args()

    reference_dir = args.reference_dir.resolve()
    if not reference_dir.is_dir():
        sys.exit(f"ERROR: reference dir does not exist: {reference_dir}")

    print(f"\n  Loading reference crops from {reference_dir} ...")
    persons = load_reference_crops(reference_dir)
    if len(persons) < 2:
        sys.exit(f"ERROR: need >=2 person folders to score impostors; found {len(persons)}.")
    for name in sorted(persons):
        print(f"    {name}: {len(persons[name])} crops")

    strangers = load_strangers(args.num_strangers, args.seed, args.lfw_dir.resolve())

    if args.aggregation == "all":
        aggregations = dict(AGGREGATIONS)
    else:
        aggregations = {args.aggregation: AGGREGATIONS[args.aggregation]}
    aggregation_names = list(aggregations)

    print(f"\n  Encoders: {', '.join(args.encoders)}")
    print(f"  Aggregations: {', '.join(aggregation_names)}\n")

    t_start = time.perf_counter()
    results = [
        evaluate_encoder(name, persons, strangers, aggregations,
                         args.enroll_fraction, args.seed, args.hist_bins)
        for name in args.encoders
    ]
    elapsed = time.perf_counter() - t_start

    print()
    print_report(results, aggregation_names, len(persons), len(strangers), args.enroll_fraction)
    print(f"  Total wall time: {elapsed:.1f}s")

    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference_dir": str(reference_dir),
        "num_persons": len(persons),
        "num_strangers": len(strangers),
        "enroll_fraction": args.enroll_fraction,
        "seed": args.seed,
        "primary_far": PRIMARY_FAR,
        "aggregations_run": aggregation_names,
        "results": results,
    }
    save_results(payload)
    print(f"  Saved -> {RESULTS_FILE.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
