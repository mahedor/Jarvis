"""
JARVIS Engineering Log — Data Layer
===================================
Reads the benchmark artifacts in results/ and reshapes them for the
/engineering pages. Separated from the routes so the shaping is pure and
unit-testable without a Flask request.

STDLIB ONLY, DELIBERATELY. It would be shorter to import the winner-selection
logic from tools/benchmark_recognition.py, but that module pulls in cv2 and
numpy through face_utils, and the web demo's whole dependency list is
'anthropic flask python-dotenv' (see CLAUDE.md). Making the chat UI require
OpenCV so a documentation page can render a table is a bad trade, so the small
amount of logic that overlaps is re-derived here from the JSON. The JSON schema
IS the contract between the two halves - that is what the artifacts are for.

FRESHNESS. Everything is read per request, cached on file mtime, so rerunning a
benchmark updates the pages on the next refresh with no regeneration step to
remember. The cache exists only so a reload does not re-parse ~400 KB for
nothing.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DETECTION_FILE = RESULTS_DIR / "benchmarks_detection.json"
RECOGNITION_FILE = RESULTS_DIR / "benchmarks_recognition.json"

# The FAR the recognition benchmark ranks encoders at. Mirrors PRIMARY_FAR in
# benchmark_recognition.py; kept as a literal to avoid importing it (see above).
PRIMARY_FAR = "0.01"

_cache = {}


# ══════════════════════════════════════════════════════════════════
# Loading (I/O) — mtime-cached JSON reads.
# ══════════════════════════════════════════════════════════════════

def load_runs(path):
    """Load a benchmark results file, re-reading only when it changes on disk.

    Inputs:
        path (Path): a results/*.json written by one of the benchmarks.
    Returns:
        list[dict]: run payloads oldest first, or [] if the file is missing or
        unreadable. A documentation page must never 500 because a benchmark has
        not been run yet.
    """
    path = Path(path)
    if not path.is_file():
        return []
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []

    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        with open(path) as f:
            runs = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(runs, list):
        runs = []

    _cache[path] = (mtime, runs)
    return runs


# ══════════════════════════════════════════════════════════════════
# Pure shaping — functions of their arguments only.
# ══════════════════════════════════════════════════════════════════

def latest_full_run_per_dataset(runs):
    """The newest UNLIMITED run for each detection dataset.

    Runs with a --limit are smoke tests over a handful of images; their AP is
    not comparable to a full-dataset pass and showing them beside real numbers
    would be misleading. They are excluded rather than flagged.

    Inputs:
        runs (list[dict]): detection run payloads, oldest first.
    Returns:
        dict[str, dict]: dataset -> newest full run, dataset keys sorted.
    """
    latest = {}
    for run in runs or []:
        if run.get("limit") is not None:
            continue
        dataset = run.get("dataset")
        if not dataset:
            continue  # legacy entries predating the dataset field
        latest[dataset] = run
    return {k: latest[k] for k in sorted(latest)}


def detection_rows(run):
    """Per-detector rows for one detection run, best AP first.

    Returns:
        tuple (rows, errors): rows are dicts of the metrics the table shows;
        errors are {"detector", "error"} for backends that were skipped.
    """
    rows, errors = [], []
    for result in (run or {}).get("results", []):
        if "error" in result:
            errors.append({"detector": result.get("detector", "?"),
                           "error": result["error"]})
            continue
        rows.append({
            "detector": result.get("detector", "?"),
            "ap": result.get("ap"),
            "roc_auc": result.get("roc_auc"),
            "f1": result.get("f1"),
            "f1_threshold": result.get("f1_threshold"),
            "mean_ms": result.get("mean_ms"),
            "p95_ms": result.get("p95_ms"),
            "fps": result.get("fps"),
        })
    rows.sort(key=lambda r: (r["ap"] is None, -(r["ap"] or 0.0)))
    return rows, errors


def recognition_matrix(run):
    """The encoder x aggregation grid for one recognition run.

    Inputs:
        run (dict): a recognition run payload.
    Returns:
        dict with:
            aggregations (list[str]): column order, as the run recorded it.
            cells (dict): (encoder, aggregation) flattened to
                "encoder|aggregation" -> metric dict, since Jinja cannot
                subscript a tuple key.
            encoders (list[str]): row order, best primary-FAR TAR first.
            errors (list[dict]): encoders that were skipped.
    """
    run = run or {}
    aggregations = run.get("aggregations_run") or []
    cells, encoders, errors = {}, [], []

    for result in run.get("results", []):
        encoder = result.get("encoder", "?")
        if "error" in result:
            errors.append({"encoder": encoder, "error": result["error"]})
            continue
        encoders.append(encoder)
        for aggregation, metrics in (result.get("aggregations") or {}).items():
            cells[f"{encoder}|{aggregation}"] = {
                "tar1": _tar(metrics, "0.01"),
                "tar01": _tar(metrics, "0.001"),
                "yardstick_threshold": _tar_threshold(metrics, PRIMARY_FAR),
                "rank1": metrics.get("rank1_accuracy"),
                "f1": (metrics.get("f1_threshold") or {}).get("f1"),
                "deployment_threshold": (metrics.get("f1_threshold") or {}).get("threshold"),
                "dprime": (metrics.get("separation") or {}).get("dprime_vs_stranger"),
            }

    def rank(encoder):
        best = max((cells[f"{encoder}|{a}"]["tar1"] or 0.0)
                   for a in aggregations if f"{encoder}|{a}" in cells) \
            if any(f"{encoder}|{a}" in cells for a in aggregations) else 0.0
        return -best

    encoders.sort(key=rank)
    latency = {r.get("encoder"): r.get("latency", {})
               for r in run.get("results", []) if "error" not in r}

    return {"aggregations": aggregations, "cells": cells, "encoders": encoders,
            "errors": errors, "latency": latency}


def pick_winner(matrix):
    """The best (encoder, aggregation) cell by TAR at the primary FAR.

    Mirrors benchmark_recognition's WINNER selection: ranked on the yardstick,
    but the threshold reported for deployment is the max-F1 one. Both are
    returned, deliberately named apart, so a template cannot confuse them.

    Returns:
        dict | None: {"encoder", "aggregation", "tar1", "yardstick_threshold",
        "deployment_threshold", "f1"}, or None if the grid is empty.
    """
    best = None
    for encoder in matrix.get("encoders", []):
        for aggregation in matrix.get("aggregations", []):
            cell = matrix["cells"].get(f"{encoder}|{aggregation}")
            if cell is None or cell["tar1"] is None:
                continue
            if best is None or cell["tar1"] > best["tar1"]:
                best = dict(cell, encoder=encoder, aggregation=aggregation)
    return best


def explorer_payload(run):
    """Score distributions for the client-side threshold explorer.

    Ships the stored 40-bin histograms rather than raw scores, because raw
    scores are not persisted - the explorer is therefore a BINNED approximation
    (bin width 0.05 over cosine [-1, 1]) and the page says so. Good enough to
    show the shape of the precision/recall trade, not a substitute for the
    exact numbers the benchmark computed.

    Returns:
        dict: "encoder|aggregation" -> {"genuine", "impostor", "stranger"},
        each {"edges": [...], "counts": [...]}, plus "meta" describing the run.
    """
    run = run or {}
    payload = {}
    for result in run.get("results", []):
        if "error" in result:
            continue
        encoder = result.get("encoder", "?")
        for aggregation, metrics in (result.get("aggregations") or {}).items():
            distributions = metrics.get("distributions") or {}
            entry = {}
            for population in ("genuine", "impostor", "stranger"):
                block = distributions.get(population) or {}
                histogram = block.get("histogram")
                if histogram:
                    entry[population] = histogram
            if entry:
                payload[f"{encoder}|{aggregation}"] = entry
    return payload


def sampling_comparison(runs, min_dev_images=30):
    """Dev-sample AP against full-dataset AP, per dataset.

    Evidence for the sampling-bias story on the overview page. Random noise
    moves detectors independently; a biased SAMPLE moves them together, because
    they are all being asked an easier or harder question than the full set
    asks. Computing it here rather than writing the numbers into the page keeps
    the claim falsifiable — rerun the benchmarks and the argument updates or
    stops being made.

    Inputs:
        runs (list[dict]): detection run payloads.
        min_dev_images (int): ignore tiny smoke runs below this size; they are
            not development samples anyone reasoned from.
    Returns:
        list[dict]: per dataset, {"dataset", "dev_images", "full_images",
        "rows": [{"detector", "dev_ap", "full_ap", "delta"}], "same_direction",
        "max_shift"}, only for datasets having both a dev and a full run.
    """
    dev, full = {}, {}
    for run in runs or []:
        dataset = run.get("dataset")
        if not dataset:
            continue
        if run.get("limit") is None:
            full[dataset] = run
        elif run.get("num_images", 0) >= min_dev_images:
            # Keep the LARGEST dev sample — the most defensible one available.
            best = dev.get(dataset)
            if best is None or run["num_images"] >= best["num_images"]:
                dev[dataset] = run

    comparisons = []
    for dataset in sorted(set(dev) & set(full)):
        dev_ap = {r["detector"]: r["ap"] for r in dev[dataset]["results"]
                  if "error" not in r and r.get("ap") is not None}
        full_ap = {r["detector"]: r["ap"] for r in full[dataset]["results"]
                   if "error" not in r and r.get("ap") is not None}
        rows = [{"detector": d, "dev_ap": dev_ap[d], "full_ap": full_ap[d],
                 "delta": full_ap[d] - dev_ap[d]}
                for d in sorted(set(dev_ap) & set(full_ap))]
        if not rows:
            continue
        deltas = [r["delta"] for r in rows]
        comparisons.append({
            "dataset": dataset,
            "dev_images": dev[dataset]["num_images"],
            "full_images": full[dataset]["num_images"],
            "rows": rows,
            "same_direction": all(d > 0 for d in deltas) or all(d < 0 for d in deltas),
            "max_shift": max(abs(d) for d in deltas),
        })
    # Strongest evidence first: datasets where every detector moved together,
    # then by how far anything moved. Alphabetical order would open with FDDB,
    # the control, which is the one dataset that barely shifted.
    comparisons.sort(key=lambda c: (not c["same_direction"], -c["max_shift"]))
    return comparisons


def candidate_box_rates(run):
    """Candidate boxes emitted per image, per detector.

    The number behind the honesty note about YOLO's confidence floor: a
    detector that proposes hundreds of boxes per image and one that proposes a
    handful are not being scored on the same kind of output, even though AP
    handles it correctly.

    Returns:
        list[dict]: {"detector", "per_image", "total", "false_positives"},
        busiest first. Empty if the run has no usable results.
    """
    run = run or {}
    images = run.get("num_images") or 0
    if not images:
        return []
    rates = []
    for result in run.get("results", []):
        if "error" in result or result.get("num_detections") is None:
            continue
        rates.append({
            "detector": result["detector"],
            "per_image": result["num_detections"] / images,
            "total": result["num_detections"],
            "false_positives": result.get("total_fp"),
        })
    rates.sort(key=lambda r: -r["per_image"])
    return rates


def _tar(metrics, far):
    return ((metrics.get("tar_at_far") or {}).get(far) or {}).get("tar")


def _tar_threshold(metrics, far):
    return ((metrics.get("tar_at_far") or {}).get(far) or {}).get("threshold")


# ══════════════════════════════════════════════════════════════════
# Page-level assembly.
# ══════════════════════════════════════════════════════════════════

def detection_page_data():
    """Everything the detection page renders."""
    runs = load_runs(DETECTION_FILE)
    latest = latest_full_run_per_dataset(runs)
    datasets = []
    for dataset, run in latest.items():
        rows, errors = detection_rows(run)
        datasets.append({
            "dataset": dataset,
            "run": run,
            "rows": rows,
            "errors": errors,
            # min_box_size only started being recorded on 2026-08-08; older
            # entries legitimately have no value and must not read as 0.
            "min_box_size_known": "min_box_size" in run,
        })
    return {"datasets": datasets, "total_runs": len(runs),
            "source": _relative(DETECTION_FILE)}


def overview_page_data():
    """Evidence the narrative on the overview page cites.

    The prose makes claims; these are the numbers backing them, read from the
    artifacts so a rerun either keeps the argument true or exposes it.
    """
    runs = load_runs(DETECTION_FILE)
    full = latest_full_run_per_dataset(runs)
    # WIDER FACE is the crowded set, so it is where per-image box counts say
    # the most; fall back to whatever full run exists.
    busiest = full.get("widerface") or (next(iter(full.values()), None))
    return {
        "sampling": sampling_comparison(runs),
        "box_rates": candidate_box_rates(busiest),
        "box_rate_run": busiest,
    }


def recognition_page_data():
    """Everything the recognition page renders."""
    runs = load_runs(RECOGNITION_FILE)
    latest = runs[-1] if runs else None
    matrix = recognition_matrix(latest)
    return {
        "run": latest,
        "matrix": matrix,
        "winner": pick_winner(matrix),
        "explorer": explorer_payload(latest),
        "total_runs": len(runs),
        "source": _relative(RECOGNITION_FILE),
    }


def _relative(path):
    try:
        return str(Path(path).relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)
