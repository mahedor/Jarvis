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
INTENT_FILE = RESULTS_DIR / "benchmarks_intent.json"
CACHING_FILE = RESULTS_DIR / "benchmarks_caching.json"
EVAL_DIR = REPO_ROOT / "tests" / "eval"
# Written by tools/collect_faces.py alongside the private manifest, and
# COMMITTED, because it is the derived, publishable half of a curation run:
# aggregates plus anonymised "folder A/B/C" labels, with no names, filenames,
# paths or per-crop rows in it.
#
# The page used to read data/reference_faces/manifest.json directly. That file
# is provenance for photographs of real people, so it lives in the gitignored
# data/ tree — which meant it was present locally and absent in every build,
# and the published page silently rendered its empty state for months. The
# anonymisation that used to happen here at render time now happens in
# collect_faces at write time, so the names never leave data/ at all and the
# thing the log reads is the thing the log can publish.
ENROLLMENT_SUMMARY = RESULTS_DIR / "enrollment_summary.json"

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


def load_document(path):
    """Load a single JSON object artifact, re-reading only when it changes.

    load_runs' sibling for artifacts that are one document rather than a list
    of runs (the curation manifest). Same mtime cache, same rule that a missing
    or malformed file yields an empty result instead of a 500.

    Inputs:
        path (Path): a JSON file whose top level is an object.
    Returns:
        dict: the parsed document, or {} if absent, unreadable, or not an
        object.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}

    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        with open(path, encoding="utf-8") as f:
            document = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(document, dict):
        document = {}

    _cache[path] = (mtime, document)
    return document


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


def pick_detection_winner(datasets):
    """The detector that wins across datasets, not just on one.

    Deliberately counts DATASET WINS before comparing AP. A detector that takes
    the easy set by a mile and loses both hard ones has not earned anything —
    the whole reason three datasets are run is that a single headline number
    hides exactly that. AP is the tie-break, averaged across datasets.

    Inputs:
        datasets (list[dict]): per-dataset blocks from detection_page_data,
            each {"dataset", "rows": [...best AP first...]}.
    Returns:
        dict | None: {"detector", "wins", "contested", "swept", "per_dataset",
        "mean_ap", "worst"} where per_dataset is [{"dataset", "ap", "f1",
        "f1_threshold", "won"}], or None if there is nothing to compare.
    """
    if not datasets:
        return None

    wins, appearances = {}, {}
    for block in datasets:
        rows = block.get("rows") or []
        if not rows:
            continue
        for row in rows:
            appearances.setdefault(row["detector"], []).append((block["dataset"], row))
        wins[rows[0]["detector"]] = wins.get(rows[0]["detector"], 0) + 1

    if not appearances:
        return None

    contested = len([b for b in datasets if b.get("rows")])

    def rank(detector):
        entries = appearances[detector]
        mean_ap = sum(r["ap"] or 0.0 for _, r in entries) / len(entries)
        return (-wins.get(detector, 0), -mean_ap)

    detector = min(appearances, key=rank)
    entries = appearances[detector]
    per_dataset = [{
        "dataset": name,
        "ap": row["ap"],
        "f1": row["f1"],
        "f1_threshold": row["f1_threshold"],
        "won": any(b["dataset"] == name and b["rows"] and b["rows"][0]["detector"] == detector
                   for b in datasets),
    } for name, row in entries]
    aps = [r["ap"] for r in per_dataset if r["ap"] is not None]

    return {
        "detector": detector,
        "wins": wins.get(detector, 0),
        "contested": contested,
        "swept": wins.get(detector, 0) == contested and contested > 1,
        "per_dataset": sorted(per_dataset, key=lambda r: -(r["ap"] or 0)),
        "mean_ap": (sum(aps) / len(aps)) if aps else None,
        "worst": min(aps) if aps else None,
    }


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
                # Curve summaries; None for runs recorded before curves were
                # persisted, so the table can blank the cell rather than
                # inventing a 0.000 that reads like a measured failure.
                "roc_auc": (metrics.get("curves") or {}).get("roc_auc"),
                "ap": (metrics.get("curves") or {}).get("average_precision"),
            }

    def rank(encoder):
        best = max((cells[f"{encoder}|{a}"]["tar1"] or 0.0)
                   for a in aggregations if f"{encoder}|{a}" in cells) \
            if any(f"{encoder}|{a}" in cells for a in aggregations) else 0.0
        return -best

    encoders.sort(key=rank)
    # Carry fps alongside the milliseconds. The detection side already reports
    # it, and real-time work gets discussed in frames per second — 190 ms/face
    # and "about 5 faces a second" land very differently.
    latency = {}
    for result in run.get("results", []):
        if "error" in result:
            continue
        stats = dict(result.get("latency") or {})
        mean_ms = stats.get("mean_ms") or 0.0
        stats["fps"] = (1000.0 / mean_ms) if mean_ms > 0 else None
        latency[result.get("encoder")] = stats

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


def _exact_counts(metrics):
    """The recorded sweep, flattened to parallel arrays for the explorer.

    The sweep is stored as 201 objects per encoder x aggregation, which is the
    right shape on disk and the wrong one to inline into a page — as JSON in an
    HTML attribute, nine of them run to six figures of markup. Only two numbers
    per grid point are actually needed (accepted genuine, accepted non-mates);
    everything the explorer displays is derived from those. Sending them as two
    integer arrays plus the grid's origin and step costs about a tenth as much
    and lets the client index straight to a threshold instead of searching.

    Inputs:
        metrics (dict): one aggregation's metric block.
    Returns:
        dict | None: {"min", "step", "tp", "fp", "genuine_total",
        "nonmate_total"} — or None for runs recorded before curves existed,
        which is the signal to fall back to the binned histograms.
    """
    sweep = ((metrics.get("curves") or {}).get("sweep")) or []
    if len(sweep) < 2:
        return None

    counts = metrics.get("counts") or {}
    nonmate_total = int(counts.get("impostor", 0)) + int(counts.get("stranger", 0))
    if not counts.get("genuine") or not nonmate_total:
        return None

    return {
        "min": sweep[0]["threshold"],
        "step": round(sweep[1]["threshold"] - sweep[0]["threshold"], 6),
        "tp": [point["tp"] for point in sweep],
        "fp": [point["fp"] for point in sweep],
        "genuine_total": int(counts["genuine"]),
        "nonmate_total": nonmate_total,
    }


def _curve_summary(metrics):
    """The curve scalars and shipped markers, for the client-side plots.

    Only what the canvas cannot derive itself. The curves are drawn from the
    tp/fp arrays already in the payload — precision, recall, TAR and FAR are
    all ratios of those — so the exact AUC/AP scalars and the two off-grid
    operating points are the entire remainder.

    Inputs:
        metrics (dict): one aggregation's metric block.
    Returns:
        dict | None: {"auc", "ap", "markers": {"yardstick", "deployment"}} —
        or None when the run recorded no curves.
    """
    curves = metrics.get("curves") or {}
    markers = curves.get("markers") or {}
    yardstick = markers.get(f"tar@far={float(PRIMARY_FAR):g}")
    deployment = markers.get("best_f1")
    if not curves.get("sweep") or not yardstick or not deployment:
        return None

    def point(marker):
        return {
            "threshold": marker.get("threshold"),
            "tar": marker.get("tar"),
            "far": marker.get("far"),
            "recall": marker.get("recall"),
            "precision": marker.get("precision"),
        }

    return {
        "auc": curves.get("roc_auc"),
        "ap": curves.get("average_precision"),
        "markers": {"yardstick": point(yardstick), "deployment": point(deployment)},
    }


def explorer_payload(run):
    """Score distributions for the client-side threshold explorer.

    Ships two things per cell, for two different jobs:

    - the stored 40-bin histograms, which draw the SHAPE of the three score
      populations on the canvas;
    - "exact", the recorded threshold sweep (see _exact_counts), which supplies
      the COUNTS the readout quotes.

    Splitting them matters because the histograms cannot answer the readout's
    question honestly. Resolved to 0.05, a threshold landing mid-bin has to
    have its bin split by interpolation, which invents fractional samples — the
    explorer used to report things like "0.3 false accepts" at a threshold
    whose true count is zero. The sweep is counted from the real scores at
    benchmark time, so wherever "exact" is present the readout is not an
    estimate at all. Runs recorded before curve persistence have no sweep and
    fall back to the old binned path.

    Returns:
        dict: "encoder|aggregation" -> {"genuine", "impostor", "stranger"}
        histograms, plus "exact" when the run carries a sweep.
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
                exact = _exact_counts(metrics)
                if exact:
                    entry["exact"] = exact
                curve = _curve_summary(metrics)
                if curve:
                    entry["curve"] = curve
                payload[f"{encoder}|{aggregation}"] = entry
    return payload


def explorer_has_curves(payload):
    """Whether any cell can be plotted client-side.

    A cell needs both halves: the tp/fp grid the curves are drawn from and the
    scalars/markers that cannot be recovered from it. Anything less and the
    section stays hidden rather than rendering empty axes.
    """
    return any("curve" in cell and "exact" in cell for cell in (payload or {}).values())


def explorer_is_exact(payload):
    """Whether every cell in the explorer payload has exact counts.

    Drives the page's caveat: the "binned approximation" warning must appear
    when any cell would fall back to interpolation, and must NOT appear when
    none do.
    """
    return bool(payload) and all("exact" in cell for cell in payload.values())


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

def detection_sweeps(run):
    """Confidence sweeps for one run, keyed by detector.

    Only runs recorded after confidence_sweep was added to benchmark_detection
    carry these. Older runs are not broken, they simply cannot drive the
    threshold explorer — the page shows it for the datasets that have data and
    says nothing for the rest, rather than blocking on a full re-run.

    Returns:
        dict[str, list[dict]]: detector -> sweep points. Empty when absent.
    """
    sweeps = {}
    for result in (run or {}).get("results", []):
        if "error" in result:
            continue
        sweep = result.get("confidence_sweep")
        if sweep:
            sweeps[result["detector"]] = sweep
    return sweeps


def detection_page_data():
    """Everything the detection page renders."""
    runs = load_runs(DETECTION_FILE)
    latest = latest_full_run_per_dataset(runs)
    datasets, explorer = [], {}
    for dataset, run in latest.items():
        rows, errors = detection_rows(run)
        sweeps = detection_sweeps(run)
        if sweeps:
            explorer[dataset] = sweeps
        datasets.append({
            "dataset": dataset,
            "run": run,
            "rows": rows,
            "errors": errors,
            # min_box_size only started being recorded on 2026-08-08; older
            # entries legitimately have no value and must not read as 0.
            "min_box_size_known": "min_box_size" in run,
            "has_sweep": bool(sweeps),
        })
    # min_box_size changes what the numbers MEAN, not just their value: it sets
    # how small a ground-truth face still counts as one to find. Runs at
    # different settings are not comparable to each other, so the page has to
    # say so rather than tabling them side by side. None = the run pre-dates
    # the field being recorded.
    regimes = [(b["dataset"], b["run"].get("min_box_size")) for b in datasets]
    distinct = {value for _, value in regimes}
    # How many confidences the explorer is actually drawn from. Counted, not
    # asserted: the prose used to claim "200" while confidence_sweep(steps=200)
    # stores steps + 1 = 201 points, and a page about being exact cannot round
    # its own description of itself.
    lengths = {len(sweep) for detectors in explorer.values()
               for sweep in detectors.values()}
    return {"datasets": datasets, "total_runs": len(runs),
            "winner": pick_detection_winner(datasets),
            "explorer": explorer,
            "sweep_points": lengths.pop() if len(lengths) == 1 else None,
            "datasets_without_sweep": [b["dataset"] for b in datasets if not b["has_sweep"]],
            "filter_regimes": regimes,
            "mixed_filters": len(distinct) > 1,
            "source": _relative(DETECTION_FILE)}


# ══════════════════════════════════════════════════════════════════
# Intent routing + prompt caching (the non-vision half of the log).
# ══════════════════════════════════════════════════════════════════

def intent_layers(run):
    """Per-layer routing latency, aggregated across the commands measured.

    The classifier is a cascade: cheap layers run first and only escalate when
    they cannot decide. Grouping the per-command results by LAYER is what shows
    the cost of each rung — the individual commands are noise around it.

    Inputs:
        run (dict): an intent benchmark payload with a "results" list of
            {label, command, tier, layer, avg_ms, p95_ms, p99_ms}.
    Returns:
        list[dict]: {"layer", "tier", "commands", "avg_ms", "p95_ms"}, cheapest
        first. Empty for a missing or malformed run.
    """
    grouped = {}
    for row in (run or {}).get("results", []):
        layer = row.get("layer")
        if not layer or row.get("avg_ms") is None:
            continue
        entry = grouped.setdefault(layer, {"layer": layer, "tier": row.get("tier"),
                                           "avg": [], "p95": []})
        entry["avg"].append(row["avg_ms"])
        entry["p95"].append(row.get("p95_ms") or row["avg_ms"])

    layers = [{
        "layer": e["layer"],
        "tier": e["tier"],
        "commands": len(e["avg"]),
        "avg_ms": sum(e["avg"]) / len(e["avg"]),
        "p95_ms": max(e["p95"]),
    } for e in grouped.values()]
    layers.sort(key=lambda e: e["avg_ms"])
    return layers


def caching_runs(runs):
    """Prompt-caching attempts, with whether the cache ACTUALLY engaged.

    The headline speedup is meaningless on its own: a 1.0x result means
    "caching did not help" only if caching actually ran. Reading
    cache_creation_input_tokens / cache_read_input_tokens tells you which of
    the two happened, so the page can distinguish a null result from a
    misconfigured experiment.

    Returns:
        list[dict]: oldest first, each {"timestamp", "input_tokens",
        "cache_created", "cache_read", "engaged", "speedup", "on_ms", "off_ms",
        "with_history"}.
    """
    rows = []
    for run in runs or []:
        on = run.get("caching_on") or {}
        off = run.get("caching_off") or {}
        usage = on.get("usage") or []
        created = sum(u.get("cache_creation_input_tokens", 0) or 0 for u in usage)
        read = sum(u.get("cache_read_input_tokens", 0) or 0 for u in usage)
        rows.append({
            "timestamp": run.get("timestamp"),
            "input_tokens": usage[0].get("input_tokens") if usage else None,
            "cache_created": created,
            "cache_read": read,
            "engaged": bool(created or read),
            "speedup": run.get("speedup_avg_x"),
            "on_ms": (on.get("latency_ms") or {}).get("avg"),
            "off_ms": (off.get("latency_ms") or {}).get("avg"),
            "with_history": bool(run.get("with_history")),
            "model": run.get("model"),
            "runs_per_mode": run.get("runs_per_mode"),
        })
    return rows


def claude_call_ms(caching):
    """Mean uncached Claude round-trip, in ms, across the caching runs.

    This is the number that makes the routing cascade worth building: it is
    what a tier-3 escalation actually costs, against the sub-millisecond to
    tens-of-milliseconds cost of deciding locally. Neither artifact states it —
    it only exists by reading the caching runs while looking at the routing
    ones.
    """
    values = [r["off_ms"] for r in caching if r.get("off_ms")]
    return (sum(values) / len(values)) if values else None


def load_eval_cases(path):
    """Read one eval .jsonl, skipping the '#' registry-rule header lines."""
    path = Path(path)
    if not path.is_file():
        return []
    cases = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return cases


def eval_corpus():
    """What the routing eval DEFINES as correct — not how well it scores.

    Deliberately reports composition only. tests/eval/results/ is empty: the
    corpus is a specification that nothing has been scored against yet, and
    the page says so rather than implying an accuracy figure exists.
    """
    routing = load_eval_cases(EVAL_DIR / "routing_eval.jsonl")
    conversation = load_eval_cases(EVAL_DIR / "conversation_eval.jsonl")

    def tally(cases, key):
        counts = {}
        for case in cases:
            counts[str(case.get(key))] = counts.get(str(case.get(key)), 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    # Look for actual result files, not just "the directory is non-empty" —
    # tests/eval/results/ holds a .gitkeep, and counting that as evidence would
    # make the page claim an accuracy figure exists when none does.
    results_dir = EVAL_DIR / "results"
    scored = bool(results_dir.is_dir() and
                  (list(results_dir.glob("*.json")) + list(results_dir.glob("*.jsonl"))))

    return {
        "routing_cases": len(routing),
        "conversation_cases": len(conversation),
        "categories": tally(routing, "category"),
        "difficulties": tally(routing, "difficulty"),
        "scenarios": sorted(p.stem for p in (EVAL_DIR / "scenarios").glob("*.jsonl"))
                     if (EVAL_DIR / "scenarios").is_dir() else [],
        "has_scored_results": scored,
        "sample": next((c for c in conversation if c.get("conversation_history")), None),
    }


def routing_tables():
    """Size of the hand-maintained lookup tables in the current classifier.

    The concrete form of "this does not scale": every one of these entries is a
    phrase someone typed out, and adding a capability means adding more of them
    in several places at once. Counted live rather than written into the page,
    so the number cannot quietly go stale while the argument stays.

    intent_classifier is imported lazily and failure is tolerated: this module
    is otherwise stdlib-only, and a documentation page must not be the reason
    the app cannot start.

    Returns:
        dict | None: {"tables": [(name, count)], "total": int}, or None if the
        classifier cannot be imported.
    """
    try:
        import intent_classifier
    except Exception:  # noqa: BLE001 - optional, never fatal
        return None

    names = ["ACTION_MAP", "CANONICAL_INTENTS", "DEVICE_ALIASES", "EVERYTHING_WORDS",
             "FILLER_PREFIXES", "GOODNIGHT_PHRASES", "GOODMORNING_PHRASES"]
    tables = []
    for name in names:
        value = getattr(intent_classifier, name, None)
        if value is not None:
            try:
                tables.append((name, len(value)))
            except TypeError:
                continue
    if not tables:
        return None
    return {"tables": sorted(tables, key=lambda t: -t[1]),
            "total": sum(count for _, count in tables)}


def routing_page_data():
    """Everything the intent-routing page renders.

    Reads the caching artifact for ONE value: claude_ms, the uncached
    round-trip that the whole local/cloud split is justified against. The
    caching experiment itself belongs to assistant_page_data() — it measures
    the model call, not the router. Keeping only the number this page argues
    from means a template change cannot silently resurrect the other one.
    """
    intent_runs = load_runs(INTENT_FILE)
    latest_intent = intent_runs[-1] if intent_runs else None
    layers = intent_layers(latest_intent)

    return {
        "intent_run": latest_intent,
        "intent_runs": len(intent_runs),
        "layers": layers,
        "cheapest": layers[0] if layers else None,
        "dearest": layers[-1] if layers else None,
        "claude_ms": claude_call_ms(caching_runs(load_runs(CACHING_FILE))),
        "corpus": eval_corpus(),
        "tables": routing_tables(),
        "intent_source": _relative(INTENT_FILE),
    }


# ══════════════════════════════════════════════════════════════════
# Enrollment curation (tools/collect_faces.py).
# ══════════════════════════════════════════════════════════════════

def curation_page_data():
    """Everything the enrollment curation page renders.

    Reads the committed, anonymised summary rather than the private manifest.
    The shaping that used to live here (anonymising folder names, grouping
    clusters, computing the folder-vote collisions) now runs in
    tools/collect_faces.py at write time, because that is the only place the
    real names exist and the only place they should ever be handled.

    Returns {"curation": None, ...} when the artifact is missing. That state is
    rendered as a VISIBLE marker, not a blank section, and
    freeze_engineering.preflight() fails --strict over it: a page that
    silently says nothing is the failure this whole change exists to fix.
    """
    summary = load_document(ENROLLMENT_SUMMARY)
    if not summary or not summary.get("clusters_detail"):
        return {"curation": None, "curation_source": _relative(ENROLLMENT_SUMMARY)}

    clusters = summary["clusters_detail"]
    # JSON has no tuples, so the per-cluster folder breakdown arrives as
    # [label, count] pairs; the template unpacks them as a pair either way.
    for cluster in clusters:
        cluster["folders"] = [tuple(entry) for entry in cluster["folders"]]

    at_risk_ids = set(summary.get("at_risk") or [])
    # rejects is a first-class key in the artifact (it is one of the things the
    # summary exists to report), but the page reads it as counts.rejects, which
    # is where it sits in the private manifest. Re-nest rather than reshape the
    # template: the artifact's own shape should read well on its own.
    counts = dict(summary.get("counts") or {})
    counts["rejects"] = summary.get("rejects") or {}

    return {
        "curation": {
            "generated_at": summary.get("generated_at"),
            "settings": summary.get("settings") or {},
            "counts": counts,
            "clusters": clusters,
            "noise": summary.get("noise_crops"),
            "clustered_crops": summary.get("clustered_crops"),
            "people": summary.get("clusters"),
            "source_folders": summary.get("source_folders"),
            "smallest": summary.get("smallest_cluster"),
            "largest": summary.get("largest_cluster"),
            "min_cluster_size": (summary.get("settings") or {}).get("min_cluster_size"),
            "headroom": summary.get("headroom"),
            "at_risk": [c for c in clusters if c["id"] in at_risk_ids],
            "collisions": summary.get("collisions") or [],
            "mislabelled_if_voted": summary.get("mislabelled_if_voted"),
            "folders": summary.get("folders") or [],
            "blur": summary.get("blur_distribution"),
        },
        "curation_source": _relative(ENROLLMENT_SUMMARY),
    }


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
    # The roadmap section names the operating points the presence service will
    # consume. They are read rather than typed for the same reason as every
    # other number on this page: a plan that cites a stale threshold is worse
    # than one that cites none.
    recognition = load_runs(RECOGNITION_FILE)
    return {
        "sampling": sampling_comparison(runs),
        "box_rates": candidate_box_rates(busiest),
        "box_rate_run": busiest,
        "recognition_winner": pick_winner(
            recognition_matrix(recognition[-1] if recognition else None)),
    }


def assistant_page_data():
    """Everything the Phase 1 assistant page renders.

    THE CACHING RUNS LIVE HERE NOW. They were on the routing page, which was
    the wrong home for them: the experiment measures a property of the Claude
    call, not of whatever decided to make it — the page itself said so in
    prose. Routing keeps claude_ms, which is the number that justifies the
    cascade; the experiment moves to the page about orchestrating Claude.

    Deliberately returns no numbers for browser speech, the filler phrases or
    the interface. Those were judgment calls and were never measured, and
    inventing a shape for them here is how a page starts implying evidence it
    does not have. The template states them as unmeasured instead.
    """
    caching = caching_runs(load_runs(CACHING_FILE))
    return {
        "caching": caching,
        "caching_engaged": [r for r in caching if r["engaged"]],
        "claude_ms": claude_call_ms(caching),
        "caching_source": _relative(CACHING_FILE),
    }


def direction_page_data():
    """The measured anchors the direction page is allowed to stand on.

    That page is about where the project is going, so almost none of it can be
    backed by an artifact — which is exactly why the few points that CAN be are
    read rather than typed. The ladder diagram marks the boundary between the
    measured stages and the intended ones, and the boundary has to move on its
    own when a benchmark is rerun. If it were hardcoded, the one claim the page
    genuinely must get right — how far along it actually is — would be the one
    claim nobody would notice going stale.

    Returns:
        dict with the recognition winner, the datasets detection has been
        measured on, and the total images behind those runs.
    """
    detection = load_runs(DETECTION_FILE)
    full = latest_full_run_per_dataset(detection)
    recognition = load_runs(RECOGNITION_FILE)
    latest_recognition = recognition[-1] if recognition else None
    winner = pick_winner(recognition_matrix(latest_recognition))

    return {
        "recognition_winner": winner,
        "recognition_run": latest_recognition,
        "recognition_counts": _winner_counts(latest_recognition, winner),
        "primary_far": PRIMARY_FAR,
        "detection_datasets": sorted(full),
        "detection_images": sum(run.get("num_images", 0) for run in full.values()),
    }


def _winner_counts(run, winner):
    """The score populations the winning cell was actually judged against.

    The caption has to name the non-mate set honestly, and "500 strangers" is
    not it: those 500 LFW faces are scored against every gallery, and the
    household's own members are scored against each other's galleries too. The
    real denominator is impostors AND strangers combined, because a deployment
    threshold has to reject both — a stranger at the door and the wrong
    housemate are different failure modes and the benchmark was built to catch
    each. Reading the counts keeps that claim honest if the run changes shape.

    Inputs:
        run (dict | None): a recognition run payload.
        winner (dict | None): the winning cell, from pick_winner.
    Returns:
        dict | None: {"genuine", "impostor", "stranger", "nonmate"} comparison
        counts, or None if the run does not record them.
    """
    if not run or not winner:
        return None
    for result in run.get("results", []):
        if result.get("encoder") != winner.get("encoder"):
            continue
        metrics = (result.get("aggregations") or {}).get(winner.get("aggregation"))
        counts = (metrics or {}).get("counts")
        if not counts:
            return None
        impostor = int(counts.get("impostor", 0))
        stranger = int(counts.get("stranger", 0))
        return {
            "genuine": int(counts.get("genuine", 0)),
            "impostor": impostor,
            "stranger": stranger,
            "nonmate": impostor + stranger,
        }
    return None


def recognition_curve_figures(matrix):
    """Which ROC/PR figures are on disk AND backed by the run being shown.

    Both conditions matter. A figure file left over from an older run would
    otherwise be served next to newer numbers it does not describe, so the
    figures are only offered when the displayed run actually carries curve
    data — the same condition under which the plotter could have drawn them.

    Inputs:
        matrix (dict): as returned by recognition_matrix.
    Returns:
        dict {"roc": bool, "pr": bool}: whether each figure can be shown.
    """
    has_curves = any(cell.get("roc_auc") is not None
                     for cell in (matrix.get("cells") or {}).values())
    if not has_curves:
        return {"roc": False, "pr": False}
    return {
        "roc": (RESULTS_DIR / "recognition_roc.svg").exists(),
        "pr": (RESULTS_DIR / "recognition_pr.svg").exists(),
    }


def recognition_page_data():
    """Everything the recognition page renders."""
    runs = load_runs(RECOGNITION_FILE)
    latest = runs[-1] if runs else None
    matrix = recognition_matrix(latest)
    explorer = explorer_payload(latest)
    return {
        "run": latest,
        "matrix": matrix,
        "winner": pick_winner(matrix),
        "explorer": explorer,
        "explorer_exact": explorer_is_exact(explorer),
        "curves_available": explorer_has_curves(explorer),
        "figures": recognition_curve_figures(matrix),
        "total_runs": len(runs),
        "source": _relative(RECOGNITION_FILE),
    }


def _relative(path):
    try:
        return str(Path(path).relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)
