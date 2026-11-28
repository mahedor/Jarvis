#!/usr/bin/env python3
"""Flight vs laptop bench (Michael 11-27): how do time_trackers.py's bench timings translate to a real flight, so a
later bench run can predict flight speed?

  cd /mnt/c/Users/mahed/OneDrive/Documents/Jarvis/jarvis-lean/followme
  python flight_vs_bench.py                                   # every flight under ~/follow_logs with detector.csv
  python flight_vs_bench.py --since 20261128 --labelled-only  # only tomorrow's comparison flights
  python flight_vs_bench.py ~/follow_logs/<run1> ~/follow_logs/<run2> --bench ~/follow_logs/tracker_speed_<date>.json

Inputs
  * Flights: run folders written by live_follow.py since 11-27 (detector.csv = one row per detector cycle; drone.json
    `tracker.bench_setup` = the bench row of the tracker that flew; follow.csv for the loop rate). Older runs have no
    detector.csv and are listed as skipped, never estimated. Fake-drone runs are skipped (--include-fake keeps them).
  * Bench: --bench, else the newest ~/follow_logs/tracker_speed_*.json (time_trackers.py), else
    results/followme_tracker_speed.json. Its rows are matched by name (`setup`) to each flight's bench_setup, and the
    flight's tracker settings must contain the bench row's settings, else the flight is skipped with the difference.

Per config (flights with the same bench row are pooled), over the cycles after the first WARMUP of each run (as the
bench) and, when follow.csv shows the drone airborne at all (flying_state), only cycles used while airborne:
  bench ms        the bench row's mean tracker.update ms (laptop, cached boxes)
  track ms        flight tracker.update ms (detector.csv track_ms): the same call, in flight
  cycle ms        flight detector period (result to result); det/s = 1000 / mean cycle ms
  busy ms         flight detector-thread work per cycle (model.track + lock-on re-ID)
  box age ms      frame arrival -> boxes published
  loop FPS        follow.csv rows per second (one row per new video frame the loop handled)
With >= 2 configs, a least-squares line y = a + b * bench_ms for track ms, cycle ms and loop FPS, with residuals.
Two points fit exactly (nothing checks the line); three points is a rough fit (one degree of freedom). Then the line
is applied to every bench row, marking extrapolations. Stdlib only.
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import flight_log as fl                 # stdlib-only readers shared with the flight log (detector.csv, follow.csv)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LOGS = Path.home() / "follow_logs"
BENCH_FALLBACK = REPO / "results" / "followme_tracker_speed.json"
WARMUP = 20                             # time_trackers.WARMUP: cycles run but not counted (test checks they agree)
METRICS = (("track_ms", "track ms"), ("cycle_ms", "cycle ms"), ("loop_fps", "loop FPS"))


# ---------------------------------------------------------------- bench

def find_bench(explicit=None, logs=LOGS, fallback=BENCH_FALLBACK):
    """--bench, else the newest tracker_speed_*.json in the logs folder (by name: it holds the date), else results/."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            sys.exit(f"STOP: bench file not found: {p}")
        return p
    cands = sorted(Path(logs).expanduser().glob("tracker_speed_*.json")) if Path(logs).expanduser().is_dir() else []
    if cands:
        return cands[-1]
    if Path(fallback).is_file():
        return Path(fallback)
    sys.exit(f"STOP: no bench JSON: no tracker_speed_*.json in {logs} and no {fallback}. Run time_trackers.py first.")


def load_bench(path):
    rec = json.loads(Path(path).read_text())
    rows = {r["setup"]: r for r in rec.get("rows", []) if r.get("setup")}
    return rec, rows


# ---------------------------------------------------------------- one flight

def settings_problem(flight_settings, bench_row):
    """The flight's tracker settings must contain the bench row's tuned values and its re-ID / camera-motion
    switches (the encoder's path may differ). -> None, or what differs."""
    if not isinstance(flight_settings, dict):
        return "the flight recorded no tracker settings"
    want = dict(bench_row.get("settings_changed") or {})
    for k in ("with_reid", "gmc_method"):
        if k in (bench_row.get("overrides") or {}):
            want[k] = bench_row["overrides"][k]
    want.pop("model", None)
    diff = [f"{k}: flight {flight_settings.get(k, '<missing>')!r}, bench {v!r}" for k, v in sorted(want.items())
            if flight_settings.get(k, "<missing>") != v]
    return ("tracker settings differ from bench row " + repr(bench_row["setup"]) + ": " + "; ".join(diff)) if diff else None


def _airborne(row):
    return str(row.get("flying_state") or "").lower() in fl.AIRBORNE


def run_data(run, warmup=WARMUP):
    """One run folder -> its cycles (after warm-up; airborne only when follow.csv knows) and loop-rate pieces."""
    cycles = fl.read_cycles(run)
    rows, _ = fl.read_rows(run)
    cycles = cycles[warmup:]
    has_fs = any(_airborne(r) for r in rows)          # never airborne (fake drone, ground test): use every cycle
    air_rows = [r for r in rows if _airborne(r)] if has_fs else rows
    if has_fs:
        used = {fl._num(r.get("det_n")) for r in air_rows} - {None}
        cycles = [c for c in cycles if c.get("det_n") in used]
    # loop rate: intervals between consecutive rows that are both counted (airborne), over their summed time
    n_int, secs = 0, 0.0
    for a, b in zip(rows, rows[1:]):
        ta, tb = fl._num(a.get("t_s")), fl._num(b.get("t_s"))
        if ta is None or tb is None or tb <= ta or (has_fs and not (_airborne(a) and _airborne(b))):
            continue
        n_int, secs = n_int + 1, secs + (tb - ta)
    return dict(cycles=cycles, loop_intervals=n_int, loop_s=secs, airborne_only=has_fs)


def flight(run, bench_rows, include_fake=False, labelled_only=False):
    """-> (record, None) or (None, why skipped)."""
    info, prob = fl.read_info(run)
    if prob:
        return None, prob
    if info.get("drone_nickname") == "fake" and not include_fake:
        return None, "fake-drone run"
    if labelled_only and not info.get("label"):
        return None, "not a labelled comparison flight"
    if not (run / fl.DETECTOR_CSV).exists():
        return None, "no detector.csv (flown before per-cycle timing was logged, 11-27)"
    trk = info.get("tracker") if isinstance(info.get("tracker"), dict) else {}
    setup = trk.get("bench_setup")
    if not setup:
        return None, f"tracker {trk.get('file')!r} has no bench row (not a followme/ yaml the bench times)"
    if setup not in bench_rows:
        return None, f"bench JSON has no row {setup!r}"
    if bench_rows[setup].get("error") or bench_rows[setup].get("mean_ms") is None:
        return None, f"bench row {setup!r} has no timing ({bench_rows[setup].get('error')})"
    sp = settings_problem(trk.get("settings"), bench_rows[setup])
    if sp:
        return None, sp
    d = run_data(run)
    if not d["cycles"]:
        return None, "no detector cycles left after the warm-up / airborne filter"
    return dict(run=run.name, setup=setup, label=info.get("label"), tracker=trk.get("file"),
                reid_device=(info.get("tracker_runtime") or {}).get("reid_device"), **d), None


# ---------------------------------------------------------------- stats + fit

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def config_stats(flights, bench_row):
    """Pool the flights of one bench row."""
    cyc = [c for f in flights for c in f["cycles"]]
    col = {k: [c.get(k) for c in cyc] for k in ("track_ms", "cycle_ms", "busy_ms", "box_age_ms", "det_ms", "infer_ms",
                                                 "n_boxes")}
    secs = sum(f["loop_s"] for f in flights)
    cyc_mean = mean(col["cycle_ms"])
    return dict(setup=bench_row["setup"], bench_ms=bench_row["mean_ms"], bench_p95_ms=bench_row.get("p95_ms"),
                bench_tracks=bench_row.get("mean_tracks"), runs=[f["run"] for f in flights],
                labels=[f["label"] for f in flights], n_cycles=len(cyc),
                track_ms=mean(col["track_ms"]), track_median_ms=fl.median(col["track_ms"]), track_p95_ms=pct(col["track_ms"], 0.95),
                cycle_ms=cyc_mean, cycle_median_ms=fl.median(col["cycle_ms"]),
                det_per_s=1000 / cyc_mean if cyc_mean else None,
                busy_ms=mean(col["busy_ms"]), det_ms_median=fl.median(col["det_ms"]), infer_ms_median=fl.median(col["infer_ms"]),
                box_age_median_ms=fl.median(col["box_age_ms"]), boxes_per_cycle=mean(col["n_boxes"]),
                loop_fps=(sum(f["loop_intervals"] for f in flights) / secs) if secs > 0 else None,
                airborne_only=all(f["airborne_only"] for f in flights),
                reid_devices=sorted({str(f["reid_device"]) for f in flights if f["reid_device"]}))


def linfit(xs, ys):
    """Least squares y = a + b x. -> dict(a, b, residuals, r2 or None, n, note) or None if < 2 usable points."""
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pts)
    if n < 2:
        return None
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    res = [p[1] - (a + b * p[0]) for p in pts]
    syy = sum((p[1] - my) ** 2 for p in pts)
    r2 = (1 - sum(r * r for r in res) / syy) if n >= 3 and syy > 0 else None
    note = ("2 points: the line goes exactly through both, nothing checks it" if n == 2 else
            "3 points: rough fit (1 degree of freedom); the residuals are the only check" if n == 3 else
            f"{n} points ({n - 2} degrees of freedom)")
    return dict(a=a, b=b, residuals=res, r2=r2, n=n, x_range=[min(p[0] for p in pts), max(p[0] for p in pts)], note=note)


def compare(flights, bench_rows):
    by = {}
    for f in flights:
        by.setdefault(f["setup"], []).append(f)
    configs = sorted((config_stats(fs, bench_rows[s]) for s, fs in by.items()), key=lambda c: c["bench_ms"])
    xs = [c["bench_ms"] for c in configs]
    fits = {k: linfit(xs, [c[k] for c in configs]) for k, _ in METRICS}
    preds = []
    if fits["cycle_ms"] or fits["track_ms"] or fits["loop_fps"]:
        for r in bench_rows.values():
            if r.get("mean_ms") is None:
                continue
            p = dict(setup=r["setup"], bench_ms=r["mean_ms"])
            for k, _ in METRICS:
                ft = fits[k]
                p[k] = None if ft is None else ft["a"] + ft["b"] * r["mean_ms"]
            p["det_per_s"] = 1000 / p["cycle_ms"] if p.get("cycle_ms") and p["cycle_ms"] > 0 else None
            rng = next((fits[k]["x_range"] for k, _ in METRICS if fits[k]), None)
            p["extrapolated"] = bool(rng) and not (rng[0] <= r["mean_ms"] <= rng[1])
            preds.append(p)
        preds.sort(key=lambda p: p["bench_ms"])
    return configs, fits, preds


# ---------------------------------------------------------------- report

def _f(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def report(configs, fits, preds, skipped, bench_path, bench_rec):
    m = bench_rec.get("machine") or {}
    det = bench_rec.get("detector") or {}
    out = [f"bench: {bench_path} (made {bench_rec.get('made')}, CPU {m.get('cpu_model', '?')}; detector alone "
           f"{_f(det.get('mean_ms'), 2)} ms mean; video {((bench_rec.get('video') or {}).get('files') or ['?'])[0]})"]
    if m and not m.get("michaels_laptop"):
        out.append("  NOTE: this bench was not run on Michael's laptop (7735HS): its numbers do not describe his flights")
    out.append(f"flights used: {sum(len(c['runs']) for c in configs)}; skipped: {len(skipped)}")
    for run, why in skipped:
        out.append(f"  skipped {run}: {why}")
    if not configs:
        out.append("\nno flight could be matched to a bench row: nothing to compare.")
        return "\n".join(out)
    head = (f"{'config (bench row)':26s} {'bench ms':>8s} {'track ms':>8s} {'trk p95':>7s} {'cycle ms':>8s} {'det/s':>6s} "
            f"{'busy ms':>7s} {'infer':>6s} {'box age':>7s} {'loop FPS':>8s} {'boxes':>5s} {'cycles':>6s}  runs")
    out += ["", head, "-" * len(head)]
    for c in configs:
        out.append(f"{c['setup'][:26]:26s} {_f(c['bench_ms'], 2):>8s} {_f(c['track_ms'], 2):>8s} {_f(c['track_p95_ms'], 1):>7s} "
                   f"{_f(c['cycle_ms']):>8s} {_f(c['det_per_s']):>6s} {_f(c['busy_ms']):>7s} {_f(c['infer_ms_median']):>6s} "
                   f"{_f(c['box_age_median_ms']):>7s} {_f(c['loop_fps']):>8s} {_f(c['boxes_per_cycle']):>5s} "
                   f"{c['n_cycles']:>6d}  {', '.join(c['runs'])}")
    out.append("means over cycles after the first %d of each run%s; track = tracker.update (same call as the bench); "
               "cycle = detector result to result; infer / box age = medians; boxes = boxes per cycle (bench: "
               "the bench video's own count)" % (WARMUP, ", airborne only" if all(c["airborne_only"] for c in configs) else ""))
    if any(c["reid_devices"] and c["reid_devices"] != ["cpu"] for c in configs):
        out.append("  NOTE: a tracker re-ID encoder ran on " + ", ".join(d for c in configs for d in c["reid_devices"])
                   + " in flight; the bench forces the CPU")
    if len(configs) < 2:
        out.append("\none config only: a relation needs at least two. Fly another tracker (see README, Flight vs bench).")
        return "\n".join(out)
    out.append("")
    for k, name in METRICS:
        ft = fits[k]
        if ft is None:
            out.append(f"{name}: no fit (fewer than 2 configs with a value, or all bench ms equal)")
            continue
        res = ", ".join(f"{c['setup'][:18]} {r:+.2f}" for c, r in zip([c for c in configs if c[k] is not None], ft["residuals"]))
        out.append(f"{name:9s} = {ft['a']:.2f} {'+' if ft['b'] >= 0 else '-'} {abs(ft['b']):.3f} x bench_ms   "
                   f"[{ft['note']}]" + (f"  R^2 {ft['r2']:.3f}" if ft["r2"] is not None else ""))
        out.append(f"          residuals (flight - line): {res}")
    if preds:
        out += ["", "applied to every bench row (* = outside the bench ms range the line was fitted on: a guess):",
                f"  {'bench row':26s} {'bench ms':>8s} {'track ms':>8s} {'cycle ms':>8s} {'det/s':>6s} {'loop FPS':>8s}"]
        for p in preds:
            out.append(f"{'*' if p['extrapolated'] else ' '} {p['setup'][:26]:26s} {_f(p['bench_ms'], 2):>8s} "
                       f"{_f(p['track_ms'], 2):>8s} {_f(p['cycle_ms']):>8s} {_f(p['det_per_s']):>6s} {_f(p['loop_fps']):>8s}")
        n = min(ft["n"] for ft in fits.values() if ft)
        out.append(f"  from {n} configs: " + ("treat as rough." if n <= 3 else "check the residuals before trusting it."))
    return "\n".join(out)


def run_folders(args):
    if args.runs:
        return [Path(r).expanduser() for r in args.runs]
    root = Path(args.root).expanduser()
    if not root.is_dir():
        sys.exit(f"STOP: logs root not found: {root}")
    runs = sorted(p for p in root.iterdir() if fl.is_run_folder(p))
    return [p for p in runs if not args.since or p.name[:8] >= args.since]


def build_parser():
    ap = argparse.ArgumentParser(description="Flight timings vs the laptop tracker bench (time_trackers.py).")
    ap.add_argument("runs", nargs="*", help="run folders (default: every run folder under --root)")
    ap.add_argument("--root", default=str(LOGS), help="logs root (default ~/follow_logs)")
    ap.add_argument("--bench", default=None, help="bench JSON (default: newest ~/follow_logs/tracker_speed_*.json, "
                                                  "else results/followme_tracker_speed.json)")
    ap.add_argument("--since", default="", help="only run folders named YYYYMMDD_... on or after this date")
    ap.add_argument("--labelled-only", action="store_true", help="only comparison flights (live_follow --label)")
    ap.add_argument("--include-fake", action="store_true", help="also use fake-drone runs (tests)")
    ap.add_argument("--save", default=None, help="also write the numbers as JSON here")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    bench_path = find_bench(args.bench, Path(args.root))
    bench_rec, bench_rows = load_bench(bench_path)
    flights, skipped = [], []
    for run in run_folders(args):
        rec, why = flight(run, bench_rows, args.include_fake, args.labelled_only)
        (flights.append(rec) if rec else skipped.append((run.name, why)))
    configs, fits, preds = compare(flights, bench_rows)
    print(report(configs, fits, preds, skipped, bench_path, bench_rec))
    result = dict(what="flight timings vs the laptop tracker bench (flight_vs_bench.py)",
                  made=datetime.now().isoformat(timespec="seconds"), bench=str(bench_path), warmup=WARMUP,
                  configs=configs, fits=fits, predictions=preds, skipped=[dict(run=r, why=w) for r, w in skipped])
    if args.save:
        Path(args.save).expanduser().write_text(json.dumps(result, indent=1) + "\n")
        print(f"saved {args.save}")
    return result


if __name__ == "__main__":
    main()
