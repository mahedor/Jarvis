"""
JARVIS Intent Classifier — Benchmark
======================================
Runs classify() 1000x per command, prints a per-command table and per-tier
aggregates, then saves results to logs/benchmarks.json.

Usage:
  python tools/benchmark.py              # run and save
  python tools/benchmark.py --compare    # run, save, diff against previous run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup — allow importing from demo/ ─────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "demo"))

from intent_classifier import (
    SPACY_AVAILABLE, EMBEDDINGS_AVAILABLE,
    TIER_DIRECT, TIER_LOCAL, TIER_CLAUDE,
    classify,
)

LOGS_DIR = REPO_ROOT / "logs"
BENCH_FILE = LOGS_DIR / "benchmarks.json"

# ── Device state fixture ─────────────────────────────────────────
DEVICE_STATES = {
    "light.bedroom":        {"state": "off",    "friendly_name": "Bedroom main light", "brightness": 0},
    "light.bedroom_lamp":   {"state": "on",     "friendly_name": "Bedroom lamp",       "brightness": 128},
    "switch.bedroom_fan":   {"state": "off",    "friendly_name": "Bedroom fan"},
    "cover.bedroom_blinds": {"state": "closed", "friendly_name": "Bedroom blinds"},
}

# ── Test commands — ~20 cases covering every tier and layer ─────
#   (label, command)
COMMANDS = [
    # Tier 1 — keyword
    ("T1 keyword: time",              "What time is it?"),
    ("T1 keyword: date",              "What's the date today?"),
    # Tier 1 — embedding
    ("T1 embed: hour",                "What hour is it?"),
    ("T1 embed: current time",        "Tell me the current time"),

    # Tier 2 — keyword (clean)
    ("T2 kw: turn on lights",         "Turn on the bedroom lights"),
    ("T2 kw: turn off lamp",          "Turn off the lamp"),
    ("T2 kw: dim lamp",               "Dim the lamp to 50%"),
    ("T2 kw: open blinds",            "Open the blinds"),
    ("T2 kw: status",                 "What's the status of everything?"),

    # Tier 2 — keyword (stacked prefixes)
    ("T2 kw+prefix: hey+please",      "Hey Jarvis, please turn on the lights"),
    ("T2 kw+prefix: deep stack",      "Hey Jarvis, can you please go ahead and turn on the lights"),

    # Tier 2 — spaCy
    ("T2 spacy: passive",             "The lamp should be switched off"),
    ("T2 spacy: need raised",         "I need the blinds raised"),

    # Tier 2 — embedding
    ("T2 embed: cut the power",       "Cut the power to the lights"),
    ("T2 embed: let light in",        "Let some light in through the blinds"),

    # Tier 2 — goodnight (time-aware, forced 9pm)
    ("T2 goodnight",                  "Goodnight"),

    # Tier 2 — negation → Claude
    ("T2 negation->T3",               "Don't turn on the lights"),

    # Tier 3
    ("T3: joke",                      "Tell me a joke"),
    ("T3: dinner",                    "What should I eat for dinner?"),
    ("T3: unknown device",            "Turn on the TV"),
]

RUNS = 1000
GOODNIGHT_NOW = datetime(2024, 1, 1, 21, 0)  # 9pm — ensures goodnight fires


# ── Benchmark runner ─────────────────────────────────────────────

def run_benchmark():
    results = []
    for label, cmd in COMMANDS:
        _now = GOODNIGHT_NOW if "goodnight" in label else None
        samples = []
        tier = None
        layer = None

        for _ in range(RUNS):
            t0 = time.perf_counter()
            r = classify(cmd, DEVICE_STATES, _now=_now)
            samples.append((time.perf_counter() - t0) * 1000)  # ms
            tier = r["tier"]
            layer = r.get("matched_layer", "?")

        samples.sort()
        avg = sum(samples) / len(samples)
        p95 = samples[int(len(samples) * 0.95)]
        p99 = samples[int(len(samples) * 0.99)]

        results.append({
            "label":   label,
            "command": cmd,
            "tier":    tier,
            "layer":   layer,
            "avg_ms":  round(avg, 3),
            "p95_ms":  round(p95, 3),
            "p99_ms":  round(p99, 3),
        })

    return results


# ── Formatting helpers ───────────────────────────────────────────

TIER_NAMES = {TIER_DIRECT: "T1-Direct", TIER_LOCAL: "T2-Local", TIER_CLAUDE: "T3-Claude"}

def print_table(results):
    col_label = max(len(r["label"]) for r in results)
    col_layer = max(len(r["layer"]) for r in results)
    w = max(col_label, 30)
    lw = max(col_layer, 14)

    header = f"  {'Label':<{w}}  {'Tier':<10}  {'Layer':<{lw}}  {'Avg ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}"
    print("=" * len(header))
    print("  JARVIS Classifier Benchmark")
    print(f"  {RUNS} runs/command  |  spaCy: {'on' if SPACY_AVAILABLE else 'off'}  |  Embeddings: {'on' if EMBEDDINGS_AVAILABLE else 'off'}")
    print("=" * len(header))
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    last_tier = None
    for r in results:
        tier_str = TIER_NAMES.get(r["tier"], str(r["tier"]))
        if last_tier is not None and r["tier"] != last_tier:
            print()
        last_tier = r["tier"]
        print(f"  {r['label']:<{w}}  {tier_str:<10}  {r['layer']:<{lw}}  {r['avg_ms']:>8.3f}  {r['p95_ms']:>8.3f}  {r['p99_ms']:>8.3f}")

    print()
    print_aggregates(results, w)


def print_aggregates(results, w=30):
    from collections import defaultdict
    by_tier = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)

    print("  Per-tier aggregates")
    print("  " + "-" * 60)
    header = f"  {'Tier':<12}  {'Cmds':>5}  {'Avg ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}"
    print(header)
    print("  " + "-" * 60)

    for tier in sorted(by_tier):
        rows = by_tier[tier]
        avgs = [r["avg_ms"] for r in rows]
        p95s = [r["p95_ms"] for r in rows]
        p99s = [r["p99_ms"] for r in rows]
        name = TIER_NAMES.get(tier, str(tier))
        print(f"  {name:<12}  {len(rows):>5}  {sum(avgs)/len(avgs):>8.3f}  {max(p95s):>8.3f}  {max(p99s):>8.3f}")

    print()


# ── Compare against previous run ────────────────────────────────

def load_previous_run(current_ts):
    """Return the most recent saved run that isn't the current one, or None."""
    if not BENCH_FILE.exists():
        return None
    with open(BENCH_FILE) as f:
        data = json.load(f)
    runs = [e for e in data if e["timestamp"] != current_ts]
    if not runs:
        return None
    return max(runs, key=lambda e: e["timestamp"])


def print_comparison(prev, curr_results):
    prev_by_label = {r["label"]: r for r in prev["results"]}
    curr_by_label = {r["label"]: r for r in curr_results}

    all_labels = list(dict.fromkeys(
        [r["label"] for r in curr_results] + list(prev_by_label)
    ))

    w = max(len(l) for l in all_labels)
    print("=" * 70)
    print(f"  Comparison vs run at {prev['timestamp']}")
    print("=" * 70)
    print()
    header = f"  {'Label':<{w}}  {'Old avg':>8}  {'New avg':>8}  {'Delta':>8}  {'Change':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label in all_labels:
        if label not in prev_by_label:
            print(f"  {label:<{w}}  {'(new)':>8}")
            continue
        if label not in curr_by_label:
            print(f"  {label:<{w}}  {'(removed)':>8}")
            continue
        old = prev_by_label[label]["avg_ms"]
        new = curr_by_label[label]["avg_ms"]
        delta = new - old
        pct = (delta / old * 100) if old > 0 else 0
        arrow = "+" if delta > 0.1 else ("-" if delta < -0.1 else "~")
        print(f"  {label:<{w}}  {old:>8.3f}  {new:>8.3f}  {delta:>+8.3f}  {arrow} {pct:+.1f}%")

    print()


# ── Persistence ──────────────────────────────────────────────────

def save_results(ts, results):
    LOGS_DIR.mkdir(exist_ok=True)
    data = []
    if BENCH_FILE.exists():
        with open(BENCH_FILE) as f:
            data = json.load(f)
    data.append({
        "timestamp": ts,
        "runs_per_command": RUNS,
        "spacy": SPACY_AVAILABLE,
        "embeddings": EMBEDDINGS_AVAILABLE,
        "results": results,
    })
    with open(BENCH_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark the JARVIS intent classifier")
    parser.add_argument("--compare", action="store_true", help="Diff against previous run")
    args = parser.parse_args()

    print()
    print("  Running benchmark...")
    print(f"  {len(COMMANDS)} commands x {RUNS} runs each\n")

    t_start = time.perf_counter()
    results = run_benchmark()
    elapsed = time.perf_counter() - t_start

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print_table(results)
    print(f"  Total wall time: {elapsed:.2f}s\n")

    if args.compare:
        prev = load_previous_run(ts)
        if prev:
            print_comparison(prev, results)
        else:
            print("  --compare: no previous run found to compare against.\n")

    save_results(ts, results)
    print(f"  Saved -> {BENCH_FILE.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
