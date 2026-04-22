"""
JARVIS — Claude API Latency Benchmark
======================================
Measures end-to-end API latency with prompt caching ON vs OFF.
Sends the same Tier 3 command N times in each mode, prints a
comparison table (avg/min/max/p95), and saves to logs/benchmarks_claude.json.

Uses the exact same system prompt as demo/jarvis_web.py.

Usage:
  python tools/benchmark_claude.py
  python tools/benchmark_claude.py --runs 20
  python tools/benchmark_claude.py --command "your Tier 3 message"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "demo"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv()

try:
    from anthropic import Anthropic
except ImportError:
    print("\n❌ Missing 'anthropic' package.  Run: pip install anthropic\n")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────
LOGS_DIR  = REPO_ROOT / "logs"
BENCH_FILE = LOGS_DIR / "benchmarks_claude.json"

MODEL         = "claude-sonnet-4-20250514"   # matches jarvis_web.py
MAX_TOKENS    = 300
DEFAULT_RUNS  = 10
DEFAULT_COMMAND = (
    "What should I have for dinner tonight? Something quick and healthy."
)

# ── Device states — mirrors jarvis_web.py ────────────────────────
DEVICE_STATES = {
    "light.bedroom":        {"state": "off",    "friendly_name": "Bedroom main light", "brightness": 0},
    "light.bedroom_lamp":   {"state": "off",    "friendly_name": "Bedroom lamp",       "brightness": 0},
    "switch.bedroom_fan":   {"state": "off",    "friendly_name": "Bedroom fan"},
    "cover.bedroom_blinds": {"state": "closed", "friendly_name": "Bedroom blinds"},
}

# ── System prompt — verbatim from jarvis_web.py ──────────────────
SYSTEM_STATIC = """You are Jarvis, a personal AI assistant for a smart home system.
Your personality is helpful, witty, and concise — like the Jarvis from Iron Man but
more casual and friendly. You're talking to Michael in his bedroom.

RULES:
- Keep responses SHORT (1-3 sentences max) since they'll eventually be spoken via TTS.
- Be natural and conversational, not robotic.
- You can be playful — you're Jarvis after all.

DEVICE CONTROL:
When the user asks to control a device, respond with a natural confirmation AND include
an action block on its own line in this exact format:
[ACTION: {"service": "light.turn_on", "entity_id": "light.bedroom", "data": {}}]

Available services:
- light.turn_on / light.turn_off (data: {"brightness": 0-255})
- switch.turn_on / switch.turn_off
- cover.open_cover / cover.close_cover

If the user says something conversational (not device-related), just respond naturally.
If you're unsure which device, ask for clarification."""


def _device_status_text():
    lines = []
    for entity_id, info in DEVICE_STATES.items():
        name  = info["friendly_name"]
        state = info["state"].upper()
        extra = ""
        if "brightness" in info and info["state"] == "on":
            pct   = round(info["brightness"] / 255 * 100)
            extra = f" (brightness: {pct}%)"
        lines.append(f"  - {name} ({entity_id}): {state}{extra}")
    return "\n".join(lines)


def _device_block_text():
    return (
        f"CURRENT DEVICE STATUS:\n{_device_status_text()}\n\n"
        "IMPORTANT: When asked about device status, use the CURRENT STATUS above "
        "to give accurate answers. If a light is already on and the user asks to "
        "turn it on, let them know it's already on. If they ask 'what’s on?' or "
        "'status?', report the current state of all devices."
    )


def build_system_cached():
    """System prompt as a list of content blocks — static part marked cacheable.
    Mirrors build_system_prompt() in jarvis_web.py."""
    return [
        {
            "type": "text",
            "text": SYSTEM_STATIC,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": _device_block_text(),
        },
    ]


def build_system_uncached():
    """System prompt as a plain string — no cache_control markers."""
    return f"{SYSTEM_STATIC}\n\n{_device_block_text()}"


# ── API call ─────────────────────────────────────────────────────

def call_once(client, system, command):
    """Make one API call; return (latency_ms, usage_dict, response_text)."""
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": command}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    usage = {
        "input_tokens":                resp.usage.input_tokens,
        "output_tokens":               resp.usage.output_tokens,
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens":     getattr(resp.usage, "cache_read_input_tokens",     0) or 0,
    }
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return latency_ms, usage, text


# ── Benchmark runner ─────────────────────────────────────────────

def run_mode(client, label, system, command, runs):
    print(f"\n  [{label}]  {runs} calls ...")
    samples = []
    usages  = []

    for i in range(runs):
        ms, usage, _ = call_once(client, system, command)
        samples.append(ms)
        usages.append(usage)

        cache_write = usage["cache_creation_input_tokens"]
        cache_read  = usage["cache_read_input_tokens"]
        if cache_write > 0:
            tag = f"WRITE ({cache_write} tok)"
        elif cache_read > 0:
            tag = f"READ  ({cache_read} tok)"
        else:
            tag = "-"

        print(f"    [{i+1:2d}/{runs}]  {ms:7.0f} ms   cache={tag}")

    return samples, usages


# ── Statistics ───────────────────────────────────────────────────

def stats(samples):
    s   = sorted(samples)
    n   = len(s)
    avg = sum(s) / n
    p95 = s[int(n * 0.95)]   # index 9 for n=10 (= max for small N)
    return avg, s[0], s[-1], p95


# ── Output ───────────────────────────────────────────────────────

def print_table(command, runs, on_samples, on_usages, off_samples, off_usages):
    on_avg,  on_min,  on_max,  on_p95  = stats(on_samples)
    off_avg, off_min, off_max, off_p95 = stats(off_samples)

    cache_writes = sum(1 for u in on_usages if u["cache_creation_input_tokens"] > 0)
    cache_hits   = sum(1 for u in on_usages if u["cache_read_input_tokens"]     > 0)

    speedup_avg = off_avg / on_avg if on_avg > 0 else 1.0
    speedup_p95 = off_p95 / on_p95 if on_p95 > 0 else 1.0
    savings_ms  = off_avg - on_avg

    W = 60
    print()
    print("=" * W)
    print("  JARVIS Claude API Latency Benchmark")
    print(f"  Model  : {MODEL}")
    print(f"  Command: \"{command[:50]}{'...' if len(command) > 50 else ''}\"")
    print(f"  Runs per mode: {runs}")
    print("=" * W)
    print()

    hdr = f"  {'Mode':<16}  {'Avg ms':>8}  {'Min ms':>8}  {'Max ms':>8}  {'p95 ms':>8}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    print(f"  {'Caching ON':<16}  {on_avg:>8.0f}  {on_min:>8.0f}  {on_max:>8.0f}  {on_p95:>8.0f}")
    print(f"  {'Caching OFF':<16}  {off_avg:>8.0f}  {off_min:>8.0f}  {off_max:>8.0f}  {off_p95:>8.0f}")
    print()

    print("  Cache activity (caching ON runs):")
    print(f"    Cache writes : {cache_writes}/{runs}")
    print(f"    Cache hits   : {cache_hits}/{runs}")

    if on_usages:
        avg_read_tok  = sum(u["cache_read_input_tokens"]     for u in on_usages) / runs
        avg_input_tok = sum(u["input_tokens"]                for u in on_usages) / runs
        avg_out_tok   = sum(u["output_tokens"]               for u in on_usages) / runs
        print(f"    Avg cache-read tokens  : {avg_read_tok:.0f}")
        print(f"    Avg uncached in-tokens : {avg_input_tok:.0f}")
        print(f"    Avg output tokens      : {avg_out_tok:.0f}")

    if cache_hits == 0 and cache_writes <= 1:
        print()
        print("  NOTE: No cache hits detected.  The system prompt may be below the")
        print(f"        ~1024-token minimum for {MODEL}.")
        print("        Results still show raw API latency; caching would reduce it further.")

    print()
    print("  Summary:")
    direction = "faster" if savings_ms > 0 else "slower"
    print(f"    Avg latency  : {on_avg:.0f} ms (cached)  vs  {off_avg:.0f} ms (uncached)")
    print(f"    Avg savings  : {abs(savings_ms):.0f} ms {direction} with caching")
    print(f"    Avg speedup  : {speedup_avg:.2f}x")
    print(f"    p95 speedup  : {speedup_p95:.2f}x")
    print()


# ── Persistence ──────────────────────────────────────────────────

def save_results(ts, command, runs, on_samples, on_usages, off_samples, off_usages):
    LOGS_DIR.mkdir(exist_ok=True)

    data = []
    if BENCH_FILE.exists():
        with open(BENCH_FILE, encoding="utf-8") as f:
            data = json.load(f)

    on_avg,  on_min,  on_max,  on_p95  = stats(on_samples)
    off_avg, off_min, off_max, off_p95 = stats(off_samples)

    entry = {
        "timestamp":    ts,
        "model":        MODEL,
        "command":      command,
        "runs_per_mode": runs,
        "caching_on": {
            "latency_ms": {
                "avg": round(on_avg,  1), "min": round(on_min,  1),
                "max": round(on_max,  1), "p95": round(on_p95,  1),
            },
            "samples_ms": [round(s, 1) for s in on_samples],
            "usage": on_usages,
        },
        "caching_off": {
            "latency_ms": {
                "avg": round(off_avg,  1), "min": round(off_min,  1),
                "max": round(off_max,  1), "p95": round(off_p95,  1),
            },
            "samples_ms": [round(s, 1) for s in off_samples],
            "usage": off_usages,
        },
        "speedup_avg_x": round(off_avg / on_avg, 3) if on_avg > 0 else None,
        "speedup_p95_x": round(off_p95 / on_p95, 3) if on_p95 > 0 else None,
    }
    data.append(entry)

    with open(BENCH_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Claude API latency: prompt caching ON vs OFF"
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"API calls per mode (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--command", type=str, default=DEFAULT_COMMAND,
        help="Tier 3 (Claude) command to benchmark",
    )
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("\n❌  Set ANTHROPIC_API_KEY first.\n")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    print()
    print(f"  Command : \"{args.command}\"")
    print(f"  Runs    : {args.runs} x 2 modes = {args.runs * 2} total API calls")

    t_wall = time.perf_counter()

    on_samples,  on_usages  = run_mode(client, "Caching ON",  build_system_cached(),   args.command, args.runs)
    off_samples, off_usages = run_mode(client, "Caching OFF", build_system_uncached(), args.command, args.runs)

    elapsed = time.perf_counter() - t_wall

    print_table(
        args.command, args.runs,
        on_samples, on_usages,
        off_samples, off_usages,
    )
    print(f"  Total wall time: {elapsed:.1f}s\n")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_results(ts, args.command, args.runs, on_samples, on_usages, off_samples, off_usages)
    print(f"  Saved -> {BENCH_FILE.relative_to(REPO_ROOT)}\n")


if __name__ == "__main__":
    main()
