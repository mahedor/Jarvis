"""Flight log: one row per real flight with the settings and situation it was flown under (Michael 11-27).

  python flight_log.py                       # backfill / refresh: scans ~/follow_logs
  python flight_log.py --root DIR            # another logs root
  python flight_log.py --rebuild             # ignore the cache and re-read every run folder

Writes flight_log.csv and flight_log.md in the logs root, newest first. live_follow.py calls update() at the end of
every run (after its logs are closed and the drone is down), so nothing has to be filled in by hand.

Everything comes from what the run itself wrote: drone.json (what was running) and follow.csv (what happened).
A field the run did not record shows "not logged", never a guess. What drone.json holds depends on the version of
live_follow.py that flew (see FIELD_HISTORY). Fake-drone runs (drone.json drone_nickname "fake") are not flights and
are left out (--include-fake lists them); runs from before drone.json existed can't be told apart and are listed.

Notes, the only hand-written part, two ways (both optional):
  * notes.txt in the run folder -> column "notes.txt" (re-read on every update)
  * type into the "notes" column of flight_log.csv -> kept on every regenerate, matched by "run folder"
    (a row whose folder has since been deleted is kept if it has a typed note, marked "folder missing").

Stdlib only, so it runs with any python 3 (no OpenCV / Olympe needed).
"""
import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

NL = "not logged"
SCHEMA = 2                      # bump when a column's meaning changes: invalidates the cache (2: precision, 11-27)
CACHE = ".flight_log_cache.json"
CSV_NAME, MD_NAME = "flight_log.csv", "flight_log.md"
AIRBORNE = {"takingoff", "hovering", "flying", "landing"}       # ANAFI flying_state values with the drone in the air
TRACKER_NAMES = {"botsort.yaml": "BoT-SORT", "bytetrack.yaml": "ByteTrack",   # Ultralytics' built-in tracker files
                 "botsort_tuned.yaml": "BoT-SORT tuned"}                      # followme/, Michael 11-27
TUNED_KEYS = (("track_high_thresh", "high"), ("new_track_thresh", "new"), ("track_low_thresh", "low"),
              ("match_thresh", "match"), ("track_buffer", "buffer"))
INT8_SUFFIX = "_int8_openvino_model"            # live_follow.INT8_SUFFIX (this file stays stdlib-only)

# what drone.json holds, by the live_follow.py commit that added it (git log -L on the drone.json write)
FIELD_HISTORY = [
    ("2026-11-22", "cc9bbc3", "drone.json first written: drone name, product, serial, firmware, ip; follow.csv `drone` column"),
    ("2026-11-22", "0af1cfb", "boxes.jsonl (every box per frame)"),
    ("2026-11-22", "aae2f0f", "`model` (detector path)"),
    ("2026-11-26", "f36ad29", "`conf_person`, `conf_vehicle` (cutoffs; no source yet)"),
    ("2026-11-26", "ee40f91", "`conf_source`, `reid`, `profile`, `running` (detector, INT8, cutoffs file, re-ID, match_sim, "
                              "profile applied, click-to-lock, search), `preflight` (pass / deviations)"),
    ("2026-11-26", "3f29585", "`running.tracker`, `running.imgsz`"),
    ("2026-11-26", "9f549ff", "`running.lost_s`"),
    ("2026-11-27", "COMMIT_TBD", "`detector` (precision, backend, input shape, the export's metadata.yaml / FakeQuantize / "
                                 "calibration), `running.detector_precision`, `tracker` (file + settings); tracker "
                                 "botsort_tuned.yaml from here on"),
]

COLUMNS = ["date/time", "run folder", "drone", "serial", "firmware", "duration", "frames", "FPS", "airborne", "battery",
           "model", "precision", "cutoffs", "tracker", "imgsz", "re-ID", "preflight", "drone profile", "FOLLOW time", "re-locks",
           "locks (clicks / N)", "spin searches", "LOOK/GO time", "video", "other settings", "notes.txt", "notes"]
MD_COLUMNS = ["date/time", "run folder", "drone", "duration", "model", "precision", "cutoffs", "tracker", "imgsz", "re-ID",
              "preflight", "FOLLOW time", "re-locks", "video", "notes"]
# drone.json keys shown in their own columns; anything else ends up in "other settings" (so new fields show up)
_SHOWN_TOP = {"drone_product_name", "drone_nickname", "drone_serial", "drone_firmware", "name", "ip", "model",
              "conf_person", "conf_vehicle", "conf_source", "reid", "profile", "running", "preflight", "detector",
              "tracker"}
_SHOWN_RUNNING = {"detector_model", "detector_int8", "detector_precision", "detection_cutoffs", "tracker", "imgsz", "reid",
                  "match_sim"}
_RUN_NAME = re.compile(r"^(\d{8})_(\d{6})")


# ---------------------------------------------------------------- reading one run folder

def is_run_folder(p):
    return p.is_dir() and ((p / "follow.csv").exists() or (p / "drone.json").exists())


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt_s(s):
    if s is None:
        return NL
    s = int(round(s))
    return f"{s // 60}m{s % 60:02d}s"


def _onsets(flags):
    """Number of False -> True transitions (a run of consecutive True rows is one event)."""
    n, prev = 0, False
    for f in flags:
        n += f and not prev
        prev = f
    return n


def start_time(folder_name):
    """The run's start time, from the folder name live_follow.py gives it (<YYYYMMDD_HHMMSS>[_drone])."""
    m = _RUN_NAME.match(folder_name)
    if not m:
        return NL
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return NL


def read_info(run):
    """drone.json -> (dict, problem or None)."""
    p = run / "drone.json"
    if not p.exists():
        return {}, None
    try:
        return json.loads(p.read_text()), None
    except (OSError, ValueError) as e:
        return {}, f"drone.json unreadable ({e.__class__.__name__})"


def read_rows(run):
    """follow.csv -> (rows, problem or None). A half-written last line (crash, dead battery) is tolerated."""
    p = run / "follow.csv"
    if not p.exists():
        return [], None
    try:
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            return list(csv.DictReader(fh)), None
    except (OSError, csv.Error) as e:
        return [], f"follow.csv unreadable ({e.__class__.__name__})"


def flight_stats(rows):
    """Cheap summary of follow.csv. Times are measured from t_s (wall clock per frame), not frames / assumed fps."""
    s = dict(duration=NL, frames=NL, fps=NL, airborne=NL, battery=NL, follow=NL, relocks=NL, locks=NL, spins=NL,
             lookgo=NL)
    if not rows:
        return s
    cols = set(rows[0])
    s["frames"] = str(len(rows))
    ts = [_num(r.get("t_s")) for r in rows]
    good = [t for t in ts if t is not None]
    if not good:
        return s
    s["duration"] = fmt_s(good[-1])
    if len(good) > 1 and good[-1] > good[0]:
        s["fps"] = f"{(len(good) - 1) / (good[-1] - good[0]):.1f}"
    # dt of row i = time until the next row (how long the program stayed in that row's mode/state)
    dts = []
    for i, t in enumerate(ts):
        nxt = ts[i + 1] if i + 1 < len(ts) else None
        dts.append(max(0.0, nxt - t) if t is not None and nxt is not None else 0.0)

    def time_where(pred):
        return sum(dt for r, dt in zip(rows, dts) if pred(r))
    if "flying_state" in cols and any(r.get("flying_state") for r in rows):
        s["airborne"] = fmt_s(time_where(lambda r: str(r.get("flying_state", "")).lower() in AIRBORNE))
    if "battery_pct" in cols:
        bats = [b for b in (_num(r.get("battery_pct")) for r in rows) if b is not None]
        if bats:
            s["battery"] = f"{bats[0]:.0f}% -> {bats[-1]:.0f}%"
    if "mode" in cols:
        fol = time_where(lambda r: r.get("mode") == "FOLLOW")
        s["follow"] = fmt_s(fol)
        if "state" in cols:
            s["follow"] += f" (on target {fmt_s(time_where(lambda r: r.get('mode') == 'FOLLOW' and r.get('state') == 'FOLLOWING'))})"
        s["lookgo"] = fmt_s(time_where(lambda r: r.get("mode") == "LOOKGO"))
    if "state" in cols:
        s["spins"] = str(_onsets([str(r.get("state", "")).startswith("SEARCH (spinning") for r in rows]))
    if "lock_event" in cols:
        s["relocks"] = str(_onsets([r.get("lock_event") == "reacquired_reid" for r in rows]))
        if "target_id" in cols:
            # same rule as replay.recorded_locks: a change of target that is not a re-ID re-lock is a click
            # (or first_lock with --auto-lock); a change to no target is N (new lock)
            clicks = resets = 0
            prev = ""
            for r in rows:
                tid = r.get("target_id") or ""
                if tid != prev:
                    if tid == "":
                        resets += 1
                    elif r.get("lock_event") != "reacquired_reid":
                        clicks += 1
                prev = tid
            s["locks"] = f"{clicks} / {resets}"
    return s


def _fmt_val(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def precision(info, run, model):
    """The detector's precision as flown. Since 11-27 drone.json records it (`detector`, read from the model files);
    older runs: INT8 only if the recorded model path is an INT8 export (<name>_int8_openvino_model), else not logged
    (a .pt path alone does not say what precision it ran at in that version)."""
    det = info.get("detector") if isinstance(info.get("detector"), dict) else {}
    if det.get("summary"):
        return str(det["summary"])
    if run.get("detector_precision"):
        return str(run["detector_precision"])
    if model and Path(str(model)).name.endswith(INT8_SUFFIX):
        return "INT8 (derived from the model path; precision not logged)"
    return NL


def settings(info):
    """drone.json -> the settings columns. Only what was written; nothing inferred from dates or notes."""
    run = info.get("running") if isinstance(info.get("running"), dict) else {}
    out = {}
    # drone
    name = info.get("name") or info.get("drone_nickname")
    prod = info.get("drone_product_name")
    out["drone"] = (f"{name} ({prod})" if name and prod and prod != name else name or prod) or NL
    out["serial"] = info.get("drone_serial") or NL
    out["firmware"] = info.get("drone_firmware") or NL
    # model (+ INT8): the running record if present, else the bare model path (11-22 .. 11-26)
    model = run.get("detector_model") or info.get("model")
    if model:
        int8 = run.get("detector_int8")
        if int8 is None:                      # 'model' only: INT8 is a property of that exact path (live_follow.is_int8)
            int8 = Path(str(model)).name.endswith(INT8_SUFFIX)
        out["model"] = f"{Path(str(model)).name} ({'INT8' if int8 else 'not INT8'})"
    else:
        out["model"] = NL
    out["precision"] = precision(info, run, model)
    # cutoffs (+ source)
    cp, cv = info.get("conf_person"), info.get("conf_vehicle")
    if cp is not None or cv is not None:
        src = info.get("conf_source") or (f"from {run['detection_cutoffs']}" if run.get("detection_cutoffs") else
                                          f"source {NL}")
        out["cutoffs"] = f"person {_fmt_val(cp) if cp is not None else NL} / vehicle {_fmt_val(cv) if cv is not None else NL} ({src})"
    else:
        out["cutoffs"] = NL
    # tracker (+ settings file)
    tr = run.get("tracker")
    out["tracker"] = (f"{TRACKER_NAMES[Path(str(tr)).name]} ({tr})" if tr and Path(str(tr)).name in TRACKER_NAMES
                      else str(tr) if tr else NL)
    tset = info["tracker"].get("settings") if isinstance(info.get("tracker"), dict) else None
    if isinstance(tset, dict) and tr:                    # the yaml's values as flown (since 11-27)
        out["tracker"] += ": " + " / ".join(f"{short} {_fmt_val(tset[k])}" for k, short in TUNED_KEYS if k in tset)
    out["imgsz"] = _fmt_val(run["imgsz"]) if run.get("imgsz") is not None else NL
    # re-ID + its re-lock cutoff
    kind = run.get("reid")
    note = str(info.get("reid") or "")
    model_name = note.split("re-ID:", 1)[1].strip() if note.startswith("re-ID:") else note
    reid = model_name or kind
    if reid and model_name and kind and kind not in model_name:
        reid = f"{model_name} ({kind})"
    ms = run.get("match_sim")
    out["re-ID"] = (f"{reid or NL}, re-lock >= {f'{ms:.2f}' if isinstance(ms, (int, float)) else NL}" if reid or ms is not None else NL)
    # pre-flight check
    pf = info.get("preflight")
    if isinstance(pf, dict):
        skipped = ", ".join(x.get("name", "?") for x in pf.get("skipped", []))
        if pf.get("passed"):
            out["preflight"] = f"PASS ({len(pf.get('checked', []))} checked" + (f"; skipped: {skipped})" if skipped else ")")
        else:
            devs = "; ".join(f"{x.get('name')}: decided {x.get('decided')!r}, ran {x.get('actual')!r}"
                             for x in pf.get("deviations", []))
            out["preflight"] = ("DEVIATIONS" + (" (--off-decisions)" if pf.get("off_decisions") else "") + f": {devs}"
                                + (f" (skipped: {skipped})" if skipped else ""))
    else:
        out["preflight"] = NL
    out["drone profile"] = str(info["profile"]) if info.get("profile") else NL
    other = [f"{k}={_fmt_val(v)}" for k, v in run.items() if k not in _SHOWN_RUNNING]
    other += [f"{k}={json.dumps(v) if isinstance(v, (dict, list)) else _fmt_val(v)}"
              for k, v in info.items() if k not in _SHOWN_TOP]
    out["other settings"] = "; ".join(other) if other else NL
    return out


def videos(run):
    clean = sorted(p.name for p in run.glob("video_*.mp4"))
    ov = sorted(p.name for p in run.glob("overlay_*.mp4"))
    if not clean and not ov:
        return "none in folder"
    txt = (clean[0] if len(clean) == 1 else f"{clean[0]} .. {clean[-1]} ({len(clean)} files)") if clean else "no clean video"
    return txt + (f" + overlay x{len(ov)}" if ov else "")


def summarize(run):
    """One run folder -> one row (dict over COLUMNS, without the typed 'notes')."""
    info, p1 = read_info(run)
    rows, p2 = read_rows(run)
    st = flight_stats(rows)
    s = settings(info)
    if s["drone"] == NL and rows and rows[0].get("drone"):
        s["drone"] = rows[0]["drone"]                      # follow.csv's own drone column (same commit as drone.json)
    notes_txt = ""
    if (run / "notes.txt").exists():
        try:
            notes_txt = " ".join((run / "notes.txt").read_text(encoding="utf-8", errors="replace").split())
        except OSError:
            notes_txt = "notes.txt unreadable"
    problems = "; ".join(p for p in (p1, p2) if p)
    row = {"date/time": start_time(run.name), "run folder": run.name, **s,
           "duration": st["duration"], "frames": st["frames"], "FPS": st["fps"], "airborne": st["airborne"],
           "battery": st["battery"], "FOLLOW time": st["follow"], "re-locks": st["relocks"],
           "locks (clicks / N)": st["locks"], "spin searches": st["spins"], "LOOK/GO time": st["lookgo"],
           "video": videos(run), "notes.txt": notes_txt}
    if problems:
        row["other settings"] = problems if row["other settings"] == NL else f"{problems}; {row['other settings']}"
    row["_fake"] = info.get("drone_nickname") == "fake"
    return row


# ---------------------------------------------------------------- the whole log

def _signature(run):
    sig = [SCHEMA]
    for n in ("drone.json", "follow.csv", "notes.txt"):
        p = run / n
        sig.append([p.stat().st_mtime_ns, p.stat().st_size] if p.exists() else None)
    sig.append(sorted(p.name for p in run.glob("*.mp4")))
    return sig


def read_typed_notes(path):
    """{run folder: (notes, whole old row)} from an existing flight_log.csv (Excel may have saved it with ; or a BOM)."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t")
    except (csv.Error, IndexError):
        dialect = csv.excel
    out = {}
    for r in csv.DictReader(text.splitlines(), dialect=dialect):
        folder, note = (r.get("run folder") or "").strip(), (r.get("notes") or "").strip()
        if folder:
            out[folder] = (note, r)
    return out


def _md_cell(v):
    return str(v).replace("|", "\\|").replace("\n", " ")


def write_md(path, rows, root, n_fake):
    lines = ["# Flight log", "",
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `followme/flight_log.py` from `{root}`, "
             f"newest first. {len(rows)} run(s)" + (f"; {n_fake} fake-drone run(s) not listed" if n_fake else "") + ".",
             "",
             "Every field is read from the run's own `drone.json` / `follow.csv`. **not logged** = the version of "
             "`live_follow.py` that ran did not record it (nothing is filled in from memory). Notes: `notes.txt` in the "
             "run folder, or the `notes` column of `flight_log.csv` (kept on regenerate). More columns (serial, "
             "firmware, frames, FPS, airborne, battery, drone profile, clicks / N, spin searches, other settings) are in "
             "`flight_log.csv`.", "",
             "| " + " | ".join(MD_COLUMNS) + " |", "|" + "---|" * len(MD_COLUMNS)]
    for r in rows:
        cells = dict(r)
        cells["duration"] = (f"{r['duration']} (air {r['airborne']}, battery {r['battery']})"
                             if r["duration"] != NL else NL)
        cells["notes"] = " / ".join(x for x in (r.get("notes.txt"), r.get("notes")) if x)
        lines.append("| " + " | ".join(_md_cell(cells.get(c, "")) for c in MD_COLUMNS) + " |")
    lines += ["", "## What drone.json holds, by date", "",
              *[f"- {d} `{c}`: {what}" for d, c, what in FIELD_HISTORY],
              "- before 2026-11-22: no drone.json; only follow.csv (no drone, model, cutoffs or settings recorded)", ""]
    _atomic_write(path, "\n".join(lines))


def _atomic_write(path, text, encoding="utf-8"):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)          # a crash mid-write never leaves a half log (or loses typed notes)


def update(root=None, include_fake=False, rebuild=False, quiet=False):
    """Rescan `root`, rewrite flight_log.csv + flight_log.md. Raises on failure (the caller decides how loud).
    Unchanged run folders come from the cache, so the call at the end of every flight stays fast.
    -> (number of rows written, path of the csv)."""
    root = Path(root or Path.home() / "follow_logs").expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"logs root not found: {root}")
    csv_path, md_path, cache_path = root / CSV_NAME, root / MD_NAME, root / CACHE
    typed = read_typed_notes(csv_path)
    cache = {}
    if not rebuild and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (OSError, ValueError):
            cache = {}
    new_cache, rows, n_fake = {}, [], 0
    for run in sorted((p for p in root.iterdir() if is_run_folder(p)), key=lambda p: p.name):
        sig = _signature(run)
        hit = cache.get(run.name)
        row = dict(hit["row"]) if hit and hit.get("sig") == sig else summarize(run)
        new_cache[run.name] = dict(sig=sig, row=dict(row))
        if row.get("_fake") and not include_fake:
            n_fake += 1
            continue
        rows.append(row)
    present = {r["run folder"] for r in rows} | {k for k, v in new_cache.items() if v["row"].get("_fake")}
    for folder, (note, old) in typed.items():          # a deleted folder's typed note is kept, never dropped
        if folder not in present and note:
            kept = {c: old.get(c, "") for c in COLUMNS}
            if not kept["video"].startswith("folder missing"):
                kept["video"] = f"folder missing; was: {kept['video']}"
            rows.append(kept)
    for r in rows:
        r["notes"] = typed.get(r["run folder"], ("", None))[0]
    rows.sort(key=lambda r: (r["date/time"] if r["date/time"] != NL else "", r["run folder"]), reverse=True)
    from io import StringIO
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    _atomic_write(csv_path, buf.getvalue(), encoding="utf-8-sig")     # BOM: Excel reads it as UTF-8
    write_md(md_path, rows, root, n_fake)
    try:
        _atomic_write(cache_path, json.dumps(new_cache))
    except OSError:
        pass                                           # the cache only saves time
    if not quiet:
        print(f"flight log: {len(rows)} run(s)" + (f" ({n_fake} fake-drone run(s) left out)" if n_fake else "")
              + f" -> {csv_path} and {md_path.name}")
    return len(rows), csv_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path.home() / "follow_logs"), help="logs root (default ~/follow_logs)")
    ap.add_argument("--include-fake", action="store_true", help="also list fake-drone runs")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache and re-read every run folder")
    a = ap.parse_args(argv)
    update(a.root, include_fake=a.include_fake, rebuild=a.rebuild)


if __name__ == "__main__":
    main()
