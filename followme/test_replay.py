"""Tests for replay.py's contract (Michael 11-26): "replay always matches whatever we're flying, unless I state
explicitly otherwise", and any deviation must be brought to his attention.

  * replay's effective settings == live_follow's for every decision in flight_decisions.json, from ONE definition
    (a replay-only flag or default, or a shadow copy of a decided constant, fails here);
  * the same settings check as live_follow's pre-flight: a mismatch STOPS, --off-decisions runs it with a DEVIATION
    banner that is also saved in replay_info.json and commands.csv;
  * click-to-lock: no automatic lock and no nearest-box re-lock; the lock comes from the flight log, --lock-id or a
    window click.
There were no replay tests before 11-26 (the old replay.main was never run by the suite)."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import live_follow as lf  # noqa: E402
import replay  # noqa: E402
from test_live_follow import make_ov_export  # noqa: E402  (an INT8-like OpenVINO export folder, 11-27)

# flags only replay has: where to read/write and how to watch, none of which changes what the program decides
REPLAY_ONLY = {"source", "lock_id", "max_frames", "window"}
SAME_FLAG_OTHER_DEFAULT = {"out"}      # both write output somewhere: live ~/follow_logs, replay <source>/replay


def _defaults(parser):
    return {a.dest: a.default for a in parser._actions}


def _int8_precision(tmp_path):
    """What detector_precision reads from the flight export's own metadata.yaml (the weights are not in the repo)."""
    return lf.detector_precision(make_ov_export(tmp_path))["precision"]


def test_replay_runs_live_follows_settings_for_every_decision(tmp_path):
    live = lf.build_parser().parse_args([])
    rep = replay.build_parser().parse_args(["--source", "x"])
    prec = _int8_precision(tmp_path)
    a_live, a_rep = lf.running_settings(live, True, precision=prec), lf.running_settings(rep, False, precision=prec)
    decisions = lf.load_decisions()
    for name, d in decisions.items():
        assert name in a_live, f"{name}: no reading in running_settings (would be a deviation in every run)"
        if "replay" in d["skip_with"]:
            continue                                   # drone_profile_applied: replay talks to no drone
        assert a_rep[name] == a_live[name] == d["value"], name
    assert [n for n, d in decisions.items() if "replay" in d["skip_with"]] == ["drone_profile_applied"]
    assert lf.follow_kwargs(rep) == lf.follow_kwargs(live)


def test_replay_of_the_int8_model_gets_the_int8_cutoffs_and_passes_the_check(tmp_path):
    """11-26: the INT8 export flies at its own measured cutoffs (0.23 / 0.29, results/followme_cutoffs_balanced_int8.json);
    replay of the same model must get the same pair from the same file, and its settings check (real detector, i.e.
    nothing skipped but drone_profile_applied) must pass."""
    rep = replay.build_parser().parse_args(["--source", "x"])
    live = lf.build_parser().parse_args([])
    assert rep.model == live.model == str(lf.INT8_MODEL)
    got = lf.resolve_cutoffs(rep.model, rep.conf_person, rep.conf_vehicle)
    assert got == lf.resolve_cutoffs(live.model, live.conf_person, live.conf_vehicle)
    assert got == (0.23, 0.29, "F1-max sweep for vd_mav_cl_s42_int8, results/followme_cutoffs_balanced_int8.json")
    actual = lf.running_settings(rep, drone_profile_applied=False, cutoffs=got[:2], precision=_int8_precision(tmp_path))
    assert actual["detection_cutoffs"] == "results/followme_cutoffs_balanced_int8.json"
    r = lf.check_decisions(actual, lf.load_decisions(), {"replay"})
    assert r["passed"], r["deviations"]
    assert "detection_cutoffs" in r["checked"] and [s["name"] for s in r["skipped"]] == ["drone_profile_applied"]
    # a .pt replay reads the .pt entry, and is (correctly) a deviation from what flies
    pt = replay.build_parser().parse_args(["--source", "x", "--model", "weights/vd_mav_cl_s42.pt"])
    assert lf.resolve_cutoffs(pt.model)[:2] == (0.24, 0.28)
    r = lf.check_decisions(lf.running_settings(pt, False, (0.24, 0.28)), lf.load_decisions(), {"replay"})
    assert {x["name"] for x in r["deviations"]} == {"detector_model", "detector_int8", "detector_precision",
                                                    "detection_cutoffs"}             # a .pt is FP32 (11-27)


def test_replay_has_no_settings_of_its_own():
    """Every flag replay shares with live_follow has live_follow's default; a new replay-only flag must be added to
    REPLAY_ONLY on purpose (i.e. someone decides it does not change what replay runs)."""
    d_live, d_rep = _defaults(lf.build_parser()), _defaults(replay.build_parser())
    assert set(d_rep) - set(d_live) == REPLAY_ONLY
    for dest in (set(d_rep) & set(d_live)) - SAME_FLAG_OTHER_DEFAULT:
        assert d_rep[dest] == d_live[dest], f"replay --{dest} default {d_rep[dest]!r} != live {d_live[dest]!r}"
    for name in ("LOST_S", "STALE_S", "MATCH_SIM", "TRACKER", "TRACKER_CHOICE_FILE", "IMGSZ", "INT8_MODEL", "CUTOFFS_FILE", "CUTOFFS_FILE_INT8",
                 "SEARCH_S",
                 "SEARCH_YAW", "SEARCH_DEG", "SEARCH_BOTH", "MARGIN"):
        assert not hasattr(replay, name), f"replay.{name} would be a second source for a flight setting"


def test_no_nearest_box_relock_left():
    """The old replay re-locked onto the nearest/biggest person by itself (pick_target). Gone; and lock-on only
    auto-locks with auto_lock=True, which is --auto-lock, a click_to_lock deviation."""
    assert not hasattr(replay, "pick_target")
    src = (HERE / "replay.py").read_text()
    assert "pick_target(" not in src and "auto_lock=True" not in src


def test_recorded_locks_are_clicks_and_resets_not_reid_relocks():
    def r(f, tid, ev):
        return dict(frame=str(f), target_id=tid, lock_event=ev)
    rows = [r(0, "", "no_target"), r(1, "3", "following"), r(2, "3", "gallery_add"), r(3, "5", "reacquired_reid"),
            r(4, "5", "following"), r(5, "", "no_target"), r(6, "8", "first_lock")]
    assert replay.recorded_locks(rows) == [(1, "pick", "3"), (5, "reset", None), (6, "pick", "8")]
    assert replay.recorded_locks([dict(frame="0", target_id="1")]) is None       # can't tell clicks from re-locks
    assert replay.parse_lock_id("") == (None, 0) and replay.parse_lock_id("4@120") == (4, 120)


def test_place_lock_by_overlap():
    tracks = [(100, 100, 200, 300, 7, 0, 0.9), (600, 100, 700, 300, 9, 0, 0.9)]
    assert replay.place_lock((110, 105, 205, 300), tracks)[4] == 7
    assert replay.place_lock((400, 100, 500, 300), tracks) is None              # nothing there: wait, never guess


# ---- full runs: a fake-drone flight from live_follow.py, then replayed (fake detector, real OSNet) ----

@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    out = tmp_path_factory.mktemp("flight")
    cmd = [sys.executable, str(HERE / "live_follow.py"), "--fake", "--fake-detector", "--headless", "--auto-lock",
           "--script", "t@0.5,m@1.5,q@4", "--seconds", "5", "--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=HERE)
    assert p.returncode == 0, p.stderr[-2000:]
    return next(p for p in out.iterdir() if p.is_dir())        # the root also holds flight_log.* (11-27)


def run_replay(source, out, *extra):
    cmd = [sys.executable, str(HERE / "replay.py"), "--source", str(source), "--fake-detector", "--out", str(out), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=HERE)


def _rows(out):
    return list(csv.DictReader(open(out / "commands.csv")))


def test_replay_of_a_flight_runs_the_flight_program(flight, tmp_path):
    p = run_replay(flight, tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "replay settings check: PASS" in p.stdout and "KNOWN OFFLINE DEVIATIONS" in p.stdout
    info = json.loads((tmp_path / "replay_info.json").read_text())
    pf = info["preflight"]
    assert pf["passed"] and not pf["off_decisions"]
    assert {s["name"] for s in pf["skipped"]} == {"detector_model", "detector_int8", "detector_precision", "tracker",
                                                  "imgsz", "drone_profile_applied"}
    assert info["tracker"]["file"] == "botsort_tuned.yaml" and info["detector"]["precision"] == "none (--fake-detector)"
    assert {"click_to_lock", "lost_s", "match_sim", "search_deg", "search_both_ways", "reid"} <= set(pf["checked"])
    assert info["offline_deviations"] == replay.OFFLINE_DEVIATIONS
    assert info["lock_source"].startswith("flight log: 1 recorded lock") and info["summary"]["locks_placed"] == 1
    rows = _rows(tmp_path)
    assert all(r["mode"] == r["flight_mode"] for r in rows)                   # the recorded MANUAL / FOLLOW
    fol = [r for r in rows if r["mode"] == "FOLLOW"]
    assert fol and all(r["state"] == "FOLLOWING" and r["target_id"] for r in fol)
    assert any(abs(float(r["yaw"])) > 0 for r in fol)
    assert all(float(r["yaw"]) == 0 and float(r["pitch"]) == 0 for r in rows if r["mode"] == "MANUAL")
    assert not any(r["deviation"] for r in rows)
    assert (tmp_path / "replay.mp4").stat().st_size > 0


def test_replay_mismatch_stops(flight, tmp_path):
    out = tmp_path / "rp"
    p = run_replay(flight, out, "--match-sim", "0.8")
    assert p.returncode != 0 and "STOP: replay would not run what is flown" in p.stderr
    assert "match_sim: decided 0.7, about to run 0.8" in p.stderr
    assert not out.exists()                                                     # stopped before any output


def test_replay_off_decisions_shows_and_saves_the_deviation(flight, tmp_path):
    p = run_replay(flight, tmp_path, "--match-sim", "0.8", "--off-decisions", "--max-frames", "30")
    assert p.returncode == 0, p.stderr[-2000:]
    assert "DEVIATION: this replay does NOT run what is flown" in p.stdout
    assert p.stdout.rstrip().endswith("#" * 78)                                # repeated at the end, not scrolled away
    pf = json.loads((tmp_path / "replay_info.json").read_text())["preflight"]
    assert not pf["passed"] and pf["off_decisions"]
    assert [(x["name"], x["decided"], x["actual"]) for x in pf["deviations"]] == [("match_sim", 0.7, 0.8)]
    rows = _rows(tmp_path)
    assert rows and all("match_sim 0.8 (flown 0.7)" in r["deviation"] for r in rows)


def test_replay_auto_lock_is_a_deviation(flight, tmp_path):
    p = run_replay(flight, tmp_path / "rp", "--auto-lock")
    assert p.returncode != 0 and "click_to_lock: decided True, about to run False" in p.stderr


def test_video_without_a_flight_log_never_locks_by_itself(flight, tmp_path):
    clip = tmp_path / "clip.mp4"
    shutil.copy(flight / "video_000.mp4", clip)
    p = run_replay(clip, tmp_path / "none", "--max-frames", "40")
    assert p.returncode == 0, p.stderr[-2000:]
    rows = _rows(tmp_path / "none")
    assert rows and all(r["target_id"] == "" and r["state"].startswith("NO TARGET") for r in rows)
    assert all(float(r["yaw"]) == 0 and float(r["pitch"]) == 0 for r in rows)
    p = run_replay(clip, tmp_path / "lock", "--max-frames", "40", "--lock-id", "1")
    assert p.returncode == 0, p.stderr[-2000:]
    rows = _rows(tmp_path / "lock")
    assert any(r["state"] == "FOLLOWING" and r["target_id"] == "1" for r in rows)
    assert json.loads((tmp_path / "lock" / "replay_info.json").read_text())["lock_source"] == "--lock-id 1"
