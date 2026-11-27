"""Tests for live_follow.py: the FOLLOW safety rules (pure logic) + a headless run against the fake drone."""
import subprocess
import sys
import time
from pathlib import Path

import csv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import live_follow as lf  # noqa: E402

W, H = 1280, 720
TEL = dict(altitude_takeoff_m=None, gimbal_pitch_deg=None)


def person(x, tid=1, w=80, h=200):
    return (x - w / 2, 300, x + w / 2, 300 + h, tid, 0, 0.9)


def logic(**kw):
    kw.setdefault("auto_lock", True)        # most tests are about following/search, not how the target was chosen
    return lf.FollowLogic({0}, max_speed=20, **kw)


def test_turns_toward_target_and_caps_speed():
    L = logic(); now = time.time()
    cmd, info = L.step([person(1250)], now, now, W, H, TEL)       # far right
    assert info["state"] == "FOLLOWING" and cmd["yaw"] > 0 and cmd["yaw"] <= 20
    cmd, _ = L.step([person(30)], now, now, W, H, TEL)
    for _ in range(10):
        cmd, _ = L.step([person(30)], now, now, W, H, TEL)         # far left (smoothing settles)
    assert -20 <= cmd["yaw"] < 0
    assert cmd["gaz"] == 0 and cmd["roll"] == 0


def test_stale_detections_hover():
    L = logic(); now = time.time()
    cmd, info = L.step([person(1250)], now - 2.0, now, W, H, TEL)
    assert info["state"].startswith("STALE") and all(v == 0 for v in cmd.values())


def test_no_detection_yet_is_stale():
    cmd, info = logic().step([], None, time.time(), W, H, TEL)
    assert all(v == 0 for v in cmd.values())


def test_brief_gap_hovers_then_lost():
    L = logic(lost_s=1.0); now = time.time()
    L.step([person(1000)], now, now, W, H, TEL)
    cmd, info = L.step([], now + 0.3, now + 0.3, W, H, TEL)
    assert info["state"].startswith("SEARCHING") and all(v == 0 for v in cmd.values())
    cmd, info = L.step([], now + 1.5, now + 1.5, W, H, TEL)          # spin in place toward the last-seen side
    assert info["state"].startswith("SEARCH (spinning") and cmd["yaw"] > 0 and cmd["pitch"] == 0 and cmd["gaz"] == 0
    assert cmd["roll"] == 0
    cmd, info = L.step([], now + 1.5 + lf.SEARCH_S, now + 1.5 + lf.SEARCH_S, W, H, TEL)   # timeout without telemetry
    assert info["state"].startswith("LOST") and all(v == 0 for v in cmd.values())


def test_search_can_be_disabled_and_is_fast_regardless_of_follow_cap():
    L = logic(lost_s=1.0, search_s=0); now = time.time()
    L.step([person(1000)], now, now, W, H, TEL)
    cmd, info = L.step([], now + 1.5, now + 1.5, W, H, TEL)
    assert info["state"].startswith("LOST") and all(v == 0 for v in cmd.values())
    L = lf.FollowLogic({0}, max_speed=5, lost_s=1.0, search_yaw=50, auto_lock=True); now = time.time()
    L.step([person(100)], now, now, W, H, TEL)                    # last seen on the LEFT
    cmd, _ = L.step([], now + 1.5, now + 1.5, W, H, TEL)
    assert cmd["yaw"] == -50 and cmd["pitch"] == 0 and cmd["roll"] == 0 and cmd["gaz"] == 0   # spins, never moves


def test_search_spins_one_way_then_the_other_then_hovers():
    L = logic(lost_s=1.0); now = time.time()
    L.step([person(1000)], now, now, W, H, dict(TEL, drone_yaw_deg=0.0))   # last seen on the RIGHT
    yaw, dirs = 0.0, []
    for k in range(100):                                         # telemetry: turning 10 deg per step, wrapping at 180
        t = now + 1.5 + 0.1 * k
        cmd, info = L.step([], t, t, W, H, dict(TEL, drone_yaw_deg=yaw))
        if not info["state"].startswith("SEARCH"):
            break
        dirs.append(1 if cmd["yaw"] > 0 else -1)
        yaw = (yaw + 10 * dirs[-1] + 180) % 360 - 180
    assert dirs[0] == 1 and dirs[-1] == -1                       # right first, then left
    assert 70 <= len(dirs) <= 75                                 # ~360 deg each way at 10 deg/step
    assert info["state"].startswith("LOST") and all(v == 0 for v in cmd.values())
    L.step([person(1000)], t + 0.2, t + 0.2, W, H, dict(TEL, drone_yaw_deg=yaw))   # found again -> counter resets
    assert L.turned_deg == 0.0


def test_search_one_way_option():
    L = logic(lost_s=1.0, search_both=False); now = time.time()
    L.step([person(1000)], now, now, W, H, dict(TEL, drone_yaw_deg=0.0))
    yaw, n = 0.0, 0
    for k in range(100):
        t = now + 1.5 + 0.1 * k
        cmd, info = L.step([], t, t, W, H, dict(TEL, drone_yaw_deg=yaw))
        if not info["state"].startswith("SEARCH"):
            break
        assert cmd["yaw"] > 0
        yaw = (yaw + 10 + 180) % 360 - 180; n += 1
    assert 35 <= n <= 38


def test_stale_detector_never_rotates():
    L = logic(lost_s=1.0); now = time.time()
    L.step([person(1000)], now, now, W, H, TEL)
    cmd, info = L.step([], now, now + 5, W, H, TEL)               # detections 5 s old
    assert info["state"].startswith("STALE") and all(v == 0 for v in cmd.values())


def test_lost_target_not_replaced_by_stranger_once_gallery_exists():
    import numpy as np
    L = logic(lost_s=1.0); now = time.time()
    a, b = np.array([1, 0, 0], np.float32), np.array([0, 1, 0], np.float32)
    for i in range(3):                                            # build a gallery of the target (id 1)
        t = now + i
        L.step([person(600, tid=1)], t, t, W, H, TEL, {1: a})
    t = now + 3.2
    L.step([person(610, tid=4)], t, t, W, H, TEL, {4: b})         # stranger appears right where the target was
    assert L.target_id == 1
    t = now + 3.6
    L.step([person(610, tid=4), person(200, tid=6)], t, t, W, H, TEL, {6: a})   # target back under id 6
    assert L.target_id == 6


def test_click_only_lock_and_no_new_lock_when_lost():
    L = logic(auto_lock=False); now = time.time()                # Michael 11-20: lock only on a click
    cmd, info = L.step([person(900, tid=1)], now, now, W, H, TEL)
    assert L.target_id is None and info["state"].startswith("NO TARGET") and all(v == 0 for v in cmd.values())
    L.pick(1, [person(900, tid=1)])
    L.step([person(900, tid=1)], now + 0.1, now + 0.1, W, H, TEL)
    assert L.target_id == 1
    # track ID changes (tracker lost and re-found) and there's no gallery yet: do NOT grab the nearest person
    L.step([person(200, tid=5, w=200, h=400), person(910, tid=9)], now + 0.2, now + 0.2, W, H, TEL)
    assert L.target_id == 1


def test_click_pick_sticks():
    L = logic(); now = time.time()
    L.pick(5)
    L.step([person(200, tid=5), person(900, tid=1, w=300, h=500)], now, now, W, H, TEL)
    assert L.target_id == 5 and L.picked_by == "you"


def run_fake(script, seconds, extra=()):
    out = HERE / "_test_follow_out"
    cmd = [sys.executable, str(HERE / "live_follow.py"), "--fake", "--fake-detector", "--headless", "--auto-lock",
           "--script", script, "--seconds", str(seconds), "--out", str(out), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=HERE)
    assert p.returncode == 0, p.stderr[-2000:]
    run = sorted(out.iterdir())[-1]
    return p.stdout, list(csv.DictReader(open(run / "follow.csv")))


def test_headless_follow_then_manual_override():
    out, rows = run_fake("t@0.5,m@1,w@4,q@6", 8)
    modes = [r["mode"] for r in rows]
    assert "FOLLOW" in modes and modes[-1] == "MANUAL"
    fol = [r for r in rows if r["mode"] == "FOLLOW" and r["state"] == "FOLLOWING"]
    assert fol, "never locked onto the fake target"
    assert all(abs(float(r["yaw"])) <= 20 and float(r["gaz"]) == 0 for r in fol)
    assert any(abs(float(r["yaw"])) > 0 for r in fol)               # it actually steers
    assert "manual key -> MANUAL" in out and "Landing" in out


def test_video_stall_drops_to_manual():
    out, rows = run_fake("t@0.5,m@1", 7, ("--fake-stall", "3", "5"))
    assert "NO VIDEO -> MANUAL" in out
    after = [r for r in rows if float(r["t_s"]) > 5.2]
    assert after and all(r["mode"] == "MANUAL" for r in after)     # does NOT silently resume following


def test_manual_mode_builds_gallery_and_follow_keeps_it():
    import numpy as np
    L = logic(auto_lock=False); now = time.time(); a = np.array([1, 0, 0], np.float32)
    L.pick(1, [person(600, tid=1)])                               # you click yourself
    for i in range(4):                                            # MANUAL: lock-on runs, no commands
        t = now + i
        L.observe([person(600, tid=1)], t, t, W, H, {1: a})
    assert L.target_id == 1 and len(L.lock.gallery) == 4          # one sample per second
    L.resume()                                                    # M -> FOLLOW keeps target + gallery
    assert L.target_id == 1 and len(L.lock.gallery) == 4
    t = now + 4.2
    cmd, info = L.step([person(1000, tid=1)], t, t, W, H, TEL)
    assert info["state"] == "FOLLOWING" and cmd["yaw"] > 0


def test_manual_mode_relocks_by_reid():
    import numpy as np
    L = logic(auto_lock=False); now = time.time(); a, b = np.array([1, 0, 0], np.float32), np.array([0, 1, 0], np.float32)
    L.pick(1, [person(600, tid=1)])
    for i in range(3):
        t = now + i
        L.observe([person(600, tid=1)], t, t, W, H, {1: a})
    t = now + 3.2
    L.observe([person(610, tid=4)], t, t, W, H, {4: b})           # stranger where you were: refused
    assert L.target_id == 1
    t = now + 3.6
    L.observe([person(610, tid=4), person(200, tid=6)], t, t, W, H, {6: a})
    assert L.target_id == 6 and L.lock.last_sim is not None and L.lock.last_sim > 0.99


def car(x, tid=7, w=200, h=100):
    return (x - w / 2, 400, x + w / 2, 400 + h, tid, 2, 0.9)


def test_clicked_vehicle_is_followed_and_not_replaced_when_lost():
    L = logic(auto_lock=False); now = time.time()
    L.pick(7, [car(1000), person(300, tid=1)])                    # you clicked a car (Michael 11-19: keep vehicles)
    cmd, info = L.step([car(1000), person(300, tid=1)], now, now, W, H, TEL)
    assert L.target_id == 7 and info["state"] == "FOLLOWING" and cmd["yaw"] > 0
    assert L.wants_embedding([car(1000)], now) == []              # no OSNet for vehicles
    t = now + 0.5                                                 # tracker lost it; another car appears right there
    L.step([car(1030, tid=9), person(1000, tid=1)], t, t, W, H, TEL)
    assert L.target_id == 7                                       # no vehicle re-ID yet: never guess (Michael 11-20)


def test_every_run_records_which_drone():
    import json
    import record
    assert record.nickname("ANAFI-G134762") == "droneA" and record.nickname("ANAFI-J102372") == "droneB"
    assert record.nickname("ANAFI-X999") == "ANAFI-X999" and record.nickname(None) == "unknown"
    out, rows = run_fake("q@1.5", 3)
    run = sorted((HERE / "_test_follow_out").iterdir())[-1]
    assert run.name.endswith("_droneA")                                # fake drone -> default name
    info = json.loads((run / "drone.json").read_text())
    assert info["drone_nickname"] == "fake" and info["name"] == "droneA"
    assert rows and all(r["drone"] == "droneA" for r in rows)


def test_boxes_logged_and_flight_index(tmp_path):
    import json
    import flight_index
    out, rows = run_fake("t@0.5,m@1,q@3", 4)
    run = sorted((HERE / "_test_follow_out").iterdir())[-1]
    lines = [json.loads(x) for x in (run / "boxes.jsonl").read_text().splitlines()]
    assert len(lines) == len(rows) and lines[0]["names"] and any(ln["boxes"] for ln in lines)
    s = flight_index.summarize(run)
    assert s["drone"] == "droneA" and s["boxes_logged"] and s["videos"] >= 1


def test_cutoffs_from_f1_sweep():
    """Michael 11-26: the flight model flies at its F1-max cutoffs, per class; unknown models must be explicit."""
    import pytest
    import live_follow as lf
    assert lf.resolve_cutoffs("weights/vd_mav_cl_s42_int8_openvino_model")[:2] == (0.23, 0.29)   # INT8's own sweep
    assert lf.resolve_cutoffs("weights/vd_mav_cl_s42.pt")[:2] == (0.24, 0.28)
    assert lf.resolve_cutoffs("other.pt", 0.3, 0.4)[:2] == (0.3, 0.4)
    with pytest.raises(SystemExit):
        lf.resolve_cutoffs("other.pt")
    assert lf.keep_box("person", 0.245, 0.24, 0.28) and not lf.keep_box("car", 0.245, 0.24, 0.28)
    assert lf.keep_box("car", 0.29, 0.24, 0.28)


# ---- pre-flight self-check against flight_decisions.json (11-26) ----

def _latest_drone_json():
    import json
    return json.loads((sorted((HERE / "_test_follow_out").iterdir())[-1] / "drone.json").read_text())


def test_code_defaults_are_the_decided_settings():
    """A default changed in code without a new decision fails here, before it can fail at the airfield."""
    import lockon
    dec = {k: v["value"] for k, v in lf.load_decisions().items()}
    assert dec["detector_model"] == lf._rel(lf.INT8_MODEL, HERE)
    assert dec["detector_int8"] is True and lf.INT8_MODEL.name.endswith(lf.INT8_SUFFIX)
    assert dec["detection_cutoffs"] == lf._rel(lf.cutoffs_file_for(lf.INT8_MODEL)) == lf._rel(lf.CUTOFFS_FILE_INT8)
    assert dec["tracker"] == lf.TRACKER and dec["imgsz"] == lf.IMGSZ
    assert dec["match_sim"] == lockon.MATCH_SIM
    assert dec["search_deg"] == lf.SEARCH_DEG and dec["search_both_ways"] == lf.SEARCH_BOTH
    assert dec["lost_s"] == lf.LOST_S == 1.0                          # Michael 11-26: keep the 1 second hover
    for d in lf.load_decisions().values():                            # replay: replay.py's own check (test_replay.py)
        assert d["source"] and d["date"] and set(d["skip_with"]) <= {"--fake", "--fake-detector", "replay"}


def test_check_decisions_finds_every_kind_of_deviation():
    dec = lf.load_decisions()
    ok = {k: v["value"] for k, v in dec.items()}
    assert lf.check_decisions(ok, dec, set())["passed"]
    bad = dict(ok, detector_model="weights/vd_mav_cl_s42.pt", detector_int8=False, tracker="bytetrack.yaml", imgsz=960)
    del bad["reid"]                                                   # no reading = deviation, never a silent pass
    r = lf.check_decisions(bad, dec, set())
    assert not r["passed"] and {x["name"] for x in r["deviations"]} == {"detector_model", "detector_int8", "tracker",
                                                                        "imgsz", "reid"}
    r = lf.check_decisions(bad, dec, {"--fake-detector"})             # only the detector/tracker checks are skipped
    assert {x["name"] for x in r["deviations"]} == {"reid"}
    assert {s["name"] for s in r["skipped"]} == {"detector_model", "detector_int8", "tracker", "imgsz"}
    assert lf.profile_ok("drone settings match drone_profile.json") and lf.profile_ok("applied drone_profile.json (2 changed)")
    assert not lf.profile_ok("applied drone_profile.json (2 changed); FAILED: {'x': 'failed: y'}")
    assert not lf.profile_ok("no drone_profile.json yet (run drone_settings.py --save on drone A)")


def test_preflight_pass_is_logged():
    out, rows = run_fake("q@1", 2)
    assert "pre-flight check: PASS" in out
    pf = _latest_drone_json()["preflight"]
    assert pf["passed"] and not pf["deviations"] and not pf["off_decisions"]
    assert {"reid", "match_sim", "detection_cutoffs", "search_deg"} <= set(pf["checked"])
    assert {s["name"] for s in pf["skipped"]} == {"detector_model", "detector_int8", "tracker", "imgsz",
                                                  "drone_profile_applied", "click_to_lock"}
    assert _latest_drone_json()["running"]["tracker"] == "botsort.yaml"
    assert _latest_drone_json()["running"]["imgsz"] == 640


def test_preflight_mismatch_stops():
    cmd = [sys.executable, str(HERE / "live_follow.py"), "--fake", "--fake-detector", "--headless", "--seconds", "2",
           "--out", str(HERE / "_test_follow_out"), "--match-sim", "0.8", "--conf-person", "0.3", "--conf-vehicle", "0.3"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=HERE)
    assert p.returncode != 0 and "STOP: not flying what was decided" in p.stderr
    assert "match_sim: decided 0.7, about to run 0.8" in p.stderr and "detection_cutoffs" in p.stderr
    assert "Logging to" not in p.stdout                               # stopped before any flight folder or loop


def test_preflight_override_is_logged():
    out, rows = run_fake("q@1", 2, ("--match-sim", "0.8", "--off-decisions"))
    assert "DELIBERATE DEVIATIONS" in out
    pf = _latest_drone_json()["preflight"]
    assert not pf["passed"] and pf["off_decisions"]
    assert [(x["name"], x["decided"], x["actual"]) for x in pf["deviations"]] == [("match_sim", 0.7, 0.8)]


def test_tracker_decision_is_the_benchmark_winner():
    """Re-derive the tracker pick from the benchmark's own output (results/followme_tracker_scores.jsonl, byte-exact
    from Drive): with the flight detector, BoT-SORT + camera-motion compensation, tracker re-ID off, has the highest
    class-average HOTA on every set. That config is Ultralytics' botsort.yaml unchanged, which is what flies."""
    import json
    p = HERE.parent / "results" / "followme_tracker_scores.jsonl"
    assert p.is_file(), f"{p} is committed evidence and must exist"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    best = {}
    for split in ("val", "test-dev", "MOT17-train", "MOT20-train"):
        mine = [r for r in rows if r["detector"] == "vd_mav_cl_s42" and r["split"] == split]
        hota = sorted(((r["scores"]["class_avg"]["HOTA"], r["config"]) for r in mine), reverse=True)
        assert hota[0][0] > hota[1][0], f"{split}: tie at the top"
        best[split] = (hota[0][1], round(hota[0][0], 4))
    assert best == {"val": ("botsort|reid-|gmc+", 0.2658), "test-dev": ("botsort|reid-|gmc+", 0.2927),
                    "MOT17-train": ("botsort|reid-|gmc+", 0.3946), "MOT20-train": ("botsort|reid-|gmc+", 0.3173)}
    assert lf.load_decisions()["tracker"]["value"] == lf.TRACKER == "botsort.yaml"
    import pytest
    pytest.importorskip("ultralytics")
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    cfg = YAML.load(check_yaml(lf.TRACKER))
    assert cfg["tracker_type"] == "botsort" and cfg["with_reid"] is False and cfg["gmc_method"] == "sparseOptFlow"
