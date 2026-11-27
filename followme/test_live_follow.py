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


def latest_run(out=HERE / "_test_follow_out"):
    """Newest run folder (the logs root also holds flight_log.csv / .md since 11-27)."""
    return sorted(p for p in out.iterdir() if p.is_dir())[-1]


def run_fake(script, seconds, extra=()):
    out = HERE / "_test_follow_out"
    cmd = [sys.executable, str(HERE / "live_follow.py"), "--fake", "--fake-detector", "--headless", "--auto-lock",
           "--script", script, "--seconds", str(seconds), "--out", str(out), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=HERE)
    assert p.returncode == 0, p.stderr[-2000:]
    run = latest_run(out)
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
    run = latest_run()
    assert run.name.endswith("_droneA")                                # fake drone -> default name
    info = json.loads((run / "drone.json").read_text())
    assert info["drone_nickname"] == "fake" and info["name"] == "droneA"
    assert rows and all(r["drone"] == "droneA" for r in rows)


def test_boxes_logged_and_flight_index(tmp_path):
    import json
    import flight_index
    out, rows = run_fake("t@0.5,m@1,q@3", 4)
    run = latest_run()
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
    return json.loads((latest_run() / "drone.json").read_text())


def test_code_defaults_are_the_decided_settings():
    """A default changed in code without a new decision fails here, before it can fail at the airfield."""
    import lockon
    dec = {k: v["value"] for k, v in lf.load_decisions().items()}
    assert dec["detector_model"] == lf._rel(lf.INT8_MODEL, HERE)
    assert dec["detector_int8"] is True and lf.INT8_MODEL.name.endswith(lf.INT8_SUFFIX)
    assert dec["detection_cutoffs"] == lf._rel(lf.cutoffs_file_for(lf.INT8_MODEL)) == lf._rel(lf.CUTOFFS_FILE_INT8)
    assert dec["tracker"] == lf.TRACKER == "botsort_tuned.yaml" and dec["imgsz"] == lf.IMGSZ
    assert dec["detector_precision"] == "INT8"                        # Michael 11-22, read from the model files (11-27)
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
    assert {s["name"] for s in r["skipped"]} == {"detector_model", "detector_int8", "detector_precision", "tracker", "imgsz"}
    assert lf.profile_ok("drone settings match drone_profile.json") and lf.profile_ok("applied drone_profile.json (2 changed)")
    assert not lf.profile_ok("applied drone_profile.json (2 changed); FAILED: {'x': 'failed: y'}")
    assert not lf.profile_ok("no drone_profile.json yet (run drone_settings.py --save on drone A)")


def test_preflight_pass_is_logged():
    out, rows = run_fake("q@1", 2)
    assert "pre-flight check: PASS" in out
    pf = _latest_drone_json()["preflight"]
    assert pf["passed"] and not pf["deviations"] and not pf["off_decisions"]
    assert {"reid", "match_sim", "detection_cutoffs", "search_deg"} <= set(pf["checked"])
    assert {s["name"] for s in pf["skipped"]} == {"detector_model", "detector_int8", "detector_precision", "tracker",
                                                  "imgsz", "drone_profile_applied", "click_to_lock"}
    assert _latest_drone_json()["running"]["tracker"] == "botsort_tuned.yaml"
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
    # what flies since 11-27 is that tracker (BoT-SORT, GMC on, re-ID off) with five settings tuned
    assert lf.load_decisions()["tracker"]["value"] == lf.TRACKER == "botsort_tuned.yaml"
    cfg = lf.resolve_tracker(lf.TRACKER)["settings"]
    assert cfg["tracker_type"] == "botsort" and cfg["with_reid"] is False and cfg["gmc_method"] == "sparseOptFlow"


# ---- 11-27: the tuned BoT-SORT settings (botsort_tuned.yaml) and their decision record ----

RESULTS = HERE.parent / "results"
SWEEP = RESULTS / "followme_tracker_settings_sweep.jsonl"
SWEEP_SUMMARY = RESULTS / "followme_tracker_settings_sweep_summary.json"
CUTS_FLIGHT = {"person": 0.23, "vehicle": 0.29}
# record set name -> (sweep row "detector", "split", "post_filter")
SETS = {"val": ("vd_mav_cl_s42", "val", None), "test-dev": ("vd_mav_cl_s42", "test-dev", None),
        "MOT17-train": ("vd_mav_cl_s42", "MOT17-train", None), "MOT20-train": ("vd_mav_cl_s42", "MOT20-train", None),
        "INT8 val": ("vd_mav_cl_s42_int8", "val", CUTS_FLIGHT),
        "INT8 test-dev": ("vd_mav_cl_s42_int8", "test-dev", CUTS_FLIGHT),
        ".pt conf 0.23 val": ("vd_mav_cl_s42_pt_c023", "val", CUTS_FLIGHT),
        ".pt conf 0.23 test-dev": ("vd_mav_cl_s42_pt_c023", "test-dev", CUTS_FLIGHT)}


def _record():
    return lf.load_tracker_choice()


def _sweep_rows():
    assert SWEEP.is_file(), f"{SWEEP} is committed evidence and must exist"
    return [__import__("json").loads(x) for x in SWEEP.read_text().splitlines() if x.strip()]


def _rows_for(rows, set_name, settings):
    det, split, pf = SETS[set_name]
    return [r for r in rows if r["detector"] == det and r["split"] == split and r["post_filter"] == pf
            and r["settings"] == settings and r["config"] == "botsort|reid-|gmc+"]


def _opt(name_start):
    return next(o for o in _record()["options"] if o["name"].startswith(name_start))


def test_tuned_yaml_equals_the_decision_record_and_is_stock_otherwise():
    import yaml
    cfg = yaml.safe_load((HERE / lf.TRACKER).read_text(encoding="utf-8"))
    rec = _record()
    assert {k: cfg[k] for k in rec["chosen"]} == rec["chosen"]
    assert rec["chosen"] == dict(track_high_thresh=0.4, new_track_thresh=0.4, track_low_thresh=0.05, match_thresh=0.9,
                                 track_buffer=60, fuse_score=True)                    # Michael 11-27
    assert rec["chosen"] == _opt("chosen")["settings"]
    stock_rec = _opt("stock")["settings"]
    for k in rec["chosen"]:                                       # fuse_score is the only one of the six left stock
        assert (cfg[k] == stock_rec[k]) == (k == "fuse_score"), k
    import pytest
    pytest.importorskip("ultralytics")
    import ultralytics
    stock = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml").read_text(encoding="utf-8"))
    assert set(cfg) == set(stock), "re-copy the installed ultralytics' botsort.yaml and change only the five values"
    assert {k: v for k, v in cfg.items() if k not in rec["chosen"] or k == "fuse_score"} == \
           {k: v for k, v in stock.items() if k not in rec["chosen"] or k == "fuse_score"}
    assert {k: stock[k] for k in stock_rec} == stock_rec          # the record's "stock" row is Ultralytics' file


def test_every_number_in_the_tracker_record_is_a_sweep_row():
    rows = _sweep_rows()
    assert len(rows) == 79
    rec = _record()
    assert RESULTS.parent / rec["evidence"]["sweep"] == SWEEP and RESULTS.parent / rec["evidence"]["summary"] == SWEEP_SUMMARY
    assert set(rec["sets"]) == set(SETS)
    for o in rec["options"]:
        for metric, path in (("class_avg_HOTA", "class_avg"), ("person_HOTA", "person")):
            for set_name in SETS:
                m = _rows_for(rows, set_name, o["settings"])
                if set_name not in o[metric]:
                    assert not m, f"{o['name']}: {set_name} was run, record leaves it out"
                    continue
                assert m, f"{o['name']}: no sweep row for {set_name}"
                got = {round(r["scores"][path]["HOTA"], 4) for r in m}      # real-GMC and replayed-GMC rows agree
                assert got == {o[metric][set_name]}, (o["name"], metric, set_name, got)
    # the script's own pick, as the summary states it
    summ = __import__("json").loads(SWEEP_SUMMARY.read_text())
    assert summ["chosen"]["settings"] == rec["sweep_auto_pick"]["settings"] == _opt("val-best")["settings"]
    assert round(summ["chosen"]["val_hota"], 4) == rec["sweep_auto_pick"]["val_class_avg_HOTA"] == _opt("val-best")["class_avg_HOTA"]["val"]
    val_runs = [r for r in rows if r["split"] == "val" and r["detector"] == "vd_mav_cl_s42"]
    assert max(round(r["scores"]["class_avg"]["HOTA"], 4) for r in val_runs) == rec["sweep_auto_pick"]["val_class_avg_HOTA"]


def test_the_reason_in_the_tracker_record_is_what_the_numbers_say():
    stock, chosen, best = (_opt(n)["class_avg_HOTA"] for n in ("stock", "chosen", "val-best"))
    assert round(best["val"] - chosen["val"], 4) == 0.0026                 # "+0.0026 on val"
    assert best["MOT20-train"] < stock["MOT20-train"]                       # "fell below default on MOT20"
    assert set(chosen) == set(stock) and all(chosen[k] > stock[k] for k in stock)   # "beats the default on every set"
    assert "0.0026" in _record()["reason"] and "0.3166" in _record()["reason"]


def test_live_and_replay_resolve_to_botsort_tuned():
    import replay
    live = lf.build_parser().parse_args([])
    rep = replay.build_parser().parse_args(["--source", "x"])
    assert live.tracker == rep.tracker == lf.TRACKER == "botsort_tuned.yaml"
    for a in (live, rep):
        t = lf.resolve_tracker(a.tracker)
        assert Path(t["path"]) == HERE / "botsort_tuned.yaml" and t["label"] == "botsort_tuned.yaml"
        assert lf.running_settings(a, True, precision="INT8")["tracker"] == "botsort_tuned.yaml"
    assert "match_thresh 0.9" in lf.tracker_line(lf.resolve_tracker(lf.TRACKER))
    stock = lf.resolve_tracker("botsort.yaml")                        # an Ultralytics built-in name, passed through
    assert stock == dict(path="botsort.yaml", label="botsort.yaml", settings=None)


def test_preflight_fails_on_stock_botsort_unless_off_decisions():
    import pytest
    import replay
    dec = lf.load_decisions()
    for program, parser, extra in (("live_follow", lf.build_parser(), []), ("replay", replay.build_parser(), ["--source", "x"])):
        a = parser.parse_args(extra + ["--tracker", "botsort.yaml"])
        r = lf.check_decisions(lf.running_settings(a, True, precision="INT8"), dec, {"replay"} if program == "replay" else set())
        assert [(x["name"], x["decided"], x["actual"]) for x in r["deviations"]] == \
               [("tracker", "botsort_tuned.yaml", "botsort.yaml")]
        with pytest.raises(SystemExit):
            lf.enforce_decisions(r, False, program=program)
        assert lf.enforce_decisions(r, True, program=program)["off_decisions"] is True


def test_preflight_stops_on_stock_botsort_end_to_end(tmp_path):
    """A real start (fake drone, INT8-like export folder, no --fake-detector) with --tracker botsort.yaml: the banner
    shows the precision read from the export, then the pre-flight check stops before anything flies."""
    model = make_ov_export(tmp_path)
    cmd = [sys.executable, str(HERE / "live_follow.py"), "--fake", "--headless", "--seconds", "2", "--model", str(model),
           "--tracker", "botsort.yaml", "--out", str(tmp_path / "out")]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=HERE)
    assert p.returncode != 0 and "STOP: not flying what was decided" in p.stderr
    assert "tracker: decided 'botsort_tuned.yaml', about to run 'botsort.yaml'" in p.stderr
    assert "detector_precision" not in p.stderr                        # the export says INT8: no deviation there
    assert "detector: INT8 OpenVINO, static 384x640" in p.stdout
    assert "tracker: botsort.yaml (Ultralytics built-in file, its stock settings)" in p.stdout
    assert not (tmp_path / "out").exists()


def test_edited_or_missing_tuned_yaml_stops(tmp_path, monkeypatch):
    import json
    import pytest
    rec = _record()
    rec["chosen"] = dict(rec["chosen"], match_thresh=0.975)           # the yaml no longer matches its record
    (tmp_path / "rec.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="differs from its decision record"):
        lf.resolve_tracker(lf.TRACKER, choice_path=tmp_path / "rec.json")
    with pytest.raises(SystemExit, match="decision record not found"):
        lf.resolve_tracker(lf.TRACKER, choice_path=tmp_path / "nope.json")
    monkeypatch.setattr(lf, "HERE", tmp_path)                         # no botsort_tuned.yaml next to the program
    with pytest.raises(SystemExit, match="tracker settings file not found"):
        lf.resolve_tracker(lf.TRACKER)


def test_drone_json_records_tracker_and_detector():
    run_fake("q@1", 2)
    info = _latest_drone_json()
    assert info["tracker"]["file"] == "botsort_tuned.yaml"
    assert {k: info["tracker"]["settings"][k] for k in _record()["chosen"]} == _record()["chosen"]
    assert info["detector"]["summary"].startswith("fake detector") and info["running"]["detector_precision"] == "none (--fake-detector)"


# ---- 11-27: detector precision read from the model files, printed, checked, logged ----

# metadata.yaml of the INT8 export that flies, verbatim (Drive jarvis_det/live_models/vd_mav_cl_s42_int8_openvino_model.zip)
REAL_INT8_META = """description: Ultralytics vd_mav_cl_s42 model trained on /content/data/arm_vd_mav_cl_3w.yaml
author: Ultralytics
date: '2026-10-29T21:20:16.587978'
version: 8.4.60
license: AGPL-3.0 License (https://ultralytics.com/license)
docs: https://docs.ultralytics.com
stride: 32
task: detect
batch: 1
imgsz:
- 384
- 640
names:
  0: person
  1: bicycle
  2: car
  3: van
  4: truck
  5: bus
  6: motorcycle
  7: three_wheeler
args:
  data: /content/data/int8_calib/calib.yaml
  batch: 1
  fraction: 1.0
  half: false
  int8: true
  dynamic: false
  nms: false
channels: 3
end2end: false
"""


def ov_xml(shape="1,3,384,640", n_fq=2, nncf=True):
    """A minimal OpenVINO IR with the parts detector_precision reads (the real one: 67 FakeQuantize layers)."""
    fq = "".join(f'<layer id="{i + 1}" name="fq{i}" type="FakeQuantize" version="opset1"><data levels="256" /></layer>'
                 for i in range(n_fq))
    rt = ('<rt_info><nncf><quantization><preset value="mixed" /><subset_size value="300" /></quantization>'
          '<version value="3.4.0" /></nncf></rt_info>') if nncf else "<rt_info />"
    return (f'<?xml version="1.0"?><net name="Model0" version="11"><layers><layer id="0" name="x" type="Parameter" '
            f'version="opset1"><data shape="{shape}" element_type="f32" /></layer>{fq}</layers>{rt}</net>')


def make_ov_export(root, meta=REAL_INT8_META, xml=None, name="vd_mav_cl_s42_int8_openvino_model"):
    d = Path(root) / name
    d.mkdir(parents=True)
    if meta is not None:
        (d / "metadata.yaml").write_text(meta)
    if xml is not False:
        (d / "vd_mav_cl_s42.xml").write_text(ov_xml() if xml is None else xml)
    return d


def test_precision_of_the_real_int8_export(tmp_path):
    d = make_ov_export(tmp_path)
    r = lf.detector_precision(d)
    assert r["precision"] == "INT8" and r["backend"] == "OpenVINO" and r["input"] == "static 384x640"
    assert r["summary"] == "INT8 OpenVINO, static 384x640"
    assert r["source"] == "metadata.yaml args.int8 = true; vd_mav_cl_s42.xml: 2 FakeQuantize layers"
    ex = r["export"]
    assert ex["ultralytics_version"] == "8.4.60" and ex["export_date"].startswith("2026-10-29")
    assert ex["input_shape"] == "1,3,384,640" and ex["imgsz"] == [384, 640] and ex["fake_quantize_layers"] == 2
    assert ex["calibration"] == dict(data="/content/data/int8_calib/calib.yaml", fraction=1.0,
                                     nncf=dict(preset="mixed", subset_size="300", nncf_version="3.4.0"))
    lines = lf.precision_lines(r, d)
    assert lines[0] == f"detector: INT8 OpenVINO, static 384x640  ({lf._rel(d, HERE)})"
    assert "calibration: data /content/data/int8_calib/calib.yaml (fraction 1.0), NNCF 3.4.0 preset mixed, subset_size 300" in lines[2]


def test_precision_other_exports_and_nothing_guessed(tmp_path):
    meta = REAL_INT8_META.replace("int8: true", "int8: false")
    fp32 = lf.detector_precision(make_ov_export(tmp_path / "a", meta, ov_xml(n_fq=0, nncf=False), "m_openvino_model"))
    assert fp32["summary"] == "FP32 OpenVINO, static 384x640"
    fp16 = lf.detector_precision(make_ov_export(tmp_path / "b", meta.replace("half: false", "half: true"),
                                                ov_xml(n_fq=0), "m_openvino_model"))
    assert fp16["precision"] == "FP16"
    newstyle = "args:\n  quantize: 8\n  dynamic: false\nimgsz: [384, 640]\n"         # ultralytics >= 8.4.2xx
    assert lf.detector_precision(make_ov_export(tmp_path / "c", newstyle))["precision"] == "INT8"
    # no metadata.yaml: the .xml's FakeQuantize layers are the only record; without them nothing is claimed
    no_meta = lf.detector_precision(make_ov_export(tmp_path / "d", None))
    assert no_meta["precision"] == "INT8" and "metadata.yaml records no precision" in no_meta["source"]
    bare = lf.detector_precision(make_ov_export(tmp_path / "e", None, False))
    assert bare["precision"] == lf.NR and bare["input"] == "input shape not recorded"
    assert bare["export"]["calibration"] == dict(data=lf.NR, fraction=lf.NR, nncf=lf.NR)
    # metadata and model disagree: never pick one
    bad = lf.detector_precision(make_ov_export(tmp_path / "f", REAL_INT8_META, ov_xml(n_fq=0)))
    assert bad["precision"].startswith("conflict")
    dyn = lf.detector_precision(make_ov_export(tmp_path / "g", REAL_INT8_META.replace("dynamic: false", "dynamic: true"),
                                               ov_xml(shape="?,3,?,?")))
    assert dyn["input"].startswith("dynamic input")
    assert lf.detector_precision("weights/vd_mav_cl_s42.pt")["summary"] == "FP32 PyTorch, imgsz 640"
    assert lf.detector_precision(tmp_path / "missing_int8_openvino_model")["precision"] == lf.NR
    assert lf.detector_precision("x", fake=True)["precision"] == "none (--fake-detector)"
    assert lf.precision_lines(lf.detector_precision("weights/vd_mav_cl_s42.pt"), "weights/vd_mav_cl_s42.pt")[0].endswith(
        "<-- NOT the INT8 model (decided 11-22)")


def test_precision_is_a_preflight_decision(tmp_path):
    import pytest
    dec = lf.load_decisions()
    assert dec["detector_precision"]["value"] == "INT8" and dec["detector_precision"]["skip_with"] == ["--fake-detector"]
    a = lf.build_parser().parse_args([])
    good = lf.detector_precision(make_ov_export(tmp_path / "int8"))["precision"]
    assert not [x for x in lf.check_decisions(lf.running_settings(a, True, precision=good), dec, set())["deviations"]
                if x["name"] == "detector_precision"]
    for bad in ("FP32", lf.NR, "conflict (x)"):
        r = lf.check_decisions(lf.running_settings(a, True, precision=bad), dec, set())
        assert [x["name"] for x in r["deviations"]] == ["detector_precision"]
        with pytest.raises(SystemExit):
            lf.enforce_decisions(r, False)
    # read from the files when not passed in: the default model's precision is whatever its folder says
    assert lf.running_settings(a, True)["detector_precision"] == lf.detector_precision(a.model)["precision"]
