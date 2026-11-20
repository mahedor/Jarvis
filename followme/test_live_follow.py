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
    L = lf.FollowLogic({0}, max_speed=5, lost_s=1.0, search_yaw=50); now = time.time()
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


def test_ignores_non_person_and_relocks_near_last_position():
    L = logic(); now = time.time()
    car = (100, 100, 300, 200, 7, 2, 0.9)
    cmd, info = L.step([car], now, now, W, H, TEL)
    assert L.target_id is None                                      # never follows a car
    L.step([person(900, tid=1)], now, now, W, H, TEL)
    assert L.target_id == 1
    # track ID changes (tracker lost and re-found): picks the person nearest the last position, not the biggest
    L.step([person(200, tid=5, w=200, h=400), person(910, tid=9)], now + 0.2, now + 0.2, W, H, TEL)
    assert L.target_id == 9


def test_click_pick_sticks():
    L = logic(); now = time.time()
    L.pick(5)
    L.step([person(200, tid=5), person(900, tid=1, w=300, h=500)], now, now, W, H, TEL)
    assert L.target_id == 5 and L.picked_by == "you"


def run_fake(script, seconds, extra=()):
    out = HERE / "_test_follow_out"
    cmd = [sys.executable, str(HERE / "live_follow.py"), "--fake", "--fake-detector", "--headless",
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
