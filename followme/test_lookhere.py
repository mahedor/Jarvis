"""Tests for lookhere.py: shared-spot geometry, the look/go state machine, the file link, and a headless worker run."""
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo  # noqa: E402
import lookhere as lh  # noqa: E402
from test_live_follow import run_fake  # noqa: E402

LAT0, LON0 = 37.0, -122.0
W, H = 1280, 720


def at(n, e, yaw=0.0, alt=10.0, gimbal=-45.0):
    lat, lon = lh.ne_to_gps(n, e, LAT0, LON0)
    return dict(gps_lat=lat, gps_lon=lon, drone_yaw_deg=yaw, altitude_takeoff_m=alt, gimbal_pitch_deg=gimbal)


def test_ne_to_gps_inverts_gps_to_ne():
    lat, lon = lh.ne_to_gps(12.5, -7.25, LAT0, LON0)
    n, e = geo.gps_to_ne(lat, lon, LAT0, LON0)
    assert abs(n - 12.5) < 1e-6 and abs(e + 7.25) < 1e-6


def test_aim_bearing_distance_and_tilt():
    tgt = lh.ne_to_gps(0, 10, LAT0, LON0)                        # 10 m due east
    a = lh.aim(at(0, 0, yaw=0, alt=10), *tgt)
    assert abs(a["bearing_deg"] - 90) < 0.01 and abs(a["dist_m"] - 10) < 0.01
    assert abs(a["yaw_err_deg"] - 90) < 0.01                       # nose north, spot to the right
    assert abs(a["gimbal_pitch_deg"] + 45) < 0.01                  # 10 m up, 10 m away -> 45 deg down
    assert abs(lh.aim(at(0, 0, yaw=350), *tgt)["yaw_err_deg"] - 100) < 0.01   # wraps through north


def test_overseer_click_to_worker_aim():
    # overseer facing east, 15 m up, camera 45 deg down: the image centre is 15 m east of it on the ground
    spot = lh.click_to_gps(W / 2, H / 2, W, H, at(0, 0, yaw=90, alt=15, gimbal=-45))
    n, e = geo.gps_to_ne(*spot, LAT0, LON0)
    assert abs(n) < 0.05 and abs(e - 15) < 0.05
    a = lh.aim(at(20, 0, yaw=0, alt=5), *spot)                     # worker 20 m north of the overseer
    assert abs(a["bearing_deg"] - math.degrees(math.atan2(15, -20)) % 360) < 0.1
    assert abs(a["dist_m"] - 25) < 0.05


def test_click_needs_gps_heading_and_ground():
    assert lh.click_to_gps(W / 2, H / 2, W, H, dict(at(0, 0), gps_lat=500.0, gps_lon=500.0)) is None
    assert lh.click_to_gps(W / 2, H / 2, W, H, dict(at(0, 0), drone_yaw_deg=None)) is None
    assert lh.click_to_gps(W / 2, 0, W, H, at(0, 0, gimbal=0)) is None   # above the horizon


def worker(target_ne, mode, **tel):
    lg = lh.LookGo(); now = time.time()
    lg.set_target(*lh.ne_to_gps(*target_ne, LAT0, LON0), now); assert lg.start(mode)
    return lg, lg.step(at(0, 0, **tel), now)


def test_look_turns_toward_the_spot_and_never_translates():
    _, (cmd, gp, info) = worker((0, 20), "look")                   # spot to the right
    assert 0 < cmd["yaw"] <= lh.LOOK_MAX_YAW and cmd["pitch"] == cmd["roll"] == cmd["gaz"] == 0
    assert gp < 0 and info["state"] == "LOOK: turning"
    _, (cmd, _, _) = worker((0, -20), "look")                      # spot to the left
    assert cmd["yaw"] < 0
    _, (cmd, _, info) = worker((20, 0.5), "look")                  # almost dead ahead
    assert cmd["yaw"] == 0 and info["state"] == "LOOKING (aligned)"


def test_go_turns_first_then_flies_slowly_and_stops_short():
    _, (cmd, _, info) = worker((0, 20), "go")                      # 90 deg off: turn only
    assert cmd["pitch"] == 0 and cmd["yaw"] > 0 and "turning" in info["state"]
    _, (cmd, _, info) = worker((30, 0), "go")                      # ahead, far: fly, capped, height kept
    assert 0 < cmd["pitch"] <= lh.GO_MAX_FWD and cmd["gaz"] == 0 and "flying" in info["state"]
    lg, (cmd, _, info) = worker((lh.STOP_M - 1, 0), "go")          # already close: hover, switch to look
    assert cmd["pitch"] == 0 and info["event"] == "arrived" and lg.mode == "look"


def test_go_refuses_far_spots_and_hovers_without_gps():
    lg, (cmd, _, info) = worker((lh.GO_MAX_DIST_M + 10, 0), "go")
    assert cmd["pitch"] == 0 and info["event"] == "go_refused_far" and lg.mode == "look"
    lg = lh.LookGo(); now = time.time()
    lg.set_target(LAT0, LON0, now); lg.start("go")
    cmd, gp, info = lg.step(dict(at(0, 0), gps_lat=500.0, gps_lon=500.0), now)
    assert cmd == dict(roll=0, pitch=0, yaw=0, gaz=0) and gp is None and info["event"] == "no_gps"


def test_stale_or_missing_spot_hovers():
    lg = lh.LookGo()
    assert not lg.start("look")                                    # nothing shared yet
    lg.set_target(LAT0, LON0, time.time() - lh.TARGET_MAX_AGE_S - 1); lg.start("look")
    cmd, _, info = lg.step(at(0, 0), time.time())
    assert cmd["yaw"] == cmd["pitch"] == 0 and info["event"] == "target_stale" and lg.mode is None


def test_file_link_delivers_each_spot_once(tmp_path):
    f = tmp_path / "spot.json"
    a, b = lh.TargetLink(f, drone="droneA"), lh.TargetLink(f, drone="droneB", listen_to="droneA")
    assert b.poll() is None
    a.publish(37.1, -122.1)
    assert b.poll() == (37.1, -122.1) and b.poll() is None
    a.publish(37.2, -122.2)
    assert b.poll() == (37.2, -122.2)


def test_headless_worker_looks_then_space_takes_control(tmp_path):
    spot = tmp_path / "spot.json"
    lat, lon = lh.ne_to_gps(0, 15, 37.0, -122.0)                   # fake drone sits at 37.0, -122.0, nose north
    spot.write_text(json.dumps(dict(lat=lat, lon=lon, t=time.time() + 5, src="droneA")))
    out, rows = run_fake("t@0.5,v@1.5, @4,q@5", 6, ("--name", "droneB", "--listen", "droneA",
                                                      "--share-file", str(spot)))
    assert "received spot" in out and "LOOK at shared spot" in out
    look = [r for r in rows if r["mode"] == "LOOKGO"]
    assert look and all(float(r["yaw"]) > 0 and float(r["pitch"]) == 0 for r in look)   # fake heading never changes
    assert rows[-1]["mode"] == "MANUAL"


def test_headless_overseer_shares_its_tracked_target(tmp_path):
    spot = tmp_path / "spot.json"
    # fake drone: 2 m up, ball below the image centre; tilt the camera down, lock (auto-lock), share with O
    out, rows = run_fake("b@0.3,b@0.5,b@0.7,o@1.5,q@4", 5, ("--name", "droneA", "--share-file", str(spot)))
    assert "SHARING the tracked target" in out
    m = json.loads(spot.read_text())
    assert m["src"] == "droneA" and abs(m["lat"] - 37.0) < 0.001 and abs(m["lon"] + 122.0) < 0.001
    assert time.time() - m["t"] < 10                                # re-shared during the run
