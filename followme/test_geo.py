"""Tests for geo.py (one-drone map position). Pure math, no drone."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo  # noqa: E402
import replay  # noqa: E402

W, H = 1280, 720


def test_center_pixel_matches_simple_trig():
    # camera tilted 45 deg down from 10 m: the image centre hits the ground 10 m ahead, straight on
    f, r = geo.ground_offset(W / 2, H / 2, W, H, 10.0, -45.0)
    assert f == pytest.approx(10.0, abs=1e-6) and r == pytest.approx(0.0, abs=1e-9)


def test_forward_distance_matches_existing_controller():
    # the vertical part must agree with replay.estimate_distance's 'angle' method (same FOV assumption)
    box = (600, 300, 680, 600)
    tel = dict(altitude_takeoff_m=4.0, gimbal_pitch_deg=-20.0)
    d, method = replay.estimate_distance(box, H, tel)
    assert method == "angle"
    f, r = geo.ground_offset(640, 600, W, H, 4.0, -20.0)
    assert f == pytest.approx(d, rel=1e-6) and r == pytest.approx(0.0, abs=1e-9)


def test_right_side_pixel_goes_right_and_scales_with_distance():
    f, r = geo.ground_offset(1100, H / 2, W, H, 5.0, -30.0)
    assert r > 0
    x_n = (1100 - W / 2) / (W / 2) * math.tan(math.radians(geo.HFOV_DEG / 2))
    slant = 5.0 / math.sin(math.radians(30))            # distance along the centre ray to the ground
    assert r == pytest.approx(slant * x_n, rel=1e-6)    # lateral offset = slant range x tan(angle right)
    fl, rl = geo.ground_offset(180, H / 2, W, H, 5.0, -30.0)
    assert rl == pytest.approx(-r, rel=1e-9) and fl == pytest.approx(f, rel=1e-9)


def test_horizon_and_missing_telemetry_give_none():
    assert geo.ground_offset(W / 2, 0, W, H, 5.0, 0.0) is None          # above the horizon
    assert geo.ground_offset(W / 2, H / 2, W, H, None, -30.0) is None
    assert geo.ground_offset(W / 2, H / 2, W, H, 5.0, None) is None
    assert geo.ground_offset(W / 2, H / 2, W, H, 0.1, -30.0) is None    # on the ground


@pytest.mark.parametrize("yaw,exp", [(0, (10, 2)), (90, (-2, 10)), (180, (-10, -2)), (270, (2, -10))])
def test_rotate_to_north_east(yaw, exp):
    n, e = geo.rotate_to_ne(10.0, 2.0, yaw)
    assert (n, e) == pytest.approx(exp, abs=1e-9)


def test_gps_to_ne_small_offsets():
    lat0, lon0 = 37.0, -122.0
    n, e = geo.gps_to_ne(lat0 + 0.0001, lon0, lat0, lon0)
    assert n == pytest.approx(11.13, abs=0.02) and e == pytest.approx(0.0, abs=1e-9)
    n, e = geo.gps_to_ne(lat0, lon0 + 0.0001, lat0, lon0)
    assert e == pytest.approx(11.13 * math.cos(math.radians(37)), abs=0.02)


def test_gps_invalid_values():
    assert not geo.gps_valid(500.0, 500.0) and not geo.gps_valid(None, 1.0) and not geo.gps_valid(0.0, 0.0)
    assert geo.gps_valid(37.1, -122.2)


def test_mapframe_origin_and_person_position():
    m = geo.MapFrame()
    tel = dict(altitude_takeoff_m=10.0, gimbal_pitch_deg=-45.0, drone_yaw_deg=90.0, gps_lat=37.0, gps_lon=-122.0)
    box = (600, 300, 680, H / 2)                         # feet at the image centre -> 10 m ahead
    out = m.update(box, W, H, tel)
    assert m.origin == (37.0, -122.0)
    assert out["drone_n_m"] == 0 and out["drone_e_m"] == 0
    assert out["person_fwd_m"] == pytest.approx(10.0) and out["person_de_m"] == pytest.approx(10.0)   # facing east
    assert out["person_n_m"] == pytest.approx(0.0, abs=0.01) and out["person_e_m"] == pytest.approx(10.0)
    # drone moves ~11 m north: the person is still placed relative to the moved drone
    out = m.update(box, W, H, dict(tel, gps_lat=37.0001))
    assert out["drone_n_m"] == pytest.approx(11.13, abs=0.02)
    assert out["person_n_m"] == pytest.approx(11.13, abs=0.02) and out["person_e_m"] == pytest.approx(10.0)


def test_mapframe_without_gps_still_gives_relative_position():
    out = geo.MapFrame().update((600, 300, 680, H / 2), W, H,
                                dict(altitude_takeoff_m=10.0, gimbal_pitch_deg=-45.0, drone_yaw_deg=0.0))
    assert out["person_dn_m"] == pytest.approx(10.0) and out["person_n_m"] is None


def test_mapframe_feet_cut_off_gives_no_position():
    out = geo.MapFrame().update((600, 300, 680, H - 1), W, H,
                                dict(altitude_takeoff_m=10.0, gimbal_pitch_deg=-45.0, drone_yaw_deg=0.0))
    assert out["person_fwd_m"] is None


def test_calibrated_path_agrees_with_pinhole_without_distortion():
    pytest.importorskip("cv2")
    fx = (W / 2) / math.tan(math.radians(geo.HFOV_DEG / 2))
    fy = (H / 2) / math.tan(math.radians(geo.VFOV_DEG / 2))
    calib = dict(K=[[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dist=[0, 0, 0, 0, 0], size=(W, H))
    a = geo.ground_offset(1000, 600, W, H, 6.0, -25.0)
    b = geo.ground_offset(1000, 600, W, H, 6.0, -25.0, calib)
    assert b == pytest.approx(a, rel=1e-6)
    # calibrated at 1920x1080, used at 1280x720: intrinsics are scaled
    calib2 = dict(K=[[fx * 1.5, 0, 960], [0, fy * 1.5, 540], [0, 0, 1]], dist=[0, 0, 0, 0, 0], size=(1920, 1080))
    assert geo.ground_offset(1000, 600, W, H, 6.0, -25.0, calib2) == pytest.approx(a, rel=1e-6)
