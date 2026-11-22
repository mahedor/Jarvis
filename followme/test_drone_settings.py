"""drone_settings.py: profile diff/apply logic with a stand-in drone (no Olympe needed)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drone_settings as ds  # noqa: E402


def test_diff_reports_only_known_different_values():
    prof = dict(max_tilt_deg=10.0, max_rotation_speed_dps=10.0, max_altitude_m=None)
    cur = dict(max_tilt_deg=10.0, max_rotation_speed_dps=45.0, max_altitude_m=30.0)
    assert ds.diff(prof, cur) == {"max_rotation_speed_dps": (45.0, 10.0)}
    assert ds.diff(prof, dict(max_tilt_deg=None)) == {}          # unknown on the drone -> not reported


def test_load_profile(tmp_path):
    p = tmp_path / "drone_profile.json"
    assert ds.load_profile(p) is None
    p.write_text(json.dumps(dict(source="ANAFI-G134762", settings=dict(max_tilt_deg=15.0))))
    assert ds.load_profile(p) == dict(max_tilt_deg=15.0)


def test_apply_sends_every_known_value():
    sent = []

    class Exp:
        def wait(self, _timeout=None): return self
        def success(self): return True

    class FakeDrone:
        def __call__(self, msg):
            sent.append(msg); return Exp()

    class Msg:
        def __init__(self, *a, **k): self.a, self.k = a, k

    orig = ds._msg
    ds._msg = lambda m, n: Msg
    try:
        res = ds.apply(FakeDrone(), dict(max_tilt_deg=12.0, max_rotation_speed_dps=None, gimbal_max_pitch_speed_dps=10.0))
    finally:
        ds._msg = orig
    assert res == {"max_tilt_deg": "ok", "gimbal_max_pitch_speed_dps": "ok"} and len(sent) == 2
