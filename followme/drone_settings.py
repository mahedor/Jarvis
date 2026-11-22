"""Keep both drones' flight "sensitivity" settings identical (Michael 2026-11-22).

Why it matters: live_follow sends stick commands as a PERCENT of each drone's own configured maximums (max tilt,
max rotation speed, max vertical speed, ...). If drone B's maximums differ from drone A's, the same 20 % command
moves them differently. These settings live ON each drone and survive power-off, so we keep one profile file in
the repo (drone_profile.json) and push it to whichever drone we connect to.

  python drone_settings.py                      # READ: print this drone's settings (sends nothing)
  python drone_settings.py --save               # READ + save them as drone_profile.json (do this on drone A)
  python drone_settings.py --apply              # APPLY drone_profile.json to this drone, then read back
live_follow.py also applies drone_profile.json automatically on every connect (unless --no-profile), so a change
to the profile reaches both drones the next time each one connects.
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROFILE = Path(__file__).resolve().parent / "drone_profile.json"

# name -> (state message module, state message, setter module, setter, unit)
SETTINGS = {
    "max_tilt_deg": ("olympe.messages.ardrone3.PilotingSettingsState", "MaxTiltChanged",
                     "olympe.messages.ardrone3.PilotingSettings", "MaxTilt", "deg"),
    "max_rotation_speed_dps": ("olympe.messages.ardrone3.SpeedSettingsState", "MaxRotationSpeedChanged",
                               "olympe.messages.ardrone3.SpeedSettings", "MaxRotationSpeed", "deg/s"),
    "max_vertical_speed_mps": ("olympe.messages.ardrone3.SpeedSettingsState", "MaxVerticalSpeedChanged",
                               "olympe.messages.ardrone3.SpeedSettings", "MaxVerticalSpeed", "m/s"),
    "max_pitch_roll_rotation_speed_dps": ("olympe.messages.ardrone3.SpeedSettingsState",
                                          "MaxPitchRollRotationSpeedChanged",
                                          "olympe.messages.ardrone3.SpeedSettings", "MaxPitchRollRotationSpeed",
                                          "deg/s"),
    "max_altitude_m": ("olympe.messages.ardrone3.PilotingSettingsState", "MaxAltitudeChanged",
                       "olympe.messages.ardrone3.PilotingSettings", "MaxAltitude", "m"),
    "max_distance_m": ("olympe.messages.ardrone3.PilotingSettingsState", "MaxDistanceChanged",
                       "olympe.messages.ardrone3.PilotingSettings", "MaxDistance", "m"),
}


def _msg(module, name):
    return getattr(__import__(module, fromlist=[name]), name)


def read(drone):
    """-> {name: value or None} for every setting, plus the gimbal's max pitch speed."""
    out = {}
    for name, (sm, sn, *_rest) in SETTINGS.items():
        try:
            out[name] = round(float(drone.get_state(_msg(sm, sn))["current"]), 3)
        except Exception:
            out[name] = None
    try:
        g = drone.get_state(_msg("olympe.messages.gimbal", "max_speed"))
        g = g[0] if 0 in g else g
        out["gimbal_max_pitch_speed_dps"] = round(float(g["current_pitch"]), 3)
    except Exception:
        out["gimbal_max_pitch_speed_dps"] = None
    return out


def diff(profile, current):
    """Settings whose current value differs from the profile (both known) -> {name: (current, wanted)}."""
    return {k: (current.get(k), v) for k, v in profile.items()
            if v is not None and current.get(k) is not None and abs(float(current[k]) - float(v)) > 1e-3}


def apply(drone, profile):
    """Push every known profile value to the drone. -> {name: "ok" | "failed: ..."}"""
    res = {}
    for name, value in profile.items():
        if value is None:
            continue
        try:
            if name == "gimbal_max_pitch_speed_dps":
                ok = drone(_msg("olympe.messages.gimbal", "set_max_speed")(
                    gimbal_id=0, yaw=0.0, pitch=float(value), roll=0.0)).wait(_timeout=5).success()
            else:
                _, _, setm, setn, _ = SETTINGS[name]
                ok = drone(_msg(setm, setn)(float(value))).wait(_timeout=5).success()
            res[name] = "ok" if ok else "failed: no acknowledgement"
        except Exception as e:
            res[name] = f"failed: {type(e).__name__}: {e}"
    return res


def load_profile(path=PROFILE):
    p = Path(path)
    return json.loads(p.read_text()).get("settings", {}) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.42.1")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--save", action="store_true", help="save this drone's settings as the shared profile")
    g.add_argument("--apply", action="store_true", help="push the shared profile to this drone")
    args = ap.parse_args()
    import olympe
    olympe.log.update_config({"loggers": {"olympe": {"level": "CRITICAL"}}})
    drone = olympe.Drone(args.ip)
    if not drone.connect():
        sys.exit(f"Could not connect to {args.ip} (on this drone's Wi-Fi?)")
    time.sleep(3)
    try:
        name = None
        try:
            name = drone.get_state(_msg("olympe.messages.common.SettingsState", "ProductNameChanged"))["name"]
        except Exception:
            pass
        cur = read(drone)
        print(f"Drone {name}:")
        for k, v in cur.items():
            print(f"  {k:36s} {v}")
        if args.save:
            PROFILE.write_text(json.dumps(dict(source=name, saved=time.strftime("%Y-%m-%d %H:%M"), settings=cur),
                                          indent=1))
            print(f"saved -> {PROFILE.name} (commit it so both laptops use the same file)")
        prof = load_profile()
        if prof is None:
            print("no drone_profile.json yet: run with --save on drone A")
        else:
            d = diff(prof, cur)
            print("matches the shared profile" if not d else
                  "DIFFERENT from the shared profile: " + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in d.items()))
            if args.apply and d:
                for k, r in apply(drone, prof).items():
                    print(f"  set {k:32s} {r}")
                time.sleep(1)
                left = diff(prof, read(drone))
                print("now matches the profile" if not left else f"STILL DIFFERENT: {left}")
    finally:
        drone.disconnect()


if __name__ == "__main__":
    main()
