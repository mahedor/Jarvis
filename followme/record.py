#!/usr/bin/env python3
"""Fly by keyboard and RECORD the live video + per-frame telemetry (for the replay tool and the test set).

Run in WSL:  source ~/jarvis-drone/bin/activate && python record.py
Output (in ~/recordings/<timestamp>/):  video_000.mp4, video_001.mp4, ...  +  telemetry.csv (one row per frame)
Video is split into 60 s chunks so a crash or dead battery only loses the last chunk, not the whole flight.

Keys (hold to keep moving; releasing for 0.3 s = hover):
  T takeoff   L land   Q quit (lands first if flying)   SPACE hover now
  W/S forward/back   A/D left/right   R/F up/down   Z/C yaw left/right   (arrow keys also work for yaw/up/down)
  1/2/3 speed 20/35/50 %     G/B gimbal tilt up/down 5 deg     V start/stop recording (starts ON)

Telemetry columns: frame, t_s, altitude_takeoff_m, gimbal_pitch_deg, drone_roll/pitch/yaw_deg,
speed_x/y/z_mps, battery_pct, flying_state, gps_lat/lon/alt_m (+accuracy, 500 = no fix), recording.
--fake runs without a drone (synthetic frames) to check that writing works.
"""
import argparse
import csv
import math
import queue
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HOLD = 0.3
SPEEDS = {ord("1"): 20, ord("2"): 35, ord("3"): 50}
ARROWS = {65362: "up", 65364: "down", 65361: "yaw_l", 65363: "yaw_r"}   # cv2.waitKeyEx codes on Linux/WSLg


class FakeDrone:
    """Stand-in so the recording/CSV path can be tested without hardware."""
    def __init__(self):
        self.t0 = time.time(); self.pitch = 0.0
    def frame(self):
        img = np.full((720, 1280, 3), 60, np.uint8)
        x = int(640 + 300 * math.sin(time.time() - self.t0))
        cv2.circle(img, (x, 400), 40, (0, 200, 255), -1)
        return img
    def telemetry(self):
        return dict(altitude_takeoff_m=2.0, gimbal_pitch_deg=self.pitch, drone_roll_deg=0, drone_pitch_deg=0,
                    drone_yaw_deg=0, speed_x_mps=0, speed_y_mps=0, speed_z_mps=0, battery_pct=99, flying_state="landed",
                    gps_lat=37.0, gps_lon=-122.0, gps_alt_m=10.0, gps_lat_acc_m=2.0, gps_lon_acc_m=2.0)


# Michael's two ANAFI 4Ks (11-22). Matched on the end of the Wi-Fi/product name (= end of the serial).
DRONES = {"G134762": "droneA", "J102372": "droneB"}


def nickname(product_name):
    for tail, nick in DRONES.items():
        if product_name and product_name.endswith(tail):
            return nick
    return product_name or "unknown"


def connect_real(ip):
    import olympe
    olympe.log.update_config({"loggers": {"olympe": {"level": "WARNING"}}})   # hide the INFO flood
    from olympe.messages.ardrone3.Piloting import Landing, PCMD, TakeOff
    from olympe.messages.ardrone3.PilotingState import (AltitudeChanged, AttitudeChanged, FlyingStateChanged,
                                                        SpeedChanged)
    from olympe.messages.common.CommonState import BatteryStateChanged
    # GPS (2026-11-08, one-drone map position). Newer firmware sends GpsLocationChanged (with accuracies), older
    # PositionChanged; both report 500 for "no fix". Imported defensively so a missing message never breaks flying.
    try:
        from olympe.messages.ardrone3.PilotingState import GpsLocationChanged
    except ImportError:
        GpsLocationChanged = None
    try:
        from olympe.messages.ardrone3.PilotingState import PositionChanged
    except ImportError:
        PositionChanged = None
    from olympe.messages.gimbal import attitude, set_target

    drone = olympe.Drone(ip)
    if not drone.connect():
        raise SystemExit("Could not connect (drone Wi-Fi? Skycontroller off?)")
    frames = queue.Queue(maxsize=4)
    CONVERT = {olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420, olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12}

    def raw_cb(yuv_frame):
        yuv_frame.ref()
        try:
            frames.put_nowait(yuv_frame)
        except queue.Full:
            yuv_frame.unref()

    def flush_cb(stream):
        # Olympe calls this when the stream is flushed/reset. Frames we still hold must be released,
        # or the video pipeline stalls (likely cause of the 2026-10-25 frozen window).
        while not frames.empty():
            try:
                frames.get_nowait().unref()
            except queue.Empty:
                break
        return True

    drone.streaming.set_callbacks(raw_cb=raw_cb, flush_raw_cb=flush_cb)
    drone.streaming.start()

    def get(msg):
        try:
            return drone.get_state(msg)
        except Exception:
            return None

    def next_frame(timeout=1.0):
        try:
            f = frames.get(timeout=timeout)
        except queue.Empty:
            return None
        try:
            return cv2.cvtColor(f.as_ndarray(), CONVERT[f.format()])
        finally:
            f.unref()

    def telemetry():
        alt, att, spd, bat, fs = get(AltitudeChanged), get(AttitudeChanged), get(SpeedChanged), get(BatteryStateChanged), get(FlyingStateChanged)
        g = get(attitude)
        g = g[0] if isinstance(g, dict) and 0 in g else g
        deg = lambda v: math.degrees(v) if v is not None else None
        return dict(altitude_takeoff_m=alt["altitude"] if alt else None,
                    gimbal_pitch_deg=g.get("pitch_absolute") if isinstance(g, dict) else None,
                    drone_roll_deg=deg(att["roll"]) if att else None, drone_pitch_deg=deg(att["pitch"]) if att else None,
                    drone_yaw_deg=deg(att["yaw"]) if att else None,
                    speed_x_mps=spd["speedX"] if spd else None, speed_y_mps=spd["speedY"] if spd else None,
                    speed_z_mps=spd["speedZ"] if spd else None, battery_pct=bat["percent"] if bat else None,
                    flying_state=getattr(fs["state"], "name", fs["state"]) if fs else None, **gps())

    def gps():
        g = get(GpsLocationChanged) if GpsLocationChanged else None
        if not g or abs(g.get("latitude", 500.0) - 500.0) < 1e-6:
            g = get(PositionChanged) if PositionChanged else g
        g = g or {}
        return dict(gps_lat=g.get("latitude"), gps_lon=g.get("longitude"), gps_alt_m=g.get("altitude"),
                    gps_lat_acc_m=g.get("latitude_accuracy"), gps_lon_acc_m=g.get("longitude_accuracy"))

    def pcmd(roll=0, pitch=0, yaw=0, gaz=0):
        flag = 1 if (roll or pitch) else 0
        drone(PCMD(flag, int(roll), int(pitch), int(yaw), int(gaz), 0))

    def gimbal(pitch_deg):
        drone(set_target(gimbal_id=0, control_mode="position", yaw_frame_of_reference="none", yaw=0.0,
                         pitch_frame_of_reference="absolute", pitch=float(pitch_deg),
                         roll_frame_of_reference="none", roll=0.0))

    def takeoff():
        drone(TakeOff())

    def land():
        drone(Landing()).wait(_timeout=30)

    def close():
        drone.streaming.stop(); drone.disconnect()

    def identity():
        """Which physical drone this is: Wi-Fi/product name, serial, firmware, and our nickname (droneA/droneB)."""
        from olympe.messages.common.SettingsState import ProductNameChanged, ProductVersionChanged
        try:
            from olympe.messages.common.SettingsState import ProductSerialHighChanged, ProductSerialLowChanged
        except ImportError:
            ProductSerialHighChanged = ProductSerialLowChanged = None
        n, v = get(ProductNameChanged), get(ProductVersionChanged)
        hi = get(ProductSerialHighChanged) if ProductSerialHighChanged else None
        lo = get(ProductSerialLowChanged) if ProductSerialLowChanged else None
        name = n["name"] if n else None
        return dict(drone_product_name=name, drone_nickname=nickname(name),
                    drone_serial=((hi or {}).get("high", "") + (lo or {}).get("low", "")) or None,
                    drone_firmware=v.get("software") if v else None)

    def apply_profile():
        """Push followme/drone_profile.json (shared max tilt / rotation / vertical speed ...) to this drone."""
        import drone_settings
        prof = drone_settings.load_profile()
        if prof is None:
            return "no drone_profile.json yet (run drone_settings.py --save on drone A)"
        d = drone_settings.diff(prof, drone_settings.read(drone))
        if not d:
            return "drone settings match drone_profile.json"
        res = drone_settings.apply(drone, prof)
        bad = {k: v for k, v in res.items() if v != "ok"}
        return f"applied drone_profile.json ({len(d)} changed)" + (f"; FAILED: {bad}" if bad else "")

    return dict(next_frame=next_frame, telemetry=telemetry, pcmd=pcmd, gimbal=gimbal, takeoff=takeoff,
                land=land, close=close, apply_profile=apply_profile, identity=identity)


def connect_fake(stall=(0, 0)):
    fd = FakeDrone()
    def next_frame(timeout=1.0):
        el = time.time() - fd.t0
        if stall[0] <= el < stall[1]:                 # simulate the video stream stalling
            time.sleep(timeout); return None
        time.sleep(1 / 30); return fd.frame()
    def gimbal(p): fd.pitch = p
    noop = lambda *a, **k: None
    ident = lambda: dict(drone_product_name="FAKE", drone_nickname="fake", drone_serial=None, drone_firmware=None)
    return dict(next_frame=next_frame, telemetry=fd.telemetry, pcmd=noop, gimbal=gimbal, takeoff=noop, land=noop,
                close=noop, identity=ident)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.42.1")
    ap.add_argument("--out", default=str(Path.home() / "recordings"))
    ap.add_argument("--fps", type=float, default=30.0, help="video file fps (ANAFI stream is ~30)")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--fake-stall", type=float, nargs=2, default=(0, 0), help="(testing) stall video from t1 to t2 s")
    ap.add_argument("--seconds", type=float, default=0, help="auto-stop after N seconds (0 = until Q)")
    ap.add_argument("--chunk", type=float, default=60.0, help="seconds per video file")
    args = ap.parse_args()

    d = connect_fake(tuple(args.fake_stall)) if args.fake else connect_real(args.ip)
    tel = d["telemetry"]()
    if not args.fake and tel["battery_pct"] is not None and tel["battery_pct"] < 30:
        print(f"Battery {tel['battery_pct']}% < 30%: charge first."); d["close"](); return

    out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    writer, csv_f, csv_w = None, open(out / "telemetry.csv", "w", newline=""), None
    recording, speed, gimbal_pitch = True, 20, 0.0
    held, last_key_t, flying, n, t0 = None, 0.0, False, 0, time.time()
    chunk, chunk_frames, name = 0, 0, None
    img, last_frame_t = None, time.time()
    print(f"Recording to {out}  (V toggles recording, Q quits)")
    try:
        while True:
            new = d["next_frame"](timeout=0.03)       # never block: keys + stick commands must keep flowing
            now = time.time()
            if new is not None:
                img, last_frame_t = new, now
            elif img is None:
                img = np.zeros((720, 1280, 3), np.uint8)
            video_ok = (now - last_frame_t) < 1.0
            t = now - t0
            tel = d["telemetry"]()
            if not recording and writer is not None:
                writer.release(); writer = None            # pausing closes the chunk cleanly
            if recording and new is not None:          # only write real frames
                per_chunk = int(args.chunk * args.fps)
                if writer is None or chunk_frames >= per_chunk:
                    if writer: writer.release()
                    h, w = img.shape[:2]
                    name = f"video_{chunk:03d}.mp4"; chunk += 1; chunk_frames = 0
                    writer = cv2.VideoWriter(str(out / name), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                writer.write(img); chunk_frames += 1
                row = {"frame": n, "t_s": round(t, 3), "video_file": name, "frame_in_file": chunk_frames - 1, **tel, "recording": 1}
                if csv_w is None:
                    csv_w = csv.DictWriter(csv_f, fieldnames=list(row)); csv_w.writeheader()
                csv_w.writerow(row); csv_f.flush(); n += 1
            show = img.copy()
            if not video_ok:
                cv2.rectangle(show, (0, 300), (show.shape[1], 420), (0, 0, 0), -1)
                cv2.putText(show, "NO VIDEO - controls still work: L = land, SPACE = hover", (30, 370),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            alt = tel["altitude_takeoff_m"]; gp = tel["gimbal_pitch_deg"]
            cv2.putText(show, f"{'REC' if recording else 'paused'} {n} fr | speed {speed}% | alt {alt if alt is None else round(alt,1)} m | "
                              f"gimbal {gp if gp is None else round(gp,1)} | bat {tel['battery_pct']}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (200, 200, 200), 2)
            cv2.imshow("record", show)
            k = cv2.waitKeyEx(1)
            if k != -1:
                kc = k & 0xFF if k < 256 else k
                ch = chr(kc).lower() if kc < 256 else ""
                if ch == "q":
                    break
                if ch == "t": d["takeoff"](); flying = True
                elif ch == "l": d["land"](); flying = False
                elif ch == "v": recording = not recording; print("recording", recording)
                elif ch == "g": gimbal_pitch = min(90, gimbal_pitch + 5); d["gimbal"](gimbal_pitch)
                elif ch == "b": gimbal_pitch = max(-90, gimbal_pitch - 5); d["gimbal"](gimbal_pitch)
                elif kc in SPEEDS: speed = SPEEDS[kc]
                elif ch == " ": held = None; d["pcmd"]()
                else:
                    move = {"w": dict(pitch=speed), "s": dict(pitch=-speed), "a": dict(roll=-speed), "d": dict(roll=speed),
                            "r": dict(gaz=speed), "f": dict(gaz=-speed), "z": dict(yaw=-speed), "c": dict(yaw=speed)}.get(ch)
                    if move is None and k in ARROWS:
                        move = {"up": dict(gaz=speed), "down": dict(gaz=-speed), "yaw_l": dict(yaw=-speed), "yaw_r": dict(yaw=speed)}[ARROWS[k]]
                    if move:
                        held, last_key_t = move, now
            if held and now - last_key_t > HOLD:
                held = None
            d["pcmd"](**(held or {}))
            if args.seconds and t > args.seconds:
                break
    finally:
        if writer: writer.release()          # save the video FIRST, so a hang below can't corrupt it
        csv_f.close()
        if flying:
            print("Landing..."); d["land"]()
        d["close"](); cv2.destroyAllWindows()
        print(f"Saved {n} frames to {out}")


if __name__ == "__main__":
    main()
