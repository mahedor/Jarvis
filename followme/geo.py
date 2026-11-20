"""One-drone map position: pixel -> ray from the camera -> where it hits flat ground -> metres north/east of takeoff.

Michael 2026-11-08: report "metres N/E of takeoff" (raw GPS lat/lon logged alongside), calibrate the camera with a
checkerboard (calibrate_camera.py writes camera_calib.json; without it we fall back to the ASSUMED 69 x 42 deg FOV).

Frames used here:
  camera:  forward (out of the lens), right, down. Pixel -> (1, x_n, y_n) with x_n, y_n = normalised image coords.
  body:    forward / right / down of the drone heading, level (the gimbal is stabilised, so drone roll/pitch don't
           tilt the camera; only the gimbal's absolute pitch does).
  world:   north / east of the takeoff point, ground = flat at takeoff height.
Assumptions (each one is an error source): flat ground; camera yaw == drone yaw (gimbal yaw frame 'none');
drone_yaw_deg is a compass heading (clockwise from north) -- VERIFY on the first flight; consumer GPS (metres).
"""
import json
import math
from pathlib import Path

HFOV_DEG = 69.0   # ASSUMED (replay.py's VFOV 42 deg comes from this at 16:9); replaced by camera_calib.json if present
VFOV_DEG = 42.0
MIN_DOWN_DEG = 3.0          # rays flatter than this hit the ground absurdly far away -> no position
EARTH_R_M = 6378137.0       # WGS-84 equatorial radius; fine for a yard-sized equirectangular approximation
GPS_INVALID = 500.0         # Parrot reports 500 for lat/lon/alt when there is no fix


def load_calib(path=None):
    """camera_calib.json from calibrate_camera.py -> dict(K, dist, size) or None."""
    p = Path(path) if path else Path(__file__).resolve().parent / "camera_calib.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return dict(K=c["K"], dist=c["dist"], size=tuple(c["size"]))


def normalised(u, v, W, H, calib=None):
    """Pixel (u, v) -> (x_n, y_n): tangent of the angle right / down of the optical axis."""
    if calib is None:
        x_n = (u - W / 2) / (W / 2) * math.tan(math.radians(HFOV_DEG / 2))
        y_n = (v - H / 2) / (H / 2) * math.tan(math.radians(VFOV_DEG / 2))
        return x_n, y_n
    import numpy as np
    import cv2
    K = np.array(calib["K"], dtype=np.float64)
    cw, ch = calib["size"]
    if (cw, ch) != (W, H):                               # calibrated at another resolution: scale the intrinsics
        K = K.copy(); K[0] *= W / cw; K[1] *= H / ch
    pt = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, np.array(calib["dist"], dtype=np.float64))
    return float(pt[0, 0, 0]), float(pt[0, 0, 1])


def ground_offset(u, v, W, H, altitude_m, gimbal_pitch_deg, calib=None):
    """Where the ray through pixel (u, v) hits flat ground, relative to the point under the drone, in the drone's
    heading frame. Returns (forward_m, right_m) or None (no telemetry / ray at or above the horizon)."""
    if altitude_m is None or gimbal_pitch_deg is None or altitude_m <= 0.3:
        return None
    x_n, y_n = normalised(u, v, W, H, calib)
    th = math.radians(-gimbal_pitch_deg)                 # tilt below horizontal (gimbal pitch is negative looking down)
    fwd = math.cos(th) - math.sin(th) * y_n              # rotate the camera ray (1, x_n, y_n) down by th
    down = math.sin(th) + math.cos(th) * y_n
    if down <= 0 or math.degrees(math.atan2(down, math.hypot(fwd, x_n))) < MIN_DOWN_DEG:
        return None
    t = altitude_m / down
    return t * fwd, t * x_n


def rotate_to_ne(forward_m, right_m, yaw_deg):
    """Heading frame -> north/east. yaw_deg = compass heading of the drone nose (0 = north, 90 = east)."""
    y = math.radians(yaw_deg)
    return (forward_m * math.cos(y) - right_m * math.sin(y),
            forward_m * math.sin(y) + right_m * math.cos(y))


def gps_valid(lat, lon):
    return (lat is not None and lon is not None and abs(lat - GPS_INVALID) > 1e-6 and abs(lon - GPS_INVALID) > 1e-6
            and -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0))


def gps_to_ne(lat, lon, lat0, lon0):
    """Small-area (equirectangular) offset in metres of (lat, lon) from the origin (lat0, lon0)."""
    n = math.radians(lat - lat0) * EARTH_R_M
    e = math.radians(lon - lon0) * EARTH_R_M * math.cos(math.radians(lat0))
    return n, e


class MapFrame:
    """Keeps the takeoff origin and turns a target box + telemetry into map numbers for the log.

    Origin = the first valid GPS fix seen (start the app on the takeoff spot). Before a fix, positions are
    drone-relative only (person_fwd_m / person_right_m / person_dn_m / person_de_m) and the world columns are empty.
    """
    def __init__(self, calib=None):
        self.calib = calib
        self.origin = None

    def update(self, box, W, H, tel):
        lat, lon = tel.get("gps_lat"), tel.get("gps_lon")
        fix = gps_valid(lat, lon)
        if fix and self.origin is None:
            self.origin = (lat, lon)
        out = dict(drone_n_m=None, drone_e_m=None, person_fwd_m=None, person_right_m=None,
                   person_dn_m=None, person_de_m=None, person_n_m=None, person_e_m=None)
        if fix and self.origin:
            out["drone_n_m"], out["drone_e_m"] = gps_to_ne(lat, lon, *self.origin)
        if box is None:
            return self._round(out)
        x1, y1, x2, y2 = box[:4]
        if y2 >= H - 3:                                  # feet cut off: the bottom of the box is not the ground point
            return self._round(out)
        off = ground_offset((x1 + x2) / 2, y2, W, H, tel.get("altitude_takeoff_m"), tel.get("gimbal_pitch_deg"),
                            self.calib)
        if off is None:
            return self._round(out)
        out["person_fwd_m"], out["person_right_m"] = off
        yaw = tel.get("drone_yaw_deg")
        if yaw is not None:
            out["person_dn_m"], out["person_de_m"] = rotate_to_ne(*off, yaw)
            if out["drone_n_m"] is not None:
                out["person_n_m"] = out["drone_n_m"] + out["person_dn_m"]
                out["person_e_m"] = out["drone_e_m"] + out["person_de_m"]
        return self._round(out)

    @staticmethod
    def _round(d):
        return {k: (None if v is None else round(v, 2)) for k, v in d.items()}
