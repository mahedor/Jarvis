"""Two-drone "look here" and "go there" (Michael 2026-11-22).

LOOK HERE: both drones hover. On the high (overseer) drone's video you press P and click a spot; that pixel becomes a
ground point (geo.py: pixel -> ray -> flat ground) and then a GPS lat/lon, which is shared with the other (worker)
drone. On the worker you press V: it stays where it is and only turns (yaw) and tilts its camera to look at the point.
GO THERE: on the worker you press X: it turns to face the point, then flies forward with the same kind of simple
proportional (P) control follow-me uses, holds its height (gaz = 0) and stops STOP_M short of the point.

The shared point is absolute GPS (lat/lon), not "metres from takeoff", because the two drones take off from
different spots. Each drone's own GPS, height and compass heading are needed. Error sources: consumer GPS is a few
metres; drone_yaw_deg must be a true compass heading (unverified on the ANAFI 4K); the ground is assumed flat and at
the same height for both takeoffs.

Exchange: a small JSON file (two programs on one laptop) or MQTT (two laptops). Topic jarvis/sensor/<drone>/target,
never under jarvis/presence/ (that branch is people only).
"""
import json
import math
import time
from pathlib import Path

import geo

# Look (turn + tilt in place)
LOOK_K_YAW = 1.2          # % yaw per degree of heading error
LOOK_MAX_YAW = 30.0       # % cap
LOOK_DEADBAND_DEG = 3.0   # close enough: stop turning
# Go (fly to the point)
GO_ALIGN_DEG = 12.0       # turn in place until the nose is within this of the bearing, then fly
GO_K_FWD = 2.0            # % forward per metre still to go
GO_MAX_FWD = 15.0         # % cap (slow on purpose)
STOP_M = 6.0              # stop this far short of the point (never fly onto people)
GO_MAX_DIST_M = 40.0      # refuse "go" if the point is farther than this from the worker
TARGET_MAX_AGE_S = 120.0  # a shared point older than this is ignored
TOPIC = "jarvis/sensor/{drone}/target"


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def ne_to_gps(n, e, lat0, lon0):
    """Inverse of geo.gps_to_ne (small-area equirectangular)."""
    lat = lat0 + math.degrees(n / geo.EARTH_R_M)
    lon = lon0 + math.degrees(e / (geo.EARTH_R_M * math.cos(math.radians(lat0))))
    return lat, lon


def click_to_gps(u, v, W, H, tel, calib=None):
    """Overseer: pixel -> (lat, lon) of the ground point, or None (no GPS fix / heading / ray above the horizon)."""
    lat, lon, yaw = tel.get("gps_lat"), tel.get("gps_lon"), tel.get("drone_yaw_deg")
    if not geo.gps_valid(lat, lon) or yaw is None:
        return None
    off = geo.ground_offset(u, v, W, H, tel.get("altitude_takeoff_m"), tel.get("gimbal_pitch_deg"), calib)
    if off is None:
        return None
    dn, de = geo.rotate_to_ne(*off, yaw)
    return ne_to_gps(dn, de, lat, lon)


def aim(tel, lat_t, lon_t):
    """Worker: where is the point from here? -> dict(bearing_deg, dist_m, yaw_err_deg, gimbal_pitch_deg) or None."""
    lat, lon, yaw, alt = tel.get("gps_lat"), tel.get("gps_lon"), tel.get("drone_yaw_deg"), tel.get("altitude_takeoff_m")
    if not geo.gps_valid(lat, lon) or yaw is None:
        return None
    n, e = geo.gps_to_ne(lat_t, lon_t, lat, lon)
    dist = math.hypot(n, e)
    bearing = math.degrees(math.atan2(e, n)) % 360.0
    pitch = -math.degrees(math.atan2(max(alt or 0.0, 0.0), max(dist, 0.1)))
    return dict(bearing_deg=bearing, dist_m=dist, yaw_err_deg=wrap180(bearing - yaw),
                gimbal_pitch_deg=max(-90.0, min(0.0, pitch)))


class LookGo:
    """Worker-side state machine. step() -> (cmd, gimbal_pitch_deg or None, info). Always hovers when unsure."""
    def __init__(self):
        self.target = None          # (lat, lon, t_received)
        self.mode = None            # None | "look" | "go"

    def set_target(self, lat, lon, now):
        self.target = (lat, lon, now)

    def start(self, mode):
        self.mode = mode if self.target else None
        return self.mode is not None

    def stop(self):
        self.mode = None

    def step(self, tel, now):
        hover = dict(roll=0, pitch=0, yaw=0, gaz=0)
        if self.mode is None:
            return hover, None, dict(state="IDLE", event="")
        if self.target is None or now - self.target[2] > TARGET_MAX_AGE_S:
            self.mode = None
            return hover, None, dict(state="NO TARGET -> HOVER", event="target_stale")
        a = aim(tel, self.target[0], self.target[1])
        if a is None:
            return hover, None, dict(state="NO GPS/HEADING -> HOVER", event="no_gps")
        err = a["yaw_err_deg"]
        yaw = 0.0 if abs(err) < LOOK_DEADBAND_DEG else max(-LOOK_MAX_YAW, min(LOOK_MAX_YAW, LOOK_K_YAW * err))
        info = dict(state="", event="", bearing=round(a["bearing_deg"], 1), dist=round(a["dist_m"], 1),
                    yaw_err=round(err, 1))
        if self.mode == "look":
            info["state"] = "LOOKING (aligned)" if yaw == 0 else "LOOK: turning"
            return dict(hover, yaw=yaw), a["gimbal_pitch_deg"], info
        # go
        if a["dist_m"] > GO_MAX_DIST_M:
            self.mode = "look"
            info.update(state=f"TOO FAR ({a['dist_m']:.0f} m) -> LOOK only", event="go_refused_far")
            return dict(hover, yaw=yaw), a["gimbal_pitch_deg"], info
        if a["dist_m"] <= STOP_M:
            self.mode = "look"
            info.update(state="ARRIVED -> LOOK", event="arrived")
            return dict(hover, yaw=yaw), a["gimbal_pitch_deg"], info
        if abs(err) > GO_ALIGN_DEG:
            info["state"] = "GO: turning to face it"
            return dict(hover, yaw=yaw), a["gimbal_pitch_deg"], info
        fwd = max(0.0, min(GO_MAX_FWD, GO_K_FWD * (a["dist_m"] - STOP_M)))
        info["state"] = f"GO: flying ({a['dist_m'] - STOP_M:.1f} m to go)"
        return dict(hover, yaw=yaw, pitch=fwd), a["gimbal_pitch_deg"], info


class TargetLink:
    """Share the clicked point between the two programs: JSON file (same laptop) or MQTT (two laptops)."""
    def __init__(self, file=None, mqtt_host=None, drone="droneA", listen_to="droneA"):
        self.file = Path(file) if file else None
        self.drone, self.last_seen, self.inbox, self.client = drone, None, None, None
        if mqtt_host:
            import paho.mqtt.client as mqtt          # optional dependency, only for two laptops
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"jarvis-{drone}-{int(time.time())}",
                                      protocol=mqtt.MQTTv311)
            self.client.on_message = lambda c, u, m: setattr(self, "inbox", m.payload.decode())
            self.client.connect(mqtt_host, 1883, 30)
            self.client.subscribe(TOPIC.format(drone=listen_to), qos=1)
            self.client.loop_start()

    def publish(self, lat, lon):
        msg = json.dumps(dict(lat=lat, lon=lon, t=time.time(), src=self.drone))
        if self.file:
            tmp = self.file.with_suffix(".tmp"); tmp.write_text(msg); tmp.replace(self.file)
        if self.client:
            self.client.publish(TOPIC.format(drone=self.drone), msg, qos=1, retain=True)

    def poll(self):
        """-> (lat, lon) if a NEW point arrived since the last poll, else None."""
        raw = self.inbox
        if raw is None and self.file and self.file.exists():
            raw = self.file.read_text()
        self.inbox = None
        if not raw:
            return None
        try:
            m = json.loads(raw)
        except ValueError:
            return None
        key = (m.get("lat"), m.get("lon"), m.get("t"))
        if key == self.last_seen or time.time() - float(m.get("t", 0)) > TARGET_MAX_AGE_S:
            return None
        self.last_seen = key
        return float(m["lat"]), float(m["lon"])
