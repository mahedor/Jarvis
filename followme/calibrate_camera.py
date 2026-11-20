#!/usr/bin/env python3
"""Calibrate the ANAFI live-stream camera from a video of a printed checkerboard -> camera_calib.json.

Why: map position turns a pixel into an angle, so it needs the camera's real field of view (and lens distortion).
replay.py / geo.py currently ASSUME 69 x 42 deg. (Michael 2026-11-08: calibrate with a checkerboard.)

1. Print a checkerboard with 10 x 7 squares (= 9 x 6 INNER corners), e.g. the OpenCV pattern.png, on plain paper;
   tape it flat to a board or wall. Exact square size doesn't matter for field of view.
2. Drone on the ground or in your hands, motors OFF. Record ~60 s with record.py (same stream as flights, 1280x720):
   move the board slowly so it appears near the centre, all four edges and corners, tilted different ways,
   near and far. Corners/edges matter most: that's where distortion shows.
3. python calibrate_camera.py --video ~/recordings/<stamp>/video_000.mp4      (several --video allowed)
   Writes camera_calib.json next to this file; live_follow.py / geo.py pick it up automatically.
Good result: reprojection error under ~0.5 px and 20+ usable views.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", nargs="+", required=True, help="video file(s) or image folder(s) of the checkerboard")
ap.add_argument("--cols", type=int, default=9, help="inner corners along the long side")
ap.add_argument("--rows", type=int, default=6, help="inner corners along the short side")
ap.add_argument("--every", type=int, default=10, help="try every Nth frame")
ap.add_argument("--max-views", type=int, default=60)
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "camera_calib.json"))
a = ap.parse_args()

pattern = (a.cols, a.rows)
obj = np.zeros((a.cols * a.rows, 3), np.float32)
obj[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def frames(src):
    p = Path(src).expanduser()
    if p.is_dir():
        for f in sorted(p.glob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                yield cv2.imread(str(f))
        return
    cap = cv2.VideoCapture(str(p))
    while True:
        ok, img = cap.read()
        if not ok:
            return
        yield img


views, size, seen_cells = [], None, set()
for src in a.video:
    for i, img in enumerate(frames(src)):
        if i % a.every:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = size or (g.shape[1], g.shape[0])
        assert (g.shape[1], g.shape[0]) == size, "all inputs must have the same resolution"
        ok, c = cv2.findChessboardCorners(g, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit).reshape(-1, 1, 2).astype(np.float32)   # shape differs by OpenCV version
        cx, cy = c[:, 0, 0].mean() / size[0], c[:, 0, 1].mean() / size[1]
        cell = (int(cx * 4), int(cy * 3), int(10 * (np.ptp(c[:, 0, 0]) / size[0])))   # where + how big: keep variety
        views.append((cell, c))
        seen_cells.add(cell[:2])

# keep a spread of positions instead of 500 near-identical views from the middle
by_cell = {}
for cell, c in views:
    by_cell.setdefault(cell, []).append(c)
picked = []
while len(picked) < a.max_views and any(by_cell.values()):
    for k in list(by_cell):
        if by_cell[k]:
            picked.append(by_cell[k].pop(len(by_cell[k]) // 2))
picked = picked[:a.max_views]
print(f"checkerboard found in {len(views)} sampled frames; using {len(picked)} spread over "
      f"{len(seen_cells)}/12 image regions")
if len(picked) < 10:
    raise SystemExit("Too few usable views (need 10+, ideally 20+). Fill the frame edges/corners and keep the board flat.")

rms, K, dist, _, _ = cv2.calibrateCamera([obj] * len(picked), picked, size, None, None)
W, H = size
hfov = 2 * math.degrees(math.atan(W / 2 / K[0, 0]))
vfov = 2 * math.degrees(math.atan(H / 2 / K[1, 1]))
res = dict(K=K.tolist(), dist=dist.ravel().tolist(), size=[W, H], rms_px=round(float(rms), 3),
           hfov_deg=round(hfov, 2), vfov_deg=round(vfov, 2), views=len(picked), pattern=list(pattern),
           regions_covered=len(seen_cells))
Path(a.out).write_text(json.dumps(res, indent=2))
print(f"reprojection error {rms:.3f} px  |  FOV {hfov:.1f} x {vfov:.1f} deg (code assumed 69 x 42)  |  "
      f"principal point ({K[0, 2]:.0f}, {K[1, 2]:.0f}) vs centre ({W / 2:.0f}, {H / 2:.0f})")
print("distortion k1 k2 p1 p2 k3:", np.round(dist.ravel(), 4).tolist())
print("wrote", a.out)
if rms > 1.0:
    print("WARNING: error > 1 px - board probably not flat, blurry frames, or wrong --cols/--rows.")
if len(seen_cells) < 8:
    print("WARNING: board never reached some image regions; corners/edges drive the distortion estimate.")
