"""Tests for side_by_side.py (Michael 11-27, talk videos) on tiny synthetic videos: every frame is a solid colour that
encodes its own index, so the output says exactly which input frame each side showed.

  * two run folders 21 s apart (the 11-22 two-drone layout) line up by wall clock: folder-name start time, refined by
    follow.csv's mtime only when that is within 2 s after it; chunked video_*.mp4 files are followed via frame_in_file;
  * overlap vs union span ("not recording yet"), --start / --duration / --offset;
  * two replay folders line up frame for frame by commands.csv t_s, labels from replay_info.json;
  * a plain video runs on frame / fps;
  * end-to-end renders with OpenCV's writer and (if installed) ffmpeg, decoded and checked pixel by pixel."""
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import side_by_side as sbs  # noqa: E402

W0, H0 = 64, 48


def colour(idx, src):
    """Frame idx (< 300) of source src (0 or 1) -> BGR with steps of 25-40 per digit, which survive two rounds of
    lossy compression (input mp4v, then the output encoder) of a solid frame: blue = ones digit * 25, green = tens
    digit * 25, red = 40 + 160 * src + hundreds digit * 40 (src 1 has < 200 frames)."""
    return (idx % 10 * 25, idx // 10 % 10 * 25, 40 + 160 * src + idx // 100 * 40)


def decode(bgr):
    b, g, r = (int(v) for v in bgr)
    src = int(r >= 180)
    return int(round((r - 40 - 160 * src) / 40)) * 100 + int(round(g / 25)) * 10 + int(round(b / 25)), src


def write_video(path, colours, fps=10.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W0, H0))
    for c in colours:
        vw.write(np.full((H0, W0, 3), c, np.uint8))
    vw.release()


def make_run(root, name, n, src, fps=10.0, chunk=100, drop_frame_in_file=False):
    """A live_follow-style run folder: video_000.mp4... chunks of `chunk` frames + follow.csv (t_s = i / fps)."""
    run = root / name
    run.mkdir(parents=True)
    rows = []
    for c0 in range(0, n, chunk):
        vf = f"video_{c0 // chunk:03d}.mp4"
        write_video(run / vf, [colour(i, src) for i in range(c0, min(c0 + chunk, n))], fps)
        for i in range(c0, min(c0 + chunk, n)):
            r = dict(frame=i, t_s=round(i / fps, 3), video_file=vf, frame_in_file=i - c0, mode="MANUAL")
            if drop_frame_in_file:
                r.pop("frame_in_file")
            rows.append(r)
    with open(run / "follow.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (run / "drone.json").write_text(json.dumps({"name": name.split("_")[-1]}))
    return run


def make_replay(root, name, n, src, label=None, dt=0.1):
    out = root / name
    out.mkdir(parents=True)
    write_video(out / "replay.mp4", [colour(i, src) for i in range(n)], 30.0)   # replay.mp4's own fps is ignored
    with open(out / "commands.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "t_s"]); w.writeheader()
        w.writerows(dict(frame=i, t_s=round(5.0 + i * dt, 3)) for i in range(n))
    if label:
        (out / "replay_info.json").write_text(json.dumps({"label": label}))
    return out


@pytest.fixture
def two_drones(tmp_path):
    """The 11-22 layout: drone A starts 16:27:19, drone B 16:27:40 (21 s later), one laptop clock."""
    a = make_run(tmp_path, "20261122_162719_droneA", 300, 0)           # 30 s at 10 fps, 3 chunks
    b = make_run(tmp_path, "20261122_162740_droneB", 150, 1)           # 15 s
    return a, b


def test_run_folders_align_by_wall_clock(two_drones):
    a, b = (sbs.load_source(p) for p in two_drones)
    assert (a.label, b.label) == ("drone A", "drone B") and a.kind == b.kind == "run"
    assert b.anchor - a.anchor == pytest.approx(21.0) and "folder name" in a.anchor_note
    times, pa, pb, wall = sbs.plan(a, b, fps=10)
    assert wall and len(times) == 90                                   # overlap: A's 21-30 s = B's 0-9 s
    assert a.frames[pa[0]][0] == pytest.approx(21.0) and pb[0] == 0
    assert [a.frames[k][2] for k in pa[:3]] == [10, 11, 12]            # A frame 210 = video_002.mp4 frame 10
    assert a.frames[pa[0]][1].name == "video_002.mp4"
    assert all(a.frames[x][0] - b.frames[y][0] == pytest.approx(21.0) for x, y in zip(pa, pb))


def test_union_span_shows_not_recording(two_drones):
    a, b = (sbs.load_source(p) for p in two_drones)
    times, pa, pb, _ = sbs.plan(a, b, fps=10, span="union")
    assert len(times) == 360                                           # 16:27:19 .. 16:27:55
    assert pb[:210] == [None] * 210 and pb[210] == 0                   # B not recording for its first 21 s
    assert pa[300:] == [None] * 60 and None not in pa[:300]            # A ended at 30 s


def test_start_duration_and_offset(two_drones):
    a, b = (sbs.load_source(p) for p in two_drones)
    times, pa, pb, _ = sbs.plan(a, b, fps=10, start=2.0, duration=3.0)
    assert len(times) == 30 and pb[0] == 20 and a.frames[pa[0]][0] == pytest.approx(23.0)
    times, pa, pb, _ = sbs.plan(a, b, fps=10, offset=-1.0)             # B shown 1 s earlier
    assert a.frames[pa[0]][0] - b.frames[pb[0]][0] == pytest.approx(20.0)
    with pytest.raises(SystemExit):
        sbs.plan(a, b, offset=60.0)                                    # never overlap -> stop, not an empty video


def test_mtime_refines_the_start_only_when_plausible(two_drones):
    a_dir, b_dir = two_drones
    ft = sbs.folder_time(b_dir.name)
    os.utime(b_dir / "follow.csv", (ft + 0.4 + 14.9, ft + 0.4 + 14.9))   # last t_s 14.9, written 0.4 s after the name
    b = sbs.load_source(b_dir)
    assert b.anchor == pytest.approx(ft + 0.4) and "mtime" in b.anchor_note
    os.utime(b_dir / "follow.csv", (ft + 5 + 14.9, ft + 5 + 14.9))       # 5 s off (a copy, an edit): not trusted
    assert sbs.load_source(b_dir).anchor == ft


def test_old_logs_without_frame_in_file(tmp_path):
    run = make_run(tmp_path, "20261029_182029", 250, 0, drop_frame_in_file=True)
    s = sbs.load_source(run)
    assert [f[2] for f in s.frames[98:102]] == [98, 99, 0, 1] and s.label == "20261029_182029"


def test_replays_align_frame_for_frame_with_their_labels(tmp_path):
    a = sbs.load_source(make_replay(tmp_path, "r070", 40, 0, label="cutoff 0.70"))
    b = sbs.load_source(make_replay(tmp_path, "r079", 40, 1, label="cutoff 0.79") / "replay.mp4")
    assert (a.label, b.label) == ("cutoff 0.70", "cutoff 0.79") and a.anchor is None
    times, pa, pb, wall = sbs.plan(a, b, fps=10)
    assert not wall and pa == pb == list(range(40))
    assert a.frames[0][0] == 5.0                                       # the flight's recorded t_s, not frame / fps
    assert sbs.load_source(make_replay(tmp_path, "r", 5, 0)).label == "r"   # no replay_info label -> folder name
    assert sbs.load_source(tmp_path / "r", "mine").label == "mine"


def test_plain_video(tmp_path):
    write_video(tmp_path / "clip.mp4", [colour(i, 0) for i in range(20)], fps=10)
    s = sbs.load_source(tmp_path / "clip.mp4")
    assert s.kind == "video" and len(s.frames) == 20 and s.frames[5][0] == pytest.approx(0.5)
    with pytest.raises(SystemExit):
        sbs.load_source(tmp_path / "nothing_here")


def _frames(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, img = cap.read()
        if not ok:
            return out
        out.append(img)


def _centre(img, side, pw):
    """The colour at the centre of the left (0) or right (1) panel's picture."""
    x = pw // 2 + side * (pw + sbs.GAP)
    return img[sbs.BAR_H + (img.shape[0] - sbs.BAR_H) // 2, x]


@pytest.mark.parametrize("encoder", ["opencv", "ffmpeg"])
def test_render_end_to_end(two_drones, tmp_path, encoder):
    if encoder == "ffmpeg" and not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg on PATH")
    out = tmp_path / f"both_{encoder}.mp4"
    r = sbs.main([str(two_drones[0]), str(two_drones[1]), "--out", str(out), "--fps", "10", "--span", "union",
                  "--start", "19", "--duration", "4", "--encoder", encoder])
    assert r["encoder"] == encoder and r["wall"] and r["frames"] == 40
    frames = _frames(out)
    assert len(frames) == 40
    pw = (r["size"][0] - sbs.GAP) // 2
    assert r["size"] == (2 * W0 + sbs.GAP, sbs.BAR_H + H0)
    for i in (0, 10, 19, 20, 21, 39):                                  # 16:27:38.0 + i / 10
        assert decode(_centre(frames[i], 0, pw)) == (190 + i, 0), i   # drone A: frame 190 + i
        if i < 20:                                                     # drone B not yet recording: dark panel
            assert int(_centre(frames[i], 1, pw).max()) < 40, i
        else:
            assert decode(_centre(frames[i], 1, pw)) == (i - 20, 1), i
    assert int(frames[0][sbs.BAR_H // 2, 2].max()) < 60                # label bar is dark


def test_render_replays(tmp_path):
    a = make_replay(tmp_path, "r070", 30, 0, label="cutoff 0.70")
    b = make_replay(tmp_path, "r079", 30, 1, label="cutoff 0.79")
    out = tmp_path / "cmp.mp4"
    r = sbs.main([str(a), str(b), "--out", str(out), "--fps", "10", "--encoder", "opencv"])
    frames = _frames(out)
    assert r["frames"] == len(frames) == 30
    pw = (r["size"][0] - sbs.GAP) // 2
    assert [decode(_centre(frames[i], 0, pw)) for i in (0, 29)] == [(0, 0), (29, 0)]
    assert [decode(_centre(frames[i], 1, pw)) for i in (0, 29)] == [(0, 1), (29, 1)]


def test_encoder_choice(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert sbs.pick_encoder("auto") == "opencv"
    with pytest.raises(SystemExit):
        sbs.pick_encoder("ffmpeg")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert sbs.pick_encoder("auto") == "ffmpeg"
    cmd = sbs.ffmpeg_cmd(Path("o.mp4"), 136, 92, 30)
    assert "libx264" in cmd and "yuv420p" in cmd and cmd[-1] == "o.mp4"
