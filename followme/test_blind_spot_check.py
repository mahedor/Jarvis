"""Tests for blind_spot_check.py (Michael 11-28, talk chapter 12 / D4): the 63 frames and the hit rule are gap_fps.py's
(11-22, committed byte-exact in results/), frames are found through follow.csv across a video file boundary, a missing
frame / file / weights file stops, and the module stays light (no ultralytics, no rfdetr until a model is asked for).
Pure logic plus one end-to-end run with a stand-in detector on tiny synthetic chunks (no model)."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blind_spot_check as bs  # noqa: E402
from test_side_by_side import decode, make_run  # noqa: E402  (frames whose colour encodes their index)

# sha256 of the 11-22 files as found on Michael's laptop (C:\Users\mahed\Downloads), copied on 11-28
EVIDENCE_SHA256 = {
    "script": "431d05cdff34c0ea793d56e4eb7b1e08cb3595e3770ece75ce19e5187a1f3118",
    "yolo": "57adf39db9b309ed3c869c55245894e79d5616ac1ec1581f763d7bf16e9f6520",
    "rfdetr": "a91942454653f60c6f7e50104cea9b707d7a6966d8d0f7b2e93da551a8f41647",
}


def test_evidence_files_are_the_byte_exact_copies():
    for k, sha in EVIDENCE_SHA256.items():
        assert hashlib.sha256(bs.EVIDENCE[k].read_bytes()).hexdigest() == sha, k


def test_frames_and_region_are_gap_fps_own():
    """The re-check must measure the same 63 frames and the same region as the 11-22 run it is compared with."""
    src = bs.EVIDENCE["script"].read_text()
    m = re.search(r"FIRST, LAST, STEP = (\d+), (\d+), (\d+)", src)
    assert (bs.FIRST, bs.LAST, bs.STEP) == tuple(int(x) for x in m.groups())
    reg = re.search(r"REG = np\.array\(\[([^\]]+)\]\)", src).group(1)
    assert bs.REGION == tuple(float(x) for x in reg.split(","))
    assert "conf=0.25" in src and bs.ALSO_CONF == 0.25
    frames = bs.sample_frames()
    assert len(frames) == 63 and frames[0] == 3507 and frames[-1] == 3879 and frames[1] - frames[0] == 6
    old = json.loads(bs.EVIDENCE["yolo"].read_text())
    assert old["YOLOv8n fine-tuned (flies today)"]["hits"] == 0 and old["YOLO11l stock COCO"]["hits"] == 54
    assert all(r["of"] == 63 for r in old.values())
    assert json.loads(bs.EVIDENCE["rfdetr"].read_text())["RF-DETR nano fine-tuned (final, 50 ep)"]["hits"] == 59


def _rows(n_per_file=(1800, 1800), with_fif=True):
    rows, f = [], 0
    for k, n in enumerate(n_per_file):
        for i in range(n):
            r = dict(frame=str(f), t_s=str(round(f / 30, 3)), video_file=f"video_{k:03d}.mp4")
            if with_fif:
                r["frame_in_file"] = str(i)
            rows.append(r); f += 1
    return rows


def test_frame_map_crosses_the_file_boundary():
    """The 10-29 layout: 1800 frames per chunk, so 3507 is video_001 frame 1707 and 3879 is video_002 frame 279."""
    rows = _rows((1800, 1800, 1800))
    m = bs.frame_map(rows, bs.sample_frames())
    assert m[3507][:2] == ("video_001.mp4", 1707) and m[3879][:2] == ("video_002.mp4", 279)
    assert m[3597][:2] == ("video_001.mp4", 1797) and m[3603][:2] == ("video_002.mp4", 3)
    assert bs.frame_map(_rows((1800, 1800, 1800), with_fif=False), [3599, 3600]) == {
        3599: ("video_001.mp4", 1799, 119.967), 3600: ("video_002.mp4", 0, 120.0)}     # old logs: counted per chunk


def test_frame_map_stops_on_a_missing_frame():
    with pytest.raises(SystemExit, match="no video frame for frame"):
        bs.frame_map(_rows((1800, 1800)), bs.sample_frames())                       # only 3600 rows


def test_check_videos_stops_on_missing_or_short_files(tmp_path):
    m = {1: ("video_000.mp4", 5, 0.0)}
    with pytest.raises(SystemExit, match="video file missing"):
        bs.check_videos(tmp_path, m)
    (tmp_path / "video_000.mp4").write_bytes(b"x")
    with pytest.raises(SystemExit, match="has 3 frames, frame_in_file 5 needed"):
        bs.check_videos(tmp_path, m, frame_count=lambda p: 3)
    bs.check_videos(tmp_path, m, frame_count=lambda p: 6)


def test_hit_rule_is_the_box_centre_in_the_region():
    reg = (100, 100, 200, 200)
    assert bs.center_in((90, 90, 110, 110), reg)                 # centre (100, 100): on the edge counts
    assert not bs.center_in((0, 0, 198, 198), reg)               # centre (99, 99)
    assert not bs.center_in((150, 150, 300, 300), reg)           # overlaps, centre (225, 225) outside
    assert bs.is_hit([dict(xyxy=(0, 0, 10, 10)), dict(xyxy=(140, 140, 160, 160))], reg)
    assert not bs.is_hit([], reg)
    s = bs.summarize([dict(frame=1, hit=True, ms=10.0), dict(frame=7, hit=False, ms=30.0), dict(frame=13, hit=True, ms=20.0)])
    assert s == dict(hits=2, of=3, hit_frames=[1, 13], ms_median=20.0, fps=50.0)


def test_also_needs_name_and_existing_weights(tmp_path):
    w = tmp_path / "yolo11l.pt"; w.write_bytes(b"x")
    r = tmp_path / "rfdetr_final_best_total.pth"; r.write_bytes(b"x")
    assert bs.parse_also(f"yolo11l={w}") == ("yolo11l", "yolo", w)
    assert bs.parse_also(f"rfdetr={r}") == ("rfdetr", "rfdetr", r)
    with pytest.raises(SystemExit, match="weights not found"):
        bs.parse_also(f"yolo11l={tmp_path / 'nope.pt'}")
    with pytest.raises(SystemExit, match="use NAME=PATH"):
        bs.parse_also("yolo11l.pt")


def test_module_stays_light():
    """Importing it (and so running it in another env with --no-flight) loads neither ultralytics nor rfdetr."""
    code = ("import sys; sys.path.insert(0, %r); import blind_spot_check; "
            "assert not {'ultralytics', 'rfdetr', 'live_follow', 'torch'} & set(sys.modules), sorted(sys.modules)") % str(HERE)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]


def test_end_to_end_with_a_stand_in_detector(tmp_path, monkeypatch):
    """Two chunks of 10 frames; frames 4..16 every 3 (crosses the boundary at 10). A stand-in 'model' sees the person
    only on frames whose index is even, so the hits and the video's frames are known exactly."""
    run = make_run(tmp_path, "20261029_182029", 20, 0, fps=10.0, chunk=10)
    monkeypatch.setattr(bs, "FIRST", 4); monkeypatch.setattr(bs, "LAST", 16); monkeypatch.setattr(bs, "STEP", 3)
    monkeypatch.setattr(bs, "REGION", (10, 10, 50, 40))

    def fake_also(name, kind, path, images, mapping, region=bs.REGION):
        def detect(img):
            idx, _ = decode(img[24, 32])
            return [dict(xyxy=[20.0, 15.0, 40.0, 35.0], conf=0.9)] if idx % 2 == 0 else [dict(xyxy=[0.0, 0.0, 6.0, 6.0], conf=0.5)]
        return dict(label=name, kind=kind, model=str(path), how="stand-in", frames=bs.measure(detect, images, mapping, (10, 10, 50, 40)))
    monkeypatch.setattr(bs, "also_model", fake_also)
    w = tmp_path / "m.pt"; w.write_bytes(b"x")
    out = tmp_path / "out"
    doc = bs.main(["--run", str(run), "--out", str(out), "--no-flight", "--also", f"standin={w}", "--encoder", "opencv"])
    frames = [4, 7, 10, 13, 16]
    assert doc["frames"]["list"] == frames
    assert doc["frames"]["span"]["first"]["video_file"] == "video_000.mp4" and doc["frames"]["span"]["last"] == dict(
        video_file="video_001.mp4", frame_in_file=6, t_s=1.6)
    m = doc["models"][0]
    assert [(f["frame"], f["video_file"], f["frame_in_file"]) for f in m["frames"]] == [
        (4, "video_000.mp4", 4), (7, "video_000.mp4", 7), (10, "video_001.mp4", 0), (13, "video_001.mp4", 3),
        (16, "video_001.mp4", 6)]
    assert m["summary"]["hits"] == 3 and m["summary"]["hit_frames"] == [4, 10, 16]
    saved = json.loads((out / "blind_spot.json").read_text())
    assert saved["models"][0]["summary"] == m["summary"] and saved["previous"]["yolo"] is not None
    cap = cv2.VideoCapture(str(out / "blind_spot.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames) * 3 == saved["video"]["frames"]   # each frame held STEP times
    # --merge: same frames -> added; other frames -> STOP
    doc2 = bs.main(["--run", str(run), "--out", str(tmp_path / "out2"), "--no-flight", "--also", f"second={w}",
                    "--merge", str(out / "blind_spot.json"), "--no-video"])
    assert [x["label"] for x in doc2["models"]] == ["second", "standin"] and doc2["models"][1]["merged_from"]
    monkeypatch.setattr(bs, "STEP", 2)
    with pytest.raises(SystemExit, match="other frames"):
        bs.main(["--run", str(run), "--out", str(tmp_path / "out3"), "--no-flight", "--also", f"second={w}",
                 "--merge", str(out / "blind_spot.json"), "--no-video"])


def test_missing_run_or_nothing_to_do_stops(tmp_path):
    with pytest.raises(SystemExit, match="no follow.csv"):
        bs.main(["--run", str(tmp_path / "nope"), "--out", str(tmp_path / "o")])
    run = make_run(tmp_path, "r", 5, 0, chunk=5)
    with pytest.raises(SystemExit, match="nothing else to measure"):
        bs.main(["--run", str(run), "--out", str(tmp_path / "o"), "--no-flight"])

