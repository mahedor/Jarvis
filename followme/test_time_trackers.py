"""Tests for time_trackers.py (laptop-CPU speed of the 15 tuned tracker setups, Michael 11-27).

  * the 15 setups and their settings come from results/followme_tracker_tuning_all_summary.json (byte-exact from Drive),
    built the way tracker_tuning_all.py / jarvis_track_v5.tracker_configs() built them;
  * the detector runs once per frame and is not in the tracker timing;
  * a setup that fails (build or mid-run) gets an error in its row and the other rows still run.
A fake detector and a tiny synthetic video stand in for the INT8 model and a flight; the trackers are the real
Ultralytics ones, with a fake re-ID encoder so nothing is downloaded."""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import time_trackers as tt  # noqa: E402

SUMMARY_SHA256 = "bb3a31f9bbcd038cffbc0a570521ed01277a60f08ffee34943212de7eabd2b14"   # Drive file, 114828 bytes
DET_SLEEP_S = 0.03
N_FRAMES, WARM = 30, 5


def summary_json():
    return json.loads(tt.SUMMARY_FILE.read_text())


@pytest.fixture
def video(tmp_path):
    """A tiny clean flight-like video: textured top half (GMC has corners to follow), a box moving right."""
    p = tmp_path / "video_000.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    rng = np.random.default_rng(0)
    for i in range(N_FRAMES):
        img = np.full((240, 320, 3), 40, np.uint8)
        img[:120] = rng.integers(0, 255, (120, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (20 + 3 * i, 60), (60 + 3 * i, 160), (0, 0, 255), -1)
        w.write(img)
    w.release()
    return p


class FakeDetector:
    """Sleeps DET_SLEEP_S per call (so a timing that included it would show) and returns three boxes."""
    def __init__(self):
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        time.sleep(DET_SLEEP_S)
        k = self.calls
        return np.array([[20 + 3 * k, 60, 60 + 3 * k, 160, 0.9, 0], [200, 100, 240, 200, 0.5, 2],
                         [100, 20, 130, 50, 0.15, 0]], np.float32)


@pytest.fixture
def fake_reid(monkeypatch):
    """The re-ID setups' encoder (ultralytics' ReID on yolo26n-cls.pt) replaced by random vectors: no download."""
    pytest.importorskip("ultralytics")
    import ultralytics.trackers.utils.reid as R

    class FakeReID:
        def __init__(self, model, *a, **k):
            self.model_path, self.device = model, k.get("device")

        def __call__(self, img, dets):
            return [np.random.rand(64).astype(np.float32) for _ in range(len(dets))]
    monkeypatch.setattr(R, "ReID", FakeReID)
    return FakeReID


def test_summary_is_the_drive_file_byte_exact():
    raw = tt.SUMMARY_FILE.read_bytes()
    assert len(raw) == 114828 and hashlib.sha256(raw).hexdigest() == SUMMARY_SHA256


def test_15_setups_with_settings_from_the_summary():
    s = summary_json()
    specs = tt.setup_specs(tt.load_summary()[0])
    assert len(specs) == 15 == len(s["per_setup"])
    assert [x["label"] for x in specs] == list(s["per_setup"])
    for x in specs:
        rec = s["per_setup"][x["label"]]
        assert x["error"] is None
        assert x["settings"] == rec["chosen"]["settings_changed"] and x["val_HOTA"] == rec["chosen"]["val_HOTA"]
        name, *sw = x["label"].split("|")
        assert x["tracker"] == name
        if not sw:                                      # bytetrack / ocsort / fasttrack: plain, as the grid ran them
            assert x["overrides"] == {} and x["reid"] is None and x["gmc"] is None
        else:
            assert x["overrides"]["with_reid"] is (sw[0] == "reid+") is x["reid"]
            assert x["overrides"]["gmc_method"] == ("sparseOptFlow" if sw[1] == "gmc+" else "none")
            assert x["overrides"].get("model") == ("yolo26n-cls.pt" if x["reid"] else None)
    assert sorted({x["tracker"] for x in specs}) == ["botsort", "bytetrack", "deepocsort", "fasttrack", "ocsort",
                                                     "tracktrack"]


def test_settings_change_follows_the_file_not_the_code():
    s = summary_json()
    s["per_setup"]["bytetrack"]["chosen"]["settings_changed"] = {"match_thresh": 0.123}
    del s["per_setup"]["ocsort"]["chosen"]
    specs = {x["label"]: x for x in tt.setup_specs(s)}
    assert specs["bytetrack"]["settings"] == {"match_thresh": 0.123}
    assert specs["ocsort"]["settings"] is None and "chosen" in specs["ocsort"]["error"]


def test_tuned_keys_exist_in_the_installed_tracker_yaml():
    pytest.importorskip("ultralytics")
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    for x in tt.setup_specs(tt.load_summary()[0]):
        cfg = YAML.load(check_yaml(f"{x['tracker']}.yaml"))
        assert set(x["settings"]) | set(x["overrides"]) <= set(cfg), x["label"]


def test_all_16_run_and_the_detector_is_outside_the_timing(video, fake_reid):
    det = FakeDetector()
    specs = tt.all_specs(tt.load_summary()[0])
    rows, d, n = tt.run([video], 0, N_FRAMES, det, specs, warmup=WARM, progress=lambda *_: None,
                        reid_model="weights/yolo26n-cls.pt")
    assert n == N_FRAMES and det.calls == N_FRAMES     # detector once per frame, not once per setup
    assert len(rows) == 16 and [r["setup"] for r in rows] == [x["label"] for x in specs]
    assert rows[3]["setup"] == "BoT-SORT + CMC (flies)" and rows[2]["setup"] == "botsort|reid-|gmc+"
    assert d["mean_ms"] >= DET_SLEEP_S * 1000 and d["n_timed"] == N_FRAMES - WARM
    for r in rows:
        assert r["error"] is None, r
        assert r["n_timed"] == N_FRAMES - WARM
        assert 0 < r["mean_ms"] < DET_SLEEP_S * 1000 <= d["mean_ms"], r   # the detector's sleep is not in it
        assert r["p95_ms"] >= 0 and r["fps"] == pytest.approx(1000 / r["mean_ms"], rel=2e-3)   # mean shown rounded
    reid_rows = [r for r in rows if r["reid"]]
    assert len(reid_rows) == 6 and all(r["overrides"]["model"] == "weights/yolo26n-cls.pt" for r in reid_rows)


def test_failures_go_in_their_row_and_do_not_abort(video, fake_reid):
    specs = tt.all_specs(tt.load_summary()[0])

    class Boom:
        def update(self, boxes, img):
            raise RuntimeError("mid-run")

    def make(name, **kw):
        if name == "ocsort":
            raise ImportError("ocsort not in this ultralytics")
        if name == "fasttrack":
            return Boom()
        return tt.make_tracker(name, **kw)
    rows, _, _ = tt.run([video], 0, N_FRAMES, FakeDetector(), specs, warmup=WARM, make=make, progress=lambda *_: None)
    assert len(rows) == 16
    bad = {r["setup"]: r["error"] for r in rows if r["error"]}
    assert set(bad) == {"ocsort", "fasttrack"}
    assert bad["ocsort"].startswith("build: ImportError") and "update (frame 0): RuntimeError" in bad["fasttrack"]
    assert all(r["mean_ms"] is not None for r in rows if not r["error"])
    assert "ERROR build: ImportError" in tt.table(rows, dict(mean_ms=1.0, p95_ms=1.0, fps=1000.0, mean_boxes=3))


def test_unknown_setting_is_an_error_not_silently_ignored():
    pytest.importorskip("ultralytics")
    with pytest.raises(KeyError):
        tt.make_tracker("bytetrack", no_such_key=1)


def test_main_writes_the_json_and_the_save_copy(video, fake_reid, tmp_path, monkeypatch, capsys):
    det = FakeDetector()
    monkeypatch.setattr(tt, "make_flight_detector", lambda args: (det, dict(
        model="fake", precision="fake", precision_source="test", cutoffs=dict(person=0.23, vehicle=0.29, source="t"),
        predict_args=dict(imgsz=640, conf=0.23, verbose=False))))
    monkeypatch.setattr(tt, "reid_weights", lambda: ("w/yolo26n-cls.pt", "re-ID encoder: test"))
    out, save = tmp_path / "logs", tmp_path / "results" / "speed.json"
    rec = tt.main(["--video", str(video), "--frames", str(N_FRAMES), "--warmup", str(WARM), "--out", str(out),
                   "--save", str(save)])
    files = list(out.glob("tracker_speed_*.json"))
    assert len(files) == 1 and save.read_bytes() == files[0].read_bytes()
    j = json.loads(save.read_text())
    assert len(j["rows"]) == 16 and det.calls == N_FRAMES
    assert j["machine"]["cpu_model"] and "ultralytics" in j["machine"]["versions"] and j["machine"]["logical_cpus"]
    assert j["settings_source"]["sha256"] == SUMMARY_SHA256 and j["video"]["frame_size"] == [320, 240]
    assert rec["detector"]["mean_ms"] >= DET_SLEEP_S * 1000
    printed = capsys.readouterr().out
    assert "bytetrack" in printed and "tracktrack|reid+|gmc+" in printed and "detector (reference)" in printed


def test_newest_flight_video_skips_fake_runs_overlays_and_empty_folders(tmp_path):
    def run(name, nick=None, files=()):
        d = tmp_path / name
        d.mkdir()
        if nick:
            (d / "drone.json").write_text(json.dumps({"drone_nickname": nick}))
        for f in files:
            (d / f).write_bytes(b"x")
        return d
    run("20261115_120000_droneA", "droneA", ["video_000.mp4", "video_001.mp4", "overlay_000.mp4"])
    run("20261122_150000_droneB", "droneB", ["overlay_000.mp4"])          # overlay only: not usable
    run("20261126_103616_droneA", "fake", ["video_000.mp4"])              # fake-drone run
    run("notes")
    folder, vids = tt.newest_flight_video(tmp_path)
    assert folder.name == "20261115_120000_droneA" and [v.name for v in vids] == ["video_000.mp4", "video_001.mp4"]
    with pytest.raises(SystemExit):
        tt.resolve_videos(str(tmp_path / "20261115_120000_droneA" / "overlay_000.mp4"))
    assert tt.newest_flight_video(tmp_path / "missing") == (None, [])


def test_flight_detector_stops_when_the_int8_export_is_missing(tmp_path):
    args = tt.build_parser().parse_args(["--model", str(tmp_path / "vd_mav_cl_s42_int8_openvino_model")])
    with pytest.raises(SystemExit, match="detector not found"):
        tt.make_flight_detector(args)


def test_reid_weights_found_or_download_note(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    p, note = tt.reid_weights([a, b])
    assert p == str(a / "yolo26n-cls.pt") and "download" in note
    (b / "yolo26n-cls.pt").write_bytes(b"x")
    p, note = tt.reid_weights([a, b])
    assert p == str(b / "yolo26n-cls.pt") and "download" not in note


def test_flies_row_is_botsort_tuned_yaml_as_live_follow_resolves_it():
    """Coordinator 11-27: the table includes the tracker that actually flies, next to the summary's BoT-SORT + CMC."""
    import live_follow as lf
    specs = tt.all_specs(tt.load_summary()[0])
    labels = [x["label"] for x in specs]
    assert len(specs) == 16 and labels.index("BoT-SORT + CMC (flies)") == labels.index("botsort|reid-|gmc+") + 1
    f = specs[labels.index("BoT-SORT + CMC (flies)")]
    trk = lf.resolve_tracker(lf.TRACKER)
    assert f["error"] is None and f["settings"] == trk["settings"] and f["overrides"] == {}
    assert f["tracker"] == "botsort" and f["gmc"] is True and f["reid"] is False
    choice = lf.load_tracker_choice()
    assert {k: f["settings"][k] for k in choice["chosen"]} == choice["chosen"] and f["settings"]["match_thresh"] == 0.9
    assert f["val_HOTA"] == next(o["class_avg_HOTA"]["val"] for o in choice["options"] if o["settings"] == choice["chosen"])
    summary_bot = specs[labels.index("botsort|reid-|gmc+")]["settings"]
    assert summary_bot["match_thresh"] == 0.95              # the summary's chosen, a different row


def test_flies_row_built_tracker_has_the_yaml_values():
    pytest.importorskip("ultralytics")
    f = next(x for x in tt.all_specs(tt.load_summary()[0]) if x["label"] == "BoT-SORT + CMC (flies)")
    t = tt.make_tracker(f["tracker"], **{**f["settings"], "device": "cpu"})
    for k, v in f["settings"].items():
        assert getattr(t.args, k) == v, k
    assert t.gmc.method is not None and t.encoder is None


def test_flies_row_error_when_the_flight_yaml_is_changed(monkeypatch):
    import live_follow as lf

    def stop(name, *a, **k):
        raise SystemExit("STOP: botsort_tuned.yaml differs from its decision record")
    monkeypatch.setattr(lf, "resolve_tracker", stop)
    specs = tt.all_specs(tt.load_summary()[0])
    f = next(x for x in specs if x["label"] == "BoT-SORT + CMC (flies)")
    assert len(specs) == 16 and f["settings"] is None and "differs" in f["error"]
