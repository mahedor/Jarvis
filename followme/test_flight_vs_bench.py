"""Tests for the flight-vs-bench work (Michael 11-27): per-cycle timing in every flight (timing only, behaviour
unchanged), the two comparison trackers (built exactly as time_trackers.py builds their bench rows, flown only as
labelled deliberate deviations), the flight log's comparison / timing columns, and flight_vs_bench.py on synthetic
flights with known numbers. Also checks that the flight plan's commands in README.md and flight_plan_1128.txt parse."""
import csv
import json
import shlex
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flight_log as fl  # noqa: E402
import flight_vs_bench as fvb  # noqa: E402
import live_follow as lf  # noqa: E402
import replay  # noqa: E402
import time_trackers as tt  # noqa: E402
from test_live_follow import make_ov_export  # noqa: E402

CMP = ("bytetrack_tuned.yaml", "botsort_reid_cmc_tuned.yaml")
FIND_REID = lf.find_reid_encoder              # the real lookup (some tests replace it)


def fake_encoder(monkeypatch, tmp_path):
    """A re-ID encoder file 'on disk' (empty) and Ultralytics' ReID replaced by random vectors: nothing downloaded."""
    enc = tmp_path / "yolo26n-cls.pt"
    enc.write_bytes(b"")
    monkeypatch.setattr(lf, "find_reid_encoder", lambda model, dirs=None: enc)
    import ultralytics.trackers.utils.reid as R

    class FakeReID:
        def __init__(self, model, *a, **k):
            self.model_path, self.device = model, k.get("device")

        def __call__(self, img, dets):
            return [np.random.rand(64).astype(np.float32) for _ in range(len(dets))]
    monkeypatch.setattr(R, "ReID", FakeReID)
    return enc


# ---------------------------------------------------------------- comparison trackers

def test_comparison_yamls_are_time_trackers_construction():
    """Each file = stock Ultralytics yaml + the grid switches + the summary row's chosen settings, key for key; the
    bench rows they name exist in time_trackers.py's table."""
    pytest.importorskip("ultralytics")
    import yaml
    summary = tt.load_summary()[0]
    labels = [s["label"] for s in tt.all_specs(summary)]
    assert set(lf.COMPARISON_TRACKERS.values()) <= set(labels) and lf.FLIES_BENCH_ROW == tt.FLIES_LABEL in labels
    assert set(lf.COMPARISON_TRACKERS) == set(CMP)
    for name, label in lf.COMPARISON_TRACKERS.items():
        cfg = yaml.safe_load((HERE / name).read_text(encoding="utf-8"))
        tracker, _, _, ov = tt.parse_label(label)
        want = lf.stock_tracker_yaml(tracker)
        assert set(cfg) == set(want), name                      # same keys as the installed stock file
        want.update(ov)
        want.update(summary["per_setup"][label]["chosen"]["settings_changed"])
        assert cfg == want == lf.comparison_expected(name)[1], name
    assert yaml.safe_load((HERE / CMP[0]).read_text())["match_thresh"] == 0.95
    reid = yaml.safe_load((HERE / CMP[1]).read_text())
    assert reid["with_reid"] is True and reid["model"] == tt.REID_MODEL and reid["gmc_method"] == tt.GMC_ON


def test_comparison_yamls_build_the_bench_trackers(monkeypatch, tmp_path):
    """The tracker live_follow resolves is the one time_trackers.time_setup builds for that row (same args)."""
    pytest.importorskip("ultralytics")
    fake_encoder(monkeypatch, tmp_path)
    specs = {s["label"]: s for s in tt.all_specs(tt.load_summary()[0])}
    for name, label in lf.COMPARISON_TRACKERS.items():
        trk = lf.resolve_tracker(name)
        assert trk["bench_setup"] == label and trk["label"] == name
        s = specs[label]
        bench = tt.make_tracker(s["tracker"], **{**s["overrides"], **s["settings"], "device": "cpu"})
        for k, v in trk["settings"].items():
            if k != "model":                                    # the flight copy holds the encoder's local path
                assert getattr(bench.args, k) == v, (name, k)
    assert lf.resolve_tracker(lf.TRACKER)["bench_setup"] == tt.FLIES_LABEL


def test_edited_comparison_yaml_stops(tmp_path, monkeypatch):
    txt = (HERE / CMP[0]).read_text(encoding="utf-8").replace("match_thresh: 0.95", "match_thresh: 0.9")
    (tmp_path / CMP[0]).write_text(txt, encoding="utf-8")
    monkeypatch.setattr(lf, "HERE", tmp_path)
    with pytest.raises(SystemExit, match=r"differs from its construction[\s\S]*match_thresh: file 0.9, built 0.95"):
        lf.resolve_tracker(CMP[0])


def test_reid_encoder_missing_stops_found_is_used_and_the_committed_file_is_untouched(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    import yaml
    before = (HERE / CMP[1]).read_bytes()
    monkeypatch.setattr(lf, "find_reid_encoder", lambda model, dirs=None: None)
    with pytest.raises(SystemExit, match="needs the tracker re-ID encoder yolo26n-cls.pt"):
        lf.resolve_tracker(CMP[1])
    enc = fake_encoder(monkeypatch, tmp_path)
    trk = lf.resolve_tracker(CMP[1])
    eff = yaml.safe_load(Path(trk["path"]).read_text())
    assert eff["model"] == trk["settings"]["model"] == trk["reid_encoder"] == str(enc.resolve())
    assert {k: v for k, v in eff.items() if k != "model"} == {k: v for k, v in lf.comparison_expected(CMP[1])[1].items() if k != "model"}
    assert (HERE / CMP[1]).read_bytes() == before and trk["label"] == CMP[1]
    assert "re-ID encoder" in lf.tracker_line(trk) and "COMPARISON tracker" in lf.tracker_line(trk)
    # lookup: absolute path must exist; a bare name is searched in the given folders
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "x.pt").write_bytes(b"")
    assert FIND_REID("x.pt", dirs=[tmp_path / "nope", tmp_path / "w"]) == tmp_path / "w" / "x.pt"
    assert FIND_REID(str(tmp_path / "missing.pt")) is None and FIND_REID("x.pt", dirs=[tmp_path]) is None


def test_comparison_needs_label_and_the_label_must_name_the_deviation(tmp_path):
    trk = dict(label=CMP[0])
    assert "--label" in lf.comparison_problem(trk, "") and lf.comparison_problem(trk, "x") is None
    assert lf.comparison_problem(dict(label=lf.TRACKER), "") is None
    dev = [dict(name="tracker", actual=CMP[0])]
    assert replay.label_problem("comparison", dev) and replay.label_problem(f"comparison {CMP[0]}", dev) is None
    # end to end, no drone: without --label it stops before connecting
    base = [sys.executable, str(HERE / "live_follow.py"), "--fake", "--headless", "--seconds", "2", "--out", str(tmp_path / "out")]
    p = subprocess.run(base + ["--fake-detector", "--tracker", CMP[0]], capture_output=True, text=True, timeout=120, cwd=HERE)
    assert p.returncode != 0 and "comparison tracker" in p.stderr and "--label" in p.stderr
    # a real detector start (INT8-like export): a label that does not name the deviating tracker stops
    model = make_ov_export(tmp_path)
    p = subprocess.run(base + ["--model", str(model), "--tracker", CMP[0], "--off-decisions", "--label", "comparison"],
                       capture_output=True, text=True, timeout=120, cwd=HERE)
    assert p.returncode != 0 and "--label must name every deviating value" in p.stderr and CMP[0] in p.stderr
    assert "DELIBERATE DEVIATIONS" in p.stdout and not (tmp_path / "out").exists()


def test_live_and_replay_share_label():
    assert lf.build_parser().parse_args(["--label", "x"]).label == "x"
    assert replay.build_parser().parse_args(["--source", "s", "--label", "x"]).label == "x"


# ---------------------------------------------------------------- per-cycle timing (behaviour unchanged)

class StubTracker:
    def __init__(self):
        self.calls = []

    def update(self, det, img, **k):
        self.calls.append((det, k))
        time.sleep(0.004)
        return ("tracks", det, k.get("feats"))


class StubModel:
    """Stands in for YOLO: model.track creates the tracker on its first call and calls update inside, as Ultralytics."""
    def __init__(self):
        self.predictor, self.got = None, []

    def track(self, img, **kw):
        if self.predictor is None:
            self.predictor = types.SimpleNamespace(trackers=[StubTracker()])
        self.got.append(self.predictor.trackers[0].update("det", img, feats="f"))
        return [types.SimpleNamespace(boxes=None, speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0})]


def stub_detector():
    det = lf.Detector(None, None, 640, (0.23, 0.29), fake=True)
    det.fake, det.model, det.args = False, StubModel(), {}
    return det


def test_tracker_timer_passes_everything_through_and_times_only_update():
    det = stub_detector()
    img = np.zeros((4, 4, 3), np.uint8)
    det.process(img, 0.0)
    assert det.timing["track_ms"] is None                            # cycle 1: the tracker is created inside the call
    det.process(img, 0.0)
    det.process(img, 0.0)
    assert det.model.got == [("tracks", "det", "f")] * 3                # results and arguments unchanged
    assert det.model.predictor.trackers[0].calls == [("det", {"feats": "f"})] * 3
    assert 3.5 < det.timing["track_ms"] < 100
    assert det.timing["pre_ms"] == 1.0 and det.timing["infer_ms"] == 2.0 and det.timing["post_ms"] == 3.0
    assert det.tracker_info["tracker_class"] == "StubTracker"
    trk = det.model.predictor.trackers[0]
    wrapped = trk.update
    det.process(img, 0.0)
    assert trk.update is wrapped                                     # wrapped once, never stacked


def test_detector_thread_publishes_one_timing_record_per_cycle():
    det = lf.Detector(None, None, 640, (0.23, 0.29), fake=True)
    det.start()
    try:
        img = np.zeros((240, 320, 3), np.uint8)
        for i in range(4):
            det.submit(img, time.time())
            time.sleep(0.05)
        time.sleep(0.1)
        cyc = det.pop_cycles()
        full = det.latest_full()
    finally:
        det.stop_flag = True
        det.join(timeout=2)
    assert [c["det_n"] for c in cyc] == list(range(1, len(cyc) + 1)) and len(cyc) >= 3
    assert cyc[0]["cycle_ms"] is None and cyc[0]["idle_ms"] is None and all(c["cycle_ms"] > 0 for c in cyc[1:])
    for a, b in zip(cyc, cyc[1:]):
        assert abs((b["t_end"] - a["t_end"]) * 1000 - b["cycle_ms"]) < 1e-6
    assert all(c["box_age_ms"] >= c["busy_ms"] >= 0 and c["track_ms"] is None for c in cyc)   # fake: no tracker
    assert full[4] is cyc[-1] and full[3] == cyc[-1]["det_ms"] and det.pop_cycles() == []
    assert set(lf.CYCLE_COLS) - {"t_s"} <= set(cyc[0])


@pytest.mark.parametrize("name", ["botsort_tuned.yaml", *CMP])
def test_real_trackers_through_model_track_get_timed(name, monkeypatch, tmp_path):
    """Ultralytics' own model.track (a small untrained YOLO from its yaml, no download) with each flight tracker file:
    it runs, and from cycle 2 on track_ms is measured; the tracker class is what the file asks for."""
    pytest.importorskip("ultralytics")
    fake_encoder(monkeypatch, tmp_path)
    trk = lf.resolve_tracker(name)
    det = lf.Detector("yolov8n.yaml", trk["path"], 160, (0.0, 0.0))
    img = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
    got = []
    for _ in range(3):
        det.process(img, 0.0)
        got.append(det.timing["track_ms"])
    assert got[0] is None and all(isinstance(x, float) and x >= 0 for x in got[1:])
    want = {"botsort": "BOTSORT", "bytetrack": "BYTETracker"}[trk["settings"]["tracker_type"]]
    assert det.tracker_info["tracker_class"] == want
    assert det.tracker_info["with_reid"] is bool(trk["settings"].get("with_reid"))


def _fake_run(out, *extra, script="t@0.5,m@1,q@4", seconds=5):
    cmd = [sys.executable, str(HERE / "live_follow.py"), "--fake", "--fake-detector", "--headless", "--auto-lock",
           "--script", script, "--seconds", str(seconds), "--out", str(out), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=HERE)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout, next(x for x in out.iterdir() if x.is_dir())


def test_flight_writes_detector_csv_and_follow_csv_points_at_its_cycles(tmp_path):
    out, run = _fake_run(tmp_path, "--label", "comparison bytetrack_tuned.yaml", "--tracker", CMP[0])
    assert "COMPARISON FLIGHT: 'comparison bytetrack_tuned.yaml'" in out
    cyc = list(csv.DictReader(open(run / lf.DETECTOR_CSV)))
    rows = list(csv.DictReader(open(run / "follow.csv")))
    assert list(cyc[0]) == list(lf.CYCLE_COLS) and len(cyc) > 20
    assert [int(c["det_n"]) for c in cyc] == list(range(1, len(cyc) + 1))
    ts = [float(c["t_s"]) for c in cyc]
    assert ts == sorted(ts) and 0 <= ts[0] < ts[-1] < 6
    used = {r["det_n"] for r in rows if r["det_n"]}
    assert used <= {c["det_n"] for c in cyc} and len(used) > 10 and "track_ms" in rows[0]
    info = json.loads((run / "drone.json").read_text())
    assert info["label"] == "comparison bytetrack_tuned.yaml" and info["tracker"]["bench_setup"] == "bytetrack"
    assert info["comparison"]["bench_setup"] == "bytetrack" and info["comparison"]["tracker"] == CMP[0]
    s = fl.summarize(run)
    assert s["comparison"].startswith("comparison flight: 'comparison bytetrack_tuned.yaml' (tracker bytetrack_tuned.yaml")
    assert s["timing"].startswith(f"{len(cyc)} detector cycles, medians: cycle ") and "tracker -" in s["timing"]
    assert s["tracker"].startswith("ByteTrack tuned (comparison) (bytetrack_tuned.yaml)")
    assert "label=" not in s["other settings"] and "comparison=" not in s["other settings"]


def test_flight_log_columns_for_old_and_normal_runs(tmp_path):
    from test_flight_log import NEW_INFO, flight_rows, make_run
    make_run(tmp_path, "20261126_120000_droneA", NEW_INFO, flight_rows())
    normal = dict(NEW_INFO, label=None, comparison=None)
    make_run(tmp_path, "20261128_120000_droneA", normal, flight_rows())
    fl.update(tmp_path, quiet=True)
    r = {x["run folder"]: x for x in csv.DictReader(open(tmp_path / fl.CSV_NAME, encoding="utf-8-sig"))}
    assert r["20261126_120000_droneA"]["comparison"] == fl.NL and r["20261126_120000_droneA"]["timing"] == fl.NL
    assert r["20261128_120000_droneA"]["comparison"] == "no (normal flight)"
    assert "comparison" in fl.MD_COLUMNS and "timing" in fl.MD_COLUMNS
    assert fl.timing_summary([dict(cycle_ms=40.0, track_ms=2.0, box_age_ms=60.0, det_ms=35.0)] * 3) == \
        "3 detector cycles, medians: cycle 40.0 ms (25.0/s), model.track 35.0 ms, tracker 2.0 ms, box age 60.0 ms"


# ---------------------------------------------------------------- flight_vs_bench.py on synthetic flights

BENCH = {"bytetrack": 0.44, tt.FLIES_LABEL: 6.22, "botsort|reid+|gmc+": 25.02, "tracktrack|reid+|gmc+": 22.36,
         "slow made-up row": 40.0}
TRACK = lambda b: 0.3 + 1.2 * b               # noqa: E731  the "true" flight relations the synthetic runs follow
CYCLE = lambda b: 42.0 + 1.5 * b              # noqa: E731
FPS = lambda b: 29.0 - 0.2 * b                # noqa: E731


def flight_settings(setup):
    if setup == tt.FLIES_LABEL:
        return lf.resolve_tracker(lf.TRACKER)["settings"]
    name = next(k for k, v in lf.COMPARISON_TRACKERS.items() if v == setup)
    return lf.comparison_expected(name)[1]


def bench_rows():
    rows = []
    summary = tt.load_summary()[0]
    for setup, ms in BENCH.items():
        if setup == tt.FLIES_LABEL:
            sc, ov = lf.resolve_tracker(lf.TRACKER)["settings"], {}
        elif setup in summary["per_setup"]:
            sc = summary["per_setup"][setup]["chosen"]["settings_changed"]
            ov = tt.parse_label(setup)[3]
            ov = dict(ov, model="/home/m/followme/weights/yolo26n-cls.pt") if "model" in ov else ov
        else:
            sc, ov = {}, {}
        rows.append(dict(setup=setup, mean_ms=ms, p95_ms=ms * 1.3, mean_tracks=3.0, settings_changed=sc, overrides=ov,
                         error=None))
    return rows


def write_bench(path, rows=None, laptop=True):
    path.write_text(json.dumps(dict(made="2026-11-28T20:00:00", detector=dict(mean_ms=15.04),
                                    machine=dict(cpu_model="AMD Ryzen 7 7735HS" if laptop else "Xeon", michaels_laptop=laptop),
                                    video=dict(files=["/home/m/follow_logs/x/video_000.mp4"]),
                                    rows=rows if rows is not None else bench_rows())))
    return path


def make_flight(root, name, setup, n_cycles=80, label="comparison", nick="droneA", settings=None, detector_csv=True):
    """A run whose airborne cycles after warm-up follow TRACK / CYCLE / FPS exactly; warm-up cycles (1-20) and cycles
    used only while landed (21-30) carry absurd values, so any filter failure shows in the means."""
    b = BENCH[setup]
    run = root / name
    run.mkdir(parents=True)
    info = dict(drone_nickname=nick, name="droneA", label=label,
                tracker=dict(file="x.yaml", settings=settings or flight_settings(setup), bench_setup=setup))
    (run / "drone.json").write_text(json.dumps(info))
    cyc_ms, t, cycles = CYCLE(b), 0.0, []
    for i in range(1, n_cycles + 1):
        t += cyc_ms / 1000
        bad = i <= 30
        cycles.append(dict(det_n=i, t_s=round(t, 4), box_age_ms=999 if bad else 60.0 + b, cycle_ms=999 if bad else cyc_ms,
                           busy_ms=999 if bad else cyc_ms - 1, idle_ms=1.0, queue_ms=0.5, det_ms=30.0, pre_ms=1.0,
                           infer_ms=20.0, post_ms=1.0, track_ms=999 if bad else TRACK(b), embed_ms=0.0, n_boxes=2))
    if detector_csv:
        with open(run / fl.DETECTOR_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lf.CYCLE_COLS))
            w.writeheader()
            w.writerows(cycles)
    dt, rows, k = 1 / FPS(b), [], 0
    t_end = cycles[-1]["t_s"]
    while k * dt <= t_end:
        tr = k * dt
        n = max((c["det_n"] for c in cycles if c["t_s"] <= tr), default=None)
        landed = n is not None and 21 <= n <= 30
        rows.append(dict(frame=k, t_s=round(tr, 5), det_n=n or "", flying_state="landed" if landed else "hovering",
                         mode="FOLLOW", state="FOLLOWING", drone="droneA"))
        k += 1
    with open(run / "follow.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return run


def test_warmup_matches_the_bench():
    assert fvb.WARMUP == tt.WARMUP == fl.TIMING_WARMUP


def test_three_configs_fit_recovers_the_relation_and_says_rough(tmp_path, capsys):
    pytest.importorskip("ultralytics")
    logs = tmp_path / "logs"
    make_flight(logs, "20261128_100000_droneA", tt.FLIES_LABEL)
    make_flight(logs, "20261128_101000_droneA", "bytetrack")
    make_flight(logs, "20261128_102000_droneA", "botsort|reid+|gmc+")
    make_flight(logs, "20261128_103000_droneA", "bytetrack", settings=dict(flight_settings("bytetrack"), match_thresh=0.9))
    make_flight(logs, "20261128_104000_droneA", "bytetrack", detector_csv=False)
    make_flight(logs, "20261128_105000_droneA", "bytetrack", nick="fake")
    make_flight(logs, "20261128_106000_droneA", "bytetrack", label=None)
    make_flight(logs, "20261127_090000_droneA", "bytetrack")                      # before --since
    bench = write_bench(tmp_path / "bench.json")
    res = fvb.main(["--root", str(logs), "--bench", str(bench), "--since", "20261128", "--labelled-only",
                    "--save", str(tmp_path / "out.json")])
    text = capsys.readouterr().out
    assert [c["setup"] for c in res["configs"]] == ["bytetrack", tt.FLIES_LABEL, "botsort|reid+|gmc+"]
    for c in res["configs"]:
        b = c["bench_ms"]
        assert c["track_ms"] == pytest.approx(TRACK(b)) and c["cycle_ms"] == pytest.approx(CYCLE(b))
        assert c["loop_fps"] == pytest.approx(FPS(b), rel=1e-3) and c["box_age_median_ms"] == pytest.approx(60 + b)
        assert c["det_per_s"] == pytest.approx(1000 / CYCLE(b)) and c["airborne_only"]
        assert 48 <= c["n_cycles"] <= 50 and len(c["runs"]) == 1      # cycles 31-80 that an airborne row used
    f = res["fits"]
    assert (f["track_ms"]["a"], f["track_ms"]["b"]) == (pytest.approx(0.3), pytest.approx(1.2))
    assert (f["cycle_ms"]["a"], f["cycle_ms"]["b"]) == (pytest.approx(42.0), pytest.approx(1.5))
    assert f["loop_fps"]["b"] == pytest.approx(-0.2, abs=1e-3)
    assert all(abs(r) < 1e-6 for r in f["cycle_ms"]["residuals"]) and "rough fit" in f["cycle_ms"]["note"]
    why = dict((s["run"], s["why"]) for s in res["skipped"])
    assert "differ from bench row 'bytetrack'" in why["20261128_103000_droneA"] and "match_thresh" in why["20261128_103000_droneA"]
    assert "no detector.csv" in why["20261128_104000_droneA"] and why["20261128_105000_droneA"] == "fake-drone run"
    assert why["20261128_106000_droneA"] == "not a labelled comparison flight" and "20261127_090000_droneA" not in why
    pred = {p["setup"]: p for p in res["predictions"]}
    assert pred["tracktrack|reid+|gmc+"]["cycle_ms"] == pytest.approx(CYCLE(22.36)) and not pred["tracktrack|reid+|gmc+"]["extrapolated"]
    assert pred["slow made-up row"]["extrapolated"] and pred["slow made-up row"]["det_per_s"] == pytest.approx(1000 / CYCLE(40))
    assert "cycle ms  = 42.00 + 1.500 x bench_ms" in text and "rough fit" in text and "treat as rough" in text
    assert "* slow made-up row" in text and "skipped 20261128_103000_droneA" in text
    assert json.loads((tmp_path / "out.json").read_text())["fits"]["cycle_ms"]["n"] == 3


def test_two_configs_fit_exactly_and_one_config_fits_nothing(tmp_path, capsys):
    pytest.importorskip("ultralytics")
    logs = tmp_path / "logs"
    make_flight(logs, "20261128_100000_droneA", tt.FLIES_LABEL)
    make_flight(logs, "20261128_100500_droneA", tt.FLIES_LABEL)                 # two runs of one config are pooled
    bench = write_bench(tmp_path / "bench.json", laptop=False)
    res = fvb.main(["--root", str(logs), "--bench", str(bench)])
    text = capsys.readouterr().out
    assert len(res["configs"]) == 1 and 96 <= res["configs"][0]["n_cycles"] <= 100 and all(v is None for v in res["fits"].values())
    assert "one config only" in text and "not run on Michael's laptop" in text and res["predictions"] == []
    make_flight(logs, "20261128_101000_droneA", "bytetrack")
    res = fvb.main(["--root", str(logs), "--bench", str(bench)])
    assert res["fits"]["track_ms"]["n"] == 2 and "exactly through both" in res["fits"]["track_ms"]["note"]
    assert res["fits"]["track_ms"]["r2"] is None and "exactly through both" in capsys.readouterr().out


def test_linfit_with_noise_reports_residuals_and_r2():
    ft = fvb.linfit([0.0, 1.0, 2.0, 3.0], [1.0, 3.1, 4.9, 7.0])
    assert ft["b"] == pytest.approx(1.98) and ft["a"] == pytest.approx(1.03) and 0.99 < ft["r2"] < 1
    assert sum(ft["residuals"]) == pytest.approx(0) and "2 degrees of freedom" in ft["note"]
    assert fvb.linfit([1.0], [2.0]) is None and fvb.linfit([1.0, 1.0], [2.0, 3.0]) is None


def test_find_bench_newest_then_fallback_then_stop(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    fb = tmp_path / "fallback.json"
    with pytest.raises(SystemExit, match="no bench JSON"):
        fvb.find_bench(None, logs, fb)
    fb.write_text("{}")
    assert fvb.find_bench(None, logs, fb) == fb
    for n in ("tracker_speed_20261127_210000.json", "tracker_speed_20261128_190000.json"):
        (logs / n).write_text("{}")
    assert fvb.find_bench(None, logs, fb).name == "tracker_speed_20261128_190000.json"
    with pytest.raises(SystemExit, match="bench file not found"):
        fvb.find_bench(str(tmp_path / "nope.json"), logs, fb)


def test_a_real_fake_drone_run_goes_through_flight_vs_bench(tmp_path, capsys):
    _, run = _fake_run(tmp_path / "logs", "--label", "comparison botsort_tuned.yaml")
    bench = write_bench(tmp_path / "bench.json")
    res = fvb.main([str(run), "--bench", str(bench), "--include-fake"])
    (c,) = res["configs"]
    assert c["setup"] == tt.FLIES_LABEL and c["n_cycles"] > 10 and c["loop_fps"] > 5 and c["track_ms"] is None
    assert c["airborne_only"] is False                         # the fake drone never reports airborne: every cycle
    assert c["labels"] == ["comparison botsort_tuned.yaml"] and "one config only" in capsys.readouterr().out
    assert fvb.main([str(run), "--bench", str(bench)])["skipped"][0]["why"] == "fake-drone run"


# ---------------------------------------------------------------- the flight plan's commands

def _plan_commands(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith(("python ", "python3 "))]


@pytest.mark.parametrize("source", ["README.md", "flight_plan_1128.txt"])
def test_flight_plan_commands_parse_and_label_every_deviation(source):
    text = (HERE / source).read_text(encoding="utf-8")
    if source == "README.md":
        text = text[text.index("### Flight plan 11-28"):text.index("## Change log")]
    cmds = _plan_commands(text)
    flights = [c for c in cmds if c.startswith("python live_follow.py") and "--fake" not in c]
    assert len(flights) == 3 and len([c for c in cmds if "live_follow.py --fake" in c]) == 2
    seen = set()
    for c in cmds:
        argv = shlex.split(c)
        if argv[1] == "live_follow.py":
            a = lf.build_parser().parse_args(argv[2:])
            seen.add(a.tracker)
            dev = [] if a.tracker == lf.TRACKER else [dict(name="tracker", actual=a.tracker)]
            assert a.label and replay.label_problem(a.label, dev) is None, c
            assert a.off_decisions == bool(dev) and not lf.comparison_problem(dict(label=a.tracker), a.label), c
        elif argv[1] == "flight_vs_bench.py":
            fvb.build_parser().parse_args(argv[2:])
        elif argv[1] == "time_trackers.py":
            tt.build_parser().parse_args(argv[2:])
        elif argv[1] == "flight_log.py":
            pass
        else:
            assert argv[1] == "-c", c
    assert seen == {lf.TRACKER, *CMP}
    assert text.count("cd /mnt/c/Users/mahed/OneDrive/Documents/Jarvis/jarvis-lean/followme") == 3
    assert text.count("source ~/jarvis-follow/bin/activate") == 3
