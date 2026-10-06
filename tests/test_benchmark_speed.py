"""
Test suite for the stats math and result format in tools/benchmark_speed.py.

No model, no torch, no images: benchmark_speed keeps every heavy import inside
the functions that need it, so these tests exercise the numbers that get
written to results/benchmarks_speed.json on hand-checkable inputs.

Run:
  pytest tests/test_benchmark_speed.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import benchmark_speed as bs  # noqa: E402


def make_timings(e2e):
    """Per-stage timings where pre/infer/post are fixed fractions of e2e."""
    return {
        "preprocess": [v * 0.1 for v in e2e],
        "inference": [v * 0.8 for v in e2e],
        "postprocess": [v * 0.1 for v in e2e],
        "end_to_end": list(e2e),
    }


def make_repeat(e2e, dets=None):
    dets = dets if dets is not None else [3] * len(e2e)
    return bs.summarize_repeat(make_timings(e2e), dets)


# ── percentile / stage_stats ───────────────────────────────────────

def test_percentile_nearest_rank():
    vals = list(range(1, 101))            # 1..100
    assert bs.percentile(vals, 50) == 51   # round(0.5 * 99) = 50 -> vals[50]
    assert bs.percentile(vals, 95) == 95   # round(0.95 * 99) = 94 -> vals[94]
    assert bs.percentile(vals, 0) == 1
    assert bs.percentile(vals, 100) == 100


def test_percentile_empty_and_single():
    assert bs.percentile([], 95) == 0.0
    assert bs.percentile([7.0], 50) == 7.0


def test_stage_stats_hand_computed():
    st = bs.stage_stats([40.0, 10.0, 30.0, 20.0])   # deliberately unsorted
    assert st["mean_ms"] == 25.0
    assert st["min_ms"] == 10.0
    assert st["max_ms"] == 40.0
    assert st["p50_ms"] == 30.0                      # round(0.5*3)=2 -> 30
    assert st["p95_ms"] == 40.0


def test_stage_stats_empty():
    assert bs.stage_stats([]) == {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                                  "min_ms": 0.0, "max_ms": 0.0}


def test_fps_is_1000_over_mean():
    assert bs.fps_from_mean(50.0) == 20.0
    assert bs.fps_from_mean(33.333) == 30.0
    assert bs.fps_from_mean(0.0) == 0.0


# ── summarize_repeat ───────────────────────────────────────────────

def test_summarize_repeat_fps_uses_end_to_end_mean():
    r = make_repeat([40.0, 60.0], dets=[2, 5])
    assert r["num_images"] == 2
    assert r["stages"]["end_to_end"]["mean_ms"] == 50.0
    assert r["stages"]["inference"]["mean_ms"] == 40.0
    assert r["fps"] == 20.0
    assert r["mean_detections"] == 3.5


def test_summarize_repeat_rejects_mismatched_lengths():
    t = make_timings([10.0, 20.0])
    t["inference"].pop()
    with pytest.raises(ValueError):
        bs.summarize_repeat(t, [1, 1])
    with pytest.raises(ValueError):
        bs.summarize_repeat(make_timings([10.0, 20.0]), [1])


def test_summarize_repeat_rejects_missing_stage():
    t = make_timings([10.0])
    del t["postprocess"]
    with pytest.raises(ValueError):
        bs.summarize_repeat(t, [1])


# ── spread / summarize_repeats ─────────────────────────────────────

def test_spread_uses_median_and_relative_range():
    sp = bs.spread([100.0, 110.0, 180.0])     # one hot, throttled repeat
    assert sp["median"] == 110.0               # not dragged to the mean (130)
    assert sp["min"] == 100.0 and sp["max"] == 180.0
    assert sp["spread_pct"] == round(80 / 110 * 100, 2)


def test_spread_identical_values_is_zero():
    assert bs.spread([5.0, 5.0, 5.0])["spread_pct"] == 0.0


def test_spread_empty():
    assert bs.spread([]) == {"median": 0.0, "min": 0.0, "max": 0.0, "spread_pct": 0.0}


def test_summarize_repeats_headline_fps_matches_headline_latency():
    repeats = [make_repeat([50.0] * 4), make_repeat([40.0] * 4), make_repeat([100.0] * 4)]
    s = bs.summarize_repeats(repeats)
    e2e = s["stages"]["end_to_end"]["mean"]
    assert e2e["median"] == 50.0
    assert s["fps"] == 20.0                    # 1000 / median e2e, not median fps
    assert s["fps_range"] == [10.0, 25.0]
    assert e2e["spread_pct"] == 120.0          # (100-40)/50
    assert s["detections_consistent"] is True


def test_summarize_repeats_flags_detection_mismatch():
    s = bs.summarize_repeats([make_repeat([10.0], dets=[3]), make_repeat([10.0], dets=[4])])
    assert s["detections_consistent"] is False


def test_summarize_repeats_empty_raises():
    with pytest.raises(ValueError):
        bs.summarize_repeats([])


# ── result record ──────────────────────────────────────────────────

ENV = {
    "cpu": "Test CPU", "os": "test", "python": "3.12", "logical_cpus": 16,
    "physical_cores": 8, "torch_threads": 8,
    "power": {"plugged_in": True, "battery_percent": 80},
    "versions": {"torch": "x", "ultralytics": "y", "onnxruntime": "z", "openvino": None, "numpy": "n"},
}


def make_entry(**over):
    kwargs = dict(
        timestamp="2026-09-30T00:00:00Z",
        backend="pytorch",
        model={"path": "data/models/m.pt", "sha256": "ab" * 32, "size_bytes": 123},
        imgsz=640,
        conf=0.172,
        image_names=["a.jpg", "b.jpg", "c.jpg"],
        warmup=10,
        environment=ENV,
        repeats=[make_repeat([10.0, 20.0, 30.0]) for _ in range(3)],
        input_shapes={"1x3x384x640": 3},
    )
    kwargs.update(over)
    return bs.build_entry(**kwargs)


def test_entry_records_every_setting_that_changes_the_numbers():
    e = make_entry()
    for key in ("timestamp", "backend", "model", "imgsz", "conf", "dataset",
                "warmup_images", "num_repeats", "input_shapes", "environment",
                "results", "summary"):
        assert key in e, key
    assert e["model"]["sha256"] == "ab" * 32
    assert e["num_repeats"] == 3 and len(e["results"]) == 3
    env = e["environment"]
    for key in ("cpu", "torch_threads", "power", "versions"):
        assert key in env
    for lib in ("torch", "ultralytics", "onnxruntime"):
        assert lib in env["versions"]


def test_entry_dataset_identifies_the_image_set():
    e = make_entry()
    ds = e["dataset"]
    assert ds["num_images"] == 3
    assert ds["first"] == "a.jpg" and ds["last"] == "c.jpg"
    assert ds["names_sha256"] == bs.names_digest(["a.jpg", "b.jpg", "c.jpg"])
    assert ds["names_sha256"] != bs.names_digest(["a.jpg", "b.jpg", "d.jpg"])


def test_entry_is_json_round_trippable():
    e = make_entry()
    assert json.loads(json.dumps(e)) == e


def test_entry_matches_other_benchmark_files_shape():
    """Every results/benchmarks_*.json is a list of dicts with timestamp + results."""
    e = make_entry()
    assert isinstance(e["timestamp"], str)
    assert isinstance(e["results"], list)


def test_append_entry_creates_then_appends(tmp_path):
    path = tmp_path / "sub" / "benchmarks_speed.json"
    bs.append_entry(make_entry(timestamp="t1"), path)
    bs.append_entry(make_entry(timestamp="t2"), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [d["timestamp"] for d in data] == ["t1", "t2"]


def test_append_entry_refuses_non_list(tmp_path):
    path = tmp_path / "benchmarks_speed.json"
    path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError):
        bs.append_entry(make_entry(), path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"not": "a list"}


def test_format_table_mentions_headline_numbers():
    e = make_entry(environment={**ENV, "power": {"plugged_in": False, "battery_percent": 40}})
    text = bs.format_table(e)
    assert "ON BATTERY" in text
    assert f"{e['summary']['fps']:.2f}" in text
    for stage in bs.STAGES:
        assert stage in text


# ── test set selection / registry ──────────────────────────────────

def test_select_images_is_sorted_prefix(tmp_path):
    for name in ["c.jpg", "a.jpg", "b.png", "notes.txt", "d.jpg"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in bs.select_images(tmp_path, 3)] == ["a.jpg", "b.png", "c.jpg"]


def test_select_images_too_few_raises(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    with pytest.raises(ValueError):
        bs.select_images(tmp_path, 2)


def test_backend_registry_entries_implement_the_interface():
    assert "pytorch" in bs.BACKENDS
    for cls in bs.BACKENDS.values():
        assert issubclass(cls, bs.Backend)
        for method in ("load", "preprocess", "infer", "postprocess"):
            assert getattr(cls, method) is not getattr(bs.Backend, method), (cls, method)


def test_importing_module_does_not_pull_in_torch():
    """Keeps this test file (and anything importing the stats) model-free."""
    import subprocess
    code = ("import sys; sys.path.insert(0, 'tools'); import benchmark_speed; "
            "assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules")
    repo = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, "-c", code], cwd=repo, check=True)


# ── imgsz / fingerprint / backend selection ────────────────────────

def test_parse_imgsz_square_and_rect():
    assert bs.parse_imgsz("640") == 640
    assert bs.parse_imgsz("384,640") == [384, 640]
    assert bs.parse_imgsz("384x640") == [384, 640]
    for bad in ("", "0", "1,2,3", "-5", "a"):
        with pytest.raises(ValueError):
            bs.parse_imgsz(bad)


def test_model_fingerprint_file_matches_file_hash(tmp_path):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"weights")
    fp = bs.model_fingerprint(f)
    assert fp == {"sha256": bs.file_sha256(f), "size_bytes": 7}


def test_model_fingerprint_directory_tracks_names_and_contents(tmp_path):
    d = tmp_path / "m_openvino_model"
    d.mkdir()
    (d / "m.xml").write_bytes(b"xml")
    (d / "m.bin").write_bytes(b"bin!")
    first = bs.model_fingerprint(d)
    assert first["size_bytes"] == 7
    (d / "m.bin").write_bytes(b"bin?")
    assert bs.model_fingerprint(d)["sha256"] != first["sha256"]
    (d / "m.bin").write_bytes(b"bin!")
    assert bs.model_fingerprint(d) == first
    (d / "m.bin").rename(d / "n.bin")
    assert bs.model_fingerprint(d)["sha256"] != first["sha256"]


def make_ov_dir(parent, name, quantized=False):
    d = parent / name
    d.mkdir()
    layer = '<layer id="1" type="FakeQuantize"/>' if quantized else '<layer id="1" type="Convolution"/>'
    (d / "m.xml").write_text(f"<net><layers>{layer}</layers></net>")
    return d


def test_backends_only_accept_their_own_weights(tmp_path):
    ov = make_ov_dir(tmp_path, "m_openvino_model")
    q = make_ov_dir(tmp_path, "m_int8_openvino_model", quantized=True)
    pt, onnx = tmp_path / "m.pt", tmp_path / "m.onnx"
    accepted = {name: [p for p in (pt, onnx, ov, q) if cls.accepts(p)] for name, cls in bs.BACKENDS.items()}
    assert accepted == {"pytorch": [pt], "onnx": [onnx], "openvino": [ov], "openvino-int8": [q]}


def test_int8_is_detected_by_contents_not_folder_name(tmp_path):
    """A quantized IR in a folder without 'int8' in its name is still INT8, and vice versa."""
    renamed_q = make_ov_dir(tmp_path, "plain_openvino_model", quantized=True)
    misnamed_fp32 = make_ov_dir(tmp_path, "x_int8_openvino_model")
    assert bs.is_quantized_ir(renamed_q) and not bs.is_quantized_ir(misnamed_fp32)
    assert bs.OpenVINOInt8Backend.accepts(renamed_q) and not bs.OpenVINOBackend.accepts(renamed_q)
    assert bs.OpenVINOBackend.accepts(misnamed_fp32) and not bs.OpenVINOInt8Backend.accepts(misnamed_fp32)


def test_backend_precision_labels():
    assert {name: cls.precision for name, cls in bs.BACKENDS.items()} == {
        "pytorch": "fp32", "onnx": "fp32", "openvino": "fp32", "openvino-int8": "int8"}


def test_openvino_backends_are_pinned_to_cpu():
    """AUTO could hand inference to the RTX 3050 / iGPU; see OpenVINOBackend."""
    assert bs.OpenVINOBackend.device == "intel:cpu"
    assert bs.OpenVINOInt8Backend.device == "intel:cpu"


def test_entry_records_runtime():
    e = make_entry(runtime={"torch_threads": 8})
    assert e["runtime"] == {"torch_threads": 8}
    assert "runtime: torch_threads=8" in bs.format_table(e)
    assert make_entry()["runtime"] == {}
