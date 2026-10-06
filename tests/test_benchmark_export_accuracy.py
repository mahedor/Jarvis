"""
Test suite for the pure helpers in tools/benchmark_export_accuracy.py.

No model, no images: label conversion, label decoding, the FP32 tolerance
comparison and backend routing, on hand-checkable inputs.

Run:
  pytest tests/test_benchmark_export_accuracy.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import benchmark_export_accuracy as ea  # noqa: E402


# ── VisDrone -> YOLO conversion ────────────────────────────────────

def test_visdrone_to_yolo_converts_and_normalizes():
    # 100x50 box at (10, 20) in a 200x100 image, category 4 (car) -> class 3
    lines = ea.visdrone_to_yolo("10,20,100,50,1,4,0,0\n", 200, 100)
    assert lines == ["3 0.300000 0.450000 0.500000 0.500000\n"]


def test_visdrone_to_yolo_drops_score_zero_rows():
    text = "\n".join([
        "0,0,10,10,0,0,0,0",      # ignored region
        "0,0,10,10,0,11,0,0",     # "others": score 0 in VisDrone val
        "0,0,10,10,1,1,0,0",      # pedestrian, kept
    ])
    lines = ea.visdrone_to_yolo(text, 100, 100)
    assert len(lines) == 1 and lines[0].startswith("0 ")


def test_visdrone_to_yolo_empty():
    assert ea.visdrone_to_yolo("", 100, 100) == []
    assert ea.visdrone_to_yolo("\n\n", 100, 100) == []


def test_yolo_round_trip_recovers_pixel_box():
    lines = ea.visdrone_to_yolo("10,20,100,50,1,4,0,0", 200, 100)
    classes, boxes = ea.yolo_to_xyxy("".join(lines), 200, 100)
    assert classes == [3]
    assert boxes[0] == pytest.approx([10, 20, 110, 70], abs=1e-3)


def test_yolo_to_xyxy_empty():
    assert ea.yolo_to_xyxy("", 640, 384) == ([], [])


# ── FP32 tolerance ─────────────────────────────────────────────────

def row(backend, m50, m):
    return {"backend": backend, "map50": m50, "map50_95": m}


def test_compare_to_baseline_deltas_and_flags():
    out = ea.compare_to_baseline([
        row("pytorch", 0.300, 0.170),
        row("onnx", 0.302, 0.169),        # within 0.005 on both
        row("openvino", 0.300, 0.163),    # 50-95 off by 0.007
    ])
    by = {r["backend"]: r for r in out}
    assert by["pytorch"]["delta_map50"] == 0 and by["pytorch"]["within_tolerance"]
    assert by["onnx"]["delta_map50"] == 0.002 and by["onnx"]["delta_map50_95"] == -0.001
    assert by["onnx"]["within_tolerance"] is True
    assert by["openvino"]["delta_map50_95"] == -0.007
    assert by["openvino"]["within_tolerance"] is False


def test_compare_to_baseline_boundary_is_inclusive():
    out = ea.compare_to_baseline([row("pytorch", 0.3, 0.17), row("onnx", 0.305, 0.165)])
    assert out[1]["within_tolerance"] is True


def test_compare_to_baseline_does_not_mutate_and_needs_baseline():
    rows = [row("pytorch", 0.3, 0.17)]
    ea.compare_to_baseline(rows)
    assert "delta_map50" not in rows[0]
    with pytest.raises(ValueError):
        ea.compare_to_baseline([row("onnx", 0.3, 0.17)])


# ── backend routing ────────────────────────────────────────────────

def test_backend_for_routes_by_weights(tmp_path):
    ov_dir = tmp_path / "m_openvino_model"
    ov_dir.mkdir()
    (ov_dir / "m.xml").write_text("<net/>")
    assert ea.backend_for(tmp_path / "m.pt") == "pytorch"
    assert ea.backend_for(tmp_path / "m.onnx") == "onnx"
    assert ea.backend_for(ov_dir) == "openvino"
    q_dir = tmp_path / "m_int8_openvino_model"
    q_dir.mkdir()
    (q_dir / "m.xml").write_text('<net><layer type="FakeQuantize"/></net>')
    assert ea.backend_for(q_dir) == "openvino-int8"
    with pytest.raises(ValueError):
        ea.backend_for(tmp_path / "m.tflite")


# ── INT8 reporting ─────────────────────────────────────────────────

def test_compare_to_baseline_tolerance_is_fp32_only():
    out = ea.compare_to_baseline([
        {**row("pytorch", 0.30, 0.17), "dtype": "fp32"},
        {**row("openvino-int8", 0.29, 0.16), "dtype": "int8"},
    ])
    assert out[0]["within_tolerance"] is True
    assert out[1]["within_tolerance"] is None          # not judged by the FP32 promise
    assert out[1]["delta_map50"] == -0.01


def qrow(backend, m50, m, per_class):
    return {"backend": backend, "map50": m50, "map50_95": m,
            "per_class": {n: {"ap50": a, "ap50_95": b, "instances": 10} for n, (a, b) in per_class.items()}}


def test_quantization_drop_math_and_flags():
    fp32 = qrow("openvino", 0.30, 0.17, {"car": (0.80, 0.60), "people": (0.20, 0.08), "bus": (0.50, 0.30)})
    int8 = qrow("openvino-int8", 0.29, 0.165, {"car": (0.79, 0.57), "people": (0.17, 0.08), "bus": (0.51, 0.30)})
    q = ea.quantization_drop(int8, fp32)
    assert q["map50_drop"] == 0.01 and q["map50_95_drop"] == 0.005    # positive = lost
    assert q["per_class"]["car"]["ap50_drop"] == 0.01
    assert q["per_class"]["car"]["ap50_95_drop"] == 0.03               # flagged on AP50-95 alone
    assert q["per_class"]["people"]["ap50_drop"] == 0.03               # flagged on AP50 alone
    assert q["per_class"]["bus"]["ap50_drop"] == -0.01                 # a gain is never flagged
    assert q["flagged_classes"] == ["car", "people"]


def test_quantization_drop_threshold_is_strict():
    fp32 = qrow("openvino", 0.3, 0.17, {"car": (0.80, 0.60)})
    int8 = qrow("openvino-int8", 0.3, 0.17, {"car": (0.78, 0.58)})    # exactly 0.02
    assert ea.quantization_drop(int8, fp32)["flagged_classes"] == []


def test_quantization_drop_missing_class_counts_as_zero():
    fp32 = qrow("openvino", 0.3, 0.17, {"car": (0.8, 0.6), "truck": (0.1, 0.05)})
    int8 = qrow("openvino-int8", 0.3, 0.17, {"car": (0.8, 0.6)})
    q = ea.quantization_drop(int8, fp32)
    assert q["per_class"]["truck"]["ap50_quant"] == 0.0
    assert q["flagged_classes"] == ["truck"]


def test_quantization_drops_pairs_int8_with_same_runtime_fp32():
    rows = [qrow("pytorch", 0.31, 0.18, {"car": (0.9, 0.7)}),
            qrow("openvino", 0.30, 0.17, {"car": (0.8, 0.6)}),
            qrow("openvino-int8", 0.29, 0.16, {"car": (0.8, 0.6)})]
    drops = ea.quantization_drops(rows)
    assert len(drops) == 1 and drops[0]["reference"] == "openvino"
    assert drops[0]["map50_drop"] == 0.01
    assert ea.quantization_drops(rows[:2]) == []                       # no int8 row: nothing to report


def test_format_table_shows_int8_section():
    rows = [{**qrow("pytorch", 0.30, 0.17, {"car": (0.8, 0.6)}), "dtype": "fp32"},
            {**qrow("openvino", 0.30, 0.17, {"car": (0.8, 0.6)}), "dtype": "fp32"},
            {**qrow("openvino-int8", 0.27, 0.15, {"car": (0.7, 0.5)}), "dtype": "int8"}]
    entry = {"dataset": {"num_images": 3}, "imgsz": [384, 640], "conf": 0.001, "iou": 0.7,
             "tolerance": 0.005, "results": ea.compare_to_baseline(rows),
             "quantization": ea.quantization_drops(rows), "reference": None}
    text = ea.format_table(entry)
    assert "n/a" in text and "FLAG" in text and "flagged: car" in text


def test_importing_module_does_not_pull_in_torch():
    code = ("import sys; sys.path.insert(0, 'tools'); import benchmark_export_accuracy; "
            "assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules")
    repo = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, "-c", code], cwd=repo, check=True)
