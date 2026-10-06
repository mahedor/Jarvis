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
    with pytest.raises(ValueError):
        ea.backend_for(tmp_path / "m.tflite")


def test_importing_module_does_not_pull_in_torch():
    code = ("import sys; sys.path.insert(0, 'tools'); import benchmark_export_accuracy; "
            "assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules")
    repo = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, "-c", code], cwd=repo, check=True)
