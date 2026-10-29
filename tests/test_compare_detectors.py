"""Pure-function tests for tools/compare_detectors.py (no models, no video)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
cd = pytest.importorskip("compare_detectors")


def test_parse_models_named_and_bare():
    out = cd.parse_models(["stock=yolov8n.pt", "w/vd_gray_s42.pt"])
    assert out == [("stock", Path("yolov8n.pt")), ("vd_gray_s42", Path("w/vd_gray_s42.pt"))]


def test_parse_models_rejects_duplicate_names():
    with pytest.raises(ValueError):
        cd.parse_models(["a=x.pt", "a=y.pt"])


def test_class_map_coco_keeps_only_our_classes():
    coco = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 16: "dog"}
    m = cd.class_map(coco)
    assert m[0] == "person" and m[3] == "motorcycle" and m[7] == "truck"
    assert m[16] is None


def test_frame_stats():
    dets = [("person", 0.4, (0, 0, 1, 1)), ("person", 0.9, (0, 0, 1, 1)), ("car", 0.5, (0, 0, 1, 1))]
    assert cd.frame_stats(dets) == {"n_person": 2, "best_person_conf": 0.9, "n_vehicle": 1}
    assert cd.frame_stats([]) == {"n_person": 0, "best_person_conf": 0.0, "n_vehicle": 0}


def test_grid_shape():
    tiles = [np.zeros((10, 20, 3), np.uint8)] * 3
    assert cd.grid(tiles, 2).shape == (20, 40, 3)


def test_summary_table_rates():
    s = cd.Summary(["a"])
    s.add("a", {"n_person": 1, "best_person_conf": 0.8, "n_vehicle": 0}, 10.0)
    s.add("a", {"n_person": 0, "best_person_conf": 0.0, "n_vehicle": 2}, 20.0)
    row = s.table().splitlines()[1].split()
    assert row[0] == "a" and row[1] == "2" and row[2] == "0.500" and row[3] == "0.800"


def test_load_cutoffs_by_name_or_weights_stem(tmp_path, capsys):
    f = tmp_path / "cut.json"
    f.write_text('{"vd_mav_cl_s42": {"person": 0.21, "vehicle": 0.27}, "stock": {"person": 0.1, "vehicle": 0.2}}')
    spec = [("vdmavL", Path("models/vd_mav_cl_s42.pt")), ("stock", Path("yolov8n.pt")), ("x", Path("x.pt"))]
    cuts = cd.load_cutoffs(f, spec)
    assert cuts == {"vdmavL": {"person": 0.21, "vehicle": 0.27}, "stock": {"person": 0.1, "vehicle": 0.2}}
    assert "no cutoff for x" in capsys.readouterr().out


def test_model_filters_each_group_at_its_own_cutoff():
    class Boxes:
        xyxy = [[0, 0, 1, 1]] * 4
        conf = [0.22, 0.22, 0.30, 0.10]
        cls = [0, 2, 2, 0]

        def __len__(self):
            return 4

    for attr in ("xyxy", "conf", "cls"):
        setattr(Boxes, attr, type("T", (), {"tolist": (lambda v: lambda self: v)(getattr(Boxes, attr))})())

    class R:
        boxes = Boxes()

    m = cd.Model.__new__(cd.Model)
    m.cut, m.conf, m.imgsz, m.device = {"person": 0.2, "vehicle": 0.25}, 0.2, 640, "cpu"
    m.map = {0: "person", 2: "car"}
    m.yolo = type("Y", (), {"predict": lambda self, *a, **k: [R()]})()
    dets, _ = m.detect(None)
    assert [(lab, c) for lab, c, _ in dets] == [("person", 0.22), ("car", 0.30)]
