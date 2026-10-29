"""Pure-function tests for tools/duel.py (no models, no video)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
duel = pytest.importorskip("duel")


def test_match_same_box_both_and_class_difference():
    a = [("person", 0.9, (0, 0, 10, 20)), ("car", 0.8, (50, 50, 90, 80))]
    b = [("person", 0.7, (1, 1, 10, 20)), ("truck", 0.6, (52, 50, 90, 82))]
    sa, sb = duel.match(a, b)
    assert sa == ["both", "class"] and sb == ["both", "class"]


def test_match_never_pairs_person_with_vehicle():
    a = [("person", 0.9, (0, 0, 10, 20))]
    b = [("car", 0.9, (0, 0, 10, 20))]
    assert duel.match(a, b) == (["only"], ["only"])


def test_match_is_one_to_one():
    a = [("car", 0.9, (0, 0, 10, 10)), ("car", 0.8, (1, 1, 11, 11))]
    b = [("car", 0.9, (0, 0, 10, 10))]
    sa, sb = duel.match(a, b)
    assert sorted(sa) == ["both", "only"] and sb == ["both"]


def test_count_only_by_group():
    dets = [("person", 0.9, (0, 0, 1, 1)), ("truck", 0.9, (0, 0, 1, 1)), ("bus", 0.9, (0, 0, 1, 1))]
    assert duel.count_only(dets, ["only", "only", "both"]) == {"person": 1, "vehicle": 1}


def test_pick_frames_spreads_out_and_skips_agreement():
    scores = [0, 5, 6, 5, 0, 0, 0, 0, 3, 0]
    assert duel.pick_frames(scores, k=3, min_gap=3) == [2, 8]


def test_sheets_written(tmp_path):
    img = np.zeros((60, 80, 3), np.uint8)
    paths = duel.make_sheets([(i, img, img) for i in range(5)], tmp_path, per_sheet=4)
    assert [p.name for p in paths] == ["sheet_01.jpg", "sheet_02.jpg"] and all(p.exists() for p in paths)
