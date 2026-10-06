"""
Test suite for the pure helpers in tools/export_int8.py.

No nncf, no model, no zip: calibration-set selection and the record written to
results/int8_calibration.json, on hand-checkable inputs.

Run:
  pytest tests/test_export_int8.py
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import export_int8 as ei  # noqa: E402


def names(n, seqs=10):
    """n VisDrone-style names spread over `seqs` sequences, in sorted order."""
    return [f"{i % seqs:07d}_{i:05d}_d_0000001.jpg" for i in range(n)]


# ── strided_selection ──────────────────────────────────────────────

def test_strided_selection_size_and_stride():
    pool = [f"{i:04d}.jpg" for i in range(100)]
    picked = ei.strided_selection(pool, 10)
    assert picked == [f"{i:04d}.jpg" for i in range(0, 100, 10)]


def test_strided_selection_exact_n_when_not_divisible():
    pool = [f"{i:04d}.jpg" for i in range(6471)]
    picked = ei.strided_selection(pool, 300)
    assert len(picked) == 300 and len(set(picked)) == 300
    assert picked[0] == "0000.jpg" and picked[1] == "0021.jpg"     # 6471 // 300 = 21


def test_strided_selection_is_order_independent():
    pool = [f"{i:04d}.jpg" for i in range(50)]
    assert ei.strided_selection(pool, 5) == ei.strided_selection(list(reversed(pool)), 5)
    assert ei.strided_selection(set(pool), 5) == ei.strided_selection(pool, 5)


def test_strided_selection_spans_the_split_unlike_a_prefix():
    pool = sorted(f"{s:07d}_{f:05d}_d.jpg" for s in range(20) for f in range(50))   # 20 flights x 50 frames
    picked = ei.strided_selection(pool, 40)
    assert len({ei.sequence_of(n) for n in picked}) == 20
    assert len({ei.sequence_of(n) for n in sorted(pool)[:40]}) == 1               # what a prefix would give


@pytest.mark.parametrize("n,size", [(0, 10), (11, 10), (-1, 10)])
def test_strided_selection_rejects_bad_n(n, size):
    with pytest.raises(ValueError):
        ei.strided_selection([f"{i}.jpg" for i in range(size)], n)


def test_sequence_of():
    assert ei.sequence_of("0000002_00005_d_0000014.jpg") == "0000002"


# ── record ─────────────────────────────────────────────────────────

def make_record(**over):
    picked = ei.strided_selection(names(60), 6)
    kwargs = dict(
        timestamp="2026-10-05T00:00:00Z",
        model={"path": "data/models/m.pt", "sha256": "a" * 64, "size_bytes": 1},
        output={"path": "data/models/m_int8_openvino_model", "sha256": "b" * 64, "size_bytes": 2},
        imgsz=[384, 640],
        names=picked,
        total_train=60,
        step=10,
        val_overlap=0,
        versions={"nncf": "3.4.0", "numpy": "2.4.6"},
        freeze_file="results/int8_export_requirements.txt",
    )
    kwargs.update(over)
    return ei.build_record(**kwargs)


def test_record_says_which_images_and_how_many():
    r = make_record()
    c = r["calibration"]
    assert c["dataset"] == "visdrone-train"
    assert c["num_images"] == 6 == len(c["names"])
    assert c["first"] == c["names"][0] and c["last"] == c["names"][-1]
    assert c["names_sha256"] == ei.bs.names_digest(c["names"])
    assert c["val_overlap"] == 0
    assert "1 in 10 of 60" in c["selection"]
    assert r["quantization"]["subset_size"] == 6


def test_record_counts_sequences():
    r = make_record(names=["0000001_1.jpg", "0000001_2.jpg", "0000009_1.jpg"])
    assert r["calibration"]["num_sequences"] == 2


def test_record_points_at_the_freeze_file_and_versions():
    r = make_record()
    assert r["export_environment"]["freeze_file"] == "results/int8_export_requirements.txt"
    assert r["export_environment"]["versions"]["nncf"] == "3.4.0"


def test_record_is_json_round_trippable():
    r = make_record()
    assert json.loads(json.dumps(r)) == r


def test_importing_module_needs_no_nncf_or_torch():
    """The record/selection helpers must be testable from the main .venv."""
    code = ("import sys; sys.path.insert(0, 'tools'); import export_int8; "
            "assert not {'torch', 'ultralytics', 'nncf'} & set(sys.modules)")
    repo = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, "-c", code], cwd=repo, check=True)
