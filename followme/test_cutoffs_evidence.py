"""The flight cutoffs can't drift from the evidence (mirrors tests/test_pipeline_config.py for the face pipeline).

results/followme_cutoffs_balanced.json and results/followme_thresholds_balanced.json are the outputs of
thresholds_balanced.py (Colab, Oct 28-29), committed byte-exact from Drive on 11-26. The thresholds file holds, per
model label and per test set, the F1 at every confidence of the sweep (0.05-0.95 in 0.01 steps = 91 points; the grid
itself is not stored, only the curves). The method: equal-weight mean of the three sets' F1, take the argmax, one cutoff
for people and one for vehicles. This recomputes that from the curves and requires it to equal the cutoffs file, which
is what live_follow.py flies at. Both files are committed, so a missing file FAILS here (no skip).
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import live_follow as lf  # noqa: E402

RESULTS = HERE.parent / "results"
CUTOFFS = RESULTS / "followme_cutoffs_balanced.json"
CURVES = RESULTS / "followme_thresholds_balanced.json"
SETS = {"visdrone_val", "mavrec_val", "coco_holdout"}       # the three equal-weight sets of thresholds_balanced.py
GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]        # 0.05..0.95 step 0.01
GROUPS = ("person", "vehicle")


def _load(p):
    assert p.is_file(), f"{p} is committed evidence and must exist"
    return json.loads(p.read_text())


CUT, CUR = _load(CUTOFFS), _load(CURVES)


def recompute(curves, group):
    """-> (cutoff, mean F1 there, {set: F1 there}, mean curve) from the per-set curves of one model label."""
    mean = [sum(curves[s][group][i] for s in SETS) / len(SETS) for i in range(len(GRID))]
    i = max(range(len(GRID)), key=mean.__getitem__)
    return GRID[i], mean[i], {s: curves[s][group][i] for s in SETS}, mean


def test_files_cover_the_same_models_and_the_flight_model():
    assert {e["label"] for e in CUT.values()} == set(CUR)
    assert lf.cutoff_key(lf.INT8_MODEL, CUT) == "vd_mav_cl_s42"


@pytest.mark.parametrize("model", sorted(CUT))
def test_curves_have_the_documented_shape(model):
    curves = CUR[CUT[model]["label"]]
    assert set(curves) == SETS
    for s in SETS:
        for g in GROUPS:
            assert len(curves[s][g]) == len(GRID), f"{model} {s} {g}: expected the 91-point 0.05-0.95 sweep"


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("model", sorted(CUT))
def test_cutoff_is_the_argmax_of_the_equal_weight_mean_f1(model, group):
    e = CUT[model]
    cut, f1, per_set, mean = recompute(CUR[e["label"]], group)
    assert cut == e[group], f"{model} {group}: curves say {cut}, cutoffs file says {e[group]}"
    assert round(f1, 4) == e[f"{group}_mean_f1"]
    assert {s: round(v, 4) for s, v in per_set.items()} == e[f"{group}_f1_per_set"]
    assert sum(1 for v in mean if v == f1) == 1, "tied maximum: the argmax would depend on tie-breaking"


def test_flight_model_flies_at_the_recomputed_cutoffs():
    """INT8 flight model -> 0.24 / 0.28, read from the file (not hardcoded), equal to the recomputed argmax."""
    p, v, src = lf.resolve_cutoffs(lf.INT8_MODEL)
    assert (p, v) == (0.24, 0.28)
    assert "results/followme_cutoffs_balanced.json" in src and "vd_mav_cl_s42" in src
    label = CUT["vd_mav_cl_s42"]["label"]
    assert (p, v) == (recompute(CUR[label], "person")[0], recompute(CUR[label], "vehicle")[0])


def test_unknown_or_lookalike_model_stops():
    for name in ("other.pt", "weights/vd_mav_cl_s42_v2.pt", "vd_mav_cl"):
        with pytest.raises(SystemExit, match="STOP: no F1-max cutoffs"):
            lf.resolve_cutoffs(name)


def test_missing_cutoffs_file_stops_unless_both_given(tmp_path):
    gone = tmp_path / "nope.json"
    with pytest.raises(SystemExit, match="STOP: F1-max cutoffs file not found"):
        lf.resolve_cutoffs(lf.INT8_MODEL, path=gone)
    with pytest.raises(SystemExit):
        lf.resolve_cutoffs(lf.INT8_MODEL, conf_person=0.3, path=gone)       # one value is not "on purpose"
    assert lf.resolve_cutoffs(lf.INT8_MODEL, 0.3, 0.4, path=gone) == (0.3, 0.4, "set on the command line")
