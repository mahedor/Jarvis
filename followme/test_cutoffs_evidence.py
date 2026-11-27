"""The flight cutoffs can't drift from the evidence (mirrors tests/test_pipeline_config.py for the face pipeline).

Two sweeps, both outputs of thresholds_balanced.py (Colab), committed byte-exact from Drive:
  * .pt models (Oct 28-29, committed 11-26): results/followme_cutoffs_balanced.json + followme_thresholds_balanced.json
  * the INT8 OpenVINO export of vd_mav_cl_s42 that actually flies (11-26, same method; only change: one image per
    predict call at imgsz=[384,640] because the export is static 384x640, the way quantize_int8.py scored it):
    results/followme_cutoffs_balanced_int8.json + followme_thresholds_balanced_int8.json (entry vd_mav_cl_s42_int8).
Each thresholds file holds, per model label and per test set, the F1 at every confidence of the sweep (0.05-0.95 in
0.01 steps = 91 points; the grid itself is not stored, only the curves). The method: equal-weight mean of the three
sets' F1, take the argmax, one cutoff for people and one for vehicles. This recomputes that from the curves and requires
it to equal the cutoffs file, which is what live_follow.py flies at: the INT8 model from the INT8 file (0.23 / 0.29),
never from its .pt entry (0.24 / 0.28). All four files are committed, so a missing file FAILS here (no skip).
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
CUTOFFS_INT8 = RESULTS / "followme_cutoffs_balanced_int8.json"
CURVES_INT8 = RESULTS / "followme_thresholds_balanced_int8.json"
SETS = {"visdrone_val", "mavrec_val", "coco_holdout"}       # the three equal-weight sets of thresholds_balanced.py
GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]        # 0.05..0.95 step 0.01
GROUPS = ("person", "vehicle")
PT_MODEL = HERE / "weights" / "vd_mav_cl_s42.pt"


def _load(p):
    assert p.is_file(), f"{p} is committed evidence and must exist"
    return json.loads(p.read_text())


CUT, CUR = _load(CUTOFFS), _load(CURVES)
CUT8, CUR8 = _load(CUTOFFS_INT8), _load(CURVES_INT8)
SWEEPS = {"pt": (CUT, CUR), "int8": (CUT8, CUR8)}
ENTRIES = [(sweep, model) for sweep, (cut, _) in SWEEPS.items() for model in sorted(cut)]


def recompute(curves, group):
    """-> (cutoff, mean F1 there, {set: F1 there}, mean curve) from the per-set curves of one model label."""
    mean = [sum(curves[s][group][i] for s in SETS) / len(SETS) for i in range(len(GRID))]
    i = max(range(len(GRID)), key=mean.__getitem__)
    return GRID[i], mean[i], {s: curves[s][group][i] for s in SETS}, mean


def test_files_cover_the_same_models_and_the_flight_model():
    for cut, cur in SWEEPS.values():
        assert {e["label"] for e in cut.values()} == set(cur)
    assert set(CUT8) == {"vd_mav_cl_s42_int8"}
    assert lf.cutoffs_file_for(lf.INT8_MODEL) == lf.CUTOFFS_FILE_INT8 == CUTOFFS_INT8
    assert lf.cutoffs_file_for(PT_MODEL) == lf.CUTOFFS_FILE == CUTOFFS
    assert lf.cutoff_key(lf.INT8_MODEL, CUT8) == "vd_mav_cl_s42_int8"
    assert lf.cutoff_key(PT_MODEL, CUT) == "vd_mav_cl_s42"


@pytest.mark.parametrize("sweep,model", ENTRIES)
def test_curves_have_the_documented_shape(sweep, model):
    cut, cur = SWEEPS[sweep]
    curves = cur[cut[model]["label"]]
    assert set(curves) == SETS
    for s in SETS:
        for g in GROUPS:
            assert len(curves[s][g]) == len(GRID), f"{model} {s} {g}: expected the 91-point 0.05-0.95 sweep"


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("sweep,model", ENTRIES)
def test_cutoff_is_the_argmax_of_the_equal_weight_mean_f1(sweep, model, group):
    cut, cur = SWEEPS[sweep]
    e = cut[model]
    c, f1, per_set, mean = recompute(cur[e["label"]], group)
    assert c == e[group], f"{model} {group}: curves say {c}, cutoffs file says {e[group]}"
    assert round(f1, 4) == e[f"{group}_mean_f1"]
    assert {s: round(v, 4) for s, v in per_set.items()} == e[f"{group}_f1_per_set"]
    assert sum(1 for v in mean if v == f1) == 1, "tied maximum: the argmax would depend on tie-breaking"


def test_flight_model_flies_at_the_recomputed_int8_cutoffs():
    """INT8 flight model -> 0.23 / 0.29, read from the INT8 file (not hardcoded), equal to the INT8 curves' argmax."""
    p, v, src = lf.resolve_cutoffs(lf.INT8_MODEL)
    assert (p, v) == (0.23, 0.29)
    assert src == "F1-max sweep for vd_mav_cl_s42_int8, results/followme_cutoffs_balanced_int8.json"
    curves = CUR8[CUT8["vd_mav_cl_s42_int8"]["label"]]
    assert (p, v) == (recompute(curves, "person")[0], recompute(curves, "vehicle")[0])


def test_pt_model_keeps_the_pt_cutoffs():
    p, v, src = lf.resolve_cutoffs(PT_MODEL)
    assert (p, v) == (0.24, 0.28)
    assert src == "F1-max sweep for vd_mav_cl_s42, results/followme_cutoffs_balanced.json"
    label = CUT["vd_mav_cl_s42"]["label"]
    assert (p, v) == (recompute(CUR[label], "person")[0], recompute(CUR[label], "vehicle")[0])


def test_int8_never_falls_back_to_the_pt_entry(tmp_path, monkeypatch):
    # the .pt file has vd_mav_cl_s42 but no vd_mav_cl_s42_int8: STOP, not 0.24 / 0.28
    with pytest.raises(SystemExit, match="STOP: no F1-max cutoffs for vd_mav_cl_s42_int8_openvino_model"):
        lf.resolve_cutoffs(lf.INT8_MODEL, path=CUTOFFS)
    # the INT8 file present but without the INT8 entry
    other = tmp_path / "int8_other.json"
    other.write_text(json.dumps({"vd_mav_cl_s42": CUT["vd_mav_cl_s42"]}))
    monkeypatch.setattr(lf, "CUTOFFS_FILE_INT8", other)
    with pytest.raises(SystemExit, match="STOP: no F1-max cutoffs"):
        lf.resolve_cutoffs(lf.INT8_MODEL)
    # the INT8 file gone (the .pt file still there)
    monkeypatch.setattr(lf, "CUTOFFS_FILE_INT8", tmp_path / "gone.json")
    with pytest.raises(SystemExit, match="STOP: F1-max cutoffs file not found"):
        lf.resolve_cutoffs(lf.INT8_MODEL)
    assert lf.resolve_cutoffs(PT_MODEL)[:2] == (0.24, 0.28)


def test_unknown_or_lookalike_model_stops(tmp_path):
    for name in ("other.pt", "weights/vd_mav_cl_s42_v2.pt", "vd_mav_cl", "vd_mav_cl_s42_int8.pt",
                 "weights/vd_mav_cl_s42_v2_int8_openvino_model", "weights/vd_mav_cl_int8_openvino_model"):
        with pytest.raises(SystemExit, match="STOP: no F1-max cutoffs"):
            lf.resolve_cutoffs(name)


def test_missing_cutoffs_file_stops_unless_both_given(tmp_path):
    gone = tmp_path / "nope.json"
    with pytest.raises(SystemExit, match="STOP: F1-max cutoffs file not found"):
        lf.resolve_cutoffs(lf.INT8_MODEL, path=gone)
    with pytest.raises(SystemExit):
        lf.resolve_cutoffs(lf.INT8_MODEL, conf_person=0.3, path=gone)       # one value is not "on purpose"
    assert lf.resolve_cutoffs(lf.INT8_MODEL, 0.3, 0.4, path=gone) == (0.3, 0.4, "set on the command line")
