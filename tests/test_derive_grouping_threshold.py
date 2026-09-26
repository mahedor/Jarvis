"""
Test suite for the pure logic in tools/derive_grouping_threshold.py.

Covers box selection, pair sampling, the FMR/FNMR maths and the artifact's
shape — everything that needs no LFW download, no YOLO and no ArcFace.

Run:
  pytest tests/test_derive_grouping_threshold.py
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import derive_grouping_threshold as dgt  # noqa: E402  (import follows the sys.path tweak above)
import pipeline_config  # noqa: E402


def rng(seed=0):
    return np.random.default_rng(seed)


# ── select_central_detection ────────────────────────────────────────

def test_central_box_beats_a_larger_off_centre_bystander():
    shape = (100, 100, 3)
    central = {"box": [40, 40, 60, 60], "confidence": 0.9}
    bystander = {"box": [0, 0, 45, 45], "confidence": 0.95}  # bigger, off centre
    assert dgt.select_central_detection([bystander, central], shape) is central


def test_central_selection_handles_empty_and_single():
    assert dgt.select_central_detection([], (100, 100, 3)) is None
    only = {"box": [0, 0, 10, 10], "confidence": 0.6}
    assert dgt.select_central_detection([only], (100, 100, 3)) is only


# ── sample_genuine_pairs ────────────────────────────────────────────

def test_genuine_pairs_stay_within_identity_and_are_ordered():
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    pairs = dgt.sample_genuine_pairs(labels, cap=10, rng=rng())
    assert np.all(labels[pairs[:, 0]] == labels[pairs[:, 1]])
    assert np.all(pairs[:, 0] < pairs[:, 1])


def test_genuine_pairs_take_all_when_under_cap():
    # C(3,2)=3, C(2,2)=1, C(4,2)=6 -> 10, all under a cap of 10.
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    pairs = dgt.sample_genuine_pairs(labels, cap=10, rng=rng())
    assert len(pairs) == 10
    assert len({tuple(p) for p in pairs}) == 10


def test_genuine_pairs_cap_gives_equal_weight_to_heavy_identities():
    # One identity with 200 images (19,900 possible pairs), five with 6 (15 each).
    labels = np.array([0] * 200 + [i for i in range(1, 6) for _ in range(6)])
    pairs = dgt.sample_genuine_pairs(labels, cap=10, rng=rng())
    per_identity = Counter(labels[pairs[:, 0]].tolist())
    assert set(per_identity.values()) == {10}
    assert len(per_identity) == 6


def test_genuine_pairs_are_sampled_without_replacement():
    labels = np.zeros(50, dtype=int)
    pairs = dgt.sample_genuine_pairs(labels, cap=10, rng=rng())
    assert len({tuple(p) for p in pairs}) == 10


def test_singleton_identities_contribute_no_genuine_pairs():
    labels = np.array([0, 1, 2, 2])
    pairs = dgt.sample_genuine_pairs(labels, cap=10, rng=rng())
    assert pairs.tolist() == [[2, 3]]
    assert dgt.sample_genuine_pairs(np.array([0, 1]), cap=10, rng=rng()).shape == (0, 2)


def test_genuine_pairs_are_deterministic_for_a_seed():
    labels = np.repeat(np.arange(20), 8)
    a = dgt.sample_genuine_pairs(labels, 10, rng(7))
    b = dgt.sample_genuine_pairs(labels, 10, rng(7))
    c = dgt.sample_genuine_pairs(labels, 10, rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ── sample_impostor_pairs ───────────────────────────────────────────

def test_impostor_pairs_always_cross_identities():
    labels = np.array([0, 0, 0, 1, 2, 2, 5, 5])
    pairs = dgt.sample_impostor_pairs(labels, 5000, rng())
    assert pairs.shape == (5000, 2)
    assert np.all(labels[pairs[:, 0]] != labels[pairs[:, 1]])


def test_impostor_pairs_weight_identities_not_images():
    # Identity 0 owns 97% of the images but must appear only ~1/3 of the time.
    labels = np.array([0] * 970 + [1] * 15 + [2] * 15)
    pairs = dgt.sample_impostor_pairs(labels, 30000, rng())
    share = np.mean(labels[pairs.ravel()] == 0)
    assert share == pytest.approx(1 / 3, abs=0.02)


def test_impostor_pairs_reach_every_image_of_an_identity():
    labels = np.array([0, 0, 0, 1, 1])
    pairs = dgt.sample_impostor_pairs(labels, 2000, rng())
    assert set(pairs.ravel().tolist()) == {0, 1, 2, 3, 4}


def test_impostor_pairs_need_two_identities():
    with pytest.raises(ValueError):
        dgt.sample_impostor_pairs(np.array([3, 3, 3]), 10, rng())


def test_impostor_pairs_are_deterministic_for_a_seed():
    labels = np.repeat(np.arange(30), 3)
    assert np.array_equal(dgt.sample_impostor_pairs(labels, 100, rng(1)),
                          dgt.sample_impostor_pairs(labels, 100, rng(1)))


# ── scoring ─────────────────────────────────────────────────────────

def test_pair_scores_are_cosines_of_unit_vectors():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [np.sqrt(0.5), np.sqrt(0.5)]], dtype=np.float32)
    scores = dgt.pair_scores(emb, np.array([[0, 1], [0, 2], [2, 2]]))
    assert scores == pytest.approx([0.0, np.sqrt(0.5), 1.0], abs=1e-6)
    assert dgt.pair_scores(emb, np.empty((0, 2))).size == 0


def test_error_rates_use_ge_for_match():
    genuine = [0.2, 0.5, 0.8]
    impostor = [0.1, 0.5, 0.6, 0.9]
    rates = dgt.error_rates(genuine, impostor, 0.5)
    assert rates["fnmr"] == pytest.approx(1 / 3)   # only 0.2 < 0.5
    assert rates["fmr"] == pytest.approx(3 / 4)    # 0.5, 0.6, 0.9 >= 0.5


def test_threshold_at_fnmr_hits_the_target_exactly_on_distinct_scores():
    genuine = np.linspace(0.0, 1.0, 1000)
    for target in (0.10, 0.05, 0.01):
        t = dgt.threshold_at_fnmr(genuine, target)
        assert np.mean(genuine < t) == pytest.approx(target)


def test_threshold_at_fnmr_never_exceeds_target_with_ties():
    genuine = np.array([0.3] * 50 + [0.7] * 50)
    t = dgt.threshold_at_fnmr(genuine, 0.10)
    assert np.mean(genuine < t) <= 0.10
    assert dgt.threshold_at_fnmr([], 0.1) == 0.0


def _synthetic_scores(seed=0):
    r = rng(seed)
    genuine = r.normal(0.6, 0.1, 5000)
    impostor = r.normal(0.05, 0.08, 100_000)
    return genuine, impostor


def test_operating_points_hit_their_targets():
    genuine, impostor = _synthetic_scores()
    ops = dgt.compute_operating_points(genuine, impostor)

    assert ops["primary"]["target_fnmr"] == dgt.PRIMARY_FNMR == 0.10
    assert ops["primary"]["fnmr"] == pytest.approx(0.10, abs=1e-3)
    assert [p["target_fnmr"] for p in ops["fnmr_targets"]] == [0.10, 0.05, 0.01]
    for p in ops["fnmr_targets"]:
        assert p["fnmr"] <= p["target_fnmr"] + 1e-9
    assert [p["target_fmr"] for p in ops["fmr_targets"]] == [0.01, 0.001]
    for p in ops["fmr_targets"]:
        assert p["fmr"] <= p["target_fmr"] + 1e-9

    # Stricter FNMR -> lower threshold -> more merges.
    thresholds = [p["threshold"] for p in ops["fnmr_targets"]]
    assert thresholds == sorted(thresholds, reverse=True)
    assert 0.0 < ops["best_f1"]["f1"] <= 1.0


def test_dprime_is_the_equal_weight_pooled_form():
    genuine, impostor = _synthetic_scores()
    expected = (genuine.mean() - impostor.mean()) / np.sqrt(0.5 * (genuine.var() + impostor.var()))
    assert dgt.compute_operating_points(genuine, impostor)["dprime"] == pytest.approx(expected, rel=1e-5)


def test_error_sweep_is_monotone_on_a_fixed_grid():
    genuine, impostor = _synthetic_scores()
    sweep = dgt.error_sweep(genuine, impostor)
    assert len(sweep) == dgt.SWEEP_STEPS + 1
    assert sweep[0]["threshold"] == -1.0 and sweep[-1]["threshold"] == 1.0
    fmr = [row["fmr"] for row in sweep]
    fnmr = [row["fnmr"] for row in sweep]
    assert fmr == sorted(fmr, reverse=True)
    assert fnmr == sorted(fnmr)


# ── LFW loading ─────────────────────────────────────────────────────

def test_slice_to_json_records_rows_then_cols():
    assert dgt.slice_to_json(dgt.LFW_SLICES["sklearn-default"]) == [[70, 195], [78, 172]]
    assert dgt.slice_to_json(dgt.LFW_SLICES["full"]) == [[0, 250], [0, 250]]


def _fake_lfw(root, counts):
    for name, n in counts.items():
        folder = root / name
        folder.mkdir()
        for i in range(n):
            (folder / f"{name}_{i + 1:04d}.jpg").write_bytes(b"")
    return root


def test_lfw_file_index_matches_sklearn_enumeration(tmp_path):
    _fake_lfw(tmp_path, {"Carol_C": 3, "Alice_A": 2, "Bob_B": 1})
    (tmp_path / "stray.txt").write_text("not a person")
    paths, labels, names = dgt.lfw_file_index(tmp_path, min_faces_per_person=2)

    assert names.tolist() == ["Alice A", "Carol C"]      # Bob dropped, "_" -> " "
    assert len(paths) == 5
    for path, label in zip(paths, labels):
        assert names[label] == path.parent.name.replace("_", " ")

    # Same shuffle as sklearn's _fetch_lfw_people.
    unshuffled = sorted(paths, key=lambda p: (p.parent.name, p.name))
    order = np.arange(5)
    np.random.RandomState(42).shuffle(order)
    assert paths == [unshuffled[i] for i in order]


def test_lfw_file_index_rejects_an_empty_subset(tmp_path):
    _fake_lfw(tmp_path, {"Solo": 1})
    with pytest.raises(ValueError):
        dgt.lfw_file_index(tmp_path, min_faces_per_person=2)


def test_lazy_lfw_decodes_the_slice_as_bgr_uint8(tmp_path):
    import cv2

    bgr = np.zeros((250, 250, 3), dtype=np.uint8)
    bgr[:, :, 0] = 200                      # blue everywhere
    bgr[70:195, 78:172, 2] = 100            # red only inside the default window
    path = tmp_path / "face.png"            # lossless, so pixels are exact
    cv2.imwrite(str(path), bgr)

    full = dgt.LazyLFW([path], dgt.LFW_SLICES["full"])
    tight = dgt.LazyLFW([path], dgt.LFW_SLICES["sklearn-default"])
    assert full.shape() == (250, 250) and tight.shape() == (125, 94)

    image = full[0]
    assert image.dtype == np.uint8 and image.shape == (250, 250, 3)
    assert abs(int(image[0, 0, 0]) - 200) <= 1 and image[0, 0, 2] == 0
    window = tight[0]
    assert window.shape == (125, 94, 3)
    assert abs(int(window[0, 0, 2]) - 100) <= 1   # the crop starts inside the red


# ── the artifact ────────────────────────────────────────────────────

def _payload():
    genuine, impostor = _synthetic_scores()
    return dgt.build_payload(
        genuine=genuine, impostor=impostor,
        settings={"encoder": "arcface"}, dataset={"identities_loaded": 3},
        pipeline_counts={"images": 1}, pairs={"seed": 42},
    )


def test_payload_has_every_required_section_and_is_json_serialisable():
    payload = _payload()
    for key in ("timestamp", "caveat", "settings", "dataset", "pipeline_counts", "pairs",
                "distributions", "operating_points", "sweep", "histograms"):
        assert key in payload
    assert payload["adopted"] is False
    import sklearn
    assert payload["sklearn_version"] == sklearn.__version__
    for name in ("genuine", "impostor"):
        assert {"mean", "std"} <= set(payload["distributions"][name])
    json.dumps(payload)


def test_caveat_names_funneled_lfw_and_higher_live_fmr():
    caveat = dgt.CAVEAT.lower()
    assert "funneled lfw" in caveat
    assert "press photography" in caveat
    assert "webcam" in caveat
    assert "fmr will be higher" in caveat


def test_save_results_appends(tmp_path):
    path = tmp_path / "results" / "grouping_threshold.json"
    dgt.save_results({"run": 1}, path)
    dgt.save_results({"run": 2}, path)
    assert json.loads(path.read_text()) == [{"run": 1}, {"run": 2}]


# ── settings that must match deployment ─────────────────────────────

def test_pipeline_settings_match_deployment():
    assert dgt.ENCODER == "arcface"
    assert dgt.CROP_MARGIN == 0.35
    assert dgt.MIN_BOX_SIZE == 40
    assert dgt.GATE_MIN_CONFIDENCE == pipeline_config.DETECTION_CONFIDENCE_THRESHOLD == 0.57
    assert dgt.GENUINE_CAP_PER_IDENTITY == 10
    assert dgt.NUM_IMPOSTOR_PAIRS == 100_000
    assert dgt.MIN_FACES_PER_PERSON == 2
    assert dgt.LFW_KWARGS == {"color": True, "resize": 1.0, "funneled": True}
    # Deployment sees whole frames, so the full image is the default.
    assert dgt.DEFAULT_SLICE == "full"
    assert dgt.LFW_SLICES["full"] == (slice(0, 250), slice(0, 250))


def test_writes_to_results_and_never_touches_the_recognition_threshold():
    assert dgt.RESULTS_FILE.parent.name == "results"
    assert dgt.RESULTS_FILE.name == "grouping_threshold.json"
    # build_gallery owns the recognition threshold and the gallery. Checked in a
    # fresh interpreter, since this test process may already have imported it.
    tools = Path(dgt.__file__).resolve().parent
    probe = (f"import sys; sys.path.insert(0, {str(tools)!r}); "
             "import derive_grouping_threshold; "
             "print('build_gallery' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    assert not any("gallery" in name.lower() for name in dir(dgt))
