"""
Test suite for the pure logic in tools/recognition_metrics.py.

Covers ONLY the numpy-only functions — splitting, aggregation, and the
open-set metrics — so they run in milliseconds with no crops, encoders, or
network. The I/O and orchestration in benchmark_recognition.py are exercised
separately (they need cv2/models/LFW).

Run:
  pytest tests/test_recognition_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from recognition_metrics import (  # noqa: E402  (import follows the sys.path tweak)
    AGGREGATIONS,
    average_precision,
    best_f1_threshold,
    dprime,
    operating_point,
    roc_auc,
    score_sweep,
    gallery_mean,
    gallery_medoid,
    gallery_multiref,
    histogram,
    l2_normalize,
    medoid_index,
    rank1_accuracy,
    score_multiref,
    score_single,
    source_image_key,
    split_enroll_probe,
    summarize_scores,
    tar_at_far,
)


# ─── l2_normalize ────────────────────────────────────────────────

def test_rows_become_unit_length():
    normalized = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_zero_vector_survives_without_nan():
    normalized = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert not np.isnan(normalized).any()
    assert np.allclose(normalized[0], [0.0, 0.0])


# ─── medoid_index ────────────────────────────────────────────────

def test_medoid_is_the_most_central_member():
    embeddings = np.array([[0.1], [0.2], [0.3], [10.0]])
    assert medoid_index(embeddings) == 1


def test_medoid_empty_raises():
    with pytest.raises(ValueError):
        medoid_index(np.empty((0, 4)))


# ─── source_image_key ────────────────────────────────────────────

def test_source_key_strips_face_index():
    assert source_image_key("christmas_IMG_001__f2.jpg") == "christmas_IMG_001"


def test_source_key_two_crops_same_photo_share_key():
    a = source_image_key("kitchen_IMG_9__f0.jpg")
    b = source_image_key("kitchen_IMG_9__f1.jpg")
    assert a == b == "kitchen_IMG_9"


def test_source_key_without_marker_falls_back_to_stem():
    assert source_image_key("weird_name.png") == "weird_name"


def test_source_key_ignores_directory_prefix():
    assert source_image_key("some/dir/michael_IMG_3__f0.jpg") == "michael_IMG_3"


# ─── split_enroll_probe ──────────────────────────────────────────

def test_split_is_image_disjoint():
    # Four photos, two crops each. No source photo may span both sides.
    keys = ["A", "A", "B", "B", "C", "C", "D", "D"]
    rng = np.random.default_rng(0)
    enroll_idx, probe_idx = split_enroll_probe(keys, 0.5, rng)
    enroll_keys = {keys[i] for i in enroll_idx}
    probe_keys = {keys[i] for i in probe_idx}
    assert enroll_keys.isdisjoint(probe_keys)
    assert enroll_idx and probe_idx


def test_split_is_reproducible_for_a_seed():
    keys = ["A", "A", "B", "B", "C", "C", "D", "D"]
    a = split_enroll_probe(keys, 0.5, np.random.default_rng(7))
    b = split_enroll_probe(keys, 0.5, np.random.default_rng(7))
    assert a == b


def test_single_photo_cannot_be_split():
    # All crops come from ONE photo -> no image-disjoint split possible.
    enroll_idx, probe_idx = split_enroll_probe(["A", "A", "A"], 0.5, np.random.default_rng(0))
    assert enroll_idx == [] and probe_idx == []


def test_split_keeps_both_sides_nonempty_with_two_photos():
    enroll_idx, probe_idx = split_enroll_probe(["A", "B"], 0.9, np.random.default_rng(1))
    assert len(enroll_idx) == 1 and len(probe_idx) == 1


# ─── aggregation strategies ──────────────────────────────────────

def test_mean_gallery_is_unit_length():
    enroll = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))
    gallery = gallery_mean(enroll)
    assert np.isclose(np.linalg.norm(gallery), 1.0)


def test_medoid_gallery_is_an_enroll_vector():
    enroll = l2_normalize(np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]]))
    gallery = gallery_medoid(enroll)
    assert any(np.allclose(gallery, row) for row in enroll)


def test_multiref_scores_take_the_best_matching_reference():
    # A probe identical to ONE enroll vector should score ~1 under multi-ref
    # even though the other reference is orthogonal.
    enroll = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]]))
    probe = l2_normalize(np.array([[1.0, 0.0]]))
    score = score_multiref(probe, gallery_multiref(enroll))
    assert np.isclose(score[0], 1.0, atol=1e-6)


def test_registry_pairs_build_and_score():
    enroll = l2_normalize(np.array([[1.0, 0.0], [0.8, 0.2]]))
    probe = l2_normalize(np.array([[0.9, 0.1]]))
    for build_fn, score_fn in AGGREGATIONS.values():
        scores = score_fn(probe, build_fn(enroll))
        assert scores.shape == (1,)
        assert -1.0 <= float(scores[0]) <= 1.0 + 1e-6


# ─── tar_at_far ──────────────────────────────────────────────────

def test_tar_is_one_when_perfectly_separated():
    genuine = np.array([0.9, 0.85, 0.8])
    nonmate = np.linspace(0.0, 0.3, 100)
    out = tar_at_far(genuine, nonmate, 0.01)
    assert out["tar"] == 1.0
    assert out["far_achieved"] <= 0.01 + 1e-9


def test_tar_drops_when_distributions_overlap():
    genuine = np.array([0.5, 0.4, 0.3])
    nonmate = np.linspace(0.0, 0.6, 100)  # overlaps the genuine range
    out = tar_at_far(genuine, nonmate, 0.01)
    assert out["tar"] < 1.0


def test_tar_empty_inputs_are_zero():
    out = tar_at_far(np.array([]), np.array([0.1]), 0.01)
    assert out == {"tar": 0.0, "threshold": 0.0, "far_achieved": 0.0}


# ─── rank1_accuracy ──────────────────────────────────────────────

def test_rank1_perfect_when_diagonal_dominates():
    matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert rank1_accuracy(matrix, [0, 1]) == 1.0


def test_rank1_zero_when_always_wrong():
    matrix = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert rank1_accuracy(matrix, [0, 1]) == 0.0


def test_rank1_empty_matrix_is_zero():
    assert rank1_accuracy(np.empty((0, 3)), []) == 0.0


# ─── best_f1_threshold ───────────────────────────────────────────

def test_f1_is_one_for_separable_scores():
    genuine = np.array([0.9, 0.8, 0.85])
    nonmate = np.array([0.1, 0.2, 0.15])
    threshold, precision, recall, f1 = best_f1_threshold(genuine, nonmate)
    assert f1 == pytest.approx(1.0)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.8


def test_f1_empty_inputs():
    assert best_f1_threshold(np.array([]), np.array([0.1])) == (0.0, 0.0, 0.0, 0.0)


# ─── dprime ──────────────────────────────────────────────────────

def test_dprime_positive_when_genuine_higher():
    assert dprime(np.array([0.8, 0.9]), np.array([0.1, 0.2])) > 0


def test_dprime_zero_variance_is_zero():
    assert dprime(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0


# ─── summarize_scores / histogram ────────────────────────────────

def test_summary_reports_count_and_median():
    summary = summarize_scores(list(np.linspace(0.0, 1.0, 101)))
    assert summary["count"] == 101
    assert summary["p50"] == pytest.approx(0.5)


def test_summary_empty_is_none():
    assert summarize_scores([]) is None


def test_histogram_counts_sum_to_input_size():
    values = np.array([-0.5, 0.0, 0.5, 0.9])
    hist = histogram(values, bins=10, value_range=(-1.0, 1.0))
    assert sum(hist["counts"]) == values.size
    assert len(hist["edges"]) == len(hist["counts"]) + 1


# ─── roc_auc ─────────────────────────────────────────────────────

def test_auc_is_one_when_perfectly_separated():
    assert roc_auc([0.8, 0.9], [0.1, 0.2]) == pytest.approx(1.0)


def test_auc_is_half_when_populations_are_identical():
    # Every pair is a tie, so each contributes 0.5 — chance, by construction.
    assert roc_auc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_auc_counts_ties_as_half():
    # 2x2 pairs: (0.5>0.1), (0.5>0.3), (0.2>0.1), (0.2 vs 0.3 loses) -> 3/4.
    assert roc_auc([0.5, 0.2], [0.1, 0.3]) == pytest.approx(0.75)


def test_auc_matches_the_rank_definition_on_random_scores():
    rng = np.random.default_rng(0)
    genuine, nonmate = rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 300)
    brute = np.mean(
        (genuine[:, None] > nonmate[None, :]) + 0.5 * (genuine[:, None] == nonmate[None, :])
    )
    assert roc_auc(genuine, nonmate) == pytest.approx(brute)


def test_auc_empty_is_zero_not_chance():
    assert roc_auc([], [0.1]) == 0.0
    assert roc_auc([0.1], []) == 0.0


# ─── average_precision ───────────────────────────────────────────

def test_ap_is_one_when_perfectly_separated():
    assert average_precision([0.8, 0.9], [0.1, 0.2]) == pytest.approx(1.0)


def test_ap_penalises_a_nonmate_ranked_top():
    # Ranking: 0.9(non), 0.8(gen), 0.7(gen). Recall 0.5 at precision 1/2,
    # recall 1.0 at precision 2/3 -> AP = 0.5*0.5 + 0.5*(2/3).
    ap = average_precision([0.8, 0.7], [0.9, 0.1])
    assert ap == pytest.approx(0.5 * 0.5 + 0.5 * (2.0 / 3.0))


def test_ap_is_lower_than_auc_under_heavy_imbalance():
    # The imbalance argument in the docstring: same scores, 50x more non-mates.
    rng = np.random.default_rng(1)
    genuine, nonmate = rng.normal(1.0, 1.0, 40), rng.normal(0.0, 1.0, 2000)
    assert average_precision(genuine, nonmate) < roc_auc(genuine, nonmate)


def test_ap_empty_inputs_are_zero():
    assert average_precision([], [0.1]) == 0.0


# ─── score_sweep ─────────────────────────────────────────────────

def test_sweep_has_one_point_per_grid_step_plus_endpoint():
    sweep = score_sweep([0.8], [0.1], steps=20)
    assert len(sweep) == 21
    assert sweep[0]["threshold"] == pytest.approx(-1.0)
    assert sweep[-1]["threshold"] == pytest.approx(1.0)


def test_sweep_thresholds_are_ascending_and_tar_is_non_increasing():
    rng = np.random.default_rng(2)
    sweep = score_sweep(rng.normal(0.6, 0.1, 50), rng.normal(0.0, 0.2, 200), steps=50)
    thresholds = [p["threshold"] for p in sweep]
    tars = [p["tar"] for p in sweep]
    assert thresholds == sorted(thresholds)
    assert all(a >= b for a, b in zip(tars, tars[1:]))  # raising the bar can only reject


def test_sweep_accepts_everything_at_the_bottom_of_the_range():
    bottom = score_sweep([0.8, 0.9], [0.1, 0.2], steps=10)[0]
    assert bottom["tar"] == 1.0 and bottom["far"] == 1.0
    assert bottom["precision"] == pytest.approx(0.5)  # 2 genuine of 4 accepted


def test_sweep_precision_is_none_where_nothing_is_accepted():
    # Every score is below 0.5, so grid points above it accept nothing and
    # precision is 0/0 — undefined, not zero.
    sweep = score_sweep([0.3], [0.1], steps=20)
    empty = [p for p in sweep if p["tp"] == 0 and p["fp"] == 0]
    assert empty and all(p["precision"] is None for p in empty)
    assert all(p["recall"] == 0.0 for p in empty)


def test_sweep_tar_and_recall_are_the_same_quantity():
    sweep = score_sweep([0.7, 0.4], [0.2, 0.5], steps=30)
    assert all(p["tar"] == p["recall"] for p in sweep)


def test_sweep_empty_inputs_give_no_curve():
    assert score_sweep([], [0.1]) == []


# ─── operating_point ─────────────────────────────────────────────

def test_operating_point_counts_accepts_at_an_offgrid_threshold():
    point = operating_point([0.8, 0.6], [0.55, 0.1], 0.5773)
    assert point["tp"] == 2 and point["fp"] == 0
    assert point["tar"] == 1.0 and point["far"] == 0.0
    assert point["precision"] == pytest.approx(1.0)


def test_operating_point_is_inclusive_of_the_threshold():
    # "Accept" means score >= threshold, matching the module's convention.
    assert operating_point([0.5], [0.9], 0.5)["tp"] == 1


def test_operating_point_agrees_with_tar_at_far():
    rng = np.random.default_rng(3)
    genuine, nonmate = rng.normal(0.7, 0.1, 60), rng.normal(0.1, 0.15, 3000)
    far = tar_at_far(genuine, nonmate, 0.01)
    point = operating_point(genuine, nonmate, far["threshold"])
    assert point["tar"] == pytest.approx(far["tar"], abs=1e-6)
    assert point["far"] == pytest.approx(far["far_achieved"], abs=1e-6)


def test_operating_point_precision_is_none_when_nothing_accepted():
    point = operating_point([0.3], [0.2], 0.99)
    assert point["precision"] is None and point["tp"] == 0
