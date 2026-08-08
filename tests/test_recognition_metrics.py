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
    best_f1_threshold,
    dprime,
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
