"""
Test suite for the pure logic in tools/collect_faces.py.

These cover ONLY the functions that are free of cv2, disk, and models —
quality_check, l2_normalize, medoid_index, summarize_distribution — which
is why they run in milliseconds and need no photos, no weights, and no
network.

Run:
  pytest tests/test_collect_faces.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from collect_faces import (  # noqa: E402  (import follows the sys.path tweak above)
    REJECT_LOW_CONFIDENCE,
    REJECT_TOO_BLURRY,
    REJECT_TOO_SMALL,
    l2_normalize,
    medoid_index,
    quality_check,
    summarize_distribution,
)

# A box whose shortest side is 100px, comfortably over any default.
GOOD_BOX = [0, 0, 120, 100]

DEFAULTS = {"min_box_size": 40, "min_confidence": 0.5}


# ─── quality_check ───────────────────────────────────────────────

def test_good_face_is_kept():
    result = quality_check(GOOD_BOX, confidence=0.9, blur_score=500.0, **DEFAULTS)
    assert result.keep is True
    assert result.reason is None


def test_low_confidence_is_rejected_with_reason():
    result = quality_check(GOOD_BOX, confidence=0.2, blur_score=500.0, **DEFAULTS)
    assert result.keep is False
    assert result.reason == REJECT_LOW_CONFIDENCE


def test_small_box_is_rejected_on_its_shortest_side():
    # 200px wide but only 30px tall: the shortest side is what disqualifies it.
    result = quality_check([0, 0, 200, 30], confidence=0.9, blur_score=500.0, **DEFAULTS)
    assert result.keep is False
    assert result.reason == REJECT_TOO_SMALL


def test_blur_check_is_off_by_default():
    # A very soft crop passes when min_blur is not supplied — that is the
    # "run once, read the distribution, then pick a cutoff" workflow.
    result = quality_check(GOOD_BOX, confidence=0.9, blur_score=1.0, **DEFAULTS)
    assert result.keep is True


def test_blurry_crop_is_rejected_when_min_blur_is_set():
    result = quality_check(GOOD_BOX, confidence=0.9, blur_score=10.0, min_blur=50.0, **DEFAULTS)
    assert result.keep is False
    assert result.reason == REJECT_TOO_BLURRY


def test_unmeasured_blur_cannot_reject():
    # blur_score=None means "not measured"; a None must never be treated as 0.
    result = quality_check(GOOD_BOX, confidence=0.9, blur_score=None, min_blur=50.0, **DEFAULTS)
    assert result.keep is True


def test_thresholds_are_inclusive_at_the_boundary():
    # Exactly at each threshold is a PASS (the checks are strict <).
    result = quality_check(
        [0, 0, 40, 40], confidence=0.5, blur_score=50.0, min_blur=50.0, **DEFAULTS
    )
    assert result.keep is True


def test_confidence_is_checked_before_size():
    # A crop failing BOTH is attributed to confidence: a box we do not believe
    # is a face at all makes its size meaningless. Ordering keeps the reject
    # counts in the report interpretable.
    result = quality_check([0, 0, 10, 10], confidence=0.1, blur_score=500.0, **DEFAULTS)
    assert result.reason == REJECT_LOW_CONFIDENCE


def test_size_is_checked_before_blur():
    result = quality_check(
        [0, 0, 10, 10], confidence=0.9, blur_score=1.0, min_blur=50.0, **DEFAULTS
    )
    assert result.reason == REJECT_TOO_SMALL


# ─── l2_normalize ────────────────────────────────────────────────

def test_rows_become_unit_length():
    embeddings = np.array([[3.0, 4.0], [1.0, 0.0], [-5.0, 12.0]])
    normalized = l2_normalize(embeddings)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_direction_is_preserved():
    # Normalizing must only change magnitude, never direction.
    normalized = l2_normalize(np.array([[3.0, 4.0]]))
    assert np.allclose(normalized, [[0.6, 0.8]])


def test_zero_vector_survives_without_nan():
    # A degenerate embedding must not poison the whole matrix with NaN, which
    # would blow up HDBSCAN for every other crop in the run.
    normalized = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert not np.isnan(normalized).any()
    assert np.allclose(normalized[0], [0.0, 0.0])
    assert np.allclose(normalized[1], [0.6, 0.8])


def test_single_vector_keeps_its_1d_shape():
    normalized = l2_normalize(np.array([3.0, 4.0]))
    assert normalized.shape == (2,)
    assert np.allclose(normalized, [0.6, 0.8])


def test_output_is_float32():
    # ONNX/sklearn are happiest on float32, and it halves the memory of a big run.
    assert l2_normalize(np.array([[3.0, 4.0]], dtype=np.float64)).dtype == np.float32


# ─── medoid_index ────────────────────────────────────────────────

def test_medoid_is_the_most_central_member():
    # Four points bunched near x=[0.1..0.3] plus one outlier far away at 10.0.
    # The medoid must be the middle of the bunch (index 1), never the outlier.
    embeddings = np.array([[0.1], [0.2], [0.3], [10.0]])
    assert medoid_index(embeddings) == 1


def test_medoid_is_an_actual_member_not_an_average():
    # The mean of these three is [0, 0], which is NOT one of the points. The
    # medoid must be a real index, since we write the medoid CROP to disk.
    embeddings = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    assert medoid_index(embeddings) == 2


def test_single_member_cluster():
    assert medoid_index(np.array([[1.0, 2.0]])) == 0


def test_empty_cluster_raises():
    with pytest.raises(ValueError):
        medoid_index(np.empty((0, 512)))


# ─── summarize_distribution ──────────────────────────────────────

def test_percentiles_of_a_known_range():
    summary = summarize_distribution(list(range(101)))  # 0..100
    assert summary["count"] == 101
    assert summary["min"] == 0.0
    assert summary["max"] == 100.0
    assert summary["p50"] == pytest.approx(50.0)
    assert summary["p5"] == pytest.approx(5.0)
    assert summary["p95"] == pytest.approx(95.0)


def test_empty_distribution_is_none():
    assert summarize_distribution([]) is None
