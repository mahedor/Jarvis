"""
Test suite for confidence_sweep in tools/detection_metrics.py.

The sweep is what the detection threshold explorer reads, so it has to be
exactly the PR curve resampled — not an approximation that drifts from the AP
and F1 numbers printed beside it. These tests use a curve small enough to work
out by hand, so a disagreement is unambiguous.

Run:
  pytest tests/test_detection_metrics.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from detection_metrics import (  # noqa: E402
    accumulate_pr,
    best_f1_threshold,
    confidence_sweep,
)

# 4 ground-truth faces; 7 detections, 4 of them true positives.
LABELED = [(0.95, 1), (0.90, 1), (0.80, 0), (0.70, 1), (0.40, 0), (0.30, 1), (0.10, 0)]
TOTAL_GT = 4


@pytest.fixture
def curve():
    return accumulate_pr(LABELED, TOTAL_GT)


@pytest.fixture
def sweep(curve):
    precisions, recalls, thresholds = curve
    return confidence_sweep(precisions, recalls, thresholds, TOTAL_GT, steps=20)


def test_grid_spans_zero_to_one_inclusive(sweep):
    assert len(sweep) == 21
    assert sweep[0]["threshold"] == 0.0
    assert sweep[-1]["threshold"] == 1.0


def test_grid_is_ordered_by_increasing_confidence(sweep):
    thresholds = [point["threshold"] for point in sweep]
    assert thresholds == sorted(thresholds)


@pytest.mark.parametrize("confidence,tp,fp", [
    (0.20, 4, 2),   # accepts .95 .90 .80 .70 .40 .30 -> 4 TP, 2 FP
    (0.60, 3, 1),   # accepts .95 .90 .80 .70          -> 3 TP, 1 FP
    (0.80, 2, 1),   # accepts .95 .90 .80              -> 2 TP, 1 FP
])
def test_counts_match_hand_computation(sweep, confidence, tp, fp):
    point = next(p for p in sweep if p["threshold"] == pytest.approx(confidence))
    assert (point["tp"], point["fp"]) == (tp, fp)


def test_precision_recall_agree_with_the_counts(sweep):
    """precision = tp/(tp+fp) and recall = tp/total_gt at every grid point."""
    for point in sweep:
        accepted = point["tp"] + point["fp"]
        if accepted == 0:
            assert point["precision"] == 0.0 and point["recall"] == 0.0
            continue
        assert point["precision"] == pytest.approx(point["tp"] / accepted, abs=5e-4)
        assert point["recall"] == pytest.approx(point["tp"] / TOTAL_GT, abs=5e-4)


def test_recall_never_rises_as_the_threshold_rises(sweep):
    """Raising the bar can only reject detections, never add them."""
    recalls = [point["recall"] for point in sweep]
    assert all(b <= a for a, b in zip(recalls, recalls[1:]))


def test_above_every_observed_confidence_nothing_is_accepted(sweep):
    top = sweep[-1]
    assert top["tp"] == 0 and top["fp"] == 0
    assert top["precision"] == 0.0 and top["recall"] == 0.0


def test_peak_f1_on_the_grid_matches_the_exact_peak(curve, sweep):
    """The grid is coarser than the true curve, so the threshold it reports may
    differ — but the F1 VALUE it finds must not be worse than the exact one."""
    _, _, _, exact_f1 = best_f1_threshold(*curve)
    grid_f1 = max(point["f1"] for point in sweep)
    assert grid_f1 == pytest.approx(exact_f1, abs=1e-3)


def test_f1_is_consistent_with_its_own_precision_and_recall(sweep):
    for point in sweep:
        p, r = point["precision"], point["recall"]
        expected = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        assert point["f1"] == pytest.approx(expected, abs=5e-4)


def test_step_count_is_configurable(curve):
    precisions, recalls, thresholds = curve
    assert len(confidence_sweep(precisions, recalls, thresholds, TOTAL_GT, steps=50)) == 51


def test_empty_curve_yields_no_sweep():
    assert confidence_sweep([], [], [], 0) == []


def test_sweep_is_small_enough_to_store(curve):
    """The point of resampling: a fixed grid regardless of detection count."""
    precisions, recalls, thresholds = curve
    assert len(confidence_sweep(precisions, recalls, thresholds, TOTAL_GT)) == 201
