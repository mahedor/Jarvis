"""
Test suite for tools/pipeline_config.py — the locked operating points.

These tests are the anti-drift mechanism. The threshold in pipeline_config is a
human decision, so nothing stops someone editing the constant; what stops it
becoming a magic number is that every claim it makes about the evidence is
re-checked here against results/benchmarks_detection.json. Change the value
without re-measuring, or let the provenance fall out of step with the stored
runs, and this fails.

Run:
  pytest tests/test_pipeline_config.py
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from pipeline_config import (  # noqa: E402  (import follows the sys.path tweak)
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_DETECTOR,
    DETECTION_THRESHOLD_PROVENANCE,
)

BENCH_FILE = REPO_ROOT / "results" / "benchmarks_detection.json"


@pytest.fixture(scope="module")
def runs():
    if not BENCH_FILE.is_file():
        pytest.skip(f"{BENCH_FILE} not present")
    with open(BENCH_FILE) as f:
        return json.load(f)


def _find_run(runs, timestamp):
    for run in runs:
        if run.get("timestamp") == timestamp:
            return run
    return None


def _yolo_result(run):
    for result in run.get("results", []):
        if result.get("detector") == DETECTION_DETECTOR and "error" not in result:
            return result
    return None


# ─── the provenance describes runs that actually exist ───────────

@pytest.mark.parametrize("measurement", DETECTION_THRESHOLD_PROVENANCE["measurements"],
                         ids=lambda m: m["dataset"])
def test_cited_run_exists(runs, measurement):
    run = _find_run(runs, measurement["run_timestamp"])
    assert run is not None, (
        f"pipeline_config cites a {measurement['dataset']} run at "
        f"{measurement['run_timestamp']} that is not in {BENCH_FILE.name}"
    )
    assert run.get("dataset") == measurement["dataset"]


@pytest.mark.parametrize("measurement", DETECTION_THRESHOLD_PROVENANCE["measurements"],
                         ids=lambda m: m["dataset"])
def test_cited_numbers_still_match_the_artifact(runs, measurement):
    """The recorded evidence must equal what the benchmark actually stored."""
    run = _find_run(runs, measurement["run_timestamp"])
    result = _yolo_result(run)
    assert result is not None, f"no {DETECTION_DETECTOR} result in the cited run"

    assert run["num_images"] == measurement["num_images"]
    assert result["f1_threshold"] == pytest.approx(measurement["f1_threshold"], abs=1e-4)
    assert result["f1"] == pytest.approx(measurement["f1"], abs=1e-4)
    assert result["ap"] == pytest.approx(measurement["ap"], abs=1e-4)


@pytest.mark.parametrize("measurement", DETECTION_THRESHOLD_PROVENANCE["measurements"],
                         ids=lambda m: m["dataset"])
def test_cited_runs_are_full_dataset(runs, measurement):
    """A threshold argued from a --limit sample would not be worth locking."""
    run = _find_run(runs, measurement["run_timestamp"])
    assert run.get("limit") is None, (
        f"{measurement['dataset']} run was limited to {run.get('limit')} images"
    )


@pytest.mark.parametrize("measurement", DETECTION_THRESHOLD_PROVENANCE["measurements"],
                         ids=lambda m: m["dataset"])
def test_cited_runs_share_one_iou_threshold(runs, measurement):
    """F1-max thresholds from different IoU settings are not comparable."""
    run = _find_run(runs, measurement["run_timestamp"])
    assert run["iou_threshold"] == DETECTION_THRESHOLD_PROVENANCE["iou_threshold"]


# ─── the shipped value follows from that evidence ────────────────

def test_shipped_value_matches_the_provenance_block():
    assert DETECTION_CONFIDENCE_THRESHOLD == DETECTION_THRESHOLD_PROVENANCE["value"]


def test_shipped_value_is_bracketed_by_its_supporting_datasets():
    """0.57 must sit within the span of the hard datasets it claims support
    from. Edit the constant without re-measuring and this is what fails."""
    supporting = DETECTION_THRESHOLD_PROVENANCE["supported_by"]
    thresholds = [m["f1_threshold"]
                  for m in DETECTION_THRESHOLD_PROVENANCE["measurements"]
                  if m["dataset"] in supporting]
    assert len(thresholds) == len(supporting), "a supporting dataset has no measurement"

    low, high = min(thresholds), max(thresholds)
    # Allow rounding to 2dp below the lower bound (0.5727 -> 0.57).
    assert low - 0.005 <= DETECTION_CONFIDENCE_THRESHOLD <= high, (
        f"{DETECTION_CONFIDENCE_THRESHOLD} is outside the range its evidence "
        f"supports ({low}-{high} from {supporting})"
    )


def test_supporting_datasets_actually_agree():
    """The argument for locking this value is that two independent hard sets
    converge. If a re-measure pulls them apart, the value needs rethinking, not
    silently keeping."""
    thresholds = [m["f1_threshold"]
                  for m in DETECTION_THRESHOLD_PROVENANCE["measurements"]
                  if m["dataset"] in DETECTION_THRESHOLD_PROVENANCE["supported_by"]]
    assert max(thresholds) - min(thresholds) < 0.05, (
        f"supporting datasets disagree by {max(thresholds) - min(thresholds):.3f}; "
        f"the 'two datasets converge' argument no longer holds"
    )


def test_excluded_datasets_are_marked_as_such():
    """FDDB is cited for context but must not be counted as support."""
    for measurement in DETECTION_THRESHOLD_PROVENANCE["measurements"]:
        if measurement["dataset"] in DETECTION_THRESHOLD_PROVENANCE["supported_by"]:
            continue
        assert "EXCLUDED" in measurement["role"], (
            f"{measurement['dataset']} is not in supported_by but its role does "
            f"not say it was excluded"
        )
