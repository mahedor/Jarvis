"""
Test suite for the pure logic in tools/collect_faces.py.

These cover ONLY the functions that are free of cv2, disk, and models —
quality_check, l2_normalize, medoid_index, summarize_distribution — which
is why they run in milliseconds and need no photos, no weights, and no
network.

Run:
  pytest tests/test_collect_faces.py
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from collect_faces import (  # noqa: E402  (import follows the sys.path tweak above)
    REJECT_LOW_CONFIDENCE,
    REJECT_TOO_BLURRY,
    REJECT_TOO_SMALL,
    build_enrollment_summary,
    folder_label_collisions,
    l2_normalize,
    medoid_index,
    quality_check,
    summarize_clusters,
    summarize_distribution,
    summarize_folders,
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


# ══════════════════════════════════════════════════════════════════
# The publishable summary.
#
# These moved here from tests/test_engineering.py when the derivation moved
# from render time to write time. The point of the move is that the real folder
# names — which are people's names — are handled only in this module and never
# reach demo/ or results/, so this is where their handling has to be tested.
# ══════════════════════════════════════════════════════════════════

def _curation_crops():
    """Two people photographed inside one person's folder, plus a stranger.

    Shaped like the real bucket: folder "alice" holds photos of Alice AND of
    Bob, so a folder-derived label would merge them.
    """
    def crop(cluster, folder, photo):
        return {"cluster": cluster, "source_folder": folder,
                "source_photo": photo, "filename": f"{photo}_{cluster}.jpg"}

    return (
        [crop(0, "alice", f"a{i}") for i in range(7)]        # Alice
        + [crop(1, "alice", f"a{i}") for i in range(3)]      # Bob, in her photos
        + [crop(2, "bob", f"b{i}") for i in range(2)]        # Bob's own folder
        + [crop(-1, "alice", "a9")]                          # a passer-by
    )


def _args(**overrides):
    settings = {"detector": "yolo", "encoder": "arcface", "margin": 0.35,
                "min_box_size": 40, "min_confidence": 0.5, "min_blur": None,
                "min_cluster_size": 6, "min_samples": None}
    settings.update(overrides)
    return SimpleNamespace(**settings)


def _stats(**overrides):
    counts = {"photos_read": 4, "unreadable_photos": 0, "faces_detected": 20,
              "crops_kept": 13, "embed_failures": 0, "alignment_fallbacks": 0,
              "rejects": {"low_confidence": 7}}
    counts.update(overrides)
    return counts


def test_summarize_clusters_counts_noise_separately():
    clusters, noise = summarize_clusters(_curation_crops())
    assert noise == 1
    assert [c["id"] for c in clusters] == [0, 1, 2]
    assert [c["crops"] for c in clusters] == [7, 3, 2]


def test_summarize_clusters_never_exposes_real_folder_names():
    """The crops carry people's names; the page's argument never needs them."""
    clusters, _ = summarize_clusters(_curation_crops())
    rendered = str(clusters)
    assert "alice" not in rendered and "bob" not in rendered
    assert "folder A" in rendered


def test_folder_labels_would_have_merged_two_identities():
    """The load-bearing claim of the flat-bucket decision.

    Clusters 0 and 1 are different people but every crop of both came out of
    the same folder, so a majority vote would file one under the other's name.
    """
    clusters, _ = summarize_clusters(_curation_crops())
    collisions, mislabelled = folder_label_collisions(clusters)

    assert len(collisions) == 1
    assert collisions[0]["identities"] == 2
    assert collisions[0]["keeps"] == 7        # the larger keeps the name
    assert mislabelled == 3                   # the smaller is absorbed


def test_no_collision_when_each_folder_owns_one_identity():
    clusters, _ = summarize_clusters([
        {"cluster": 0, "source_folder": "x", "source_photo": "p1"},
        {"cluster": 1, "source_folder": "y", "source_photo": "p2"},
    ])
    collisions, mislabelled = folder_label_collisions(clusters)
    assert collisions == [] and mislabelled == 0


def test_summarize_folders_counts_photos_not_paths():
    """Distinct photos per folder, as a COUNT — the paths must not survive."""
    rows = summarize_folders(_curation_crops())
    assert [r["label"] for r in rows] == ["folder A", "folder B"]
    assert rows[0]["crops"] == 11          # alice's folder fed 11 crops
    assert rows[0]["photos"] == 8          # a0..a6 plus a9
    assert all(isinstance(r["photos"], int) for r in rows)


def test_summary_carries_no_names_filenames_or_paths():
    """The privacy contract of the published artifact, asserted directly.

    Checks the VALUES that came out of the crops, not key names: the summary
    legitimately has a "source_folders" key (a count), so a naive substring
    sweep for "source_folder" reports itself.
    """
    summary = build_enrollment_summary(_curation_crops(), _args(), _stats(), None)
    blob = json.dumps(summary)

    for secret in ("alice", "bob", ".jpg", "a0", "b0", "input_dir"):
        assert secret not in blob, f"{secret!r} reached the published summary"

    # No per-crop rows survive: every list in the summary is either numbers or
    # the aggregate dicts the page renders, never one entry per photograph.
    assert "clusters_detail" in summary
    for cluster in summary["clusters_detail"]:
        assert set(cluster) == {"id", "crops", "folders", "sole_folder",
                                "majority_folder"}
        assert all(label.startswith("folder ") for label, _ in cluster["folders"])


def test_summary_reports_the_shape_the_page_renders():
    summary = build_enrollment_summary(_curation_crops(), _args(), _stats(), None)
    assert summary["clusters"] == 3
    assert summary["cluster_sizes"] == [7, 3, 2]
    assert summary["clustered_crops"] == 12
    assert summary["noise_crops"] == 1
    assert summary["source_folders"] == 2
    assert summary["mislabelled_if_voted"] == 3
    assert summary["settings"]["encoder"] == "arcface"
    assert summary["rejects"] == {"low_confidence": 7}


def test_summary_headroom_is_slack_before_min_cluster_size_deletes_someone():
    """The trap the enrollment page is written around."""
    summary = build_enrollment_summary(_curation_crops(), _args(min_cluster_size=2),
                                       _stats(), None)
    assert summary["smallest_cluster"] == 2
    assert summary["headroom"] == 0          # one crop thinner and Bob vanishes


def test_summary_flags_clusters_at_risk_of_deletion():
    summary = build_enrollment_summary(_curation_crops(), _args(min_cluster_size=3),
                                       _stats(), None)
    # crops < min_cluster_size * 2, i.e. under 6: clusters 1 (3) and 2 (2).
    assert summary["at_risk"] == [1, 2]


def test_summary_of_an_empty_run_does_not_crash():
    summary = build_enrollment_summary([], _args(), _stats(), None)
    assert summary["clusters"] == 0
    assert summary["cluster_sizes"] == []
    assert summary["smallest_cluster"] is None
    assert summary["headroom"] is None
    assert summary["collisions"] == []
