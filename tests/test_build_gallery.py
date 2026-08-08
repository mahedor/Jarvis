"""
Test suite for the pure logic in tools/build_gallery.py, plus the gallery
file round trip.

Covers the numpy-only half — aggregation into per-person vectors, stacking,
regrouping, the threshold-calibration guard and metadata assembly — and the
save/load pair, which is I/O but needs nothing more than a tmp directory (no
crops, no encoders, no network). Building the encoder and embedding real crops
is exercised by running the tool itself.

Run:
  pytest tests/test_build_gallery.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from build_gallery import (  # noqa: E402  (import follows the sys.path tweak)
    DEFAULT_THRESHOLD,
    GALLERY_FORMAT_VERSION,
    THRESHOLD_CALIBRATED_FOR,
    build_metadata,
    build_person_vectors,
    f1_threshold_from_runs,
    load_benchmark_runs,
    load_gallery,
    save_gallery,
    stack_gallery,
    threshold_is_calibrated,
    threshold_warning,
    vectors_by_person,
)
from recognition_metrics import AGGREGATIONS, l2_normalize  # noqa: E402


@pytest.fixture
def people():
    """Three people with 5 / 3 / 1 unit-length 8-D embeddings."""
    rng = np.random.default_rng(0)
    return {
        "aaron": l2_normalize(rng.normal(size=(5, 8))),
        "mom": l2_normalize(rng.normal(size=(3, 8))),
        "dad": l2_normalize(rng.normal(size=(1, 8))),
    }


# ─── build_person_vectors ────────────────────────────────────────

@pytest.mark.parametrize("aggregation,expected_rows", [
    ("mean-renormalize", {"aaron": 1, "mom": 1, "dad": 1}),
    ("medoid", {"aaron": 1, "mom": 1, "dad": 1}),
    ("multi-reference", {"aaron": 5, "mom": 3, "dad": 1}),
])
def test_row_count_per_strategy(people, aggregation, expected_rows):
    build_fn, _ = AGGREGATIONS[aggregation]
    vectors = build_person_vectors(people, build_fn)
    assert {name: v.shape[0] for name, v in vectors.items()} == expected_rows


@pytest.mark.parametrize("aggregation", sorted(AGGREGATIONS))
def test_single_vector_strategies_are_promoted_to_2d(people, aggregation):
    """Every person comes out (K, D) so the live scorer never branches."""
    build_fn, _ = AGGREGATIONS[aggregation]
    for block in build_person_vectors(people, build_fn).values():
        assert block.ndim == 2
        assert block.dtype == np.float32


@pytest.mark.parametrize("aggregation", sorted(AGGREGATIONS))
def test_stored_vectors_stay_unit_length(people, aggregation):
    """Cosine == dot product only holds if aggregation preserves unit length."""
    build_fn, _ = AGGREGATIONS[aggregation]
    for block in build_person_vectors(people, build_fn).values():
        assert np.allclose(np.linalg.norm(block, axis=1), 1.0, atol=1e-5)


def test_person_with_no_embeddings_is_rejected():
    build_fn, _ = AGGREGATIONS["medoid"]
    with pytest.raises(ValueError, match="no embeddings"):
        build_person_vectors({"ghost": np.empty((0, 8), dtype=np.float32)}, build_fn)


def test_mixed_dimensions_are_rejected():
    """Two encoders' vectors in one gallery is exactly the silent failure the
    metadata exists to prevent, so it must not even build."""
    people = {
        "a": l2_normalize(np.ones((2, 8))),
        "b": l2_normalize(np.ones((2, 16))),
    }
    build_fn, _ = AGGREGATIONS["multi-reference"]
    with pytest.raises(ValueError, match="inconsistent embedding dimensions"):
        build_person_vectors(people, build_fn)


# ─── stack_gallery / vectors_by_person ───────────────────────────

def test_stack_labels_every_row_with_its_owner(people):
    build_fn, _ = AGGREGATIONS["multi-reference"]
    person_vectors = build_person_vectors(people, build_fn)
    vectors, ids = stack_gallery(person_vectors)

    assert vectors.shape == (9, 8)
    assert len(ids) == 9
    assert sorted(set(ids)) == ["aaron", "dad", "mom"]
    # rows arrive in sorted-name order, blocked by person
    assert list(ids) == ["aaron"] * 5 + ["dad"] * 1 + ["mom"] * 3


def test_regrouping_is_the_exact_inverse_of_stacking(people):
    build_fn, _ = AGGREGATIONS["multi-reference"]
    person_vectors = build_person_vectors(people, build_fn)
    vectors, ids = stack_gallery(person_vectors)

    regrouped = vectors_by_person(vectors, ids)
    assert set(regrouped) == set(person_vectors)
    for name, block in person_vectors.items():
        assert np.array_equal(regrouped[name], block)


def test_empty_gallery_does_not_explode():
    vectors, ids = stack_gallery({})
    assert vectors.shape == (0, 0)
    assert len(ids) == 0


# ─── threshold_is_calibrated ─────────────────────────────────────

def test_default_threshold_is_calibrated_for_its_own_pair():
    assert threshold_is_calibrated(*THRESHOLD_CALIBRATED_FOR, DEFAULT_THRESHOLD)


@pytest.mark.parametrize("encoder,aggregation", [
    ("facenet512", "multi-reference"),
    ("arcface", "medoid"),
    ("vgg-face", "mean-renormalize"),
])
def test_default_threshold_on_another_pair_is_flagged(encoder, aggregation):
    assert not threshold_is_calibrated(encoder, aggregation, DEFAULT_THRESHOLD)


def test_an_explicit_threshold_is_always_the_callers_call():
    assert threshold_is_calibrated("facenet512", "medoid", 0.567)


# ─── f1_threshold_from_runs ──────────────────────────────────────

def _run(encoder, aggregation, f1_threshold, tar_threshold=0.214):
    """A benchmark run payload holding one scored cell."""
    return {"results": [{
        "encoder": encoder,
        "aggregations": {aggregation: {
            "f1_threshold": {"threshold": f1_threshold, "f1": 0.98},
            "tar_at_far": {"0.01": {"tar": 0.985, "threshold": tar_threshold}},
        }},
    }]}


def test_reads_the_max_f1_threshold_for_the_cell():
    runs = [_run("arcface", "multi-reference", 0.342)]
    assert f1_threshold_from_runs(runs, "arcface", "multi-reference") == 0.342


def test_never_returns_the_tar_at_far_yardstick():
    """The TAR@FAR threshold sits in the same record; it must not leak out as
    the deployment number."""
    runs = [_run("arcface", "multi-reference", 0.342, tar_threshold=0.214)]
    assert f1_threshold_from_runs(runs, "arcface", "multi-reference") != 0.214


def test_the_newest_run_wins():
    runs = [_run("arcface", "medoid", 0.300), _run("arcface", "medoid", 0.366)]
    assert f1_threshold_from_runs(runs, "arcface", "medoid") == 0.366


@pytest.mark.parametrize("encoder,aggregation", [
    ("facenet512", "multi-reference"),   # encoder never scored
    ("arcface", "medoid"),               # aggregation never scored
])
def test_missing_cell_is_none(encoder, aggregation):
    runs = [_run("arcface", "multi-reference", 0.342)]
    assert f1_threshold_from_runs(runs, encoder, aggregation) is None


def test_no_runs_at_all_is_none():
    assert f1_threshold_from_runs([], "arcface", "multi-reference") is None
    assert f1_threshold_from_runs(None, "arcface", "multi-reference") is None


def test_an_errored_encoder_record_is_skipped():
    """A run where the encoder was skipped has no 'aggregations' to read."""
    runs = [{"results": [{"encoder": "vgg-face", "error": "not installed"}]}]
    assert f1_threshold_from_runs(runs, "vgg-face", "multi-reference") is None


def test_missing_results_file_degrades_quietly(tmp_path):
    assert load_benchmark_runs(tmp_path / "nope.json") == []


def test_corrupt_results_file_degrades_quietly(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    assert load_benchmark_runs(path) == []


# ─── threshold_warning ───────────────────────────────────────────

def test_matching_the_benchmarked_max_f1_is_silent():
    assert threshold_warning(0.342, "arcface", "multi-reference", 0.342) is None


def test_rounding_noise_is_not_a_mismatch():
    """The benchmark JSON rounds to 4dp, so 0.3420 vs 0.342 is the same number."""
    assert threshold_warning(0.342, "arcface", "multi-reference", 0.34204) is None


def test_mismatch_names_both_numbers():
    warning = threshold_warning(0.214, "arcface", "multi-reference", 0.342)
    assert warning is not None
    assert "0.214" in warning and "0.342" in warning
    assert "max-F1" in warning
    assert "arcface" in warning and "multi-reference" in warning


def test_mismatch_calls_out_the_yardstick_confusion():
    """The likeliest mistake is pasting the TAR@FAR threshold, so say so."""
    warning = threshold_warning(0.214, "arcface", "multi-reference", 0.342)
    assert "TAR@FAR" in warning and "yardstick" in warning


def test_unbenchmarked_pair_falls_back_to_the_default_check():
    warning = threshold_warning(DEFAULT_THRESHOLD, "facenet512", "medoid", None)
    assert warning is not None
    assert "no\n           benchmark result on file" in warning


def test_unbenchmarked_pair_with_an_explicit_threshold_is_silent():
    assert threshold_warning(0.567, "facenet512", "medoid", None) is None


def test_unbenchmarked_default_on_its_own_pair_is_silent():
    assert threshold_warning(DEFAULT_THRESHOLD, *THRESHOLD_CALIBRATED_FOR, None) is None


def test_every_warning_is_ascii_printable():
    """A Windows cp437 console must never crash on the warning text."""
    for warning in [threshold_warning(0.214, "arcface", "multi-reference", 0.342),
                    threshold_warning(DEFAULT_THRESHOLD, "facenet512", "medoid", None)]:
        warning.encode("ascii")


# ─── build_metadata ──────────────────────────────────────────────

def test_metadata_records_what_the_live_system_must_check():
    stats = {"aaron": {"crops": 5, "embedded": 5, "vectors": 5, "sources": []}}
    metadata = build_metadata("arcface", "multi-reference", 0.342, 512, stats, "ref")

    assert metadata["encoder"] == "arcface"
    assert metadata["aggregation"] == "multi-reference"
    assert metadata["threshold"] == 0.342
    assert metadata["dimension"] == 512
    assert metadata["num_persons"] == 1
    assert metadata["num_vectors"] == 5
    assert metadata["format_version"] == GALLERY_FORMAT_VERSION


def test_metadata_is_json_serializable():
    stats = {"aaron": {"crops": 1, "embedded": 1, "vectors": 1, "sources": ["a.jpg"]}}
    metadata = build_metadata("arcface", "medoid", 0.3, 512, stats, "ref",
                              extras={"embed_failures": 0})
    assert json.loads(json.dumps(metadata)) == metadata


# ─── save_gallery / load_gallery ─────────────────────────────────

@pytest.fixture
def saved_gallery(tmp_path, people):
    build_fn, _ = AGGREGATIONS["multi-reference"]
    person_vectors = build_person_vectors(people, build_fn)
    vectors, ids = stack_gallery(person_vectors)
    stats = {n: {"crops": len(v), "embedded": len(v),
                 "vectors": int(person_vectors[n].shape[0]), "sources": []}
             for n, v in people.items()}
    metadata = build_metadata("arcface", "multi-reference", DEFAULT_THRESHOLD,
                              8, stats, "ref")
    path = tmp_path / "gallery.npz"
    save_gallery(path, vectors, ids, metadata)
    return path, vectors, list(ids), metadata


def test_round_trip_is_bit_exact(saved_gallery):
    path, vectors, ids, metadata = saved_gallery
    loaded_vectors, loaded_ids, loaded_metadata = load_gallery(path)

    assert np.array_equal(loaded_vectors, vectors)  # float32 binary, not text
    assert loaded_ids == ids
    assert loaded_metadata == metadata


def test_file_loads_without_pickle(saved_gallery):
    """Nothing in a gallery file may be executable on load."""
    path, _, _, _ = saved_gallery
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"vectors", "person_ids", "metadata"}


def test_matching_encoder_passes_the_guard(saved_gallery):
    path, _, _, _ = saved_gallery
    vectors, _, _ = load_gallery(path, expected_encoder="arcface",
                                 expected_aggregation="multi-reference")
    assert vectors.shape[0] == 9


def test_wrong_encoder_is_refused(saved_gallery):
    """The whole point: a mismatched encoder must raise, not score badly."""
    path, _, _, _ = saved_gallery
    with pytest.raises(ValueError, match="ENCODER MISMATCH"):
        load_gallery(path, expected_encoder="facenet512")


def test_wrong_aggregation_is_refused(saved_gallery):
    path, _, _, _ = saved_gallery
    with pytest.raises(ValueError, match="AGGREGATION MISMATCH"):
        load_gallery(path, expected_aggregation="medoid")


def test_unreadable_format_version_is_refused(tmp_path):
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        vectors=np.zeros((1, 8), dtype=np.float32),
        person_ids=np.asarray(["aaron"], dtype=np.str_),
        metadata=np.asarray(json.dumps({"format_version": 999, "encoder": "arcface"})),
    )
    with pytest.raises(ValueError, match="format version"):
        load_gallery(path)


def test_a_file_that_is_not_a_gallery_is_refused(tmp_path):
    path = tmp_path / "junk.npz"
    np.savez_compressed(path, something_else=np.zeros(3))
    with pytest.raises(ValueError, match="not a valid JARVIS gallery"):
        load_gallery(path)
