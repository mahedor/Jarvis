"""
Test suite for the /engineering log — data shaping and routes.

Two things are worth guarding here. First, the pages must never 500 because an
artifact is missing, malformed or empty: documentation that crashes when a
benchmark has not been run is worse than no documentation. Second, the
yardstick/deployment distinction must survive into the rendered HTML — the same
distinction guarded in the terminal report by test_benchmark_report.py. It has
regressed twice in the report; there is no reason the page is immune.

The routes are exercised through Flask's test client, which needs no server and
no API key (the blueprint never calls Claude).

Run:
  pytest tests/test_engineering.py
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "demo"))

import engineering_data as data  # noqa: E402


# ══════════════════════════════════════════════════════════════════
# Pure shaping.
# ══════════════════════════════════════════════════════════════════

def _detection_run(dataset, timestamp, limit=None, ap=0.86, **extra):
    run = {
        "timestamp": timestamp,
        "dataset": dataset,
        "iou_threshold": 0.5,
        "limit": limit,
        "num_images": 3226,
        "results": [
            {"detector": "yolo", "ap": ap, "roc_auc": 0.9, "f1": 0.8,
             "f1_threshold": 0.5727, "mean_ms": 25.0, "p95_ms": 37.0, "fps": 39.0},
            {"detector": "mediapipe", "ap": 0.07, "roc_auc": 0.08, "f1": 0.13,
             "f1_threshold": 0.4, "mean_ms": 25.5, "p95_ms": 37.2, "fps": 39.2},
            {"detector": "mtcnn", "error": "not installed"},
        ],
    }
    run.update(extra)
    return run


def test_limited_runs_are_excluded():
    """--limit smoke tests have incomparable AP and must not reach the page."""
    runs = [_detection_run("widerface", "T1", limit=5),
            _detection_run("widerface", "T2")]
    latest = data.latest_full_run_per_dataset(runs)
    assert latest["widerface"]["timestamp"] == "T2"


def test_only_limited_runs_yields_nothing():
    runs = [_detection_run("widerface", "T1", limit=5)]
    assert data.latest_full_run_per_dataset(runs) == {}


def test_newest_full_run_per_dataset_wins():
    runs = [_detection_run("mafa", "T1"), _detection_run("widerface", "T2"),
            _detection_run("mafa", "T3")]
    latest = data.latest_full_run_per_dataset(runs)
    assert latest["mafa"]["timestamp"] == "T3"
    assert sorted(latest) == ["mafa", "widerface"]


def test_legacy_runs_without_a_dataset_are_skipped():
    runs = [{"timestamp": "T0", "limit": None, "results": []}]
    assert data.latest_full_run_per_dataset(runs) == {}


def test_detection_rows_sort_by_ap_and_split_errors():
    rows, errors = data.detection_rows(_detection_run("widerface", "T1"))
    assert [r["detector"] for r in rows] == ["yolo", "mediapipe"]
    assert [e["detector"] for e in errors] == ["mtcnn"]


def _recognition_run():
    def cell(tar, f1_thr, tar_thr):
        return {
            "rank1_accuracy": 0.985,
            "tar_at_far": {"0.01": {"tar": tar, "threshold": tar_thr},
                           "0.001": {"tar": tar - 0.02, "threshold": tar_thr + 0.08}},
            "f1_threshold": {"threshold": f1_thr, "f1": 0.984},
            "separation": {"dprime_vs_stranger": 5.77},
            "distributions": {
                "genuine": {"histogram": {"edges": [0.0, 0.5, 1.0], "counts": [1, 9]}},
                "impostor": {"histogram": {"edges": [0.0, 0.5, 1.0], "counts": [8, 2]}},
                "stranger": {"histogram": {"edges": [0.0, 0.5, 1.0], "counts": [10, 0]}},
            },
        }
    return {
        "timestamp": "T9",
        "aggregations_run": ["mean-renormalize", "multi-reference"],
        "num_persons": 6, "num_strangers": 500, "enroll_fraction": 0.5, "seed": 42,
        "results": [
            {"encoder": "arcface", "latency": {"mean_ms": 187.8},
             "aggregations": {"mean-renormalize": cell(0.969, 0.307, 0.191),
                              "multi-reference": cell(0.985, 0.342, 0.214)}},
            {"encoder": "facenet512", "latency": {"mean_ms": 123.1},
             "aggregations": {"mean-renormalize": cell(0.939, 0.560, 0.413),
                              "multi-reference": cell(0.954, 0.641, 0.492)}},
            {"encoder": "dlib", "error": "not installed"},
        ],
    }


def test_matrix_orders_encoders_by_best_tar():
    matrix = data.recognition_matrix(_recognition_run())
    assert matrix["encoders"] == ["arcface", "facenet512"]
    assert [e["encoder"] for e in matrix["errors"]] == ["dlib"]


def test_matrix_keeps_the_two_thresholds_apart():
    """If these ever collapse into one field, the page cannot label them."""
    matrix = data.recognition_matrix(_recognition_run())
    cell = matrix["cells"]["arcface|multi-reference"]
    assert cell["yardstick_threshold"] == 0.214
    assert cell["deployment_threshold"] == 0.342


def test_winner_is_the_best_cell_across_aggregations():
    winner = data.pick_winner(data.recognition_matrix(_recognition_run()))
    assert winner["encoder"] == "arcface"
    assert winner["aggregation"] == "multi-reference"
    assert winner["deployment_threshold"] == 0.342
    assert winner["yardstick_threshold"] == 0.214


def test_winner_of_an_empty_grid_is_none():
    assert data.pick_winner({"encoders": [], "aggregations": [], "cells": {}}) is None


def test_explorer_payload_carries_histograms_per_cell():
    payload = data.explorer_payload(_recognition_run())
    assert "arcface|multi-reference" in payload
    assert set(payload["arcface|multi-reference"]) == {"genuine", "impostor", "stranger"}


def test_explorer_payload_is_json_serializable():
    json.dumps(data.explorer_payload(_recognition_run()))


def _sweep_cell(points):
    """A recognition cell carrying a recorded sweep, for the exact-path tests."""
    cell = {
        "counts": {"genuine": 10, "impostor": 20, "stranger": 30},
        "curves": {"sweep": points},
        "distributions": {
            "genuine": {"histogram": {"edges": [0.0, 0.5, 1.0], "counts": [1, 9]}},
            "impostor": {"histogram": {"edges": [0.0, 0.5, 1.0], "counts": [8, 2]}},
            "stranger": {"histogram": {"edges": [0.0, 0.5, 1.0], "counts": [10, 0]}},
        },
    }
    return {"timestamp": "T10", "aggregations_run": ["multi-reference"],
            "results": [{"encoder": "arcface", "latency": {"mean_ms": 1.0},
                         "aggregations": {"multi-reference": cell}}]}


def test_explorer_payload_flattens_a_sweep_into_parallel_arrays():
    run = _sweep_cell([
        {"threshold": -1.0, "tp": 10, "fp": 50},
        {"threshold": -0.99, "tp": 10, "fp": 49},
        {"threshold": -0.98, "tp": 9, "fp": 40},
    ])
    exact = data.explorer_payload(run)["arcface|multi-reference"]["exact"]
    assert exact["min"] == -1.0
    assert exact["step"] == 0.01
    assert exact["tp"] == [10, 10, 9]
    assert exact["fp"] == [50, 49, 40]
    assert exact["genuine_total"] == 10
    assert exact["nonmate_total"] == 50  # impostor + stranger


def test_explorer_payload_omits_exact_counts_for_a_run_without_curves():
    """The absence is the fallback signal; a half-filled block would not be."""
    payload = data.explorer_payload(_recognition_run())
    assert "exact" not in payload["arcface|multi-reference"]
    assert not data.explorer_is_exact(payload)


def test_explorer_is_exact_only_when_every_cell_has_a_sweep():
    run = _sweep_cell([{"threshold": -1.0, "tp": 1, "fp": 1},
                       {"threshold": -0.99, "tp": 1, "fp": 1}])
    assert data.explorer_is_exact(data.explorer_payload(run))

    # One cell missing its sweep is enough to put the page back on the caveat.
    run["aggregations_run"].append("medoid")
    run["results"][0]["aggregations"]["medoid"] = {
        "distributions": {"genuine": {"histogram": {"edges": [0.0, 1.0], "counts": [5]}}},
    }
    assert not data.explorer_is_exact(data.explorer_payload(run))


def test_explorer_is_exact_is_false_for_an_empty_payload():
    """No cells is not 'all cells are exact' — the page must not claim it."""
    assert not data.explorer_is_exact({})


def _curve_run():
    """A run carrying everything the client-side curves need."""
    run = _sweep_cell([{"threshold": -1.0, "tp": 10, "fp": 50},
                       {"threshold": -0.99, "tp": 9, "fp": 40}])
    metrics = run["results"][0]["aggregations"]["multi-reference"]
    metrics["curves"].update({
        "roc_auc": 0.994,
        "average_precision": 0.981,
        "markers": {
            "tar@far=0.01": {"threshold": 0.21, "tar": 0.98, "far": 0.0099,
                             "recall": 0.98, "precision": 0.66},
            "tar@far=0.001": {"threshold": 0.28, "tar": 0.97, "far": 0.0009,
                              "recall": 0.97, "precision": 0.95},
            "best_f1": {"threshold": 0.34, "tar": 0.97, "far": 0.0,
                        "recall": 0.97, "precision": 1.0},
        },
    })
    return run


def test_curve_summary_carries_the_scalars_and_both_shipped_markers():
    cell = data.explorer_payload(_curve_run())["arcface|multi-reference"]
    assert cell["curve"]["auc"] == 0.994
    assert cell["curve"]["ap"] == 0.981
    # The two markers the page names, mapped off their artifact keys.
    assert cell["curve"]["markers"]["yardstick"]["threshold"] == 0.21
    assert cell["curve"]["markers"]["deployment"]["threshold"] == 0.34
    assert cell["curve"]["markers"]["deployment"]["precision"] == 1.0
    assert data.explorer_has_curves(data.explorer_payload(_curve_run()))


def test_curve_summary_is_withheld_when_a_shipped_marker_is_missing():
    """Half a marker pair cannot be drawn, so the cell must not claim to be
    plottable — the section hides rather than rendering a curve with no
    operating point on it."""
    run = _curve_run()
    del run["results"][0]["aggregations"]["multi-reference"]["curves"]["markers"]["best_f1"]
    payload = data.explorer_payload(run)
    assert "curve" not in payload["arcface|multi-reference"]
    assert not data.explorer_has_curves(payload)


def test_a_run_without_curves_cannot_be_plotted():
    assert not data.explorer_has_curves(data.explorer_payload(_recognition_run()))
    assert not data.explorer_has_curves({})


def test_recognition_page_renders_the_interactive_curve_controls(client):
    """The curves are a control surface, not a picture of one."""
    html = client.get("/engineering/recognition").get_data(as_text=True)
    for element in ("curve-aggregation", "curve-threshold", "curve-log",
                    "curve-pr", "curve-roc", "curve-explorer.js"):
        assert element in html


def test_curve_section_hides_when_the_run_predates_curves(client, monkeypatch):
    monkeypatch.setattr(data, "load_runs", lambda path: [_recognition_run()])
    html = client.get("/engineering/recognition").get_data(as_text=True)
    assert "The curves behind the numbers" not in html
    assert 'id="curves"' not in html


# ══════════════════════════════════════════════════════════════════
# Sampling-bias evidence (the overview page's opening argument).
# ══════════════════════════════════════════════════════════════════

def _ap_run(dataset, timestamp, limit, images, aps):
    return {"timestamp": timestamp, "dataset": dataset, "limit": limit,
            "iou_threshold": 0.5, "num_images": images,
            "results": [{"detector": d, "ap": ap} for d, ap in aps.items()]}


def test_sampling_comparison_pairs_dev_against_full():
    runs = [_ap_run("widerface", "T1", 100, 100, {"yolo": 0.75, "mtcnn": 0.51}),
            _ap_run("widerface", "T2", None, 3226, {"yolo": 0.86, "mtcnn": 0.62})]
    block = data.sampling_comparison(runs)[0]
    assert block["dev_images"] == 100 and block["full_images"] == 3226
    deltas = {r["detector"]: r["delta"] for r in block["rows"]}
    assert deltas["yolo"] == pytest.approx(0.11)
    assert deltas["mtcnn"] == pytest.approx(0.11)


def test_uniform_movement_is_flagged_as_same_direction():
    """All detectors moving together is the signature of a biased sample."""
    runs = [_ap_run("widerface", "T1", 100, 100, {"a": 0.5, "b": 0.6}),
            _ap_run("widerface", "T2", None, 3000, {"a": 0.6, "b": 0.7})]
    assert data.sampling_comparison(runs)[0]["same_direction"] is True


def test_mixed_movement_is_not_flagged():
    runs = [_ap_run("mafa", "T1", 100, 100, {"a": 0.5, "b": 0.6}),
            _ap_run("mafa", "T2", None, 3000, {"a": 0.6, "b": 0.5})]
    assert data.sampling_comparison(runs)[0]["same_direction"] is False


def test_strongest_evidence_is_ordered_first():
    """Alphabetical order would open the argument with the control dataset."""
    runs = [
        _ap_run("fddb", "T1", 100, 100, {"a": 0.90, "b": 0.95}),
        _ap_run("fddb", "T2", None, 2845, {"a": 0.89, "b": 0.96}),   # mixed, tiny
        _ap_run("widerface", "T3", 100, 100, {"a": 0.50, "b": 0.60}),
        _ap_run("widerface", "T4", None, 3226, {"a": 0.62, "b": 0.72}),  # uniform
    ]
    assert [b["dataset"] for b in data.sampling_comparison(runs)] == ["widerface", "fddb"]


def test_tiny_smoke_runs_are_not_treated_as_dev_samples():
    runs = [_ap_run("widerface", "T1", 5, 5, {"a": 0.3}),
            _ap_run("widerface", "T2", None, 3226, {"a": 0.86})]
    assert data.sampling_comparison(runs) == []


def test_dataset_without_a_full_run_is_omitted():
    runs = [_ap_run("widerface", "T1", 100, 100, {"a": 0.5})]
    assert data.sampling_comparison(runs) == []


def test_candidate_box_rates_are_per_image_and_sorted():
    run = {"num_images": 100, "results": [
        {"detector": "retinaface", "num_detections": 650, "total_fp": 200},
        {"detector": "yolo", "num_detections": 29370, "total_fp": 28000},
    ]}
    rates = data.candidate_box_rates(run)
    assert [r["detector"] for r in rates] == ["yolo", "retinaface"]
    assert rates[0]["per_image"] == pytest.approx(293.7)


def test_box_rates_of_an_empty_run_are_empty():
    assert data.candidate_box_rates(None) == []
    assert data.candidate_box_rates({"num_images": 0, "results": []}) == []


# ══════════════════════════════════════════════════════════════════
# Intent routing + prompt caching.
# ══════════════════════════════════════════════════════════════════

def test_layers_are_grouped_and_ordered_cheapest_first():
    run = {"results": [
        {"layer": "tier2_embedding", "tier": 2, "avg_ms": 30.0, "p95_ms": 40.0},
        {"layer": "tier1_keyword", "tier": 1, "avg_ms": 0.01, "p95_ms": 0.02},
        {"layer": "tier1_keyword", "tier": 1, "avg_ms": 0.03, "p95_ms": 0.05},
    ]}
    layers = data.intent_layers(run)
    assert [l["layer"] for l in layers] == ["tier1_keyword", "tier2_embedding"]
    assert layers[0]["commands"] == 2
    assert layers[0]["avg_ms"] == pytest.approx(0.02)
    assert layers[0]["p95_ms"] == pytest.approx(0.05)  # worst p95, not averaged


def test_layers_of_an_empty_run():
    assert data.intent_layers(None) == []
    assert data.intent_layers({"results": []}) == []


def _caching_run(created, read, input_tokens, speedup=1.0, **extra):
    run = {
        "timestamp": "2026-04-22T08:00:00Z",
        "speedup_avg_x": speedup,
        "caching_on": {"latency_ms": {"avg": 2600.0},
                       "usage": [{"input_tokens": input_tokens,
                                  "cache_creation_input_tokens": created,
                                  "cache_read_input_tokens": read}]},
        "caching_off": {"latency_ms": {"avg": 2610.0}},
    }
    run.update(extra)
    return run


def test_caching_run_with_zero_cache_tokens_is_not_engaged():
    """The whole point: a 1.0x speedup from a run that never cached anything
    must not be presented as evidence that caching does not help."""
    rows = data.caching_runs([_caching_run(0, 0, 411)])
    assert rows[0]["engaged"] is False


def test_caching_run_with_cache_tokens_is_engaged():
    rows = data.caching_runs([_caching_run(1039, 9351, 19)])
    assert rows[0]["engaged"] is True
    assert rows[0]["cache_created"] == 1039
    assert rows[0]["cache_read"] == 9351


def test_cache_tokens_are_summed_across_all_runs_in_a_mode():
    run = _caching_run(0, 0, 19)
    run["caching_on"]["usage"] = [
        {"input_tokens": 19, "cache_creation_input_tokens": 1039, "cache_read_input_tokens": 0},
        {"input_tokens": 19, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1039},
    ]
    row = data.caching_runs([run])[0]
    assert row["cache_created"] == 1039 and row["cache_read"] == 1039
    assert row["engaged"] is True


def test_claude_call_ms_averages_the_uncached_latency():
    rows = data.caching_runs([_caching_run(0, 0, 411), _caching_run(1, 1, 19)])
    assert data.claude_call_ms(rows) == pytest.approx(2610.0)


def test_claude_call_ms_without_data_is_none():
    assert data.claude_call_ms([]) is None


def test_eval_corpus_reports_composition_not_accuracy():
    corpus = data.eval_corpus()
    assert corpus["routing_cases"] > 0
    assert corpus["conversation_cases"] > 0
    assert dict(corpus["categories"]).get("adversarial", 0) > 0
    # No scored results exist; a .gitkeep must not be mistaken for one.
    assert corpus["has_scored_results"] is False


def test_eval_loader_skips_the_registry_rule_header(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('# EVAL ROUTE REGISTRY RULE: header\n'
                    '{"id": "a", "expected_route": "x"}\n'
                    '\n'
                    'not json\n', encoding="utf-8")
    cases = data.load_eval_cases(path)
    assert len(cases) == 1 and cases[0]["id"] == "a"


def test_eval_loader_missing_file_is_empty(tmp_path):
    assert data.load_eval_cases(tmp_path / "nope.jsonl") == []


# ══════════════════════════════════════════════════════════════════
# Detection winner + confidence sweeps.
# ══════════════════════════════════════════════════════════════════

def test_detection_winner_prefers_dataset_wins_over_mean_ap():
    """A detector that takes the easy set by a mile and loses the hard ones
    must not win — that is the argument the page makes about MediaPipe."""
    datasets = [
        {"dataset": "easy", "rows": [{"detector": "showoff", "ap": 0.99, "f1": .9, "f1_threshold": .5},
                                     {"detector": "steady", "ap": 0.80, "f1": .8, "f1_threshold": .5}]},
        {"dataset": "hard1", "rows": [{"detector": "steady", "ap": 0.70, "f1": .7, "f1_threshold": .5},
                                      {"detector": "showoff", "ap": 0.20, "f1": .2, "f1_threshold": .5}]},
        {"dataset": "hard2", "rows": [{"detector": "steady", "ap": 0.68, "f1": .7, "f1_threshold": .5},
                                      {"detector": "showoff", "ap": 0.18, "f1": .2, "f1_threshold": .5}]},
    ]
    winner = data.pick_detection_winner(datasets)
    assert winner["detector"] == "steady"
    assert winner["wins"] == 2 and winner["contested"] == 3
    assert winner["swept"] is False


def test_detection_winner_reports_a_sweep():
    datasets = [
        {"dataset": "a", "rows": [{"detector": "yolo", "ap": 0.9, "f1": .8, "f1_threshold": .5}]},
        {"dataset": "b", "rows": [{"detector": "yolo", "ap": 0.7, "f1": .8, "f1_threshold": .5}]},
    ]
    winner = data.pick_detection_winner(datasets)
    assert winner["swept"] is True and winner["worst"] == 0.7


def test_detection_winner_of_nothing_is_none():
    assert data.pick_detection_winner([]) is None
    assert data.pick_detection_winner([{"dataset": "a", "rows": []}]) is None


def test_sweeps_are_only_reported_when_present():
    """Older runs have no confidence_sweep; the explorer must simply not
    appear for them rather than the page failing."""
    assert data.detection_sweeps({"results": [{"detector": "yolo", "ap": 0.9}]}) == {}
    run = {"results": [{"detector": "yolo", "confidence_sweep": [{"threshold": 0.5}]}]}
    assert list(data.detection_sweeps(run)) == ["yolo"]


# ══════════════════════════════════════════════════════════════════
# Enrollment curation.
# ══════════════════════════════════════════════════════════════════

def _summary(**overrides):
    """A minimal enrollment_summary.json, already anonymised.

    The shaping tests for how this is DERIVED live in tests/test_collect_faces.py,
    beside the code that handles the real folder names. These tests are about
    what the page does with the artifact once it exists.
    """
    summary = {
        "generated_at": "2026-01-01T00:00:00",
        "clusters": 2,
        "cluster_sizes": [7, 3],
        "clusters_detail": [
            {"id": 0, "crops": 7, "folders": [["folder A", 7]],
             "sole_folder": "folder A", "majority_folder": "folder A"},
            {"id": 1, "crops": 3, "folders": [["folder A", 3]],
             "sole_folder": "folder A", "majority_folder": "folder A"},
        ],
        "folders": [{"label": "folder A", "crops": 10, "photos": 8}],
        "source_folders": 1,
        "collisions": [{"folder": "folder A", "clusters": [0, 1],
                        "identities": 2, "keeps": 7, "absorbed": 3}],
        "mislabelled_if_voted": 3,
        "clustered_crops": 10,
        "noise_crops": 1,
        "smallest_cluster": 3,
        "largest_cluster": 7,
        "headroom": 1,
        "at_risk": [1],
        "rejects": {"low_confidence": 7},
        "counts": {"photos_read": 4, "crops_kept": 11},
        "settings": {"min_cluster_size": 2, "encoder": "arcface"},
        "blur_distribution": None,
    }
    summary.update(overrides)
    return summary


def test_curation_page_data_shapes_the_artifact_for_the_template(monkeypatch):
    monkeypatch.setattr(data, "load_document", lambda path: _summary())
    curation = data.curation_page_data()["curation"]

    assert curation["people"] == 2
    assert curation["noise"] == 1
    assert curation["source_folders"] == 1
    assert curation["mislabelled_if_voted"] == 3
    assert curation["min_cluster_size"] == 2
    # rejects is top-level in the artifact but counts.rejects to the template.
    assert curation["counts"]["rejects"] == {"low_confidence": 7}
    # at_risk arrives as ids and is resolved to the cluster dicts the page renders.
    assert [c["id"] for c in curation["at_risk"]] == [1]


def test_curation_folders_arrive_as_pairs_the_template_can_unpack(monkeypatch):
    """JSON has no tuples, so the [label, count] lists must survive the trip."""
    monkeypatch.setattr(data, "load_document", lambda path: _summary())
    clusters = data.curation_page_data()["curation"]["clusters"]
    for cluster in clusters:
        for entry in cluster["folders"]:
            label, count = entry          # the template does exactly this
            assert isinstance(label, str) and isinstance(count, int)


def test_curation_page_data_is_none_without_the_artifact(monkeypatch):
    """results/enrollment_summary.json is committed, so absent means broken."""
    monkeypatch.setattr(data, "ENROLLMENT_SUMMARY", data.REPO_ROOT / "nope.json")
    assert data.curation_page_data()["curation"] is None


def test_curation_page_data_is_none_for_an_artifact_with_no_clusters(monkeypatch):
    """A summary that parsed but describes nothing is still nothing to render."""
    monkeypatch.setattr(data, "load_document",
                        lambda path: _summary(clusters_detail=[]))
    assert data.curation_page_data()["curation"] is None


def test_missing_artifact_renders_a_visible_marker_not_a_quiet_empty_state(
        client, monkeypatch):
    """The regression this whole artifact exists to prevent.

    The page shipped a polite "no curation run on this machine" to production
    for months. An absent COMMITTED artifact is a broken build, and the page has
    to say so in a way a reader cannot mistake for "nothing to report yet".
    """
    monkeypatch.setattr(data, "ENROLLMENT_SUMMARY", data.REPO_ROOT / "nope.json")
    html = client.get("/engineering/enrollment").get_data(as_text=True)

    assert "Enrollment summary unavailable" in html
    assert "this page is incomplete" in html
    assert "eng-unavailable" in html


def test_the_rendered_curation_page_leaks_no_real_folder_names(client):
    """Guards the anonymisation end-to-end against this machine's real manifest.

    The source folders are named after real people. The page now renders from
    the derived artifact rather than the manifest, so this asserts the split
    actually holds: nothing from the private file reaches the published page.
    """
    manifest_path = data.REPO_ROOT / "data" / "reference_faces" / "manifest.json"
    manifest = data.load_document(manifest_path)
    if not manifest.get("crops"):
        pytest.skip("no curation manifest on this machine")

    html = client.get("/engineering/enrollment").get_data(as_text=True)
    for folder in {c.get("source_folder") for c in manifest["crops"]}:
        if folder and folder != ".":
            assert folder not in html, f"source folder {folder!r} reached the page"


def test_the_committed_artifact_itself_carries_no_real_folder_names():
    """The artifact is committed and published, so it is checked directly too."""
    manifest = data.load_document(
        data.REPO_ROOT / "data" / "reference_faces" / "manifest.json")
    summary = data.load_document(data.ENROLLMENT_SUMMARY)
    if not manifest.get("crops") or not summary:
        pytest.skip("needs both the private manifest and the built artifact")

    blob = json.dumps(summary)
    for folder in {c.get("source_folder") for c in manifest["crops"]}:
        if folder and folder != ".":
            assert folder not in blob, f"{folder!r} reached results/"


# ══════════════════════════════════════════════════════════════════
# Loading degrades instead of raising.
# ══════════════════════════════════════════════════════════════════

def test_missing_results_file_is_empty(tmp_path):
    assert data.load_runs(tmp_path / "nope.json") == []


def test_corrupt_results_file_is_empty(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    assert data.load_runs(path) == []


def test_reload_picks_up_a_changed_file(tmp_path):
    """Rerunning a benchmark must show up without restarting the server."""
    path = tmp_path / "runs.json"
    path.write_text(json.dumps([{"timestamp": "T1"}]))
    assert len(data.load_runs(path)) == 1

    path.write_text(json.dumps([{"timestamp": "T1"}, {"timestamp": "T2"}]))
    import os
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))  # force a distinct mtime
    assert len(data.load_runs(path)) == 2


def test_empty_run_list_shapes_cleanly():
    assert data.recognition_matrix(None)["encoders"] == []
    assert data.detection_rows(None) == ([], [])
    assert data.explorer_payload(None) == {}


# ══════════════════════════════════════════════════════════════════
# Routes.
# ══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    import jarvis_web
    jarvis_web.app.config["TESTING"] = True
    return jarvis_web.app.test_client()


@pytest.mark.parametrize("url", ["/engineering/", "/engineering/assistant",
                                 "/engineering/architecture",
                                 "/engineering/routing", "/engineering/detection",
                                 "/engineering/recognition", "/engineering/enrollment"])
def test_pages_render(client, url):
    response = client.get(url)
    assert response.status_code == 200


def test_detection_explorer_is_told_what_the_threshold_was_decided_from(client):
    """The shipped marker must not be drawn as an operating point on curves it
    was never derived from.

    0.57 came from yolo on mafa + widerface; FDDB was deliberately excluded
    (pipeline_config.py). The chart can only honour that if the page hands it
    the scope, so the scope has to survive in the markup.
    """
    html = client.get("/engineering/detection").get_data(as_text=True)
    assert "data-shipped-detector='yolo'" in html

    datasets = html[html.index("data-shipped-datasets='"):]
    datasets = datasets[:datasets.index("'>")]
    assert "mafa" in datasets and "widerface" in datasets
    assert "fddb" not in datasets, "FDDB had no vote in the shipped threshold"


def test_detection_page_counts_its_own_sweep_resolution():
    """The prose claimed 200 confidences; confidence_sweep(steps=200) stores
    201. Count it rather than restate it."""
    page = data.detection_page_data()
    sweeps = [s for detectors in page["explorer"].values()
              for s in detectors.values()]
    if not sweeps:
        pytest.skip("no confidence sweeps recorded")
    assert page["sweep_points"] == len(sweeps[0])


def test_sweep_resolution_is_withheld_when_runs_disagree(monkeypatch):
    """Runs resampled onto different grids have no single honest number, so
    the page must say nothing rather than pick one."""
    def two_grids(_path):
        return [
            {"dataset": "a", "timestamp": "2026-01-01T00:00:00Z", "limit": None,
             "results": [{"detector": "yolo", "ap": 0.9,
                          "confidence_sweep": [{"threshold": 0.0}] * 201}]},
            {"dataset": "b", "timestamp": "2026-01-02T00:00:00Z", "limit": None,
             "results": [{"detector": "yolo", "ap": 0.9,
                          "confidence_sweep": [{"threshold": 0.0}] * 101}]},
        ]
    monkeypatch.setattr(data, "load_runs", two_grids)
    assert data.detection_page_data()["sweep_points"] is None


def _flat(html):
    """Rendered HTML with its line wrapping collapsed, so assertions can quote
    a sentence the way it reads rather than the way it happens to wrap."""
    return " ".join(html.split())


def test_the_overview_never_hardcodes_a_threshold_it_could_read(client):
    """No literal copy of either recognition threshold in the overview source.

    Same failure class as the min_box_size that had to be reverse engineered
    and the shipped marker drawn on curves it was never measured on: a number
    typed in once is a number that can silently stop matching the run it
    claims to come from. The overview cites both thresholds in three places,
    and all three have to resolve from the artifact.
    """
    winner = data.overview_page_data()["recognition_winner"]
    if not winner:
        pytest.skip("no recognition run recorded")

    template = (data.REPO_ROOT / "demo" / "templates" / "engineering"
                / "index.html").read_text(encoding="utf-8")
    for key in ("deployment_threshold", "yardstick_threshold"):
        literal = "%.3f" % winner[key]
        assert literal not in template, (
            f"{literal} is typed into index.html; render it from "
            f"recognition_winner.{key} instead"
        )


def test_the_overview_renders_the_locked_recognition_threshold(client):
    """...and the value that reaches the page is the artifact's."""
    winner = data.overview_page_data()["recognition_winner"]
    if not winner:
        pytest.skip("no recognition run recorded")

    html = _flat(client.get("/engineering/").get_data(as_text=True))
    expected = "%.3f" % winner["deployment_threshold"]
    assert f'<span class="eng-lock-number"> {expected} </span>' in html \
        or f'<span class="eng-lock-number">{expected}</span>' in html


def test_the_roadmap_is_marked_as_unmeasured(client):
    """The one section with no artifact behind it has to say so, loudly.

    The whole page's credibility rests on every claim being backed by a run.
    A roadmap that reads like the rest of the page borrows that credibility
    without earning it.
    """
    html = _flat(client.get("/engineering/").get_data(as_text=True))
    assert "What's next" in html
    assert "Everything above this point is backed by an artifact. "\
           "Nothing below it is." in html
    assert 'class="eng-badge planned">not measured' in html


def test_the_roadmap_comes_after_the_measured_work(client):
    """Unchanged in intent; the heading it anchors on was renamed when the
    stage grid stopped being a face-only list of "The measurements"."""
    html = client.get("/engineering/").get_data(as_text=True)
    assert html.index("What's next") > html.index("Every stage, and where it stands")


def test_the_roadmap_quotes_locked_thresholds_instead_of_new_ones(client):
    """The planned pipeline cites two operating points. Both must be the ones
    already justified on this page, not numbers typed into the roadmap."""
    winner = data.overview_page_data()["recognition_winner"]
    if not winner:
        pytest.skip("no recognition run recorded")

    html = client.get("/engineering/").get_data(as_text=True)
    pipeline = html[html.index("<pre class=\"eng-code\">frame"):]
    pipeline = pipeline[:pipeline.index("</pre>")]
    assert "%.3f" % winner["deployment_threshold"] in pipeline
    assert winner["encoder"] in pipeline
    assert "MQTT" in pipeline


def test_routing_page_states_the_accuracy_gap(client):
    """Latency is not accuracy, and the page must not let that slide."""
    html = client.get("/engineering/routing").get_data(as_text=True)
    assert "never been measured" in html


# The engaged/unengaged distinction is still guarded, on the page that now owns
# the experiment: test_caching_experiment_lives_on_the_assistant_page. It was
# asserted here while the routing page hosted the table, and moved with it
# rather than being dropped — the property is about the write-up, not the URL.


def test_log_is_not_only_face_work(client):
    """The assistant is an LLM product; a build log that is entirely vision
    misrepresents the project."""
    html = client.get("/engineering/").get_data(as_text=True)
    assert "/engineering/routing" in html
    assert "/engineering/assistant" in html


# ══════════════════════════════════════════════════════════════════
# The assistant page — Phase 1, the system every other page serves.
#
# Its defining risk is the opposite of the architecture page's. That page is
# wrong if it implies a result; this one is wrong if it LEVELS UP three
# judgment calls into the same confident register as the benchmark tables it
# sits beside. Most of what follows guards the seam between the two.
# ══════════════════════════════════════════════════════════════════

def test_assistant_page_renders(client):
    assert client.get("/engineering/assistant").status_code == 200


def test_assistant_page_covers_the_four_decisions(client):
    """Orchestration, speech, the wait, and caching — the brief for the page."""
    html = client.get("/engineering/assistant").get_data(as_text=True)
    assert "build_system_prompt()" in html          # device state ownership
    assert "speechSynthesis.speak()" in html        # browser TTS
    assert "/filler" in html                        # streaming / filler design
    assert "Prompt caching was measured and rejected" in html


def test_assistant_page_cites_the_tts_adr_as_its_source(client):
    """The latency figures are the ADR's, recorded while deciding. Presenting
    them without saying so would pass them off as a benchmark this repo can
    rerun, which it cannot."""
    html = client.get("/engineering/assistant").get_data(as_text=True)
    assert "docs/adr/002-browser-tts-over-server-tts.md" in html


def test_browser_voice_is_framed_as_temporary_not_final(client):
    """Both halves of the trade have to be on the page.

    Browser speech is presented as a win — no server cost, no latency, and
    Edge's recognition is genuinely good. Left there it reads as the finished
    answer, which would contradict both the roadmap and ADR 002's own
    demo-phase scoping. The page has to carry the exit alongside the win.
    """
    html = client.get("/engineering/assistant").get_data(as_text=True)
    # The win.
    assert "Edge's integrated speech recognition is genuinely" in html
    assert "no added latency" in html
    # The exit.
    assert "Whisper" in html
    assert "OpenWakeWord" in html
    assert "not the destination" in html
    # And the reason the swap is forced, not optional.
    assert "headless" in html


def test_voice_replacement_is_marked_planned_not_in_progress(client):
    """THE distinction this page is most likely to blur.

    "Being replaced by Whisper" and "blocked on hardware nobody has ordered"
    are different states, and the second one is true. A log that lets naming a
    successor imply work underway is doing the thing this whole site exists to
    avoid.
    """
    html = client.get("/engineering/assistant").get_data(as_text=True)
    assert "None of that replacement is in progress." in html
    assert "no branch, no prototype, no benchmark" in html


def test_assistant_page_does_not_claim_an_stt_benchmark(client):
    """The "Edge is good" claim is an impression from daily use. There is no
    transcription benchmark in this repo, and the page must not imply one."""
    html = client.get("/engineering/assistant").get_data(as_text=True)
    assert "an impression, not a number" in html


def test_assistant_page_marks_its_unmeasured_claims(client):
    """THE constraint for this page.

    Three of its four decisions are arguments, not results. The site's whole
    credibility rests on a reader being able to tell which is which at a
    glance, so the unmeasured ones carry their own marker rather than sharing
    the register of the measured one.
    """
    html = client.get("/engineering/assistant").get_data(as_text=True)
    assert html.count("eng-unmeasured") >= 3
    assert "Reasoned, not measured." in html
    assert "What has no measurement behind it" in html


def test_assistant_page_does_not_invent_evidence_for_the_ui(client):
    """The theme, the screensaver and the personality were chosen by taste.
    Saying so is the point; a page that quietly omitted them would read as
    though everything on the product had been justified."""
    html = client.get("/engineering/assistant").get_data(as_text=True)
    assert "no A/B" in html or "no A/B test" in html
    assert "never been measured" in html            # STT word error rate


def test_assistant_page_says_devices_are_simulated(client):
    html = client.get("/engineering/assistant").get_data(as_text=True)
    assert "Devices are simulated" in html


def test_caching_experiment_lives_on_the_assistant_page(client):
    """It measures the model call, not the router — which is what the routing
    page's own prose said while hosting it. One home, and this is it."""
    html = client.get("/engineering/assistant").get_data(as_text=True)
    assert "caching never engaged" in html
    assert "cache genuinely used" in html


def test_routing_page_keeps_the_claude_cost_but_not_the_experiment(client):
    """Routing still needs what a tier-3 escalation costs — that number is the
    entire justification for the cascade. It does not need the experiment."""
    html = client.get("/engineering/routing").get_data(as_text=True)
    assert "cache genuinely used" not in html
    assert "cache written" not in html
    # The cost argument survives the move.
    assert "ms. Anything" in html


# ══════════════════════════════════════════════════════════════════
# Direction page — the destination. The tests here are almost entirely
# about ONE property: it must never read as something that was built or
# measured. Everywhere else in this suite a page is wrong if it hides a
# result; this page is wrong if it implies one.
# ══════════════════════════════════════════════════════════════════

def test_measured_work_leads_and_ambition_does_not(client):
    """THE ordering constraint, in one test.

    The same architecture prose reads as a pitch when it opens the log and as
    a conclusion when it closes one. So the landing page must show measured
    work and must NOT be where the destination is ARGUED — that lives last.

    Rewritten when the log widened from a face-pipeline record to the whole
    project. The old version banned two capitalised phrases, which enforced the
    right thing by accident: it happened to catch the architecture page's
    headings while letting the same words through in lowercase prose. The
    constraint was never about vocabulary. Naming the destination on the way
    out is required — the closing section does exactly that — so what this now
    asserts is that the landing page does not carry the architecture page's
    STRUCTURE: its diagram, its memory-system breakdown, its decision-layer
    thesis. Mentioning where the project is going stays legal; making the case
    for it here does not.
    """
    html = client.get("/engineering/").get_data(as_text=True)
    assert "eng-decision" in html                    # the judgment calls
    assert "eng-lock-number" in html                 # the locked operating points

    # The unbuilt system is named on the way out, not on the way in.
    assert html.index("Decisions") < html.index("where all of it is heading")

    # The architecture page's load-bearing apparatus, none of which may appear
    # before a reader has been through the evidence.
    for argument in ("The Conductor decides. The LLM speaks.",
                     "eng-memory-limit",
                     "eng-arch-flow",
                     "eng-loop-step"):
        assert argument not in html, argument


def test_landing_page_introduces_the_project_before_any_workstream(client):
    """A reader arriving cold does not know what JARVIS is.

    The log opened for months on "the decisions behind the face pipeline",
    which answered a question nobody had been given a reason to ask and
    misrepresented a voice assistant as a vision project. The system has to be
    introduced before any single strand of it is.
    """
    html = client.get("/engineering/").get_data(as_text=True)
    lede = html[html.index('class="eng-lede"'):html.index("</section>")]
    assert "voice-driven assistant" in lede
    # The face pipeline may not be what the page opens on.
    assert "face pipeline" not in lede.lower() or (
        lede.lower().index("assistant") < lede.lower().index("face pipeline"))
    # And the workstreams are introduced before the face workstream's decisions.
    assert html.index("The workstreams") < html.index("eng-decision")


def test_landing_page_frames_face_decisions_as_one_workstream(client):
    """The decision cards are the face workstream's record, not the site's
    identity. Unlabelled, five vision decisions in a row re-assert exactly the
    scoping this restructure removed."""
    html = client.get("/engineering/").get_data(as_text=True)
    heading = html[html.index("<h2>Decisions"):]
    heading = heading[:heading.index("</h2>")]
    assert "face pipeline" in heading


def test_every_workstream_is_on_the_landing_page(client):
    html = client.get("/engineering/").get_data(as_text=True)
    import engineering as eng
    for group in eng.WORKSTREAMS:
        assert group["title"] in html, group["title"]


def test_nav_is_reading_order_assistant_first_architecture_last(client):
    """Nav order is reading order, and it makes two claims at once.

    The assistant comes first because it is the only page describing something
    that exists and runs; routing sits with it because it is the assistant's
    front door rather than a stage in a vision pipeline. The one page with no
    artifact behind it comes after every page that has one.
    """
    html = client.get("/engineering/").get_data(as_text=True)
    nav = html[html.index('class="eng-nav"'):html.index("</nav>")]
    order = ("assistant", "routing", "detection", "enrollment",
             "recognition", "architecture")
    positions = [nav.index(f'/engineering/{p}') for p in order]
    assert positions == sorted(positions), dict(zip(order, positions))
    assert nav.index("/engineering/architecture") == max(positions)


def test_nav_does_not_link_out_to_the_running_assistant(client):
    """The log must not be a door into the app.

    An "← assistant" link used to sit at the end of the nav whenever the
    blueprint was mounted on the real app, pointing at url_for('index'). The
    assistant is not ready to be walked into from a documentation page, and the
    log is meant to be readable standalone regardless. The `client` fixture
    mounts the blueprint on an app that DOES have an index view, so this test
    sees the exact condition under which the link used to appear.
    """
    html = client.get("/engineering/").get_data(as_text=True)
    nav = html[html.index('class="eng-nav"'):html.index("</nav>")]
    assert "eng-exit" not in nav
    # Every href in the nav stays inside the log.
    hrefs = re.findall(r'href="([^"]+)"', nav)
    assert hrefs, "nav has no links at all"
    for href in hrefs:
        assert href.startswith("/engineering"), href


def test_routing_is_not_listed_among_the_face_stages(client):
    """It was, and that was the bug: a flat list put intent routing between
    enrollment and recognition, implying the assistant is a step in the face
    pipeline."""
    import engineering as eng
    face = next(g for g in eng.WORKSTREAMS if g["id"] == "face")
    assert [s["id"] for s in face["stages"]] == [
        "detection", "enrollment", "recognition", "presence"]
    assistant = next(g for g in eng.WORKSTREAMS if g["id"] == "assistant")
    assert "routing" in [s["id"] for s in assistant["stages"]]


def test_stages_is_derived_from_workstreams(client):
    """Two lists that can disagree eventually do."""
    import engineering as eng
    assert eng.STAGES == [s for g in eng.WORKSTREAMS for s in g["stages"]]


def test_architecture_page_states_it_is_unmeasured_up_front(client):
    """A page of confident prose about behavioural prediction has to say, in
    its first screen, that none of it has been built."""
    html = client.get("/engineering/architecture").get_data(as_text=True)
    assert "This is the argument. The evidence was the rest of the log." in html
    assert 'class="eng-badge measured"' not in html


def test_architecture_page_shows_both_memory_systems(client):
    """Neither replaces the other, so neither may be dropped — and the pair
    only justifies itself if each one's limit is stated too."""
    html = client.get("/engineering/architecture").get_data(as_text=True)
    assert "Embedding space" in html and "Knowledge graph" in html
    assert html.count('class="eng-memory-limit"') == 2


def test_architecture_page_names_all_eight_paths(client):
    html = client.get("/engineering/architecture").get_data(as_text=True)
    for path in ("Transcript", "Resonance", "Perception", "Reflex",
                 "Reasoning", "Intuition", "Truth", "Reflection"):
        assert path in html, path
    for kind in ("write", "read", "feedback", "deferred"):
        assert f'eng-path-kind {kind}' in html, kind


def test_resonance_is_singled_out(client):
    """Path 2 is the least obvious and the most load-bearing: clustering
    discovers graph nodes no conversation ever stated. If it reads as one of
    eight equals, the diagram has failed to make its own argument."""
    html = client.get("/engineering/architecture").get_data(as_text=True)
    assert "eng-arch-flow key" in html          # the one solid arrow
    assert "eng-arch-chip key" in html
    assert "no single conversation ever stated" in html


def test_architecture_page_separates_deciding_from_speaking(client):
    html = client.get("/engineering/architecture").get_data(as_text=True)
    assert "The Conductor decides. The LLM speaks." in html


def test_architecture_page_carries_the_standing_commitments(client):
    """Six positions that were invisible on the site before, each one a thing
    a reader would otherwise have to be told in person."""
    html = client.get("/engineering/architecture").get_data(as_text=True)
    for commitment in ("Backbone stable", "Privacy is the representation",
                       "data-history-bound", "Rules ship before learning",
                       "Classic ML has three permanent roles",
                       "brain-systems decomposition"):
        assert commitment in html, commitment


def test_architecture_page_keeps_the_prediction_loop_falsifiable(client):
    html = client.get("/engineering/architecture").get_data(as_text=True)
    assert "eng-loop-step" in html
    assert "five cadences" in html


def test_ladder_grades_the_unbuilt_rungs(client):
    """One dashed style for all three flattened 'decided but unbuilt' into the
    same bucket as 'open research bet'."""
    html = client.get("/engineering/architecture").get_data(as_text=True)
    for grade in ("next · thresholds locked", "planned · phases 3–5",
                  "research · open bet"):
        assert grade in html, grade


def test_ladder_shows_where_the_other_sensors_enter(client):
    """The goal is multimodal; a camera-only ladder hid that entirely."""
    html = client.get("/engineering/architecture").get_data(as_text=True)
    assert "+ mmWave radar" in html
    assert "+ wearables" in html


def test_ladder_caption_names_the_real_selection_metric(client):
    """The winner was crowned on TAR@FAR, not on identification accuracy, and
    the non-mate set was impostors AND strangers by design."""
    html = client.get("/engineering/architecture").get_data(as_text=True)
    flat = " ".join(html.split())
    assert "TAR at FAR" in flat
    assert "3,325 non-mate comparisons" in flat
    assert "3,000 stranger and 325 impostor" in flat


def test_direction_data_survives_empty_artifacts(monkeypatch):
    """A fresh clone with no runs must still render the destination — the goal
    does not depend on having measured anything yet."""
    monkeypatch.setattr(data, "load_runs", lambda path: [])
    payload = data.direction_page_data()
    assert payload["recognition_winner"] is None
    assert payload["detection_images"] == 0
    assert payload["detection_datasets"] == []
    assert payload["recognition_counts"] is None


def test_architecture_page_renders_without_any_runs(client, monkeypatch):
    monkeypatch.setattr(data, "load_runs", lambda path: [])
    response = client.get("/engineering/architecture")
    assert response.status_code == 200


def test_every_page_offers_the_architecture(client):
    for path in ("/engineering/", "/engineering/detection",
                 "/engineering/recognition", "/engineering/routing"):
        html = client.get(path).get_data(as_text=True)
        assert "/engineering/architecture" in html, path


def test_the_old_urls_still_land(client):
    """Both of these were real for part of a day while the ordering moved."""
    assert client.get("/engineering/decisions").headers["Location"].endswith("/engineering/")
    assert client.get("/engineering/direction").headers["Location"].endswith("/architecture")


def test_index_links_the_roadmap_anchor_the_architecture_promises(client):
    overview = client.get("/engineering/").get_data(as_text=True)
    architecture = client.get("/engineering/architecture").get_data(as_text=True)
    assert 'id="roadmap"' in overview
    assert "#roadmap" in architecture


def test_recognition_page_labels_both_thresholds(client):
    """The same distinction test_benchmark_report guards in the terminal."""
    html = client.get("/engineering/recognition").get_data(as_text=True)
    assert "yardstick" in html
    assert "deployment" in html.lower()
    assert "max-F1" in html


def test_recognition_page_claims_exactness_only_with_a_recorded_sweep(client):
    """The live artifact carries a sweep, so the explorer is exact and says so.

    The claim and the caveat are a matched pair: exactly one of them belongs on
    the page at a time, and which one is decided by the data, never by an
    editor remembering to update the prose.
    """
    html = client.get("/engineering/recognition").get_data(as_text=True)
    assert "readout is <strong>exact</strong>" in html
    assert "binned approximation" not in html


def test_recognition_page_states_the_binned_caveat_without_a_sweep(client, monkeypatch):
    """A run predating curve persistence must admit the explorer is estimating."""
    monkeypatch.setattr(data, "load_runs", lambda path: [_recognition_run()])
    html = client.get("/engineering/recognition").get_data(as_text=True)
    assert "binned approximation" in html
    assert "readout is <strong>exact</strong>" not in html


def test_explorer_names_the_shipped_encoder_and_threshold(client):
    """The explorer must say what was actually shipped, not just offer a slider.

    A threshold with no model beside it is unactionable, and a slider with no
    marked starting point invites reading any position as the decision.
    """
    html = client.get("/engineering/recognition").get_data(as_text=True)
    shipped = html[html.find('class="eng-shipped"'):][:600]
    assert "arcface" in shipped
    assert "multi-reference" in shipped
    assert "0.342" in shipped


def test_explorer_is_given_the_shipped_marker_provenance(client):
    """The marker needs BOTH halves of its provenance, not just the number.

    The encoder decides whether the line is drawn at all (a cosine score does
    not transfer between embedding spaces) and the aggregation decides whether
    it is drawn live or dim. A threshold shipped without them would be drawn
    everywhere, which is the bug this scoping exists to prevent.
    """
    html = client.get("/engineering/recognition").get_data(as_text=True)
    # Unrounded on purpose: the marker is drawn at the threshold that actually
    # ships, and only rounded for its label. Feeding it 0.342 would place the
    # line a thousandth away from the value the gallery decides with.
    assert "data-shipped='0.3416'" in html
    assert "data-shipped-encoder='arcface'" in html
    assert "data-shipped-aggregation='multi-reference'" in html
    assert 'id="explorer-scope"' in html


def test_recognition_page_embeds_explorer_data(client):
    html = client.get("/engineering/recognition").get_data(as_text=True)
    assert "data-distributions" in html


def test_assistant_page_links_to_the_log_from_the_header_tagline(client):
    """Permanent and quiet, and NOT among the controls.

    The link hangs off the descriptive tagline rather than sitting in the
    status bar (a row of live state and toggles) or in a footer bar (which
    would permanently shorten the chat transcript in a 100vh shell).
    """
    html = client.get("/").get_data(as_text=True)
    assert "/engineering/" in html
    assert "header-doc-link" in html

    header = html[html.index("<h1>J.A.R.V.I.S."):html.index('class="status-bar"')]
    assert "/engineering/" in header, "the link is not in the header block"


def test_the_log_link_is_not_a_control(client):
    """It must not live in the status bar beside the TTS toggle."""
    html = client.get("/").get_data(as_text=True)
    status_bar = html[html.index('class="status-bar"'):]
    status_bar = status_bar[:status_bar.index("</div>", status_bar.index("tts-toggle"))]
    assert "/engineering/" not in status_bar


def test_overview_leads_with_reasoning(client):
    """The decisions must come before the tables, not after them."""
    html = client.get("/engineering/").get_data(as_text=True)
    assert "eng-decision" in html
    assert html.index("Decisions") < html.index("What got locked in")


def test_overview_states_its_limits(client):
    html = client.get("/engineering/").get_data(as_text=True)
    assert "don&#39;t tell you" in html or "don't tell you" in html
    assert "like-for-like" in html   # the AP-vs-F1 honesty note


def test_engineering_pages_do_not_touch_the_assistant(client):
    """No conversation state, no device state, no Claude call."""
    import jarvis_web
    before = list(jarvis_web.conversation_history)
    client.get("/engineering/recognition")
    assert jarvis_web.conversation_history == before
