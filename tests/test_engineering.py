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


@pytest.mark.parametrize("url", ["/engineering/", "/engineering/detection",
                                 "/engineering/recognition"])
def test_pages_render(client, url):
    response = client.get(url)
    assert response.status_code == 200


def test_recognition_page_labels_both_thresholds(client):
    """The same distinction test_benchmark_report guards in the terminal."""
    html = client.get("/engineering/recognition").get_data(as_text=True)
    assert "yardstick" in html
    assert "deployment" in html.lower()
    assert "max-F1" in html


def test_recognition_page_states_the_binned_caveat(client):
    """The explorer is an approximation and must never imply otherwise."""
    html = client.get("/engineering/recognition").get_data(as_text=True)
    assert "binned approximation" in html


def test_recognition_page_embeds_explorer_data(client):
    html = client.get("/engineering/recognition").get_data(as_text=True)
    assert "data-distributions" in html


def test_assistant_page_links_to_the_log_from_the_footer(client):
    """Permanent and quiet: in the footer, not among the status-bar controls."""
    html = client.get("/").get_data(as_text=True)
    assert "/engineering/" in html
    assert "app-footer" in html
    footer = html[html.index("app-footer"):]
    assert "/engineering/" in footer[:400]


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
