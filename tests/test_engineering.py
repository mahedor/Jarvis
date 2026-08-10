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


@pytest.mark.parametrize("url", ["/engineering/", "/engineering/routing",
                                 "/engineering/detection", "/engineering/recognition"])
def test_pages_render(client, url):
    response = client.get(url)
    assert response.status_code == 200


def test_routing_page_states_the_accuracy_gap(client):
    """Latency is not accuracy, and the page must not let that slide."""
    html = client.get("/engineering/routing").get_data(as_text=True)
    assert "never been measured" in html


def test_routing_page_distinguishes_engaged_from_unengaged_caching(client):
    html = client.get("/engineering/routing").get_data(as_text=True)
    assert "caching never engaged" in html
    assert "cache genuinely used" in html


def test_log_is_not_only_face_work(client):
    """The assistant is an LLM product; a build log that is entirely vision
    misrepresents the project."""
    html = client.get("/engineering/").get_data(as_text=True)
    assert "/engineering/routing" in html


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
