"""
Test suite for the printed report in tools/benchmark_recognition.py.

Guards ONE distinction, which has now regressed twice:

    the TAR@FAR threshold is the YARDSTICK - it pins every encoder to the same
    false-accept budget so they can be ranked fairly, and must never be
    deployed;

    the max-F1 threshold is what SHIPS - it goes into the gallery.

Both must appear in the WINNER block, each next to its own label, and the
build_gallery command the report hands over must carry the deployment one. A
report that prints a single unlabelled "recommended threshold" is the bug.

print_report is a plain function over already-parsed results, so this needs no
crops, encoders, LFW or disk - just a synthetic payload and capsys.

Run:
  pytest tests/test_benchmark_report.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from benchmark_recognition import print_report  # noqa: E402

# Deliberately far apart so no assertion can pass by coincidence.
YARDSTICK_THRESHOLD = 0.214
DEPLOYMENT_THRESHOLD = 0.342


def _cell(tar, f1_threshold, tar_threshold, f1=0.984):
    """One (encoder, aggregation) result, shaped like score_aggregation's."""
    return {
        "rank1_accuracy": 0.985,
        "tar_at_far": {
            "0.01": {"tar": tar, "threshold": tar_threshold, "far_achieved": 0.0099},
            "0.001": {"tar": tar - 0.02, "threshold": tar_threshold + 0.08,
                      "far_achieved": 0.0009},
        },
        "f1_threshold": {"threshold": f1_threshold, "precision": 1.0,
                         "recall": 0.969, "f1": f1},
        "separation": {"dprime_vs_impostor": 5.8, "dprime_vs_stranger": 5.77},
        "distributions": {"genuine": None, "impostor": None, "stranger": None},
        "counts": {"genuine": 65, "impostor": 325, "stranger": 3000},
    }


@pytest.fixture
def report(capsys):
    """Print a two-encoder x two-aggregation report and return its stdout.

    The winner is arcface + multi-reference on TAR, and its two thresholds
    differ, which is the whole point of the fixture.
    """
    results = [
        {
            "encoder": "arcface",
            "aggregations": {
                "mean-renormalize": _cell(0.969, 0.307, 0.191),
                "multi-reference": _cell(0.985, DEPLOYMENT_THRESHOLD, YARDSTICK_THRESHOLD),
            },
            "latency": {"count": 633, "mean_ms": 187.8, "p50_ms": 180.0, "p95_ms": 249.5},
            "embed_failures": 0, "fallback_count": 3,
            "persons_scored": ["a", "b"], "persons_skipped": [], "num_strangers": 500,
        },
        {
            "encoder": "facenet512",
            "aggregations": {
                "mean-renormalize": _cell(0.939, 0.560, 0.413),
                "multi-reference": _cell(0.954, 0.641, 0.492),
            },
            "latency": {"count": 633, "mean_ms": 123.1, "p50_ms": 120.0, "p95_ms": 251.5},
            "embed_failures": 0, "fallback_count": 7,
            "persons_scored": ["a", "b"], "persons_skipped": [], "num_strangers": 500,
        },
    ]
    print_report(results, ["mean-renormalize", "multi-reference"], 6, 500, 0.5)
    return capsys.readouterr().out


def _winner_block(report):
    """The WINNER section only — everything from the WINNER line onward."""
    assert "WINNER" in report, "the report printed no WINNER line"
    return report[report.index("WINNER"):]


# ─── the winner itself ───────────────────────────────────────────

def test_winner_names_the_encoder_and_the_aggregation(report):
    """Ranking only the default aggregation was the original bug."""
    block = _winner_block(report)
    assert "arcface" in block
    assert "multi-reference" in block


def test_winner_is_the_best_cell_not_the_first_aggregation(report):
    block = _winner_block(report)
    first_line = block.splitlines()[0]
    assert "mean-renormalize" not in first_line


# ─── both thresholds, each labelled ──────────────────────────────

def test_both_thresholds_are_printed(report):
    block = _winner_block(report)
    assert f"{YARDSTICK_THRESHOLD:.3f}" in block, "yardstick threshold missing"
    assert f"{DEPLOYMENT_THRESHOLD:.3f}" in block, "deployment threshold missing"


def test_yardstick_threshold_is_labelled_as_such(report):
    """The TAR@FAR number must be findable as the yardstick, on its own line."""
    lines = [ln for ln in _winner_block(report).splitlines()
             if f"{YARDSTICK_THRESHOLD:.3f}" in ln]
    assert lines, "no line carries the yardstick threshold"
    assert any("yardstick" in ln.lower() for ln in lines), (
        f"yardstick threshold is printed unlabelled: {lines}"
    )


def test_deployment_threshold_is_labelled_as_such(report):
    lines = [ln for ln in _winner_block(report).splitlines()
             if f"{DEPLOYMENT_THRESHOLD:.3f}" in ln]
    assert lines, "no line carries the deployment threshold"
    assert any("deployment" in ln.lower() for ln in lines), (
        f"deployment threshold is printed unlabelled: {lines}"
    )
    assert any("max-f1" in ln.lower() for ln in lines), (
        "the deployment threshold does not say it is the max-F1 point"
    )


def test_the_two_labels_are_not_on_the_same_line(report):
    """Collapsing them onto one line is how the distinction gets lost."""
    block = _winner_block(report)
    for line in block.splitlines():
        both = (f"{YARDSTICK_THRESHOLD:.3f}" in line
                and f"{DEPLOYMENT_THRESHOLD:.3f}" in line)
        assert not both, f"both thresholds crammed into one line: {line!r}"


def test_ranking_metric_is_named_next_to_the_yardstick(report):
    lines = [ln for ln in _winner_block(report).splitlines()
             if f"{YARDSTICK_THRESHOLD:.3f}" in ln]
    assert any("TAR@FAR" in ln for ln in lines)


# ─── the handoff command carries the RIGHT number ────────────────

def test_handoff_command_is_printed(report):
    block = _winner_block(report)
    assert "build_gallery.py" in block, "the report does not hand over a command"


def test_handoff_command_carries_the_deployment_threshold(report):
    """The strongest guard: the command must ship max-F1, never the yardstick.
    Pasting the yardstick into build_gallery is the exact mistake this whole
    labelling exists to prevent."""
    command = next(ln for ln in _winner_block(report).splitlines()
                   if "build_gallery.py" in ln)
    assert f"--threshold {DEPLOYMENT_THRESHOLD:.3f}" in command
    assert f"{YARDSTICK_THRESHOLD:.3f}" not in command


def test_handoff_command_carries_the_winning_pair(report):
    command = next(ln for ln in _winner_block(report).splitlines()
                   if "build_gallery.py" in ln)
    assert "--encoder arcface" in command
    assert "--aggregation multi-reference" in command


# ─── the convention is stated in the banner too ──────────────────

def test_banner_states_which_metric_ranks_and_which_ships(report):
    header = report[:report.index("Aggregation:")]
    assert "yardstick" in header.lower()
    assert "ship" in header.lower()
