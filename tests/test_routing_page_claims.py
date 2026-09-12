"""
Pins the behavioural claims the /engineering/routing page makes in prose.

WHY THIS EXISTS. The routing page describes the classifier's mechanism with
worked examples — "don't turn on the light" escalates, "turn on the heater"
resolves its verb and dies on the device, "current time in London" is not
answered from the local clock. Prose like that is a claim about behaviour, and
the first draft of that section shipped two examples that were simply false:
spaCy was described as extracting the object (it never touches it), and the
ceiling was illustrated with a phrase whose verb no layer actually recovers.
Both read perfectly well and both were wrong.

So the examples are pinned here. If the classifier's behaviour moves, the page
becomes a lie and the build says so — the same bargain test_pipeline_config.py
strikes for the locked detection threshold.

NOTE ON SCOPE: this guards the page's claims, not the classifier's correctness.
The classifier's own suite lives in test_intent_classifier.py, which is a
standalone script (no test_* functions) and is NOT collected by pytest.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from intent_classifier import (  # noqa: E402
    EMBEDDING_THRESHOLD,
    SPACY_AVAILABLE,
    TIER_CLAUDE,
    TIER_DIRECT,
    TIER_LOCAL,
    _extract_action,
    _extract_device,
    _spacy_extract_action,
    classify,
    normalize,
)

from datetime import datetime  # noqa: E402

STATES = {
    "light.bedroom":        {"state": "off",    "friendly_name": "Bedroom main light", "brightness": 0},
    "switch.bedroom_fan":   {"state": "on",     "friendly_name": "Bedroom fan"},
    "cover.bedroom_blinds": {"state": "closed", "friendly_name": "Bedroom blinds"},
}

needs_spacy = pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy layer disabled at import")


def tier(text, now=None):
    return classify(text, STATES, _now=now)["tier"]


# ─── "normalize" bullet ────────────────────────────────────────

def test_stacked_prefixes_come_off_in_one_pass():
    assert normalize("Hey Jarvis, could you please turn on the light") == "turn on the light"


def test_word_boundary_guard_does_not_eat_a_real_word():
    # The page cites this exact phrase: "please" must not strip "pleased".
    assert normalize("pleased to meet you") == "pleased to meet you"


# ─── "Three places it refuses on purpose" ──────────────────────

def test_negation_escalates_rather_than_acting():
    assert tier("don't turn on the light") == TIER_CLAUDE


def test_negation_guard_sits_after_the_status_checks():
    # The page's argument for the guard's POSITION: this must still answer.
    assert tier("is the fan not on?") == TIER_LOCAL


def test_local_clock_does_not_answer_for_another_timezone():
    assert tier("what's the current time in London") == TIER_CLAUDE
    assert tier("what time is it") == TIER_DIRECT


def test_entity_type_validation_rejects_a_nonsense_pair():
    # Verb and device both resolve; the PAIR is invalid, so it escalates.
    assert _extract_action("open the fan") is not None
    assert _extract_device("open the fan") is not None
    assert tier("open the fan") == TIER_CLAUDE


# ─── "What each layer actually does" ───────────────────────────

@needs_spacy
def test_parse_layer_recovers_a_verb_the_tables_do_not_contain():
    text = normalize("the fan should be switched off")
    assert _extract_action(text) is None, "keyword layer should miss this phrasing"
    assert _spacy_extract_action(text) == "turn_off"
    assert classify("the fan should be switched off", STATES)["matched_layer"] == "tier2_spacy"


def test_embedding_threshold_is_what_the_page_quotes():
    assert EMBEDDING_THRESHOLD == 0.70


# ─── "The ceiling: the cascade only resolves the verb" ─────────

@pytest.mark.parametrize("text", ["turn on the heater", "shut off the kettle"])
def test_a_known_verb_with_an_unknown_device_still_escalates(text):
    """The page's headline example. The verb resolves; the device has no rung."""
    normalized = normalize(text)
    assert _extract_action(normalized) is not None, "verb should resolve"
    assert _extract_device(normalized) is None, "device should not"
    assert tier(text) == TIER_CLAUDE


@needs_spacy
def test_the_kettle_case_has_its_verb_agreed_by_two_layers():
    # The page claims keyword AND parse both resolve this one.
    normalized = normalize("shut off the kettle")
    assert _extract_action(normalized) == "turn_off"
    assert _spacy_extract_action(normalized) == "turn_off"


def test_device_extraction_has_no_fallback_rung():
    """The structural claim behind the ceiling, not just an example of it."""
    import intent_classifier as ic
    source = Path(ic.__file__).read_text(encoding="utf-8")
    body = source.split("def _extract_device(")[1].split("\ndef ")[0]
    for escape_hatch in ("_spacy", "_embed", "_nlp", "_embedder"):
        assert escape_hatch not in body, (
            f"_extract_device now references {escape_hatch} — the routing page's "
            "'no fallback rung behind it' claim is stale"
        )


# ─── "How a command moves through it" ──────────────────────────

def test_probe_two_is_time_gated_in_both_directions():
    assert tier("goodnight", datetime(2024, 1, 1, 21, 0)) == TIER_LOCAL
    assert tier("goodnight", datetime(2024, 1, 1, 14, 0)) == TIER_CLAUDE


def test_probe_two_reports_itself_as_tier_two_keyword():
    result = classify("goodnight", STATES, _now=datetime(2024, 1, 1, 21, 0))
    assert result["tier"] == TIER_LOCAL
    assert result["matched_layer"] == "tier2_keyword"


def test_tier_three_is_an_empty_shell_not_a_decision():
    result = classify("what do you think about jazz", STATES)
    assert result["tier"] == TIER_CLAUDE
    assert result["confidence"] == 0.0
    assert result["response"] is None
    assert result["actions"] == []
    assert result["matched_layer"] == "tier3_claude"
