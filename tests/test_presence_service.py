"""
Test suite for the state machine in tools/presence_service.py.

Covers PresenceTracker only — the part that decides when somebody has arrived
and when they have left. It is pure: no camera, no broker, no model. Every test
injects the clock (`now_mono`) rather than sleeping, so a ten-second departure
timeout is exercised in microseconds and the results cannot flake on a slow
machine.

The pipeline around it (capture, detect, embed, match) needs real hardware and
real weights, and is exercised by running the service.

Run:
  pytest tests/test_presence_service.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from presence_service import (  # noqa: E402  (import follows the sys.path tweak)
    STATE_ABSENT,
    STATE_PRESENT,
    PresenceConfig,
    PresenceTracker,
    Transition,
    topic_for,
)

NAME = "michael"
SCORE = 0.9


@pytest.fixture
def config():
    return PresenceConfig()


@pytest.fixture
def tracker(config):
    return PresenceTracker([NAME], config)


def hit(tracker, at, score=SCORE):
    """One processed frame in which `NAME` was recognised."""
    return tracker.update({NAME: score}, now_mono=at, now_iso="2026-08-30T00:00:00Z")


def miss(tracker, at):
    """One processed frame in which nobody was recognised."""
    return tracker.update({}, now_mono=at, now_iso="2026-08-30T00:00:00Z")


def arrive(tracker, start=0.0, step=0.1):
    """Drive the tracker to 'present' and return the arrival transition."""
    transitions = []
    for i in range(PresenceConfig().hits_required):
        transitions += hit(tracker, start + i * step)
    assert tracker.state(NAME) == STATE_PRESENT
    return transitions[-1]


# ── arrival: N of M ──────────────────────────────────────────────

def test_arrival_fires_on_the_nth_hit_and_not_before(tracker, config):
    """Nothing is published until the Nth hit, and it fires exactly there."""
    for i in range(config.hits_required - 1):
        assert hit(tracker, i * 0.1) == []
        assert tracker.state(NAME) == STATE_ABSENT

    transitions = hit(tracker, (config.hits_required - 1) * 0.1)
    assert len(transitions) == 1
    assert transitions[0].state == STATE_PRESENT
    assert transitions[0].name == NAME
    assert tracker.state(NAME) == STATE_PRESENT


def test_misses_inside_the_window_still_allow_arrival(tracker):
    """3 hits in 5 frames is enough even when they are not consecutive."""
    hit(tracker, 0.0)
    hit(tracker, 0.1)
    assert miss(tracker, 0.2) == []
    transitions = hit(tracker, 0.3)
    assert [t.state for t in transitions] == [STATE_PRESENT]


def test_hits_spread_wider_than_the_window_do_not_fire(tracker, config):
    """Hits that never coexist inside M frames must not accumulate.

    One hit every M frames means the window only ever holds one at a time, so
    no matter how long this runs the count never reaches N.
    """
    now = 0.0
    for _ in range(6):
        assert hit(tracker, now) == []
        now += 0.1
        for _ in range(config.window_frames - 1):
            assert miss(tracker, now) == []
            now += 0.1
    assert tracker.state(NAME) == STATE_ABSENT


def test_oldest_hit_falls_out_of_the_frame_window(tracker):
    """A hit pushed out by maxlen stops counting.

    Hits at frames 0 and 4 sit M=5 apart; by frame 5 the first has been evicted,
    so frames 4 and 5 are only two hits and arrival must not fire.
    """
    hit(tracker, 0.0)
    for i in range(1, 4):
        miss(tracker, i * 0.1)
    hit(tracker, 0.4)
    assert hit(tracker, 0.5) == []
    assert tracker.state(NAME) == STATE_ABSENT


# ── arrival: the age ceiling ─────────────────────────────────────

# These pin explicit numbers instead of reading max_observation_age off the
# config. Deriving the test's timings from the value under test makes the test
# scale with it, so it keeps passing when the ceiling is widened or disabled -
# which is exactly the regression it is supposed to catch.
AGE_CFG = PresenceConfig(max_observation_age=2.0)


def test_stale_observations_do_not_count_toward_arrival():
    """Hits inside the frame window but older than the ceiling are ignored -
    the throughput-collapse case the ceiling exists for."""
    tracker = PresenceTracker([NAME], AGE_CFG)
    tracker.update({NAME: SCORE}, now_mono=0.0, now_iso="T")
    tracker.update({NAME: SCORE}, now_mono=0.5, now_iso="T")
    # The third frame lands 10s later, as it would if the pipeline stalled. The
    # first two are still among the last M frames, but at 10s and 9.5s old they
    # are well past the 2.0s ceiling and are no longer evidence of anybody.
    assert tracker.update({NAME: SCORE}, now_mono=10.0, now_iso="T") == []
    assert tracker.state(NAME) == STATE_ABSENT


def test_the_same_frame_pattern_fires_when_the_hits_are_fresh():
    """Contrast with the test above: identical frames, only the clock differs."""
    tracker = PresenceTracker([NAME], AGE_CFG)
    tracker.update({NAME: SCORE}, now_mono=0.0, now_iso="T")
    tracker.update({NAME: SCORE}, now_mono=0.5, now_iso="T")
    transitions = tracker.update({NAME: SCORE}, now_mono=1.0, now_iso="T")
    assert [t.state for t in transitions] == [STATE_PRESENT]


def test_hits_exactly_at_the_age_ceiling_still_count():
    """The boundary is inclusive: exactly max_observation_age old counts."""
    tracker = PresenceTracker([NAME], AGE_CFG)
    tracker.update({NAME: SCORE}, now_mono=0.0, now_iso="T")
    tracker.update({NAME: SCORE}, now_mono=0.0, now_iso="T")
    transitions = tracker.update({NAME: SCORE}, now_mono=2.0, now_iso="T")
    assert [t.state for t in transitions] == [STATE_PRESENT]


def test_one_tick_past_the_ceiling_does_not_count():
    """...and just past it does not."""
    tracker = PresenceTracker([NAME], AGE_CFG)
    tracker.update({NAME: SCORE}, now_mono=0.0, now_iso="T")
    tracker.update({NAME: SCORE}, now_mono=0.0, now_iso="T")
    assert tracker.update({NAME: SCORE}, now_mono=2.001, now_iso="T") == []


def test_default_config_is_the_locked_operating_point():
    """N/M/T and the age ceiling are a locked operating point, not knobs.

    This test exists so that changing one is a deliberate, reviewed act rather
    than a quiet edit - the same reason the detection threshold is pinned to
    its benchmark in test_pipeline_config.py.
    """
    config = PresenceConfig()
    assert config.hits_required == 3
    assert config.window_frames == 5
    assert config.absent_after == 10.0
    assert config.max_observation_age == 5.0


def test_default_ceiling_is_inert_at_realistic_throughput():
    """At the ~7 fps the pipeline measures at, the default never interferes."""
    config = PresenceConfig()
    tracker = PresenceTracker([NAME], config)
    for i in range(config.hits_required - 1):
        assert tracker.update({NAME: SCORE}, now_mono=i / 7.0, now_iso="T") == []
    fired = tracker.update({NAME: SCORE}, now_mono=(config.hits_required - 1) / 7.0,
                           now_iso="T")
    assert [t.state for t in fired] == [STATE_PRESENT]


# ── departure: T seconds ─────────────────────────────────────────

def test_departure_fires_at_exactly_t_and_not_just_under(tracker, config):
    """The boundary is T inclusive, so T-epsilon must not fire."""
    arrive(tracker)
    last_hit = (config.hits_required - 1) * 0.1

    assert miss(tracker, last_hit + config.absent_after - 0.001) == []
    assert tracker.state(NAME) == STATE_PRESENT

    transitions = miss(tracker, last_hit + config.absent_after)
    assert len(transitions) == 1
    assert transitions[0].state == STATE_ABSENT
    assert tracker.state(NAME) == STATE_ABSENT


def test_the_clock_runs_from_the_last_hit_not_from_arrival(tracker, config):
    """Someone who keeps being seen never times out."""
    arrive(tracker)
    now = 10.0
    for _ in range(5):
        assert hit(tracker, now) == []
        now += config.absent_after - 1.0
    assert tracker.state(NAME) == STATE_PRESENT


# ── publishing discipline ────────────────────────────────────────

def test_a_long_run_of_hits_publishes_exactly_once(tracker):
    """Staying put produces no traffic after the arrival."""
    published = []
    for i in range(200):
        published += hit(tracker, i * 0.1)
    assert len(published) == 1
    assert published[0].state == STATE_PRESENT


def test_a_long_run_of_misses_publishes_nothing(tracker):
    """Never having been there is not a transition."""
    published = []
    for i in range(200):
        published += miss(tracker, i * 0.1)
    assert published == []
    assert tracker.state(NAME) == STATE_ABSENT


# ── recovery ─────────────────────────────────────────────────────

def test_present_absent_present_round_trip(tracker, config):
    """The full cycle, and the second arrival must be published too."""
    first = arrive(tracker)
    assert first.state == STATE_PRESENT

    departure = miss(tracker, 100.0)
    assert [t.state for t in departure] == [STATE_ABSENT]

    second = []
    for i in range(config.hits_required):
        second += hit(tracker, 200.0 + i * 0.1)
    assert [t.state for t in second] == [STATE_PRESENT]
    assert tracker.state(NAME) == STATE_PRESENT


def test_departure_clears_the_window_so_one_hit_cannot_readmit(tracker):
    """Hits from before the timeout must not survive it.

    Without clearing, the deque would still hold the pre-departure hits and a
    single new one could push the count back to N immediately.
    """
    arrive(tracker)
    miss(tracker, 100.0)
    assert tracker.state(NAME) == STATE_ABSENT

    assert hit(tracker, 101.0) == []
    assert hit(tracker, 101.1) == []
    assert tracker.state(NAME) == STATE_ABSENT
    assert hit(tracker, 101.2) != []


# ── independence and payload ─────────────────────────────────────

def test_identities_are_tracked_independently(config):
    tracker = PresenceTracker(["michael", "aaron"], config)
    transitions = []
    for i in range(config.hits_required):
        transitions += tracker.update({"michael": SCORE}, now_mono=i * 0.1, now_iso="T")
    assert [(t.name, t.state) for t in transitions] == [("michael", STATE_PRESENT)]
    assert tracker.state("aaron") == STATE_ABSENT


def test_one_person_leaving_does_not_evict_the_other(config):
    tracker = PresenceTracker(["michael", "aaron"], config)
    for i in range(config.hits_required):
        tracker.update({"michael": SCORE, "aaron": SCORE}, now_mono=i * 0.1, now_iso="T")
    assert tracker.state("michael") == STATE_PRESENT
    assert tracker.state("aaron") == STATE_PRESENT

    # Only michael keeps being seen.
    now = 1.0
    transitions = []
    while now < 1.0 + config.absent_after + 1.0:
        transitions += tracker.update({"michael": SCORE}, now_mono=now, now_iso="T")
        now += 1.0
    assert [(t.name, t.state) for t in transitions] == [("aaron", STATE_ABSENT)]
    assert tracker.state("michael") == STATE_PRESENT


def test_arrival_payload_carries_the_triggering_confidence(tracker):
    hit(tracker, 0.0, score=0.40)
    hit(tracker, 0.1, score=0.50)
    transition = hit(tracker, 0.2, score=0.77)[0]
    assert transition.confidence == pytest.approx(0.77)
    assert transition.last_seen == "2026-08-30T00:00:00Z"


def test_departure_payload_keeps_the_last_sighting(tracker):
    """'absent' reports when the person was last actually seen, not now."""
    for i in range(PresenceConfig().hits_required):
        tracker.update({NAME: SCORE}, now_mono=i * 0.1, now_iso="SEEN")
    transition = tracker.update({}, now_mono=100.0, now_iso="NOW")[0]
    assert transition.state == STATE_ABSENT
    assert transition.timestamp == "NOW"
    assert transition.last_seen == "SEEN"


def test_transition_payload_shape():
    """The published JSON keys are a contract for every subscriber."""
    payload = Transition(
        name=NAME, state=STATE_PRESENT, confidence=0.123456,
        timestamp="2026-08-30T00:00:00Z", last_seen="2026-08-30T00:00:00Z",
    ).payload()
    assert set(payload) == {"state", "name", "confidence", "timestamp", "last_seen"}
    assert payload["confidence"] == 0.1235


def test_never_seen_person_has_null_last_seen(config):
    """A subscriber must be able to tell 'gone' from 'never here'."""
    tracker = PresenceTracker([NAME], config)
    for i in range(config.hits_required):
        tracker.update({NAME: SCORE}, now_mono=i * 0.1, now_iso="T")
    assert tracker._last_seen_iso[NAME] is not None
    assert PresenceTracker(["nobody"], config)._last_seen_iso["nobody"] is None


# ── topic mapping ────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("michael", "jarvis/presence/michael"),
    ("Michael", "jarvis/presence/michael"),
    ("mary jane", "jarvis/presence/mary-jane"),
    ("o'brien", "jarvis/presence/o-brien"),
])
def test_topic_slugging(name, expected):
    assert topic_for(name) == expected


@pytest.mark.parametrize("name", ["a/b", "a+b", "a#b"])
def test_topic_never_contains_mqtt_metacharacters(name):
    """'+' and '#' are illegal in a published topic; '/' would forge a level."""
    topic = topic_for(name)
    assert topic.count("/") == 2
    assert "+" not in topic and "#" not in topic


def test_status_topic_is_outside_the_presence_branch():
    """jarvis/presence/+ must mean 'people' and nothing else."""
    from presence_service import SERVICE_TOPIC, TOPIC_PREFIX

    assert not SERVICE_TOPIC.startswith(TOPIC_PREFIX + "/")
