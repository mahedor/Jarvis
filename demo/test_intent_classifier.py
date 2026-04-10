"""
Test suite for the JARVIS intent classifier.

Run directly:
  python demo/test_intent_classifier.py
"""

from datetime import datetime
from intent_classifier import (
    TIER_DIRECT, TIER_LOCAL, TIER_CLAUDE,
    SPACY_AVAILABLE, EMBEDDINGS_AVAILABLE,
    classify, _classify_input_type,
)

# ─── Shared device state fixture ───────────────────────────────
TEST_STATES = {
    "light.bedroom":        {"state": "off",    "friendly_name": "Bedroom main light", "brightness": 0},
    "light.bedroom_lamp":   {"state": "on",     "friendly_name": "Bedroom lamp",       "brightness": 128},
    "switch.bedroom_fan":   {"state": "off",    "friendly_name": "Bedroom fan"},
    "cover.bedroom_blinds": {"state": "closed", "friendly_name": "Bedroom blinds"},
}

# Convenient time fixtures for time-aware tests
_9pm   = datetime(2024, 1, 1, 21, 0)
_2pm   = datetime(2024, 1, 1, 14, 0)
_8pm   = datetime(2024, 1, 1, 20, 0)    # boundary: exactly 8pm → triggers
_759pm = datetime(2024, 1, 1, 19, 59)   # one minute before → falls to Claude


# ─── Intent Classifier Tests ───────────────────────────────────
# Each entry is (command, expected_tier) or (command, expected_tier, _now).
# _now overrides datetime.now() — used to test time-aware commands.

CLASSIFY_TESTS = [
    # ── Tier 1 — keyword path ────────────────────────────
    ("What time is it?",          TIER_DIRECT),
    ("What's the date today?",    TIER_DIRECT),
    # Tier 1 via embeddings (novel phrasings)
    ("Tell me the current time",  TIER_DIRECT),
    ("What hour is it?",          TIER_DIRECT),
    ("What day is today?",        TIER_DIRECT),

    # ── Tier 2 — keyword path ────────────────────────────
    ("Turn on the bedroom lights",   TIER_LOCAL),
    ("Turn off the lamp",            TIER_LOCAL),
    ("Dim the lamp to 50%",          TIER_LOCAL),
    ("Open the blinds",              TIER_LOCAL),
    ("Close the blinds",             TIER_LOCAL),
    ("Turn on the fan",              TIER_LOCAL),
    ("Turn off everything",          TIER_LOCAL),
    ("Switch on the lights",         TIER_LOCAL),
    ("Kill the lights",              TIER_LOCAL),
    ("Shut the blinds",              TIER_LOCAL),

    # ── Tier 2 — normalization (Option A) ───────────────
    ("Hey Jarvis, turn on the lights",  TIER_LOCAL),
    ("Can you please turn off the fan", TIER_LOCAL),
    ("Could you open the blinds",       TIER_LOCAL),
    ("Please dim the lamp to 30%",      TIER_LOCAL),

    # ── Tier 2 — stacked filler prefixes ────────────────
    ("Hey Jarvis, can you turn on the lights",          TIER_LOCAL),
    ("Hey Jarvis, please turn off the fan",             TIER_LOCAL),
    ("Hey Jarvis, could you open the blinds",           TIER_LOCAL),
    ("Hey Jarvis, can you please turn on the lights",   TIER_LOCAL),
    ("Hey Jarvis, could you please close the blinds",   TIER_LOCAL),
    ("Jarvis, please can you turn off the lamp",        TIER_LOCAL),
    ("Hey Jarvis, would you please turn on the fan",    TIER_LOCAL),

    # ── Tier 2 — spaCy path (Option C) ──────────────────
    ("The lamp should be switched off", TIER_LOCAL),
    ("I need the blinds raised",        TIER_LOCAL),

    # ── Tier 2 — embedding path (Option D) ──────────────
    ("Cut the power to the lights",           TIER_LOCAL),
    ("Let some light in through the blinds",  TIER_LOCAL),
    ("Put the fan on",                        TIER_LOCAL),

    # ── Tier 2 — status queries ──────────────────────────
    ("What's the status of everything?", TIER_LOCAL),
    ("Is the light on?",                 TIER_LOCAL),
    ("Status?",                          TIER_LOCAL),
    ("What's on?",                       TIER_LOCAL),

    # ── Tier 2 — already done (lamp is already on) ──────
    ("Turn on the lamp", TIER_LOCAL),

    # ── Goodmorning — time-aware ─────────────────────────
    ("Good morning",     TIER_LOCAL,  datetime(2024, 1, 1,  7, 0)),  # 7am → triggers
    ("Goodmorning",      TIER_LOCAL,  datetime(2024, 1, 1,  5, 0)),  # exactly 5am → triggers
    ("Morning",          TIER_LOCAL,  datetime(2024, 1, 1,  9, 30)), # phrase variant
    ("Good morning",     TIER_LOCAL,  datetime(2024, 1, 1, 11, 59)), # 11:59am → triggers
    ("Good morning",     TIER_CLAUDE, datetime(2024, 1, 1, 12, 0)),  # noon → falls to Claude
    ("Good morning",     TIER_CLAUDE, datetime(2024, 1, 1,  3, 0)),  # 3am → falls to Claude

    # ── Goodnight — time-aware ───────────────────────────
    ("Goodnight",        TIER_LOCAL,  _9pm),    # 9pm → triggers
    ("Good night",       TIER_LOCAL,  _9pm),    # alternate spelling
    ("I'm going to bed", TIER_LOCAL,  _9pm),    # phrase variant
    ("Goodnight",        TIER_LOCAL,  _8pm),    # exactly 8pm → triggers
    ("Goodnight",        TIER_CLAUDE, _759pm),  # 7:59pm → falls to Claude
    ("Goodnight",        TIER_CLAUDE, _2pm),    # afternoon → falls to Claude

    # ── Tier 3 — should go to Claude ────────────────────
    ("What should I eat for dinner?",   TIER_CLAUDE),
    ("Tell me a joke",                  TIER_CLAUDE),
    ("How productive was I today?",     TIER_CLAUDE),
    ("What's the meaning of life?",     TIER_CLAUDE),
    ("Set the mood for a movie",        TIER_CLAUDE),

    # ── Edge cases ───────────────────────────────────────
    ("Make things dark",   TIER_CLAUDE),
    ("Turn the heater on", TIER_CLAUDE),
    ("Set an alarm",       TIER_CLAUDE),
    ("Play Spotify",       TIER_CLAUDE),
    ("Call Dad",           TIER_CLAUDE),

    ("Hey man, what's the weather gonna be like?", TIER_CLAUDE),

    # ── Tier 1 false-positive regression tests ───────────
    # "what time" / "current time" in a non-time-query context
    ("What time does the muffin man pull up?",  TIER_CLAUDE),
    ("What's the current time in London?",      TIER_CLAUDE),

    # ── Tier 1 — filler prefix stripped before matching ──
    ("Hey Jarvis, what time is it?",            TIER_DIRECT),
    ("Please tell me the current time",         TIER_DIRECT),

    # ── Tier 2 — already done (more device states) ───────
    ("Turn off the bedroom light",              TIER_LOCAL),   # already off
    ("Turn off the fan",                        TIER_LOCAL),   # already off

    # ── Tier 2 — unknown device → falls to Claude ────────
    ("Turn on the TV",                          TIER_CLAUDE),
    ("Switch on the oven",                      TIER_CLAUDE),

    # ── Tier 2 — action present but no device → Claude ───
    ("Turn on",                                 TIER_CLAUDE),

    # ── Tier 2 — mixed case / shouting ───────────────────
    ("TURN ON THE LIGHTS",                      TIER_LOCAL),
    ("Turn ON the LAMP",                        TIER_LOCAL),

    # ── Tier 2 — untested filler prefixes ────────────────
    ("I'd like to turn off the lamp",           TIER_LOCAL),
    ("I want you to open the blinds",           TIER_LOCAL),
    ("I need you to turn on the fan",           TIER_LOCAL),

    # ── Tier 2 — prefix not in FILLER_PREFIXES (passthrough) ──
    # "i need to" is NOT a listed prefix; keyword match still fires
    ("I need to turn on the lights",            TIER_LOCAL),

    # ── Wake word alone → nothing left after strip → Claude ──
    ("Hey Jarvis",                              TIER_CLAUDE),
    ("Jarvis",                                  TIER_CLAUDE),

    # ── Goodmorning / goodnight with filler prefix ───────
    ("Hey Jarvis, good morning",   TIER_LOCAL,  datetime(2024, 1, 1, 7, 0)),
    ("Hey Jarvis, goodnight",      TIER_LOCAL,  _9pm),
]


# ─── Filler Input-Type Classifier Tests ────────────────────────

FILLER_TESTS = [
    # ── Questions — punctuation shortcut ────────────────────
    ("What should I eat for dinner?",            "question"),
    ("How productive was I today?",              "question"),
    ("Is there anything interesting happening?", "question"),
    ("Why do I always procrastinate?",           "question"),
    ("Can you recommend a movie?",               "question"),
    # '?' beats 'do you think' → should still be question
    ("What do you think about my day?",          "question"),

    # ── Questions — no '?', spaCy wh-word ───────────────────
    ("What's the best way to study",             "question"),
    ("How should I spend my evening",            "question"),
    ("Who invented the internet",                "question"),

    # ── Questions — aux inversion, no '?' ───────────────────
    ("Is this a good idea",                      "question"),
    ("Can you help me think through this",       "question"),
    ("Should I take a break",                    "question"),

    # ── Commands — imperative (spaCy VB, no nsubj) ──────────
    ("Tell me a joke",                           "command"),
    ("Summarize my day",                         "command"),
    ("Recommend something to watch tonight",     "command"),
    ("Help me write a grocery list",             "command"),
    ("Explain how black holes work",             "command"),
    ("Give me a motivational quote",             "command"),
    ("Play something relaxing",                  "command"),

    # ── Statements — declarative (spaCy nsubj) ──────────────
    ("I had a really productive day",            "statement"),
    ("The weather looks nice today",             "statement"),
    ("I'm thinking about learning guitar",       "statement"),
    ("That documentary was really good",         "statement"),
    ("My schedule is packed this week",          "statement"),

    # ── Complaints — keyword ─────────────────────────────────
    ("Ugh I can't focus today",                  "complaint"),
    ("I'm so frustrated right now",              "complaint"),
    ("I hate when this happens",                 "complaint"),
    ("I'm so tired of everything",               "complaint"),
    ("I can't stand how disorganised I am",      "complaint"),

    # ── Reflection — keyword ─────────────────────────────────
    ("I wonder if I'm making the right choices", "reflection"),
    ("Sometimes I think about the future",       "reflection"),
    ("I've been thinking about changing careers","reflection"),
    ("What if things had gone differently",      "reflection"),
]


# ─── Runners ───────────────────────────────────────────────────

def run_classify_tests():
    print("=" * 60)
    print("  JARVIS Intent Classifier — Test Suite")
    print()
    print(f"  spaCy:      {'available' if SPACY_AVAILABLE      else 'NOT INSTALLED'}")
    print(f"  Embeddings: {'available' if EMBEDDINGS_AVAILABLE else 'NOT INSTALLED'}")
    print("=" * 60)
    print()

    passed = failed = 0
    for entry in CLASSIFY_TESTS:
        command, expected_tier, *rest = entry
        _now = rest[0] if rest else None
        result = classify(command, TEST_STATES, _now=_now)
        actual_tier = result["tier"]
        ok = actual_tier == expected_tier
        passed += ok
        failed += not ok
        status = "OK  " if ok else "FAIL"
        response = result["response"] or "(--> Claude)"
        layer = result.get("matched_layer", "?")
        print(f"  [{status}] [Tier {actual_tier}] [{layer}] \"{command}\"")
        print(f"           Intent: {result['intent']} | {response}")
        if result["actions"]:
            for a in result["actions"]:
                extra = f" (data: {a['data']})" if a.get("data") else ""
                print(f"           => {a['service']} -> {a['entity_id']}{extra}")
        print()

    print("-" * 60)
    print(f"  Results: {passed}/{passed+failed} passed", end="")
    print(f" ({failed} failed)" if failed else " — all clear!")
    print()
    return failed


def run_filler_tests():
    print("=" * 60)
    print("  Filler Input-Type Classifier — Test Suite")
    print("=" * 60)
    print()

    f_passed = f_failed = 0
    for phrase, expected in FILLER_TESTS:
        actual = _classify_input_type(phrase)
        ok = actual == expected
        f_passed += ok
        f_failed += not ok
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] [{actual:<10}] \"{phrase}\"", end="")
        if not ok:
            print(f"  <-- expected {expected}", end="")
        print()

    print()
    print("-" * 60)
    print(f"  Results: {f_passed}/{f_passed+f_failed} passed", end="")
    print(f" ({f_failed} failed)" if f_failed else " — all clear!")
    print()
    return f_failed


if __name__ == "__main__":
    failures = run_classify_tests()
    failures += run_filler_tests()
    raise SystemExit(failures)  # non-zero exit if anything failed
