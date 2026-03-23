"""
Jarvis Intent Classifier
=========================
Routes user commands to the right handler BEFORE hitting Claude.

Tiers:
  Tier 1 — Direct (0ms): time/date queries
  Tier 2 — Local (~10ms keyword, ~20ms spaCy, ~20ms embedding): device commands
  Tier 3 — Claude API (~1-2s): complex, conversational, ambiguous

Parsing pipeline (applied in order, each is a fallback for the one above):
  A. Normalize   — strip filler prefixes ("can you", "please", "hey jarvis", etc.)
  B. Keywords    — substring match against ACTION_MAP / DEVICE_ALIASES (existing)
  C. spaCy       — dependency-parse root verb + particle for novel phrasings
  D. Embeddings  — cosine similarity against canonical phrases for semantic coverage

Startup note: spaCy and sentence-transformers models load at import time (~2-4s
one-time cost). After that, inference is fast and fully local.

Install:
  pip install spacy sentence-transformers numpy
  python -m spacy download en_core_web_sm
"""

import re
from datetime import datetime

# ─── Optional ML dependencies ──────────────────────────────────
try:
    import spacy as _spacy_lib
    _nlp = _spacy_lib.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    _nlp = None
    SPACY_AVAILABLE = False

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    _embedder = None
    EMBEDDINGS_AVAILABLE = False


# ─── Tier Constants ────────────────────────────────────────────
TIER_DIRECT = 1   # Instant responses — time, date, simple lookups
TIER_LOCAL = 2    # Local device command parsing — no model needed
TIER_CLAUDE = 3   # Complex/conversational — requires Claude API


# ─── Option A: Normalization ───────────────────────────────────
FILLER_PREFIXES = [
    # Longest first so "can you please" matches before "can you"
    "hey jarvis", "jarvis,", "jarvis",
    "can you please", "could you please", "would you please",
    "can you", "could you", "would you",
    "i want you to", "i'd like you to", "i need you to",
    "i want to", "i'd like to",
    "please", "go ahead and",
]

def normalize(text):
    """Lowercase, strip one filler prefix, collapse whitespace."""
    s = text.lower().strip()
    for prefix in sorted(FILLER_PREFIXES, key=len, reverse=True):
        if s.startswith(prefix):
            s = s[len(prefix):].strip().lstrip(",").strip()
            break
    return s


# ─── Device Knowledge Base ──────────────────────────────────────
# Maps natural language names to entity IDs.
# Extend this as you add devices.
DEVICE_ALIASES = {
    # Bedroom main light
    "bedroom light": "light.bedroom",
    "bedroom lights": "light.bedroom",
    "main light": "light.bedroom",
    "main lights": "light.bedroom",
    "the light": "light.bedroom",
    "the lights": "light.bedroom",
    "lights": "light.bedroom",
    "light": "light.bedroom",
    # Bedroom lamp
    "lamp": "light.bedroom_lamp",
    "bedroom lamp": "light.bedroom_lamp",
    "the lamp": "light.bedroom_lamp",
    "desk lamp": "light.bedroom_lamp",
    # Fan
    "fan": "switch.bedroom_fan",
    "bedroom fan": "switch.bedroom_fan",
    "the fan": "switch.bedroom_fan",
    # Blinds
    "blinds": "cover.bedroom_blinds",
    "bedroom blinds": "cover.bedroom_blinds",
    "the blinds": "cover.bedroom_blinds",
    "shades": "cover.bedroom_blinds",
    "curtains": "cover.bedroom_blinds",
}

# Maps action keywords to Home Assistant services
ACTION_MAP = {
    # Turn on
    "turn on": "turn_on",
    "switch on": "turn_on",
    "enable": "turn_on",
    "activate": "turn_on",
    "power on": "turn_on",
    # Turn off
    "turn off": "turn_off",
    "switch off": "turn_off",
    "disable": "turn_off",
    "deactivate": "turn_off",
    "power off": "turn_off",
    "kill": "turn_off",
    "shut off": "turn_off",
    # Open
    "open": "open_cover",
    "raise": "open_cover",
    # Close
    "close": "close_cover",
    "lower": "close_cover",
    "shut": "close_cover",
    # Dim / brightness
    "dim": "turn_on",
    "brighten": "turn_on",
    "set": "turn_on",
}

# Words that signal "everything" / all devices
EVERYTHING_WORDS = ["everything", "all", "all the devices", "all devices", "every device", "whole room"]


# ─── Option D: Semantic Embeddings ─────────────────────────────
# Representative phrases per intent. Device extraction still uses
# DEVICE_ALIASES after the intent/action is resolved.
# "status" is a special label that routes to _build_status_response.

CANONICAL_INTENTS = [
    # turn_on
    ("turn on the light", "turn_on"),
    ("switch on the fan", "turn_on"),
    ("power up the device", "turn_on"),
    ("activate the bedroom light", "turn_on"),
    ("enable the lamp", "turn_on"),
    ("put the lights on", "turn_on"),
    # turn_off
    ("turn off the light", "turn_off"),
    ("switch off the fan", "turn_off"),
    ("shut off the lamp", "turn_off"),
    ("kill the lights", "turn_off"),
    ("disable the device", "turn_off"),
    ("cut the lights", "turn_off"),
    ("power down the fan", "turn_off"),
    # dim / brightness (action is still turn_on with brightness data)
    ("dim the lights to fifty percent", "turn_on"),
    ("brighten the lamp", "turn_on"),
    ("set the light to thirty percent", "turn_on"),
    ("lower the brightness of the lamp", "turn_on"),
    # open_cover
    ("open the blinds", "open_cover"),
    ("raise the shades", "open_cover"),
    ("let some light in through the blinds", "open_cover"),
    # close_cover
    ("close the blinds", "close_cover"),
    ("lower the curtains", "close_cover"),
    ("shut the shades", "close_cover"),
    # status
    ("what is the current status of my devices", "status"),
    ("which lights are on right now", "status"),
    ("is the fan running", "status"),
    ("tell me what's currently on", "status"),
    ("show device status", "status"),
    ("are the blinds open", "status"),
]

CANONICAL_TIER1 = [
    ("what time is it", "time_query"),
    ("tell me the current time", "time_query"),
    ("what is the time right now", "time_query"),
    ("what hour is it", "time_query"),
    ("what day is it today", "date_query"),
    ("what is today's date", "date_query"),
    ("tell me the date", "date_query"),
    ("what's the date today", "date_query"),
]

EMBEDDING_THRESHOLD = 0.70  # minimum cosine similarity to trust a match

# Pre-compute embeddings once at import time
if EMBEDDINGS_AVAILABLE:
    _intent_labels     = [label for _, label in CANONICAL_INTENTS]
    _intent_embeddings = _embedder.encode(
        [phrase for phrase, _ in CANONICAL_INTENTS], normalize_embeddings=True
    )
    _tier1_labels      = [label for _, label in CANONICAL_TIER1]
    _tier1_embeddings  = _embedder.encode(
        [phrase for phrase, _ in CANONICAL_TIER1], normalize_embeddings=True
    )
else:
    _intent_labels = _intent_embeddings = None
    _tier1_labels  = _tier1_embeddings  = None


def _embed_classify(text, embeddings, labels):
    """
    Returns the label of the best-matching canonical phrase,
    or None if best cosine similarity is below EMBEDDING_THRESHOLD.
    """
    if not EMBEDDINGS_AVAILABLE or embeddings is None:
        return None
    query_vec = _embedder.encode([text], normalize_embeddings=True)
    scores = (query_vec @ embeddings.T)[0]
    best_idx = int(np.argmax(scores))
    if scores[best_idx] >= EMBEDDING_THRESHOLD:
        return labels[best_idx]
    return None


# ─── Tier 1: Direct Responses ──────────────────────────────────
def check_tier1(text):
    """
    Instant responses that don't need any intelligence.
    Returns a response dict or None if not Tier 1.
    """
    lower = normalize(text)

    # Fast keyword path
    if any(p in lower for p in ["what time", "current time", "time is it", "what's the time"]):
        return _make_time_response()
    if any(p in lower for p in ["what day", "what's the date", "today's date", "what date"]):
        return _make_date_response()

    # Embedding fallback — catches "tell me the time", "what hour is it", etc.
    label = _embed_classify(lower, _tier1_embeddings, _tier1_labels)
    if label == "time_query":
        return _make_time_response()
    if label == "date_query":
        return _make_date_response()

    return None


def _make_time_response():
    time_str = datetime.now().strftime("%I:%M %p").lstrip("0")
    return {
        "intent": "time_query", "tier": TIER_DIRECT,
        "confidence": 1.0, "response": f"It's {time_str}.", "actions": [],
    }

def _make_date_response():
    date_str = datetime.now().strftime("%A, %B %d")
    return {
        "intent": "date_query", "tier": TIER_DIRECT,
        "confidence": 1.0, "response": f"It's {date_str}.", "actions": [],
    }


# ─── Tier 2: Device Command Parsing ────────────────────────────
def check_tier2(text, device_states):
    """
    Parses device commands using a layered pipeline:
      A. Normalize text
      B. Keyword match (fast, exact)
      C. spaCy dependency parse (handles novel phrasing / polite prefixes)
      D. Embedding similarity (semantic fallback)
    Returns a result dict or None if not a device command.
    """
    lower = normalize(text)  # A

    # ── Status queries ──────────────────────
    status_patterns = [
        r"what.?s the status", r"what.?s on", r"what.?s off",
        r"status of", r"is the .+ on", r"is the .+ off",
        r"are the .+ on", r"are the .+ off",
        r"device status", r"what.?s everything", r"status\??$",
    ]
    if any(re.search(p, lower) for p in status_patterns):
        return _build_status_response(lower, device_states)

    # ── "Turn everything off/on" ────────────
    if any(re.search(r'\b' + re.escape(word) + r'\b', lower) for word in EVERYTHING_WORDS):
        action = _extract_action(lower)
        if action:
            return _build_everything_response(action, device_states)

    # ── Standard device commands ────────────
    # B: keyword extraction on normalized text
    action = _extract_action(lower)

    # C: spaCy if keyword failed
    if action is None:
        action = _spacy_extract_action(lower)

    # D: embeddings if both failed
    if action is None:
        label = _embed_classify(lower, _intent_embeddings, _intent_labels)
        if label == "status":
            return _build_status_response(lower, device_states)
        action = label  # turn_on / turn_off / open_cover / close_cover / None

    if action is None:
        return None

    device_id = _extract_device(lower)
    if device_id is None:
        return None

    brightness = _extract_brightness(lower)
    data = {}
    entity_type = device_id.split(".")[0]

    # Validate action is compatible with entity type before building service
    if entity_type in ("light", "switch") and action not in ("turn_on", "turn_off"):
        return None
    if entity_type == "cover" and action not in ("open_cover", "close_cover"):
        return None

    if entity_type == "light":
        service = f"light.{action}"
        if brightness is not None:
            data["brightness"] = brightness
            service = "light.turn_on"
    elif entity_type == "switch":
        service = f"switch.{action}"
    elif entity_type == "cover":
        service = f"cover.{action}"
    else:
        return None

    current = device_states.get(device_id, {})
    current_state = current.get("state", "unknown")
    friendly_name = current.get("friendly_name", device_id)

    if _check_already_done(action, current_state, entity_type):
        return {
            "intent": "device_control", "tier": TIER_LOCAL, "confidence": 0.9,
            "response": f"The {friendly_name.lower()} is already {current_state}.",
            "actions": [],
        }

    return {
        "intent": "device_control", "tier": TIER_LOCAL, "confidence": 0.9,
        "response": _generate_response(action, friendly_name, brightness),
        "actions": [{"service": service, "entity_id": device_id, "data": data}],
    }


# ─── Option C: spaCy Action Extraction ─────────────────────────
def _spacy_extract_action(text):
    """
    Parse the dependency tree to find the root verb + optional particle.

    Handles constructs keyword matching can't:
      - Phrasal verbs: "turn on/off" -> root=turn, particle=on/off
      - Lemmatization: "dims" -> "dim", "opened" -> "open"
      - Arbitrary prefix: "the fan should be switched off" -> root=switched

    Returns an action string from ACTION_MAP, or None.
    """
    if not SPACY_AVAILABLE:
        return None

    doc = _nlp(text)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return None

    # Verb + particle: "turn on", "switch off", "shut off"
    for child in root.children:
        if child.dep_ == "prt":
            phrase = f"{root.lemma_} {child.text.lower()}"
            action = ACTION_MAP.get(phrase)
            if action:
                return action

    # Verb lemma alone: "dim", "open", "close", "kill", "enable"
    return ACTION_MAP.get(root.lemma_) or ACTION_MAP.get(root.text.lower())


# ─── Extraction Helpers ────────────────────────────────────────

def _extract_action(text):
    """Keyword word-boundary match, longest phrase first."""
    for phrase in sorted(ACTION_MAP.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(phrase) + r'\b', text):
            return ACTION_MAP[phrase]
    return None


def _extract_device(text):
    """Alias word-boundary match, longest alias first."""
    for alias in sorted(DEVICE_ALIASES.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(alias) + r'\b', text):
            return DEVICE_ALIASES[alias]
    return None


def _extract_brightness(text):
    """Extract brightness percentage from text. Returns 0-255 or None."""
    match = re.search(r'(\d+)\s*%', text)
    if match:
        pct = max(0, min(100, int(match.group(1))))
        return round(pct / 100 * 255)
    if "dim" in text and not re.search(r'\d', text):
        return round(0.3 * 255)  # ~77 — default dim level
    return None


def _check_already_done(action, current_state, entity_type):
    """Return True if the device is already in the requested state."""
    if entity_type == "cover":
        return (action == "open_cover" and current_state == "open") or \
               (action == "close_cover" and current_state == "closed")
    return (action == "turn_on" and current_state == "on") or \
           (action == "turn_off" and current_state == "off")


def _generate_response(action, friendly_name, brightness=None):
    """Generate a natural spoken response for a device action."""
    name = friendly_name.lower()
    if brightness is not None:
        return f"Setting the {name} to {round(brightness / 255 * 100)}%."
    return {
        "turn_on":     f"Done, {name} is on.",
        "turn_off":    f"{name} is off.",
        "open_cover":  f"Opening the {name}.",
        "close_cover": f"Closing the {name}.",
    }.get(action, "Done.")


def _build_status_response(text, device_states):
    """Build a response for device status queries."""
    device_id = _extract_device(text)

    if device_id and device_id in device_states:
        info = device_states[device_id]
        name = info["friendly_name"].lower()
        state = info["state"]
        extra = ""
        if "brightness" in info and state == "on":
            extra = f" at {round(info['brightness'] / 255 * 100)}% brightness"
        return {
            "intent": "device_status", "tier": TIER_LOCAL, "confidence": 0.95,
            "response": f"The {name} is {state}{extra}.",
            "actions": [],
        }

    lines = []
    for eid, info in device_states.items():
        name = info["friendly_name"]
        state = info["state"]
        extra = ""
        if "brightness" in info and state == "on":
            extra = f" at {round(info['brightness'] / 255 * 100)}%"
        lines.append(f"{name}: {state}{extra}")

    return {
        "intent": "device_status", "tier": TIER_LOCAL, "confidence": 0.95,
        "response": ". ".join(lines) + ".",
        "actions": [],
    }


def _build_everything_response(action, device_states):
    """Handle 'turn everything off' / 'turn on everything' commands."""
    actions = []
    for entity_id, info in device_states.items():
        entity_type = entity_id.split(".")[0]
        current_state = info["state"]
        if action == "turn_on" and current_state in ("off", "closed"):
            service = {"cover": "cover.open_cover", "switch": "switch.turn_on"}.get(entity_type, "light.turn_on")
            actions.append({"service": service, "entity_id": entity_id, "data": {}})
        elif action == "turn_off" and current_state in ("on", "open"):
            service = {"cover": "cover.close_cover", "switch": "switch.turn_off"}.get(entity_type, "light.turn_off")
            actions.append({"service": service, "entity_id": entity_id, "data": {}})
        elif action == "open_cover" and entity_type == "cover" and current_state == "closed":
            actions.append({"service": "cover.open_cover", "entity_id": entity_id, "data": {}})
        elif action == "close_cover" and entity_type == "cover" and current_state == "open":
            actions.append({"service": "cover.close_cover", "entity_id": entity_id, "data": {}})

    if not actions:
        state_word = "off" if action == "turn_off" else "on"
        return {
            "intent": "device_control", "tier": TIER_LOCAL, "confidence": 0.9,
            "response": f"Everything is already {state_word}.",
            "actions": [],
        }

    verb = "off" if action == "turn_off" else "on"
    return {
        "intent": "device_control", "tier": TIER_LOCAL, "confidence": 0.9,
        "response": f"Turning {verb} everything.",
        "actions": actions,
    }


# ─── Main Classification Entry Point ──────────────────────────

def classify(text, device_states):
    """
    Classify a user message and return the intent + tier.

    Returns a dict with:
      - intent: str (time_query, date_query, device_control, device_status, conversation)
      - tier: int (1, 2, or 3)
      - confidence: float (0.0-1.0)
      - response: str (if tier 1 or 2, the response to send)
      - actions: list (if tier 2, HA actions to execute)
    """
    result = check_tier1(text)
    if result:
        return result

    result = check_tier2(text, device_states)
    if result:
        return result

    return {
        "intent": "conversation", "tier": TIER_CLAUDE,
        "confidence": 0.0, "response": None, "actions": [],
    }


# ─── Test Suite ────────────────────────────────────────────────
# Run this file directly to test the classifier:
#   python intent_classifier.py

if __name__ == "__main__":
    test_states = {
        "light.bedroom":        {"state": "off",    "friendly_name": "Bedroom main light", "brightness": 0},
        "light.bedroom_lamp":   {"state": "on",     "friendly_name": "Bedroom lamp",       "brightness": 128},
        "switch.bedroom_fan":   {"state": "off",    "friendly_name": "Bedroom fan"},
        "cover.bedroom_blinds": {"state": "closed", "friendly_name": "Bedroom blinds"},
    }

    test_commands = [
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

        # ── Tier 3 — should go to Claude ────────────────────
        ("What should I eat for dinner?",   TIER_CLAUDE),
        ("Tell me a joke",                  TIER_CLAUDE),
        ("How productive was I today?",     TIER_CLAUDE),
        ("What's the meaning of life?",     TIER_CLAUDE),
        ("Set the mood for a movie",        TIER_CLAUDE),

        # ── Edge cases ─────────────────────────────────────────
        ("Make things dark", TIER_CLAUDE),  # "dark" isn't a recognized action or device 
        ("Turn the heater on", TIER_CLAUDE),  
        ("Set an alarm", TIER_CLAUDE),  
        ("Play Spotify", TIER_CLAUDE),  
        ("Call Dad", TIER_CLAUDE),  
    ]

    print("=" * 60)
    print("  JARVIS Intent Classifier — Test Suite")
    print()
    print(f"  spaCy:      {'available' if SPACY_AVAILABLE      else 'NOT INSTALLED'}")
    print(f"  Embeddings: {'available' if EMBEDDINGS_AVAILABLE else 'NOT INSTALLED'}")
    print("=" * 60)
    print()

    passed = failed = 0
    for command, expected_tier in test_commands:
        result = classify(command, test_states)
        actual_tier = result["tier"]
        ok = actual_tier == expected_tier
        passed += ok
        failed += not ok
        status = "OK  " if ok else "FAIL"
        response = result["response"] or "(--> Claude)"
        print(f"  [{status}] [Tier {actual_tier}] \"{command}\"")
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