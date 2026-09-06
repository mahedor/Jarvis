"""
JARVIS Presence — State-Machine Operating Point
===============================================
The debounce constants the presence state machine runs on: how much evidence
declares somebody present, and how long silence takes to declare them absent.

WHY THIS IS ITS OWN MODULE AND NOT PART OF presence_service.py. It was, and it
could not stay there. presence_service.py imports cv2, numpy, face_utils and
build_gallery at module level, so importing it costs the whole vision stack.
The /engineering log renders these values in a caption, and demo/ is documented
as stdlib-only on purpose (see engineering_data.py: "Making the chat UI require
OpenCV so a documentation page can render a table is a bad trade"). Splitting
the values out means the documentation layer can read them without the
assistant growing a dependency on OpenCV.

WHY NOT IN pipeline_config.py, WHICH IS ALREADY DEPENDENCY-FREE. That file's
load-bearing claim is ANTI-DRIFT: "Every number below is pinned to the
benchmark run that produced it, by timestamp", enforced by
tests/test_pipeline_config.py against results/benchmarks_detection.json. The
numbers HERE are pinned to nothing. They are hand-chosen starting points that
have never been measured against a recorded run, and filing them next to values
that carry timestamped provenance would quietly borrow credibility they have
not earned. Different epistemic status, different module.

*** THESE VALUES ARE UNVALIDATED. ***
N, M and T were chosen by reasoning about a camera nobody has pointed at a
room yet. They have been exercised against a video file and never against live
hardware, and no measurement has been taken that would justify any of them over
a neighbouring value. Treat them as placeholders with a rationale, not as an
operating point in the sense pipeline_config.py means it. They are pinned by
tests/test_presence_service.py so that changing one is deliberate - which
guards against drift, not against being wrong.
"""

from dataclasses import dataclass

__all__ = ["PresenceConfig"]


@dataclass(frozen=True)
class PresenceConfig:
    """The state-machine constants. Deliberately not exposed as CLI flags -
    these are a locked operating point like the thresholds, not a knob to tune
    per run."""

    hits_required: int = 3       # N: hits needed to declare presence
    window_frames: int = 5       # M: over this many recent processed frames
    absent_after: float = 10.0   # T: seconds of no hit before declaring absence

    # The arrival window counts FRAMES, but frames are not a unit of time. At
    # the ~7 fps this pipeline actually runs at, M=5 spans about 0.7s; if
    # throughput collapses - a busy CPU, a frame full of faces to embed - the
    # same five frames can span a minute, and "3 hits in the last 5 frames"
    # silently becomes "3 hits at some point recently-ish". An observation
    # older than this many seconds is therefore ignored when counting toward
    # arrival, so the window has a ceiling in wall time as well as in frames.
    #
    # 5.0 is deliberately loose: ~7x the healthy span of the window, so it
    # never interferes with normal operation and only bites once throughput has
    # fallen below about 1 fps, where the frame count has stopped meaning
    # anything. It bounds the failure mode without becoming a second tuning
    # knob competing with M.
    max_observation_age: float = 5.0
