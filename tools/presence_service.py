"""
JARVIS Presence Service - real-time face presence over MQTT
===========================================================
Reads a camera, decides who is in front of it, and publishes presence
transitions onto the MQTT bus. This is the live consumer of the Phase 2 face
pipeline: the detector, the encoder, and both operating points are taken from
the artifacts that were benchmarked and locked, never retyped here.

WHERE EVERY NUMBER COMES FROM. Nothing about the pipeline is hardcoded in this
file:
    detection confidence  <- pipeline_config.DETECTION_CONFIDENCE_THRESHOLD
    detection min box     <- pipeline_config.DETECTION_THRESHOLD_PROVENANCE
    detector + weights    <- pipeline_config.DETECTION_DETECTOR / _WEIGHTS
    recognition threshold <- data/gallery.npz metadata["threshold"]
    encoder + aggregation <- data/gallery.npz metadata
The recognition threshold lives inside the gallery because a cosine threshold
is only meaningful beside the encoder that produced the vectors it scores
against; the gallery therefore also decides which encoder gets loaded, so the
two can never drift apart.

FRAME HANDLING. A capture thread pulls frames at the camera's native rate into
a SINGLE-SLOT buffer that overwrites. The worker takes whatever is sitting in
the slot when it becomes free. Frames that arrive while the worker is busy are
dropped, not queued.

That is deliberate and it is not the same as "process every Nth frame". We do
not know the processing time - it varies with how many faces are in shot, and
it drifts with load - so any fixed N is wrong somewhere. Worse, a queue would
make the service fall progressively further behind real time while looking
healthy: a presence answer computed from a frame twelve seconds old is not a
late answer, it is a wrong one. Dropping keeps latency bounded by one frame
regardless of how slow the pipeline gets, and the drop count is measured
(--stats) rather than hidden.

STATE MACHINE. Per identity, independently:
    absent  -> present : N hits within the last M processed frames, counting
                         only observations younger than max_observation_age
    present -> absent  : T seconds elapsed with no hit
N/M/T are PresenceConfig (3, 5, 10.0). Hysteresis in both directions is the
point - a single missed frame must not evict someone who is standing there,
and a single false match must not announce someone who is not. Only
transitions are published; a person who stays put produces no traffic.

The age ceiling exists because a frame count is not a duration. M=5 spans
about 0.7s at the ~7 fps this pipeline measures at, but nothing pins it there:
if throughput drops the same five frames can straddle a minute, and the
arrival rule quietly degrades into "3 hits some time recently". Bounding the
window in seconds as well as in frames keeps the rule meaning what it reads
like at any throughput. Departure is already in wall time, so it needs no
equivalent.

MQTT TOPICS
    jarvis/presence/<name>          retained, JSON {state, name, confidence,
                                    timestamp, last_seen}
    jarvis/status/presence_service  retained, "online" on connect, "offline"
                                    as the LAST WILL and on clean shutdown

WHY THE STATUS TOPIC IS ON A DIFFERENT BRANCH. It would be natural to park it
at jarvis/presence/_service, and that would be a bug. Wildcards do not honour
naming conventions: a subscriber on jarvis/presence/+ receives every child of
that level, and the broker has no notion of a leading underscore meaning
"internal". The service's own liveness would arrive at every presence consumer
looking exactly like a seventh person called "_service". Hanging it off
jarvis/status/ instead keeps jarvis/presence/+ meaning precisely "all people",
which is what makes that wildcard safe to subscribe to.

The status topic is not decoration. Without it "absent" is ambiguous: it means
either "nobody is there" or "this service died and its retained state is a
fossil". The will covers the crash case (the broker publishes it when our
socket drops); a clean shutdown SUPPRESSES the will, so we publish "offline"
ourselves before disconnecting.

Usage:
    python tools/presence_service.py --stats
    python tools/presence_service.py --dry-run --source clip.mp4
    python tools/presence_service.py --mqtt-host 127.0.0.1 --duration 60
    python tools/presence_service.py --preview --dry-run     # watch it work

--preview opens a window with the detection boxes, the matched identity and
score (or "unknown"), the current presence state and the drop counter. It is a
debugging aid and it is OFF by default: when off, no annotations are collected
and nothing is drawn, so the headless path pays nothing for it. When ON it
costs worker time, which delays the next frame - so it inflates slot_wait and
the drop count as well as adding its own row. --stats says so loudly rather
than letting preview numbers pass as headless ones.

Requires a broker (see README, "MQTT hello-world"). --dry-run needs no broker
and does not import paho.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import face_utils  # noqa: E402  - after sys.path so tools/ is importable standalone
import pipeline_config  # noqa: E402
import recognition_metrics  # noqa: E402
from build_gallery import load_gallery  # noqa: E402

# Lives in its own module so the /engineering log can render these numbers
# without importing cv2 through this file. Re-exported here because this is
# where callers expect to find it.
from presence_config import PresenceConfig  # noqa: E402,F401

DEFAULT_GALLERY = REPO_ROOT / "data" / "gallery.npz"
DEFAULT_WEIGHTS = _HERE / "weights" / pipeline_config.DETECTION_WEIGHTS

TOPIC_PREFIX = "jarvis/presence"
# Deliberately NOT derived from TOPIC_PREFIX - it must not sit under the
# presence branch at all, or jarvis/presence/+ would deliver it as a person.
# See "WHY THE STATUS TOPIC IS ON A DIFFERENT BRANCH" above.
SERVICE_TOPIC = "jarvis/status/presence_service"
STATE_PRESENT = "present"
STATE_ABSENT = "absent"

# Crop padding handed to the encoder. 0.35 is what collect_faces.py used to cut
# the gallery crops, and probe geometry must match enrollment geometry - the
# encoder re-detects and aligns inside .embed(), and it needs the same room to
# do it that it had at enrollment time.
CROP_MARGIN = 0.35

# Consecutive failed reads before we call the camera dead and exit. Exiting is
# the correct response: a dead camera must become "offline" on the service
# topic, not silently frozen presence state.
MAX_READ_FAILURES = 30


@dataclass(frozen=True)
class Transition:
    """One published state change."""

    name: str
    state: str
    confidence: float
    timestamp: str
    last_seen: str | None

    def payload(self):
        return {
            "state": self.state,
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp,
            "last_seen": self.last_seen,
        }


def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_TOPIC_UNSAFE = re.compile(r"[^a-z0-9_-]+")


def topic_for(name):
    """Map an identity to its retained state topic.

    MQTT forbids '+' and '#' in published topics and treats '/' as a level
    separator, so a name is slugged rather than interpolated raw.
    """
    slug = _TOPIC_UNSAFE.sub("-", name.strip().lower()).strip("-")
    return f"{TOPIC_PREFIX}/{slug or 'unknown'}"


# ════════════════════════════════════════════════════════════════════
# Frame transport - the single-slot buffer and the capture thread.
# ════════════════════════════════════════════════════════════════════

class LatestFrame:
    """A one-deep buffer that overwrites instead of queueing.

    put() never blocks and never grows. If a frame is already sitting unread,
    it is discarded in favour of the newer one and counted as dropped, so the
    worker always gets the most recent view of the room rather than the oldest
    unprocessed one.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._slot = None
        self._closed = False
        self.captured = 0
        self.dropped = 0

    def put(self, frame, started, finished):
        with self._cond:
            self.captured += 1
            if self._slot is not None:
                # The worker never picked up the previous frame. Overwrite it.
                self.dropped += 1
            self._slot = (frame, started, finished)
            self._cond.notify()

    def get(self, timeout=0.5):
        """Take the pending frame, or None on timeout / after close()."""
        with self._cond:
            if self._slot is None and not self._closed:
                self._cond.wait(timeout)
            item = self._slot
            self._slot = None
            return item

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class CaptureThread(threading.Thread):
    """Pulls frames at the camera's native rate into the slot."""

    def __init__(self, source, slot, stop_event):
        super().__init__(name="capture", daemon=True)
        self._source = source
        self._slot = slot
        self._stop_event = stop_event  # NOT self._stop: that name is threading.Thread._stop()
        self.failed = False
        self.error = None

    def run(self):
        capture = cv2.VideoCapture(self._source)
        # Ask the driver to keep its own buffer shallow. Best effort - not every
        # backend honours it - but where it works it removes a second queue
        # sitting upstream of ours, which would reintroduce exactly the latency
        # the single slot exists to prevent.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass

        if not capture.isOpened():
            self.failed = True
            self.error = f"could not open video source {self._source!r}"
            self._slot.close()
            return

        # A camera's read() blocks at the sensor's native rate on its own. A
        # FILE does not - it decodes as fast as the CPU allows, which is not a
        # simulation of a camera but a different system entirely: the worker
        # would see one enormous burst, drop almost all of it, and the wall-clock
        # timeouts in the state machine would never line up with the footage. So
        # file sources are paced to the rate they were recorded at.
        interval = None
        if not isinstance(self._source, int):
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0 and fps == fps:  # fps == fps rejects NaN
                interval = 1.0 / fps

        failures = 0
        next_due = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                if interval is not None:
                    pause = next_due - time.perf_counter()
                    if pause > 0:
                        time.sleep(pause)
                    next_due = max(next_due + interval, time.perf_counter())

                started = time.perf_counter()
                ok, frame = capture.read()
                finished = time.perf_counter()
                if not ok or frame is None:
                    if interval is not None:
                        # A file that stops returning frames has simply ended.
                        # That is a clean stop, not a camera fault - only a live
                        # device gets the retry-then-declare-dead treatment.
                        break
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        self.failed = True
                        self.error = (
                            f"{failures} consecutive read failures from "
                            f"{self._source!r} - treating the camera as down"
                        )
                        break
                    continue
                failures = 0
                self._slot.put(frame, started, finished)
        finally:
            capture.release()
            self._slot.close()


# ════════════════════════════════════════════════════════════════════
# Instrumentation.
# ════════════════════════════════════════════════════════════════════

class Stats:
    """Counters and per-stage timings.

    This service is the only place real end-to-end latency numbers come from,
    so the measurement is a feature, not debug scaffolding.
    """

    STAGES = ("capture_decode", "slot_wait", "detect", "crop_align", "embed", "match",
              "preview_draw")

    # Which population each stage samples. This is printed next to every row
    # because the two are not addable: a per-frame stage is measured once per
    # processed frame, a per-face stage once per face found, and a frame with
    # no face in it contributes to the first and not the second. Summing the
    # means across units produces a number that can exceed the end-to-end mean,
    # which is what makes an unlabelled table actively misleading rather than
    # merely incomplete.
    STAGE_UNITS = {
        "capture_decode": "per frame",
        "slot_wait": "per frame",
        "detect": "per frame",
        "crop_align": "per face",
        "embed": "per face",
        "match": "per face",
        "preview_draw": "per frame",
    }

    def __init__(self, preview=False):
        self.frames_processed = 0
        self.faces_detected = 0
        self.faces_matched = 0
        self.faces_unknown = 0
        # Recorded so the report can say out loud that these are preview
        # numbers. Drawing does not just add its own row - it occupies the
        # worker, so it inflates slot_wait and the drop count too.
        self.preview = preview
        self._timings = {stage: [] for stage in self.STAGES}
        self._end_to_end = []

    def record(self, stage, seconds):
        self._timings[stage].append(seconds * 1000.0)

    def record_end_to_end(self, seconds):
        self._end_to_end.append(seconds * 1000.0)

    @staticmethod
    def _summarize(values):
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "mean_ms": float(array.mean()),
            "p95_ms": float(np.percentile(array, 95)),
        }

    def report(self, slot, elapsed):
        lines = ["", "=== presence service stats ==="]
        lines.append(f"  wall clock            : {elapsed:.1f}s")
        lines.append(f"  frames captured       : {slot.captured}")
        dropped = f"  frames dropped        : {slot.dropped}"
        if slot.captured:
            dropped += f"   ({slot.dropped / slot.captured:.1%} of captured)"
        lines.append(dropped)
        lines.append(f"  frames processed      : {self.frames_processed}")
        if elapsed > 0:
            lines.append(f"  capture rate          : {slot.captured / elapsed:.2f} fps")
            lines.append(f"  processed rate        : {self.frames_processed / elapsed:.2f} fps")
        lines.append(f"  faces detected        : {self.faces_detected}")
        lines.append(f"  faces matched         : {self.faces_matched}")
        lines.append(f"  faces unknown         : {self.faces_unknown}")
        lines.append("")
        width = 58
        lines.append(f"  {'stage':<16} {'unit':<10} {'n':>6} {'mean':>9} {'p95':>9}")
        lines.append("  " + "-" * width)

        for stage in self.STAGES:
            unit = self.STAGE_UNITS[stage]
            summary = self._summarize(self._timings[stage])
            if summary is None:
                lines.append(f"  {stage:<16} {unit:<10} {'-':>6} {'-':>9} {'-':>9}")
                continue
            lines.append(
                f"  {stage:<16} {unit:<10} {summary['n']:>6} "
                f"{summary['mean_ms']:>7.1f}ms {summary['p95_ms']:>7.1f}ms"
            )

        end_to_end = self._summarize(self._end_to_end)
        if end_to_end is not None:
            lines.append("  " + "-" * width)
            lines.append(
                f"  {'END-TO-END':<16} {'per frame':<10} {end_to_end['n']:>6} "
                f"{end_to_end['mean_ms']:>7.1f}ms {end_to_end['p95_ms']:>7.1f}ms"
            )

        if self.preview:
            drawing = self._summarize(self._timings["preview_draw"])
            cost = f"{drawing['mean_ms']:.1f}ms/frame" if drawing else "unmeasured"
            lines.append("")
            lines.append("  *** PREVIEW WAS ON - THESE ARE NOT HEADLESS NUMBERS ***")
            lines.append(f"  Drawing and the window event loop cost {cost}, and that time")
            lines.append("  is spent IN THE WORKER. It is excluded from END-TO-END (which")
            lines.append("  stops at the match), but it is not free: it delays the next")
            lines.append("  frame, so slot_wait, the processed rate and the drop count are")
            lines.append("  all worse here than the same run without --preview. Quote")
            lines.append("  headless numbers when reporting pipeline latency.")

        lines.append("")
        lines.append("  THE ROWS DO NOT SUM. per-frame stages are sampled once per")
        lines.append("  processed frame; per-face stages once per face found, and most")
        lines.append("  frames contain no face. Compare n before comparing means - a")
        lines.append("  per-face mean is an average over a different, smaller population,")
        lines.append("  so adding it to a per-frame mean can exceed END-TO-END.")
        lines.append("")
        lines.append("  END-TO-END is capture start -> match complete, so it includes")
        lines.append("  slot_wait (how long the frame sat waiting for the worker).")
        lines.append("  crop_align times the crop only: alignment and resizing live")
        lines.append("  inside the encoder per the face_utils contract, so that cost is")
        lines.append("  counted in embed.")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# Recognition and the per-identity state machine.
# ════════════════════════════════════════════════════════════════════

class GalleryMatcher:
    """Scores an embedding against every enrolled identity.

    Uses the rule the gallery documents in its own metadata: cosine of
    L2-normalized embeddings, accept if the MAX over that person's vectors
    clears the threshold. For a one-vector-per-person aggregation the max over
    one row is just that row's cosine, so this is correct for every aggregation
    the gallery builder can produce.
    """

    def __init__(self, vectors, person_ids, threshold):
        self.threshold = float(threshold)
        self.names = sorted(set(person_ids))
        self._by_person = {
            name: vectors[[i for i, p in enumerate(person_ids) if p == name]]
            for name in self.names
        }

    def match(self, embedding):
        """Returns (name | None, best_score). None means "below threshold"."""
        probe = recognition_metrics.l2_normalize(np.asarray(embedding)).reshape(1, -1)
        best_name, best_score = None, -1.0
        for name, person_vectors in self._by_person.items():
            score = float(recognition_metrics.score_multiref(probe, person_vectors)[0])
            if score > best_score:
                best_name, best_score = name, score
        if best_score >= self.threshold:
            return best_name, best_score
        return None, best_score


class PresenceTracker:
    """The per-identity state machine. Returns only transitions."""

    def __init__(self, names, config):
        self.config = config
        self._window = {n: collections.deque(maxlen=config.window_frames) for n in names}
        self._state = {n: STATE_ABSENT for n in names}
        self._last_hit_mono = {n: None for n in names}
        self._last_seen_iso = {n: None for n in names}
        self._last_confidence = {n: 0.0 for n in names}

    def state(self, name):
        return self._state[name]

    def names(self):
        """Every identity being tracked, in gallery order."""
        return list(self._window)

    def _fresh_hits(self, name, now_mono):
        """Hits in the window that are still recent enough to count.

        The frame window (M) and the age ceiling are both upper bounds, and an
        observation has to satisfy both: it must be one of the last M, AND no
        older than max_observation_age. Stale entries are left in the deque
        rather than pruned - they age out of relevance here and get pushed out
        by maxlen in the normal way, which keeps "the last M frames" honest
        instead of quietly extending the window backwards as entries expire.
        """
        max_age = self.config.max_observation_age
        return sum(
            1 for observed_at, hit in self._window[name]
            if hit and (now_mono - observed_at) <= max_age
        )

    def update(self, hits, now_mono=None, now_iso=None):
        """Advance one processed frame.

        Inputs:
            hits (dict[str, float]): identity -> best confidence THIS frame.
                Absent keys are treated as a miss for that identity.
            now_mono (float | None): monotonic clock, injectable for tests.
            now_iso (str | None): wall-clock stamp, injectable for tests.
        Returns:
            list[Transition]: empty when nothing changed.
        """
        now_mono = time.monotonic() if now_mono is None else now_mono
        now_iso = utc_now_iso() if now_iso is None else now_iso
        transitions = []

        for name in self._window:
            hit = name in hits
            # Each observation carries the time it was made, so the window can
            # be bounded in wall time as well as in frames.
            self._window[name].append((now_mono, hit))
            if hit:
                self._last_hit_mono[name] = now_mono
                self._last_seen_iso[name] = now_iso
                self._last_confidence[name] = hits[name]

            if self._state[name] == STATE_ABSENT:
                if self._fresh_hits(name, now_mono) >= self.config.hits_required:
                    self._state[name] = STATE_PRESENT
                    transitions.append(self._transition(name, STATE_PRESENT, now_iso))
            else:
                last = self._last_hit_mono[name]
                if last is not None and (now_mono - last) >= self.config.absent_after:
                    self._state[name] = STATE_ABSENT
                    # Clear the window on the way out. Without this, hits still
                    # sitting in it from before the timeout could immediately
                    # re-trigger presence - possible whenever the processed
                    # frame rate is low enough that M frames span more than T
                    # seconds.
                    self._window[name].clear()
                    transitions.append(self._transition(name, STATE_ABSENT, now_iso))

        return transitions

    def _transition(self, name, state, now_iso):
        return Transition(
            name=name,
            state=state,
            # For 'absent' this is the confidence at the last sighting, which is
            # the only confidence that ever meant anything for this person.
            confidence=self._last_confidence[name],
            timestamp=now_iso,
            last_seen=self._last_seen_iso[name],
        )


# ════════════════════════════════════════════════════════════════════
# MQTT.
# ════════════════════════════════════════════════════════════════════

class PresencePublisher:
    """Retained presence topics plus the service liveness topic.

    paho is imported lazily so --dry-run runs the full pipeline on a machine
    that has neither a broker nor the client library.
    """

    def __init__(self, host, port, dry_run=False):
        self.host = host
        self.port = port
        self.dry_run = dry_run
        self._client = None

    def connect(self):
        if self.dry_run:
            print("[mqtt] dry run - nothing will be published")
            return
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"jarvis-presence-{int(time.time())}",
            protocol=mqtt.MQTTv311,
        )
        # Registered before connect: it rides inside the CONNECT packet, and the
        # broker publishes it if our socket ever drops without a DISCONNECT.
        # Retained, so a subscriber arriving after the crash still learns the
        # service is gone.
        self._client.will_set(SERVICE_TOPIC, "offline", qos=1, retain=True)
        self._client.connect(self.host, self.port, keepalive=30)
        self._client.loop_start()
        info = self._client.publish(SERVICE_TOPIC, "online", qos=1, retain=True)
        info.wait_for_publish(timeout=5)
        print(f"[mqtt] connected to {self.host}:{self.port} - {SERVICE_TOPIC} = online")

    def publish(self, transition):
        body = json.dumps(transition.payload())
        topic = topic_for(transition.name)
        if self.dry_run:
            print(f"[dry-run] would publish (retained) {topic}  {body}")
            return
        self._client.publish(topic, body, qos=1, retain=True).wait_for_publish(timeout=5)
        print(f"[mqtt] {topic}  {body}")

    def close(self):
        if self.dry_run or self._client is None:
            return
        # A clean DISCONNECT suppresses the will, so say it ourselves. Otherwise
        # the retained "online" would outlive the process and every subscriber
        # would go on trusting stale presence forever.
        info = self._client.publish(SERVICE_TOPIC, "offline", qos=1, retain=True)
        info.wait_for_publish(timeout=5)
        self._client.loop_stop()
        self._client.disconnect()
        print(f"[mqtt] {SERVICE_TOPIC} = offline, disconnected")


# ════════════════════════════════════════════════════════════════════
# Pipeline.
# ════════════════════════════════════════════════════════════════════

def process_frame(frame, detector, encoder, matcher, min_box_size, stats, annotate=False):
    """frame -> detect -> crop -> embed -> match.

    Inputs:
        annotate (bool): also return per-face results for the preview window.
            Off by default, and when off nothing is collected at all - the
            headless path must not pay for a feature it is not using.
    Returns:
        tuple (hits, annotations):
            hits (dict[str, float]): identity -> best confidence this frame.
            annotations (list[tuple] | None): (box, name_or_None, score) per
                face, or None when annotate is False.
    """
    started = time.perf_counter()
    detections = detector.detect(frame)
    stats.record("detect", time.perf_counter() - started)

    # The same filter the detection threshold was measured under: boxes whose
    # width OR height is under min_box_size were excluded from the benchmark, so
    # 0.57 is only calibrated for that filtered task. A face that small also
    # carries too few pixels for the encoder to do anything useful with.
    kept = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue
        kept.append(detection)

    stats.faces_detected += len(kept)
    hits = {}
    annotations = [] if annotate else None
    for detection in kept:
        started = time.perf_counter()
        crop = face_utils.crop_face(frame, detection["box"], margin=CROP_MARGIN)
        stats.record("crop_align", time.perf_counter() - started)
        if crop.size == 0:
            continue

        started = time.perf_counter()
        try:
            embedding = encoder.embed(crop)
        except Exception as exc:  # one bad crop must not take the service down
            print(f"[warn] embed failed on a {crop.shape[1]}x{crop.shape[0]} crop: {exc}")
            continue
        stats.record("embed", time.perf_counter() - started)
        if embedding is None:
            continue

        started = time.perf_counter()
        name, score = matcher.match(embedding)
        stats.record("match", time.perf_counter() - started)

        if annotations is not None:
            # score is the best cosine found either way, so an unknown face
            # still carries the number it was rejected on - which is the most
            # useful thing the preview can show about a miss.
            annotations.append((detection["box"], name, score))

        if name is None:
            stats.faces_unknown += 1
            continue
        stats.faces_matched += 1
        # Two crops of the same person in one frame: keep the better score.
        if score > hits.get(name, -1.0):
            hits[name] = score

    return hits, annotations


# ════════════════════════════════════════════════════════════════════
# Preview window (--preview). Entirely optional and entirely off the
# headless path.
# ════════════════════════════════════════════════════════════════════

PREVIEW_WINDOW = "JARVIS presence"
_MATCHED_COLOR = (0, 200, 0)      # BGR green
_UNKNOWN_COLOR = (0, 165, 255)    # BGR amber
_HUD_TEXT = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(frame, text, x, y, color, top_margin=0):
    """A filled label box with text, clamped to stay inside the frame.

    top_margin reserves the HUD band at the top. A face detected high in the
    frame would otherwise have its label drawn underneath the HUD and painted
    over - the identity, which is the whole point of the window, would be the
    one thing invisible.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, 0.5, 1)
    box_h = text_h + baseline + 2
    top = y - box_h                        # preferred: just above the box
    if top < top_margin:                   # would collide with the HUD or the
        top = max(y + 2, top_margin)       # frame edge, so drop it inside instead
    top = min(top, frame.shape[0] - box_h)
    x = max(0, min(x, frame.shape[1] - text_w - 2))
    cv2.rectangle(frame, (x, top), (x + text_w + 2, top + box_h), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 1, top + text_h + 1), _FONT, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)


def draw_preview(frame, annotations, tracker, slot, processed):
    """Draw boxes, identities and a status HUD onto `frame`, IN PLACE.

    In place on purpose: the frame is dead after this point in the loop, and
    copying a 960x540 BGR array per frame would be a real cost charged to a
    debugging aid.
    """
    present = sorted(n for n in tracker.names() if tracker.state(n) == STATE_PRESENT)
    dropped = f"{slot.dropped}"
    if slot.captured:
        dropped += f" ({slot.dropped / slot.captured:.0%})"
    hud = [
        f"present: {', '.join(present) if present else 'nobody'}",
        f"captured {slot.captured}   dropped {dropped}   processed {processed}",
    ]

    # HUD first, boxes second. The boxes are the primary content, so they draw
    # over the HUD band rather than under it, and _label is told to keep every
    # label clear of that band.
    hud_height = 8 + 22 * len(hud)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], hud_height), (0, 0, 0), cv2.FILLED)
    for i, line in enumerate(hud):
        cv2.putText(frame, line, (8, 22 + 22 * i), _FONT, 0.55, _HUD_TEXT, 1, cv2.LINE_AA)

    for box, name, score in annotations:
        x1, y1, x2, y2 = (int(v) for v in box)
        matched = name is not None
        color = _MATCHED_COLOR if matched else _UNKNOWN_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _label(frame, f"{name if matched else 'unknown'} {score:.2f}", x1, y1, color,
               top_margin=hud_height)


def open_preview_window():
    """Create the window, or report that this OpenCV build cannot.

    Headless OpenCV builds raise on namedWindow. That must degrade to running
    without a preview, not take the service down.
    """
    try:
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        return True
    except cv2.error as exc:
        print(f"[warn] --preview unavailable (no GUI support in this OpenCV build): {exc}")
        print("[warn] continuing headless")
        return False


def close_preview_window():
    try:
        cv2.destroyWindow(PREVIEW_WINDOW)
        cv2.waitKey(1)      # let the window manager actually process the close
    except cv2.error:
        pass


def load_pipeline(gallery_path, weights_path):
    """Build detector, encoder and matcher from the locked artifacts."""
    vectors, person_ids, metadata = load_gallery(gallery_path)
    encoder_name = metadata["encoder"]
    threshold = metadata["threshold"]
    min_box_size = pipeline_config.DETECTION_THRESHOLD_PROVENANCE["min_box_size"]

    print("-- pipeline ------------------------------------------")
    print(f"  gallery         : {gallery_path}")
    print(f"  built           : {metadata.get('created_at')}")
    print(f"  encoder         : {encoder_name} ({metadata['dimension']}-d)")
    print(f"  aggregation     : {metadata['aggregation']}")
    print(f"  identities      : {metadata['num_persons']} "
          f"({metadata['num_vectors']} vectors)")
    print(f"  recog threshold : {threshold}    <- gallery metadata")
    print(f"  detector        : {pipeline_config.DETECTION_DETECTOR}")
    print(f"  weights         : {weights_path}")
    print(f"  det threshold   : {pipeline_config.DETECTION_CONFIDENCE_THRESHOLD}"
          f"    <- pipeline_config")
    print(f"  min box size    : {min_box_size}px    <- detection provenance")

    detector = face_utils.load_detector(
        pipeline_config.DETECTION_DETECTOR,
        min_confidence=pipeline_config.DETECTION_CONFIDENCE_THRESHOLD,
        weights=str(weights_path),
    )
    encoder = face_utils.load_encoder(encoder_name)
    matcher = GalleryMatcher(vectors, person_ids, threshold)
    print(f"  enrolled        : {', '.join(matcher.names)}")
    print("------------------------------------------------------")
    return detector, encoder, matcher, min_box_size


def parse_source(value):
    """'0' -> camera index 0; anything else -> a path/URL for VideoCapture."""
    return int(value) if value.isdigit() else value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Real-time face presence over MQTT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default="0",
                        help="camera index or video file/URL (default: 0)")
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--dry-run", action="store_true",
                        help="run the full pipeline and print transitions, publish nothing")
    parser.add_argument("--stats", action="store_true",
                        help="print counters and per-stage mean/p95 timings on exit")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after N seconds (default 0 = until Ctrl-C or end of source)")
    parser.add_argument("--preview", action="store_true",
                        help="open a window showing boxes, identities and presence "
                             "state. Debugging aid: it costs worker time, so it "
                             "skews the --stats numbers (which say so). q or ESC quits.")
    args = parser.parse_args(argv)

    if not args.gallery.is_file():
        parser.error(f"gallery not found: {args.gallery} - build it with build_gallery.py")
    if not args.weights.is_file():
        parser.error(f"detector weights not found: {args.weights}")

    config = PresenceConfig()
    print(f"-- state machine -- N={config.hits_required} hits in the last "
          f"M={config.window_frames} processed frames -> present; "
          f"T={config.absent_after}s with no hit -> absent")

    detector, encoder, matcher, min_box_size = load_pipeline(args.gallery, args.weights)
    tracker = PresenceTracker(matcher.names, config)

    # A headless OpenCV build cannot open a window; that downgrades the preview
    # rather than failing the run, so `preview` is what the loop tests, not the
    # flag the user passed.
    preview = args.preview and open_preview_window()
    if preview:
        print("[preview] window open - press q or ESC to quit")
        print("[preview] NOTE: drawing costs worker time; --stats numbers will "
              "not match a headless run")
    stats = Stats(preview=preview)
    publisher = PresencePublisher(args.mqtt_host, args.mqtt_port, dry_run=args.dry_run)
    publisher.connect()

    slot = LatestFrame()
    stop = threading.Event()

    def request_stop(signum, frame):
        print("\n[signal] stopping...")
        stop.set()
        slot.close()

    signal.signal(signal.SIGINT, request_stop)

    capture = CaptureThread(parse_source(args.source), slot, stop)
    capture.start()

    started = time.perf_counter()
    exit_code = 0
    try:
        while not stop.is_set():
            if args.duration and (time.perf_counter() - started) >= args.duration:
                break
            item = slot.get(timeout=0.5)
            if item is None:
                if not capture.is_alive():
                    break
                continue

            frame, capture_started, capture_finished = item
            worker_started = time.perf_counter()
            stats.record("capture_decode", capture_finished - capture_started)
            stats.record("slot_wait", worker_started - capture_finished)

            hits, annotations = process_frame(
                frame, detector, encoder, matcher, min_box_size, stats,
                annotate=preview,
            )
            stats.frames_processed += 1
            # Recorded BEFORE any drawing: END-TO-END means "how long until we
            # had an answer", and a debugging window must not inflate it.
            stats.record_end_to_end(time.perf_counter() - capture_started)

            for transition in tracker.update(hits):
                publisher.publish(transition)

            if preview:
                drawing_started = time.perf_counter()
                draw_preview(frame, annotations, tracker, slot, stats.frames_processed)
                try:
                    cv2.imshow(PREVIEW_WINDOW, frame)
                    # waitKey is not optional - it is what pumps the window's
                    # event loop. Without it the window never paints.
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    print(f"[warn] preview failed, continuing headless: {exc}")
                    preview = False
                    key = 0xFF
                stats.record("preview_draw", time.perf_counter() - drawing_started)
                if key in (27, ord("q")):
                    print("[preview] quit requested")
                    break
    except KeyboardInterrupt:
        print("\n[interrupt] stopping...")
    finally:
        stop.set()
        slot.close()
        capture.join(timeout=2.0)
        elapsed = time.perf_counter() - started
        if preview:
            close_preview_window()
        if capture.failed:
            print(f"[error] capture: {capture.error}")
            exit_code = 1
        publisher.close()
        if args.stats:
            print(stats.report(slot, elapsed))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
