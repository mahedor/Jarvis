"""
JARVIS Stream Viewer — any OpenCV source, live detections on top
================================================================
Groundwork for piping Parrot Anafi drone video into Jarvis. Reads a webcam, a
video file or a network stream, runs a YOLO model on every frame it shows, and
draws the boxes, class names, confidences, the display FPS and the per-frame
inference time.

    python tools/view_stream.py                                   # webcam 0
    python tools/view_stream.py --source clip.mp4 --record out.mp4
    python tools/view_stream.py --source rtsp://192.168.42.1/live # Anafi, on its Wi-Fi
    python tools/view_stream.py --source rtsp://... --rtsp-transport tcp
    python tools/view_stream.py --model yolov8n.pt                # any YOLO weights

q or ESC quits. A standalone tool: nothing in demo/ imports it.

THE DETECTOR IS THE PRESENCE SERVICE'S DETECTOR. It is built with
face_utils.load_detector() - the same factory presence_service.py calls - so
the viewer and the service cannot disagree about how YOLO is run, clamped or
thresholded. --model only swaps the weights file. The confidence threshold
defaults to pipeline_config.DETECTION_CONFIDENCE_THRESHOLD (0.57); note that
value is calibrated for the face weights, not for anything --model loads.

WHY A WATCHDOG AND NOT JUST VideoCapture. An RTSP source that is unreachable,
or reachable but silent, makes cv2.VideoCapture.read() block - the classic
symptom is a black window that never updates and never errors. Capture runs
on its own thread and the display loop waits on it with a deadline, so a
source that produces no frame within --timeout seconds ends the run with a
message saying so. FFmpeg's own open/read timeouts are also set for URL
sources, but the thread-level deadline is what guarantees the behaviour for
every backend.

LIVE VS FILE. A live source (camera, URL) is read into a one-deep slot that
overwrites, so the display always shows the newest frame and inference that
cannot keep up drops frames rather than falling behind. A file is handed over
frame by frame with no drops, so every frame of a clip is shown and recorded.

TWO RECORDINGS. --record saves what the window shows: annotated, and only the
frames detection processed. --record-raw saves what the camera produced, for
building labeling datasets: no overlay, and EVERY captured frame, including the
ones a live source dropped before detection saw them. That is why it is
written on the capture thread rather than in the display loop, and written
before the frame is handed over, so the overlay (drawn in place) can never
reach it. Frames from a live source do not arrive at a steady rate, so the
container's fps is only nominal; the truth is the sidecar
<name>.timestamps.csv, one row per frame with its capture time.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import face_utils  # noqa: E402  - after sys.path so tools/ is importable standalone
import pipeline_config  # noqa: E402

DEFAULT_WEIGHTS = _HERE / "weights" / pipeline_config.DETECTION_WEIGHTS
DEFAULT_TIMEOUT = 10.0
WINDOW = "JARVIS stream"

SOURCE_CAMERA = "camera"
SOURCE_FILE = "file"
SOURCE_URL = "url"

# Frames the FPS readout averages over.
FPS_WINDOW = 30
# Used for --record when the source does not report a usable frame rate.
FALLBACK_RECORD_FPS = 30.0
# How long shutdown waits for the capture thread to close --record-raw.
READER_JOIN_TIMEOUT = 3.0

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_TEXT = (255, 255, 255)
# BGR. Picked per class so that multi-class weights stay readable.
_PALETTE = [(0, 200, 0), (0, 165, 255), (255, 128, 0), (200, 0, 200),
            (0, 220, 220), (255, 0, 0), (0, 0, 255), (128, 255, 128)]


class StreamTimeout(RuntimeError):
    """No frame arrived within the deadline."""


class StreamError(RuntimeError):
    """The source could not be opened or failed while reading."""


# ════════════════════════════════════════════════════════════════════
# Sources.
# ════════════════════════════════════════════════════════════════════

def parse_source(value):
    """Classify a --source string.

    Returns:
        tuple (kind, source): kind is SOURCE_CAMERA, SOURCE_URL or
        SOURCE_FILE; source is what cv2.VideoCapture should be given - an int
        for a camera index, the string unchanged otherwise.
    """
    text = value.strip()
    if text.isdigit():
        return SOURCE_CAMERA, int(text)
    if "://" in text:
        return SOURCE_URL, text
    return SOURCE_FILE, text


def is_rtsp(source):
    return isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://"))


def ffmpeg_capture_options(transport):
    """Value for OPENCV_FFMPEG_CAPTURE_OPTIONS ('key;value' pairs, '|'-separated)."""
    if transport not in ("udp", "tcp"):
        raise ValueError(f"rtsp transport must be udp or tcp, not {transport!r}")
    return f"rtsp_transport;{transport}"


def open_capture(kind, source, rtsp_transport="udp", timeout=DEFAULT_TIMEOUT):
    """Open `source` with cv2.VideoCapture, configured for its kind."""
    if kind != SOURCE_URL:
        return cv2.VideoCapture(source)

    if is_rtsp(source):
        # Read by OpenCV's FFmpeg backend when the capture is opened, so it has
        # to be in the environment before the constructor runs.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_capture_options(rtsp_transport)
    timeout_ms = int(timeout * 1000)
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
    ])


def sidecar_path(video_path):
    """raw.mp4 -> raw.timestamps.csv, next to the video."""
    return Path(video_path).with_suffix(".timestamps.csv")


def _open_writer(path, fps, frame):
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise StreamError(f"could not open {path} for writing")
    return writer


class RawRecorder:
    """Unannotated frames plus a per-frame capture-timestamp sidecar.

    Sidecar columns:
        frame      0-based index into the video
        t_s        seconds since the first captured frame (monotonic clock)
        unix_time  wall-clock capture time
        pos_ms     the source's own position (CAP_PROP_POS_MSEC). For a file
                   this is the media timestamp and the column to trust, since
                   a file is read as fast as it is consumed, not in real time.
    """

    COLUMNS = ("frame", "t_s", "unix_time", "pos_ms")

    def __init__(self, path, fps=FALLBACK_RECORD_FPS):
        self.path = Path(path)
        self.sidecar = sidecar_path(self.path)
        # Container fps only - set by FrameReader from the source once opened.
        self.fps = fps
        self._writer = None
        self._csv_file = None
        self._csv = None
        self._first = None
        self.frames = 0

    def write(self, frame, captured, unix_time, pos_ms):
        if self._writer is None:
            self._writer = _open_writer(self.path, self.fps, frame)
            self._csv_file = open(self.sidecar, "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(self.COLUMNS)
            self._first = captured
            print(f"[record-raw] {self.path} (+ {self.sidecar.name}), "
                  f"nominal {self.fps:g} fps")
        self._writer.write(frame)
        self._csv.writerow((self.frames, f"{captured - self._first:.6f}",
                            f"{unix_time:.6f}", f"{pos_ms:.3f}"))
        self.frames += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._csv_file.close()
            self._writer = None


_END = object()


class FrameReader(threading.Thread):
    """Reads frames on a background thread; next_frame() waits with a deadline.

    open_fn is a zero-argument callable returning a cv2.VideoCapture-like
    object (isOpened / read / get / release). Injected so the timeout path can
    be tested with a source that never produces a frame.

    raw_recorder, if given, receives every frame read, on this thread, before
    the frame is handed to the display loop. The reader closes it on exit.
    """

    def __init__(self, open_fn, live, description="source", raw_recorder=None):
        super().__init__(name="capture", daemon=True)
        self._open = open_fn
        self._live = live
        self._description = description
        self.raw_recorder = raw_recorder
        self._queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()  # NOT self._stop: that is Thread._stop()
        self.error = None
        self.source_fps = 0.0
        self.frames_delivered = 0

    def run(self):
        capture = None
        try:
            capture = self._open()
            if not capture.isOpened():
                self.error = f"could not open {self._description}"
                return
            self.source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if self.raw_recorder is not None:
                self.raw_recorder.fps = _record_fps(self.source_fps)
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if ok and frame is not None:
                    if self.raw_recorder is not None:
                        # Before the hand-over: once the display loop has the
                        # frame it draws on it in place.
                        self.raw_recorder.write(frame, time.perf_counter(), time.time(),
                                        float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
                    self._hand_over(frame)
                elif self._live:
                    # A live source can hiccup. Keep asking; if it stays silent,
                    # next_frame()'s deadline is what ends the run.
                    time.sleep(0.01)
                else:
                    return  # end of file
        except Exception as exc:  # surfaced to the display loop, not swallowed
            self.error = f"{self._description} failed: {exc}"
        finally:
            if capture is not None:
                capture.release()
            if self.raw_recorder is not None:
                self.raw_recorder.close()
            self._put_blocking(_END)

    def _hand_over(self, frame):
        if not self._live:
            self._put_blocking(frame)
            return
        # Live: drop whatever the display loop has not picked up yet. Single
        # producer, so the slot is guaranteed empty for put_nowait.
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(frame)

    def _put_blocking(self, item):
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if item is _END and self._live:
                    # Replace a stale frame with the end marker.
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass

    def next_frame(self, timeout):
        """The next frame, or None at a clean end of stream.

        Raises:
            StreamTimeout: nothing arrived within `timeout` seconds.
            StreamError: the source failed to open or errored while reading.
        """
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            if self.frames_delivered == 0:
                raise StreamTimeout(
                    f"no frame from {self._description} within {timeout:g}s") from None
            raise StreamTimeout(
                f"{self._description} stalled: no new frame for {timeout:g}s "
                f"after {self.frames_delivered} frames") from None
        if item is _END:
            if self.error:
                raise StreamError(self.error)
            if self.frames_delivered == 0:
                raise StreamError(f"{self._description} ended without producing a frame")
            return None
        self.frames_delivered += 1
        return item

    def stop(self):
        self._stop_event.set()


# ════════════════════════════════════════════════════════════════════
# Overlay.
# ════════════════════════════════════════════════════════════════════

def _color_for(label):
    # Deterministic across runs (unlike hash() on str, which is salted).
    return _PALETTE[sum(label.encode()) % len(_PALETTE)]


def _label_box(frame, text, x, y, color, top_margin):
    """Filled label above (x, y), dropped inside the box if it would hit the HUD."""
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, 0.5, 1)
    box_h = text_h + baseline + 2
    top = y - box_h
    if top < top_margin:
        top = max(y + 2, top_margin)
    top = min(top, frame.shape[0] - box_h)
    x = max(0, min(x, frame.shape[1] - text_w - 2))
    cv2.rectangle(frame, (x, top), (x + text_w + 2, top + box_h), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 1, top + text_h + 1), _FONT, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)


def draw_overlay(frame, detections, fps, inference_ms):
    """Draw boxes, '<class> <conf>' labels and an FPS / inference HUD, IN PLACE.

    detections are face_utils detector dicts. "label" is present for YOLO
    weights; anything without one is drawn as "object".
    """
    hud = f"{fps:5.1f} FPS   inference {inference_ms:6.1f} ms   {len(detections)} det"
    hud_height = 30
    cv2.rectangle(frame, (0, 0), (frame.shape[1], hud_height), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, hud, (8, 21), _FONT, 0.55, _HUD_TEXT, 1, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        label = det.get("label", "object")
        color = _color_for(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _label_box(frame, f"{label} {det['confidence']:.2f}", x1, y1, color,
                   top_margin=hud_height)


class FpsMeter:
    """Frames per second over the last `window` ticks."""

    def __init__(self, window=FPS_WINDOW):
        self._stamps = collections.deque(maxlen=window)

    def tick(self, now=None):
        self._stamps.append(time.perf_counter() if now is None else now)

    @property
    def fps(self):
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


# ════════════════════════════════════════════════════════════════════
# Main loop.
# ════════════════════════════════════════════════════════════════════

def _open_window():
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        return True
    except cv2.error:
        return False


def _record_fps(source_fps):
    return source_fps if 0 < source_fps <= 240 else FALLBACK_RECORD_FPS


def run(detector, reader, timeout, record_path=None, display=True):
    """Pull frames, detect, draw, show/record until the stream ends or q/ESC.

    Returns the number of frames shown.
    """
    fps = FpsMeter()
    writer = None
    shown = 0
    try:
        while True:
            frame = reader.next_frame(timeout)
            if frame is None:
                print(f"[end] source finished after {shown} frames")
                break

            started = time.perf_counter()
            detections = detector.detect(frame)
            inference_ms = (time.perf_counter() - started) * 1000.0

            fps.tick()
            draw_overlay(frame, detections, fps.fps, inference_ms)
            shown += 1

            if record_path is not None:
                if writer is None:
                    rate = _record_fps(reader.source_fps)
                    writer = _open_writer(record_path, rate, frame)
                    h, w = frame.shape[:2]
                    print(f"[record] {record_path} at {w}x{h}, {rate:g} fps")
                writer.write(frame)

            if display:
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print(f"[quit] after {shown} frames")
                    break
    finally:
        reader.stop()
        if writer is not None:
            writer.release()
        # The raw recorder is closed by the reader thread, so give it the
        # chance to finish - an mp4 that is never released has no index and
        # will not play.
        reader.join(READER_JOIN_TIMEOUT)
        if reader.is_alive() and reader.raw_recorder is not None:
            print(f"[warn] capture thread still blocked in read(); "
                  f"{reader.raw_recorder.path} may be unplayable")
    return shown


def build_parser():
    # Built per call, not at import, so the --conf default is read from
    # pipeline_config when the tool runs rather than frozen at import time.
    parser = argparse.ArgumentParser(
        description="Live viewer: any OpenCV source, YOLO detections drawn on top.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default="0",
                        help="camera index, video file path, or stream URL "
                             "(Anafi: rtsp://192.168.42.1/live). Default 0")
    parser.add_argument("--model", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"YOLO weights (default: {DEFAULT_WEIGHTS.name}, "
                             "the presence service's face detector)")
    parser.add_argument("--conf", type=float,
                        default=pipeline_config.DETECTION_CONFIDENCE_THRESHOLD,
                        help="minimum confidence (default: %(default)s, "
                             "from pipeline_config)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="give up if no frame arrives for this many seconds "
                             "(default: %(default)s)")
    parser.add_argument("--rtsp-transport", choices=("udp", "tcp"), default="udp",
                        help="RTSP transport for rtsp:// sources (default: udp). "
                             "Try tcp if udp shows smearing or never connects")
    parser.add_argument("--record", type=Path, default=None, metavar="PATH",
                        help="also save the annotated stream here (.mp4)")
    parser.add_argument("--record-raw", type=Path, default=None, metavar="PATH",
                        help="also save every captured frame, unannotated, here "
                             "(.mp4), with capture times in a .timestamps.csv "
                             "beside it (raw.mp4 -> raw.timestamps.csv). "
                             "For labeling datasets; can be combined with --record")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.model.is_file():
        parser.error(f"model weights not found: {args.model}")
    kind, source = parse_source(args.source)
    if kind == SOURCE_FILE and not Path(source).is_file():
        parser.error(f"video file not found: {source}")
    if args.record is not None and args.record_raw is not None \
            and args.record.resolve() == args.record_raw.resolve():
        parser.error("--record and --record-raw must be different files")

    print(f"  source    : {source!r} ({kind}"
          + (f", rtsp over {args.rtsp_transport}" if is_rtsp(source) else "") + ")")
    print(f"  model     : {args.model}")
    print(f"  conf      : {args.conf}")
    print(f"  timeout   : {args.timeout:g}s")

    # The same factory presence_service.py builds its detector with.
    detector = face_utils.load_detector(
        pipeline_config.DETECTION_DETECTOR,
        min_confidence=args.conf,
        weights=str(args.model),
    )

    display = _open_window()
    if not display:
        if args.record is None and args.record_raw is None:
            print("[error] this OpenCV build has no GUI support; "
                  "pass --record or --record-raw to run headless")
            return 1
        print("[warn] no GUI support in this OpenCV build - recording headless")

    reader = FrameReader(
        lambda: open_capture(kind, source, args.rtsp_transport, args.timeout),
        live=kind != SOURCE_FILE,
        description=f"{kind} {source!r}",
        raw_recorder=RawRecorder(args.record_raw) if args.record_raw else None,
    )
    reader.start()

    try:
        run(detector, reader, args.timeout, record_path=args.record, display=display)
    except StreamTimeout as exc:
        print(f"[error] {exc}")
        if is_rtsp(source):
            print("        check this machine is on the drone's Wi-Fi, "
                  "or retry with --rtsp-transport "
                  + ("tcp" if args.rtsp_transport == "udp" else "udp"))
        return 2
    except StreamError as exc:
        print(f"[error] {exc}")
        return 2
    except KeyboardInterrupt:
        print("\n[interrupt] stopping")
    finally:
        if display:
            try:
                cv2.destroyWindow(WINDOW)
                cv2.waitKey(1)
            except cv2.error:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
