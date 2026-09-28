"""
Test suite for tools/view_stream.py.

No camera, no network, no model weights: captures and detectors are fakes
injected through FrameReader's open_fn and run()'s detector argument. The
timeout tests use deadlines of a fraction of a second, so the "no frame ever
arrives" path is exercised without waiting out the real 10s default.

Run:
  pytest tests/test_view_stream.py
"""

import csv
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import face_utils  # noqa: E402  (import follows the sys.path tweak)
import pipeline_config  # noqa: E402
import view_stream  # noqa: E402
from view_stream import (  # noqa: E402
    SOURCE_CAMERA,
    SOURCE_FILE,
    SOURCE_URL,
    FrameReader,
    RawRecorder,
    StreamError,
    StreamTimeout,
    draw_overlay,
    ffmpeg_capture_options,
    parse_source,
)

SHORT = 0.3  # seconds - the deadline used by every timeout test


# ── Fakes ────────────────────────────────────────────────────────

class FakeCapture:
    """cv2.VideoCapture stand-in that plays a fixed list of frames."""

    def __init__(self, frames=(), opened=True, block=False, delay=0.0):
        self._frames = list(frames)
        self._opened = opened
        self._block = block
        self._delay = delay
        self.released = threading.Event()

    def isOpened(self):  # noqa: N802 - mirrors the cv2 API
        return self._opened

    def get(self, prop):
        return 25.0

    def read(self):
        if self._block:
            # The RTSP failure mode: read() just never returns.
            self.released.wait()
            return False, None
        if self._frames:
            time.sleep(self._delay)
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released.set()


class FakeDetector:
    def __init__(self, detections=()):
        self.detections = list(detections)
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [dict(d) for d in self.detections]


def blank(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def start_reader(capture, live, raw_recorder=None):
    reader = FrameReader(lambda: capture, live=live, description="test source",
                         raw_recorder=raw_recorder)
    reader.start()
    return reader


class ListRecorder:
    """RawRecorder stand-in that keeps what it is given."""

    def __init__(self):
        self.fps = None
        self.frames = []
        self.captured = []
        self.closed = False

    def write(self, frame, captured, unix_time, pos_ms):
        self.frames.append(frame.copy())
        self.captured.append(captured)

    def close(self):
        self.closed = True


def read_video(path):
    cv2 = pytest.importorskip("cv2")
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


# ── Source parsing ───────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [
    ("0", (SOURCE_CAMERA, 0)),
    ("2", (SOURCE_CAMERA, 2)),
    (" 1 ", (SOURCE_CAMERA, 1)),
])
def test_digits_are_a_camera_index(value, expected):
    kind, source = parse_source(value)
    assert (kind, source) == expected
    assert isinstance(source, int)


@pytest.mark.parametrize("value", [
    "rtsp://192.168.42.1/live",
    "RTSP://192.168.42.1/live",
    "http://10.0.0.5:8080/video",
    "udp://@:5000",
])
def test_scheme_urls_are_urls(value):
    assert parse_source(value) == (SOURCE_URL, value)


@pytest.mark.parametrize("value", [
    "clip.mp4",
    "data/videos/hallway.avi",
    r"C:\videos\drone.mp4",
    "0.mp4",            # starts with a digit, is still a file
    "12345_flight.mov",
])
def test_everything_else_is_a_file(value):
    assert parse_source(value) == (SOURCE_FILE, value)


def test_rtsp_detection():
    assert view_stream.is_rtsp("rtsp://192.168.42.1/live")
    assert view_stream.is_rtsp("RTSPS://host/x")
    assert not view_stream.is_rtsp("http://host/x")
    assert not view_stream.is_rtsp(0)


def test_ffmpeg_capture_options():
    assert ffmpeg_capture_options("udp") == "rtsp_transport;udp"
    assert ffmpeg_capture_options("tcp") == "rtsp_transport;tcp"
    with pytest.raises(ValueError):
        ffmpeg_capture_options("http")


# ── Timeout path ─────────────────────────────────────────────────

def test_live_source_that_never_yields_a_frame_times_out():
    """Opens fine, read() keeps failing: must raise, not spin forever."""
    capture = FakeCapture(frames=[])
    reader = start_reader(capture, live=True)
    try:
        with pytest.raises(StreamTimeout, match=r"no frame from test source within 0.3s"):
            reader.next_frame(SHORT)
    finally:
        reader.stop()


def test_source_whose_read_blocks_forever_times_out():
    """The black-screen case: read() never returns at all."""
    capture = FakeCapture(block=True)
    reader = start_reader(capture, live=True)
    try:
        with pytest.raises(StreamTimeout, match="no frame"):
            reader.next_frame(SHORT)
    finally:
        reader.stop()
        capture.release()   # unblock the fake so the thread can exit


def test_stall_after_frames_says_it_stalled():
    capture = FakeCapture(frames=[blank()])
    reader = start_reader(capture, live=True)
    try:
        assert reader.next_frame(2.0) is not None
        with pytest.raises(StreamTimeout, match="stalled.*after 1 frames"):
            reader.next_frame(SHORT)
    finally:
        reader.stop()


def test_source_that_fails_to_open_is_an_error_not_a_timeout():
    reader = start_reader(FakeCapture(opened=False), live=True)
    with pytest.raises(StreamError, match="could not open"):
        reader.next_frame(2.0)


def test_empty_file_is_an_error():
    reader = start_reader(FakeCapture(frames=[]), live=False)
    with pytest.raises(StreamError, match="without producing a frame"):
        reader.next_frame(2.0)


def test_file_delivers_every_frame_then_ends_cleanly():
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(5)]
    capture = FakeCapture(frames=frames)
    reader = start_reader(capture, live=False)
    got = []
    while (frame := reader.next_frame(2.0)) is not None:
        got.append(int(frame[0, 0, 0]))
    assert got == [0, 1, 2, 3, 4]       # no drops for a file
    assert capture.released.wait(2.0)


def test_timeout_propagates_out_of_run():
    reader = start_reader(FakeCapture(frames=[]), live=True)
    with pytest.raises(StreamTimeout):
        view_stream.run(FakeDetector(), reader, SHORT, display=False)


# ── Overlay ──────────────────────────────────────────────────────

def test_overlay_draws_box_label_and_hud():
    frame = blank(240, 320)
    det = {"box": [100, 120, 200, 220], "confidence": 0.87, "label": "face"}
    color = view_stream._color_for("face")

    draw_overlay(frame, [det], fps=29.7, inference_ms=41.2)

    # Box edges carry the class colour.
    assert tuple(frame[170, 100]) == color      # left edge
    assert tuple(frame[220, 150]) == color      # bottom edge
    # Interior untouched.
    assert not frame[170, 150].any()
    # Label box filled just above the box.
    assert (frame[105:119, 101:140] == color).all(axis=-1).any()
    # HUD band at the top has (anti-aliased, so not pure white) text in it.
    assert (frame[0:30].min(axis=-1) > 200).sum() > 50


def test_overlay_is_in_place_and_returns_nothing():
    frame = blank()
    assert draw_overlay(frame, [], fps=0.0, inference_ms=0.0) is None
    assert frame.any()      # HUD drawn even with no detections


def test_overlay_without_label_falls_back_to_object():
    frame = blank()
    draw_overlay(frame, [{"box": [10, 50, 60, 100], "confidence": 0.6}], 10.0, 5.0)
    assert tuple(frame[75, 10]) == view_stream._color_for("object")


def test_overlay_clamps_boxes_at_the_frame_edges():
    """A detection hugging the top-right corner must not raise."""
    frame = blank(100, 100)
    dets = [{"box": [80, 0, 100, 20], "confidence": 0.99, "label": "a-very-long-class-name"}]
    draw_overlay(frame, dets, 1.0, 1.0)


def test_fps_meter():
    meter = view_stream.FpsMeter(window=5)
    assert meter.fps == 0.0
    for i in range(5):
        meter.tick(now=i * 0.1)
    assert meter.fps == pytest.approx(10.0)


# ── End to end, headless ─────────────────────────────────────────

def test_run_records_every_frame_of_a_file(tmp_path):
    frames = [blank(120, 160) for _ in range(6)]
    detector = FakeDetector([{"box": [20, 40, 80, 100], "confidence": 0.9, "label": "face"}])
    reader = start_reader(FakeCapture(frames=frames), live=False)
    out = tmp_path / "out.mp4"

    shown = view_stream.run(detector, reader, 2.0, record_path=out, display=False)

    assert shown == 6 and detector.calls == 6
    assert len(read_video(out)) == 6


# ── --conf default ───────────────────────────────────────────────

def test_conf_default_is_pipeline_config_value():
    args = view_stream.build_parser().parse_args([])
    assert args.conf == pipeline_config.DETECTION_CONFIDENCE_THRESHOLD


def test_conf_default_follows_pipeline_config(monkeypatch):
    """Read from the config at run time, not a literal that happens to match."""
    monkeypatch.setattr(pipeline_config, "DETECTION_CONFIDENCE_THRESHOLD", 0.123)
    assert view_stream.build_parser().parse_args([]).conf == 0.123


def test_conf_can_be_overridden():
    assert view_stream.build_parser().parse_args(["--conf", "0.3"]).conf == 0.3


# ── --record-raw ─────────────────────────────────────────────────

def test_sidecar_path():
    assert view_stream.sidecar_path(Path("out/raw.mp4")) == Path("out/raw.timestamps.csv")


def test_record_raw_parses_alongside_record():
    args = view_stream.build_parser().parse_args(
        ["--record", "a.mp4", "--record-raw", "b.mp4"])
    assert (args.record, args.record_raw) == (Path("a.mp4"), Path("b.mp4"))


def test_record_and_record_raw_must_differ(capsys):
    with pytest.raises(SystemExit):
        view_stream.main(["--record", "same.mp4", "--record-raw", "same.mp4",
                          "--model", __file__, "--source", "0"])
    assert "must be different files" in capsys.readouterr().err


def test_raw_gets_frames_a_live_source_dropped():
    """Nobody consumes the live slot, so display sees at most one frame -
    the raw recorder must still get all of them."""
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(10)]
    recorder = ListRecorder()
    reader = start_reader(FakeCapture(frames=frames), live=True, raw_recorder=recorder)
    time.sleep(0.2)
    reader.stop()
    reader.join(2.0)

    assert [int(f[0, 0, 0]) for f in recorder.frames] == list(range(10))
    assert recorder.closed
    assert recorder.fps == 25.0     # nominal fps taken from the source


def test_raw_timestamps_are_capture_times():
    """Frames arrive 50 ms apart; the recorded times must say so."""
    recorder = ListRecorder()
    reader = start_reader(FakeCapture(frames=[blank()] * 4, delay=0.05), live=True,
                          raw_recorder=recorder)
    time.sleep(0.5)
    reader.stop()
    reader.join(2.0)

    gaps = np.diff(recorder.captured)
    assert len(recorder.captured) == 4
    assert (gaps >= 0.04).all()


def test_record_raw_is_unannotated_and_complete_alongside_record(tmp_path):
    gray = 128
    frames = [np.full((120, 160, 3), gray, dtype=np.uint8) for _ in range(6)]
    box = [20, 40, 80, 100]
    detector = FakeDetector([{"box": box, "confidence": 0.9, "label": "face"}])
    raw_path, annotated_path = tmp_path / "raw.mp4", tmp_path / "annotated.mp4"
    reader = start_reader(FakeCapture(frames=frames), live=False,
                          raw_recorder=RawRecorder(raw_path))

    view_stream.run(detector, reader, 2.0, record_path=annotated_path, display=False)

    raw, annotated = read_video(raw_path), read_video(annotated_path)
    assert len(raw) == len(annotated) == 6
    color = np.array(view_stream._color_for("face"))
    for r, a in zip(raw, annotated):
        # Box edge: coloured in the annotated video, still plain grey in raw.
        assert np.abs(a[70, 20].astype(int) - color).max() < 40
        assert np.abs(r[70, 20].astype(int) - gray).max() < 10
        # HUD band: black in the annotated video, grey in raw.
        assert a[5, 150].max() < 40
        assert np.abs(r[5, 150].astype(int) - gray).max() < 10

    with open(view_stream.sidecar_path(raw_path), newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["frame"]) for row in rows] == list(range(6))
    t = [float(row["t_s"]) for row in rows]
    assert t[0] == 0.0 and t == sorted(t)
    assert all(float(row["unix_time"]) > 1e9 for row in rows)
    assert set(rows[0]) == set(RawRecorder.COLUMNS)


# ── face_utils: YOLO detections carry a class label ──────────────

class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Tensor(xyxy), _Tensor(conf), _Tensor(cls)

    def __len__(self):
        return len(self.conf.numpy())


class _Model:
    names = {0: "person", 1: "car"}

    def predict(self, image, **kwargs):
        boxes = _Boxes([[10, 10, 50, 50], [60, 60, 90, 90], [0, 0, 5, 5]],
                       [0.9, 0.7, 0.2], [0, 1, 0])
        return [type("R", (), {"boxes": boxes})()]


def test_yolo_detector_attaches_class_names():
    detector = face_utils._YOLOFaceDetector.__new__(face_utils._YOLOFaceDetector)
    detector._model = _Model()
    detector._device = "cpu"
    detector._min_confidence = 0.57

    dets = detector.detect(blank(100, 100))

    assert [d["label"] for d in dets] == ["person", "car"]     # 0.2 filtered out
    assert [round(d["confidence"], 2) for d in dets] == [0.9, 0.7]
