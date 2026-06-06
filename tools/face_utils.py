"""
JARVIS Face Recognition — Shared Utilities
============================================
A dependency-light library of reusable helpers for the face-recognition
pipeline. This module ONLY defines functions/classes — it is imported by
benchmark_detection.py, benchmark_recognition.py, and the live presence
service. It is not meant to be run directly.

----------------------------------------------------------------------
BOX FORMAT CONVENTION (every other file depends on this):
    A box is [x1, y1, x2, y2] in ABSOLUTE PIXELS, where
    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right
    corner, with x1 <= x2 and y1 <= y2. Coordinates are clamped to the
    image bounds. Detector backends differ natively (MediaPipe Tasks API
    uses origin_x/origin_y + width/height in pixels, MTCNN uses
    [x, y, width, height], InsightFace uses [x1, y1, x2, y2]); load_detector()
    normalizes ALL of them to the convention above so calling code never
    branches on the backend.
----------------------------------------------------------------------

Heavy backend libraries (mediapipe, mtcnn, insightface) are imported
LAZILY inside each detector — importing face_utils never requires all
three to be installed, only the one(s) you actually load.

Environment: CPU-only (AMD Ryzen, no CUDA). InsightFace runs on ONNX
Runtime's CPUExecutionProvider. Cross-platform; developed on Windows 11.
"""

import cv2
import numpy as np

__all__ = [
    "load_detector",
    "compute_iou",
    "crop_face",
    "draw_detections",
]


# ── Detector backends ────────────────────────────────────────────
# Each detector exposes the SAME interface:
#   .detect(image) -> list[dict]  where image is a BGR numpy array and
#   each dict is {"box": [x1, y1, x2, y2], "confidence": float} in
#   absolute pixels (see the box-format convention at the top of this file).

class _MediaPipeDetector:
    """MediaPipe Face Detection wrapper (Tasks API, BlazeFace short-range).

    Uses the modern mediapipe Tasks API (mediapipe>=0.10.x); the legacy
    mp.solutions.face_detection API was removed in 0.10.18+. The Tasks API
    returns bounding boxes already in ABSOLUTE pixels (origin_x/origin_y +
    width/height), so .detect() only repackages and clamps them.

    Requires the blaze_face_short_range.tflite model file (passed via
    model_asset_path). Short-range BlazeFace is tuned for faces within ~2m
    that fill a good fraction of the frame — it downsamples the input to
    128x128, so small/distant faces in large images are easily missed.
    """

    def __init__(self, min_confidence=0.5, model_asset_path="blaze_face_short_range.tflite"):
        """
        Inputs:
            min_confidence (float): minimum detection score to keep, 0-1.
            model_asset_path (str): path to the blaze_face_short_range.tflite
                model file.
        """
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_asset_path),
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image):
        """Detect faces in a BGR image.

        Inputs:
            image (np.ndarray): BGR image, shape (H, W, 3).
        Returns:
            list[dict]: each {"box": [x1, y1, x2, y2] absolute pixels,
                        "confidence": float}.
        """
        h, w = image.shape[:2]
        # Tasks API wants an mp.Image of contiguous RGB uint8 pixels.
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        detections = []
        if not result.detections:
            return detections

        for det in result.detections:
            bb = det.bounding_box
            x1 = bb.origin_x
            y1 = bb.origin_y
            x2 = bb.origin_x + bb.width
            y2 = bb.origin_y + bb.height
            box = _clamp_box([x1, y1, x2, y2], w, h)
            if box is None:
                continue
            confidence = float(det.categories[0].score) if det.categories else 0.0
            detections.append({"box": box, "confidence": confidence})
        return detections


class _MTCNNDetector:
    """MTCNN wrapper.

    MTCNN returns boxes as [x, y, width, height] in absolute pixels;
    .detect() converts them to [x1, y1, x2, y2].
    """

    def __init__(self, min_confidence=0.5):
        """
        Inputs:
            min_confidence (float): minimum detection score to keep, 0-1.
        """
        from mtcnn import MTCNN

        self._detector = MTCNN()
        self._min_confidence = min_confidence

    def detect(self, image):
        """Detect faces in a BGR image.

        Inputs:
            image (np.ndarray): BGR image, shape (H, W, 3).
        Returns:
            list[dict]: each {"box": [x1, y1, x2, y2] absolute pixels,
                        "confidence": float}.
        """
        h, w = image.shape[:2]
        # MTCNN expects RGB.
        results = self._detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        detections = []
        for det in results:
            confidence = float(det.get("confidence", 0.0))
            if confidence < self._min_confidence:
                continue
            x, y, bw, bh = det["box"]
            box = _clamp_box([x, y, x + bw, y + bh], w, h)
            if box is None:
                continue
            detections.append({"box": box, "confidence": confidence})
        return detections


class _RetinaFaceDetector:
    """RetinaFace detector via InsightFace (ONNX Runtime, CPU).

    InsightFace returns boxes already as [x1, y1, x2, y2] in absolute
    pixels; .detect() only clamps and repackages them.
    """

    def __init__(self, min_confidence=0.5, det_size=(640, 640)):
        """
        Inputs:
            min_confidence (float): minimum detection score to keep, 0-1.
            det_size (tuple[int, int]): detection input size (w, h).
        """
        from insightface.app import FaceAnalysis

        # CPU-only: force the ONNX Runtime CPU provider and a CPU context.
        # (ctx_id=-1 selects CPU; 0+ would select a GPU device).
        self._app = FaceAnalysis(
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=det_size)
        self._min_confidence = min_confidence

    def detect(self, image):
        """Detect faces in a BGR image.

        Inputs:
            image (np.ndarray): BGR image, shape (H, W, 3). InsightFace
                consumes BGR natively, so no color conversion is done.
        Returns:
            list[dict]: each {"box": [x1, y1, x2, y2] absolute pixels,
                        "confidence": float}.
        """
        h, w = image.shape[:2]
        faces = self._app.get(image)

        detections = []
        for face in faces:
            confidence = float(face.det_score)
            if confidence < self._min_confidence:
                continue
            x1, y1, x2, y2 = face.bbox
            box = _clamp_box([x1, y1, x2, y2], w, h)
            if box is None:
                continue
            detections.append({"box": box, "confidence": confidence})
        return detections


class _YOLOFaceDetector:
    """YOLOv8-face detector via Ultralytics.

    Ultralytics YOLO returns boxes already as [x1, y1, x2, y2] in absolute
    pixels (results[0].boxes.xyxy); .detect() only clamps and repackages
    them. Requires a YOLOv8-face weights file (e.g. yolov8n-face.pt) — the
    stock COCO yolov8n.pt detects "person", not faces.
    """

    def __init__(self, min_confidence=0.5, weights="yolov8n-face.pt", device="cpu"):
        """
        Inputs:
            min_confidence (float): minimum detection score to keep, 0-1.
            weights (str): path to a YOLOv8-face .pt weights file.
            device (str): inference device. CPU-only here, so "cpu".
        """
        from ultralytics import YOLO

        self._model = YOLO(weights)
        # CPU-only: ultralytics auto-selects CUDA if available, so pin the
        # device. ("cpu" forces CPU; "0"/"cuda:0" would select a GPU). The
        # device is also passed to each predict() call below to be safe.
        self._device = device
        self._min_confidence = min_confidence

    def detect(self, image):
        """Detect faces in a BGR image.

        Inputs:
            image (np.ndarray): BGR image, shape (H, W, 3). Ultralytics YOLO
                consumes numpy arrays as BGR natively (PIL inputs are RGB),
                so no color conversion is done — matching InsightFace.
        Returns:
            list[dict]: each {"box": [x1, y1, x2, y2] absolute pixels,
                        "confidence": float}.
        """
        h, w = image.shape[:2]
        # conf=0.0 keeps every detection so callers can sweep PR curves; we
        # apply min_confidence ourselves below (as MTCNN/InsightFace do).
        results = self._model.predict(
            image, device=self._device, conf=0.0, verbose=False
        )

        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # xyxy is already [x1, y1, x2, y2] absolute pixels; move off any
        # device and out of torch into plain numpy.
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            confidence = float(conf)
            if confidence < self._min_confidence:
                continue
            box = _clamp_box([x1, y1, x2, y2], w, h)
            if box is None:
                continue
            detections.append({"box": box, "confidence": confidence})
        return detections


_DETECTOR_REGISTRY = {
    "mediapipe": _MediaPipeDetector,
    "mtcnn": _MTCNNDetector,
    "retinaface": _RetinaFaceDetector,
    "yolo": _YOLOFaceDetector,
    "yolo-face": _YOLOFaceDetector,
}


def load_detector(name, **kwargs):
    """Factory returning a uniform face detector for a given backend.

    Inputs:
        name (str): backend identifier, case-insensitive. One of
            "mediapipe", "mtcnn", "retinaface" (RetinaFace via InsightFace).
        **kwargs: forwarded to the backend constructor (e.g. min_confidence).
    Returns:
        A detector object exposing .detect(image) -> list[dict], where each
        dict is {"box": [x1, y1, x2, y2] absolute pixels, "confidence": float}.
        The interface is identical across backends — calling code must never
        branch on which backend it is.
    Raises:
        ValueError: if `name` is not a supported backend.
    """
    key = name.strip().lower()
    if key not in _DETECTOR_REGISTRY:
        supported = ", ".join(sorted(_DETECTOR_REGISTRY))
        raise ValueError(f"Unknown detector backend {name!r}. Supported: {supported}.")
    return _DETECTOR_REGISTRY[key](**kwargs)


# ── Geometry helpers ─────────────────────────────────────────────

def _clamp_box(box, width, height):
    """Clamp a box to image bounds and validate it.

    Inputs:
        box (sequence of 4 numbers): [x1, y1, x2, y2] in absolute pixels.
        width (int): image width in pixels.
        height (int): image height in pixels.
    Returns:
        list[int] | None: [x1, y1, x2, y2] clamped to [0, width]/[0, height]
        as ints, or None if the box has zero/negative area after clamping.
    """
    x1, y1, x2, y2 = box
    # Normalize ordering so x1 <= x2 and y1 <= y2.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    x1 = int(round(max(0, min(x1, width))))
    y1 = int(round(max(0, min(y1, height))))
    x2 = int(round(max(0, min(x2, width))))
    y2 = int(round(max(0, min(y2, height))))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def compute_iou(box_a, box_b):
    """Intersection over Union of two boxes.

    Inputs:
        box_a, box_b (sequence of 4 numbers): [x1, y1, x2, y2] in absolute
            pixels (top-left / bottom-right corners).
    Returns:
        float: IoU in [0.0, 1.0]. Returns 0.0 when the boxes do not overlap
        or when either box has non-positive area.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection rectangle.
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return float(inter_area / union)


def crop_face(image, box, margin=0.0):
    """Crop the face region from an image, with optional margin padding.

    Inputs:
        image (np.ndarray): BGR image, shape (H, W, 3).
        box (sequence of 4 numbers): [x1, y1, x2, y2] in absolute pixels.
        margin (float): fraction of the box's width/height to pad on each
            side (e.g. 0.2 = 20% padding). The padded box is clamped to the
            image bounds. Defaults to 0.0 (no padding).
    Returns:
        np.ndarray: the cropped sub-image (a copy). Shape is
        (crop_h, crop_w, 3). Returns an empty array if the box is degenerate.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin
    padded = _clamp_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], w, h)
    if padded is None:
        return np.empty((0, 0, image.shape[2] if image.ndim == 3 else 0), dtype=image.dtype)

    px1, py1, px2, py2 = padded
    return image[py1:py2, px1:px2].copy()


# ── Visualization ────────────────────────────────────────────────

def draw_detections(image, detections, color=(0, 255, 0), thickness=2):
    """Draw detection boxes and confidence scores onto a copy of the image.

    Inputs:
        image (np.ndarray): BGR image, shape (H, W, 3).
        detections (list[dict]): each {"box": [x1, y1, x2, y2] absolute
            pixels, "confidence": float}, as returned by a detector's
            .detect() method.
        color (tuple[int, int, int]): BGR box/label color. Default green.
        thickness (int): box line thickness in pixels.
    Returns:
        np.ndarray: a COPY of the input image with boxes and confidence
        labels drawn. The input image is not modified.
    """
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        confidence = det.get("confidence", 0.0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Place the label just above the box, or just inside if near the top.
        label_y = y1 - baseline - 2
        if label_y - text_h < 0:
            label_y = y1 + text_h + baseline + 2
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated
