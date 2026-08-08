"""
JARVIS Face Recognition — Shared Utilities
============================================
A dependency-light library of reusable helpers for the face-recognition
pipeline. This module ONLY defines functions/classes — it is imported by
benchmark_detection.py, benchmark_recognition.py, build_gallery.py, and the
live presence service. It is not meant to be run directly.

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

----------------------------------------------------------------------
ENCODER CONTRACT (load_encoder, mirrors load_detector):
    An encoder exposes exactly ONE method:
        .embed(bgr_crop) -> np.ndarray, shape (D,), float32
    where bgr_crop is a RAW, generous-margin face crop (BGR). Every
    encoder receives the SAME crop; no caller ever pre-aligns or
    pre-resizes for it.

    EACH ENCODER OWNS ITS INPUT GEOMETRY. Alignment and resizing to the
    model's expected input live INSIDE .embed() — ArcFace wants an
    aligned 112x112, FaceNet512 wants 160x160, dlib wants 150x150,
    VGG-Face wants 224x224. Those numbers must never leak into shared
    pipeline code, so that adding an encoder means writing one class plus
    one registry entry and nothing else.
----------------------------------------------------------------------

Heavy backend libraries (mediapipe, mtcnn, insightface, facenet-pytorch,
face_recognition/dlib, deepface) are
imported LAZILY inside each detector/encoder — importing face_utils never
requires all of them to be installed, only the one(s) you actually load.

Environment: CPU-only (AMD Ryzen, no CUDA). InsightFace runs on ONNX
Runtime's CPUExecutionProvider. Cross-platform; developed on Windows 11.
"""

import os
import time

import cv2
import numpy as np

from recognition_metrics import l2_normalize

__all__ = [
    "load_detector",
    "load_encoder",
    "load_reference_crops",
    "embed_crops",
    "compute_iou",
    "crop_face",
    "draw_detections",
]

# Image files load_reference_crops() will read; anything else in a person
# folder (manifest.json, notes.txt) is ignored.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


# ── Encoder backends ─────────────────────────────────────────────
# Each encoder exposes the SAME interface:
#   .embed(bgr_crop) -> np.ndarray of shape (D,), float32
# The crop handed in is the raw, generous-margin crop produced by
# crop_face() — NOT aligned, NOT resized. Turning that crop into whatever
# geometry the model wants is the encoder's own job (see the encoder
# contract at the top of this file).

class _ArcFaceEncoder:
    """ArcFace (buffalo_l) via InsightFace, 512-D embeddings, ONNX Runtime CPU.

    Owns its geometry: .embed() hands the generous-margin crop to
    InsightFace's FULL pipeline (detection + recognition), so InsightFace
    re-detects the face inside the crop and warps it to ArcFace's canonical
    aligned 112x112 using the 5-point landmarks before embedding. That
    alignment is what ArcFace was trained on — feeding it an unaligned,
    naively-resized crop measurably degrades the embedding.

    If re-detection finds no face inside the crop (small, blurry, or
    extreme-pose crops), .embed() FALLS BACK to an unaligned resize to
    112x112 straight into the recognition model, and increments
    .fallback_count. The embedding is still usable but lower quality, so
    callers should report that count.
    """

    def __init__(self, model_name="buffalo_l", det_size=(320, 320), model_root=None):
        """
        Inputs:
            model_name (str): InsightFace model pack. "buffalo_l" bundles
                SCRFD detection + the ArcFace r100 recognition model.
            det_size (tuple[int, int]): input size (w, h) for the internal
                re-detection pass. Tuned for face CROPS, not full photos: the
                face already fills most of the crop, so 320x320 is plenty and
                is ~4x cheaper than the 640x640 used for whole scenes.
            model_root (str | None): directory holding the model pack.
                None uses InsightFace's default (~/.insightface/models).
        """
        from insightface.app import FaceAnalysis

        kwargs = {
            "name": model_name,
            "allowed_modules": ["detection", "recognition"],
            "providers": ["CPUExecutionProvider"],
        }
        if model_root is not None:
            kwargs["root"] = model_root

        # CPU-only: ctx_id=-1 selects CPU (0+ would select a GPU device).
        self._app = FaceAnalysis(**kwargs)
        self._app.prepare(ctx_id=-1, det_size=det_size)

        # Kept for the fallback path: the bare recognition model, which takes
        # a 112x112 BGR image and does its own blob preprocessing internally.
        self._recognition = self._app.models["recognition"]
        self.input_size = (112, 112)
        self.fallback_count = 0

    def embed(self, bgr_crop):
        """Embed a raw face crop.

        Inputs:
            bgr_crop (np.ndarray): BGR crop, shape (h, w, 3). Any size —
                alignment and resizing happen inside this method.
        Returns:
            np.ndarray: float32 embedding, shape (512,). NOT L2-normalized
            (callers normalize, so the choice of metric stays theirs).
        Raises:
            ValueError: if the crop is empty.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            raise ValueError("embed() received an empty crop")

        faces = self._app.get(bgr_crop)
        face = _most_central_face(faces, bgr_crop.shape)

        if face is not None:
            # InsightFace already aligned this to 112x112 internally.
            return np.asarray(face.embedding, dtype=np.float32).reshape(-1)

        # Fallback: no face re-detected inside the crop. Resize the whole
        # crop to 112x112 unaligned and embed it anyway.
        self.fallback_count += 1
        resized = cv2.resize(bgr_crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        feat = self._recognition.get_feat(resized)  # shape (1, 512)
        return np.asarray(feat, dtype=np.float32).reshape(-1)


class _FaceNet512Encoder:
    """FaceNet512 (InceptionResnetV1, VGGFace2) via facenet-pytorch, 512-D.

    Owns its geometry, exactly like ArcFace owns InsightFace's pipeline:
    facenet-pytorch's own MTCNN re-detects and aligns the face INSIDE the raw
    crop and warps it to the 160x160 the network expects, with the standard
    fixed image standardization ((x - 127.5) / 128) that VGGFace2 was trained
    on. Feeding an unaligned naive resize instead measurably degrades the
    embedding, which is why alignment lives here and never in the pipeline.

    If MTCNN re-detects no face inside the crop (small/blurry/extreme pose),
    .embed() FALLS BACK to an unaligned standardized resize to 160x160 and
    increments .fallback_count — the same tolerance and reporting as ArcFace.
    """

    def __init__(self, pretrained="vggface2"):
        """
        Inputs:
            pretrained (str): InceptionResnetV1 weights, "vggface2" (default)
                or "casia-webface". Downloaded and cached by facenet-pytorch on
                first use.
        Raises:
            ImportError: if facenet-pytorch (or torch) is not installed.
        """
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
        except ImportError as exc:
            raise ImportError(
                "The 'facenet512' encoder requires facenet-pytorch, which is not "
                "installed.\n"
                "  Install it with:  pip install facenet-pytorch\n"
                "It depends on torch (already present in this venv). NOTE: some "
                "facenet-pytorch releases pin an older numpy — install with "
                "'--no-deps' if pip tries to downgrade numpy below 2.0 and break "
                "insightface/mediapipe. Use --encoders arcface to stay on InsightFace."
            ) from exc

        self._torch = torch
        # select_largest=False + a center bias is not offered by MTCNN, so keep
        # the largest face; a generous single-face crop makes that the subject.
        self._mtcnn = MTCNN(image_size=160, margin=0, post_process=True,
                            select_largest=True, device="cpu")
        self._resnet = InceptionResnetV1(pretrained=pretrained).eval()
        self.input_size = (160, 160)
        self.fallback_count = 0

    def embed(self, bgr_crop):
        """Embed a raw face crop.

        Inputs:
            bgr_crop (np.ndarray): BGR crop, shape (h, w, 3). Any size —
                alignment and resizing happen inside this method.
        Returns:
            np.ndarray: float32 embedding, shape (512,). NOT L2-normalized
            (callers normalize, keeping the choice of metric theirs).
        Raises:
            ValueError: if the crop is empty.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            raise ValueError("embed() received an empty crop")

        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

        # MTCNN detects + aligns + standardizes to a (3, 160, 160) tensor, or
        # returns None when it finds no face inside the crop.
        aligned = self._mtcnn(rgb)
        if aligned is None:
            # Fallback: unaligned resize with the SAME standardization MTCNN
            # applies, so the embedding stays on the network's expected scale.
            self.fallback_count += 1
            resized = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
            tensor = self._torch.from_numpy(resized).permute(2, 0, 1).float()
            aligned = (tensor - 127.5) / 128.0

        with self._torch.no_grad():
            feat = self._resnet(aligned.unsqueeze(0))
        return feat.squeeze(0).cpu().numpy().astype(np.float32).reshape(-1)


class _AdaFaceEncoder:
    """AdaFace (IR-101) encoder — DEFERRED, not implemented yet.

    Registered so '--encoders adaface' is a recognized name, but construction
    raises so the benchmark SKIPS it with a clear message (handled identically
    to a missing dependency). AdaFace ships no clean pip package: wiring it up
    means vendoring the IR-101 architecture into the repo plus downloading an
    official checkpoint (e.g. adaface_ir101_webface12m.ckpt) into tools/weights/.
    Planned as a follow-up once arcface / facenet512 / dlib are validated.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "The 'adaface' encoder is not implemented yet (deferred): it needs "
            "the IR-101 architecture vendored plus an official checkpoint in "
            "tools/weights/. Run with --encoders arcface facenet512 dlib for now."
        )

    def embed(self, bgr_crop):  # pragma: no cover - unreachable; __init__ raises
        raise NotImplementedError("The 'adaface' encoder is deferred.")


class _DlibEncoder:
    """dlib's 128-D ResNet face encoder via the face_recognition library.

    Owns its geometry: face_recognition detects the face inside the crop and
    warps it to dlib's aligned 150x150 using 68-point landmarks before running
    the ResNet, producing a 128-D descriptor. dlib descriptors were trained for
    EUCLIDEAN distance, but the benchmark L2-normalizes every encoder's output
    and compares by cosine uniformly — valid for ranking, just note dlib is not
    playing on its native metric.

    If no face is re-detected inside the crop, .embed() FALLS BACK to encoding
    the whole crop as a single face box and increments .fallback_count.
    """

    def __init__(self, num_jitters=1, model="large"):
        """
        Inputs:
            num_jitters (int): times to re-sample/augment each face when
                encoding; higher is slower but slightly more stable.
            model (str): landmark predictor for alignment, "large" (68-point,
                default, more accurate) or "small" (5-point, faster).
        Raises:
            ImportError: if face_recognition (dlib) is not installed.
        """
        try:
            import face_recognition
        except ImportError as exc:
            raise ImportError(
                "The 'dlib' encoder requires the face_recognition library, which "
                "is not installed.\n"
                "  Install it with:  pip install face_recognition\n"
                "It depends on dlib, which on Windows needs CMake + Visual Studio "
                "Build Tools to compile. Use --encoders arcface to skip it."
            ) from exc

        self._fr = face_recognition
        self._num_jitters = num_jitters
        self._model = model
        self.input_size = (150, 150)
        self.fallback_count = 0

    def embed(self, bgr_crop):
        """Embed a raw face crop.

        Inputs:
            bgr_crop (np.ndarray): BGR crop, shape (h, w, 3). Any size.
        Returns:
            np.ndarray: float32 embedding, shape (128,). NOT L2-normalized.
        Raises:
            ValueError: if the crop is empty or dlib returns no descriptor.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            raise ValueError("embed() received an empty crop")

        # face_recognition wants a contiguous uint8 RGB image.
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))

        locations = self._fr.face_locations(rgb)
        if not locations:
            # Fallback: treat the whole crop as the face box (top, right,
            # bottom, left), the face_recognition location convention.
            self.fallback_count += 1
            h, w = rgb.shape[:2]
            locations = [(0, w, h, 0)]
        else:
            locations = [_most_central_location(locations, rgb.shape)]

        encodings = self._fr.face_encodings(
            rgb, known_face_locations=locations,
            num_jitters=self._num_jitters, model=self._model,
        )
        if not encodings:
            raise ValueError("face_recognition returned no descriptor for this crop")
        return np.asarray(encodings[0], dtype=np.float32).reshape(-1)


class _VGGFaceEncoder:
    """VGG-Face via DeepFace, 4096-D embeddings (Keras/TensorFlow, CPU).

    Included as a THIRD LOSS LINEAGE, which is the whole point of having it:
    VGG-Face is a softmax classification-trained network, where ArcFace uses an
    additive angular margin and FaceNet512 uses a triplet loss. Three different
    training objectives disagree in different places, so a face all three agree
    on is a much stronger signal than one only two agree on.

    Owns its geometry, like every other encoder here: .embed() hands the raw
    generous-margin crop to DeepFace, which re-detects the face inside it,
    aligns it on the eye line, and resizes to the 224x224 VGG-Face expects,
    applying its own pixel normalization. None of those numbers ever leak out
    of this class.

    If re-detection finds no face inside the crop, .embed() FALLS BACK to
    DeepFace's "skip" backend (unaligned whole-crop resize to 224x224) and
    increments .fallback_count — the same tolerance and reporting as ArcFace
    and FaceNet512.

    NOTE: DeepFace's VGG-Face client already L2-normalizes its output, so
    unlike the other encoders here this one returns a unit vector. Callers
    normalize anyway, and re-normalizing a unit vector is a no-op, so the
    contract still holds.
    """

    def __init__(self, detector_backend="mtcnn", normalization="base"):
        """
        Inputs:
            detector_backend (str): DeepFace detector used to re-detect and
                align the face INSIDE the crop. "mtcnn" (default) matches what
                FaceNet512 uses and is a fair counterpart to the SCRFD pass
                ArcFace runs, so latency and quality are compared on equal
                footing. "opencv" is faster but misses more; "skip" disables
                alignment entirely.
            normalization (str): DeepFace pixel normalization. "base" is what
                DeepFace's own published VGG-Face thresholds are calibrated on.
        Raises:
            ImportError: if deepface is not installed.
        """
        # WINDOWS LANDMINE: deepface's logger prints emoji, and on a cp1252
        # console print() then raises UnicodeEncodeError. Its weight downloader
        # logs one such line INSIDE the download try/except, so the encoding
        # error is swallowed and re-raised as a bogus "an exception occurred
        # while downloading" — the download never actually starts, and the
        # message sends you chasing a network problem that does not exist.
        # Silencing the logger (above CRITICAL) removes the entire class of
        # failure. This must be set BEFORE the import: deepface's Logger is a
        # singleton that reads the level once, at import time.
        os.environ.setdefault("DEEPFACE_LOG_LEVEL", "51")

        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise ImportError(
                "The 'vgg-face' encoder requires deepface, which is not installed.\n"
                "  Install it with:  pip install deepface tf-keras\n"
                "tf-keras is required alongside it: deepface loads legacy .h5 "
                "weights, which needs the pre-Keras-3 API on TensorFlow >= 2.16. "
                "The VGG-Face weights (~580 MB) download automatically on first "
                "use into ~/.deepface/weights. Use --encoders arcface to skip it."
            ) from exc

        self._deepface = DeepFace
        self._detector_backend = detector_backend
        self._normalization = normalization
        # Build the model now so model construction is never billed to the
        # first .embed() call (the benchmark times embed() alone). DeepFace
        # caches it internally, so represent() below reuses this instance.
        try:
            self._deepface.build_model("VGG-Face")
        except Exception as exc:
            # Surfaced by callers that print the message (the benchmark prints
            # skip reasons), so strip the emoji deepface puts in its errors —
            # printing them would crash the caller on a cp1252 console.
            raise RuntimeError(
                f"deepface could not build VGG-Face: {_ascii_safe(exc)}"
            ) from exc
        self.input_size = (224, 224)
        self.fallback_count = 0

    def embed(self, bgr_crop):
        """Embed a raw face crop.

        Inputs:
            bgr_crop (np.ndarray): BGR crop, shape (h, w, 3). Any size —
                alignment and resizing happen inside this method. DeepFace
                treats a numpy array input as BGR, so no color conversion.
        Returns:
            np.ndarray: float32 embedding, shape (4096,).
        Raises:
            ValueError: if the crop is empty or DeepFace returns no embedding.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            raise ValueError("embed() received an empty crop")

        try:
            results = self._deepface.represent(
                img_path=bgr_crop,
                model_name="VGG-Face",
                detector_backend=self._detector_backend,
                align=True,
                enforce_detection=True,
                normalization=self._normalization,
            )
        except ValueError:
            # DeepFace raises ValueError when it detects no face and
            # enforce_detection is on. Fall back to an unaligned whole-crop
            # resize, exactly as the other encoders do.
            results = None

        if not results:
            self.fallback_count += 1
            results = self._deepface.represent(
                img_path=bgr_crop,
                model_name="VGG-Face",
                detector_backend="skip",
                align=False,
                enforce_detection=False,
                normalization=self._normalization,
            )
            if not results:
                raise ValueError("deepface returned no embedding for this crop")

        # A generous-margin crop can catch a neighbour's face at the edge, so
        # take the detection nearest the crop centre (see _most_central_index).
        boxes = [_facial_area_to_box(r.get("facial_area")) for r in results]
        index = _most_central_index(boxes, bgr_crop.shape)
        return np.asarray(results[index]["embedding"], dtype=np.float32).reshape(-1)


_ENCODER_REGISTRY = {
    "arcface": _ArcFaceEncoder,
    "facenet512": _FaceNet512Encoder,
    "adaface": _AdaFaceEncoder,
    "dlib": _DlibEncoder,
    "vgg-face": _VGGFaceEncoder,
    "vggface": _VGGFaceEncoder,
}


def load_encoder(name, **kwargs):
    """Factory returning a uniform face encoder for a given backend.

    Inputs:
        name (str): backend identifier, case-insensitive. One of
            "arcface" (InsightFace buffalo_l, 512-D), "facenet512"
            (facenet-pytorch, InceptionResnetV1/VGGFace2, 512-D), "vgg-face"
            (DeepFace, 4096-D), "dlib" (face_recognition, 128-D), or
            "adaface" (IR-101, deferred — construction raises NotImplementedError).
        **kwargs: forwarded to the backend constructor.
    Returns:
        An encoder object exposing .embed(bgr_crop) -> np.ndarray of shape
        (D,). The interface is identical across backends — calling code must
        never branch on which backend it is, and must never pre-align or
        pre-resize the crop it passes in.
    Raises:
        ValueError: if `name` is not a supported backend.
        ImportError: if the backend's library is not installed.
    """
    key = name.strip().lower()
    if key not in _ENCODER_REGISTRY:
        supported = ", ".join(sorted(_ENCODER_REGISTRY))
        raise ValueError(f"Unknown encoder backend {name!r}. Supported: {supported}.")
    return _ENCODER_REGISTRY[key](**kwargs)


def _most_central_face(faces, crop_shape):
    """Pick the face nearest the centre of a crop, from an InsightFace result.

    A generous-margin crop of one face routinely catches a SLICE of a
    neighbouring face at the edge (group photos are exactly this shape). The
    subject of the crop is the one at its centre, so pick by distance from
    centre — not by area, which a half-visible bystander standing closer to
    the camera can easily win.

    Inputs:
        faces (list): InsightFace Face objects (each with .bbox = [x1,y1,x2,y2]).
        crop_shape (tuple): the crop's .shape, i.e. (h, w, ...).
    Returns:
        The Face nearest the crop centre (ties broken by larger area), or
        None if `faces` is empty.
    """
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]

    h, w = crop_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def rank(face):
        x1, y1, x2, y2 = face.bbox
        face_cx = (x1 + x2) / 2.0
        face_cy = (y1 + y2) / 2.0
        distance = ((face_cx - cx) ** 2 + (face_cy - cy) ** 2) ** 0.5
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return (distance, -area)  # nearest first; larger area wins ties

    return min(faces, key=rank)


def _most_central_location(locations, crop_shape):
    """Pick the face nearest the centre of a crop, from face_recognition output.

    The dlib analogue of _most_central_face: a generous-margin crop can catch a
    neighbour's face at the edge, so choose the box closest to the crop centre
    (ties broken by larger area) rather than whichever dlib happened to list
    first.

    Inputs:
        locations (list[tuple]): (top, right, bottom, left) boxes in pixels, the
            face_recognition convention.
        crop_shape (tuple): the crop's .shape, i.e. (h, w, ...).
    Returns:
        tuple: the (top, right, bottom, left) nearest the centre. Assumes
        `locations` is non-empty (callers guarantee it).
    """
    if len(locations) == 1:
        return locations[0]

    h, w = crop_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def rank(loc):
        top, right, bottom, left = loc
        face_cx = (left + right) / 2.0
        face_cy = (top + bottom) / 2.0
        distance = ((face_cx - cx) ** 2 + (face_cy - cy) ** 2) ** 0.5
        area = max(0.0, right - left) * max(0.0, bottom - top)
        return (distance, -area)  # nearest first; larger area wins ties

    return min(locations, key=rank)


def _most_central_index(boxes, crop_shape):
    """Index of the box nearest the centre of a crop — the plain-box analogue
    of _most_central_face / _most_central_location.

    Used when the caller needs to index back into a PARALLEL list (DeepFace
    returns one embedding per detected face, so picking the box is only half
    the job — you need its position to pick the matching embedding).

    Inputs:
        boxes (list): [x1, y1, x2, y2] per detection, in the crop's pixel
            coordinates. A None entry ranks last (unknown position).
        crop_shape (tuple): the crop's .shape, i.e. (h, w, ...).
    Returns:
        int: index of the box nearest the crop centre (ties broken by larger
        area), or 0 if `boxes` is empty.
    """
    if not boxes:
        return 0
    if len(boxes) == 1:
        return 0

    h, w = crop_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def rank(index):
        box = boxes[index]
        if box is None:
            return (float("inf"), 0.0)
        x1, y1, x2, y2 = box
        face_cx = (x1 + x2) / 2.0
        face_cy = (y1 + y2) / 2.0
        distance = ((face_cx - cx) ** 2 + (face_cy - cy) ** 2) ** 0.5
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return (distance, -area)  # nearest first; larger area wins ties

    return min(range(len(boxes)), key=rank)


def _ascii_safe(message):
    """Strip non-ASCII from a message so printing it can never raise.

    Windows consoles default to cp1252, where print()ing an emoji raises
    UnicodeEncodeError — which would turn "this backend was skipped" into a
    crash of the whole benchmark. Used on third-party error text before it is
    handed to callers that print it.

    Inputs:
        message: anything str()-able.
    Returns:
        str: the message with un-encodable characters replaced by "?".
    """
    return str(message).encode("ascii", "replace").decode("ascii")


def _facial_area_to_box(area):
    """Convert DeepFace's facial_area dict to the [x1, y1, x2, y2] convention.

    Inputs:
        area (dict | None): {"x", "y", "w", "h"} in pixels, as returned in each
            DeepFace.represent() result.
    Returns:
        list[float] | None: [x1, y1, x2, y2], or None if the dict is missing or
        malformed (the "skip" backend reports no meaningful region).
    """
    if not isinstance(area, dict):
        return None
    try:
        x, y = float(area["x"]), float(area["y"])
        w, h = float(area["w"]), float(area["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return [x, y, x + w, y + h]


# ── Reference-crop I/O ───────────────────────────────────────────
# Shared by benchmark_recognition.py (which scores encoders on these crops)
# and build_gallery.py (which ships them to the live system). Both must read
# the SAME folders by the SAME rules and embed them the SAME way, or the
# threshold the benchmark recommends would not describe the gallery that gets
# deployed. Keeping one copy here is what guarantees that.

def load_reference_crops(reference_dir):
    """Load the verified reference crops, grouped by person.

    Only real identity folders are read: entries whose name starts with "_"
    (e.g. _noise/, _rejected/) or "." are skipped, as are loose files like
    manifest.json. Inside each person folder, files starting with "_" (the
    _face_card.jpg medoid thumbnail) are skipped too.

    Inputs:
        reference_dir (Path): data/reference_faces.
    Returns:
        dict[str, list[tuple[str, np.ndarray]]]: person -> [(filename, bgr), ...],
        persons and files both sorted for determinism. Unreadable images skipped.
    """
    persons = {}
    for person_dir in sorted(p for p in reference_dir.iterdir() if p.is_dir()):
        if person_dir.name.startswith(("_", ".")):
            continue
        crops = []
        for path in sorted(person_dir.iterdir()):
            if not path.is_file() or path.name.startswith("_"):
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(path))
            if image is None:
                print(f"    ! unreadable, skipping: {path.name}")
                continue
            crops.append((path.name, image))
        if crops:
            persons[person_dir.name] = crops
    return persons


def embed_crops(encoder, images):
    """Embed a list of BGR crops, timing each embed() call and dropping failures.

    Only encoder.embed() is timed (with perf_counter), so latency reflects
    inference alone. A crop the encoder chokes on is counted, not fatal.

    Inputs:
        encoder: an encoder from load_encoder (.embed(bgr) -> np.ndarray).
        images (list[np.ndarray]): BGR crops.
    Returns:
        tuple (kept_indices, embeddings, latencies_ms, failures):
            kept_indices (list[int]): input positions that embedded, in order.
            embeddings (np.ndarray): (len(kept), D), L2-NORMALIZED (empty (0,0)).
            latencies_ms (list[float]): one per successful embed.
            failures (int): crops that raised.
    """
    vectors, kept, latencies, failures = [], [], [], 0
    for index, image in enumerate(images):
        t0 = time.perf_counter()
        try:
            vector = encoder.embed(image)
        except Exception:  # a crop this model simply cannot handle
            failures += 1
            continue
        latencies.append((time.perf_counter() - t0) * 1000.0)
        vectors.append(vector)
        kept.append(index)

    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32), latencies, failures
    return kept, l2_normalize(np.vstack(vectors)), latencies, failures


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
