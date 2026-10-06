"""
JARVIS — CPU Detection Speed Benchmark
======================================
Measures how fast a YOLO detection model runs on this machine's CPU. It is the
BASELINE for an optimization series (PyTorch -> ONNX -> OpenVINO -> INT8), so
every choice below is about making two runs comparable, not about making one
run look fast.

    fixed VisDrone val subset  ->  decode all images into RAM
        ->  warm up on the first 10
        ->  per image: time preprocess | inference | postprocess | end-to-end
        ->  repeat the whole pass 3x  ->  per-repeat stats + spread across repeats

FAIRNESS RULES
  * Same images every run: the first N VisDrone val images SORTED BY FILENAME.
    The entry records the first/last name and a hash of the whole name list, so
    two stored runs can be checked to have measured the same set.
  * Images are decoded before timing starts. Disk I/O is not part of model speed
    and on a OneDrive-synced folder it is the noisiest thing on the machine.
  * Warm-up (default 10 images) runs at the start of EVERY repeat and is never
    timed: first calls pay for allocator growth, lazy init and cold caches.
  * End-to-end is measured as its own span around the three stages, not as
    their sum, so it includes the glue between them.
  * Repeats exist because laptop timing drifts with load and heat — ArcFace
    latency drifted 106 -> 188 ms across identical runs in the recognition
    benchmark. A single pass is an anecdote. The headline is the MEDIAN of the
    per-repeat means, and the spread (max-min over median) says how much to
    trust it. Treat a delta between two configurations that is smaller than
    their spread as noise.
  * Power state is recorded where psutil can see a battery. On battery, Windows
    power plans throttle the CPU; do not compare a plugged-in run with an
    unplugged one.
  * The input tensor shape is recorded. Ultralytics' PyTorch path letterboxes
    to the MINIMUM stride-aligned rectangle (a 1360x765 VisDrone frame runs at
    640x384), while a static-shape ONNX/OpenVINO export pads to 640x640 — ~1.7x
    the pixels. Compare backends only at matching input shapes, or export with
    dynamic shapes.

BACKENDS
  A backend is a class with load() / preprocess() / infer() / postprocess() and
  is registered in BACKENDS. "pytorch", "onnx", "openvino" and
  "openvino-int8" all subclass UltralyticsBackend: AutoBackend picks the
  runtime from the weights (.pt, .onnx, *_openvino_model/), so preprocess and
  postprocess are the SAME code for all of them and only infer() differs. That
  is the point — a speedup in the table is then attributable to the runtime
  (and, for int8, the precision) alone. Everything else — timing, stats,
  persistence — is backend-agnostic and must stay that way. Each entry records
  its precision under runtime.precision.

  Exports are static-shape. Pass the shape they were exported at as
  --imgsz H,W (e.g. 384,640) so the letterbox produces exactly that tensor; the
  recorded input_shapes are the check that it did.

Usage:
    python tools/benchmark_speed.py --model data/models/yolov8n_visdrone_640_e50_s42.pt --conf 0.172
    python tools/benchmark_speed.py --model data/models/yolov8n_visdrone_640_e50_s42.onnx \
        --backend onnx --imgsz 384,640 --conf 0.172
    python tools/benchmark_speed.py --model data/models/yolov8n_visdrone_640_e50_s42_openvino_model \
        --backend openvino --imgsz 384,640 --conf 0.172
    python tools/benchmark_speed.py --model tools/weights/yolov8n-face.pt --images 50

Results are appended to results/benchmarks_speed.json.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "benchmarks_speed.json"
VISDRONE_DIR = REPO_ROOT / "data" / "visdrone"
VISDRONE_VAL_IMAGES = VISDRONE_DIR / "VisDrone2019-DET-val" / "images"
# The same archive Ultralytics' VisDrone.yaml downloads; val only (~80 MB).
VISDRONE_VAL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
STAGES = ("preprocess", "inference", "postprocess", "end_to_end")

DEFAULT_IMAGES = 100
DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 3
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25          # Ultralytics' own default


# ════════════════════════════════════════════════════════════════════
# Stats — pure, unit-tested (tests/test_benchmark_speed.py).
# ════════════════════════════════════════════════════════════════════

def percentile(sorted_vals, pct):
    """Nearest-rank percentile of an already-sorted list (0.0 if empty).

    The same definition benchmark_detection uses, so p95s are comparable
    across the two files.
    """
    if not sorted_vals:
        return 0.0
    k = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def stage_stats(samples_ms):
    """Mean / p50 / p95 / min / max (ms) of one stage's per-image timings."""
    if not samples_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(samples_ms)
    return {
        "mean_ms": round(sum(ordered) / len(ordered), 3),
        "p50_ms": round(percentile(ordered, 50), 3),
        "p95_ms": round(percentile(ordered, 95), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def fps_from_mean(mean_ms):
    """Throughput implied by a mean end-to-end latency: 1000 / mean_ms."""
    return round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0


def summarize_repeat(timings, detections):
    """Stats for ONE full pass over the image set.

    Inputs:
        timings (dict[str, list[float]]): per-image ms for every name in STAGES.
        detections (list[int]): detections per image, same order.
    Returns:
        dict: {"num_images", "stages": {stage: stage_stats}, "fps",
               "mean_detections"}. fps is from the END-TO-END mean.
    """
    missing = [s for s in STAGES if s not in timings]
    if missing:
        raise ValueError(f"timings missing stages: {missing}")
    n = len(timings["end_to_end"])
    if any(len(timings[s]) != n for s in STAGES) or len(detections) != n:
        raise ValueError("every stage and the detection list must cover the same images")
    stages = {s: stage_stats(timings[s]) for s in STAGES}
    return {
        "num_images": n,
        "stages": stages,
        "fps": fps_from_mean(stages["end_to_end"]["mean_ms"]),
        "mean_detections": round(sum(detections) / n, 2) if n else 0.0,
    }


def spread(values):
    """How far a metric moved across repeats.

    Returns median / min / max and spread_pct = (max - min) / median * 100. The
    median, not the mean, is the headline: one repeat that caught a background
    task should not drag the reported number.
    """
    if not values:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "spread_pct": 0.0}
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    return {
        "median": round(med, 3),
        "min": round(lo, 3),
        "max": round(hi, 3),
        "spread_pct": round((hi - lo) / med * 100.0, 2) if med > 0 else 0.0,
    }


def summarize_repeats(repeats):
    """Collapse per-repeat summaries into the cross-repeat headline.

    For each stage, the mean / p50 / p95 of each repeat are put through
    spread(). fps is DERIVED from the median end-to-end mean (1000 / median)
    rather than being the median of per-repeat fps, so the headline fps and the
    headline latency always agree exactly.
    """
    if not repeats:
        raise ValueError("no repeats to summarize")
    stages = {}
    for s in STAGES:
        stages[s] = {
            key: spread([r["stages"][s][f"{key}_ms"] for r in repeats])
            for key in ("mean", "p50", "p95")
        }
    e2e_median = stages["end_to_end"]["mean"]["median"]
    return {
        "stages": stages,
        "fps": fps_from_mean(e2e_median),
        "fps_range": [min(r["fps"] for r in repeats), max(r["fps"] for r in repeats)],
        # Identical across repeats for a deterministic model; a mismatch means
        # the runs did not measure the same work.
        "mean_detections": statistics.median(r["mean_detections"] for r in repeats),
        "detections_consistent": len({r["mean_detections"] for r in repeats}) == 1,
    }


# ════════════════════════════════════════════════════════════════════
# Result record — pure, unit-tested.
# ════════════════════════════════════════════════════════════════════

def names_digest(names):
    """sha256 over the newline-joined image names: identifies the test set."""
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def build_entry(*, timestamp, backend, model, imgsz, conf, image_names, warmup,
                environment, repeats, input_shapes, runtime=None):
    """One appendable record for results/benchmarks_speed.json.

    Top-level shape matches the other benchmark files: a dict with a
    "timestamp", every setting that changes what the numbers MEAN, a "results"
    list (here one element per repeat, in run order) and a "summary".
    """
    return {
        "timestamp": timestamp,
        "backend": backend,
        "model": model,
        "imgsz": imgsz,
        "conf": conf,
        "dataset": {
            "name": "visdrone-val",
            "selection": "first N by sorted filename",
            "num_images": len(image_names),
            "first": image_names[0] if image_names else None,
            "last": image_names[-1] if image_names else None,
            "names_sha256": names_digest(image_names),
        },
        "warmup_images": warmup,
        "num_repeats": len(repeats),
        "input_shapes": input_shapes,
        # Backend-specific execution settings (threads, providers, perf hint):
        # torch_threads in environment says nothing about ORT or OpenVINO.
        "runtime": runtime or {},
        "environment": environment,
        "results": repeats,
        "summary": summarize_repeats(repeats),
    }


def append_entry(entry, path=RESULTS_FILE):
    """Append to the JSON list at `path`, creating it if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a JSON list; refusing to overwrite it")
    data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# ════════════════════════════════════════════════════════════════════
# Environment
# ════════════════════════════════════════════════════════════════════

def cpu_model():
    """Human CPU name. platform.processor() on Windows is only a family string."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _version(module_name):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except ImportError:
        return None


def power_state():
    """{"plugged_in": bool|None, "battery_percent": float|None}; None = not detectable."""
    try:
        import psutil
        battery = psutil.sensors_battery()
    except Exception:
        battery = None
    if battery is None:
        return {"plugged_in": None, "battery_percent": None}
    return {"plugged_in": bool(battery.power_plugged), "battery_percent": battery.percent}


def collect_environment():
    import torch
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        physical = None
    return {
        "cpu": cpu_model(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": physical,
        "torch_threads": torch.get_num_threads(),
        "power": power_state(),
        "versions": {
            "torch": _version("torch"),
            "ultralytics": _version("ultralytics"),
            "onnxruntime": _version("onnxruntime"),
            "openvino": _version("openvino"),
            "numpy": _version("numpy"),
        },
    }


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_fingerprint(path):
    """{"sha256", "size_bytes"} for a weights file OR a model directory.

    An OpenVINO export is a directory (.xml + .bin + metadata.yaml). Its hash is
    over every file's relative name and contents, in sorted order, so it changes
    if any file is added, renamed or edited.
    """
    path = Path(path)
    if path.is_file():
        return {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    h, size = hashlib.sha256(), 0
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode("utf-8") + b"\0")
        h.update(file_sha256(f).encode("ascii"))
        size += f.stat().st_size
    return {"sha256": h.hexdigest(), "size_bytes": size}


def parse_imgsz(text):
    """"640" -> 640 (square, Ultralytics' default); "384,640" -> [384, 640] (H, W)."""
    parts = [int(p) for p in str(text).replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1 and parts[0] > 0:
        return parts[0]
    if len(parts) == 2 and all(p > 0 for p in parts):
        return parts
    raise ValueError(f"imgsz must be N or H,W; got {text!r}")


def display_path(path):
    """Repo-relative POSIX path when inside the repo, else absolute."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# ════════════════════════════════════════════════════════════════════
# Test set
# ════════════════════════════════════════════════════════════════════

def ensure_visdrone_val(images_dir=VISDRONE_VAL_IMAGES):
    """Download and unpack ONLY the VisDrone val split if it is not present."""
    if images_dir.is_dir() and any(images_dir.iterdir()):
        return images_dir
    import urllib.request

    VISDRONE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = VISDRONE_DIR / "VisDrone2019-DET-val.zip"
    if not zip_path.exists():
        print(f"  VisDrone val not found - downloading {VISDRONE_VAL_URL}")
        tmp = zip_path.with_suffix(".zip.part")
        urllib.request.urlretrieve(VISDRONE_VAL_URL, tmp)
        tmp.replace(zip_path)
    print(f"  Extracting {zip_path.name} -> {VISDRONE_DIR}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(VISDRONE_DIR)
    zip_path.unlink()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"archive did not contain {images_dir}")
    return images_dir


def select_images(images_dir, n):
    """The first n images by filename. Sorted, never sampled: same set every run."""
    paths = sorted((p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS),
                   key=lambda p: p.name)
    if len(paths) < n:
        raise ValueError(f"asked for {n} images, {images_dir} has {len(paths)}")
    return paths[:n]


# ════════════════════════════════════════════════════════════════════
# Backends
# ════════════════════════════════════════════════════════════════════

class Backend:
    """Interface every backend implements. Each stage is timed separately.

    preprocess(img_bgr) -> x        decoded image -> model input
    infer(x)            -> raw      the forward pass only
    postprocess(raw, x, img_bgr) -> int   decode + NMS; returns detection count
    input_shape(x)      -> list     shape of the model input, for the record
    """

    name = "abstract"
    precision = "fp32"

    def load(self, model_path, imgsz, conf):
        raise NotImplementedError

    def preprocess(self, img):
        raise NotImplementedError

    def infer(self, x):
        raise NotImplementedError

    def postprocess(self, raw, x, img):
        raise NotImplementedError

    def input_shape(self, x):
        return list(getattr(x, "shape", []))

    def runtime_info(self):
        """Execution settings worth recording (threads, providers...). Optional."""
        return {}


class UltralyticsBackend(Backend):
    """Ultralytics' own predictor, driven one stage at a time.

    These are the exact stage methods model.predict() calls (and that its
    built-in speed= dict times), so the numbers mean what Ultralytics users
    expect. One real predict() call sets the predictor up; after that the
    stages are invoked directly. AutoBackend picks the runtime from the
    weights, so a subclass only declares its `name` and which weights it
    accepts — checked up front, so `--backend onnx --model x.pt` cannot
    silently benchmark PyTorch under the wrong label.
    """

    name = "pytorch"
    device = "cpu"

    @staticmethod
    def accepts(model_path):
        return Path(model_path).suffix == ".pt"

    def load(self, model_path, imgsz, conf):
        import numpy as np
        import torch
        from ultralytics import YOLO

        if not self.accepts(model_path):
            raise ValueError(f"{self.name} backend cannot load {model_path}")
        self._torch = torch
        self.model = YOLO(str(model_path), task="detect")
        h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=imgsz, conf=conf, device=self.device, verbose=False)
        self.predictor = self.model.predictor
        self.check_cpu_only()

    def check_cpu_only(self):
        """Raise if the runtime could execute anywhere but the CPU."""

    def runtime_info(self):
        return {"torch_threads": self._torch.get_num_threads()}

    def preprocess(self, img):
        # construct_results reads image paths from predictor.batch.
        self.predictor.batch = (["image"], [img], [""])
        return self.predictor.preprocess([img])

    def infer(self, x):
        with self._torch.inference_mode():
            return self.predictor.inference(x)

    def postprocess(self, raw, x, img):
        with self._torch.inference_mode():
            results = self.predictor.postprocess(raw, x, [img])
        return len(results[0].boxes)


class OnnxBackend(UltralyticsBackend):
    """ONNX Runtime, CPUExecutionProvider, via AutoBackend."""

    name = "onnx"

    @staticmethod
    def accepts(model_path):
        return Path(model_path).suffix == ".onnx"

    def check_cpu_only(self):
        providers = self.predictor.model.backend.session.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError(f"ONNX Runtime providers are {providers}, not CPU only")

    def runtime_info(self):
        import onnxruntime

        session = self.predictor.model.backend.session
        opts = session.get_session_options()
        return {
            "onnxruntime": onnxruntime.__version__,
            "providers": session.get_providers(),
            # 0 = ORT's default: one intra-op thread per physical core.
            "intra_op_num_threads": opts.intra_op_num_threads,
            "graph_optimization_level": str(opts.graph_optimization_level),
        }


class OpenVINOBackend(UltralyticsBackend):
    """OpenVINO CPU plugin, via AutoBackend (LATENCY hint at batch 1).

    PINNED TO CPU. Left to itself, Ultralytics compiles on OpenVINO's AUTO
    device whenever anything besides the CPU is visible — and on this laptop
    OpenVINO sees the RTX 3050 and the Radeon iGPU through OpenCL. AUTO starts
    on the CPU and may hand inference to a GPU once that compile finishes, which
    would make this a GPU number labelled CPU. "intel:cpu" compiles straight on
    the CPU plugin, and check_cpu_only() refuses to benchmark if it did not.
    """

    name = "openvino"
    device = "intel:cpu"

    def check_cpu_only(self):
        devices = list(self.predictor.model.backend.ov_compiled_model.get_property("EXECUTION_DEVICES"))
        if devices != ["CPU"]:
            raise RuntimeError(f"OpenVINO is executing on {devices}, not CPU only")

    @staticmethod
    def accepts(model_path):
        return is_openvino_dir(model_path) and not is_quantized_ir(model_path)

    def conv_precisions(self):
        """{runtime precision: count} over the convolutions the CPU plugin compiled.

        The compiled graph, not the IR: this is what actually executes. The
        runtime-model references are dropped before returning — holding them
        past the compiled model's lifetime segfaults the interpreter at exit.
        """
        runtime = self.predictor.model.backend.ov_compiled_model.get_runtime_model()
        counts = {}
        for op in runtime.get_ordered_ops():
            rt = op.get_rt_info()
            if "layerType" in rt and rt["layerType"].astype(str) == "Convolution":
                prec = rt["runtimePrecision"].astype(str)
                counts[prec] = counts.get(prec, 0) + 1
            del rt, op
        del runtime
        return counts

    def runtime_info(self):
        import openvino

        compiled = self.predictor.model.backend.ov_compiled_model
        info = {"openvino": openvino.__version__, "conv_precisions": self.conv_precisions()}
        for prop in ("PERFORMANCE_HINT", "INFERENCE_NUM_THREADS", "NUM_STREAMS",
                     "INFERENCE_PRECISION_HINT", "EXECUTION_DEVICES"):
            try:
                value = compiled.get_property(prop)
                info[prop.lower()] = value if isinstance(value, (int, float, list)) else str(value)
            except Exception:
                pass
        return info


class OpenVINOInt8Backend(OpenVINOBackend):
    """OpenVINO INT8 IR (tools/export_int8.py), same CPU pinning as FP32.

    Told apart from FP32 by the IR's CONTENTS (FakeQuantize ops), not its
    folder name, so a renamed directory cannot be benchmarked under the wrong
    precision. check_cpu_only() additionally refuses to run unless every
    convolution the CPU plugin compiled executes in an 8-bit integer type.
    """

    name = "openvino-int8"
    precision = "int8"

    @staticmethod
    def accepts(model_path):
        return is_openvino_dir(model_path) and is_quantized_ir(model_path)

    def check_cpu_only(self):
        super().check_cpu_only()
        precisions = self.conv_precisions()
        if not precisions or set(precisions) - {"u8", "i8"}:
            raise RuntimeError(f"INT8 IR compiled with non-int8 convolutions: {precisions}")


def is_openvino_dir(model_path):
    p = Path(model_path)
    return p.is_dir() and any(p.glob("*.xml"))


def is_quantized_ir(model_path):
    """True when the OpenVINO IR holds FakeQuantize ops (an NNCF-quantized model)."""
    return any('type="FakeQuantize"' in x.read_text(encoding="utf-8", errors="ignore")
               for x in Path(model_path).glob("*.xml"))


BACKENDS = {
    "pytorch": UltralyticsBackend,
    "onnx": OnnxBackend,
    "openvino": OpenVINOBackend,
    "openvino-int8": OpenVINOInt8Backend,
}


# ════════════════════════════════════════════════════════════════════
# Timing
# ════════════════════════════════════════════════════════════════════

def time_one(backend, img):
    """Run one image through all stages; return (stage_ms dict, n_dets, x)."""
    clock = time.perf_counter
    t0 = clock()
    x = backend.preprocess(img)
    t1 = clock()
    raw = backend.infer(x)
    t2 = clock()
    n = backend.postprocess(raw, x, img)
    t3 = clock()
    return ({"preprocess": (t1 - t0) * 1000.0,
             "inference": (t2 - t1) * 1000.0,
             "postprocess": (t3 - t2) * 1000.0,
             "end_to_end": (t3 - t0) * 1000.0}, n, x)


def run_repeat(backend, images, warmup):
    """Warm up on the first `warmup` images, then time every image once."""
    for img in images[:warmup]:
        time_one(backend, img)
    timings = {s: [] for s in STAGES}
    detections, shapes = [], {}
    for img in images:
        stage_ms, n, x = time_one(backend, img)
        for s in STAGES:
            timings[s].append(stage_ms[s])
        detections.append(n)
        key = "x".join(str(d) for d in backend.input_shape(x))
        shapes[key] = shapes.get(key, 0) + 1
    return summarize_repeat(timings, detections), shapes


# ════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════

def format_table(entry):
    """Readable summary of one entry, as a string."""
    s = entry["summary"]
    env = entry["environment"]
    power = env["power"]["plugged_in"]
    power_txt = {True: "plugged in", False: "ON BATTERY", None: "unknown"}[power]
    shapes = ", ".join(f"{k} ({v})" for k, v in entry["input_shapes"].items())
    lines = [
        "",
        f"Speed benchmark — {entry['backend']} — {entry['model']['path']}",
        f"  {env['cpu']} | {env['torch_threads']} torch threads "
        f"({env['physical_cores']} cores / {env['logical_cpus']} logical) | power: {power_txt}",
        f"  imgsz {entry['imgsz']}, conf {entry['conf']}, {entry['dataset']['num_images']} VisDrone val "
        f"images x {entry['num_repeats']} repeats (warm-up {entry['warmup_images']})",
        f"  input tensors: {shapes}",
    ]
    runtime = entry.get("runtime") or {}
    if runtime:
        lines.append("  runtime: " + ", ".join(f"{k}={v}" for k, v in runtime.items()))
    lines += [
        "",
        "  Median of per-repeat values (spread = (max-min)/median across repeats), ms",
        f"  {'stage':<12} {'mean':>9} {'p50':>9} {'p95':>9}   {'mean range':>17} {'spread':>7}",
        "  " + "-" * 70,
    ]
    for stage in STAGES:
        st = s["stages"][stage]
        rng = f"{st['mean']['min']:.2f}-{st['mean']['max']:.2f}"
        lines.append(
            f"  {stage:<12} {st['mean']['median']:>9.2f} {st['p50']['median']:>9.2f} "
            f"{st['p95']['median']:>9.2f}   {rng:>17} {st['mean']['spread_pct']:>6.1f}%"
        )
    lines += [
        "  " + "-" * 70,
        f"  FPS (1000 / median end-to-end mean): {s['fps']:.2f}   "
        f"[per-repeat range {s['fps_range'][0]:.2f}-{s['fps_range'][1]:.2f}]",
        f"  Detections per image (mean): {s['mean_detections']}"
        + ("" if s["detections_consistent"] else "   WARNING: differed between repeats"),
        "",
        "  Per repeat:  " + "   ".join(
            f"#{i + 1} e2e {r['stages']['end_to_end']['mean_ms']:.2f} ms / {r['fps']:.2f} fps"
            for i, r in enumerate(entry["results"])),
        "",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CPU speed benchmark for YOLO detection models.")
    p.add_argument("--model", required=True, type=Path,
                   help="YOLO weights: .pt, .onnx, or an *_openvino_model/ directory")
    p.add_argument("--backend", default="pytorch", choices=sorted(BACKENDS))
    p.add_argument("--imgsz", type=parse_imgsz, default=DEFAULT_IMGSZ,
                   help="N (square) or H,W — use the export shape for static exports")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--images", type=int, default=DEFAULT_IMAGES, help="size of the fixed VisDrone val subset")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="untimed images at the start of each repeat")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--images-dir", type=Path, default=VISDRONE_VAL_IMAGES)
    p.add_argument("--no-save", action="store_true", help="print only; do not append to results/")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.model.exists():
        sys.exit(f"model not found: {args.model}")
    if args.images < 1 or args.repeats < 1 or args.warmup < 0:
        sys.exit("--images and --repeats must be >= 1, --warmup >= 0")
    if not BACKENDS[args.backend].accepts(args.model):
        sys.exit(f"--backend {args.backend} cannot load {args.model}")

    import cv2

    images_dir = args.images_dir
    if images_dir == VISDRONE_VAL_IMAGES:
        ensure_visdrone_val(images_dir)
    paths = select_images(images_dir, args.images)
    print(f"  Decoding {len(paths)} images from {display_path(images_dir)} ...")
    images = [cv2.imread(str(p)) for p in paths]
    if any(img is None for img in images):
        sys.exit("an image failed to decode")

    print(f"  Loading {args.backend} backend: {display_path(args.model)}")
    backend = BACKENDS[args.backend]()
    backend.load(args.model, args.imgsz, args.conf)

    environment = collect_environment()
    if environment["power"]["plugged_in"] is False:
        print("  WARNING: running on battery — CPU may be throttled; don't compare with plugged-in runs.")

    repeats, input_shapes = [], {}
    for i in range(args.repeats):
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        summary, shapes = run_repeat(backend, images, args.warmup)
        summary["started"] = started
        repeats.append(summary)
        for k, v in shapes.items():
            input_shapes[k] = input_shapes.get(k, 0) + v
        print(f"  repeat {i + 1}/{args.repeats}: e2e mean "
              f"{summary['stages']['end_to_end']['mean_ms']:.2f} ms, {summary['fps']:.2f} fps")
    # Shapes counted per timed image across all repeats; report per-pass counts.
    input_shapes = {k: v // args.repeats for k, v in sorted(input_shapes.items())}

    entry = build_entry(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        backend=args.backend,
        model={"path": display_path(args.model), **model_fingerprint(args.model)},
        imgsz=args.imgsz,
        conf=args.conf,
        image_names=[p.name for p in paths],
        warmup=args.warmup,
        environment=environment,
        repeats=repeats,
        input_shapes=input_shapes,
        runtime={"precision": backend.precision, **backend.runtime_info()},
    )
    print(format_table(entry))
    if not args.no_save:
        append_entry(entry)
        print(f"  Appended to {display_path(RESULTS_FILE)}")


if __name__ == "__main__":
    main()
