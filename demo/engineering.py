"""
JARVIS Engineering Log — Routes
===============================
A Flask blueprint serving /engineering: a "how this was built" record of how
each stage of the face pipeline was designed, measured and tuned. It is NOT
part of the assistant UI — nothing here talks to Claude, changes device state,
or is meant for someone using JARVIS as an assistant.

Registered by jarvis_web.py under url_prefix="/engineering". A blueprint rather
than more routes in jarvis_web.py because the two have nothing to say to each
other: this one reads only static artifacts from results/, holds no
conversation history, and needs no API key. Keeping it separable means the
documentation can never break the assistant.

Pages are generated from results/*.json on every request (see
engineering_data), so rerunning a benchmark is immediately reflected — there is
no build step to forget and no copy of the numbers to fall out of date.
"""

import sys
from pathlib import Path

from flask import Blueprint, render_template

import engineering_data as data

# tools/ holds the locked operating points the live pipeline reads.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tools"))

try:
    from pipeline_config import (
        DETECTION_CONFIDENCE_THRESHOLD,
        DETECTION_DETECTOR,
        DETECTION_THRESHOLD_PROVENANCE,
    )
except ImportError:  # tools/ missing or renamed — the log still renders
    DETECTION_CONFIDENCE_THRESHOLD = None
    DETECTION_DETECTOR = None
    DETECTION_THRESHOLD_PROVENANCE = None

engineering = Blueprint(
    "engineering",
    __name__,
    url_prefix="/engineering",
    template_folder="templates",
)

# The pipeline as a reader should understand it: what each stage does, what was
# compared, and what was locked in. Rendered on the index as the spine of the
# log. Stages without a page yet are shown as pending rather than hidden, so
# the log describes the whole design and not just the finished parts.
STAGES = [
    {
        "id": "detection",
        "title": "Face detection",
        "question": "Which detector finds faces in a frame, and at what confidence?",
        "status": "measured",
        "endpoint": "engineering.detection",
        "summary": "Four backends compared on three datasets: WIDER FACE, MAFA "
                   "and FDDB. Ranked by AP, with the operating threshold taken "
                   "from where F1 peaks.",
    },
    {
        "id": "recognition",
        "title": "Face recognition",
        "question": "Which encoder tells household members apart, and where is "
                    "the accept threshold?",
        "status": "measured",
        "endpoint": "engineering.recognition",
        "summary": "Encoders compared on open-set identification against LFW "
                   "strangers, with image-disjoint enroll/probe splits and "
                   "three gallery aggregation strategies.",
    },
    {
        "id": "routing",
        "title": "Intent routing & latency",
        "question": "Which commands can be answered without calling an LLM at all?",
        "status": "provisional",
        "endpoint": "engineering.routing",
        "summary": "The local/cloud split and the architecture-agnostic eval "
                   "are settled; the hand-written cascade doing the routing "
                   "today is measured, flagged and up for replacement.",
    },
    {
        "id": "presence",
        "title": "Presence service",
        "question": "How does a recognized face become 'Michael is home'?",
        "status": "pending",
        "endpoint": None,
        "summary": "Not built yet. Will consume the locked detection threshold "
                   "and the gallery produced by build_gallery.py.",
    },
]


@engineering.route("/")
def index():
    """Overview: the pipeline stages and the operating points locked so far."""
    return render_template(
        "engineering/index.html",
        stages=STAGES,
        detection_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        detection_detector=DETECTION_DETECTOR,
        provenance=DETECTION_THRESHOLD_PROVENANCE,
        **data.overview_page_data(),
    )


@engineering.route("/detection")
def detection():
    """Detection benchmark: AP, ROC AUC and latency per detector per dataset."""
    return render_template(
        "engineering/detection.html",
        stages=STAGES,
        detection_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        provenance=DETECTION_THRESHOLD_PROVENANCE,
        **data.detection_page_data(),
    )


@engineering.route("/routing")
def routing():
    """Intent routing: the tiered cascade and the prompt-caching experiment."""
    return render_template(
        "engineering/routing.html",
        stages=STAGES,
        **data.routing_page_data(),
    )


@engineering.route("/recognition")
def recognition():
    """Recognition benchmark: the encoder grid plus the threshold explorer."""
    return render_template(
        "engineering/recognition.html",
        stages=STAGES,
        primary_far=data.PRIMARY_FAR,
        **data.recognition_page_data(),
    )
