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

from flask import Blueprint, abort, redirect, render_template, send_from_directory, url_for

import engineering_data as data

# tools/ holds the locked operating points the live pipeline reads.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tools"))
_RESULTS_DIR = _REPO_ROOT / "results"

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
        "id": "enrollment",
        "title": "Enrollment curation",
        "question": "How does a flat bucket of photos become a named gallery "
                    "without anyone labelling a face?",
        "status": "measured",
        "endpoint": "engineering.enrollment",
        "summary": "Detect every face, embed all of them, and let one HDBSCAN "
                   "pass discover the identities. The only manual step is "
                   "renaming the cluster folders it produces.",
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
    """The build log: the judgment calls, the caveats, what got locked in.

    MEASURED WORK LEADS, DELIBERATELY. The ambition is the last page in the
    nav, not the first, because the same content lands differently depending
    on what a reader has already seen. Opened with, it reads as a pitch;
    reached after the detector benchmark, the encoder race and the rejected
    caching result, it reads as where the evidence is heading. See
    architecture().
    """
    return render_template(
        "engineering/index.html",
        stages=STAGES,
        detection_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        detection_detector=DETECTION_DETECTOR,
        provenance=DETECTION_THRESHOLD_PROVENANCE,
        **data.overview_page_data(),
    )


@engineering.route("/architecture")
def architecture():
    """The last page: the system this is all being built toward.

    Everything here is argued rather than measured, which is why it sits at
    the end of the log and says so at the top. It earns its ambition from the
    pages before it, not from its own prose.
    """
    return render_template(
        "engineering/architecture.html",
        stages=STAGES,
        detection_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        detection_detector=DETECTION_DETECTOR,
        **data.direction_page_data(),
    )


# Both of these were real URLs for part of a day while the ordering was being
# worked out. Redirect rather than 404 — a link that rots is worse than a
# redirect that lingers.
@engineering.route("/decisions")
def decisions():
    return redirect(url_for("engineering.index"), code=301)


@engineering.route("/direction")
def direction():
    return redirect(url_for("engineering.architecture"), code=301)


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


@engineering.route("/enrollment")
def enrollment():
    """Enrollment curation: how the unlabelled photo bucket became a gallery."""
    return render_template(
        "engineering/enrollment.html",
        stages=STAGES,
        **data.curation_page_data(),
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


# Figures live in results/ next to the JSON they are drawn from, not in
# static/, so a benchmark artifact and its plot stay in one place and cannot
# drift apart. This serves them read-only, by explicit allow-list — the
# directory also holds raw artifacts, and a page should not be able to reach
# for an arbitrary file in it just by naming one.
FIGURES = {
    "recognition_roc.svg",
    "recognition_roc.png",
    "recognition_pr.svg",
    "recognition_pr.png",
}


@engineering.route("/figure/<path:name>")
def figure(name):
    """Serve one generated benchmark figure out of results/."""
    if name not in FIGURES:
        abort(404)
    if not (_RESULTS_DIR / name).exists():
        abort(404)  # not plotted yet; the page hides the section in that case
    return send_from_directory(_RESULTS_DIR, name, max_age=0)
