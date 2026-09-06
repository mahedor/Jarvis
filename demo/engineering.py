"""
JARVIS Engineering Log — Routes
===============================
A Flask blueprint serving /engineering: a "how this was built" record for the
WHOLE project — the assistant that ships today, the intent routing in front of
it, the face pipeline being built for presence, and the architecture all of it
is aimed at. It is NOT part of the assistant UI — nothing here talks to Claude,
changes device state, or is meant for someone using JARVIS as an assistant.

SCOPE. This log used to be a face-pipeline record that happened to have a
routing page bolted on, which misrepresented the project: JARVIS is a
voice-driven assistant first, and the vision work is one workstream serving it.
The overview introduces the system, then the workstreams; the face stages are
the face workstream's record rather than the site's identity. See WORKSTREAMS.

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

from flask import (
    Blueprint,
    abort,
    current_app,
    redirect,
    render_template,
    send_from_directory,
    url_for,
)

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

# The presence debounce constants, read live so the diagram's caption can never
# disagree with the service. Imported from presence_config rather than
# presence_service because that module pulls in cv2/numpy and demo/ is
# stdlib-only by design — see the note at the top of presence_config.py.
#
# THE None CASE IS NOT SILENT. An unavailable config renders a visible
# "unavailable" marker in place of the caption (never an empty gap), and
# freeze_engineering.preflight() fails --strict over it. A missing caption that
# still builds clean is how the enrollment page once shipped empty; the whole
# point of reading these live is defeated if their absence looks like a design
# choice.
try:
    from presence_config import PresenceConfig

    PRESENCE_CONFIG = PresenceConfig()
except ImportError:
    PRESENCE_CONFIG = None

engineering = Blueprint(
    "engineering",
    __name__,
    url_prefix="/engineering",
    template_folder="templates",
)


# ══════════════════════════════════════════════════════════════════
# What each page needs in order to say anything.
# ══════════════════════════════════════════════════════════════════
#
# WHY THIS EXISTS. freeze_engineering used to carry a hand-written checklist of
# things to verify before publishing. It caught blank thresholds and empty
# benchmark tables — for the pages someone remembered to add. The enrollment
# page was never added, so it published an empty state for months while every
# build reported success. A checklist maintained by memory fails silently and
# fails late.
#
# Inverted: the page declares what it needs, and the freezer walks every
# registered route and enforces those declarations. A NEW PAGE WITH NO
# DECLARATION IS A BUILD ERROR, not a silent pass — that is the whole point,
# because the failure being designed out is forgetting.
#
# Declarations are DATA, not callables: names of artifacts and of named data
# predicates the freezer knows how to run. That keeps build-time validation
# logic out of demo/ (which is stdlib-only and must stay importable without the
# tools/ dependencies) while letting the page own the statement of what it
# needs.

def requires(artifacts=(), data_checks=(), nothing=None):
    """Declare the data a view needs before its page is worth publishing.

    Inputs:
        artifacts (seq[str]): filenames under results/ that must exist and be
            non-empty.
        data_checks (seq[str]): names of predicates freeze_engineering knows
            (see DATA_CHECKS there), for requirements a file's existence cannot
            express — e.g. "a caching run must actually have engaged the cache".
        nothing (str | None): an explicit, REASONED statement that this route
            needs no data (a redirect, a file server). Required rather than
            allowing a bare @requires(), so "this needs nothing" is always a
            decision someone wrote down.
    Returns:
        The view function, tagged with `eng_requires`.
    """
    if not artifacts and not data_checks and not nothing:
        raise ValueError(
            "requires() with no arguments is meaningless. Name the artifacts or "
            "data checks the page needs, or pass nothing='<why>' to state on "
            "the record that it needs none."
        )

    def decorate(view):
        view.eng_requires = {
            "artifacts": tuple(artifacts),
            "data_checks": tuple(data_checks),
            "nothing": nothing,
        }
        return view
    return decorate

@engineering.context_processor
def _shell_context():
    """Values the shared shell needs, resolved per request rather than assumed.

    The blueprint is registered on an app it does not own, so this must not
    take anything for granted about that app.

    build_info: whatever the host app chose to record about the build, or None
        when it recorded nothing. The log renders from artifacts at request
        time, so it cannot derive this itself and must not invent it.

    There was also an assistant_url here, resolving url_for('index') to link
    the log back to the running assistant. The link was removed deliberately —
    the assistant is not something a documentation page should hand a reader a
    door into yet — and the lookup went with it rather than being left dangling
    for a template that no longer asks for it.
    """
    return {
        "build_info": current_app.config.get("ENG_BUILD_INFO"),
    }


# The project as a reader should understand it: which workstreams exist, what
# each one was asked to answer, and what got locked in. Rendered on the index as
# the spine of the log.
#
# WHY WORKSTREAMS AND NOT A FLAT STAGE LIST. The flat list put "intent routing"
# between two face-pipeline stages, which quietly claimed the assistant was a
# step in a vision pipeline. It is the other way round: the assistant is the
# product, and detection/enrollment/recognition are one workstream feeding a
# presence feature it does not have yet. Grouping is the fix.
#
# Stages without a page yet are shown as pending rather than hidden, so the log
# describes the whole design and not just the finished parts.
WORKSTREAMS = [
    {
        "id": "assistant",
        "title": "The assistant",
        "blurb": "The system that exists and runs today: speak or type, it "
                 "answers aloud and the lights change. Everything else in this "
                 "log is either in front of it or being built to feed it.",
        "stages": [
            {
                "id": "assistant",
                "title": "Assistant core",
                "question": "What does the working system actually do, and which "
                            "parts of it were chosen with evidence?",
                "status": "measured",
                "endpoint": "engineering.assistant",
                "summary": "Claude orchestration, device state as the source of "
                           "truth, browser voice I/O taken as a deliberate "
                           "stopgap, and a prompt-caching experiment that came "
                           "back negative for a reason worth writing down.",
            },
            {
                "id": "routing",
                "title": "Intent routing & latency",
                "question": "Which commands can be answered without calling an LLM at all?",
                "status": "provisional",
                "endpoint": "engineering.routing",
                "summary": "The local/cloud split and the architecture-agnostic "
                           "eval are settled; the hand-written cascade doing the "
                           "routing today is measured, flagged and up for "
                           "replacement.",
            },
        ],
    },
    {
        "id": "face",
        "title": "Face pipeline",
        "blurb": "Three stages in sequence — find a face, learn who lives here, "
                 "recognise them again — measured so a presence service can be "
                 "built on numbers instead of defaults. None of it is wired to a "
                 "live camera yet.",
        "stages": [],  # filled from FACE_STAGES below
    },
    {
        "id": "direction",
        "title": "Where it is going",
        "blurb": "The system the sensor work is for. Argued rather than "
                 "measured, and last in the log on purpose.",
        "stages": [
            {
                "id": "architecture",
                "title": "Architecture direction",
                "question": "What is all of this actually being built toward?",
                "status": "planned",
                "endpoint": "engineering.architecture",
                # Deliberately a POINTER, not a precis. Reproducing that page's
                # argument here would put the pitch first, which is the one
                # ordering this log exists to avoid — see index() and
                # architecture(). Name the subject; make the case there.
                "summary": "The system the sensor work is for. Argued from the "
                           "evidence in the pages before it, and built by "
                           "nothing yet.",
            },
        ],
    },
]

FACE_STAGES = [
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
        "id": "presence",
        "title": "Presence service",
        "question": "How does a recognized face become 'Michael is home'?",
        # PRELIMINARY, not measured, and the distinction is load-bearing: the
        # code runs and has been verified end to end from a video file, but the
        # live-camera path has never executed and the debounce constants are
        # unvalidated placeholders. Every other stage marked "measured" earned
        # it with a recorded benchmark run. This one has not, and must not
        # borrow that word — freeze_engineering.preflight() enforces it.
        "status": "preliminary",
        "endpoint": "engineering.presence",
        "summary": "Built and running against the locked thresholds, but shown "
                   "as a design rather than a result: verified only from a "
                   "video file, never a live camera, with debounce values "
                   "chosen by argument and measured by nothing.",
    },
]

# The face workstream owns its three measured stages plus the unbuilt one they
# feed. Assembled here rather than inline above so the sequence reads in one
# place.
for _workstream in WORKSTREAMS:
    if _workstream["id"] == "face":
        _workstream["stages"] = FACE_STAGES

# Flat view, in reading order, for templates that want every stage regardless of
# grouping (the overview's measurement grid, the architecture ladder). Derived
# rather than maintained separately — two lists that can disagree eventually do.
STAGES = [stage for group in WORKSTREAMS for stage in group["stages"]]


@engineering.route("/")
@requires(artifacts=["benchmarks_detection.json", "benchmarks_recognition.json"],
          data_checks=["detection_threshold"])
def index():
    """The build log: what the project is, then the judgment calls behind it.

    THE PROJECT IS INTRODUCED BEFORE ANY WORKSTREAM. A reader landing here does
    not yet know what JARVIS is, and opening on a detector comparison answers a
    question they have not been given a reason to care about.

    MEASURED WORK LEADS, DELIBERATELY. The ambition is the last page in the
    nav, not the first, because the same content lands differently depending
    on what a reader has already seen. Opened with, it reads as a pitch;
    reached after the detector benchmark, the encoder race and the rejected
    caching result, it reads as where the evidence is heading. See
    architecture().
    """
    return render_template(
        "engineering/index.html",
        workstreams=WORKSTREAMS,
        stages=STAGES,
        detection_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        detection_detector=DETECTION_DETECTOR,
        provenance=DETECTION_THRESHOLD_PROVENANCE,
        **data.overview_page_data(),
    )


@engineering.route("/assistant")
@requires(artifacts=["benchmarks_caching.json"],
          data_checks=["caching_runs_engaged"])
def assistant():
    """Phase 1: the working system every other page is in service of.

    Placed first among the workstream pages because it is the only one a reader
    can go and use. The face stages are measurements for a feature that does
    not exist yet; this is the thing that answers when you talk to it.

    Its evidence is deliberately uneven, and the page says so rather than
    levelling it up: the caching experiment and the TTS latency comparison are
    real measurements, while the voice UX and the interface were chosen by
    judgment and never measured at all.
    """
    return render_template(
        "engineering/assistant.html",
        stages=STAGES,
        **data.assistant_page_data(),
    )


@engineering.route("/architecture")
@requires(data_checks=["detection_threshold"])
def architecture():
    """The last page: the system this is all being built toward.

    Everything here is argued rather than measured, which is why it sits at
    the end of the log and says so at the top. It earns its ambition from the
    pages before it, not from its own prose.
    """
    return render_template(
        "engineering/architecture.html",
        workstreams=WORKSTREAMS,
        stages=STAGES,
        detection_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        detection_detector=DETECTION_DETECTOR,
        **data.direction_page_data(),
    )


# Both of these were real URLs for part of a day while the ordering was being
# worked out. Redirect rather than 404 — a link that rots is worse than a
# redirect that lingers.
@engineering.route("/decisions")
@requires(nothing="301 redirect to the index; renders no content of its own")
def decisions():
    return redirect(url_for("engineering.index"), code=301)


@engineering.route("/direction")
@requires(nothing="301 redirect to /architecture; renders no content of its own")
def direction():
    return redirect(url_for("engineering.architecture"), code=301)


@engineering.route("/detection")
@requires(artifacts=["benchmarks_detection.json"],
          data_checks=["detection_threshold"])
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
@requires(artifacts=["enrollment_summary.json"])
def enrollment():
    """Enrollment curation: how the unlabelled photo bucket became a gallery."""
    return render_template(
        "engineering/enrollment.html",
        stages=STAGES,
        **data.curation_page_data(),
    )


@engineering.route("/routing")
@requires(artifacts=["benchmarks_intent.json"], data_checks=["routing_tables"])
def routing():
    """Intent routing: the tiered cascade and the prompt-caching experiment."""
    return render_template(
        "engineering/routing.html",
        stages=STAGES,
        **data.routing_page_data(),
    )


@engineering.route("/presence")
@requires(data_checks=["presence_config", "presence_not_measured"])
def presence():
    """Presence service: the pipeline diagram and the topic layout.

    PRELIMINARY. Every other stage page reports a benchmark; this one reports a
    design that runs. The service works end to end against a real broker from a
    video file, but no live camera has ever driven it and the debounce
    constants have no measurement behind them. The page says so at the top
    rather than in a footnote, because a reader arriving from a log full of
    measured pages will otherwise assume this one is measured too.

    The debounce values are passed live from PresenceConfig instead of being
    written into the template or the SVG, so the caption cannot drift from the
    service the way a hardcoded number silently would.
    """
    return render_template(
        "engineering/presence.html",
        stages=STAGES,
        presence_config=PRESENCE_CONFIG,
        detection_threshold=DETECTION_CONFIDENCE_THRESHOLD,
        detection_detector=DETECTION_DETECTOR,
    )


@engineering.route("/recognition")
@requires(artifacts=["benchmarks_recognition.json"])
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
@requires(nothing="serves generated figures out of results/ by allow-list; the "
                  "freezer enumerates them and reports any that are missing")
def figure(name):
    """Serve one generated benchmark figure out of results/."""
    if name not in FIGURES:
        abort(404)
    if not (_RESULTS_DIR / name).exists():
        abort(404)  # not plotted yet; the page hides the section in that case
    return send_from_directory(_RESULTS_DIR, name, max_age=0)
