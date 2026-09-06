"""
Test suite for the engineering log's build gate (tools/freeze_engineering.py).

WHY THIS FILE EXISTS. preflight() is the thing that decides whether a build is
fit to publish, and until now nothing tested it. That is exactly backwards: it
is the last check before the site goes public, and its failure mode is silence
— every gap in it publishes a page that looks fine and says nothing. The
enrollment page did that for months.

The tests are about BEHAVIOUR, not the mechanism: "an absent artifact fails the
build", "an undeclared page fails the build". They should survive a rewrite of
how the declarations are stored.

Nothing here writes to results/ or renders a site; preflight is called directly
against a freshly built app.

Run:
  pytest tests/test_freeze_engineering.py
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "demo"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import freeze_engineering as fz  # noqa: E402  (import follows the sys.path tweak)


@pytest.fixture
def build():
    """A fresh app plus its modules, one per test.

    Fresh because several tests add a route, and Flask refuses to accept new
    routes on an app that has already handled a request.
    """
    return fz.make_app()


def _problems(build, **_):
    app, eng, data = build
    return fz.preflight(app, eng, data)


# ══════════════════════════════════════════════════════════════════
# The healthy case.
# ══════════════════════════════════════════════════════════════════

def test_a_complete_checkout_reports_no_problems(build):
    """If this fails, one of the committed artifacts is missing or empty."""
    assert _problems(build) == []


# ══════════════════════════════════════════════════════════════════
# Declarations are mandatory — the systemic fix.
# ══════════════════════════════════════════════════════════════════

def test_a_page_with_no_declaration_is_a_build_error(build):
    """The regression the inversion exists to prevent.

    A new page used to be checked only if someone remembered to add a line to
    preflight. Now the absence of a declaration is itself the failure, so
    forgetting is loud and immediate instead of silent and permanent.
    """
    app, eng, data = build

    @app.route("/engineering/undeclared", endpoint="engineering.undeclared")
    def _undeclared():
        return "hi"

    problems = fz.preflight(app, eng, data)
    assert any("declares no data requirements" in p for p in problems)
    assert any("engineering.undeclared" in p for p in problems)


def test_a_declared_page_with_everything_present_passes(build):
    app, eng, data = build

    @app.route("/engineering/declared", endpoint="engineering.declared")
    @eng.requires(nothing="test route that renders no data")
    def _declared():
        return "hi"

    assert fz.preflight(app, eng, data) == []


def test_requires_refuses_an_empty_declaration(build):
    """'I need nothing' has to be stated, not implied by an empty call."""
    _, eng, _ = build
    with pytest.raises(ValueError, match="meaningless"):
        eng.requires()


def test_an_unknown_data_check_name_is_reported(build):
    """A typo in a declaration must not silently check nothing."""
    app, eng, data = build

    @app.route("/engineering/typo", endpoint="engineering.typo")
    @eng.requires(data_checks=["detectoin_threshold"])   # deliberate typo
    def _typo():
        return "hi"

    problems = fz.preflight(app, eng, data)
    assert any("unknown data check" in p for p in problems)


# ══════════════════════════════════════════════════════════════════
# Artifact requirements.
# ══════════════════════════════════════════════════════════════════

def test_a_missing_declared_artifact_fails_the_build(build, tmp_path, monkeypatch):
    app, eng, data = build
    monkeypatch.setattr(fz, "RESULTS_DIR", tmp_path)   # an empty results/

    problems = fz.preflight(app, eng, data)
    assert any("missing artifact: results/enrollment_summary.json" in p
               for p in problems)
    # and the message says who wanted it, so the fix is obvious
    assert any("engineering.enrollment" in p for p in problems
               if "enrollment_summary.json" in p)


def test_an_empty_declared_artifact_fails_the_build(build, tmp_path, monkeypatch):
    """A zero-byte file exists but describes nothing."""
    app, eng, data = build
    (tmp_path / "enrollment_summary.json").write_text("")
    monkeypatch.setattr(fz, "RESULTS_DIR", tmp_path)

    problems = fz.preflight(app, eng, data)
    assert any("empty artifact: results/enrollment_summary.json" in p
               for p in problems)


def test_an_artifact_no_page_declares_is_reported(build, monkeypatch):
    """Keeps the declarations honest: a required artifact nobody claims is
    either dead weight or a page that forgot to declare it."""
    app, eng, data = build
    monkeypatch.setattr(fz, "REQUIRED_ARTIFACTS",
                        fz.REQUIRED_ARTIFACTS + ("benchmarks_nobody_wants.json",))

    problems = fz.preflight(app, eng, data)
    assert any("is required by no page" in p for p in problems)


# ══════════════════════════════════════════════════════════════════
# Shaped-data checks — the ones a file's existence cannot express.
# ══════════════════════════════════════════════════════════════════

def test_a_blank_detection_threshold_fails_the_build(build, monkeypatch):
    app, eng, data = build
    monkeypatch.setattr(eng, "DETECTION_CONFIDENCE_THRESHOLD", None)

    problems = fz.preflight(app, eng, data)
    assert any("pipeline_config did not import" in p for p in problems)


def test_an_unavailable_presence_config_fails_the_build(build, monkeypatch):
    """The presence page's N/M/T caption would render as 'unavailable'."""
    app, eng, data = build
    monkeypatch.setattr(eng, "PRESENCE_CONFIG", None)

    problems = fz.preflight(app, eng, data)
    assert any("presence_config did not import" in p for p in problems)


def test_the_presence_stage_may_not_claim_to_be_measured(build, monkeypatch):
    """It runs, but only from a video file, and N/M/T rest on an argument.

    Every other stage carrying "measured" earned it from a recorded benchmark.
    This guard stops that word being borrowed during a tidy-up.
    """
    app, eng, data = build
    stages = [dict(stage) for stage in eng.STAGES]
    for stage in stages:
        if stage["id"] == "presence":
            stage["status"] = "measured"
    monkeypatch.setattr(eng, "STAGES", stages)

    problems = fz.preflight(app, eng, data)
    assert any("claims status 'measured'" in p for p in problems)


def test_the_presence_stage_passes_while_it_says_preliminary(build):
    """The complement: the guard fires on the claim, not on the page existing."""
    _, eng, _ = build
    presence = next(s for s in eng.STAGES if s["id"] == "presence")
    assert presence["status"] == "preliminary"
    assert _problems(build) == []


def test_an_empty_caching_table_fails_the_build(build, monkeypatch):
    """The assistant page's heading promises six runs and a reversal."""
    app, eng, data = build
    monkeypatch.setattr(data, "assistant_page_data", lambda: {"caching": []})

    problems = fz.preflight(app, eng, data)
    assert any("no caching runs" in p for p in problems)


def test_caching_runs_that_never_engaged_fail_the_build(build, monkeypatch):
    app, eng, data = build
    monkeypatch.setattr(data, "assistant_page_data",
                        lambda: {"caching": [{"engaged": False}]})

    problems = fz.preflight(app, eng, data)
    assert any("ever engaged the cache" in p for p in problems)


def test_a_raising_data_module_is_reported_not_propagated(build, monkeypatch):
    """A documentation build should report a problem, not traceback."""
    app, eng, data = build

    def boom():
        raise RuntimeError("artifact is malformed")

    monkeypatch.setattr(data, "routing_tables", boom)
    problems = fz.preflight(app, eng, data)
    assert any("routing_tables() raised" in p for p in problems)


# ══════════════════════════════════════════════════════════════════
# Structural checks.
# ══════════════════════════════════════════════════════════════════

def test_a_missing_nav_view_fails_the_build(build, monkeypatch):
    """A nav entry pointing at a missing endpoint 500s every page, not one."""
    app, eng, data = build
    monkeypatch.setattr(fz, "EXPECTED_PAGES", fz.EXPECTED_PAGES + ("nonexistent",))

    problems = fz.preflight(app, eng, data)
    assert any("missing expected view functions" in p for p in problems)


def test_a_workstream_without_stages_fails_the_build(build, monkeypatch):
    """Grouping is the log's argument; an empty workstream is a content failure
    the renderer cannot see."""
    app, eng, data = build
    monkeypatch.setattr(eng, "WORKSTREAMS", [{"id": "hollow", "stages": []}])

    problems = fz.preflight(app, eng, data)
    assert any("has no stages" in p for p in problems)


def test_every_route_in_the_blueprint_carries_a_declaration(build):
    """The inventory check: no route ships undeclared, today or after a merge."""
    app, _, _ = build
    undeclared = [
        rule.endpoint for rule in app.url_map.iter_rules()
        if rule.endpoint.startswith("engineering.")
        and getattr(app.view_functions.get(rule.endpoint), "eng_requires", None) is None
    ]
    assert undeclared == []
