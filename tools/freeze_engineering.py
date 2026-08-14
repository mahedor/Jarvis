"""
JARVIS Engineering Log — static site freezer
============================================
Renders /engineering into a directory of flat HTML that any static host can
serve, with no Python, no Flask and no API key at the other end.

WHY THIS DOES NOT IMPORT jarvis_web. The log is a Flask BLUEPRINT precisely so
it can be lifted out of the assistant, and importing the assistant to freeze
the log would throw that away: jarvis_web constructs an Anthropic client, reads
ANTHROPIC_API_KEY, loads an intent classifier and holds conversation state,
none of which a documentation build has any business touching. So this builds
its own bare Flask app, registers the blueprint on it, and asserts afterwards
that neither jarvis_web nor anthropic ended up in sys.modules — a constraint
worth checking rather than merely intending, since it would be broken by a
single stray import in any module the blueprint touches.

WHAT COMES OUT.
    <out>/index.html                        redirect to the log
    <out>/engineering/index.html            the blueprint index
    <out>/engineering/<page>/index.html     one directory per page, so URLs
                                            stay extensionless on a static host
    <out>/engineering/figure/<name>         the generated benchmark figures
    <out>/static/...                        only the assets the HTML references

Run:
  python tools/freeze_engineering.py
  python tools/freeze_engineering.py --out dist --strict
"""

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
DEMO_DIR = REPO_ROOT / "demo"
STATIC_SRC = DEMO_DIR / "static"
RESULTS_DIR = REPO_ROOT / "results"

# The blueprint lives in demo/ and imports its siblings by bare name.
sys.path.insert(0, str(DEMO_DIR))

DEFAULT_OUT = Path("_site")

# Artifacts every page's arguments are read from. A missing one does not crash
# the log — the pages degrade to empty states by design — which is exactly why
# --strict has to check: a silently empty benchmark table is a worse published
# artifact than a failed build.
REQUIRED_ARTIFACTS = (
    "benchmarks_detection.json",
    "benchmarks_recognition.json",
    "benchmarks_intent.json",
    "benchmarks_caching.json",
)

# What the frozen log is expected to reference. Not a whitelist that blocks
# anything — the copier follows the HTML — but a tripwire: a surprise here
# means a template started pulling in something new, and the two directions
# that matters in are opposite. Something missing is a broken page; something
# extra from the ASSISTANT means the log has picked up a dependency on the app
# it is supposed to be separable from.
EXPECTED_ASSETS = {
    "css/engineering.css",
    "js/curve-explorer.js",
    "js/detection-explorer.js",
    "js/threshold-explorer.js",
}
ASSISTANT_ASSETS = ("styles.css", "chat.js", "voice-engine.js", "screensaver.js")

STATIC_REF = re.compile(r"/static/([A-Za-z0-9_\-./]+)")

ROOT_REDIRECT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="0; url=./engineering/">
<title>J.A.R.V.I.S. — Engineering Log</title>
</head>
<body>
<p>Redirecting to the <a href="./engineering/">engineering log</a>.</p>
</body>
</html>
"""


# ════════════════════════════════════════════════════════════════════
# Build metadata.
# ════════════════════════════════════════════════════════════════════

def git_commit():
    """Short HEAD sha, or None outside a git checkout.

    Returns:
        str | None: e.g. "48c3f4a". None when git is absent or this is not a
        repository — a published copy without a commit stamp is worse than no
        build at all only if you pretend to know the commit.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def artifacts_mtime():
    """Modification time of the most recently written results/*.json.

    This is the honest "how fresh is this page" number. The build timestamp
    only says when the HTML was generated; this says when the evidence behind
    it last changed, which is the thing a reader actually wants to know.

    Returns:
        str | None: ISO-8601 UTC, or None if there are no artifacts at all.
    """
    stamps = [path.stat().st_mtime for path in RESULTS_DIR.glob("*.json")]
    if not stamps:
        return None
    return datetime.fromtimestamp(max(stamps), tz=timezone.utc).isoformat(timespec="seconds")


def build_info():
    """The ENG_BUILD_INFO payload the base template renders in its footer."""
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit() or "unknown",
        "artifacts_mtime": artifacts_mtime() or "never",
    }


# ════════════════════════════════════════════════════════════════════
# The app under the freezer.
# ════════════════════════════════════════════════════════════════════

def make_app():
    """A bare Flask app with only the engineering blueprint on it.

    static_folder is absolute rather than "demo/static": Flask resolves a
    relative static_folder against the app's root_path, which for a module in
    tools/ would point at tools/demo/static and silently serve nothing.

    Returns:
        tuple (app, engineering_module, data_module).
    """
    from flask import Flask

    import engineering as engineering_module
    import engineering_data

    app = Flask(__name__, static_folder=str(STATIC_SRC))
    app.config["ENG_BUILD_INFO"] = build_info()
    app.register_blueprint(engineering_module.engineering)
    return app, engineering_module, engineering_data


def assert_assistant_never_imported():
    """The separability claim, enforced.

    Raises:
        SystemExit: if importing the blueprint dragged in the assistant or the
        Anthropic SDK. Both would still "work" here, which is what makes the
        regression easy to miss and worth failing loudly over.
    """
    leaked = [name for name in ("jarvis_web", "anthropic") if name in sys.modules]
    if leaked:
        raise SystemExit(
            f"freeze imported {', '.join(leaked)} — the engineering log is meant to "
            "render without the assistant. Something in demo/ grew an import it "
            "should not have."
        )


# ════════════════════════════════════════════════════════════════════
# Pre-flight checks (what --strict is willing to fail the build over).
# ════════════════════════════════════════════════════════════════════

def preflight(engineering_module, data_module):
    """Conditions that produce a technically-valid but hollow site.

    Each of these renders a page that looks fine and says nothing: blank
    thresholds, empty benchmark tables, a missing scale argument. None of them
    raises on its own, which is the whole reason they are checked here.

    Returns:
        list[str]: human-readable problems, empty when everything is present.
    """
    problems = []

    for name in REQUIRED_ARTIFACTS:
        if not (RESULTS_DIR / name).exists():
            problems.append(f"missing artifact: results/{name}")

    # engineering.py degrades to None when tools/pipeline_config.py cannot be
    # imported, and every locked-threshold figure on the overview renders blank.
    if engineering_module.DETECTION_CONFIDENCE_THRESHOLD is None:
        problems.append(
            "pipeline_config did not import: the locked detection threshold and "
            "its provenance table will render blank"
        )

    # Counted live from the classifier; None means the import failed and the
    # routing page loses the numbers its argument is built on.
    try:
        if data_module.routing_tables() is None:
            problems.append("engineering_data.routing_tables() returned None")
    except Exception as exc:  # a documentation build should report, not traceback
        problems.append(f"engineering_data.routing_tables() raised: {exc}")

    return problems


# ════════════════════════════════════════════════════════════════════
# Rendering.
# ════════════════════════════════════════════════════════════════════

def page_output_path(url_path, out):
    """Where a rendered page lands.

    "/engineering/"          -> <out>/engineering/index.html
    "/engineering/detection" -> <out>/engineering/detection/index.html

    The directory-per-page form keeps the published URLs identical to the live
    ones, so every link already in the templates stays correct without a
    rewrite pass.
    """
    return out / url_path.strip("/") / "index.html"


def freeze_figures(client, engineering_module, out, rows):
    """Write the generated figures, enumerated from the route's own allowlist.

    The rule is dynamic, so it cannot be walked — but its handler already
    refuses anything outside FIGURES, and reading that same set here means the
    freezer and the route cannot disagree about what exists. Absent files are
    skipped rather than failed: the PNGs are gitignored, so a clean checkout
    legitimately has only the SVGs.
    """
    figure_dir = out / "engineering" / "figure"
    for name in sorted(engineering_module.FIGURES):
        url = f"/engineering/figure/{name}"
        if not (RESULTS_DIR / name).exists():
            rows.append((url, "skip", "not generated"))
            continue
        response = client.get(url)
        if response.status_code != 200:
            rows.append((url, str(response.status_code), "unexpected"))
            continue
        figure_dir.mkdir(parents=True, exist_ok=True)
        target = figure_dir / name
        target.write_bytes(response.get_data())
        rows.append((url, "200", f"{target.relative_to(out)} ({target.stat().st_size // 1024} KB)"))


def freeze_pages(app, engineering_module, out):
    """Render every engineering route into `out`.

    Returns:
        tuple (rows, html_documents, failures):
            rows: per-route summary tuples for the printed table.
            html_documents: rendered HTML, for the asset scan.
            failures: routes that answered with something unexpected.
    """
    rows, documents, failures = [], [], []
    client = app.test_client()

    rules = [rule for rule in app.url_map.iter_rules()
             if rule.endpoint.startswith("engineering.")]

    for rule in sorted(rules, key=lambda r: str(r)):
        url = str(rule)

        if rule.arguments:
            if rule.endpoint == "engineering.figure":
                freeze_figures(client, engineering_module, out, rows)
                continue
            # A new dynamic rule cannot be walked from the url_map alone, and
            # quietly dropping it would publish a site missing a page nobody
            # noticed was missing.
            raise SystemExit(
                f"{rule.endpoint} takes arguments {sorted(rule.arguments)} and has no "
                "enumeration strategy. Add one to freeze_engineering.py before shipping it."
            )

        if "GET" not in rule.methods:
            rows.append((url, "skip", "not a GET route"))
            continue

        response = client.get(url)

        if response.status_code == 301:
            target = response.headers.get("Location", "?")
            rows.append((url, "301", f"skipped — redirects to {target}"))
            continue

        if response.status_code != 200:
            rows.append((url, str(response.status_code), "UNEXPECTED"))
            failures.append(f"{url} returned {response.status_code}")
            continue

        html = response.get_data(as_text=True)
        documents.append(html)
        target = page_output_path(url, out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        rows.append((url, "200", f"{target.relative_to(out)} ({len(html) // 1024} KB)"))

    return rows, documents, failures


# ════════════════════════════════════════════════════════════════════
# Assets.
# ════════════════════════════════════════════════════════════════════

def copy_referenced_assets(documents, out):
    """Copy only the static files the rendered HTML actually asks for.

    Following the HTML rather than copying demo/static wholesale is what keeps
    the assistant's stylesheet, chat client and voice engine out of a
    documentation build — they live in the same directory and are irrelevant
    here.

    Returns:
        tuple (copied, missing, unexpected): sets of "css/engineering.css"-style
        relative paths.
    """
    referenced = set()
    for html in documents:
        referenced.update(STATIC_REF.findall(html))

    copied, missing = set(), set()
    for relative in sorted(referenced):
        source = STATIC_SRC / relative
        if not source.is_file():
            missing.add(relative)
            continue
        target = out / "static" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.add(relative)

    unexpected = {r for r in referenced
                  if any(marker in r for marker in ASSISTANT_ASSETS)}
    return copied, missing, unexpected


# ════════════════════════════════════════════════════════════════════
# Entry point.
# ════════════════════════════════════════════════════════════════════

def print_summary(rows):
    print()
    print(f"  {'route':<34} {'code':<6} output")
    print(f"  {'-' * 34} {'-' * 6} {'-' * 44}")
    for url, code, note in rows:
        print(f"  {url:<34} {code:<6} {note}")


def main():
    parser = argparse.ArgumentParser(
        description="Freeze /engineering into a static site.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Directory to write the frozen site into.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail the build on a bad response, a missing artifact, "
                             "or data the pages would render blank.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove the output directory before writing.")
    args = parser.parse_args()

    out = args.out.resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    app, engineering_module, data_module = make_app()
    assert_assistant_never_imported()

    info = app.config["ENG_BUILD_INFO"]
    print(f"Freezing /engineering -> {out}")
    print(f"  commit {info['commit']} · built {info['built_at']} · "
          f"artifacts {info['artifacts_mtime']}")

    problems = preflight(engineering_module, data_module)
    for problem in problems:
        print(f"  ! {problem}")

    rows, documents, failures = freeze_pages(app, engineering_module, out)
    copied, missing, unexpected = copy_referenced_assets(documents, out)

    (out / "index.html").write_text(ROOT_REDIRECT, encoding="utf-8")
    rows.append(("/", "200", "index.html (redirect to /engineering/)"))

    print_summary(rows)

    print()
    print(f"  assets copied ({len(copied)}):")
    for relative in sorted(copied):
        print(f"    {relative}")

    for relative in sorted(missing):
        print(f"  ! referenced but not found in demo/static: {relative}")
    for relative in sorted(unexpected):
        print(f"  ! WARNING: the log now references an assistant asset: {relative}")
    for relative in sorted(EXPECTED_ASSETS - copied):
        print(f"  ! WARNING: expected asset was never referenced: {relative}")
    for relative in sorted(copied - EXPECTED_ASSETS):
        print(f"  ! note: new asset not on the expected list: {relative}")

    if failures:
        print()
        for failure in failures:
            print(f"  ! {failure}")

    blocking = failures + problems + [f"asset referenced but missing: {m}" for m in sorted(missing)]
    if args.strict and blocking:
        print()
        print(f"FAILED (--strict): {len(blocking)} problem(s).")
        return 1

    print()
    print(f"Done — {sum(1 for _, code, _ in rows if code == '200')} files written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
