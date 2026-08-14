"""
JARVIS — public-output leak scanner
===================================
Fails the build when something that should have stayed on the machine shows up
in something that gets published. Two surfaces, because they leak differently:

    TRACKED   every git-tracked file — what a `git push` makes public.
    SITE      a frozen engineering log (tools/freeze_engineering.py --out) —
              what a static host serves to the world.

WHAT IT LOOKS FOR.
    credentials     sk-ant-… and ntn_… key shapes, on BOTH surfaces. A key is
                    the one finding here that is unambiguous and unrecoverable
                    — everything else is embarrassing, this is exploitable.
    local layout    data/… references and local artifact filenames
                    (gallery.npz and its .npz/.pt/.tflite siblings), on SITE
                    only. In source these are ordinary — a default path, a
                    docstring, an argparse help string. On a published page
                    they describe a filesystem the reader cannot see and the
                    host does not have, which is either a broken link or a
                    description of someone's hard drive.

WHAT IT DELIBERATELY DOES NOT LOOK FOR. Personal names and absolute paths were
both checked here once and are not any more. Name scanning needed a list of
real names to match against, which cannot live in a public repo and so had to
be read from a gitignored file — a check that silently becomes a no-op on any
machine without that file, which is every machine that matters (CI, the Render
build box). A guard that passes vacuously is worse than no guard, because it
gets trusted. Keep names out of published artifacts at the point they are
written instead; that is a property of the writer, not something to audit for
afterwards.

ALLOWLIST. One deliberate exemption, in ALLOWED_PROSE below, carrying its
reason. It matches the exact sentence, so rewording the paragraph re-flags it
rather than widening the hole.

Run:
  python tools/scan_public_output.py
  python tools/scan_public_output.py --site _site
  python tools/scan_public_output.py --site _site --skip-tracked
"""

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent

# Surfaces a pattern applies to.
TRACKED, SITE = "tracked", "site"
BOTH = frozenset({TRACKED, SITE})
SITE_ONLY = frozenset({SITE})

# Only text is scanned. A suffix not listed here is skipped outright rather
# than sniffed — the binaries in this repo (weights, .npz galleries, PNG
# figures) are large and none of them carry prose.
TEXT_SUFFIXES = {".py", ".html", ".js", ".css", ".json", ".jsonl", ".md", ".txt",
                 ".sh", ".svg", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".env"}

PATTERNS = (
    # The length floor is the whole discriminator. Real keys run ~100 chars;
    # the placeholders this repo documents are the longest decoys around
    # ("sk-ant-placeholder-for-qa-tests" is 24 after the prefix), so 32 clears
    # every one of them while no genuine key comes close to being that short.
    ("anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_-]{32,}"), BOTH),
    ("notion key", re.compile(r"\bntn_[A-Za-z0-9]{32,}"), BOTH),
    # Leading guard keeps this off "metadata/", "somedata/" and URL tails.
    ("local data/ path", re.compile(r"(?<![A-Za-z0-9_/.-])data/[A-Za-z0-9_./-]+"), SITE_ONLY),
    ("local artifact file", re.compile(r"\b[A-Za-z0-9_.-]+\.(?:npz|pt|tflite)\b"), SITE_ONLY),
)

# ── Allowlist ───────────────────────────────────────────────────────
# (reason, path globs, sentence). A hit is dropped only when the file matches a
# glob AND the offending line matches the sentence. Globs cover both the
# template and its frozen output, since the same prose appears in each.
ALLOWED_PROSE = (
    (
        "the enrollment page documents the crop pipeline by naming its real "
        "directories and the gallery it writes; those paths are the subject of "
        "the page, not a leak from it",
        ("demo/templates/engineering/enrollment.html", "engineering/enrollment/index.html"),
        re.compile(r"data/reference_faces|cluster_000|data/gallery\.npz"),
    ),
)


# ════════════════════════════════════════════════════════════════════
# What to scan.
# ════════════════════════════════════════════════════════════════════

def tracked_files():
    """Every git-tracked path, as absolute Paths.

    Returns:
        list[Path]: empty if this is not a git checkout (reported by caller).
    """
    try:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [REPO_ROOT / name for name in result.stdout.split("\0") if name]


# ════════════════════════════════════════════════════════════════════
# Scanning.
# ════════════════════════════════════════════════════════════════════

def allowed(relative, line):
    """True when an allowlist entry covers this file and this exact line."""
    for _reason, globs, sentence in ALLOWED_PROSE:
        if any(fnmatch.fnmatch(relative, glob) for glob in globs) and sentence.search(line):
            return True
    return False


def scan_file(path, root, surface):
    """Every non-allowlisted match in one file.

    Returns:
        list[tuple[str, str, int, str]]: (label, relative path, line no, line).
    """
    if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError:
        relative = path.name

    lines = text.splitlines()
    hits = []
    for label, pattern, surfaces in PATTERNS:
        if surface not in surfaces:
            continue
        seen_lines = set()
        for match in pattern.finditer(text):
            number = text.count("\n", 0, match.start()) + 1
            if number in seen_lines:      # one report per line per pattern
                continue
            seen_lines.add(number)
            line = lines[number - 1].strip()
            if allowed(relative, line):
                continue
            hits.append((label, relative, number, line))
    return hits


def scan(paths, root, surface, label):
    """Scan a set of paths and print every hit as file:line.

    Returns:
        list: the accumulated hits.
    """
    hits = []
    for path in sorted(paths):
        hits.extend(scan_file(path, root, surface))

    print(f"\n  {label}: {len(paths)} file(s)")
    for what, relative, number, line in sorted(hits, key=lambda h: (h[1], h[2])):
        # Windows consoles are cp1252 and the log's prose is full of em-dashes;
        # a UnicodeEncodeError here would kill the scan over a punctuation mark.
        snippet = line[:100].encode("ascii", "replace").decode()
        print(f"    ! {relative}:{number}: [{what}] {snippet}")
    if not hits:
        print("    clean")
    return hits


# ════════════════════════════════════════════════════════════════════
# Entry point.
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Scan public output for credentials and local-layout references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--site", type=Path, default=None,
                        help="A frozen site directory to scan as well.")
    parser.add_argument("--skip-tracked", action="store_true",
                        help="Scan only --site, not the git-tracked files.")
    args = parser.parse_args()

    print("Scanning public output for leaks")
    hits = []

    if not args.skip_tracked:
        files = tracked_files()
        if not files:
            print("\n  ! not a git checkout (or git unavailable) — tracked scan skipped")
        else:
            hits += scan(files, REPO_ROOT, TRACKED, "git-tracked")

    if args.site:
        site = args.site.resolve()
        if not site.is_dir():
            sys.exit(f"ERROR: --site is not a directory: {site}")
        hits += scan(list(site.rglob("*")), site, SITE, f"site {site}")

    print()
    if hits:
        print(f"FAILED: {len(hits)} leak(s).")
        return 1
    print("CLEAN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
