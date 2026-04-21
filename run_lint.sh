#!/usr/bin/env bash
# Run all linters and report errors.
# Usage: bash run_lint.sh
# Exit code: 0 if clean, 1 if any linter reports errors.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=".venv/Scripts/python"
FAILED=0

echo "============================="
echo "  JARVIS — Lint Check"
echo "============================="
echo

# ── Ruff (Python: demo/) ────────────────────────────────────────
echo "── ruff  demo/ ─────────────────────────────────"
if "$PYTHON" -m ruff check demo/; then
  echo "  ✓ ruff: no issues"
else
  FAILED=1
fi
echo

# ── ESLint (JS: demo/static/js/) ────────────────────────────────
echo "── eslint  demo/static/js/ ─────────────────────"
if npx --no-install eslint demo/static/js/; then
  echo "  ✓ eslint: no issues"
else
  FAILED=1
fi
echo

# ── Summary ─────────────────────────────────────────────────────
if [ "$FAILED" -eq 0 ]; then
  echo "============================="
  echo "  All checks passed."
  echo "============================="
else
  echo "============================="
  echo "  Lint errors found — see above."
  echo "============================="
fi

exit "$FAILED"
