"""
JARVIS Face Recognition — ROC / PR curve figures
================================================
Renders the ROC and precision-recall curves behind the recognition benchmark's
headline scalars, one panel per gallery aggregation, one line per encoder.

WHY THIS EXISTS SEPARATELY FROM THE BENCHMARK. benchmark_recognition.py owns
the measuring; this owns the drawing. It reads the stored artifact and never
loads an encoder, so a figure can be restyled or re-cut without re-running a
7-minute embedding job — and the figure can never silently disagree with the
JSON, because the JSON is its only input.

WHAT IS PLOTTED. Each curve comes from the `curves.sweep` grid recorded per
encoder x aggregation (see recognition_metrics.score_sweep). The AUC/AP printed
in the legend are the exact full-resolution scalars from the same record, NOT
integrated from the plotted grid, so a legend number always matches the table
on the engineering page.

Markers follow the page's colour semantics:
    amber  = the TAR@FAR yardstick threshold (for comparing encoders, never shipped)
    green  = the max-F1 deployment threshold (the one actually in the gallery)

Run:
  python tools/plot_recognition_curves.py
  python tools/plot_recognition_curves.py --run -2 --outdir results
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display needed, and none exists on the Pi
import matplotlib.pyplot as plt

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
RESULTS_FILE = REPO_ROOT / "results" / "benchmarks_recognition.json"
DEFAULT_OUTDIR = REPO_ROOT / "results"

# Palette lifted from demo/static/css/engineering.css so the figures sit on the
# engineering page without looking like a foreign object.
BG = "#0a0a0f"
SURFACE = "#101018"
TEXT = "#e0e0e0"
MUTED = "#8a8a99"
GRID = "#22222c"
YARDSTICK = "#d19a3a"
DEPLOYMENT = "#3ad19a"

# Encoder colours deliberately avoid amber and green: on this page those two
# mean "yardstick" and "deployment", and an encoder line wearing either would
# collide with the marker semantics.
ENCODER_COLOURS = ["#7c3aed", "#4a9eff", "#e0559e", "#35c5cf", "#c0c0d0"]


def load_run(path, index):
    """Load one run from the results file.

    Inputs:
        path (Path): benchmarks_recognition.json.
        index (int): Python list index; -1 is the most recent run.
    Returns:
        dict: the run payload.
    Raises:
        SystemExit: if the file is missing, empty, or the index is out of range.
    """
    if not path.exists():
        raise SystemExit(f"No results file at {path} — run benchmark_recognition.py first.")
    with open(path) as f:
        runs = json.load(f)
    if not runs:
        raise SystemExit(f"{path} contains no runs.")
    try:
        return runs[index]
    except IndexError:
        raise SystemExit(f"--run {index} out of range: {len(runs)} runs on file.")


def curve_cells(run):
    """Extract every (encoder, aggregation) record that carries curve data.

    Runs recorded before curve persistence existed have no `curves` key; they
    are skipped rather than half-plotted, and the caller reports the shortfall.

    Inputs:
        run (dict): a recognition run payload.
    Returns:
        tuple (cells, aggregations, encoders, missing):
            cells (dict): "encoder|aggregation" -> the curves record.
            aggregations (list[str]): panel order, as the run recorded it.
            encoders (list[str]): line order, as the run recorded it.
            missing (list[str]): "encoder|aggregation" keys that had no curves.
    """
    cells, encoders, missing = {}, [], []
    aggregations = run.get("aggregations_run") or []

    for result in run.get("results", []):
        if "error" in result:
            continue
        encoder = result.get("encoder", "?")
        encoders.append(encoder)
        for aggregation, metrics in (result.get("aggregations") or {}).items():
            curves = metrics.get("curves")
            key = f"{encoder}|{aggregation}"
            if curves and curves.get("sweep"):
                cells[key] = curves
            else:
                missing.append(key)

    return cells, aggregations, encoders, missing


def _style_axes(ax, xlabel, ylabel):
    """Apply the shared dark styling to one panel."""
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _marker_points(curves, kind):
    """The shipped operating points for one cell, as (x, y, colour, label).

    Inputs:
        curves (dict): one cell's curves record.
        kind (str): "roc" or "pr" — selects which axes the point maps onto.
    Returns:
        list[tuple]: (x, y, colour, label) per marker, skipping any marker whose
        precision is undefined (nothing accepted at that threshold).
    """
    points = []
    markers = curves.get("markers") or {}
    wanted = [
        ("tar@far=0.01", YARDSTICK, "yardstick TAR@FAR 1%"),
        ("best_f1", DEPLOYMENT, "deployment max-F1"),
    ]
    for key, colour, label in wanted:
        point = markers.get(key)
        if not point:
            continue
        if kind == "roc":
            points.append((point["far"], point["tar"], colour, label))
        else:
            if point.get("precision") is None:
                continue  # 0/0 — undefined, not a point on the curve
            points.append((point["recall"], point["precision"], colour, label))
    return points


def draw_figure(cells, aggregations, encoders, kind, run):
    """Draw one figure: ROC or PR, with a panel per aggregation.

    Inputs:
        cells, aggregations, encoders: as returned by curve_cells.
        kind (str): "roc" or "pr".
        run (dict): the run payload, for the subtitle.
    Returns:
        matplotlib.figure.Figure
    """
    panels = [a for a in aggregations if any(f"{e}|{a}" in cells for e in encoders)]
    if not panels:
        raise SystemExit("No curve data to plot in this run.")

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 4.6), facecolor=BG, squeeze=False
    )
    axes = axes[0]

    is_roc = kind == "roc"
    # A log FAR axis cannot show FAR == 0 ("no false accept observed"), and the
    # smallest FAR this run can even resolve is one non-mate out of the whole
    # set. Start the axis there rather than at an arbitrary decade, so the left
    # edge means "the resolution limit of this benchmark" and not "zero".
    floor_far = min(
        (p["far"] for c in cells.values() for p in c["sweep"] if p["far"] > 0),
        default=1e-4,
    )

    for ax, aggregation in zip(axes, panels):
        for i, encoder in enumerate(encoders):
            curves = cells.get(f"{encoder}|{aggregation}")
            if not curves:
                continue
            sweep = curves["sweep"]
            colour = ENCODER_COLOURS[i % len(ENCODER_COLOURS)]

            if is_roc:
                # Points at FAR == 0 are real (no non-mate got in) but have no
                # position on a log axis; the line therefore starts at the
                # first FAR the run can resolve.
                measurable = [p for p in sweep if p["far"] > 0]
                xs = [p["far"] for p in measurable]
                ys = [p["tar"] for p in measurable]
                score = f"AUC {curves.get('roc_auc', 0):.3f}"
            else:
                # Precision is None where nothing is accepted; dropping those
                # points ends the line instead of drawing it down to a
                # fabricated zero the encoder never reached.
                defined = [p for p in sweep if p["precision"] is not None]
                xs = [p["recall"] for p in defined]
                ys = [p["precision"] for p in defined]
                score = f"AP {curves.get('average_precision', 0):.3f}"

            ax.plot(xs, ys, color=colour, linewidth=1.8, label=f"{encoder} · {score}",
                    zorder=3, solid_capstyle="round")

            for x, y, marker_colour, _ in _marker_points(curves, kind):
                if is_roc and x <= 0:
                    continue  # off the left edge of a log axis
                ax.plot(x, y, marker="o", markersize=5.5, color=marker_colour,
                        markeredgecolor=BG, markeredgewidth=1.2, zorder=5)

        _style_axes(
            ax,
            "false accept rate (non-mates, log)" if is_roc else "recall (genuine accepted)",
            "true accept rate (genuine)" if is_roc else "precision (accepts that are genuine)",
        )
        if is_roc:
            # Log FAR, as biometric ROC is conventionally read. On a linear
            # axis every encoder is a right angle in the top-left corner and
            # the only region anyone operates in — FAR <= 1% — is a few pixels
            # wide. The shipped operating points are AT 1% and 0.1%, so that
            # region is the entire point of the plot.
            ax.set_xscale("log")
            ax.set_xlim(floor_far, 1.0)
            for far_line in (0.01, 0.001):
                ax.axvline(far_line, color=MUTED, linewidth=0.7,
                           linestyle=(0, (3, 4)), alpha=0.45, zorder=1)
        else:
            ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(aggregation, color=TEXT, fontsize=11, pad=8)
        legend = ax.legend(loc="lower left" if is_roc else "lower center",
                           fontsize=8, framealpha=0.0, labelcolor=TEXT)
        for text in legend.get_texts():
            text.set_color(TEXT)

    title = "ROC — true accepts vs false accepts (log FAR)" if is_roc else \
            "Precision-recall — the honest view under 50:1 imbalance"
    fig.suptitle(title, color=TEXT, fontsize=13, y=0.99)

    subtitle = (f"{run.get('num_persons', '?')} people · "
                f"{run.get('num_strangers', '?')} LFW strangers · "
                f"seed {run.get('seed', '?')} · {run.get('timestamp', '')}")
    fig.text(0.5, 0.925, subtitle, color=MUTED, fontsize=8.5, ha="center")

    # Marker key, once per figure rather than once per encoder line.
    fig.text(0.5, 0.018,
             "● amber = TAR@FAR yardstick threshold (comparison only)     "
             "● green = max-F1 deployment threshold (shipped)",
             color=MUTED, fontsize=8.5, ha="center")

    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot ROC and PR curves from the recognition benchmark artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", type=Path, default=RESULTS_FILE,
                        help="Recognition results JSON to read.")
    parser.add_argument("--run", type=int, default=-1,
                        help="Which run to plot; -1 is the most recent.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="Where the figures are written.")
    parser.add_argument("--formats", nargs="+", default=["svg", "png"],
                        help="Image formats to emit.")
    parser.add_argument("--dpi", type=int, default=160,
                        help="Raster resolution for png output.")
    args = parser.parse_args()

    run = load_run(args.results, args.run)
    cells, aggregations, encoders, missing = curve_cells(run)

    if not cells:
        raise SystemExit(
            f"Run {run.get('timestamp', args.run)} has no curve data. Curves are "
            "recorded only by benchmark_recognition.py runs from the version that "
            "added recognition_metrics.score_sweep — re-run the benchmark."
        )
    if missing:
        print(f"  ! {len(missing)} cell(s) had no curve data and were skipped: "
              f"{', '.join(sorted(missing))}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, stem in (("roc", "recognition_roc"), ("pr", "recognition_pr")):
        figure = draw_figure(cells, aggregations, encoders, kind, run)
        for fmt in args.formats:
            path = args.outdir / f"{stem}.{fmt}"
            figure.savefig(path, format=fmt, facecolor=BG, dpi=args.dpi)
            written.append(path)
        plt.close(figure)

    print(f"\nPlotted run {run.get('timestamp')} — "
          f"{len(cells)} cells, {len(encoders)} encoders.")
    for path in written:
        size = path.stat().st_size
        print(f"  {path.relative_to(REPO_ROOT)}  ({size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
