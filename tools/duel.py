"""
JARVIS detector duel - show only the frames where two models DISAGREE
=====================================================================
Watching two models side by side for three minutes is slow, and most frames
look the same. This runs both models on the same video and keeps only the
frames where one model sees something the other doesn't, then saves those
frames side by side as image sheets for a person to judge.

    python tools/duel.py --source flight.mp4 \
        --models visdrone=vd_gray_s42.pt vd+cocoM=vd_coco_m_s42.pt --out duel_vd_vs_cocoM

On each pair of images, a box drawn THICK and RED is one the other model did not
find (a possible miss on one side, or a false alarm on the other - that is the
judgement call this tool exists to hand to a human). THIN GRAY boxes were found
by both. YELLOW = both found something there but called it a different class
(e.g. car vs truck).

No labels are needed and none are produced: the summary counts disagreements,
it cannot say which side was right. "More vehicles" is not "better" - a model
that calls a house a truck scores more vehicles.

Outputs in --out:
  sheet_01.jpg ...   the most-disagreed frames, side by side, spread out in time
  duel.csv           one row per processed frame: A-only / B-only counts by group
  summary.txt        totals, also printed
A standalone tool: nothing in demo/ imports it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compare_detectors import Model, parse_models  # noqa: E402

GROUPS = {"person": "person"}            # everything else is a vehicle
MATCH_IOU = 0.4
RED, GRAY, YELLOW = (0, 0, 255), (170, 170, 170), (0, 220, 255)


def group(label):
    return GROUPS.get(label, "vehicle")


def parse_cutoffs(item, default):
    """'name=w.pt@0.31' -> ('name=w.pt', {'person': .31, 'vehicle': .31}); '@0.31,0.42' = person, vehicle."""
    spec, sep, cut = item.rpartition("@")
    if not sep or "=" in cut or "/" in cut:
        return item, {"person": default, "vehicle": default}
    vals = [float(v) for v in cut.split(",")]
    return spec, {"person": vals[0], "vehicle": vals[-1]}


def apply_cutoffs(dets, cut):
    """Keep detections at or above their group's cutoff."""
    return [d for d in dets if d[1] >= cut[group(d[0])]]


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(dets_a, dets_b, thr=MATCH_IOU):
    """Greedy match within the same group (person / vehicle), best IoU first.

    Returns (status_a, status_b): per detection 'both', 'class' (matched, different class) or 'only'.
    """
    pairs = []
    for i, (la, _, ba) in enumerate(dets_a):
        for j, (lb, _, bb) in enumerate(dets_b):
            if group(la) == group(lb):
                v = iou(ba, bb)
                if v >= thr:
                    pairs.append((v, i, j))
    sa, sb = ["only"] * len(dets_a), ["only"] * len(dets_b)
    for _, i, j in sorted(pairs, reverse=True):
        if sa[i] == "only" and sb[j] == "only":
            same = dets_a[i][0] == dets_b[j][0]
            sa[i] = sb[j] = "both" if same else "class"
    return sa, sb


def count_only(dets, status):
    out = {"person": 0, "vehicle": 0}
    for (lab, _, _), s in zip(dets, status):
        if s == "only":
            out[group(lab)] += 1
    return out


def pick_frames(scores, k, min_gap):
    """Indices of the k highest-scoring frames, at least min_gap apart (so the sheets aren't one moment)."""
    chosen = []
    for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
        if scores[i] <= 0 or len(chosen) >= k:
            break
        if all(abs(i - c) >= min_gap for c in chosen):
            chosen.append(i)
    return sorted(chosen)


def draw(frame, name, dets, status, width):
    scale = width / frame.shape[1]
    img = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    for (lab, c, (x1, y1, x2, y2)), s in zip(dets, status):
        x1, y1, x2, y2 = (int(v * scale) for v in (x1, y1, x2, y2))
        col, th = {"only": (RED, 3), "class": (YELLOW, 2), "both": (GRAY, 1)}[s]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
        if s != "both":
            txt = f"{lab.replace('_', '-')} {c:.2f}"
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(tht + 2, y1 - 3)
            cv2.rectangle(img, (x1, ty - tht - 2), (x1 + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(img, txt, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    only = count_only(dets, status)
    bar = np.zeros((44, img.shape[1], 3), np.uint8)
    cv2.putText(bar, name, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, f"only this model: {only['person']} person, {only['vehicle']} vehicle", (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def make_sheets(rows, out, per_sheet=4):
    """rows: list of (frame_idx, imgA, imgB). Each sheet = per_sheet pairs stacked vertically."""
    paths = []
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        strips = []
        for fi, a, b in chunk:
            sep = np.full((a.shape[0], 6, 3), 255, np.uint8)
            strip = np.hstack([a, sep, b])
            tag = np.zeros((22, strip.shape[1], 3), np.uint8)
            cv2.putText(tag, f"frame {fi}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            strips.append(np.vstack([tag, strip]))
        p = out / f"sheet_{s // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(p), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Frames where two detectors disagree, side by side.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--source", required=True, help="video file")
    ap.add_argument("--models", nargs=2, required=True, metavar="NAME=WEIGHTS[@CUT[,VEHCUT]]",
                    help="optional per-model cutoff: name=w.pt@0.31 (both groups) or @0.31,0.42 (person,vehicle)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default duel_<A>_vs_<B>)")
    ap.add_argument("--conf", type=float, default=0.35, help="cutoff for models given without @CUT (0.35)")
    ap.add_argument("--every", type=int, default=3, help="process every Nth frame (3)")
    ap.add_argument("--frames", type=int, default=24, help="how many disagreeing frames to save (24)")
    ap.add_argument("--min-gap", type=int, default=15, help="min processed frames between saved frames (15)")
    ap.add_argument("--width", type=int, default=640, help="width of each image in the sheets")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    (spec_a, cut_a), (spec_b, cut_b) = (parse_cutoffs(m, args.conf) for m in args.models)
    (na, wa), (nb, wb) = parse_models([spec_a, spec_b])
    out = args.out or Path(f"duel_{na}_vs_{nb}".replace("+", "-"))
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        print(f"[error] can't open {args.source}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    A = Model(na, wa, min(cut_a.values()), args.imgsz, args.device)
    B = Model(nb, wb, min(cut_b.values()), args.imgsz, args.device)
    print(f"cutoffs  {na}: person {cut_a['person']}, vehicle {cut_a['vehicle']}  |  "
          f"{nb}: person {cut_b['person']}, vehicle {cut_b['vehicle']}")

    records, n = [], 0           # (frame_idx, dets_a, sa, dets_b, sb)
    with open(out / "duel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", f"{na}_only_person", f"{na}_only_vehicle", f"{nb}_only_person", f"{nb}_only_vehicle",
                    "class_differs", "both"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if (n - 1) % args.every:
                continue
            da = apply_cutoffs(A.detect(frame)[0], cut_a)
            db = apply_cutoffs(B.detect(frame)[0], cut_b)
            sa, sb = match(da, db)
            oa, ob = count_only(da, sa), count_only(db, sb)
            w.writerow([n - 1, oa["person"], oa["vehicle"], ob["person"], ob["vehicle"],
                        sa.count("class"), sa.count("both")])
            records.append((n - 1, da, sa, db, sb))
            if len(records) % 20 == 0:
                print(f"\r{n}/{total} frames", end="", flush=True)
    print()
    scores = [sum(s == "only" for s in r[2]) + sum(s == "only" for s in r[4]) for r in records]
    keep = pick_frames(scores, args.frames, args.min_gap)
    rows = []
    if keep:
        wanted = {records[i][0]: records[i] for i in keep}
        cap = cv2.VideoCapture(str(args.source))
        fi = -1
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            fi += 1
            if fi in wanted:
                _, da, sa, db, sb = wanted.pop(fi)
                rows.append((fi, draw(frame, f"{na}  (cutoff p{cut_a['person']:.2f} v{cut_a['vehicle']:.2f})", da, sa,
                                      args.width),
                             draw(frame, f"{nb}  (cutoff p{cut_b['person']:.2f} v{cut_b['vehicle']:.2f})", db, sb,
                                  args.width)))
    sheets = make_sheets(rows, out)

    tot = {k: sum(count_only(r[1], r[2])[k] for r in records) for k in ("person", "vehicle")}
    totb = {k: sum(count_only(r[3], r[4])[k] for r in records) for k in ("person", "vehicle")}
    agree = sum(1 for s in scores if s == 0)
    lines = [f"{len(records)} frames compared (every {args.every}th); cutoffs {na} {cut_a}, {nb} {cut_b}",
             f"frames where they fully agree: {agree} ({100 * agree / max(len(records), 1):.0f}%)",
             f"{na} only: {tot['person']} person boxes, {tot['vehicle']} vehicle boxes",
             f"{nb} only: {totb['person']} person boxes, {totb['vehicle']} vehicle boxes",
             "Counts say WHERE they differ, not who is right - judge the sheets.",
             f"saved {len(sheets)} sheet(s) with {len(rows)} disagreeing frames to {out}/"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
