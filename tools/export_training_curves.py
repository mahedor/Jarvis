"""
Follow-me training curves -> results/followme_training_curves.json
===================================================================
Ultralytics writes one results.csv per training run (one row per epoch: train
and val box / cls / dfl loss, val precision / recall / mAP, learning rates).
This script curates those files into the committed artifact the
/engineering/followme page draws its loss curves from.

    python tools/export_training_curves.py RUNS_DIR [RUNS_DIR ...] --out results/followme_training_curves.json

RUNS_DIR is a folder of run folders (e.g. Drive jarvis_det/runs_3w), each with
results.csv. Only seed-42 runs of the recipes the page names are kept.

What it records per run, and nothing else: per-epoch losses and val mAP rounded
to 4 decimals, the optimizer Ultralytics' `optimizer=auto` actually picked
(read from the logged learning rate, see optimizer_of), the epoch where the
mosaic augmentation switched off, and the best epoch by mAP50-95 (the
checkpoint best.pt keeps). Stdlib only, like the rest of what demo/ reads.
"""

import argparse
import csv
import json
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "followme_training_curves.json"

# run folder -> recipe id used by followme_detection.json. Order = page order.
RUNS = {
    "vd_drop_s42": "vd_drop",
    "vd_gray_s42": "vd_gray",
    "vd_ign_s42": "vd_ign",
    "vd_coco_s_s42": "vd_coco_s",
    "vd_coco_m_s42": "vd_coco_m",
    "vd_coco_l_s42": "vd_coco_l",
    "mav_cm_s42": "mav_cm",
    "vd_mav_cm_s42": "vd_mav_cm",
    "vd_mav_cl_s42": "vd_mav_cl",
    "vd_mav_gnd_cm_s42": "vd_mav_gnd_cm",
}
COLUMNS = {
    "train_box": "train/box_loss", "train_cls": "train/cls_loss", "train_dfl": "train/dfl_loss",
    "val_box": "val/box_loss", "val_cls": "val/cls_loss", "val_dfl": "val/dfl_loss",
    "map50": "metrics/mAP50(B)", "map50_95": "metrics/mAP50-95(B)",
}
CLOSE_MOSAIC = 10   # Ultralytics default: mosaic off for the last 10 epochs


def optimizer_of(lr_pg0):
    """Which optimizer optimizer=auto chose, read from the logged learning rate.

    Ultralytics 8.4: MuSGD at lr0 0.01 when the run has > 10,000 iterations, else AdamW at
    round(0.002 * 5 / (4 + nc), 6) (~0.00083 for 8 classes). Warmup only ever ramps towards lr0,
    so the peak logged rate separates them: AdamW never exceeds ~0.001."""
    peak = max(lr_pg0)
    return ("MuSGD", 0.01) if peak > 0.002 else ("AdamW", round(peak, 6))


def read_run(path):
    rows = list(csv.DictReader(Path(path).open()))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    out = {k: [round(float(r[c]), 4) for r in rows] for k, c in COLUMNS.items()}
    out["epochs"] = [int(float(r["epoch"])) for r in rows]
    opt, lr = optimizer_of([float(r["lr/pg0"]) for r in rows])
    best = max(range(len(rows)), key=lambda i: out["map50_95"][i])
    n = len(rows)
    return {**out, "optimizer": opt, "lr0": lr, "best_epoch": out["epochs"][best],
            "mosaic_off_epoch": out["epochs"][n - CLOSE_MOSAIC] if n > CLOSE_MOSAIC else None,
            "hours": round(float(rows[-1]["time"]) / 3600, 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    found = {}
    for d in args.runs_dirs:
        for run in RUNS:
            f = d / run / "results.csv"
            if run not in found and f.exists():
                found[run] = f
    runs = []
    for run, recipe in RUNS.items():
        if run in found:
            runs.append({"run": run, "recipe": recipe, **read_run(found[run])})
            print(f"{run:20s} {len(runs[-1]['epochs'])} epochs  {runs[-1]['optimizer']}")
        else:
            print(f"{run:20s} missing")
    doc = {"generated": date.today().isoformat(),
           "source": "Ultralytics results.csv per run (Drive jarvis_det/runs_3w)",
           "losses_as_logged": "Each loss is logged already multiplied by its weight (box 7.5, cls 0.5, dfl 1.5)",
           "runs": runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
