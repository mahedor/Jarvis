"""
Follow-me detector ablations -> results/followme_detection.json
================================================================
The ablations train on Colab and the fair scorer appends one JSON line per
(model, test set) to Google Drive (jarvis_det/fair_scores_3w.jsonl). That file
is the raw record; this script is the curation step that turns it into the
committed artifact the /engineering/followme page reads.

    python tools/export_followme_results.py --scores fair_scores_3w.jsonl

WHAT IT DOES, AND DOES NOT, DO
  - Groups seeds of the same recipe (vd_drop_s42/43/44 -> "drop") and reports
    mean and spread, because a single-seed difference smaller than the seed
    spread is not a difference.
  - Keeps three test sets: VisDrone test-dev, MAVREC test, COCO val. JATAYU test
    is deliberately EXCLUDED: its labels were found to be sparse (median 2 boxes
    per image in busy traffic), so a score on it is not a measurement of
    anything. The exclusion is written into the output, not silently applied.
  - Never invents a row. A model with no score on a test set is absent from
    that test set, and the page says "not scored".
Stdlib only, like the rest of what demo/ reads.
"""

import argparse
import json
import math
import re
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "followme_detection.json"

TEST_SETS = {
    "visdrone_test_3w": {"id": "visdrone", "label": "VisDrone test-dev",
                         "note": "Drone footage, China, many heights. The headline set: seed spread is ±0.003."},
    "mavrec_test_3w": {"id": "mavrec", "label": "MAVREC test",
                       "note": "Drone footage, Europe, 25-45 m. Optimistic for MAVREC-trained models: its official "
                               "train and test splits share the same 12 scenes. Swings ~0.07 between seeds."},
    "coco_val": {"id": "coco", "label": "COCO val (forgetting)",
                 "note": "Everyday ground-level photos. Measures how much normal vision the drone training destroyed."},
}
EXCLUDED = {"jatayu_test_3w": "JATAYU labels are sparse (median 2 boxes per image in busy scenes); "
                              "scores on it are not a valid measure. Dataset removed from training and testing."}

# recipe id -> how the page names it. Order = reading order on the page.
RECIPES = {
    "stock": "Stock YOLOv8n (COCO only)",
    "vd_drop": "VisDrone, ignore regions dropped",
    "vd_gray": "VisDrone, ignore regions grayed",
    "vd_ign": "VisDrone, ignore regions masked in the loss",
    "vd_coco_s": "VisDrone + 1.6k COCO",
    "vd_coco_m": "VisDrone + 6.5k COCO",
    "vd_coco_l": "VisDrone + 13k COCO",
    "mav_cm": "MAVREC + 6.5k COCO",
    "vd_mav_cm": "VisDrone + MAVREC + 6.5k COCO",
    "vd_mav_cl": "VisDrone + MAVREC + 13k COCO",
    "vd_mav_gnd_cm": "VisDrone + MAVREC aerial + MAVREC ground + 6.5k COCO",
}
SEED_RE = re.compile(r"^(?P<recipe>.+)_s(?P<seed>\d+)$")
CLASSES_OF_INTEREST = ("person", "motorcycle", "car", "bicycle")


def recipe_of(model):
    if model == "stock":
        return "stock", None
    m = SEED_RE.match(model)
    return (m["recipe"], int(m["seed"])) if m else (model, None)


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    spread = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else None
    return {"mean": round(mean, 4), "std": None if spread is None else round(spread, 4), "n": len(vals)}


def build(rows):
    """rows: parsed jsonl lines from the fair scorer -> the artifact dict."""
    grouped = {}            # (recipe, test) -> list of rows (one per seed)
    unknown = set()
    for r in rows:
        test = Path(r["dataset"]).name
        recipe, seed = recipe_of(r["model"])
        if recipe not in RECIPES:
            unknown.add(r["model"])
            continue
        if test in TEST_SETS:
            grouped.setdefault((recipe, test), {})[seed] = r     # last line per seed wins (re-scores)
    scores = {}
    for (recipe, test), by_seed in grouped.items():
        rs = list(by_seed.values())
        entry = {"seeds": sorted(s for s in by_seed if s is not None),
                 "mAP50": _agg([_num(r["mAP50"]) for r in rs]),
                 "mAP50_95": _agg([_num(r["mAP50_95"]) for r in rs]),
                 "images": rs[0].get("images"),
                 "per_class": {c: _agg([_num(r["per_class"].get(c, {}).get("AP50")) for r in rs])
                               for c in CLASSES_OF_INTEREST}}
        scores.setdefault(recipe, {})[TEST_SETS[test]["id"]] = entry
    return {
        "generated": date.today().isoformat(),
        "source": "Google Drive jarvis_det/fair_scores_3w.jsonl (fair scorer, Colab T4)",
        "scorer": "COCO-style AP over our 8 classes; an unmatched detection mostly inside an ignore box is "
                  "not counted as a false positive (the VisDrone toolkit's rule).",
        "base_model": "YOLOv8n pretrained on COCO, 640 px, batch 16, 50 epochs; only the training data changes.",
        "test_sets": [{"id": v["id"], "key": k, "label": v["label"], "note": v["note"]} for k, v in TEST_SETS.items()],
        "excluded_test_sets": [{"key": k, "reason": v} for k, v in EXCLUDED.items()],
        "recipes": [{"id": k, "label": v, "scored": k in scores} for k, v in RECIPES.items()],
        "scores": scores,
        "unrecognised_models": sorted(unknown),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scores", type=Path, required=True, help="fair_scores_3w.jsonl from Drive")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in a.scores.read_text(encoding="utf-8").splitlines() if line.strip()]
    art = build(rows)
    a.out.write_text(json.dumps(art, indent=2) + "\n", encoding="utf-8")
    n = sum(r["scored"] for r in art["recipes"])
    print(f"wrote {a.out} ({n} scored recipes; unrecognised: {art['unrecognised_models'] or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
