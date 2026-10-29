"""The follow-me detector page: exporter grouping, chart geometry, and the page itself."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "tools"))

import engineering_data as data  # noqa: E402
import export_followme_results as exp  # noqa: E402


def row(model, test, m50, person=0.2):
    return {"model": model, "dataset": f"/content/data/{test}", "mAP50": m50, "mAP50_95": m50 / 2, "images": 10,
            "per_class": {"person": {"AP50": person}, "motorcycle": {"AP50": float("nan")}}}


def test_exporter_groups_seeds_and_excludes_jatayu():
    rows = [row("vd_drop_s42", "visdrone_test_3w", 0.28), row("vd_drop_s43", "visdrone_test_3w", 0.30),
            row("vd_drop_s42", "jatayu_test_3w", 0.04), row("stock", "coco_val", 0.58),
            row("mystery_s1", "coco_val", 0.1)]
    art = exp.build(rows)
    v = art["scores"]["vd_drop"]["visdrone"]
    assert v["mAP50"]["mean"] == pytest.approx(0.29) and v["mAP50"]["n"] == 2 and v["seeds"] == [42, 43]
    assert v["mAP50"]["std"] == pytest.approx(0.01414, abs=1e-4)
    assert "jatayu" not in json.dumps(art["scores"])                      # never scored on the excluded set
    assert art["excluded_test_sets"][0]["key"] == "jatayu_test_3w"
    assert art["unrecognised_models"] == ["mystery_s1"]
    assert v["per_class"]["motorcycle"] is None                           # NaN AP is "no value", not 0
    assert {r["id"] for r in art["recipes"] if r["scored"]} == {"vd_drop", "stock"}


def test_exporter_last_line_per_seed_wins():
    art = exp.build([row("vd_gray_s42", "coco_val", 0.02), row("vd_gray_s42", "coco_val", 0.03)])
    assert art["scores"]["vd_gray"]["coco"]["mAP50"] == {"mean": 0.03, "std": None, "n": 1}


def doc_with(scores):
    return {"recipes": [{"id": r, "label": r.upper(), "scored": r in scores} for r in ("a", "b", "c")],
            "test_sets": [{"id": "visdrone", "label": "VD"}, {"id": "coco", "label": "CO"}], "scores": scores}


def agg(m, s=None, n=1):
    return {"mean": m, "std": s, "n": n}


def test_chart_bars_are_proportional_and_start_at_zero():
    doc = doc_with({"a": {"visdrone": {"mAP50": agg(0.2), "per_class": {}}},
                    "b": {"visdrone": {"mAP50": agg(0.4, 0.05, 3)}, "coco": {"mAP50": agg(0.1)}}})
    fig = {"id": "f", "title": "t", "recipes": ["a", "b", "c"], "measures": [("visdrone", None), ("coco", None)]}
    c = data.followme_chart(fig, doc)
    a, b = c["groups"][0]["bars"]
    assert b["h"] == pytest.approx(2 * a["h"], rel=1e-2)                  # 0.4 is twice 0.2
    assert a["y"] + a["h"] == pytest.approx(c["base_y"], abs=0.2)         # bars sit on the zero line
    assert c["ticks"][0]["v"] == 0.0 and c["ticks"][-1]["v"] == 0.5       # 0.4 + 0.05 rounds up to 0.5
    assert c["groups"][1]["bars"][0]["missing"]                           # a has no COCO score: gap, not 0
    assert "err_y1" in b and "err_y1" not in a
    assert c["pending"] == ["C"] and [x["label"] for x in c["legend"]] == ["A", "B"]


def test_every_figure_names_known_recipes_and_tests():
    known = set(exp.RECIPES)
    tests = {v["id"] for v in exp.TEST_SETS.values()}
    for f in data.FOLLOWME_FIGURES:
        assert set(f["recipes"]) <= known, f["id"]
        assert {t for t, _ in f["measures"]} <= tests, f["id"]


@pytest.fixture
def client():
    from flask import Flask
    import engineering
    app = Flask(__name__, template_folder=str(ROOT / "demo" / "templates"), static_folder=str(ROOT / "demo" / "static"))
    app.register_blueprint(engineering.engineering, url_prefix="/engineering")
    return app.test_client()


def test_page_renders_from_committed_artifact(client):
    r = client.get("/engineering/followme")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Follow-me detector" in html and "<svg" in html and "unavailable" not in html
    curves = data.followme_curves_data({})
    assert curves, "results/followme_training_curves.json is committed and must load"
    n_curves = len(curves["runs"]) + (4 if curves["featured"] else 0)
    assert html.count('class="fm-chart"') == len(data.FOLLOWME_FIGURES) + len(data.UNSEEN_FIGURES) + n_curves


def test_page_says_so_when_artifact_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(data, "FOLLOWME_FILE", tmp_path / "nope.json")
    html = client.get("/engineering/followme").get_data(as_text=True)
    assert "results unavailable" in html and "<svg" not in html


def test_route_declares_its_artifact():
    import engineering
    assert engineering.followme.eng_requires["artifacts"] == ("followme_detection.json",)


def fake_run(recipe, n=12, opt_lr=0.01):
    e = list(range(1, n + 1))
    return {"run": recipe + "_s42", "recipe": recipe, "epochs": e,
            **{f"{side}_{p}": [round(1.0 - 0.02 * i + (0.1 if side == "val" else 0), 4) for i in e]
               for side in ("train", "val") for p in ("box", "cls", "dfl")},
            "map50": [0.1 + 0.01 * i for i in e], "map50_95": [0.05 + 0.005 * i for i in e],
            "optimizer": "MuSGD" if opt_lr > 0.002 else "AdamW", "lr0": opt_lr, "best_epoch": n,
            "mosaic_off_epoch": n - 2, "hours": 1.5}


def test_curve_chart_maps_values_into_the_box():
    c = data.curve_chart([1, 2, 3], [("t", [0.0, 0.5, 1.0], "x")], 0.0, 1.0, mosaic_off=2, width=100, height=60,
                         pad_left=10, pad_bottom=10)
    pts = [tuple(map(float, p.split(","))) for p in c["lines"][0]["points"].split()]
    assert pts[0] == (10.0, 50.0) and pts[-1] == (94.0, 8.0)             # low-left to high-right corner
    assert pts[1][1] == pytest.approx(29.0)                                # 0.5 sits halfway up
    assert c["mosaic_x"] == pytest.approx(52.0)


def test_curves_share_one_scale_and_feature_the_winner(monkeypatch, tmp_path):
    f = tmp_path / "curves.json"
    f.write_text(json.dumps({"generated": "2026-10-29", "runs": [fake_run("vd_gray", opt_lr=0.0008),
                                                                 fake_run("vd_mav_cl")]}))
    monkeypatch.setattr(data, "FOLLOWME_CURVES_FILE", f)
    c = data.followme_curves_data({"vd_mav_cl": "VD+MAV+COCO-L"})
    assert [r["optimizer"] for r in c["runs"]] == ["AdamW", "MuSGD"]
    assert c["runs"][0]["chart"]["ticks"] == c["runs"][1]["chart"]["ticks"]   # small multiples, one y scale
    assert c["runs"][1]["gap"] == pytest.approx(0.3)                            # val - train total at the end
    assert c["featured"]["label"] == "VD+MAV+COCO-L" and len(c["featured"]["parts"]) == 3


def test_page_says_so_when_curves_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(data, "FOLLOWME_CURVES_FILE", tmp_path / "nope.json")
    html = client.get("/engineering/followme").get_data(as_text=True)
    assert "Training curves unavailable" in html and 'class="fm-explain"' not in html


def test_page_renders_curves(client, monkeypatch, tmp_path):
    f = tmp_path / "curves.json"
    f.write_text(json.dumps({"generated": "2026-10-29", "runs": [fake_run("vd_gray"), fake_run("vd_mav_cl")]}))
    monkeypatch.setattr(data, "FOLLOWME_CURVES_FILE", f)
    html = client.get("/engineering/followme").get_data(as_text=True)
    assert "What the three losses measure" in html and "Box loss" in html and "DFL" in html
    assert html.count("<polyline") == 2 * 2 + 3 * 2 + 1      # 2 runs x (train, val) + 3 parts x 2 + mAP


def test_unseen_charts_match_the_artifact():
    labels = {r["id"]: r["label"] for r in data.load_document(data.FOLLOWME_FILE)["recipes"]}
    u = data.followme_unseen_data(labels)
    doc = json.loads(data.FOLLOWME_UNSEEN_FILE.read_text())
    named = {r for f in data.UNSEEN_FIGURES for r in f["recipes"]}
    assert named == set(doc["f1"])                                        # every model appears, once
    for c in u["charts"]:
        assert [g["label"].split(" · ")[1] for g in c["groups"]] == ["person", "vehicle", "person", "person"]
        for g, t in zip(c["groups"], [t["id"] for t in doc["test_sets"]]):
            for b in g["bars"]:
                assert b["value"] == doc["f1"][b["recipe"]][t]             # bar height is the stored F1


def test_page_renders_unseen_video(client):
    html = client.get("/engineering/followme").get_data(as_text=True)
    assert 'id="unseen-video"' in html and "MOT20" in html and "fig-unseen_final" in html


def test_page_says_so_when_unseen_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(data, "FOLLOWME_UNSEEN_FILE", tmp_path / "missing.json")
    html = client.get("/engineering/followme").get_data(as_text=True)
    assert "Unseen-video scores unavailable" in html
