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
    assert html.count('class="fm-chart"') == len(data.FOLLOWME_FIGURES)


def test_page_says_so_when_artifact_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(data, "FOLLOWME_FILE", tmp_path / "nope.json")
    html = client.get("/engineering/followme").get_data(as_text=True)
    assert "results unavailable" in html and "<svg" not in html


def test_route_declares_its_artifact():
    import engineering
    assert engineering.followme.eng_requires["artifacts"] == ("followme_detection.json",)
