"""The re-ID re-lock cutoff can't drift from its evidence (the test_cutoffs_evidence.py treatment for lockon.MATCH_SIM).

0.70 was NOT computed by anything: Michael chose it on 11-21 among measured options. So this does not re-derive 0.70;
it checks that the decision record (results/followme_reid_cutoff.json) only states what the evidence says, that the
chosen value is one of the options that were actually measured, and that it is what runs. Evidence, all committed
byte-exact (a missing file FAILS here, no skip):
  * results/followme_reid_bench_v3.json + followme_reid_scores_osnet_x0_25.npz: reid_bench_v3.py (Colab, 11-22), the
    summary metrics and every raw target / stranger score. The summary is recomputed from the raw scores with the
    script's own metric code, then the stranger / target pass rate at every option is read from the same scores.
  * results/followme_reid_replay_1029_before_fix.json (11-21, 7 cutoffs, the table Michael chose from) and
    results/followme_reid_replay_1029.json (11-22, lockon.py with the 0f86828 search fixes, 3 cutoffs): the 10-29
    18:20 flight replayed through lockon.TargetLock, re-locks checked by eye.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lockon  # noqa: E402
import live_follow as lf  # noqa: E402

RESULTS = HERE.parent / "results"
RECORD = RESULTS / "followme_reid_cutoff.json"
BENCH = RESULTS / "followme_reid_bench_v3.json"
SCORES = RESULTS / "followme_reid_scores_osnet_x0_25.npz"
REPLAYS = {"replay_before_fix": RESULTS / "followme_reid_replay_1029_before_fix.json",
           "replay_after_fix": RESULTS / "followme_reid_replay_1029.json"}
REPLAY_FIELDS = ("right_relocks", "wrong_relocks", "on_me_pct", "lost_pct")


def _load_json(p):
    assert p.is_file(), f"{p} is committed evidence and must exist"
    return json.loads(p.read_text())


REC, BEN = _load_json(RECORD), _load_json(BENCH)
REP = {k: {r["cut"]: r for r in _load_json(p)} for k, p in REPLAYS.items()}
assert SCORES.is_file(), f"{SCORES} is committed evidence and must exist"
NPZ = np.load(SCORES)
GROUPS = sorted({k.rsplit("|", 1)[0] for k in NPZ.files},                        # (dataset, gap) like the script
                key=lambda g: (g.split("|")[0], float(g.split("|")[1][3:-1])))


def bench_metrics(pos, neg, rank1):
    """reid_bench_v3.py's metrics() (Drive jarvis_det/reid_bench_v3.py), unchanged apart from float64 inputs."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; s = np.r_[pos, neg]
    order = np.argsort(-s); y = y[order]; s = s[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr, fpr = tp / max(1, len(pos)), fp / max(1, len(neg))
    auc = float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))

    def tar_at(far):
        i = np.searchsorted(fpr, far, side="right") - 1
        return (float(tpr[i]), float(s[i])) if i >= 0 else (0.0, float(s[0]))
    prec = tp / np.maximum(1, tp + fp); f1 = 2 * prec * tpr / np.maximum(1e-9, prec + tpr); k = int(np.argmax(f1))
    return dict(n_pos=len(pos), n_neg=len(neg), rank1=round(float(np.mean(rank1)), 4),
                auc=round(auc, 4), f1_max=round(float(f1[k]), 4), f1_cutoff=round(float(s[k]), 4),
                tar_far1=[round(v, 4) for v in tar_at(0.01)], tar_far01=[round(v, 4) for v in tar_at(0.001)],
                stranger_p99=round(float(np.percentile(neg, 99)), 4),
                stranger_p999=round(float(np.percentile(neg, 99.9)), 4),
                target_median=round(float(np.median(pos)), 4), target_p10=round(float(np.percentile(pos, 10)), 4))


def pooled(kind):
    return np.concatenate([NPZ[f"{g}|{kind}"] for g in GROUPS]).astype(np.float64)


def test_record_points_at_the_committed_evidence():
    ev = REC["evidence"]
    assert RESULTS.parent / ev["benchmark"]["file"] == BENCH and RESULTS.parent / ev["benchmark"]["scores"] == SCORES
    assert ev["benchmark"]["model"] == "osnet_x0_25" and lockon.OSNetEmbedder.name.startswith("osnet_x0_25")
    for k, p in REPLAYS.items():
        assert RESULTS.parent / ev[k]["file"] == p


@pytest.mark.parametrize("group", GROUPS + ["ALL"])
def test_benchmark_summary_recomputes_from_the_raw_scores(group):
    """The summary file and the raw scores are one run: the script's metrics on the scores give the file, exactly."""
    if group == "ALL":
        got = bench_metrics(pooled("pos"), pooled("neg"), pooled("rank1"))
    else:
        got = bench_metrics(NPZ[f"{group}|pos"], NPZ[f"{group}|neg"], NPZ[f"{group}|rank1"])
    assert got == BEN["osnet_x0_25"][group]


def test_benchmark_operating_points_are_quoted_right_and_are_not_the_chosen_value():
    a = BEN["osnet_x0_25"]["ALL"]
    assert REC["benchmark_operating_points"] == {"f1_cutoff": a["f1_cutoff"], "stranger_p99": a["stranger_p99"]}
    assert REC["match_sim"] not in REC["benchmark_operating_points"].values()      # judgment, not one of these


def test_options_are_exactly_the_measured_cutoffs():
    measured = set(REP["replay_before_fix"]) | set(REP["replay_after_fix"])
    assert [o["cut"] for o in REC["options"]] == sorted(measured), "an option nobody measured, or a measured one left out"


@pytest.mark.parametrize("opt", REC["options"], ids=lambda o: str(o["cut"]))
def test_every_option_states_what_the_evidence_says(opt):
    c = opt["cut"]
    assert opt["bench_stranger_pass"] == round(float(np.mean(pooled("neg") >= c)), 4)   # lockon accepts best >= cut
    assert opt["bench_target_pass"] == round(float(np.mean(pooled("pos") >= c)), 4)
    for k, rows in REP.items():
        if c in rows:
            assert opt[k] == {f: rows[c][f] for f in REPLAY_FIELDS}
        else:
            assert k not in opt, f"{c}: no {k} run at this cutoff"


def test_chosen_value_was_measured_and_never_re_locked_a_stranger():
    c = REC["match_sim"]
    assert c in {o["cut"] for o in REC["options"]}
    for k, rows in REP.items():
        assert c in rows, f"0.70 must have been replayed in {k}"
        assert rows[c]["wrong_relocks"] == 0 and rows[c]["on_other_pct"] == 0.0


def test_what_runs_is_the_recorded_choice():
    """lockon reads it from the record; the pre-flight decision and live_follow's default agree with it."""
    assert lockon.MATCH_SIM == REC["match_sim"] == lockon.load_match_sim() == 0.70
    assert lf.load_decisions()["match_sim"]["value"] == REC["match_sim"]
    assert lockon.TargetLock({0}).match_sim == REC["match_sim"]
    assert lf.FollowLogic({0}, 20.0).lock.match_sim == REC["match_sim"]


def test_missing_record_stops(tmp_path):
    with pytest.raises(SystemExit, match="STOP: re-ID re-lock cutoff file not found"):
        lockon.load_match_sim(tmp_path / "nope.json")
