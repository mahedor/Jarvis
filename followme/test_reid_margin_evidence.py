"""The re-ID re-lock margin can't drift from its evidence (the test_reid_evidence.py treatment for lockon.MARGIN).

0.05 was NOT computed by anything: Michael chose it on 2026-11-27 23:21 PST among the four measured margins. So this does
not re-derive 0.05; it checks that the decision record (results/followme_reid_margin_choice.json) only states what the
evidence says, that the chosen value is one of the measured options, and that it is what runs. Evidence, committed
byte-exact from Drive jarvis_det (a missing or changed file FAILS here, no skip):
  * results/followme_lockon_settings_sweep.jsonl: lockon_settings_sweep.py (Colab, 11-27), one row per
    config x dataset x split, plus the supplementary margin x cutoff rows.
  * results/followme_lockon_settings_sweep_summary.json: the same run's summary (split, reproduction of
    reid_bench_v3's scores, one-at-a-time tables, the script's own pick).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lockon  # noqa: E402
import live_follow as lf  # noqa: E402

RESULTS = HERE.parent / "results"
RECORD = RESULTS / "followme_reid_margin_choice.json"
SWEEP = RESULTS / "followme_lockon_settings_sweep.jsonl"
SUMMARY = RESULTS / "followme_lockon_settings_sweep_summary.json"
# Drive file sizes (metadata) and the sha256 of the downloaded bytes, 2026-11-27
PINNED = {SWEEP: (1288046, "5d44ee085ba8fec28ef9b51887c6952ddbd630d2de079bf420ec90b26f05c155"),
          SUMMARY: (26246, "1e517a68f2e344ff19260d63e3c6af10d2ef4d9cdca4de5dfe54dd61d1d32ae6")}
RATES = ("right", "wrong_present", "none_present", "wrong_absent")


def _read(p):
    assert p.is_file(), f"{p} is committed evidence and must exist"
    return p.read_bytes()


REC = json.loads(_read(RECORD))
SUM = json.loads(_read(SUMMARY))
ROWS = [json.loads(line) for line in _read(SWEEP).decode().splitlines() if line.strip()]
GRID = {(r["config"], r["dataset"], r["split"]): r for r in ROWS if "config" in r}
CUR = SUM["current"]


def cfg_key(margin):
    """The sweep's config name (lockon_settings_sweep.cfg_key) at the then-current settings with this margin."""
    return f"G{CUR['G']}|sp{CUR['spacing']:g}|{CUR['agg']}|m{margin:g}"


def row(margin, key):
    ds, split = key.split("|")
    return GRID[(cfg_key(margin), ds, split)]


@pytest.mark.parametrize("path", list(PINNED), ids=lambda p: p.name)
def test_evidence_is_byte_exact(path):
    raw = _read(path)
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == PINNED[path]


def test_record_points_at_the_committed_evidence():
    ev = REC["evidence"]
    assert RESULTS.parent / ev["sweep"] == SWEEP and RESULTS.parent / ev["summary"] == SUMMARY
    assert SUM["files"]["rows"] == "lockon_settings_sweep.jsonl" and SWEEP.name.endswith(SUM["files"]["rows"])


def test_row_counts_in_the_record_are_the_files():
    n_grid, n_supp = len(GRID), sum("supplementary" in r for r in ROWS)
    assert len(ROWS) == n_grid + n_supp and len(GRID) == sum("config" in r for r in ROWS)    # no duplicate keys
    assert f"({len(ROWS)} rows: {n_grid} config x dataset x split rows + {n_supp} supplementary" in REC["evidence"]["from"]
    assert REC["evidence"]["from"].count("2026-11-28T06:52:02Z") == 1 and SUM["made"] == "2026-11-28T06:52:02+00:00"


def test_dataset_description_is_the_summarys():
    ev = REC["evidence"]
    for k in ("what", "embedder", "match_sim", "split_rule", "halves", "n_sweep_trials", "caveats"):
        assert ev[k] == SUM[k], k
    assert ev["embedder"] == "osnet_x0_25" and lockon.OSNetEmbedder.name.startswith(ev["embedder"])
    assert ev["match_sim"] == lockon.MATCH_SIM                    # measured at the cutoff that flies
    assert "MOT17 train" in ev["what"] and "VisDrone-MOT val" in ev["what"]
    # the halves really are the stated rule: per dataset, sorted names, even index -> choose, odd -> check
    for ds in SUM["halves"]["choose"]:
        seqs = sorted(SUM["halves"]["choose"][ds] + SUM["halves"]["check"][ds])
        assert SUM["halves"]["choose"][ds] == seqs[0::2] and SUM["halves"]["check"][ds] == seqs[1::2]
    assert set(ev["halves_use"]) == {"choose", "check"}


def test_reproduction_of_the_re_id_benchmark():
    rp, sv = REC["evidence"]["reproduction"], SUM["repro_v3"]
    assert rp["n_v3_trials"] == SUM["n_v3_trials"]
    assert rp["all_counts_match"] is sv["all_counts_match"] is True
    assert rp["worst_abs_diff"] == sv["worst_abs_diff"] == max(v["max_abs_diff"] for k, v in sv.items() if "|" in k)
    assert rp["worst_abs_diff"] < 5e-7
    for k, v in sv.items():
        if "|" in k:
            assert v["n_pos"][0] == v["n_pos"][1] and v["n_neg"][0] == v["n_neg"][1], k


def test_measured_at_what_lockon_runs():
    """The margins were swept with everything else at lockon.py's settings; if those change, re-measure."""
    assert CUR == {"G": lockon.GALLERY_MAX, "spacing": lockon.ADD_EVERY_S, "agg": "max", "margin": REC["margin"]}
    assert f"GALLERY_MAX {CUR['G']}, ADD_EVERY_S {CUR['spacing']} s, max cosine" in REC["metric"]
    assert f"match_sim {SUM['match_sim']:.2f}" in REC["metric"]
    assert "1 / 3 / 10 s" in REC["metric"] and SUM["grid"]["gaps_s"] == [1.0, 3.0, 10.0]


def test_options_are_exactly_the_measured_margins():
    assert [o["margin"] for o in REC["options"]] == SUM["grid"]["margin"] == [0.0, 0.02, 0.05, 0.1]


@pytest.mark.parametrize("opt", REC["options"], ids=lambda o: str(o["margin"]))
def test_every_option_states_what_the_evidence_says(opt):
    m = opt["margin"]
    blk = SUM["one_at_a_time_from_current"]["margin"][str(m)]
    keys = [k for k in opt if "|" in k]
    assert sorted(keys) == sorted(blk) and set(opt) == set(keys) | {"margin", "note"}
    for k in keys:
        assert opt[k] == {f: row(m, k)[f] for f in RATES} == blk[k], f"{m} {k}"
        assert row(m, k)["match_sim"] == SUM["match_sim"]


def test_trial_counts_are_the_rows():
    for k, n in REC["trial_counts"].items():
        for m in SUM["grid"]["margin"]:                            # the same trials for every margin
            assert n == {"n_present_multi": row(m, k)["n_present_multi"], "n_absent_multi": row(m, k)["n_absent_multi"]}
    assert set(REC["trial_counts"]) == {k for k in REC["options"][0] if "|" in k}


def test_summary_check_block_agrees_with_the_rows():
    """The summary's 'current' blocks are rows of the jsonl at margin 0.05: the two files are one run."""
    for k, v in SUM["check"].items():
        assert v["current"] == {f: row(CUR["margin"], k)[f] for f in v["current"]}, k


def test_reason_quotes_the_check_half_numbers():
    opt = {o["margin"]: o["ALL|check"] for o in REC["options"]}
    wrong_002, right_010 = opt[0.02]["wrong_present"], opt[0.1]["right"]
    assert round(wrong_002, 2) == 0.07 and round(right_010, 2) == 0.60          # Michael's ~7% and ~60%
    assert "(~7%)" in REC["reason"] and "(~60%)" in REC["reason"]
    assert f"wrong_present {wrong_002} at 0.02" in REC["reason"] and f"right {right_010} at 0.10" in REC["reason"]
    notes = {o["margin"]: o["note"] for o in REC["options"]}
    assert f"ALL check {wrong_002}" in notes[0.02] and f"ALL check {right_010}" in notes[0.1]
    assert notes[REC["margin"]].startswith("Chosen")


def test_every_number_in_the_flight_decision_is_the_records():
    src = lf.load_decisions()["reid_margin"]["source"]
    opt = {o["margin"]: o["ALL|check"] for o in REC["options"]}
    rights = " / ".join(f"{opt[m]['right']:g}" for m in sorted(opt))
    wrongs = " / ".join(f"{opt[m]['wrong_present']:g}" for m in sorted(opt))
    assert f"right re-locks {rights}" in src and f"present {wrongs}" in src
    assert f"within {REC['evidence']['reproduction']['worst_abs_diff']:.1e}" in src
    for num in re.findall(r"\d+\.\d+", src):                        # nothing else crept in
        assert num in {"0.05", "0.02", "0.10", "0.70", "0.7875", "0.7583", "0.725", "0.5958", "0.1125",
                       "0.0667", "0.0167", "0.0042", "4.8"}, num


def test_sweep_auto_pick_is_the_summarys():
    ap = REC["sweep_auto_pick"]
    assert ap["settings"] == SUM["chosen"] and ap["rule"] == SUM["rule"] and ap["n_feasible"] == SUM["n_feasible"]
    assert SUM["chosen_is_current"] is False and ap["settings"]["G"] != CUR["G"]   # only the margin is decided here


def test_chosen_value_was_measured_and_who_decided():
    assert REC["margin"] in {o["margin"] for o in REC["options"]}
    assert REC["decided_by"] == "Michael" and REC["date"] == "2026-11-27" and REC["decided_at"] == "2026-11-27 23:21 PST"
    assert REC["in_talk"] is False and "not presented in the talk" in REC["talk_note"]


def test_what_runs_is_the_recorded_choice():
    """lockon reads it from the record; the pre-flight decision and what live_follow / replay run agree with it."""
    assert lockon.MARGIN == REC["margin"] == lockon.load_margin() == 0.05
    assert lf.load_decisions()["reid_margin"]["value"] == REC["margin"]
    assert lockon.TargetLock({0}).margin == REC["margin"]
    assert lf.FollowLogic({0}, 20.0).lock.margin == REC["margin"]
    args = lf.build_parser().parse_args([])
    assert lf.running_settings(args, True, precision="INT8")["reid_margin"] == REC["margin"]


def test_missing_record_stops(tmp_path):
    with pytest.raises(SystemExit, match="STOP: re-ID re-lock margin file not found"):
        lockon.load_margin(tmp_path / "nope.json")
