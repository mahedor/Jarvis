"""Tests for flight_log.py (Michael 11-27): one row per flight with the settings and situation it flew under, built
automatically from each run folder; nothing guessed ("not logged"), typed notes survive a regenerate, newest first,
and live_follow.py updates it at the end of every run without ever breaking the shutdown path."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flight_log as fl  # noqa: E402

NL = fl.NL
CSV_COLS = ["frame", "t_s", "video_file", "mode", "state", "target_id", "lock_event", "drone", "battery_pct",
            "flying_state"]

# drone.json as live_follow.py writes it today (ee40f91 + 3f29585 + 9f549ff), for a real drone
NEW_INFO = {
    "drone_product_name": "ANAFI-G134762", "drone_nickname": "droneA", "drone_serial": "PI040G134762",
    "drone_firmware": "1.5.7", "name": "droneA", "ip": "192.168.42.1",
    "model": "/home/m/jarvis/followme/weights/vd_mav_cl_s42_int8_openvino_model",
    "conf_person": 0.23, "conf_vehicle": 0.29,
    "conf_source": "F1-max sweep for vd_mav_cl_s42_int8, results/followme_cutoffs_balanced_int8.json",
    "reid": "re-ID: osnet_x0_25_msmt17", "profile": "drone settings match drone_profile.json",
    "running": {"detector_model": "weights/vd_mav_cl_s42_int8_openvino_model", "detector_int8": True,
                "detection_cutoffs": "results/followme_cutoffs_balanced_int8.json", "tracker": "botsort.yaml",
                "imgsz": 640, "reid": "osnet", "match_sim": 0.7, "drone_profile_applied": True, "click_to_lock": True,
                "search_deg": 360.0, "search_both_ways": True, "lost_s": 1.0},
    "preflight": {"passed": True, "deviations": [], "skipped": [], "checked": ["a"] * 12, "off_decisions": False},
}


def row(i, t, mode="MANUAL", state="MANUAL", target="", event="", bat=90, fs="hovering", drone="droneA"):
    return dict(frame=i, t_s=t, video_file="video_000.mp4", mode=mode, state=state, target_id=target,
                lock_event=event, drone=drone, battery_pct=bat, flying_state=fs)


def make_run(root, name, info=None, rows=None, cols=CSV_COLS, videos=("video_000.mp4",), notes=None, raw_info=None):
    run = root / name
    run.mkdir(parents=True)
    if raw_info is not None:
        (run / "drone.json").write_text(raw_info)
    elif info is not None:
        (run / "drone.json").write_text(json.dumps(info))
    if rows is not None:
        with open(run / "follow.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    for v in videos:
        (run / v).write_bytes(b"")
    if notes is not None:
        (run / "notes.txt").write_text(notes)
    return run


def flight_rows():
    """10 s at 10 fps: MANUAL 0-2 s (click at 1 s), FOLLOW 2-8 s with a loss, a spin and a re-ID re-lock, N at 8.5 s."""
    rs = []
    for i in range(101):
        t = round(i * 0.1, 1)
        if t < 1.0:
            rs.append(row(i, t, fs="landed" if t < 0.5 else "takingoff", bat=95))
        elif t < 2.0:
            rs.append(row(i, t, target="3", event="gallery_add"))
        elif t < 4.0:
            rs.append(row(i, t, "FOLLOW", "FOLLOWING", "3", "following"))
        elif t < 5.0:
            rs.append(row(i, t, "FOLLOW", "SEARCH (spinning 90/720 deg)", "3", "lost"))
        elif t < 5.1:
            rs.append(row(i, t, "FOLLOW", "FOLLOWING", "7", "reacquired_reid"))
        elif t < 8.0:
            rs.append(row(i, t, "FOLLOW", "FOLLOWING", "7", "following"))
        elif t < 8.5:
            rs.append(row(i, t, target="7"))
        else:
            rs.append(row(i, t, bat=80, fs="landed" if t > 9.5 else "landing"))
    return rs


def by_folder(root):
    with open(root / fl.CSV_NAME, encoding="utf-8-sig", newline="") as fh:
        return {r["run folder"]: r for r in csv.DictReader(fh)}


# ---- one run folder ----

def test_new_format_run_every_column(tmp_path):
    make_run(tmp_path, "20261127_101500_droneA", NEW_INFO, flight_rows(),
             videos=("video_000.mp4", "video_001.mp4", "overlay_000.mp4", "overlay_001.mp4"), notes="windy\nfield 2")
    fl.update(tmp_path, quiet=True)
    r = by_folder(tmp_path)["20261127_101500_droneA"]
    assert r["date/time"] == "2026-11-27 10:15:00"
    assert r["drone"] == "droneA (ANAFI-G134762)" and r["serial"] == "PI040G134762" and r["firmware"] == "1.5.7"
    assert r["duration"] == "0m10s" and r["frames"] == "101" and r["FPS"] == "10.0"
    assert r["battery"] == "95% -> 80%" and r["airborne"] == "0m09s"
    assert r["model"] == "vd_mav_cl_s42_int8_openvino_model (INT8)"
    assert r["cutoffs"] == ("person 0.23 / vehicle 0.29 (F1-max sweep for vd_mav_cl_s42_int8, "
                            "results/followme_cutoffs_balanced_int8.json)")
    assert r["tracker"] == "BoT-SORT (botsort.yaml)" and r["imgsz"] == "640"
    assert r["re-ID"] == "osnet_x0_25_msmt17, re-lock >= 0.70"
    assert r["preflight"] == "PASS (12 checked)"
    assert r["drone profile"] == "drone settings match drone_profile.json"
    assert r["FOLLOW time"] == "0m06s (on target 0m05s)"
    assert r["re-locks"] == "1" and r["locks (clicks / N)"] == "1 / 1" and r["spin searches"] == "1"
    assert r["video"] == "video_000.mp4 .. video_001.mp4 (2 files) + overlay x2"
    assert r["notes.txt"] == "windy field 2" and r["notes"] == ""
    assert "lost_s=1" in r["other settings"] and "click_to_lock=True" in r["other settings"]


def test_lock_counts_follow_replays_rule():
    """Clicks / N use replay.recorded_locks' rule; the two must not drift apart."""
    replay = pytest.importorskip("replay")
    rs = [{k: str(v) for k, v in r.items()} for r in flight_rows()]
    ev = replay.recorded_locks(rs)
    s = fl.flight_stats(rs)
    assert s["locks"] == f"{sum(e[1] == 'pick' for e in ev)} / {sum(e[1] == 'reset' for e in ev)}"


def test_deviations_are_spelled_out(tmp_path):
    info = json.loads(json.dumps(NEW_INFO))
    info["running"]["detector_model"], info["running"]["detector_int8"] = "weights/vd_mav_cl_s42.pt", False
    info["preflight"] = {"passed": False, "off_decisions": True, "checked": ["a"] * 12,
                         "skipped": [{"name": "click_to_lock", "because": ["--fake"]}],
                         "deviations": [{"name": "detector_model", "decided": "weights/vd_mav_cl_s42_int8_openvino_model",
                                         "actual": "weights/vd_mav_cl_s42.pt", "source": "x"}]}
    make_run(tmp_path, "20261127_110000_droneB", info, flight_rows())
    fl.update(tmp_path, quiet=True)
    r = by_folder(tmp_path)["20261127_110000_droneB"]
    assert r["model"] == "vd_mav_cl_s42.pt (not INT8)"
    assert r["preflight"] == ("DEVIATIONS (--off-decisions): detector_model: decided "
                              "'weights/vd_mav_cl_s42_int8_openvino_model', ran 'weights/vd_mav_cl_s42.pt' "
                              "(skipped: click_to_lock)")


@pytest.mark.parametrize("info,expect", [
    # cc9bbc3 (11-22): identity only
    (dict(drone_product_name="ANAFI-J102372", drone_nickname="droneB", drone_serial="X", drone_firmware="1.5.7",
          name="droneB", ip="192.168.53.1"),
     dict(drone="droneB (ANAFI-J102372)", model=NL, cutoffs=NL, tracker=NL, imgsz=NL, **{"re-ID": NL}, preflight=NL)),
    # aae2f0f (11-22): + model path; INT8 is a property of that exact path
    (dict(name="droneA", drone_nickname="droneA", model="/x/weights/vd_mav_cl_s42.pt"),
     dict(drone="droneA", model="vd_mav_cl_s42.pt (not INT8)", cutoffs=NL, tracker=NL, preflight=NL)),
    # f36ad29 (11-26): + cutoffs, no source
    (dict(name="droneA", model="/x/weights/vd_mav_cl_s42_int8_openvino_model", conf_person=0.24, conf_vehicle=0.28),
     dict(model="vd_mav_cl_s42_int8_openvino_model (INT8)", cutoffs="person 0.24 / vehicle 0.28 (source not logged)",
          tracker=NL, imgsz=NL)),
    # ee40f91 (11-26): running + preflight, but no tracker / imgsz / lost_s yet
    (dict(name="droneA", model="m", conf_person=0.24, conf_vehicle=0.28, conf_source="F1-max sweep for vd_mav_cl_s42",
          reid="re-ID: osnet_x0_25_msmt17",
          running=dict(detector_model="weights/vd_mav_cl_s42_int8_openvino_model", detector_int8=True, reid="osnet",
                       match_sim=0.7),
          preflight=dict(passed=True, checked=["a"] * 8, skipped=[], deviations=[])),
     dict(tracker=NL, imgsz=NL, **{"re-ID": "osnet_x0_25_msmt17, re-lock >= 0.70"}, preflight="PASS (8 checked)")),
])
def test_older_drone_json_formats(tmp_path, info, expect):
    make_run(tmp_path, "20261122_120000_droneA", info, flight_rows())
    fl.update(tmp_path, quiet=True)
    r = by_folder(tmp_path)["20261122_120000_droneA"]
    for k, v in expect.items():
        assert r[k] == v, k


def test_pre_drone_json_run_guesses_nothing(tmp_path):
    """Before 11-22: no drone.json, old follow.csv without lock_event / drone / telemetry columns."""
    old_cols = ["frame", "t_s", "video_file", "mode", "state", "target_id"]
    make_run(tmp_path, "20261115_143000", None, flight_rows(), cols=old_cols)
    fl.update(tmp_path, quiet=True)
    r = by_folder(tmp_path)["20261115_143000"]
    for k in ("drone", "serial", "firmware", "model", "cutoffs", "tracker", "imgsz", "re-ID", "preflight",
              "drone profile", "airborne", "battery", "re-locks", "locks (clicks / N)", "other settings"):
        assert r[k] == NL, k
    assert r["duration"] == "0m10s" and r["FOLLOW time"] == "0m06s (on target 0m05s)"


def test_missing_and_broken_files(tmp_path):
    make_run(tmp_path, "20261127_090000_droneA", NEW_INFO, rows=None, videos=())        # crashed before frame 1
    make_run(tmp_path, "20261127_090100_droneA", None, rows=[], videos=())               # header only, no drone.json
    make_run(tmp_path, "20261127_090200_droneA", raw_info="{not json", rows=flight_rows())
    (tmp_path / "not_a_run").mkdir()                                                     # no follow.csv / drone.json
    (tmp_path / "stray.txt").write_text("x")
    fl.update(tmp_path, quiet=True)
    rows = by_folder(tmp_path)
    assert set(rows) == {"20261127_090000_droneA", "20261127_090100_droneA", "20261127_090200_droneA"}
    a, b, c = (rows[k] for k in sorted(rows))
    assert a["duration"] == a["frames"] == a["FOLLOW time"] == a["re-locks"] == NL and a["video"] == "none in folder"
    assert a["model"] == "vd_mav_cl_s42_int8_openvino_model (INT8)"
    assert b["duration"] == NL and b["drone"] == NL
    assert c["model"] == NL and c["other settings"].startswith("drone.json unreadable") and c["duration"] == "0m10s"


# ---- the whole log ----

def test_newest_first_and_md_matches_csv(tmp_path):
    for name in ("20261029_101000", "20261126_225332_droneA", "20261126_225332_droneB", "20261115_120000"):
        make_run(tmp_path, name, None, flight_rows())
    n, _ = fl.update(tmp_path, quiet=True)
    with open(tmp_path / fl.CSV_NAME, encoding="utf-8-sig", newline="") as fh:
        order = [r["run folder"] for r in csv.DictReader(fh)]
    assert n == 4 and order == ["20261126_225332_droneB", "20261126_225332_droneA", "20261115_120000", "20261029_101000"]
    md = (tmp_path / fl.MD_NAME).read_text()
    table = [ln for ln in md.splitlines() if ln.startswith("| 2026")]
    assert [ln.split(" | ")[1] for ln in table] == order
    assert "| " + " | ".join(fl.MD_COLUMNS) + " |" in md


def test_typed_notes_survive_regenerate(tmp_path):
    make_run(tmp_path, "20261127_100000_droneA", NEW_INFO, flight_rows(), notes="gusty")
    make_run(tmp_path, "20261127_120000_droneA", NEW_INFO, flight_rows())
    fl.update(tmp_path, quiet=True)
    # Michael types into the csv; Excel saves it back with ';' and a BOM
    rows = list(by_folder(tmp_path).values())
    rows[0]["notes"], rows[1]["notes"] = "lost him behind the van", "first INT8 flight; ok"
    with open(tmp_path / fl.CSV_NAME, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fl.COLUMNS, delimiter=";")
        w.writeheader()
        w.writerows(rows)
    (tmp_path / "20261127_100000_droneA" / "notes.txt").write_text("gusty, 20 km/h")    # notes.txt edited too
    make_run(tmp_path, "20261127_130000_droneA", NEW_INFO, flight_rows())                # a new flight arrives
    fl.update(tmp_path, quiet=True)
    r = by_folder(tmp_path)
    assert r["20261127_120000_droneA"]["notes"] == "lost him behind the van"
    assert r["20261127_100000_droneA"]["notes"] == "first INT8 flight; ok"
    assert r["20261127_100000_droneA"]["notes.txt"] == "gusty, 20 km/h"                   # re-read, not stale
    assert r["20261127_130000_droneA"]["notes"] == ""
    md = (tmp_path / fl.MD_NAME).read_text()
    assert "gusty, 20 km/h / first INT8 flight; ok" in md


def test_deleted_folder_keeps_its_typed_note(tmp_path):
    import shutil
    make_run(tmp_path, "20261127_100000_droneA", NEW_INFO, flight_rows())
    fl.update(tmp_path, quiet=True)
    rows = list(by_folder(tmp_path).values())
    rows[0]["notes"] = "crash: keep this"
    with open(tmp_path / fl.CSV_NAME, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fl.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    shutil.rmtree(tmp_path / "20261127_100000_droneA")
    fl.update(tmp_path, quiet=True)
    r = by_folder(tmp_path)["20261127_100000_droneA"]
    assert r["notes"] == "crash: keep this" and r["video"].startswith("folder missing")
    assert r["model"] == "vd_mav_cl_s42_int8_openvino_model (INT8)"


def test_fake_drone_runs_left_out_unless_asked(tmp_path):
    fake = dict(NEW_INFO, drone_product_name="FAKE", drone_nickname="fake", drone_serial=None)
    make_run(tmp_path, "20261127_100000_droneA", fake, flight_rows())
    make_run(tmp_path, "20261127_110000_droneA", NEW_INFO, flight_rows())
    assert fl.update(tmp_path, quiet=True)[0] == 1
    assert "1 fake-drone run(s) not listed" in (tmp_path / fl.MD_NAME).read_text()
    assert fl.update(tmp_path, include_fake=True, quiet=True)[0] == 2


def test_cache_skips_unchanged_folders(tmp_path, monkeypatch):
    make_run(tmp_path, "20261127_100000_droneA", NEW_INFO, flight_rows())
    make_run(tmp_path, "20261127_110000_droneA", NEW_INFO, flight_rows())
    fl.update(tmp_path, quiet=True)
    calls = []
    real = fl.summarize
    monkeypatch.setattr(fl, "summarize", lambda run: calls.append(run.name) or real(run))
    fl.update(tmp_path, quiet=True)
    assert calls == []
    (tmp_path / "20261127_110000_droneA" / "notes.txt").write_text("new note")
    fl.update(tmp_path, quiet=True)
    assert calls == ["20261127_110000_droneA"]
    fl.update(tmp_path, quiet=True, rebuild=True)
    assert len(calls) == 3


def test_cli_backfill_and_missing_root(tmp_path, capsys):
    make_run(tmp_path, "20261127_100000_droneA", NEW_INFO, flight_rows())
    fl.main(["--root", str(tmp_path)])
    assert "flight log: 1 run(s)" in capsys.readouterr().out
    assert (tmp_path / fl.CSV_NAME).exists() and (tmp_path / fl.MD_NAME).exists()
    with pytest.raises(FileNotFoundError):
        fl.update(tmp_path / "nope")


# ---- the hook at the end of every live_follow.py run ----

def _fly(out, script="t@0.5,m@1,q@2.5"):
    cmd = [sys.executable, str(HERE / "live_follow.py"), "--fake", "--fake-detector", "--headless", "--auto-lock",
           "--script", script, "--seconds", "4", "--out", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=HERE)


def test_live_follow_updates_the_log_at_exit(tmp_path):
    make_run(tmp_path, "20261126_120000_droneA", NEW_INFO, flight_rows())               # an earlier real flight
    p = _fly(tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "flight log updated" in p.stdout and p.stdout.index("Landing") < p.stdout.index("flight log updated")
    assert "20261126_120000_droneA" in by_folder(tmp_path)
    assert "1 fake-drone run(s) not listed" in (tmp_path / fl.MD_NAME).read_text()      # the run just made (fake)


def test_log_failure_warns_and_never_breaks_shutdown(tmp_path):
    (tmp_path / fl.CSV_NAME).mkdir()                    # the log can't be written (stands in for Excel holding a lock)
    p = _fly(tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "Landing..." in p.stdout and "Saved" in p.stdout
    assert "WARNING: flight log not updated" in p.stdout and "python flight_log.py --root" in p.stdout
