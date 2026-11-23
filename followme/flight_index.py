"""One table of every flight: which drone, when, how long, what it did, and where the files are (Michael 11-22).

  python flight_index.py                       # scans ~/follow_logs, prints the table, writes flights.csv there
  python flight_index.py --copy-to /mnt/c/Users/mahed/Downloads      # also drop flights.csv where Windows can see it

Runs from before 11-22 have no drone label; they show drone "?" (firmware 1.5.7 = droneA if drone.json exists).
"""
import argparse
import csv
import json
import shutil
from pathlib import Path


def summarize(run):
    f = run / "follow.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    info = json.loads((run / "drone.json").read_text()) if (run / "drone.json").exists() else {}

    def num(r, k):
        try:
            return float(r[k])
        except (KeyError, TypeError, ValueError):
            return None
    fps = 30.0
    bats = [b for b in (num(r, "battery_pct") for r in rows) if b is not None]
    states = [r.get("state", "") for r in rows]
    events = [r.get("lock_event", "") for r in rows]
    return dict(
        folder=run.name,
        drone=info.get("name") or (rows[0].get("drone") if rows and rows[0].get("drone") else "?"),
        firmware=info.get("drone_firmware", ""),
        start=f"{run.name[:4]}-{run.name[4:6]}-{run.name[6:8]} {run.name[9:11]}:{run.name[11:13]}",
        minutes=round((num(rows[-1], "t_s") or 0) / 60, 1) if rows else 0,
        airborne_min=round(sum(r.get("flying_state") in ("hovering", "flying") for r in rows) / fps / 60, 1),
        follow_min=round(sum(s == "FOLLOWING" for s in states) / fps / 60, 1),
        reid_relocks=sum(e == "reacquired_reid" for e in events),
        spin_searches=sum(1 for a, b in zip(states, states[1:]) if b.startswith("SEARCH (spinning") and not a.startswith("SEARCH (spinning")),
        look_go=sum(1 for r in rows if r.get("mode") == "LOOKGO") > 0,
        battery=f"{int(bats[0])}->{int(bats[-1])}%" if bats else "",
        videos=len(list(run.glob("video_*.mp4"))),
        boxes_logged=(run / "boxes.jsonl").exists(),
        path=str(run),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(Path.home() / "follow_logs"))
    ap.add_argument("--copy-to", default="")
    args = ap.parse_args()
    runs = sorted(p for p in Path(args.logs).iterdir() if p.is_dir())
    table = [s for s in map(summarize, runs) if s]
    out = Path(args.logs) / "flights.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]) if table else ["folder"]); w.writeheader(); w.writerows(table)
    print(f"{'start':16s} {'drone':8s} {'min':>5s} {'air':>5s} {'follow':>6s} {'relock':>6s} {'spins':>5s} {'battery':>9s}  folder")
    for s in table:
        print(f"{s['start']:16s} {s['drone']:8s} {s['minutes']:5.1f} {s['airborne_min']:5.1f} {s['follow_min']:6.1f} "
              f"{s['reid_relocks']:6d} {s['spin_searches']:5d} {s['battery']:>9s}  {s['folder']}")
    print("saved", out)
    if args.copy_to:
        shutil.copy(out, Path(args.copy_to) / "flights.csv"); print("copied to", args.copy_to)


if __name__ == "__main__":
    main()
