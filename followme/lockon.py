"""Lock-on: follow ONE chosen person and get them back after the tracker loses them (2026-11-10).

Re-ID = making an appearance vector for a person crop and comparing two vectors (cosine similarity).
Lock-on = the job built on top of the tracker: choose the target, keep following their track ID, and when that ID
disappears, only re-lock onto someone whose appearance matches the target's stored vectors.

Replaces the old behaviour (replay.pick_target, deleted 11-26): "track ID vanished -> grab whoever is nearest the last-seen box",
which happily latched onto a stranger walking past.

Pieces:
  * Embedder:   crop -> L2-normalised vector. OSNet x0_25 (boxmot, re-ID trained on MSMT17). If it can't load,
                the program stops (no fallback). An HSV colour histogram exists only as a named test baseline.
  * Gallery:    up to GALLERY_MAX vectors of the target, added at most every ADD_EVERY_S, and only when the target
                is confident, big enough, and not overlapping anyone else (so a stranger never leaks in).
  * Re-acquire: candidate = any person track that isn't a recently rejected one. Score = max cosine similarity to
                the gallery (multi-reference, like the face pipeline). Accept if >= MATCH_SIM and it beats the
                runner-up by MARGIN. Each candidate gets a few quick checks (CHECKS_PER_TRACK), then is looked at
                again every RETRY_S. Only clean boxes are candidates (same rule as gallery samples).
  * No gallery yet (embedder missing, or lost before the first sample): stays lost; it never guesses.

Lock rule (Michael 2026-11-20): lock on ONLY when you click someone, and stay on that track ID. If the ID is lost,
try to find THEM again (re-ID match against their gallery), but never lock onto anyone new: no nearest-box fallback
and no automatic "biggest person" (that is only available as auto_lock=True, used by the fake-drone test).
Vehicles (11-19): a clicked vehicle can be the target. OSNet only knows people, so a lost vehicle is not re-acquired
automatically (click it again) until a vehicle re-ID model is added.

MATCH_SIM is READ from results/followme_reid_cutoff.json (Michael's 11-21 choice among measured options: the re-ID
benchmark and the 10-29 replay, both committed in results/); test_reid_evidence.py checks it against them. Missing
file -> stop. MARGIN is READ the same way from results/followme_reid_margin_choice.json (Michael's 11-27 choice among
the four margins of the 11-27 lock-on settings sweep, committed in results/); test_reid_margin_evidence.py checks it.
Pure logic + a thin embedder wrapper; no drone I/O, so it is unit-tested.
"""
import json
import time
from pathlib import Path

import numpy as np

GALLERY_MAX = 12
ADD_EVERY_S = 1.0
MIN_CONF_ADD = 0.5
MIN_BOX_H_ADD = 40          # px; tiny crops make junk vectors
ISOLATION_IOU = 0.0         # target must overlap no other person box to be added to the gallery
EDGE_MARGIN = 4             # px; a box touching the frame edge is a partial body (2026-11-14: first gallery sample
                            # on the real flight was a blurry close-up face at takeoff) -> never a gallery sample
REID_CUTOFF_FILE = Path(__file__).resolve().parent.parent / "results" / "followme_reid_cutoff.json"
REID_MARGIN_FILE = Path(__file__).resolve().parent.parent / "results" / "followme_reid_margin_choice.json"


def load_match_sim(path=REID_CUTOFF_FILE):
    """The re-lock cutoff from its decision record (no default in code); missing file -> stop."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"STOP: re-ID re-lock cutoff file not found: {p}\n"
                         f"  It is committed in results/ next to its evidence; restore it.")
    return float(json.loads(p.read_text())["match_sim"])


def load_margin(path=REID_MARGIN_FILE):
    """The re-lock margin from its decision record (no default in code); missing file -> stop."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"STOP: re-ID re-lock margin file not found: {p}\n"
                         f"  It is committed in results/ next to its evidence; restore it.")
    return float(json.loads(p.read_text())["margin"])


# Michael 11-21: keep 0.70 (set 11-14 from the 10-29 18:20 flight). Chosen by judgment, not computed: re-ID benchmark
# (OSNet x0_25, ROC-AUC 0.965) passes 8.1% of single stranger looks at 0.70 and 1% at 0.789; the 10-29 replay had no
# wrong re-lock at any cutoff 0.65-0.85, but at 0.79 he scored 0.779 and stayed lost ~2.5 min. With the search fixes
# below every re-lock was him (0.70-0.91), highest clean stranger 0.621. All in results/followme_reid_*.
MATCH_SIM = load_match_sim()
# Best match must beat the runner-up by MARGIN. Michael 11-27 23:21 PST: keep 0.05. Lock-on settings sweep (OSNet x0_25,
# MOT17 train + VisDrone-MOT val, check half): margin 0.02 -> wrong re-locks with the target present 0.0667 (too high),
# 0.10 -> right re-locks 0.5958 (too low), 0.05 -> 0.725 right / 0.0167 wrong. All in results/followme_lockon_settings_*.
MARGIN = load_margin()
CHECKS_PER_TRACK = 3        # quick checks of a candidate track when it first appears while lost
RECHECK_S = 0.3             # ...no more often than this
RETRY_S = 2.0               # after the quick checks, look at a refused track again every RETRY_S (Michael 11-21:
                            # the 10-29 replay at a 0.79 cutoff refused him 3x at 0.779 and then never looked again
                            # for the ~60 s the tracker kept that ID)
# Search candidates must pass the same clean-sample rule as the gallery (Michael 11-21): confident, not tiny, not
# cut off by the frame edge, not overlapping another person. On the 10-29 replay the only non-target boxes scoring
# >= 0.70 were fragments (conf 0.26-0.54, a sliver on the top edge, part-body boxes inside another person's box).


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ------------------------------------------------------------------ embedders
class HistEmbedder:
    """HSV colour histogram of the box (upper and lower half separately: shirt vs trousers). Baseline only."""
    name = "hsv_hist"

    def __call__(self, frame, boxes):
        import cv2
        out = []
        H, W = frame.shape[:2]
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                out.append(np.zeros(2 * 8 * 4, np.float32)); continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mid = hsv.shape[0] // 2
            parts = [cv2.calcHist([p], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel() for p in (hsv[:mid], hsv[mid:])]
            v = np.concatenate(parts).astype(np.float32)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.array(out, np.float32).reshape(len(boxes), -1)


class OSNetEmbedder:
    """OSNet x0_25 trained for person re-ID (MSMT17), via boxmot's ReID runtime. Weights auto-download once."""
    name = "osnet_x0_25_msmt17"

    def __init__(self, weights="osnet_x0_25_msmt17.pt", device="cpu"):
        from boxmot.reid.core.runtime import ReID      # imported lazily: boxmot is optional
        from pathlib import Path
        local = Path(__file__).resolve().parent / "weights" / weights   # followme/weights/ (gitignored)
        if local.exists():
            weights = str(local)
        self.reid = ReID(weights=weights, device=device)

    def __call__(self, frame, boxes):
        if len(boxes) == 0:
            return np.empty((0, 0), np.float32)
        return np.asarray(self.reid(frame, boxes=np.asarray(boxes, np.float32)), np.float32)


def make_embedder(kind="osnet"):
    """'osnet', 'hist' or 'none'. Returns (embedder|None, note).
    No silent fallback (Michael 11-22): if OSNet can't load, this raises instead of quietly switching to the much
    weaker colour histogram at the same cutoff. The histogram only runs when asked for by name (tests/baseline)."""
    if kind == "none":
        return None, "re-ID OFF: a lost target is never re-acquired"
    if kind == "osnet":
        e = OSNetEmbedder()
        return e, f"re-ID: {e.name}"
    if kind == "hist":
        return HistEmbedder(), "re-ID: HSV histogram baseline (NOT for flying)"
    raise ValueError(f"unknown re-ID kind {kind!r} (use osnet, hist or none)")


# ------------------------------------------------------------------ the lock
class TargetLock:
    """Tracks: (x1, y1, x2, y2, id, cls, conf). Feed it every detector result with the vectors you were asked for."""

    def __init__(self, person_ids, match_sim=MATCH_SIM, margin=MARGIN, auto_lock=False):
        self.person_ids, self.match_sim, self.margin = set(person_ids), match_sim, margin
        self.auto_lock = auto_lock        # False = lock only on a click (Michael 11-20)
        self.reset()

    def reset(self):
        self.target_id = None
        self.gallery = []                 # list of vectors
        self.last_add = -1e9
        self.last_box = None
        self.lost_since = None
        self.checked = {}                 # tid -> (n_checks, last_check_t, best_sim) while lost
        self.picked_by = "auto"
        self.last_event = ""
        self.target_is_person = True      # False when a vehicle was clicked (11-19)
        self.last_sim = None              # similarity of the most recent re-ID re-lock (logged)

    # -- choosing ---------------------------------------------------------
    def pick(self, tid, tracks=None, by="you"):
        """Lock onto track `tid`. Picking a new person starts a fresh gallery."""
        if tid != self.target_id:
            self.gallery, self.last_add = [], -1e9
        self.target_id, self.picked_by, self.lost_since, self.checked = tid, by, None, {}
        if tracks is not None:
            t = next((t for t in tracks if t[4] == tid), None)
            if t is not None:
                self.last_box = tuple(t[:4])
                self.target_is_person = t[5] in self.person_ids

    def _people(self, tracks):
        return [t for t in tracks if t[5] in self.person_ids]

    def _same_kind(self, tracks):
        """Candidates the target can be: people for a person target, non-person boxes for a vehicle target."""
        if self.target_is_person:
            return self._people(tracks)
        return [t for t in tracks if t[5] not in self.person_ids]

    # -- which crops to embed this frame (cheap: usually 0 or 1) -----------
    def wants_embedding(self, tracks, now, W=None, H=None):
        if not self.target_is_person:
            return []                                       # OSNet only knows people
        people = self._people(tracks)
        tgt = next((t for t in people if t[4] == self.target_id), None)
        if tgt is not None:
            if now - self.last_add >= ADD_EVERY_S and self._clean_sample(tgt, people, W, H):
                return [tgt[4]]
            return []
        if self.target_id is None or not self.gallery:
            return []
        want = []
        for t in people:
            if not self._clean_sample(t, people, W, H):
                continue                                    # fragment / partial / crowded box: never a candidate
            n, last_t, _ = self.checked.get(t[4], (0, -1e9, None))
            if (n < CHECKS_PER_TRACK and now - last_t >= RECHECK_S) or now - last_t >= RETRY_S:
                want.append(t[4])
        return want

    def _clean_sample(self, tgt, people, W=None, H=None):
        if tgt[6] < MIN_CONF_ADD or (tgt[3] - tgt[1]) < MIN_BOX_H_ADD:
            return False
        if W and H and (tgt[0] < EDGE_MARGIN or tgt[1] < EDGE_MARGIN or tgt[2] > W - EDGE_MARGIN or tgt[3] > H - EDGE_MARGIN):
            return False                                    # cut off by the frame: not the whole person
        return all(iou(tgt[:4], o[:4]) <= ISOLATION_IOU for o in people if o[4] != tgt[4])

    def similarity(self, vec):
        if not self.gallery or vec is None or len(vec) == 0:
            return None
        return float(np.max(np.stack(self.gallery) @ np.asarray(vec, np.float32)))

    # -- the per-frame decision ------------------------------------------
    def update(self, tracks, vecs, now, W=None, H=None):
        """vecs: {track_id: vector} for the ids wants_embedding returned (missing ones are fine).
        Returns (target_track or None, event). Events: following, first_lock, gallery_add, reacquired_reid,
        lost, searching, no_target."""
        people = self._people(tracks)
        if self.target_id is None:
            if not self.auto_lock:
                return None, self._ev("no_target")         # waits for a click
            if not people:
                return None, self._ev("searching")
            t = max(people, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
            self.pick(t[4], by="auto")
            self.last_box, self.target_is_person = tuple(t[:4]), True
            return t, self._ev("first_lock")

        if not self.target_is_person:
            return self._update_vehicle(tracks, now, W, H)

        tgt = next((t for t in people if t[4] == self.target_id), None)
        if tgt is not None:
            self.lost_since, self.checked, self.last_box = None, {}, tuple(tgt[:4])
            v = vecs.get(tgt[4])
            if v is not None and len(v):
                self.gallery.append(np.asarray(v, np.float32))
                self.gallery = self.gallery[-GALLERY_MAX:]
                self.last_add = now
                return tgt, self._ev("gallery_add")
            return tgt, self._ev("following")

        # target's track ID is gone
        if self.lost_since is None:
            self.lost_since = now
        if not people or not self.gallery:                          # no fingerprints: never guess (Michael 11-20)
            return None, self._ev("lost")
        for tid, v in vecs.items():
            if v is None or not len(v):
                continue
            n, _, best = self.checked.get(tid, (0, -1e9, None))
            s = self.similarity(v)
            self.checked[tid] = (n + 1, now, s if best is None else max(best, s))
        scored = sorted(((c[2], tid) for tid, c in self.checked.items()
                         if c[2] is not None and any(p[4] == tid for p in people)), reverse=True)
        if scored:
            best, tid = scored[0]
            runner = scored[1][0] if len(scored) > 1 else -1.0
            if best >= self.match_sim and best - runner >= self.margin:
                t = next(p for p in people if p[4] == tid)
                self.target_id, self.lost_since, self.checked, self.last_box = tid, None, {}, tuple(t[:4])
                self.last_sim = best
                return t, self._ev("reacquired_reid")
        return None, self._ev("lost")

    def _update_vehicle(self, tracks, now, W, H):
        cands = self._same_kind(tracks)
        tgt = next((t for t in cands if t[4] == self.target_id), None)
        if tgt is not None:
            self.lost_since, self.last_box = None, tuple(tgt[:4])
            return tgt, self._ev("following")
        if self.lost_since is None:
            self.lost_since = now
        return None, self._ev("lost")                               # no vehicle re-ID yet: wait for a click

    def _last_center(self, W, H):
        if self.last_box is None:
            return (W or 0) / 2, (H or 0) / 2
        return (self.last_box[0] + self.last_box[2]) / 2, (self.last_box[1] + self.last_box[3]) / 2

    def _ev(self, e):
        self.last_event = e
        return e

    def lost_for(self, now):
        return 0.0 if self.lost_since is None else now - self.lost_since

    def last_side(self, W):
        """-1 if the target was last seen left of centre, +1 if right (which way to turn when searching)."""
        if self.last_box is None or not W:
            return 1
        return -1 if (self.last_box[0] + self.last_box[2]) / 2 < W / 2 else 1


def now_s():
    return time.time()
