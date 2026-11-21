"""Tests for lockon.py: gallery building, verified re-acquisition, refusing strangers, fallback without re-ID."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lockon  # noqa: E402

P = {0}
A = np.array([1, 0, 0, 0], np.float32)                     # the target's appearance
B = np.array([0, 1, 0, 0], np.float32)                     # a stranger
A2 = np.array([0.95, 0.31, 0, 0], np.float32); A2 /= np.linalg.norm(A2)   # target, different pose/lighting


def trk(tid, x=100, w=60, h=150, conf=0.9, cls=0):
    return (x, 100, x + w, 100 + h, tid, cls, conf)


def build_gallery(lock, tid=1, vec=A, n=3, x=100):
    t = 0.0
    for _ in range(n):
        tracks = [trk(tid, x)]
        want = lock.wants_embedding(tracks, t)
        lock.update(tracks, {i: vec for i in want}, t)
        t += lockon.ADD_EVERY_S
    return t


def test_first_lock_is_biggest_person_and_gallery_fills():
    lock = lockon.TargetLock(P, auto_lock=True)                     # auto-lock: fake-drone tests only
    tgt, ev = lock.update([trk(1, w=40), trk(2, x=400, w=80)], {}, 0.0)
    assert ev == "first_lock" and tgt[4] == 2
    build_gallery(lock, tid=2, x=400)
    assert len(lock.gallery) == 3


def test_no_gallery_sample_when_overlapping_someone():
    lock = lockon.TargetLock(P)
    lock.pick(1)
    assert lock.wants_embedding([trk(1, x=100), trk(2, x=130)], 0.0) == []      # boxes overlap -> no sample
    assert lock.wants_embedding([trk(1, x=100), trk(2, x=400)], 0.0) == [1]


def test_no_sample_from_low_confidence_or_tiny_box():
    lock = lockon.TargetLock(P)
    lock.pick(1)
    assert lock.wants_embedding([trk(1, conf=0.3)], 0.0) == []
    assert lock.wants_embedding([trk(1, h=20)], 0.0) == []


def test_gallery_rate_limited_and_capped():
    lock = lockon.TargetLock(P)
    lock.pick(1)
    lock.update([trk(1)], {1: A}, 0.0)
    assert lock.wants_embedding([trk(1)], 0.5) == []                            # too soon
    t = 1.0
    for _ in range(lockon.GALLERY_MAX + 5):
        lock.update([trk(1)], {1: A}, t); t += 1
    assert len(lock.gallery) == lockon.GALLERY_MAX


def test_reacquires_target_under_new_id_and_ignores_stranger():
    lock = lockon.TargetLock(P)
    lock.pick(1)
    t = build_gallery(lock)
    tracks = [trk(7, x=300), trk(9, x=500)]                                      # id 1 gone; 7 = stranger, 9 = target
    want = lock.wants_embedding(tracks, t)
    assert set(want) == {7, 9}
    tgt, ev = lock.update(tracks, {7: B, 9: A2}, t)
    assert ev == "reacquired_reid" and tgt[4] == 9 and lock.target_id == 9


def test_refuses_stranger_even_if_nearest():
    lock = lockon.TargetLock(P)
    lock.pick(1)
    t = build_gallery(lock, x=100)
    tracks = [trk(7, x=105)]                                                     # right where the target was
    tgt, ev = lock.update(tracks, {7: B}, t)
    assert tgt is None and ev == "lost" and lock.target_id == 1


def test_ambiguous_match_waits():
    lock = lockon.TargetLock(P)
    lock.pick(1)
    t = build_gallery(lock)
    tgt, ev = lock.update([trk(7, x=300), trk(8, x=500)], {7: A2, 8: A2}, t)     # two equally good -> don't guess
    assert tgt is None and ev == "lost"


def test_candidate_checked_limited_times():
    lock = lockon.TargetLock(P)
    lock.pick(1)
    t = build_gallery(lock)
    tracks = [trk(7, x=300)]
    for _ in range(lockon.CHECKS_PER_TRACK):
        assert lock.wants_embedding(tracks, t) == [7]
        lock.update(tracks, {7: B}, t)
        t += lockon.RECHECK_S + 0.01
    assert lock.wants_embedding(tracks, t) == []                                 # gave up on this stranger


def test_never_guesses_without_gallery():
    lock = lockon.TargetLock(P)                                    # Michael 11-20: never lock onto anyone new
    lock.pick(1, tracks=[trk(1, x=100)])
    tgt, ev = lock.update([trk(7, x=110), trk(8, x=600)], {}, 0.0)
    assert tgt is None and ev == "lost" and lock.target_id == 1


def test_nothing_locks_without_a_click():
    lock = lockon.TargetLock(P)
    tgt, ev = lock.update([trk(1, x=100), trk(2, x=600)], {}, 0.0)
    assert tgt is None and ev == "no_target" and lock.target_id is None


def test_non_people_ignored():
    lock = lockon.TargetLock(P, auto_lock=True)
    tgt, ev = lock.update([trk(1, cls=2)], {}, 0.0)
    assert tgt is None and ev == "searching"


def test_last_side_for_search_direction():
    lock = lockon.TargetLock(P)
    lock.pick(1, tracks=[trk(1, x=100)])
    assert lock.last_side(1280) == -1
    lock.pick(1, tracks=[trk(1, x=1000)])
    assert lock.last_side(1280) == 1


def test_hist_embedder_tells_colours_apart():
    import cv2  # noqa: F401
    img = np.zeros((300, 400, 3), np.uint8)
    img[:, :200] = (0, 0, 255)      # red person region
    img[:, 200:] = (255, 0, 0)      # blue person region
    e = lockon.HistEmbedder()
    v = e(img, [(10, 10, 100, 290), (110, 10, 190, 290), (210, 10, 390, 290)])
    assert v.shape[0] == 3
    assert v[0] @ v[1] > 0.99 and v[0] @ v[2] < 0.2


def test_no_gallery_sample_when_box_touches_frame_edge():
    lock = lockon.TargetLock(P)
    lock.pick(1)
    assert lock.wants_embedding([trk(1, x=0)], 0.0, 1280, 720) == []          # cut off on the left
    assert lock.wants_embedding([trk(1, x=500, h=700)], 0.0, 1280, 720) == []  # runs off the bottom (close-up)
    assert lock.wants_embedding([trk(1, x=500)], 0.0, 1280, 720) == [1]
