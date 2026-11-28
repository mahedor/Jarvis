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


def test_refused_candidate_rechecked_slowly_not_never():
    lock = lockon.TargetLock(P)
    lock.pick(1)
    t = build_gallery(lock)
    tracks = [trk(7, x=300)]
    for _ in range(lockon.CHECKS_PER_TRACK):
        assert lock.wants_embedding(tracks, t) == [7]
        lock.update(tracks, {7: B}, t)
        t += lockon.RECHECK_S + 0.01
    assert lock.wants_embedding(tracks, t) == []                                 # quick checks used up
    t += lockon.RETRY_S
    assert lock.wants_embedding(tracks, t) == [7]                                # Michael 11-21: look again later


def test_target_refused_at_first_is_taken_back_on_a_later_look():
    # 10-29 replay: at a high cutoff he was refused 3x on one track ID and then never looked at again
    lock = lockon.TargetLock(P)
    lock.pick(1)
    t = build_gallery(lock)
    tracks = [trk(9, x=500)]
    for _ in range(lockon.CHECKS_PER_TRACK):                                     # blurry first looks: no match
        lock.update(tracks, {i: B for i in lock.wants_embedding(tracks, t)}, t)
        t += lockon.RECHECK_S + 0.01
    t += lockon.RETRY_S
    want = lock.wants_embedding(tracks, t)
    tgt, ev = lock.update(tracks, {i: A2 for i in want}, t)                      # clear view now
    assert ev == "reacquired_reid" and tgt[4] == 9


def test_search_ignores_fragment_boxes():
    # 10-29 replay: the only non-target boxes scoring >= 0.70 were fragments (low confidence, on the edge, or
    # inside another person's box). Search candidates now pass the same clean-sample rule as the gallery.
    lock = lockon.TargetLock(P)
    lock.pick(1)
    t = build_gallery(lock)
    W, H = 1280, 720
    assert lock.wants_embedding([trk(7, x=300, conf=0.36)], t, W, H) == []       # low confidence
    assert lock.wants_embedding([trk(7, x=0)], t, W, H) == []                    # cut off by the frame edge
    assert lock.wants_embedding([trk(7, x=300, h=20)], t, W, H) == []            # tiny
    assert lock.wants_embedding([trk(7, x=300), trk(8, x=320, w=20, h=60)], t, W, H) == []  # box inside a box
    assert lock.wants_embedding([trk(7, x=300)], t, W, H) == [7]


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


def test_no_silent_reid_fallback():
    """Michael 11-22: no 'auto' that quietly swaps OSNet for the histogram."""
    import pytest
    with pytest.raises(ValueError):
        lockon.make_embedder("auto")
    e, note = lockon.make_embedder("hist")
    assert "NOT for flying" in note


def test_target_id_moved_onto_a_car_is_lost_and_never_followed():
    """10-29 replay 2:02-2:19: the tracker (class-blind IoU matching) gave lost person ID 43 to the car he stood by.
    A person target whose ID sits on a car is LOST (no FOLLOWING, no gallery sample), target_track() says None,
    and re-acquisition still only looks at people."""
    lock = lockon.TargetLock(P)
    lock.pick(1, tracks=[trk(1)])
    t = build_gallery(lock, tid=1, vec=A)
    car = trk(1, x=105, w=120, h=80, cls=2)                    # same ID 1, now a car box
    for k in range(5):
        want = lock.wants_embedding([car], t + k)
        assert want == []                                     # never embeds the car
        tgt, ev = lock.update([car], {}, t + k)
        assert tgt is None and ev == "lost" and lock.target_id == 1
        assert lock.target_track([car]) is None
    him = trk(9, x=400)                                       # he comes back as a person under a new ID
    want = lock.wants_embedding([car, him], t + 6)
    assert want == [9]
    tgt, ev = lock.update([car, him], {9: A}, t + 6)
    assert ev == "reacquired_reid" and tgt[4] == 9 and lock.target_track([car, him])[4] == 9


def test_target_track_respects_vehicle_target_kind():
    lock = lockon.TargetLock(P)
    lock.pick(5, tracks=[trk(5, cls=2)])                       # a clicked car
    assert lock.target_track([trk(5, cls=2)])[4] == 5
    assert lock.target_track([trk(5, cls=0)]) is None          # its ID moved onto a person: not the target


def test_draw_does_not_mark_a_car_holding_the_target_id_as_target():
    import replay
    img = np.zeros((400, 600, 3), np.uint8)
    car = (100.0, 100.0, 300.0, 200.0, 43, 2, 0.9)
    c = dict(yaw=0, pitch=0, gimbal_delta=0.0, dist=0.0, method="searching")
    names = {0: "person", 2: "car"}

    def green(im):                                                # left edge of the box drawn in target green?
        return bool((im[150, 99:102] == (0, 255, 0)).all(axis=-1).any())
    assert green(replay.draw(img.copy(), [car], names, 43, c, 0, {}, 0, 0, "you"))   # old rule: drawn as TARGET
    assert not green(replay.draw(img.copy(), [car], names, 43, c, 0, {}, 0, 0, "you", target_track=None))
    assert green(replay.draw(img.copy(), [car], names, 43, c, 0, {}, 0, 0, "you", target_track=car))
