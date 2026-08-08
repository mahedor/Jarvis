"""
JARVIS Face Recognition — Stage 2 Recognition Metrics
=====================================================
Pure-function logic for the face-ENCODER benchmark: enroll/probe splitting,
gallery aggregation strategies, and the open-set identification metrics
(TAR@FAR, rank-1, F1 accept/reject threshold, d-prime, distributions).

Like detection_metrics.py this module is I/O-FREE: no cv2, no disk, no model
loading, no CLI. Everything is a function of its arguments and numpy only, so
it is unit-testable in milliseconds (see tests/test_recognition_metrics.py) and
imported by benchmark_recognition.py, which owns all the I/O.

----------------------------------------------------------------------
SCORE CONVENTION:
    Embeddings are L2-normalized before they reach this module, so a similarity
    is a plain dot product == cosine similarity in [-1, 1]. Higher = more alike.
    "Accept" always means score >= threshold.

    genuine  : a probe scored against its OWN identity's gallery (one per probe).
    impostor : a probe scored against a DIFFERENT identity's gallery (cross pairs).
    stranger : an off-dataset face (LFW) scored against a gallery.
    non-mate : the union used as the false-accept set (impostor + stranger),
               i.e. everything that SHOULD be rejected.
----------------------------------------------------------------------
"""

import numpy as np

__all__ = [
    "l2_normalize",
    "medoid_index",
    "source_image_key",
    "split_enroll_probe",
    "AGGREGATIONS",
    "gallery_mean",
    "gallery_medoid",
    "gallery_multiref",
    "score_single",
    "score_multiref",
    "tar_at_far",
    "rank1_accuracy",
    "best_f1_threshold",
    "dprime",
    "summarize_scores",
    "histogram",
]


# ══════════════════════════════════════════════════════════════════
# Vector helpers (numpy-only; deliberately NOT imported from collect_faces,
# whose module import pulls in cv2 + sklearn — this module stays light).
# ══════════════════════════════════════════════════════════════════

def l2_normalize(embeddings):
    """Scale each embedding to unit length so dot product == cosine similarity.

    Inputs:
        embeddings (np.ndarray): (N, D), or a single (D,) vector.
    Returns:
        np.ndarray: float32, same shape, each row unit length. A zero-length
        row is passed through as zeros (cannot be normalized) rather than
        producing NaN.
    """
    array = np.asarray(embeddings, dtype=np.float32)
    single = array.ndim == 1
    if single:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normalized = (array / safe).astype(np.float32)
    return normalized[0] if single else normalized


def medoid_index(embeddings):
    """Index of the member with the smallest total distance to all others.

    The medoid is an ACTUAL member of the set (unlike a mean), so it maps back
    to a real enroll crop. Used by the 'medoid' aggregation.

    Inputs:
        embeddings (np.ndarray): (N, D).
    Returns:
        int: index of the medoid. Returns 0 for N == 1.
    Raises:
        ValueError: if there are no embeddings.
    """
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("medoid_index() needs a non-empty (N, D) array")
    if array.shape[0] == 1:
        return 0
    diffs = array[:, None, :] - array[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return int(np.argmin(distances.sum(axis=1)))


# ══════════════════════════════════════════════════════════════════
# Enroll / probe splitting — IMAGE-DISJOINT.
# ══════════════════════════════════════════════════════════════════

def source_image_key(filename):
    """The source-photo identity of a crop, from its provenance filename.

    collect_faces names crops "<flattened source path>__f<face index>.jpg", so
    every crop from one photo shares the prefix before "__f". Splitting on that
    prefix keeps all crops of a single photo together, so the same photo can
    never contribute to BOTH the enroll and probe sets (which would leak
    near-duplicate pixels and inflate genuine scores).

    Inputs:
        filename (str): a crop filename (basename or path).
    Returns:
        str: the source-photo key (the part before the last "__f", extension
        stripped). Falls back to the stem if the "__f" marker is absent.
    """
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    marker = stem.rsplit("__f", 1)
    return marker[0] if len(marker) == 2 else stem


def split_enroll_probe(source_keys, enroll_fraction, rng):
    """Split one identity's crops into enroll/probe by WHOLE source photo.

    Photos (not individual crops) are shuffled and partitioned, so the split is
    image-disjoint. Reproducible for a given `rng` seed.

    Inputs:
        source_keys (sequence[str]): source_image_key of each crop, in crop
            order. Indices returned index back into this same order.
        enroll_fraction (float): fraction of PHOTOS (not crops) to enroll,
            in (0, 1). The count is rounded so at least one photo lands on each
            side whenever there are >= 2 photos.
        rng (np.random.Generator): seeded generator for a reproducible shuffle.
    Returns:
        tuple (enroll_idx, probe_idx): two lists of int crop indices. BOTH are
        empty when the identity has fewer than 2 distinct source photos (it
        cannot be split image-disjoint — the caller should skip that person).
    """
    keys = list(source_keys)
    groups = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)

    unique_keys = sorted(groups)  # sorted first -> deterministic before shuffle
    if len(unique_keys) < 2:
        return [], []

    order = rng.permutation(len(unique_keys))
    shuffled = [unique_keys[i] for i in order]

    n_enroll = int(round(enroll_fraction * len(shuffled)))
    n_enroll = max(1, min(n_enroll, len(shuffled) - 1))  # keep both sides non-empty

    enroll_idx, probe_idx = [], []
    for rank, key in enumerate(shuffled):
        (enroll_idx if rank < n_enroll else probe_idx).extend(groups[key])
    return sorted(enroll_idx), sorted(probe_idx)


# ══════════════════════════════════════════════════════════════════
# Gallery aggregation strategies — pluggable pure functions.
# Each strategy is a (build_fn, score_fn) pair:
#   build_fn(enroll_embs (K, D)) -> gallery representation
#   score_fn(probe_embs (N, D), gallery_repr) -> (N,) similarity per probe
# All embeddings are assumed L2-normalized on the way in.
# ══════════════════════════════════════════════════════════════════

def gallery_mean(enroll_embs):
    """Mean of the enroll vectors, RE-normalized to unit length (one template)."""
    return l2_normalize(np.asarray(enroll_embs, dtype=np.float32).mean(axis=0))


def gallery_medoid(enroll_embs):
    """The single most central enroll vector (already unit length)."""
    array = np.asarray(enroll_embs, dtype=np.float32)
    return array[medoid_index(array)]


def gallery_multiref(enroll_embs):
    """Keep ALL enroll vectors; the gallery is the full (K, D) matrix."""
    return np.asarray(enroll_embs, dtype=np.float32)


def score_single(probe_embs, gallery_vec):
    """Cosine of each probe against a single (D,) gallery template -> (N,)."""
    return np.asarray(probe_embs, dtype=np.float32) @ np.asarray(gallery_vec, dtype=np.float32)


def score_multiref(probe_embs, gallery_mat):
    """MAX cosine of each probe against any of the (K, D) gallery vectors -> (N,)."""
    sims = np.asarray(probe_embs, dtype=np.float32) @ np.asarray(gallery_mat, dtype=np.float32).T
    return sims.max(axis=1)


# --aggregation selects one; "all" iterates every entry.
AGGREGATIONS = {
    "mean-renormalize": (gallery_mean, score_single),
    "medoid": (gallery_medoid, score_single),
    "multi-reference": (gallery_multiref, score_multiref),
}


# ══════════════════════════════════════════════════════════════════
# Open-set metrics.
# ══════════════════════════════════════════════════════════════════

def tar_at_far(genuine, nonmate, far_target):
    """True-accept rate at a target false-accept rate.

    Sets the acceptance threshold so at most `far_target` of the non-mate
    scores are accepted, then reports the fraction of genuine scores accepted at
    that same threshold. This is the headline open-set operating point: "if I
    tolerate 1% of impostors/strangers getting in, what fraction of the real
    person's probes do I let in?"

    Inputs:
        genuine (array-like): genuine similarity scores (should be accepted).
        nonmate (array-like): impostor + stranger scores (should be rejected).
        far_target (float): tolerated false-accept fraction, e.g. 0.01 for 1%.
    Returns:
        dict {"tar", "threshold", "far_achieved"}: TAR in [0, 1], the similarity
        threshold, and the FAR actually realised at it (<= far_target up to
        score ties). All zeros if either input is empty.
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    nonmate = np.asarray(nonmate, dtype=np.float64)
    if genuine.size == 0 or nonmate.size == 0:
        return {"tar": 0.0, "threshold": 0.0, "far_achieved": 0.0}

    n = nonmate.size
    k = int(np.floor(far_target * n))  # how many false accepts the budget allows
    desc = np.sort(nonmate)[::-1]
    if k <= 0:
        # No false accepts allowed: sit just ABOVE the top non-mate score.
        threshold = float(np.nextafter(desc[0], np.inf))
    else:
        # Accept scores >= the k-th highest non-mate -> exactly k of them get in.
        threshold = float(desc[k - 1])

    tar = float(np.mean(genuine >= threshold))
    far_achieved = float(np.mean(nonmate >= threshold))
    return {"tar": tar, "threshold": threshold, "far_achieved": far_achieved}


def rank1_accuracy(score_matrix, true_indices):
    """Closed-set rank-1 identification accuracy on probes.

    For each probe, the predicted identity is the gallery with the highest
    score; it is correct iff that argmax is the probe's own identity.

    Inputs:
        score_matrix (np.ndarray): (num_probes, num_galleries), score of each
            probe against each identity's gallery.
        true_indices (array-like): (num_probes,) column index of each probe's
            OWN gallery.
    Returns:
        float: fraction correct in [0, 1]. 0.0 if there are no probes.
    """
    matrix = np.asarray(score_matrix, dtype=np.float64)
    truth = np.asarray(true_indices)
    if matrix.size == 0 or matrix.shape[0] == 0:
        return 0.0
    predicted = np.argmax(matrix, axis=1)
    return float(np.mean(predicted == truth))


def best_f1_threshold(genuine, nonmate):
    """Similarity threshold that maximizes F1 for the accept/reject decision.

    Mirrors detection_metrics.best_f1_threshold, but operates directly on two
    score populations instead of a precision-recall curve. Positives are the
    genuine scores (we WANT to accept them); negatives are the non-mate scores
    (we want to reject them). Every distinct observed score is tried as the
    "accept if score >= t" cut-off, and the highest-F1 one is returned as the
    recommended deployment threshold.

    Inputs:
        genuine (array-like): genuine similarity scores.
        nonmate (array-like): impostor + stranger similarity scores.
    Returns:
        tuple (threshold, precision, recall, f1) — all floats, at the point of
        maximum F1. Points where precision + recall == 0 get F1 = 0.0 (never a
        division by zero). Returns (0.0, 0.0, 0.0, 0.0) if either input is empty.
    """
    genuine = np.asarray(genuine, dtype=np.float64)
    nonmate = np.asarray(nonmate, dtype=np.float64)
    if genuine.size == 0 or nonmate.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    thresholds = np.unique(np.concatenate([genuine, nonmate]))
    gen_sorted = np.sort(genuine)
    non_sorted = np.sort(nonmate)

    # At each threshold t: accept everything >= t. searchsorted(side="left")
    # gives the count strictly below t, so (size - that) counts accepted.
    tp = genuine.size - np.searchsorted(gen_sorted, thresholds, side="left")
    fp = nonmate.size - np.searchsorted(non_sorted, thresholds, side="left")

    accepted = tp + fp
    precision = np.divide(tp, accepted, out=np.zeros_like(tp, dtype=float), where=accepted > 0)
    recall = tp / float(genuine.size)

    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    best = int(np.argmax(f1))
    return float(thresholds[best]), float(precision[best]), float(recall[best]), float(f1[best])


def dprime(positive, negative):
    """Separation between two score distributions (higher = better separated).

    d' = (mean_pos - mean_neg) / sqrt(0.5 * (var_pos + var_neg)). A single,
    scale-free number summarizing how far apart genuine and non-mate scores sit.

    Inputs:
        positive (array-like): genuine scores.
        negative (array-like): impostor or stranger scores.
    Returns:
        float: d-prime. 0.0 if either input is empty or both variances are 0.
    """
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        return 0.0
    pooled_var = 0.5 * (positive.var() + negative.var())
    if pooled_var == 0:
        return 0.0
    return float((positive.mean() - negative.mean()) / np.sqrt(pooled_var))


def summarize_scores(values):
    """Compact summary of a score population, for the report and the JSON log.

    Inputs:
        values (array-like): similarity scores.
    Returns:
        dict | None: count, mean, std, min, p5, p25, p50, p75, p95, max — or
        None if `values` is empty.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    p5, p25, p50, p75, p95 = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(array.max()),
    }


def histogram(values, bins=40, value_range=(-1.0, 1.0)):
    """Fixed-range histogram so distributions are comparable across encoders.

    The range is pinned to cosine's [-1, 1] by default (not derived from the
    data) so histograms from different encoders/aggregations line up bin-for-bin.

    Inputs:
        values (array-like): similarity scores.
        bins (int): number of bins.
        value_range (tuple[float, float]): (low, high) edge span.
    Returns:
        dict {"edges": [...len bins+1], "counts": [...len bins]} — or None if
        `values` is empty.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    counts, edges = np.histogram(array, bins=bins, range=value_range)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}
