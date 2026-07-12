"""
JARVIS Face Recognition — Detection Metrics
============================================
Pure-function metrics for evaluating face DETECTION results against ground
truth. This module ONLY defines functions — no I/O, no model loading, no cv2,
no CLI. It is imported by benchmark_detection.py and is independently
unit-testable. Geometry (IoU) is reused from face_utils rather than
reimplemented here.

----------------------------------------------------------------------
BOX FORMAT CONVENTION (matches face_utils):
    A box is [x1, y1, x2, y2] in ABSOLUTE PIXELS (top-left / bottom-right
    corners, x1 <= x2, y1 <= y2). A single detection is a dict
    {"box": [x1, y1, x2, y2], "confidence": float}. Ground truth for one
    image is a plain list of boxes [[x1, y1, x2, y2], ...] (no scores).
----------------------------------------------------------------------

A NOTE ON ROC FOR DETECTION (read before trusting the ROC numbers):
    Precision/Recall and Average Precision (AP) are the PRIMARY metrics here —
    they are the industry standard for object/face detection (Pascal VOC, COCO,
    WIDER FACE all report AP/mAP). ROC is included for LEARNING and side-by-side
    comparison ONLY. ROC's x-axis is the False Positive Rate = FP / N_negatives,
    but detection has no fixed, countable set of "negative" examples — a detector
    can emit an unbounded number of false boxes anywhere in an image. There is no
    well-defined denominator for FPR. We therefore APPROXIMATE N_negatives by the
    number of false-positive candidates actually produced (see roc_points), which
    makes the ROC curve dataset/detector-dependent and NOT comparable across runs
    the way a classification ROC would be. Treat AP as the metric of record and
    ROC as an illustrative extra.
"""

import numpy as np

from face_utils import compute_iou  # same directory; reuse, don't reimplement

__all__ = [
    "match_detections",
    "accumulate_pr",
    "average_precision",
    "best_f1_threshold",
    "roc_points",
    "roc_auc",
]


def match_detections(detections, gt_boxes, iou_threshold=0.5):
    """Greedily match one image's predicted detections to its ground-truth boxes.

    Greedy, confidence-ordered matching (the standard Pascal VOC / COCO scheme):
    detections are processed from highest confidence to lowest; each detection
    claims the single unmatched ground-truth box it overlaps most, and counts as
    a true positive (TP) iff that best IoU is >= iou_threshold. Otherwise it is a
    false positive (FP). Each ground-truth box can be claimed by at most one
    detection (the highest-confidence one that qualifies); ground-truth boxes
    left unclaimed after all detections are processed are false negatives (FN).

    Inputs:
        detections (list[dict]): predictions for ONE image, each
            {"box": [x1, y1, x2, y2] absolute pixels, "confidence": float}.
        gt_boxes (list[list]): ground-truth boxes for the SAME image, each
            [x1, y1, x2, y2] in absolute pixels. Unscored.
        iou_threshold (float): minimum IoU for a detection to count as a TP.
            Default 0.5 (the Pascal VOC / WIDER FACE convention).

    Returns:
        tuple (labeled, fn_count):
            labeled (list[tuple[float, bool]]): one entry PER DETECTION, ordered
                by DESCENDING confidence (the order they were matched in), each
                (confidence, is_tp) where is_tp is True for a TP and False for an
                FP. Feed these, aggregated across all images, into accumulate_pr
                / roc_points.
            fn_count (int): number of ground-truth boxes in this image that no
                detection matched (missed faces). Note TP count == len(gt_boxes)
                - fn_count, so FNs are recoverable downstream from total_gt.

    Assumptions:
        - Confidence is only used for ORDERING here; no score threshold is
          applied (sweep the threshold later in accumulate_pr / roc_points).
        - Matching is greedy, not globally optimal (no Hungarian assignment) —
          this matches how VOC/COCO score detectors.
    """
    # Highest confidence first; ties keep their relative input order (stable sort).
    order = sorted(
        range(len(detections)),
        key=lambda i: detections[i].get("confidence", 0.0),
        reverse=True,
    )

    gt_used = [False] * len(gt_boxes)
    labeled = []

    for i in order:
        det = detections[i]
        confidence = float(det.get("confidence", 0.0))

        best_iou = 0.0
        best_gt = -1
        for j, gt in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            iou = compute_iou(det["box"], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        is_tp = best_gt >= 0 and best_iou >= iou_threshold
        if is_tp:
            gt_used[best_gt] = True
        labeled.append((confidence, is_tp))

    fn_count = gt_used.count(False)
    return labeled, fn_count


def accumulate_pr(all_labeled_detections, total_gt):
    """Sweep the confidence threshold to build a precision-recall curve.

    Aggregates per-detection (confidence, is_tp) labels from match_detections
    across the WHOLE dataset and sweeps the score threshold from high to low.
    At each threshold t we "accept" every detection with confidence >= t and
    compute:
        precision = TP / (TP + FP)   — fraction of accepted boxes that are right
        recall    = TP / total_gt    — fraction of all true faces recovered
    Lowering the threshold can only add detections, so recall is monotonically
    non-decreasing along the returned arrays (precision is not monotonic).

    One point is emitted per DISTINCT confidence value (detections sharing a
    score cannot be separated by any threshold, so they are accumulated as a
    group), ordered from the highest threshold (low recall) to the lowest
    threshold (full recall) — i.e. by INCREASING recall, ready to plot or to
    pass straight to average_precision.

    Inputs:
        all_labeled_detections (list[tuple[float, bool]]): every detection from
            every image, each (confidence, is_tp). Concatenate the `labeled`
            lists returned by match_detections across all images.
        total_gt (int): total number of ground-truth faces across all images
            (the recall denominator). Equivalent to sum of TPs + sum of FNs.

    Returns:
        tuple (precisions, recalls, thresholds), all 1-D np.ndarray of equal
        length (one element per distinct confidence threshold), float dtype:
            precisions[k]: precision when accepting all detections with
                confidence >= thresholds[k].
            recalls[k]:    recall at that same threshold, in [0, 1].
            thresholds[k]: the confidence cut-off, ordered HIGH -> LOW (so
                recalls is ascending).
        Returns three empty arrays if there are no detections. If total_gt == 0
        recall is reported as 0.0 throughout (no faces to recover).

    Units: confidences are raw detector scores (typically 0-1); precision and
        recall are dimensionless ratios in [0, 1].
    """
    if len(all_labeled_detections) == 0:
        empty = np.array([], dtype=float)
        return empty, empty.copy(), empty.copy()

    # Sort all detections by descending confidence so a high->low threshold
    # sweep is just a running cumulative sum.
    labeled = sorted(all_labeled_detections, key=lambda d: d[0], reverse=True)
    confidences = np.array([c for c, _ in labeled], dtype=float)
    is_tp = np.array([1.0 if tp else 0.0 for _, tp in labeled], dtype=float)

    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(1.0 - is_tp)

    # One PR point per distinct threshold: detections sharing a confidence
    # cannot be split by any cut-off, so take the cumulative value at the LAST
    # index of each run of equal confidences. `confidences` is descending, so a
    # new distinct value starts wherever it strictly decreases.
    distinct_end = np.where(np.diff(confidences) != 0)[0]
    # np.diff drops the final element; the last run always ends at the last index.
    last_idx = np.append(distinct_end, len(confidences) - 1)

    tp_at = cum_tp[last_idx]
    fp_at = cum_fp[last_idx]
    thresholds = confidences[last_idx]

    # precision = TP / (TP + FP); the denominator is the count accepted so far,
    # which is >= 1 at every point, so this never divides by zero.
    precisions = tp_at / (tp_at + fp_at)
    if total_gt > 0:
        recalls = tp_at / float(total_gt)
    else:
        recalls = np.zeros_like(tp_at)

    return precisions.astype(float), recalls.astype(float), thresholds.astype(float)


def average_precision(precisions, recalls):
    """Average Precision (AP): area under the precision-recall curve.

    Uses the ALL-POINTS interpolation method (Pascal VOC 2010+, also COCO):
    the precision envelope is made monotonically non-increasing as recall grows
    (each point's precision is raised to the max precision achievable at any
    higher recall), then AP is the exact area under that envelope:
        AP = sum_k (recall[k] - recall[k-1]) * precision_envelope[k]
    This is the area under the curve, NOT the older 11-point sampled average.

    Inputs:
        precisions (array-like): precision values, as returned by accumulate_pr.
        recalls (array-like): recall values aligned with `precisions`, expected
            ordered by INCREASING recall (accumulate_pr's output already is).

    Returns:
        float: AP in [0, 1]. Returns 0.0 if the inputs are empty.

    Method notes:
        - Sentinels (recall 0 with precision 0, and recall 1 with precision 0)
          bracket the curve so partial coverage isn't over-credited.
        - Inputs are sorted by recall internally, so order is not assumed beyond
          alignment of the two arrays.
    """
    precisions = np.asarray(precisions, dtype=float)
    recalls = np.asarray(recalls, dtype=float)
    if precisions.size == 0:
        return 0.0

    # Ensure ascending recall (defensive; accumulate_pr already provides this).
    order = np.argsort(recalls, kind="stable")
    recalls = recalls[order]
    precisions = precisions[order]

    # Bracket with sentinels: (r=0, p=0) on the left and (r=1, p=0) on the right.
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Monotonic-decreasing precision envelope: sweep right-to-left taking maxima.
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Sum the area only where recall actually increases.
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))
    return ap


def best_f1_threshold(precisions, recalls, thresholds):
    """Find the confidence threshold that maximizes F1 on a PR curve.

    F1 = 2*P*R / (P+R) is computed at every point on the curve returned by
    accumulate_pr; the point with the highest F1 is the recommended DEPLOYMENT
    operating point (the single confidence cut-off to actually ship), as
    opposed to AP which summarizes the whole curve into one area-under-curve
    number.

    Inputs:
        precisions (array-like): precision values, as returned by accumulate_pr.
        recalls (array-like): recall values aligned with `precisions`.
        thresholds (array-like): confidence thresholds aligned with both,
            as returned by accumulate_pr.

    Returns:
        tuple (threshold, precision, recall, f1) — all floats, taken at the
        point of maximum F1. Points where precision + recall == 0 are given
        F1 = 0.0 (not a division by zero). Returns (0.0, 0.0, 0.0, 0.0) if the
        inputs are empty.
    """
    precisions = np.asarray(precisions, dtype=float)
    recalls = np.asarray(recalls, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if precisions.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    denom = precisions + recalls
    safe_denom = np.where(denom > 0, denom, 1.0)
    f1 = np.where(denom > 0, 2.0 * precisions * recalls / safe_denom, 0.0)

    best_idx = int(np.argmax(f1))
    return (
        float(thresholds[best_idx]),
        float(precisions[best_idx]),
        float(recalls[best_idx]),
        float(f1[best_idx]),
    )


def roc_points(all_labeled_detections, total_gt, total_negatives):
    """Build ROC points (FPR, TPR) by sweeping the confidence threshold.

    Provided for learning/comparison only — see the module-level note: ROC is
    NOT the standard detection metric because FPR's denominator (the number of
    "negatives") is ill-defined for detection. There is no fixed negative set;
    a detector can place a false box anywhere. We APPROXIMATE the negative count
    by `total_negatives`, which the caller should set to the total number of
    false-positive candidates produced across the dataset (i.e. the total count
    of detections that are NOT true positives). With that approximation:
        TPR = TP / total_gt          (= recall; well-defined)
        FPR = FP / total_negatives   (APPROXIMATE; denominator is a stand-in)
    Because the FPR denominator depends on how many false boxes this particular
    detector happened to emit, ROC curves here are NOT comparable across
    detectors the way a classification ROC would be. Report AP, not ROC AUC.

    Inputs:
        all_labeled_detections (list[tuple[float, bool]]): every detection from
            every image, each (confidence, is_tp) (from match_detections).
        total_gt (int): total ground-truth faces (the TPR denominator).
        total_negatives (int): approximate negative count (the FPR denominator);
            recommended value is the total number of FP detections in the
            dataset. See the approximation caveat above.

    Returns:
        tuple (fpr, tpr), two 1-D np.ndarray of equal length, float dtype,
        ordered from the highest threshold to the lowest (so both rise from the
        origin). A (0, 0) point is prepended so the curve starts at the origin.
        Returns two single-element arrays [0.0] if there are no detections.

    Units: TPR and FPR are dimensionless ratios in [0, 1].
    """
    if len(all_labeled_detections) == 0:
        return np.array([0.0]), np.array([0.0])

    labeled = sorted(all_labeled_detections, key=lambda d: d[0], reverse=True)
    confidences = np.array([c for c, _ in labeled], dtype=float)
    is_tp = np.array([1.0 if tp else 0.0 for _, tp in labeled], dtype=float)

    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(1.0 - is_tp)

    # One point per distinct threshold (see accumulate_pr for the rationale).
    distinct_end = np.where(np.diff(confidences) != 0)[0]
    last_idx = np.append(distinct_end, len(confidences) - 1)

    tp_at = cum_tp[last_idx]
    fp_at = cum_fp[last_idx]

    tpr = tp_at / float(total_gt) if total_gt > 0 else np.zeros_like(tp_at)
    fpr = fp_at / float(total_negatives) if total_negatives > 0 else np.zeros_like(fp_at)

    # Start the curve at the origin (threshold above every score => nothing
    # accepted => TP=FP=0).
    fpr = np.concatenate(([0.0], fpr))
    tpr = np.concatenate(([0.0], tpr))
    return fpr.astype(float), tpr.astype(float)


def roc_auc(fpr, tpr):
    """Area under the ROC curve via the trapezoidal rule.

    Inputs:
        fpr (array-like): false-positive-rate values (ROC x-axis).
        tpr (array-like): true-positive-rate values (ROC y-axis), aligned.

    Returns:
        float: AUC in [0, 1]. Returns 0.0 if fewer than two points are given.

    Notes:
        - Points are sorted by increasing FPR before integrating, so input
          order does not matter.
        - Carries the SAME approximation caveat as roc_points: the FPR axis uses
          an approximate negative count, so this AUC is illustrative, not a
          headline metric. Prefer average_precision (AP).
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.size < 2:
        return 0.0

    order = np.argsort(fpr, kind="stable")
    fpr = fpr[order]
    tpr = tpr[order]
    # np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(tpr, fpr))


if __name__ == "__main__":
    # ── Tiny hand-built toy example for eyeballing correctness ──────────
    # Three fake images. Ground truth and detections are chosen so the
    # outcomes are obvious by inspection (boxes are [x1, y1, x2, y2] pixels).
    #
    # Image 1: two GT faces.
    #   - det A overlaps GT0 almost exactly (high conf)  -> TP
    #   - det B overlaps GT1 well            (mid conf)  -> TP
    #   - det C is off in empty space        (low conf)  -> FP
    # Image 2: one GT face.
    #   - det D overlaps it well             (high conf) -> TP
    # Image 3: one GT face, ZERO detections -> that GT is a false negative.
    #
    # Totals: total_gt = 4, TP = 3, FP = 1, FN = 1.

    images = [
        {
            "gt": [[10, 10, 50, 50], [100, 100, 140, 140]],
            "dets": [
                {"box": [11, 11, 51, 51], "confidence": 0.95},   # TP vs GT0
                {"box": [102, 98, 141, 139], "confidence": 0.80}, # TP vs GT1
                {"box": [300, 300, 330, 330], "confidence": 0.40}, # FP
            ],
        },
        {
            "gt": [[60, 60, 100, 100]],
            "dets": [
                {"box": [58, 62, 99, 101], "confidence": 0.90},   # TP
            ],
        },
        {
            "gt": [[200, 200, 240, 240]],
            "dets": [],  # missed face -> FN
        },
    ]

    all_labeled = []
    total_gt = 0
    total_fn = 0
    total_fp = 0
    for idx, img in enumerate(images):
        labeled, fn = match_detections(img["dets"], img["gt"], iou_threshold=0.5)
        all_labeled.extend(labeled)
        total_gt += len(img["gt"])
        total_fn += fn
        total_fp += sum(1 for _, tp in labeled if not tp)
        print(f"image {idx}: labeled={labeled}  FN={fn}")

    total_tp = sum(1 for _, tp in all_labeled if tp)
    print(f"\ntotals: total_gt={total_gt}  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print("(expected: total_gt=4  TP=3  FP=1  FN=1)\n")

    precisions, recalls, thresholds = accumulate_pr(all_labeled, total_gt)
    print("PR sweep (high threshold -> low):")
    for p, r, t in zip(precisions, recalls, thresholds):
        print(f"  thr>={t:.2f}  precision={p:.3f}  recall={r:.3f}")

    ap = average_precision(precisions, recalls)
    print(f"\nAverage Precision (AP) = {ap:.3f}")

    best_thr, best_p, best_r, best_f1 = best_f1_threshold(precisions, recalls, thresholds)
    print(f"\nF1-max operating point: thr>={best_thr:.2f}  "
          f"precision={best_p:.3f}  recall={best_r:.3f}  F1={best_f1:.3f}")

    # total_negatives approximated by the number of FP candidates (here, 1).
    fpr, tpr = roc_points(all_labeled, total_gt, total_negatives=total_fp)
    auc = roc_auc(fpr, tpr)
    print("\nROC points (approx -- see module note):")
    for f, t in zip(fpr, tpr):
        print(f"  fpr={f:.3f}  tpr={t:.3f}")
    print(f"ROC AUC (illustrative only) = {auc:.3f}")
