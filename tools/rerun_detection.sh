#!/usr/bin/env bash
#
# Sequential re-run of the detection benchmark across all three datasets.
#
# SEQUENTIAL ON PURPOSE. Two detectors running at once contend for the same
# cores, and every latency number in the results becomes a function of what
# else happened to be scheduled beside it. Do not "speed this up" by
# backgrounding the steps.
#
# STOPPING IT HAS TO ACTUALLY STOP IT. An earlier version of this script piped
# each run through `tail`, which made the pipeline's exit status tail's (always
# 0) — so `set -e` never fired, killing a python child just advanced the script
# to the next dataset, and two orphaned benchmarks ended up running in
# parallel. Hence: pipefail, no pipe on the python call, and a trap that takes
# the whole process group down on Ctrl-C or SIGTERM.
#
# Usage:
#   bash tools/rerun_detection.sh                 # min_box_size=20 (default)
#   MIN_BOX_SIZE=0 bash tools/rerun_detection.sh  # unfiltered
#   DATASETS="widerface" bash tools/rerun_detection.sh
#
# Progress goes to logs/rerun_detection_<stamp>.log; the summary tables are in
# there too. Results append to results/benchmarks_detection.json as usual.

set -euo pipefail

cd "$(dirname "$0")/.."

# min_box_size is the setting that silently changed what the numbers meant once
# already: at 0, WIDER FACE counts 39,697 ground-truth boxes and YOLO scores
# AP 0.642; at 20 it counts 16,072 and scores 0.864. It is recorded in the
# artifact now, but pick it deliberately.
MIN_BOX_SIZE="${MIN_BOX_SIZE:-20}"
DATASETS="${DATASETS:-widerface mafa fddb}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="logs/rerun_detection_${STAMP}.log"

# Kill the whole process group, so stopping the wrapper stops the python child.
trap 'echo "[$(date -Is)] interrupted — stopping" | tee -a "$LOG"; kill 0' INT TERM

wider_dir="data/WIDER_FACE/WIDER_val/images"
wider_ann="data/WIDER_FACE/wider_face_split/wider_face_val_bbx_gt.txt"
mafa_dir="data/MAFA/test-images/images"
mafa_ann="data/MAFA/MAFA-Label-Test/LabelTestAll.mat"
fddb_dir="data/FDDB/originalPics"
fddb_ann="data/FDDB/FDDB-folds-all-ellipseList.txt"

echo "=== detection re-run  min_box_size=${MIN_BOX_SIZE}  datasets: ${DATASETS}" | tee -a "$LOG"
echo "=== log: ${LOG}"
echo "=== expect roughly an hour per large dataset on CPU; nothing else should"
echo "    be running, including the web app, or the latency numbers are noise."

for dataset in $DATASETS; do
    case "$dataset" in
        widerface) data_dir="$wider_dir"; annotations="$wider_ann" ;;
        mafa)      data_dir="$mafa_dir";  annotations="$mafa_ann"  ;;
        fddb)      data_dir="$fddb_dir";  annotations="$fddb_ann"  ;;
        *) echo "unknown dataset: $dataset" | tee -a "$LOG"; exit 2 ;;
    esac

    echo "" | tee -a "$LOG"
    echo "--- ${dataset} starting $(date -Is)" | tee -a "$LOG"

    # No pipe here: the exit status must be python's, not a filter's.
    python tools/benchmark_detection.py \
        --dataset "$dataset" \
        --min-box-size "$MIN_BOX_SIZE" \
        --data-dir "$data_dir" \
        --annotations "$annotations" >>"$LOG" 2>&1

    echo "--- ${dataset} done $(date -Is)" | tee -a "$LOG"
    tail -n 14 "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== all datasets complete $(date -Is)" | tee -a "$LOG"
