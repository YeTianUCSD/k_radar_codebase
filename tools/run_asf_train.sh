#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-train_seq1_20epoch_test_seq1}"
EPOCHS="${2:-20}"
FULL_EVAL_EVERY="${3:-1}"

REPO_ROOT="${REPO_ROOT:-/home/code/hyperradar/k_radar_codebase}"
CONFIG="${CONFIG:-./configs/ASF_v2_0_seq1.yml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}"
BEST_METRIC_CLS="${BEST_METRIC_CLS:-auto}"
BEST_METRIC_KIND="${BEST_METRIC_KIND:-3d}"
BEST_METRIC_IOUS="${BEST_METRIC_IOUS:-0.3 0.5}"
BEST_METRIC_CONF="${BEST_METRIC_CONF:-0.3}"

export PATH=/home/miniconda/bin:$PATH
source /home/miniconda/etc/profile.d/conda.sh
conda activate kradar_asf

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

echo "[START] ${RUN_NAME} at $(date)"
echo "[PWD] $(pwd)"
echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES}"

EXTRA_ARGS=()
if [[ "$SKIP_FINAL_EVAL" == "1" ]]; then
  EXTRA_ARGS+=(--skip_final_eval)
fi

python main_train_0_args.py \
  --config "$CONFIG" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --full_eval_every "$FULL_EVAL_EVERY" \
  --best_metric_cls "$BEST_METRIC_CLS" \
  --best_metric_kind "$BEST_METRIC_KIND" \
  --best_metric_ious $BEST_METRIC_IOUS \
  --best_metric_conf "$BEST_METRIC_CONF" \
  "${EXTRA_ARGS[@]}"

echo "[DONE] ${RUN_NAME} finished at $(date)"
