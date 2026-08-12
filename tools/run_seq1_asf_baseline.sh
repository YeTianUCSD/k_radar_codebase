#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
CHECKPOINT_DIR="${2:-}"
REPO_ROOT="${REPO_ROOT:-/home/code/hyperradar/k_radar_codebase}"
PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
CONFIG="${CONFIG:-./configs/ASF_v2_0_seq1.yml}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results}"
RUN_NAME="${RUN_NAME:-train_seq1_20epoch_test_seq1}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${RESULT_ROOT}/${RUN_NAME}}"
EVAL_NAME="${EVAL_NAME:-eval_train_seq1_checkpoints_on_seq1}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${RESULT_ROOT}/${EVAL_NAME}}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FULL_EVAL_EVERY="${FULL_EVAL_EVERY:-1}"
CONF_THR="${CONF_THR:-0.3}"
BEST_METRIC_CLS="${BEST_METRIC_CLS:-auto}"
BEST_METRIC_KIND="${BEST_METRIC_KIND:-3d}"
BEST_METRIC_IOUS="${BEST_METRIC_IOUS:-0.3 0.5}"
BEST_METRIC_CONF="${BEST_METRIC_CONF:-0.3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -x "$PYTHON_BIN" ]] || { echo "[ERROR] Python not found: $PYTHON_BIN" >&2; exit 1; }
case "$MODE" in train|eval|all) ;; *) echo "Usage: $0 {train|eval|all} [checkpoint_dir]" >&2; exit 2;; esac

cd "$REPO_ROOT"
mkdir -p "$RUN_OUTPUT_ROOT" "$EVAL_OUTPUT_ROOT"

run_train() {
  echo "[START] Sequence 1 ASF training at $(date)"
  "$PYTHON_BIN" -u main_train_0_args.py \
    --config "$CONFIG" \
    --output_root "$RUN_OUTPUT_ROOT" \
    --run_name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --full_eval_every "$FULL_EVAL_EVERY" \
    --interval_epoch_model 1 \
    --interval_epoch_util 1 \
    --best_metric_cls "$BEST_METRIC_CLS" \
    --best_metric_kind "$BEST_METRIC_KIND" \
    --best_metric_ious $BEST_METRIC_IOUS \
    --best_metric_conf "$BEST_METRIC_CONF" \
    --skip_final_eval
  echo "[DONE] Sequence 1 ASF training at $(date)"
}

resolve_checkpoint_dir() {
  [[ -n "$CHECKPOINT_DIR" ]] && return
  local latest_run="" candidate
  for candidate in "$RUN_OUTPUT_ROOT"/"${RUN_NAME}"_exp_*; do
    if [[ -d "$candidate/models" && ( -z "$latest_run" || "$candidate" -nt "$latest_run" ) ]]; then
      latest_run="$candidate"
    fi
  done
  [[ -n "$latest_run" ]] || { echo "[ERROR] No run found under $RUN_OUTPUT_ROOT" >&2; exit 1; }
  CHECKPOINT_DIR="$latest_run/models"
}

run_eval() {
  resolve_checkpoint_dir
  [[ -d "$CHECKPOINT_DIR" ]] || { echo "[ERROR] Checkpoint directory not found: $CHECKPOINT_DIR" >&2; exit 1; }
  echo "[START] Sequence 1 checkpoint evaluation at $(date)"
  "$PYTHON_BIN" -u tools/eval_checkpoints.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --eval "seq1=$CONFIG" \
    --output_root "$EVAL_OUTPUT_ROOT/eval_outputs" \
    --summary_csv "$EVAL_OUTPUT_ROOT/summary.csv" \
    --conf_thr "$CONF_THR" \
    --best_metric_cls "$BEST_METRIC_CLS" \
    --best_metric_kind "$BEST_METRIC_KIND" \
    --best_metric_ious $BEST_METRIC_IOUS \
    --best_metric_conf "$BEST_METRIC_CONF"
  echo "[SUMMARY] $EVAL_OUTPUT_ROOT/summary.csv"
}

if [[ "$MODE" == train || "$MODE" == all ]]; then
  run_train
fi
if [[ "$MODE" == eval || "$MODE" == all ]]; then
  run_eval
fi
