#!/usr/bin/env bash
set -euo pipefail

INIT_MODEL="${1:-}"
TRAINABLE="${2:-fuser_head}"
[[ -n "$INIT_MODEL" ]] || { echo "Usage: $0 /path/to/checkpoint [cls|head|fuser_head|full]" >&2; exit 2; }
case "$TRAINABLE" in cls|head|fuser_head|full) ;; *) echo "[ERROR] Invalid trainable scope: $TRAINABLE" >&2; exit 2;; esac

REPO_ROOT="${REPO_ROOT:-/home/code/hyperradar/k_radar_codebase}"
PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
CONFIG="${CONFIG:-./configs/ASF_v2_0_seq58_adapt.yml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results}"
RUN_NAME="${RUN_NAME:-online_seq58_${TRAINABLE}_from_seq1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_STEPS="${MAX_STEPS:--1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-0}"
EVAL_EVERY_UPDATES="${EVAL_EVERY_UPDATES:-50}"
SAVE_EVERY_UPDATES="${SAVE_EVERY_UPDATES:-50}"
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
[[ -f "$INIT_MODEL" ]] || { echo "[ERROR] Checkpoint not found: $INIT_MODEL" >&2; exit 1; }
cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

echo "[START] Sequence 58 online adaptation at $(date)"
echo "[INIT_MODEL] $INIT_MODEL"
echo "[TRAINABLE] $TRAINABLE"
echo "[RUN_NAME] $RUN_NAME"

"$PYTHON_BIN" -u tools/online_adapt_asf.py \
  --config "$CONFIG" \
  --init_model "$INIT_MODEL" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --trainable "$TRAINABLE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_steps "$MAX_STEPS" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --grad_clip "$GRAD_CLIP" \
  --eval_every_updates "$EVAL_EVERY_UPDATES" \
  --save_every_updates "$SAVE_EVERY_UPDATES" \
  --conf_thr "$CONF_THR" \
  --best_metric_cls "$BEST_METRIC_CLS" \
  --best_metric_kind "$BEST_METRIC_KIND" \
  --best_metric_ious $BEST_METRIC_IOUS \
  --best_metric_conf "$BEST_METRIC_CONF"

echo "[DONE] Sequence 58 online adaptation at $(date)"
