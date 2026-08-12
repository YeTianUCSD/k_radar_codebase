#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/run_superposition_pipeline.sh all
#   bash tools/run_superposition_pipeline.sh train
#   bash tools/run_superposition_pipeline.sh adapt
#   bash tools/run_superposition_pipeline.sh eval
#   bash tools/run_superposition_pipeline.sh eval_seq58
#   bash tools/run_superposition_pipeline.sh eval_seq1
#
# Resume/override examples:
#   SEQ1_CHECKPOINT=/path/to/seq1/best.checkpoint bash tools/run_superposition_pipeline.sh adapt
#   ADAPTED_CHECKPOINT=/path/to/seq58/best.checkpoint bash tools/run_superposition_pipeline.sh eval

MODE="${1:-all}"
case "$MODE" in
  all|train|adapt|eval|eval_seq58|eval_seq1) ;;
  *)
    echo "Usage: $0 {all|train|adapt|eval|eval_seq58|eval_seq1}" >&2
    exit 2
    ;;
esac

REPO_ROOT="${REPO_ROOT:-/home/code/hyperradar/k_radar_codebase}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/Superposition/v3}"
PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
CONDA_SH="${CONDA_SH:-/home/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-kradar_asf}"

SEQ1_CONFIG="${SEQ1_CONFIG:-./configs/ASF_v2_0_seq1_psp.yml}"
SEQ58_ADAPT_CONFIG="${SEQ58_ADAPT_CONFIG:-./configs/ASF_v2_0_seq58_adapt_psp.yml}"
SEQ58_EVAL_CONFIG="${SEQ58_EVAL_CONFIG:-./configs/ASF_v2_0_seq58_eval_psp.yml}"
SEQ1_EVAL_CONFIG="${SEQ1_EVAL_CONFIG:-./configs/ASF_v2_0_seq1_eval_psp.yml}"

SEQ1_RUN_NAME="${SEQ1_RUN_NAME:-train_seq1_20epoch_test_seq1_psp_v3}"
SEQ58_RUN_NAME="${SEQ58_RUN_NAME:-online_seq58_fuser_head_residual_only_from_seq1_psp_best_v3}"
SEQ58_EVAL_NAME="${SEQ58_EVAL_NAME:-eval_seq58_scene_weight_residual_best_on_seq58_v3}"
SEQ1_EVAL_NAME="${SEQ1_EVAL_NAME:-eval_seq1_scene_weight_residual_best_on_seq1_v3}"

EPOCHS="${EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
ONLINE_BATCH_SIZE="${ONLINE_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FULL_EVAL_EVERY="${FULL_EVAL_EVERY:-1}"
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

# Optional explicit checkpoints for resuming individual stages.
SEQ1_CHECKPOINT="${SEQ1_CHECKPOINT:-}"
ADAPTED_CHECKPOINT="${ADAPTED_CHECKPOINT:-}"
SEQ58_RUN_DIR="${SEQ58_RUN_DIR:-}"

export PATH="/home/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

[[ -x "$PYTHON_BIN" ]] || { echo "[ERROR] Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$CONDA_SH" ]] || { echo "[ERROR] Conda setup not found: $CONDA_SH" >&2; exit 1; }

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$REPO_ROOT"
mkdir -p "$RESULT_ROOT"

for required_file in \
  "$SEQ1_CONFIG" \
  "$SEQ58_ADAPT_CONFIG" \
  "$SEQ58_EVAL_CONFIG" \
  "$SEQ1_EVAL_CONFIG" \
  tools/online_adapt_asf_psp.py \
  tools/eval_scene_context_asf.py; do
  [[ -f "$required_file" ]] || { echo "[ERROR] Required file not found: $required_file" >&2; exit 1; }
done

read -r -a BEST_METRIC_IOUS_ARGS <<< "$BEST_METRIC_IOUS"
STAMP="$(/bin/date +%y%m%d_%H%M%S)"
SEQ1_LOG="${RESULT_ROOT}/${SEQ1_RUN_NAME}_${STAMP}.log"
SEQ58_LOG="${RESULT_ROOT}/${SEQ58_RUN_NAME}_${STAMP}.log"

find_latest_run() {
  local run_name="$1"
  local latest="" candidate
  for candidate in "$RESULT_ROOT"/"${run_name}"_exp_*; do
    if [[ -d "$candidate" && ( -z "$latest" || "$candidate" -nt "$latest" ) ]]; then
      latest="$candidate"
    fi
  done
  [[ -n "$latest" ]] || return 1
  printf '%s\n' "$latest"
}

resolve_seq1_checkpoint() {
  if [[ -n "$SEQ1_CHECKPOINT" ]]; then
    [[ -f "$SEQ1_CHECKPOINT" ]] || { echo "[ERROR] Seq1 checkpoint not found: $SEQ1_CHECKPOINT" >&2; exit 1; }
    return
  fi

  local run_dir
  run_dir="$(find_latest_run "$SEQ1_RUN_NAME")" || {
    echo "[ERROR] No Seq1 run found for ${SEQ1_RUN_NAME} under ${RESULT_ROOT}" >&2
    echo "Run the train stage first or set SEQ1_CHECKPOINT=/path/to/best.checkpoint" >&2
    exit 1
  }
  SEQ1_CHECKPOINT="${run_dir}/models/best.checkpoint"
  [[ -f "$SEQ1_CHECKPOINT" ]] || { echo "[ERROR] Seq1 best checkpoint not found: $SEQ1_CHECKPOINT" >&2; exit 1; }
}

resolve_adapted_checkpoint() {
  if [[ -n "$ADAPTED_CHECKPOINT" ]]; then
    [[ -f "$ADAPTED_CHECKPOINT" ]] || { echo "[ERROR] Adapted checkpoint not found: $ADAPTED_CHECKPOINT" >&2; exit 1; }
    if [[ -z "$SEQ58_RUN_DIR" ]]; then
      SEQ58_RUN_DIR="$(cd "$(dirname "$ADAPTED_CHECKPOINT")/.." && pwd)"
    fi
    return
  fi

  if [[ -z "$SEQ58_RUN_DIR" ]]; then
    SEQ58_RUN_DIR="$(find_latest_run "$SEQ58_RUN_NAME")" || {
      echo "[ERROR] No Seq58 run found for ${SEQ58_RUN_NAME} under ${RESULT_ROOT}" >&2
      echo "Run the adapt stage first or set ADAPTED_CHECKPOINT=/path/to/best.checkpoint" >&2
      exit 1
    }
  fi
  ADAPTED_CHECKPOINT="${SEQ58_RUN_DIR}/models/best.checkpoint"
  [[ -f "$ADAPTED_CHECKPOINT" ]] || { echo "[ERROR] Seq58 best checkpoint not found: $ADAPTED_CHECKPOINT" >&2; exit 1; }
}

run_seq1_train() {
  echo "[START] ${SEQ1_RUN_NAME} at $(date)"
  echo "[CONFIG] ${SEQ1_CONFIG}"
  echo "[RESULT_ROOT] ${RESULT_ROOT}"
  echo "[SCENE_CONTEXT] seq1"

  "$PYTHON_BIN" -u main_train_0_args.py \
    --config "$SEQ1_CONFIG" \
    --output_root "$RESULT_ROOT" \
    --run_name "$SEQ1_RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --full_eval_every "$FULL_EVAL_EVERY" \
    --best_metric_cls "$BEST_METRIC_CLS" \
    --best_metric_kind "$BEST_METRIC_KIND" \
    --best_metric_ious "${BEST_METRIC_IOUS_ARGS[@]}" \
    --best_metric_conf "$BEST_METRIC_CONF" \
    --skip_final_eval

  echo "[DONE] ${SEQ1_RUN_NAME} at $(date)"
}

run_seq58_adaptation() {
  resolve_seq1_checkpoint
  echo "[START] ${SEQ58_RUN_NAME} at $(date)"
  echo "[CONFIG] ${SEQ58_ADAPT_CONFIG}"
  echo "[INIT_MODEL] ${SEQ1_CHECKPOINT}"
  echo "[TRAINABLE] fuser_head"
  echo "[SHARED_WEIGHT_POLICY] residual_only"
  echo "[TRAIN_SCENE] seq58"
  echo "[EVAL_SCENE] seq58"

  "$PYTHON_BIN" -u tools/online_adapt_asf_psp.py \
    --config "$SEQ58_ADAPT_CONFIG" \
    --init_model "$SEQ1_CHECKPOINT" \
    --output_root "$RESULT_ROOT" \
    --run_name "$SEQ58_RUN_NAME" \
    --trainable fuser_head \
    --shared_weight_policy residual_only \
    --batch_size "$ONLINE_BATCH_SIZE" \
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
    --best_metric_ious "${BEST_METRIC_IOUS_ARGS[@]}" \
    --best_metric_conf "$BEST_METRIC_CONF" \
    --scene_context_train seq58 \
    --scene_context_eval seq58

  echo "[DONE] ${SEQ58_RUN_NAME} at $(date)"
}

run_scene_eval() {
  local eval_name="$1"
  local eval_config="$2"
  local scene_name="$3"
  local run_dir="${SEQ58_RUN_DIR}/${eval_name}"
  local eval_log="${run_dir}/run.log"

  mkdir -p "$run_dir"
  {
    echo "[START] ${eval_name} at $(date)"
    echo "[MODEL] ${ADAPTED_CHECKPOINT}"
    echo "[EVAL_CONFIG] ${eval_config}"
    echo "[SCENE_CONTEXT] ${scene_name}"
    echo "[RUN_DIR] ${run_dir}"

    "$PYTHON_BIN" -u tools/eval_scene_context_asf.py \
      --checkpoints "$ADAPTED_CHECKPOINT" \
      --eval "${scene_name}=${eval_config}" \
      --scene "${scene_name}=${scene_name}" \
      --output_root "${run_dir}/eval_outputs" \
      --summary_csv "${run_dir}/summary.csv" \
      --conf_thr "$CONF_THR" \
      --best_metric_cls "$BEST_METRIC_CLS" \
      --best_metric_kind "$BEST_METRIC_KIND" \
      --best_metric_ious "${BEST_METRIC_IOUS_ARGS[@]}" \
      --best_metric_conf "$BEST_METRIC_CONF"

    echo "[DONE] ${eval_name} at $(date)"
  } > "$eval_log" 2>&1

  echo "[EVAL_LOG] ${eval_log}"
  echo "[EVAL_SUMMARY] ${run_dir}/summary.csv"
}

if [[ "$MODE" == all || "$MODE" == train ]]; then
  run_seq1_train > "$SEQ1_LOG" 2>&1
  resolve_seq1_checkpoint
  echo "[SEQ1_LOG] $SEQ1_LOG"
fi

if [[ "$MODE" == all || "$MODE" == adapt ]]; then
  run_seq58_adaptation > "$SEQ58_LOG" 2>&1
  resolve_adapted_checkpoint
  echo "[SEQ58_LOG] $SEQ58_LOG"
fi

if [[ "$MODE" == all || "$MODE" == eval || "$MODE" == eval_seq58 || "$MODE" == eval_seq1 ]]; then
  resolve_adapted_checkpoint
fi

if [[ "$MODE" == all || "$MODE" == eval || "$MODE" == eval_seq58 ]]; then
  run_scene_eval "$SEQ58_EVAL_NAME" "$SEQ58_EVAL_CONFIG" seq58
fi

if [[ "$MODE" == all || "$MODE" == eval || "$MODE" == eval_seq1 ]]; then
  run_scene_eval "$SEQ1_EVAL_NAME" "$SEQ1_EVAL_CONFIG" seq1
fi

if [[ -n "$SEQ1_CHECKPOINT" ]]; then
  echo "[SEQ1_CHECKPOINT] $SEQ1_CHECKPOINT"
fi
if [[ -n "$ADAPTED_CHECKPOINT" ]]; then
  echo "[ADAPTED_CHECKPOINT] $ADAPTED_CHECKPOINT"
fi
