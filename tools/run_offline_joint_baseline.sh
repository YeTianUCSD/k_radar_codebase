#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
CHECKPOINT_PATH="${2:-}"
REPO_ROOT="${REPO_ROOT:-/home/code/hyperradar/k_radar_codebase}"
PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/baselines/ASF_v2_0_10scenes_offline_60_40.yml}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/OfflineJoint/10scenes_60_40}"
RUN_STAMP="${RUN_STAMP:-$(date +%y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-offline_joint_10scenes_normal_20epoch_${RUN_STAMP}}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${RESULT_ROOT}}"
JOB_DIR="${JOB_DIR:-${RESULT_ROOT}/${RUN_NAME}_job}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${JOB_DIR}/scene_eval}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FULL_EVAL_EVERY="${FULL_EVAL_EVERY:-1}"
CONF_THR="${CONF_THR:-0.3}"
BEST_METRIC_CLS="${BEST_METRIC_CLS:-auto}"
BEST_METRIC_KIND="${BEST_METRIC_KIND:-3d}"
BEST_METRIC_IOUS="${BEST_METRIC_IOUS:-0.3 0.5}"
BEST_METRIC_CONF="${BEST_METRIC_CONF:-0.3}"
SEQUENCES=(1 5 9 19 22 34 35 38 46 58)
TRAIN_SUMMARY_CSV="${TRAIN_SUMMARY_CSV:-}"
TRAINING_WALL_TIME_SEC="${TRAINING_WALL_TIME_SEC:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"

usage() {
  echo "Usage: $0 {train|eval|all} [best.checkpoint]"
  echo "The script runs in the foreground; use nohup on this wrapper for a background experiment."
}

[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "[ERROR] Config not found: ${CONFIG}" >&2; exit 1; }
case "${MODE}" in train|eval|all) ;; *) usage >&2; exit 2;; esac

cd "${REPO_ROOT}"
mkdir -p "${RUN_OUTPUT_ROOT}" "${EVAL_OUTPUT_ROOT}"
JOB_START_SECONDS=$(date +%s)

run_train() {
  local train_start_seconds train_end_seconds
  train_start_seconds=$(date +%s)
  echo "[TRAIN START] $(date -Iseconds)"
  echo "[CONFIG] ${CONFIG}"
  echo "[RUN_NAME] ${RUN_NAME}"
  "${PYTHON_BIN}" -u main_train_0_args.py \
    --config "${CONFIG}" \
    --output_root "${RUN_OUTPUT_ROOT}" \
    --run_name "${RUN_NAME}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --full_eval_every "${FULL_EVAL_EVERY}" \
    --interval_epoch_model 1 \
    --interval_epoch_util 1 \
    --best_metric_cls "${BEST_METRIC_CLS}" \
    --best_metric_kind "${BEST_METRIC_KIND}" \
    --best_metric_ious ${BEST_METRIC_IOUS} \
    --best_metric_conf "${BEST_METRIC_CONF}" \
    --skip_final_eval
  train_end_seconds=$(date +%s)
  TRAINING_WALL_TIME_SEC=$((train_end_seconds - train_start_seconds))
  echo "[TRAIN DONE] $(date -Iseconds)"
  echo "[TRAIN WALL TIME SEC] ${TRAINING_WALL_TIME_SEC}"
}

resolve_checkpoint() {
  if [[ -n "${CHECKPOINT_PATH}" ]]; then
    [[ -f "${CHECKPOINT_PATH}" ]] || { echo "[ERROR] Checkpoint not found: ${CHECKPOINT_PATH}" >&2; exit 1; }
  else
    local latest_run="" candidate
    for candidate in "${RUN_OUTPUT_ROOT}"/"${RUN_NAME}"_exp_*; do
      if [[ -f "${candidate}/models/best.checkpoint" && ( -z "${latest_run}" || "${candidate}" -nt "${latest_run}" ) ]]; then
        latest_run="${candidate}"
      fi
    done
    [[ -n "${latest_run}" ]] || { echo "[ERROR] No completed run found for ${RUN_NAME}" >&2; exit 1; }
    CHECKPOINT_PATH="${latest_run}/models/best.checkpoint"
  fi

  local experiment_dir
  experiment_dir=$(dirname "$(dirname "${CHECKPOINT_PATH}")")
  if [[ -z "${TRAIN_SUMMARY_CSV}" && -f "${experiment_dir}/train_summary.csv" ]]; then
    TRAIN_SUMMARY_CSV="${experiment_dir}/train_summary.csv"
  fi
  if [[ -z "${TRAINING_WALL_TIME_SEC}" && -f "${JOB_DIR}/training_wall_time_sec.txt" ]]; then
    TRAINING_WALL_TIME_SEC=$(<"${JOB_DIR}/training_wall_time_sec.txt")
  fi
  echo "[BEST CHECKPOINT] ${CHECKPOINT_PATH}"
  echo "[TRAIN SUMMARY CSV] ${TRAIN_SUMMARY_CSV:-unavailable}"
}

run_eval() {
  resolve_checkpoint
  local command=(
    "${PYTHON_BIN}" -u tools/eval_checkpoints.py
    --checkpoints "${CHECKPOINT_PATH}"
    --eval "joint=${CONFIG}"
    --sequences "${SEQUENCES[@]}"
    --output_root "${EVAL_OUTPUT_ROOT}/eval_outputs"
    --summary_csv "${EVAL_OUTPUT_ROOT}/scene_summary.csv"
    --conf_thr "${CONF_THR}"
    --best_metric_cls "${BEST_METRIC_CLS}"
    --best_metric_kind "${BEST_METRIC_KIND}"
    --best_metric_ious ${BEST_METRIC_IOUS}
    --best_metric_conf "${BEST_METRIC_CONF}"
  )
  if [[ -n "${TRAIN_SUMMARY_CSV}" ]]; then
    command+=(--training_summary_csv "${TRAIN_SUMMARY_CSV}")
  fi
  if [[ -n "${TRAINING_WALL_TIME_SEC}" ]]; then
    command+=(--training_wall_time_sec "${TRAINING_WALL_TIME_SEC}")
  fi

  echo "[EVAL START] $(date -Iseconds)"
  "${command[@]}"
  echo "[EVAL DONE] $(date -Iseconds)"
  echo "[SCENE SUMMARY] ${EVAL_OUTPUT_ROOT}/scene_summary.csv"
  echo "[AGGREGATE SUMMARY] ${EVAL_OUTPUT_ROOT}/scene_summary_aggregate.csv"
}

if [[ "${MODE}" == train || "${MODE}" == all ]]; then
  run_train
  mkdir -p "${JOB_DIR}"
  printf "%s\n" "${TRAINING_WALL_TIME_SEC}" > "${JOB_DIR}/training_wall_time_sec.txt"
fi
if [[ "${MODE}" == eval || "${MODE}" == all ]]; then
  run_eval
fi

JOB_END_SECONDS=$(date +%s)
JOB_WALL_TIME_SEC=$((JOB_END_SECONDS - JOB_START_SECONDS))
mkdir -p "${JOB_DIR}"
printf "job_start_epoch_sec=%s\njob_end_epoch_sec=%s\njob_wall_time_sec=%s\ntraining_wall_time_sec=%s\n" \
  "${JOB_START_SECONDS}" "${JOB_END_SECONDS}" "${JOB_WALL_TIME_SEC}" "${TRAINING_WALL_TIME_SEC:-}" \
  > "${JOB_DIR}/job_timing.env"
echo "[TOTAL WALL TIME SEC] ${JOB_WALL_TIME_SEC}"
echo "[JOB TIMING] ${JOB_DIR}/job_timing.env"
