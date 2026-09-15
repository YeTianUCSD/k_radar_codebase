#!/usr/bin/env bash
set -euo pipefail

# Train independent per-scene upper-bound models serially on one GPU.
# Every epoch runs full validation and participates in best-checkpoint selection.

REPO_ROOT="${REPO_ROOT:-/home/code/hyperradar/k_radar_codebase}"
PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/baselines/ASF_v2_0_10scenes_offline_60_40.yml}"
RUN_STAMP="${RUN_STAMP:-$(date +%y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/results/SceneCeiling/10scenes_60_40/all_10scenes_${RUN_STAMP}}"
RUNS_DIR="${RUN_DIR}/runs"
LOGS_DIR="${RUN_DIR}/logs"
SUMMARY_CSV="${RUN_DIR}/ceiling_summary.csv"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FULL_EVAL_EVERY="${FULL_EVAL_EVERY:-1}"
CONF_THR="${CONF_THR:-0.3}"
BEST_METRIC_CLS="${BEST_METRIC_CLS:-auto}"
BEST_METRIC_KIND="${BEST_METRIC_KIND:-3d}"
BEST_METRIC_IOUS="${BEST_METRIC_IOUS:-0.3 0.5}"
if (( $# > 0 )); then
  SEQUENCES=("$@")
else
  SEQUENCES=(1 35 46 19 58 5 22 34 9 38)
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"

[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "[ERROR] Config not found: ${CONFIG}" >&2; exit 1; }

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${RUN_DIR}/completed"
cd "${REPO_ROOT}"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  printf '%s\n' 'sequence,status,epochs,best_epoch,best_score,wall_time_sec,experiment_dir,best_checkpoint,train_summary,error_log,time_to_best_sec' > "${SUMMARY_CSV}"
fi

echo "[CEILING START] $(date -Iseconds)"
echo "[RUN DIR] ${RUN_DIR}"
echo "[SEQUENCES] ${SEQUENCES[*]}"
echo "[CONFIG] ${CONFIG}"

for sequence in "${SEQUENCES[@]}"; do
  sequence="${sequence#seq}"
  completion_file="${RUN_DIR}/completed/seq${sequence}.env"
  if [[ -f "${completion_file}" ]]; then
    echo "[SKIP] seq${sequence} already completed: ${completion_file}"
    continue
  fi

  run_name="ceiling_seq${sequence}_normal_${EPOCHS}epoch_${RUN_STAMP}"
  scene_log="${LOGS_DIR}/seq${sequence}.log"
  start_seconds=$(date +%s)
  echo "[SCENE START] seq${sequence} $(date -Iseconds)"

  if "${PYTHON_BIN}" -u main_train_0_args.py \
    --config "${CONFIG}" \
    --sequences "${sequence}" \
    --output_root "${RUNS_DIR}" \
    --run_name "${run_name}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --full_eval_every "${FULL_EVAL_EVERY}" \
    --interval_epoch_model 1 \
    --interval_epoch_util 1 \
    --best_metric_cls "${BEST_METRIC_CLS}" \
    --best_metric_kind "${BEST_METRIC_KIND}" \
    --best_metric_ious ${BEST_METRIC_IOUS} \
    --best_metric_conf "${CONF_THR}" \
    --skip_final_eval \
    2>&1 | tee "${scene_log}"; then
    train_status=0
  else
    train_status=$?
  fi

  end_seconds=$(date +%s)
  wall_time_sec=$((end_seconds - start_seconds))
  if (( train_status != 0 )); then
    printf 'seq%s,failed_train,%s,,,%s,,,,%s,\n' \
      "${sequence}" "${EPOCHS}" "${wall_time_sec}" "${scene_log}" >> "${SUMMARY_CSV}"
    echo "[SCENE FAILED] seq${sequence} exit=${train_status} wall=${wall_time_sec}s; continuing with next scene"
    continue
  fi

  experiment_dir=""
  for candidate in "${RUNS_DIR}/${run_name}"_exp_*; do
    if [[ -d "${candidate}" && (-z "${experiment_dir}" || "${candidate}" -nt "${experiment_dir}") ]]; then
      experiment_dir="${candidate}"
    fi
  done
  if [[ -z "${experiment_dir}" ]]; then
    printf 'seq%s,failed_artifacts,%s,,,%s,,,,%s,\n' \
      "${sequence}" "${EPOCHS}" "${wall_time_sec}" "${scene_log}" >> "${SUMMARY_CSV}"
    echo "[SCENE FAILED] Experiment directory not found for seq${sequence}; continuing with next scene" >&2
    continue
  fi

  best_checkpoint="${experiment_dir}/models/best.checkpoint"
  best_summary="${experiment_dir}/best_summary.txt"
  train_summary="${experiment_dir}/train_summary.csv"
  if [[ ! -f "${best_checkpoint}" || ! -f "${best_summary}" || ! -f "${train_summary}" ]]; then
    printf 'seq%s,failed_artifacts,%s,,,%s,%s,,,%s,\n' \
      "${sequence}" "${EPOCHS}" "${wall_time_sec}" "${experiment_dir}" "${scene_log}" >> "${SUMMARY_CSV}"
    echo "[SCENE FAILED] Best checkpoint, best summary, or train summary missing for seq${sequence}; continuing with next scene" >&2
    continue
  fi

  best_epoch=$(awk -F': ' '$1 == "best_epoch" {print $2}' "${best_summary}")
  best_score=$(awk -F': ' '$1 == "score" {print $2}' "${best_summary}")
  if [[ -z "${best_epoch}" || -z "${best_score}" ]]; then
    printf 'seq%s,failed_summary,%s,,,%s,%s,%s,%s,%s,\n' \
      "${sequence}" "${EPOCHS}" "${wall_time_sec}" "${experiment_dir}" \
      "${best_checkpoint}" "${train_summary}" "${scene_log}" >> "${SUMMARY_CSV}"
    echo "[SCENE FAILED] Invalid best summary for seq${sequence}; continuing with next scene" >&2
    continue
  fi

  read -r best_epoch_time_sec all_epoch_time_sec < <(
    awk -F, -v best_epoch="${best_epoch}" '
      NR == 1 {
        for (i = 1; i <= NF; i++) {
          if ($i == "epoch_time_sec") time_col = i
        }
        next
      }
      time_col {
        all_time += $time_col
        if (($1 + 0) <= (best_epoch + 0)) best_time += $time_col
      }
      END {printf "%.3f %.3f\n", best_time, all_time}
    ' "${train_summary}"
  )
  time_to_best_sec=$(awk \
    -v wall_time="${wall_time_sec}" \
    -v best_epoch_time="${best_epoch_time_sec}" \
    -v all_epoch_time="${all_epoch_time_sec}" '
      BEGIN {
        setup_time = wall_time - all_epoch_time
        if (setup_time < 0) setup_time = 0
        printf "%.3f", setup_time + best_epoch_time
      }
    ')

  printf 'seq%s,complete,%s,%s,%s,%s,%s,%s,%s,,%s\n' \
    "${sequence}" "${EPOCHS}" "${best_epoch}" "${best_score}" "${wall_time_sec}" \
    "${experiment_dir}" "${best_checkpoint}" "${train_summary}" "${time_to_best_sec}" >> "${SUMMARY_CSV}"
  printf 'sequence=seq%s\ntime_to_best_sec=%s\nwall_time_sec=%s\nexperiment_dir=%s\nbest_checkpoint=%s\n' \
    "${sequence}" "${time_to_best_sec}" "${wall_time_sec}" "${experiment_dir}" "${best_checkpoint}" > "${completion_file}"
  echo "[SCENE DONE] seq${sequence} best=${best_score}@epoch${best_epoch} time_to_best=${time_to_best_sec}s wall=${wall_time_sec}s $(date -Iseconds)"
done

echo "[CEILING DONE] $(date -Iseconds)"
echo "[SUMMARY] ${SUMMARY_CSV}"
