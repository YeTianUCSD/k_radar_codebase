#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/code/hyperradar/k_radar_codebase
RESULT_ROOT="${REPO_ROOT}/results/Superposition"
PYTHON_BIN=/home/miniconda/envs/kradar_asf/bin/python
CONDA_SH=/home/miniconda/etc/profile.d/conda.sh
CONDA_ENV=kradar_asf

export PATH=/home/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${REPO_ROOT}"
mkdir -p "${RESULT_ROOT}"

SEQ1_RUN_NAME=train_seq1_20epoch_test_seq1_psp
SEQ58_RUN_NAME=online_seq58_fuser_head_from_seq1_psp_best
STAMP="$(/bin/date +%y%m%d_%H%M%S)"

SEQ1_LOG="${RESULT_ROOT}/${SEQ1_RUN_NAME}_${STAMP}.log"
SEQ58_LOG="${RESULT_ROOT}/${SEQ58_RUN_NAME}_${STAMP}.log"

SEQ1_BEST_MODEL=""
SEQ58_RUN_DIR=""
SEQ58_BEST_MODEL=""

run_seq1_train() {
  echo "[START] ${SEQ1_RUN_NAME} at $(date)"
  "${PYTHON_BIN}" -u main_train_0_args.py \
    --config ./configs/ASF_v2_0_seq1_psp.yml \
    --output_root "${RESULT_ROOT}" \
    --run_name "${SEQ1_RUN_NAME}" \
    --epochs 20 \
    --batch_size 2 \
    --num_workers 0 \
    --full_eval_every 1 \
    --best_metric_cls auto \
    --best_metric_kind 3d \
    --best_metric_ious 0.3 0.5 \
    --best_metric_conf 0.3 \
    --skip_final_eval
  echo "[DONE] ${SEQ1_RUN_NAME} at $(date)"
}

resolve_seq1_best() {
  SEQ1_BEST_MODEL="$(ls -td "${RESULT_ROOT}/${SEQ1_RUN_NAME}"_exp_* 2>/dev/null | head -n 1)/models/best.checkpoint"
  if [[ ! -f "${SEQ1_BEST_MODEL}" ]]; then
    echo "[ERROR] seq1 best checkpoint not found: ${SEQ1_BEST_MODEL}" >&2
    exit 1
  fi
}

run_seq58_online_update() {
  echo "[START] ${SEQ58_RUN_NAME} at $(date)"
  echo "[INIT_MODEL] ${SEQ1_BEST_MODEL}"
  "${PYTHON_BIN}" -u tools/online_adapt_asf_psp.py \
    --config ./configs/ASF_v2_0_seq58_adapt_psp.yml \
    --init_model "${SEQ1_BEST_MODEL}" \
    --output_root "${RESULT_ROOT}" \
    --run_name "${SEQ58_RUN_NAME}" \
    --trainable fuser_head \
    --batch_size 1 \
    --num_workers 0 \
    --max_steps -1 \
    --optimizer adamw \
    --lr 0.0001 \
    --weight_decay 0.0 \
    --grad_clip 0 \
    --eval_every_updates 50 \
    --save_every_updates 50 \
    --conf_thr 0.3 \
    --best_metric_cls auto \
    --best_metric_kind 3d \
    --best_metric_ious 0.3 0.5 \
    --best_metric_conf 0.3 \
    --scene_context_train seq58 \
    --scene_context_eval seq58
  echo "[DONE] ${SEQ58_RUN_NAME} at $(date)"
}

resolve_seq58_best() {
  SEQ58_RUN_DIR="$(ls -td "${RESULT_ROOT}/${SEQ58_RUN_NAME}"_exp_* 2>/dev/null | head -n 1)"
  SEQ58_BEST_MODEL="${SEQ58_RUN_DIR}/models/best.checkpoint"
  if [[ ! -f "${SEQ58_BEST_MODEL}" ]]; then
    echo "[ERROR] seq58 best checkpoint not found: ${SEQ58_BEST_MODEL}" >&2
    exit 1
  fi
}

run_scene_eval() {
  local eval_name="$1"
  local eval_config="$2"
  local scene_name="$3"
  local run_dir="${SEQ58_RUN_DIR}/${eval_name}"

  mkdir -p "${run_dir}"
  echo "[START] ${eval_name} at $(date)"
  echo "[MODEL] ${SEQ58_BEST_MODEL}"
  "${PYTHON_BIN}" -u tools/eval_scene_context_asf.py \
    --checkpoints "${SEQ58_BEST_MODEL}" \
    --eval "${scene_name}=${eval_config}" \
    --scene "${scene_name}=${scene_name}" \
    --output_root "${run_dir}/eval_outputs" \
    --summary_csv "${run_dir}/summary.csv" \
    --conf_thr 0.3 \
    --best_metric_cls auto \
    --best_metric_kind 3d \
    --best_metric_ious 0.3 0.5 \
    --best_metric_conf 0.3
  echo "[DONE] ${eval_name} at $(date)"
}

run_seq1_train > "${SEQ1_LOG}" 2>&1
resolve_seq1_best

run_seq58_online_update > "${SEQ58_LOG}" 2>&1
resolve_seq58_best

run_scene_eval "eval_seq58_shared_psp_best_on_seq58" "./configs/ASF_v2_0_seq58_eval_psp.yml" "seq58"
run_scene_eval "eval_seq1_shared_psp_best_on_seq1" "./configs/ASF_v2_0_seq1_eval_psp.yml" "seq1"

echo "SEQ1_LOG=${SEQ1_LOG}"
echo "SEQ58_LOG=${SEQ58_LOG}"
echo "SEQ1_BEST_MODEL=${SEQ1_BEST_MODEL}"
echo "SEQ58_BEST_MODEL=${SEQ58_BEST_MODEL}"
