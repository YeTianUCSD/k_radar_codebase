#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/code/hyperradar/k_radar_codebase"
PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
INIT_MODEL="${INIT_MODEL:-/home/code/hyperradar/k_radar_codebase/results/Superposition/v3/train_seq1_20epoch_test_seq1_psp_v3_exp_260620_091955/models/best.checkpoint}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/ContextRouting/window${WINDOW_SIZE}_rp2}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-${ROOT_DIR}/configs/context_adaptation/auto_context_rp2.yml}"
if [[ -z "${STREAM_CONFIG:-}" ]]; then
    case "${WINDOW_SIZE}" in
        32)
            STREAM_CONFIG="${ROOT_DIR}/configs/context_adaptation/stream_normal_seq58_windows32.yml"
            ;;
        80)
            STREAM_CONFIG="${ROOT_DIR}/configs/context_adaptation/stream_normal_seq58_windows.yml"
            ;;
        *)
            echo "Set STREAM_CONFIG when WINDOW_SIZE is not 32 or 80." >&2
            exit 1
            ;;
    esac
fi
BOOTSTRAP_CHECKPOINT="${BOOTSTRAP_CHECKPOINT:-${RESULT_ROOT}/bootstrap_normal_from_seq1.context.checkpoint}"
RUN_NAME="${RUN_NAME:-routing_only_normal_seq58_window${WINDOW_SIZE}}"

if [[ ! -f "${INIT_MODEL}" ]]; then
    echo "INIT_MODEL does not exist: ${INIT_MODEL}" >&2
    exit 1
fi
if [[ ! -f "${STREAM_CONFIG}" ]]; then
    echo "STREAM_CONFIG does not exist: ${STREAM_CONFIG}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${RESULT_ROOT}"
cd "${ROOT_DIR}"

"${PYTHON_BIN}" -u tools/context_adaptation/bootstrap_normal_context.py \
    --config ./configs/ASF_v2_0_seq1_psp.yml \
    --init_model "${INIT_MODEL}" \
    --context_config "${CONTEXT_CONFIG}" \
    --output "${BOOTSTRAP_CHECKPOINT}" \
    --split train \
    --batch_size 1 \
    --num_workers "${NUM_WORKERS:-0}" \
    --max_samples "${BOOTSTRAP_MAX_SAMPLES:--1}" \
    --base_context_name normal \
    --base_model_context_name seq1 \
    --allow_incompatible_checkpoint

"${PYTHON_BIN}" -u tools/context_adaptation/online_adapt_auto_context.py \
    --config ./configs/ASF_v2_0_seq58_adapt_psp.yml \
    --init_context_checkpoint "${BOOTSTRAP_CHECKPOINT}" \
    --context_config "${CONTEXT_CONFIG}" \
    --stream_config "${STREAM_CONFIG}" \
    --output_root "${RESULT_ROOT}" \
    --run_name "${RUN_NAME}" \
    --batch_size 1 \
    --num_workers "${NUM_WORKERS:-0}" \
    --max_steps -1 \
    --save_every_updates 0 \
    --routing_only

RUN_DIR="$(find "${RESULT_ROOT}" -maxdepth 1 -type d -name "${RUN_NAME}_exp_*" -print | sort -V | tail -n 1)"
if [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/routing_log.csv" ]]; then
    echo "Routing output was not found for run ${RUN_NAME}." >&2
    exit 1
fi

"${PYTHON_BIN}" -u tools/context_adaptation/summarize_routing_accuracy.py \
    --routing_csv "${RUN_DIR}/routing_log.csv" \
    --output "${RUN_DIR}/routing_metrics.yml" \
    --base_label normal \
    --target_label seq58 \
    --base_context_id 0 \
    --ignore_transition_frames "${IGNORE_TRANSITION_FRAMES:-0}"

echo "RUN_DIR=${RUN_DIR}"
echo "ROUTING_LOG=${RUN_DIR}/routing_log.csv"
echo "ROUTING_METRICS=${RUN_DIR}/routing_metrics.yml"
