#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/code/hyperradar/k_radar_codebase"
PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/ContextAdaptation/v1}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-${ROOT_DIR}/configs/context_adaptation/auto_context_rp2.yml}"
NORMAL_CONFIG="${NORMAL_CONFIG:-${ROOT_DIR}/configs/ASF_v2_0_seq1_psp.yml}"
SEQ58_ADAPT_CONFIG="${SEQ58_ADAPT_CONFIG:-${ROOT_DIR}/configs/ASF_v2_0_seq58_adapt_psp.yml}"
SEQ58_EVAL_CONFIG="${SEQ58_EVAL_CONFIG:-${ROOT_DIR}/configs/ASF_v2_0_seq58_eval_psp.yml}"
NORMAL_EVAL_CONFIG="${NORMAL_EVAL_CONFIG:-${ROOT_DIR}/configs/ASF_v2_0_seq1_eval_psp.yml}"
RETURN_STREAM_CONFIG="${RETURN_STREAM_CONFIG:-${ROOT_DIR}/configs/context_adaptation/stream_seq58_return_normal.yml}"

TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-train_normal_psp_auto_context}"
ADAPT_RUN_NAME="${ADAPT_RUN_NAME:-online_auto_context_seq58}"
RETURN_RUN_NAME="${RETURN_RUN_NAME:-online_auto_context_seq58_return_normal}"
BOOTSTRAP_CHECKPOINT="${BOOTSTRAP_CHECKPOINT:-${RESULT_ROOT}/bootstrap_normal_rp2.context.checkpoint}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${RESULT_ROOT}"
cd "${ROOT_DIR}"

resolve_normal_model() {
    if [[ -n "${NORMAL_MODEL:-}" ]]; then
        if [[ ! -f "${NORMAL_MODEL}" ]]; then
            echo "NORMAL_MODEL does not exist: ${NORMAL_MODEL}" >&2
            return 1
        fi
        printf '%s\n' "${NORMAL_MODEL}"
        return 0
    fi

    local checkpoint
    checkpoint="$(find "${RESULT_ROOT}" -path "*/${TRAIN_RUN_NAME}_exp_*/models/best.checkpoint" -type f -print 2>/dev/null | sort -V | tail -n 1)"
    if [[ -z "${checkpoint}" ]]; then
        checkpoint="$(find "${RESULT_ROOT}" -path "*/${TRAIN_RUN_NAME}_exp_*/models/model_*.pt" -type f -print 2>/dev/null | sort -V | tail -n 1)"
    fi
    if [[ -z "${checkpoint}" ]]; then
        echo "No normal model was found. Run the train stage or set NORMAL_MODEL." >&2
        return 1
    fi
    printf '%s\n' "${checkpoint}"
}

resolve_adapted_checkpoint() {
    local source_run_name="${1:-${EVAL_RUN_NAME:-${ADAPT_RUN_NAME}}}"
    if [[ -n "${ADAPTED_CHECKPOINT:-}" ]]; then
        if [[ ! -f "${ADAPTED_CHECKPOINT}" ]]; then
            echo "ADAPTED_CHECKPOINT does not exist: ${ADAPTED_CHECKPOINT}" >&2
            return 1
        fi
        printf '%s\n' "${ADAPTED_CHECKPOINT}"
        return 0
    fi

    local checkpoint
    checkpoint="$(find "${RESULT_ROOT}" -path "*/${source_run_name}_exp_*/models/last.context.checkpoint" -type f -print 2>/dev/null | sort -V | tail -n 1)"
    if [[ -z "${checkpoint}" ]]; then
        echo "No adapted checkpoint was found for run ${source_run_name}." >&2
        echo "Run the matching adapt stage or set ADAPTED_CHECKPOINT." >&2
        return 1
    fi
    printf '%s\n' "${checkpoint}"
}

run_train() {
    "${PYTHON_BIN}" -u tools/context_adaptation/train_normal_psp.py \
        --config "${NORMAL_CONFIG}" \
        --context_config "${CONTEXT_CONFIG}" \
        --output_root "${RESULT_ROOT}" \
        --run_name "${TRAIN_RUN_NAME}" \
        --epochs "${NORMAL_EPOCHS:-20}" \
        --batch_size "${NORMAL_BATCH_SIZE:-2}" \
        --num_workers "${NUM_WORKERS:-0}" \
        --full_eval_every "${FULL_EVAL_EVERY:-1}" \
        --best_metric_cls auto \
        --best_metric_kind 3d \
        --best_metric_ious 0.3 0.5 \
        --best_metric_conf 0.3 \
        --skip_final_eval
}

run_bootstrap() {
    local normal_model
    normal_model="$(resolve_normal_model)"
    "${PYTHON_BIN}" -u tools/context_adaptation/bootstrap_normal_context.py \
        --config "${NORMAL_CONFIG}" \
        --init_model "${normal_model}" \
        --context_config "${CONTEXT_CONFIG}" \
        --output "${BOOTSTRAP_CHECKPOINT}" \
        --split train \
        --batch_size "${BOOTSTRAP_BATCH_SIZE:-1}" \
        --num_workers "${NUM_WORKERS:-0}" \
        --max_samples "${BOOTSTRAP_MAX_SAMPLES:--1}"
}

run_adapt() {
    if [[ ! -f "${BOOTSTRAP_CHECKPOINT}" ]]; then
        echo "Bootstrap checkpoint does not exist: ${BOOTSTRAP_CHECKPOINT}" >&2
        return 1
    fi
    "${PYTHON_BIN}" -u tools/context_adaptation/online_adapt_auto_context.py \
        --config "${SEQ58_ADAPT_CONFIG}" \
        --init_context_checkpoint "${BOOTSTRAP_CHECKPOINT}" \
        --context_config "${CONTEXT_CONFIG}" \
        --output_root "${RESULT_ROOT}" \
        --run_name "${ADAPT_RUN_NAME}" \
        --batch_size 1 \
        --num_workers "${NUM_WORKERS:-0}" \
        --max_steps "${MAX_STEPS:--1}"
}

run_adapt_return() {
    if [[ ! -f "${BOOTSTRAP_CHECKPOINT}" ]]; then
        echo "Bootstrap checkpoint does not exist: ${BOOTSTRAP_CHECKPOINT}" >&2
        return 1
    fi
    "${PYTHON_BIN}" -u tools/context_adaptation/online_adapt_auto_context.py \
        --config "${SEQ58_ADAPT_CONFIG}" \
        --init_context_checkpoint "${BOOTSTRAP_CHECKPOINT}" \
        --context_config "${CONTEXT_CONFIG}" \
        --stream_config "${RETURN_STREAM_CONFIG}" \
        --output_root "${RESULT_ROOT}" \
        --run_name "${RETURN_RUN_NAME}" \
        --batch_size 1 \
        --num_workers "${NUM_WORKERS:-0}" \
        --max_steps "${MAX_STEPS:--1}"
}

run_eval_seq58() {
    local source_run_name="${1:-${EVAL_RUN_NAME:-${ADAPT_RUN_NAME}}}"
    local output_suffix="${2:-}"
    local checkpoint
    checkpoint="$(resolve_adapted_checkpoint "${source_run_name}")"
    "${PYTHON_BIN}" -u tools/context_adaptation/eval_auto_context.py \
        --config "${SEQ58_EVAL_CONFIG}" \
        --checkpoint "${checkpoint}" \
        --output_root "${RESULT_ROOT}" \
        --run_name "eval_auto_context_seq58${output_suffix}" \
        --mode auto \
        --metric_scene_name seq58 \
        --conf_thr 0.3
}

run_eval_normal() {
    local source_run_name="${1:-${EVAL_RUN_NAME:-${ADAPT_RUN_NAME}}}"
    local output_suffix="${2:-}"
    local checkpoint
    checkpoint="$(resolve_adapted_checkpoint "${source_run_name}")"
    "${PYTHON_BIN}" -u tools/context_adaptation/eval_auto_context.py \
        --config "${NORMAL_EVAL_CONFIG}" \
        --checkpoint "${checkpoint}" \
        --output_root "${RESULT_ROOT}" \
        --run_name "eval_auto_context_normal${output_suffix}" \
        --mode auto \
        --metric_scene_name normal_seq1 \
        --conf_thr 0.3
}

usage() {
    echo "Usage: bash tools/run_auto_context_pipeline.sh {train|bootstrap|adapt|adapt_return|eval_seq58|eval_normal|eval|eval_return|all}"
}

stage="${1:-}"
case "${stage}" in
    train)
        run_train
        ;;
    bootstrap)
        run_bootstrap
        ;;
    adapt)
        run_adapt
        ;;
    adapt_return)
        run_adapt_return
        ;;
    eval_seq58)
        run_eval_seq58
        ;;
    eval_normal)
        run_eval_normal
        ;;
    eval)
        run_eval_seq58
        run_eval_normal
        ;;
    eval_return)
        run_eval_seq58 "${RETURN_RUN_NAME}" "_return"
        run_eval_normal "${RETURN_RUN_NAME}" "_return"
        ;;
    all)
        run_train
        run_bootstrap
        run_adapt
        run_eval_seq58
        run_eval_normal
        ;;
    *)
        usage
        exit 2
        ;;
esac
