#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/code/hyperradar/k_radar_codebase"
CONDA_SH="${CONDA_SH:-/home/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-kradar_asf}"
PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${ROOT_DIR}/configs/oracle_superposition/seq5_10scenes.yml}"
ACTION="${1:-}"

usage() {
    echo "Usage: INIT_MODEL=/path/to/seq5/best.checkpoint bash tools/run_oracle_superposition.sh {baseline|adapt|eval|all} [experiment.yml]"
    echo "For eval, set FINAL_CHECKPOINT or reuse the same RUN_DIR from adapt/all."
}

if [[ -z "${ACTION}" ]]; then
    usage
    exit 2
fi
if [[ $# -ge 2 ]]; then
    EXPERIMENT_CONFIG="$2"
fi
case "${ACTION}" in
    baseline|adapt|eval|all) ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "CONDA_SH does not exist: ${CONDA_SH}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${EXPERIMENT_CONFIG}" ]]; then
    echo "EXPERIMENT_CONFIG does not exist: ${EXPERIMENT_CONFIG}" >&2
    exit 1
fi
if [[ "${ACTION}" != "eval" && -z "${INIT_MODEL:-}" && -z "${RESUME_CHECKPOINT:-}" ]]; then
    echo "Set INIT_MODEL to the Seq5 PSP best.checkpoint." >&2
    exit 1
fi
if [[ -n "${INIT_MODEL:-}" && ! -f "${INIT_MODEL}" ]]; then
    echo "INIT_MODEL does not exist: ${INIT_MODEL}" >&2
    exit 1
fi
if [[ -n "${FINAL_CHECKPOINT:-}" && ! -f "${FINAL_CHECKPOINT}" ]]; then
    echo "FINAL_CHECKPOINT does not exist: ${FINAL_CHECKPOINT}" >&2
    exit 1
fi
if [[ -n "${RESUME_CHECKPOINT:-}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "RESUME_CHECKPOINT does not exist: ${RESUME_CHECKPOINT}" >&2
    exit 1
fi

export PATH="/home/miniconda/bin:${PATH}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"

DEFAULT_RESULT_ROOT="${ROOT_DIR}/results/OracleSuperposition/10scenes_seq5"
RUN_STAMP="${RUN_STAMP:-$(date +%y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${DEFAULT_RESULT_ROOT}/oracle_seq5_10scenes_${RUN_STAMP}}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${ACTION}_$(date +%y%m%d_%H%M%S).log}"
mkdir -p "${LOG_DIR}"

command=(
    "${PYTHON_BIN}" -u
    "${ROOT_DIR}/tools/superposition/oracle_sequential.py"
    "${ACTION}"
    --experiment-config "${EXPERIMENT_CONFIG}"
    --output-dir "${RUN_DIR}"
)

if [[ -n "${INIT_MODEL:-}" ]]; then
    command+=(--init-model "${INIT_MODEL}")
fi
if [[ -n "${FINAL_CHECKPOINT:-}" ]]; then
    command+=(--final-checkpoint "${FINAL_CHECKPOINT}")
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    command+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ -n "${MAX_TARGET_SCENES:-}" ]]; then
    command+=(--max-target-scenes "${MAX_TARGET_SCENES}")
fi
if [[ -n "${MAX_STEPS_PER_SCENE:-}" ]]; then
    command+=(--max-steps-per-scene "${MAX_STEPS_PER_SCENE}")
fi
if [[ -n "${CONTEXT_INITIALIZATION:-}" ]]; then
    command+=(--context-initialization "${CONTEXT_INITIALIZATION}")
fi
if [[ -n "${EVAL_EVERY_UPDATES:-}" ]]; then
    command+=(--eval-every-updates "${EVAL_EVERY_UPDATES}")
fi
if [[ -n "${SCENE_PRE_EVAL:-}" ]]; then
    if [[ "${SCENE_PRE_EVAL}" == "1" ]]; then
        command+=(--scene-pre-eval)
    elif [[ "${SCENE_PRE_EVAL}" == "0" ]]; then
        command+=(--skip-scene-pre-eval)
    else
        echo "SCENE_PRE_EVAL must be 0 or 1." >&2; exit 2
    fi
elif [[ "${SKIP_SCENE_PRE_EVAL:-0}" == "1" ]]; then
    command+=(--skip-scene-pre-eval)
fi
if [[ -n "${SCENE_FINAL_EVAL:-}" ]]; then
    if [[ "${SCENE_FINAL_EVAL}" == "1" ]]; then
        command+=(--scene-final-eval)
    elif [[ "${SCENE_FINAL_EVAL}" == "0" ]]; then
        command+=(--skip-scene-final-eval)
    else
        echo "SCENE_FINAL_EVAL must be 0 or 1." >&2; exit 2
    fi
elif [[ "${SKIP_SCENE_FINAL_EVAL:-0}" == "1" ]]; then
    command+=(--skip-scene-final-eval)
fi
if [[ -n "${BACKTEST_POLICY:-}" ]]; then
    command+=(--backtest-policy "${BACKTEST_POLICY}")
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
    command+=(--num-workers "${NUM_WORKERS}")
fi

{
    echo "[START] $(date -Iseconds)"
    echo "[ACTION] ${ACTION}"
    echo "[RUN_DIR] ${RUN_DIR}"
    echo "[EXPERIMENT_CONFIG] ${EXPERIMENT_CONFIG}"
    echo "[INIT_MODEL] ${INIT_MODEL:-}"
    echo "[FINAL_CHECKPOINT] ${FINAL_CHECKPOINT:-}"
    echo "[RESUME_CHECKPOINT] ${RESUME_CHECKPOINT:-}"
    echo "[CONTEXT_INITIALIZATION] ${CONTEXT_INITIALIZATION:-yaml}"
    echo "[EVAL_EVERY_UPDATES] ${EVAL_EVERY_UPDATES:-yaml}"
    echo "[SCENE_PRE_EVAL] ${SCENE_PRE_EVAL:-yaml}"
    echo "[SCENE_FINAL_EVAL] ${SCENE_FINAL_EVAL:-yaml}"
    echo "[BACKTEST_POLICY] ${BACKTEST_POLICY:-yaml}"
    echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES}"
    printf '[COMMAND]'
    printf ' %q' "${command[@]}"
    printf '\n'
} | tee -a "${LOG_FILE}"

cd "${ROOT_DIR}"
if [[ "${FOREGROUND:-0}" == "1" ]]; then
    "${command[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
    /usr/bin/nohup "${command[@]}" >> "${LOG_FILE}" 2>&1 &
    pid=$!
    echo "PID=${pid}"
    echo "RUN_DIR=${RUN_DIR}"
    echo "LOG_FILE=${LOG_FILE}"
fi
