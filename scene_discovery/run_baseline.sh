#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-/home/code/hyperradar/k_radar_codebase}"
PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
FEATURE_BANK="${FEATURE_BANK:-${REPOSITORY_ROOT}/results/EncoderFeatureBank/10scenes_fp16}"
RESULT_ROOT="${RESULT_ROOT:-${REPOSITORY_ROOT}/results/SceneDiscovery}"
DESCRIPTOR_BANK="${DESCRIPTOR_BANK:-${RESULT_ROOT}/descriptors/v1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "${REPOSITORY_ROOT}" || exit 1

run_step() {
    local name="$1"
    shift
    echo "[START] ${name} at $(date --iso-8601=seconds)"
    if "$@"; then
        echo "[DONE] ${name} at $(date --iso-8601=seconds)"
    else
        local status=$?
        echo "[FAILED] ${name} with exit code ${status} at $(date --iso-8601=seconds)" >&2
        return "${status}"
    fi
}

run_step validate_feature_bank     "${PYTHON_BIN}" scene_discovery/scripts/validate_feature_bank.py     --feature-bank "${FEATURE_BANK}"     --output "${RESULT_ROOT}/feature_bank_validation.json"     --finite-mode sample

run_step build_or_verify_descriptors     "${PYTHON_BIN}" scene_discovery/scripts/build_descriptors.py     --feature-bank "${FEATURE_BANK}"     --output-root "${DESCRIPTOR_BANK}"     --descriptors mean mean_std spatial     --chunk-size 2

run_step analyze_separability     "${PYTHON_BIN}" scene_discovery/scripts/analyze_separability.py     --descriptor-bank "${DESCRIPTOR_BANK}"     --descriptor mean_std     --windows 1 10 20 30     --stride 5

run_step clustering_baselines     "${PYTHON_BIN}" scene_discovery/scripts/run_clustering_baselines.py     --descriptor-bank "${DESCRIPTOR_BANK}"     --descriptor mean_std     --windows 1 10 20 30     --stride 5

run_step evaluate_open_set     "${PYTHON_BIN}" scene_discovery/scripts/evaluate_open_set.py     --descriptor-bank "${DESCRIPTOR_BANK}"     --descriptor mean_std     --windows 1 20     --stride 5     --threshold-quantiles 0.9 0.95 0.975 0.99 0.995

run_step simulate_stream_discovery     "${PYTHON_BIN}" scene_discovery/scripts/simulate_stream_discovery.py     --descriptor-bank "${DESCRIPTOR_BANK}"     --descriptor mean_std     --modalities camera lidar radar     --projector-fit-scenes 1     --window 20     --stride 5     --threshold-multipliers 0.8 1.0 1.2     --num-random-orders 4
