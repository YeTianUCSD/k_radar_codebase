#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/code/hyperradar/k_radar_codebase"
CONDA_SH="${CONDA_SH:-/home/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-kradar_asf}"

export PATH="/home/miniconda/bin:${PATH}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

PYTHON_BIN="${PYTHON_BIN:-/home/miniconda/envs/kradar_asf/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/ContextAdaptation/10scenes_seq5_v2_dim16_auto_bw}"
BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/configs/ASF_v2_0_seq5_psp.yml}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-${ROOT_DIR}/configs/context_adaptation/auto_context_10scenes_v2.yml}"
DISCOVERY_STREAM_CONFIG="${DISCOVERY_STREAM_CONFIG:-${ROOT_DIR}/configs/context_adaptation/stream_10scenes_discovery.yml}"
RETURN_STREAM_CONFIG="${RETURN_STREAM_CONFIG:-${ROOT_DIR}/configs/context_adaptation/stream_10scenes_return.yml}"
BOOTSTRAP_CHECKPOINT="${BOOTSTRAP_CHECKPOINT:-${RESULT_ROOT}/bootstrap_seq5_v2_dim16_auto_bw.context.checkpoint}"
DISCOVERY_RUN_NAME="${DISCOVERY_RUN_NAME:-routing_10scenes_discovery_v2_dim16_auto_bw}"
RETURN_RUN_NAME="${RETURN_RUN_NAME:-routing_10scenes_return_v2_dim16_auto_bw}"
METRICS_DIR="${METRICS_DIR:-${RESULT_ROOT}/metrics}"
PIPELINE_STATE="${PIPELINE_STATE:-${RESULT_ROOT}/pipeline_state.yml}"
DISCOVERY_RUN_POINTER="${DISCOVERY_RUN_POINTER:-${RESULT_ROOT}/discovery_run.path}"
RETURN_RUN_POINTER="${RETURN_RUN_POINTER:-${RESULT_ROOT}/return_run.path}"

EXPECTED_LABELS=(seq5 seq1 seq22 seq34 seq9 seq38 seq35 seq46 seq58 seq19)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

require_file() {
    local path="${1}"
    local label="${2}"
    if [[ ! -f "${path}" ]]; then
        echo "${label} does not exist: ${path}" >&2
        return 1
    fi
}

resolve_latest_run_dir() {
    local run_name="${1}"
    local run_dir
    run_dir="$(find "${RESULT_ROOT}" -maxdepth 1 -type d -name "${run_name}_exp_*" -print 2>/dev/null | sort -V | tail -n 1)"
    if [[ -z "${run_dir}" ]]; then
        echo "No run directory found for ${run_name} under ${RESULT_ROOT}." >&2
        return 1
    fi
    printf '%s\n' "${run_dir}"
}

read_run_pointer() {
    local pointer="${1}"
    local label="${2}"
    require_file "${pointer}" "${label} pointer" || return 1
    local run_dir
    run_dir="$(head -n 1 "${pointer}")"
    if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
        echo "${label} run directory is invalid: ${run_dir}" >&2
        return 1
    fi
    printf '%s\n' "${run_dir}"
}

resolve_discovery_run_dir() {
    if [[ -n "${DISCOVERY_RUN_DIR:-}" ]]; then
        if [[ ! -d "${DISCOVERY_RUN_DIR}" ]]; then
            echo "DISCOVERY_RUN_DIR does not exist: ${DISCOVERY_RUN_DIR}" >&2
            return 1
        fi
        printf '%s\n' "${DISCOVERY_RUN_DIR}"
        return 0
    fi
    read_run_pointer "${DISCOVERY_RUN_POINTER}" "Discovery"
}

resolve_return_run_dir() {
    if [[ -n "${RETURN_RUN_DIR:-}" ]]; then
        if [[ ! -d "${RETURN_RUN_DIR}" ]]; then
            echo "RETURN_RUN_DIR does not exist: ${RETURN_RUN_DIR}" >&2
            return 1
        fi
        printf '%s\n' "${RETURN_RUN_DIR}"
        return 0
    fi
    read_run_pointer "${RETURN_RUN_POINTER}" "Return"
}

resolve_latest_resume_checkpoint() {
    local run_dir="$1"
    local newest=""
    local candidate
    for candidate in "${run_dir}/models"/context_created_*.checkpoint "${run_dir}/models"/step_*.context.checkpoint "${run_dir}/models"/update_*.checkpoint; do
        [[ -f "${candidate}" ]] || continue
        if [[ -z "${newest}" || "${candidate}" -nt "${newest}" ]]; then
            newest="${candidate}"
        fi
    done
    if [[ -z "${newest}" ]]; then
        echo "No resumable checkpoint found under ${run_dir}/models." >&2
        return 1
    fi
    printf "%s\n" "${newest}"
}

resolve_discovery_checkpoint() {
    if [[ -n "${DISCOVERY_CHECKPOINT:-}" ]]; then
        require_file "${DISCOVERY_CHECKPOINT}" "DISCOVERY_CHECKPOINT" || return 1
        printf '%s\n' "${DISCOVERY_CHECKPOINT}"
        return 0
    fi
    local run_dir
    run_dir="$(resolve_discovery_run_dir)" || return 1
    local checkpoint="${run_dir}/models/last.context.checkpoint"
    require_file "${checkpoint}" "Discovery checkpoint" || return 1
    printf '%s\n' "${checkpoint}"
}

write_pipeline_state() {
    local discovery_run=""
    local discovery_checkpoint=""
    local return_run=""
    local discovery_csv=""
    local return_csv=""
    local init_model="${INIT_MODEL:-}"
    if [[ -z "${init_model}" && -f "${PIPELINE_STATE}" ]]; then
        init_model="$(awk -F": " '$1 == "init_model" { value=$2; gsub(/^"|"$/, "", value); print value; exit }' "${PIPELINE_STATE}")"
    fi

    if [[ -f "${DISCOVERY_RUN_POINTER}" ]]; then
        discovery_run="$(head -n 1 "${DISCOVERY_RUN_POINTER}")"
        discovery_checkpoint="${discovery_run}/models/last.context.checkpoint"
        discovery_csv="${discovery_run}/routing_log.csv"
    fi
    if [[ -f "${RETURN_RUN_POINTER}" ]]; then
        return_run="$(head -n 1 "${RETURN_RUN_POINTER}")"
        return_csv="${return_run}/routing_log.csv"
    fi

    {
        printf 'experiment: 10scenes_seq5_v2_dim16_auto_bw\n'
        printf 'updated_at_utc: "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'conda_environment: "%s"\n' "${CONDA_ENV_NAME}"
        printf 'python: "%s"\n' "${PYTHON_BIN}"
        printf 'init_model: "%s"\n' "${init_model}"
        printf 'bootstrap_checkpoint: "%s"\n' "${BOOTSTRAP_CHECKPOINT}"
        printf 'context_config: "%s"\n' "${CONTEXT_CONFIG}"
        printf 'discovery_stream_config: "%s"\n' "${DISCOVERY_STREAM_CONFIG}"
        printf 'return_stream_config: "%s"\n' "${RETURN_STREAM_CONFIG}"
        printf 'discovery_run_dir: "%s"\n' "${discovery_run}"
        printf 'discovery_checkpoint: "%s"\n' "${discovery_checkpoint}"
        printf 'discovery_routing_csv: "%s"\n' "${discovery_csv}"
        printf 'return_run_dir: "%s"\n' "${return_run}"
        printf 'return_routing_csv: "%s"\n' "${return_csv}"
        printf 'metrics_dir: "%s"\n' "${METRICS_DIR}"
    } > "${PIPELINE_STATE}"
}

run_validate() {
    require_file "${CONDA_SH}" "CONDA_SH" || return 1
    require_file "${PYTHON_BIN}" "PYTHON_BIN" || return 1
    require_file "${BASE_CONFIG}" "BASE_CONFIG" || return 1
    require_file "${CONTEXT_CONFIG}" "CONTEXT_CONFIG" || return 1
    require_file "${DISCOVERY_STREAM_CONFIG}" "DISCOVERY_STREAM_CONFIG" || return 1
    require_file "${RETURN_STREAM_CONFIG}" "RETURN_STREAM_CONFIG" || return 1

    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" - "${BASE_CONFIG}" "${CONTEXT_CONFIG}" "${DISCOVERY_STREAM_CONFIG}" "${RETURN_STREAM_CONFIG}" <<'PY'
import sys
from collections import Counter
from pathlib import Path

import yaml

from context_adaptation.stream import load_stream_manifest

base_path = Path(sys.argv[1]).resolve()
context_path = Path(sys.argv[2]).resolve()
discovery_path = Path(sys.argv[3]).resolve()
return_path = Path(sys.argv[4]).resolve()
base = yaml.safe_load(base_path.read_text())
context = yaml.safe_load(context_path.read_text())
if list(base["DATASET"]["portion"]) != ["5"]:
    raise ValueError("BASE_CONFIG must select only Sequence 5.")
superposition = base["MODEL"]["SUPERPOSITION"]
if list(superposition["SCENE_LIST"]) != ["seq5"]:
    raise ValueError("BASE_CONFIG must contain only the seq5 PSP context.")
if str(superposition["BASE_SCENE"]) != "seq5":
    raise ValueError("BASE_CONFIG BASE_SCENE must be seq5.")
if int(context["PROJECTION"]["DIM"]) not in {2, 4, 8, 16}:
    raise ValueError("PROJECTION.DIM must be one of 2, 4, 8, or 16.")
if int(context["EXPERIMENT"]["EXPECTED_CONTEXTS"]) != 10:
    raise ValueError("This experiment requires EXPECTED_CONTEXTS=10.")
if str(context["BASE_CONTEXT"]["MODEL_CONTEXT_NAME"]) != "seq5":
    raise ValueError("Context bootstrap model name must be seq5.")

expected = ("5", "1", "22", "34", "9", "38", "35", "46", "58", "19")
discovery = load_stream_manifest(discovery_path, repository_root=Path.cwd())
returned = load_stream_manifest(return_path, repository_root=Path.cwd())
if len(discovery) != 10 or len(returned) != 10:
    raise ValueError("Discovery and return manifests must each contain 10 segments.")
if tuple(segment.sequences[0] for segment in discovery) != expected:
    raise ValueError("Discovery sequence order does not match the experiment contract.")
if any(len(segment.sequences) != 1 for segment in [*discovery, *returned]):
    raise ValueError("Every segment must contain exactly one sequence.")
if any(segment.split != "train" for segment in discovery):
    raise ValueError("Discovery segments must use the train split.")
if any(segment.split != "test" for segment in returned):
    raise ValueError("Return segments must use the test split.")
if set(segment.sequences[0] for segment in returned) != set(expected):
    raise ValueError("Return manifest must contain the same 10 sequences.")
for segment in [*discovery, *returned]:
    if Path(segment.config_path).resolve() != base_path:
        raise ValueError("Every stream segment must use BASE_CONFIG.")
    expected_metric_name = f"seq{segment.sequences[0]}"
    if segment.name_for_metrics_only != expected_metric_name:
        raise ValueError(
            "Metric-only label does not match SEQUENCES: "
            f"{segment.name_for_metrics_only} != {expected_metric_name}"
        )
split_paths = base["DATASET"]["path_data"]["split"]
if len(split_paths) != 2:
    raise ValueError("BASE_CONFIG must define global train and test split files.")
split_counts = {}
for split_name, raw_path in zip(("train", "test"), split_paths):
    split_path = Path(raw_path)
    if not split_path.is_absolute():
        split_path = Path.cwd() / split_path
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing {split_name} split: {split_path}")
    counts = Counter(
        line.split(",", 1)[0].strip()
        for line in split_path.read_text().splitlines()
        if line.strip()
    )
    missing_sequences = [sequence for sequence in expected if counts[sequence] == 0]
    if missing_sequences:
        raise ValueError(
            f"{split_name} split is missing sequences: {missing_sequences}"
        )
    split_counts[split_name] = {sequence: counts[sequence] for sequence in expected}
print("VALIDATION_OK")
print(f"conda_prefix={sys.prefix}")
print(f"projection_dim={context['PROJECTION']['DIM']}")
print(f"discovery_segments={len(discovery)}")
print(f"return_segments={len(returned)}")
print(f"split_counts={split_counts}")
PY
}

run_bootstrap() {
    if [[ -z "${INIT_MODEL:-}" ]]; then
        echo "Set INIT_MODEL to the final Seq5 best.checkpoint." >&2
        return 1
    fi
    require_file "${INIT_MODEL}" "INIT_MODEL" || return 1
    run_validate || return 1

    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -u tools/context_adaptation/bootstrap_normal_context.py \
        --config "${BASE_CONFIG}" \
        --init_model "${INIT_MODEL}" \
        --context_config "${CONTEXT_CONFIG}" \
        --output "${BOOTSTRAP_CHECKPOINT}" \
        --split train \
        --batch_size "${BOOTSTRAP_BATCH_SIZE:-1}" \
        --num_workers "${NUM_WORKERS:-0}" \
        --max_samples "${BOOTSTRAP_MAX_SAMPLES:--1}" \
        --base_context_name normal \
        --base_model_context_name seq5 || return 1

    rm -f "${DISCOVERY_RUN_POINTER}" "${RETURN_RUN_POINTER}"
    write_pipeline_state
}

run_route_discovery() {
    require_file "${BOOTSTRAP_CHECKPOINT}" "BOOTSTRAP_CHECKPOINT" || return 1
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -u tools/context_adaptation/online_adapt_auto_context.py \
        --config "${BASE_CONFIG}" \
        --init_context_checkpoint "${BOOTSTRAP_CHECKPOINT}" \
        --context_config "${CONTEXT_CONFIG}" \
        --stream_config "${DISCOVERY_STREAM_CONFIG}" \
        --output_root "${RESULT_ROOT}" \
        --run_name "${DISCOVERY_RUN_NAME}" \
        --batch_size 1 \
        --num_workers "${NUM_WORKERS:-0}" \
        --max_steps "${MAX_STEPS:--1}" \
        --save_every_updates 0 \
        --save_every_steps "${SAVE_EVERY_STEPS:-500}" \
        --routing_only || return 1

    local run_dir
    run_dir="$(resolve_latest_run_dir "${DISCOVERY_RUN_NAME}")" || return 1
    require_file "${run_dir}/routing_log.csv" "Discovery routing log" || return 1
    require_file "${run_dir}/models/last.context.checkpoint" "Discovery checkpoint" || return 1
    printf '%s\n' "${run_dir}" > "${DISCOVERY_RUN_POINTER}"
    rm -f "${RETURN_RUN_POINTER}"
    write_pipeline_state
    echo "DISCOVERY_RUN_DIR=${run_dir}"
    echo "DISCOVERY_CHECKPOINT=${run_dir}/models/last.context.checkpoint"
}

run_resume_discovery() {
    local run_dir="${DISCOVERY_RESUME_RUN_DIR:-}"
    local checkpoint="${DISCOVERY_RESUME_CHECKPOINT:-}"
    if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
        echo "Set DISCOVERY_RESUME_RUN_DIR to the interrupted discovery directory." >&2
        return 1
    fi
    if [[ -z "${checkpoint}" ]]; then
        checkpoint="$(resolve_latest_resume_checkpoint "${run_dir}")"
    fi
    require_file "${checkpoint}" "DISCOVERY_RESUME_CHECKPOINT" || return 1

    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -u tools/context_adaptation/online_adapt_auto_context.py \
        --config "${BASE_CONFIG}" \
        --init_context_checkpoint "${checkpoint}" \
        --context_config "${CONTEXT_CONFIG}" \
        --stream_config "${DISCOVERY_STREAM_CONFIG}" \
        --output_root "${RESULT_ROOT}" \
        --run_name "${DISCOVERY_RUN_NAME}_resume_runtime" \
        --resume_run_dir "${run_dir}" \
        --batch_size 1 \
        --num_workers "${NUM_WORKERS:-0}" \
        --max_steps "${MAX_STEPS:--1}" \
        --save_every_updates 0 \
        --save_every_steps "${SAVE_EVERY_STEPS:-500}" \
        --routing_only || return 1

    require_file "${run_dir}/routing_log.csv" "Discovery routing log" || return 1
    require_file "${run_dir}/models/last.context.checkpoint" "Discovery checkpoint" || return 1
    printf "%s\n" "${run_dir}" > "${DISCOVERY_RUN_POINTER}"
    rm -f "${RETURN_RUN_POINTER}"
    write_pipeline_state
    echo "DISCOVERY_RUN_DIR=${run_dir}"
    echo "DISCOVERY_CHECKPOINT=${run_dir}/models/last.context.checkpoint"
}

run_route_return() {
    local checkpoint
    checkpoint="$(resolve_discovery_checkpoint)" || return 1
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -u tools/context_adaptation/online_adapt_auto_context.py \
        --config "${BASE_CONFIG}" \
        --init_context_checkpoint "${checkpoint}" \
        --context_config "${CONTEXT_CONFIG}" \
        --stream_config "${RETURN_STREAM_CONFIG}" \
        --output_root "${RESULT_ROOT}" \
        --run_name "${RETURN_RUN_NAME}" \
        --batch_size 1 \
        --num_workers "${NUM_WORKERS:-0}" \
        --max_steps "${MAX_STEPS:--1}" \
        --save_every_updates 0 \
        --save_every_steps "${SAVE_EVERY_STEPS:-500}" \
        --routing_only || return 1

    local run_dir
    run_dir="$(resolve_latest_run_dir "${RETURN_RUN_NAME}")" || return 1
    require_file "${run_dir}/routing_log.csv" "Return routing log" || return 1
    printf '%s\n' "${run_dir}" > "${RETURN_RUN_POINTER}"
    write_pipeline_state
    echo "RETURN_RUN_DIR=${run_dir}"
}

run_summarize() {
    local discovery_run
    local return_run
    discovery_run="$(resolve_discovery_run_dir)" || return 1
    return_run="$(resolve_return_run_dir)" || return 1
    local discovery_csv="${discovery_run}/routing_log.csv"
    local return_csv="${return_run}/routing_log.csv"
    require_file "${discovery_csv}" "Discovery routing log" || return 1
    require_file "${return_csv}" "Return routing log" || return 1

    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -u \
        tools/context_adaptation/summarize_multiscene_routing.py \
        --discovery_csv "${discovery_csv}" \
        --return_csv "${return_csv}" \
        --output_dir "${METRICS_DIR}" \
        --expected_labels "${EXPECTED_LABELS[@]}" \
        --base_label seq5 \
        --base_context_id 0 \
        --ignore_transition_frames "${IGNORE_TRANSITION_FRAMES:-16}" \
        --minimum_context_purity "${MINIMUM_CONTEXT_PURITY:-0.95}" \
        --minimum_context_support "${MINIMUM_CONTEXT_SUPPORT:-8}" || return 1
    write_pipeline_state
    echo "ROUTING_METRICS=${METRICS_DIR}/routing_metrics.yml"
}

print_status() {
    if [[ -f "${PIPELINE_STATE}" ]]; then
        cat "${PIPELINE_STATE}"
    else
        echo "Pipeline state does not exist: ${PIPELINE_STATE}"
    fi
}

usage() {
    echo "Usage: bash tools/run_auto_context_10scenes_pipeline.sh {validate|bootstrap|route_discovery|resume_discovery|route_return|summarize|routing_all|resume_routing_all|status}"
}

mkdir -p "${RESULT_ROOT}"
stage="${1:-}"
case "${stage}" in
    validate)
        run_validate
        ;;
    bootstrap)
        run_bootstrap
        ;;
    route_discovery)
        run_route_discovery
        ;;
    resume_discovery)
        run_resume_discovery
        ;;
    route_return)
        run_route_return
        ;;
    summarize)
        run_summarize
        ;;
    routing_all)
        run_bootstrap
        run_route_discovery
        run_route_return
        run_summarize
        ;;
    resume_routing_all)
        run_resume_discovery
        run_route_return
        run_summarize
        ;;
    status)
        print_status
        ;;
    *)
        usage
        exit 2
        ;;
esac
