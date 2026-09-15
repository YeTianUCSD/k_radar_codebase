"""Oracle-context sequential superposition adaptation for K-Radar ASF.

This runner deliberately bypasses feature projection, KDE, and automatic context
routing. Each scene boundary and scene name come from an experiment YAML. Each
target context is initialized according to the configured policy, and only that
context residual is optimized while shared weights and older contexts stay frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import yaml
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_adaptation.dynamic_psp import (  # noqa: E402
    activate_context_residuals,
    add_dynamic_context,
    add_inherited_dynamic_context,
    get_context_manifest,
    restore_dynamic_contexts,
    restore_inherited_contexts,
)
from context_adaptation.model_adapter import (  # noqa: E402
    ContextAwareModelAdapter,
    configure_residual_only_training,
    freeze_encoder_batch_stats,
)
from context_adaptation.runtime import (  # noqa: E402
    load_yaml,
    write_single_context_runtime_config,
)


ORACLE_CHECKPOINT_VERSION = 2
RESIDUAL_MARKERS = (
    "aware_query_scene_bank.params.",
    "scene_weight_bank.params.",
    "scene_bias_bank.params.",
)


@dataclass(frozen=True)
class Scene:
    name: str
    sequence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Known-context sequential PSP adaptation and evaluation."
    )
    parser.add_argument("action", choices=("baseline", "adapt", "eval", "all"))
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--init-model",
        default=None,
        help="Base-scene PSP best.checkpoint; required by baseline/all and fresh adapt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run directory. Defaults to EXPERIMENT.RESULT_ROOT/EXPERIMENT.NAME.",
    )
    parser.add_argument(
        "--final-checkpoint",
        default=None,
        help="Final Oracle checkpoint for eval; defaults to output-dir/checkpoints/final.oracle.checkpoint.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume adapt from an after_*.oracle.checkpoint.",
    )
    parser.add_argument("--max-target-scenes", type=int, default=None)
    parser.add_argument("--max-steps-per-scene", type=int, default=None)
    parser.add_argument(
        "--context-initialization",
        choices=("inherit_previous", "inherit_base", "zero_random"),
        default=None,
        help="How a new scene context is initialized.",
    )
    parser.add_argument(
        "--eval-every-updates", type=int, default=None,
        help="Evaluate only the current scene every N updates; 0 disables periodic evaluation.",
    )
    parser.add_argument(
        "--backtest-policy", choices=("all", "final"), default=None
    )
    pre_group = parser.add_mutually_exclusive_group()
    pre_group.add_argument(
        "--scene-pre-eval", dest="scene_pre_eval", action="store_true"
    )
    pre_group.add_argument(
        "--skip-scene-pre-eval", dest="scene_pre_eval", action="store_false"
    )
    final_group = parser.add_mutually_exclusive_group()
    final_group.add_argument(
        "--scene-final-eval", dest="scene_final_eval", action="store_true"
    )
    final_group.add_argument(
        "--skip-scene-final-eval", dest="scene_final_eval", action="store_false"
    )
    parser.set_defaults(scene_pre_eval=None, scene_final_eval=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def load_experiment(path: Path) -> tuple[dict[str, Any], Scene, list[Scene], Path, Path]:
    config = load_yaml(path)
    experiment = require_mapping(config.get("EXPERIMENT"), "EXPERIMENT")
    base_cfg = require_mapping(config.get("BASE"), "BASE")
    base_scene = Scene(str(base_cfg["NAME"]), str(base_cfg["SEQUENCE"]))
    raw_scenes = config.get("SCENES")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("SCENES must be a non-empty list.")
    scenes = [
        Scene(str(require_mapping(item, "SCENES item")["NAME"]), str(item["SEQUENCE"]))
        for item in raw_scenes
    ]
    all_names = [base_scene.name, *(scene.name for scene in scenes)]
    all_sequences = [base_scene.sequence, *(scene.sequence for scene in scenes)]
    if any(not value for value in [*all_names, *all_sequences]):
        raise ValueError("Scene names and sequence identifiers must not be empty.")
    if len(set(all_names)) != len(all_names):
        raise ValueError(f"Scene names must be unique: {all_names}")
    if len(set(all_sequences)) != len(all_sequences):
        raise ValueError(f"Sequence identifiers must be unique: {all_sequences}")
    base_config = resolve_path(str(experiment["BASE_CONFIG"]))
    result_root = resolve_path(str(experiment["RESULT_ROOT"]))
    if not base_config.is_file():
        raise FileNotFoundError(f"Base ASF config does not exist: {base_config}")
    return config, base_scene, scenes, base_config, result_root


def validate_base_config(base_config: Path, base_scene: Scene) -> None:
    config = load_yaml(base_config)
    superposition = require_mapping(
        require_mapping(config.get("MODEL"), "MODEL").get("SUPERPOSITION"),
        "MODEL.SUPERPOSITION",
    )
    if not bool(superposition.get("ENABLED", False)):
        raise ValueError("Base config must enable MODEL.SUPERPOSITION.")
    manifest = tuple(str(item) for item in superposition.get("SCENE_LIST", ()))
    if manifest != (base_scene.name,):
        raise ValueError(
            "Base config must contain exactly the base PSP context: "
            f"expected={(base_scene.name,)}, actual={manifest}."
        )
    portion = tuple(
        str(item)
        for item in require_mapping(config.get("DATASET"), "DATASET").get("portion", ())
    )
    if portion != (base_scene.sequence,):
        raise ValueError(
            "Base config DATASET.portion must contain exactly the base sequence: "
            f"expected={(base_scene.sequence,)}, actual={portion}."
        )


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        yaml.safe_dump(dict(payload), output, sort_keys=False)



def materialize_effective_base_config(
    source_config: Path,
    *,
    base_scene: Scene,
    data_config: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Create a single-context config with explicit data protocol overrides."""
    config = load_yaml(source_config)
    model = require_mapping(config.get("MODEL"), "MODEL")
    superposition = require_mapping(
        model.get("SUPERPOSITION"), "MODEL.SUPERPOSITION"
    )
    superposition["SCENE_LIST"] = [base_scene.name]
    superposition["BASE_SCENE"] = base_scene.name
    superposition["ACTIVE_SCENE"] = base_scene.name

    dataset = require_mapping(config.get("DATASET"), "DATASET")
    dataset["portion"] = [base_scene.sequence]
    path_data = require_mapping(dataset.get("path_data"), "DATASET.path_data")
    roots = data_config.get("LIST_DIR_KRADAR")
    train_split = data_config.get("TRAIN_SPLIT")
    test_split = data_config.get("TEST_SPLIT")
    if roots is not None:
        if not isinstance(roots, (list, tuple)) or not roots:
            raise ValueError("DATA.LIST_DIR_KRADAR must be a non-empty list.")
        path_data["list_dir_kradar"] = [str(value) for value in roots]
    if (train_split is None) != (test_split is None):
        raise ValueError("DATA.TRAIN_SPLIT and DATA.TEST_SPLIT must be set together.")
    if train_split is not None:
        path_data["split"] = [str(train_split), str(test_split)]
    write_yaml(output_path, config)
    return output_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_split_sequences(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(newline="") as source:
        for row in csv.reader(source):
            if not row:
                continue
            sequence = str(row[0]).strip()
            counts[sequence] = counts.get(sequence, 0) + 1
    return counts


def audit_data_protocol(
    base_config: Path,
    *,
    scenes: Sequence[Scene],
    run_dir: Path,
) -> list[dict[str, Any]]:
    config = load_yaml(base_config)
    dataset = require_mapping(config.get("DATASET"), "DATASET")
    path_data = require_mapping(dataset.get("path_data"), "DATASET.path_data")
    raw_splits = path_data.get("split")
    if not isinstance(raw_splits, (list, tuple)) or len(raw_splits) != 2:
        raise ValueError("DATASET.path_data.split must contain train and test paths.")
    train_path = resolve_path(str(raw_splits[0]))
    test_path = resolve_path(str(raw_splits[1]))
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Split file missing: train={train_path}, test={test_path}"
        )
    train_counts = _count_split_sequences(train_path)
    test_counts = _count_split_sequences(test_path)
    roots = path_data.get("list_dir_kradar", ())
    if not isinstance(roots, (list, tuple)):
        roots = (roots,)
    train_hash = _sha256(train_path)
    test_hash = _sha256(test_path)
    rows = []
    print(f"* Data roots: {[str(value) for value in roots]}")
    print(f"* Train split: {train_path} (sha256={train_hash})")
    print(f"* Test split: {test_path} (sha256={test_hash})")
    for scene in scenes:
        row = {
            "scene": scene.name,
            "sequence": scene.sequence,
            "train_count": train_counts.get(scene.sequence, 0),
            "test_count": test_counts.get(scene.sequence, 0),
            "data_roots": ";".join(str(value) for value in roots),
            "train_split": str(train_path),
            "test_split": str(test_path),
            "train_split_sha256": train_hash,
            "test_split_sha256": test_hash,
        }
        if row["train_count"] <= 0 or row["test_count"] <= 0:
            raise RuntimeError(f"Empty train/test split for {scene.name}: {row}")
        rows.append(row)
        append_csv(run_dir / "metrics" / "data_protocol.csv", tuple(row), row)
        print(
            f"* Data scene={scene.name} sequence={scene.sequence}: "
            + "train={}, test={}".format(row["train_count"], row["test_count"])
        )
    return rows

def append_csv(path: Path, fieldnames: Sequence[str], row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def clear_batch(batch: Any) -> None:
    if not isinstance(batch, dict):
        return
    if "pointer" in batch:
        for item in batch["pointer"]:
            for key in list(item):
                if key != "meta":
                    item[key] = None
    for key in list(batch):
        batch[key] = None


def close_writers(pipeline: Any) -> None:
    for name in ("log_train_iter", "log_train_epoch", "log_test"):
        writer = getattr(pipeline, name, None)
        if writer is not None:
            writer.close()


def build_pipeline(
    *,
    base_config: Path,
    base_scene: Scene,
    run_dir: Path,
    num_workers: int,
    best_metric: Mapping[str, Any],
) -> tuple[Any, Path, dict[str, Any]]:
    runtime_path, runtime_config = write_single_context_runtime_config(
        base_config,
        model_context_name=base_scene.name,
        output_root=run_dir / "runtime",
        run_name="oracle_runtime",
        batch_size=1,
        num_workers=num_workers,
        enable_logging=True,
        enable_validation=True,
    )
    runtime_config["GENERAL"]["LOGGING"]["BEST_METRIC"] = {
        "CLS": str(best_metric.get("BEST_METRIC_CLS", "auto")),
        "KIND": str(best_metric.get("BEST_METRIC_KIND", "3d")),
        "IOUS": [float(value) for value in best_metric.get("BEST_METRIC_IOUS", [0.3, 0.5])],
        "CONF_THR": float(best_metric.get("CONF_THR", 0.3)),
        "ONLY_CLASSES_WITH_GT": True,
    }
    with open(runtime_path, "w") as output:
        yaml.safe_dump(runtime_config, output, sort_keys=False)

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pipeline = PipelineDetection_v1_0(path_cfg=runtime_path, mode="train")
    return pipeline, Path(runtime_path), runtime_config


def _extract_model_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a model state dictionary.")
    return state_dict


def _load_compatible_base_context(
    network: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load shared/base tensors while pruning pre-registered extra contexts."""
    target = network.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    ignored_extra = []
    shape_mismatches = []
    for name, value in state_dict.items():
        if name not in target:
            if any(marker in name for marker in RESIDUAL_MARKERS):
                ignored_extra.append(name)
                continue
            raise RuntimeError(f"Unexpected non-residual checkpoint tensor: {name}")
        if target[name].shape != value.shape:
            shape_mismatches.append(
                f"{name}:{tuple(value.shape)}!={tuple(target[name].shape)}"
            )
            continue
        filtered[name] = value
    if shape_mismatches:
        raise RuntimeError(
            "Base checkpoint tensor shape mismatch: " + ";".join(shape_mismatches[:12])
        )
    missing = [name for name in target if name not in filtered]
    forbidden_missing = [
        name for name in missing
        if not any(marker in name for marker in RESIDUAL_MARKERS)
    ]
    if forbidden_missing:
        raise RuntimeError(
            "Missing non-residual base tensors: " + ";".join(forbidden_missing[:12])
        )
    incompatible = network.load_state_dict(filtered, strict=False)
    if set(incompatible.missing_keys) != set(missing) or incompatible.unexpected_keys:
        raise RuntimeError(f"Compatible base restore mismatch: {incompatible}")
    print(
        "* Compatible base checkpoint load: "
        f"loaded={len(filtered)}, zero_initialized_residuals={len(missing)}, "
        f"ignored_extra_context_residuals={len(ignored_extra)}"
    )


def load_raw_or_oracle_checkpoint(
    network: torch.nn.Module,
    path: Path,
    *,
    base_scene: Scene,
    trainable_scope: str,
    checkpoint_load_policy: str = "strict",
) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and payload.get("oracle_checkpoint_version") is not None:
        version = int(payload["oracle_checkpoint_version"])
        if version != ORACLE_CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported Oracle checkpoint version: {version}")
        manifest = tuple(str(name) for name in payload["context_manifest"])
        if not manifest or manifest[0] != base_scene.name:
            raise ValueError(
                f"Oracle checkpoint has incompatible manifest: {manifest}."
            )
        initialization = str(payload.get("context_initialization", ""))
        context_parents = {
            str(key): str(value)
            for key, value in require_mapping(
                payload.get("context_parents", {}), "context_parents"
            ).items()
        }
        if initialization in {"inherit_previous", "inherit_base"}:
            restore_inherited_contexts(
                network,
                manifest,
                context_parents,
                trainable_scope=trainable_scope,
            )
        elif initialization == "zero_random":
            if context_parents:
                raise ValueError("zero_random checkpoint must not contain context parents.")
            restore_dynamic_contexts(
                network,
                manifest,
                trainable_scope=trainable_scope,
            )
        else:
            raise ValueError(
                f"Unsupported checkpoint context initialization: {initialization}"
            )
        incompatible = network.load_state_dict(payload["model_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict Oracle restore failed: {incompatible}")
        return dict(payload)

    policy = str(checkpoint_load_policy)
    state_dict = _extract_model_state_dict(payload)
    if policy == "strict":
        incompatible = network.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict base checkpoint load failed: {incompatible}")
    elif policy == "compatible_base_context":
        _load_compatible_base_context(network, state_dict)
    else:
        raise ValueError(f"Unsupported base checkpoint load policy: {policy}")
    return None


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_oracle_checkpoint(
    path: Path,
    *,
    network: torch.nn.Module,
    base_scene: Scene,
    processed_scenes: Sequence[str],
    context_initialization: str,
    context_parents: Mapping[str, str],
    scene_to_sequence: Mapping[str, str],
    step_idx: int,
    update_idx: int,
    experiment_config: Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    manifest = get_context_manifest(network)
    expected = (base_scene.name, *tuple(processed_scenes))
    initialization = str(context_initialization)
    normalized_parents = {str(key): str(value) for key, value in context_parents.items()}
    expected_children = set(expected[1:])
    if initialization in {"inherit_previous", "inherit_base"}:
        if set(normalized_parents) != expected_children:
            raise RuntimeError(
                f"Checkpoint parent map mismatch: parents={sorted(normalized_parents)}, "
                f"expected={sorted(expected_children)}"
            )
    elif initialization == "zero_random":
        if normalized_parents:
            raise RuntimeError("zero_random checkpoint cannot store context parents.")
    else:
        raise ValueError(f"Unsupported context initialization: {initialization}")
    if manifest != expected:
        raise RuntimeError(
            f"Checkpoint manifest mismatch: model={manifest}, expected={expected}."
        )
    atomic_torch_save(
        {
            "oracle_checkpoint_version": ORACLE_CHECKPOINT_VERSION,
            "context_initialization": initialization,
            "context_parents": normalized_parents,
            "model_state_dict": network.state_dict(),
            "context_manifest": list(manifest),
            "context_ids": {name: index for index, name in enumerate(manifest)},
            "scene_to_sequence": dict(scene_to_sequence),
            "processed_scenes": list(processed_scenes),
            "step_idx": int(step_idx),
            "update_idx": int(update_idx),
            "experiment_config": str(experiment_config),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def make_dataset(base_config: Path, scene: Scene, split: str) -> Any:
    from context_adaptation.stream import StreamSegment, build_segment_dataset

    return build_segment_dataset(
        StreamSegment(
            config_path=str(base_config),
            split=split,
            name_for_metrics_only=scene.name,
            sequences=(scene.sequence,),
        )
    )


def aggregate_score(rows: Sequence[Mapping[str, Any]], metric: Mapping[str, Any]) -> float | None:
    confidence = float(metric.get("CONF_THR", 0.3))
    kind = str(metric.get("BEST_METRIC_KIND", "3d")).lower()
    cls = str(metric.get("BEST_METRIC_CLS", "auto")).lower()
    ious = [float(value) for value in metric.get("BEST_METRIC_IOUS", [0.3, 0.5])]
    values = []
    for row in rows:
        if abs(float(row["conf_thr"]) - confidence) > 1e-6:
            continue
        if cls != "auto" and str(row["cls"]).lower() != cls:
            continue
        if not any(abs(float(row["iou"]) - value) <= 1e-6 for value in ious):
            continue
        if not bool(row.get("has_gt", True)):
            continue
        values.append(float(row[kind]))
    return None if not values else sum(values) / len(values)


def evaluate_scene(
    pipeline: Any,
    *,
    base_config: Path,
    scene: Scene,
    context_name: str,
    phase: str,
    run_dir: Path,
    metric: Mapping[str, Any],
    evaluation_index: int,
) -> tuple[float | None, list[dict[str, Any]]]:
    dataset = make_dataset(base_config, scene, "test")
    previous_dataset = pipeline.dataset_test
    previous_path = Path(pipeline.path_log)
    previous_context = getattr(pipeline.network, "default_scene_context", None)
    evaluation_root = run_dir / "evaluations" / phase / scene.name
    evaluation_root.mkdir(parents=True, exist_ok=True)
    pipeline.dataset_test = dataset
    pipeline.path_log = str(evaluation_root)
    pipeline.network.default_scene_context = context_name
    try:
        rows = pipeline.validate_kitti(
            epoch=evaluation_index,
            list_conf_thr=[float(metric.get("CONF_THR", 0.3))],
            is_subset=False,
        )
    finally:
        pipeline.dataset_test = previous_dataset
        pipeline.path_log = str(previous_path)
        pipeline.network.default_scene_context = previous_context

    raw_fields = (
        "phase", "scene", "context", "cls", "conf_thr", "iou", "bev", "3d", "has_gt"
    )
    raw_path = run_dir / "metrics" / "evaluation_rows.csv"
    for row in rows:
        append_csv(
            raw_path,
            raw_fields,
            {
                "phase": phase,
                "scene": scene.name,
                "context": context_name,
                **row,
            },
        )
    return aggregate_score(rows, metric), list(rows)


def state_snapshot(network: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }



def parameter_snapshot(
    network: torch.nn.Module,
    names: Sequence[str],
) -> dict[str, torch.Tensor]:
    parameters = dict(network.named_parameters())
    missing = [name for name in names if name not in parameters]
    if missing:
        raise KeyError(f"Unknown parameters in snapshot: {missing[:12]}")
    return {
        name: parameters[name].detach().cpu().clone()
        for name in names
    }


def restore_parameter_snapshot(
    network: torch.nn.Module,
    snapshot: Mapping[str, torch.Tensor],
) -> None:
    parameters = dict(network.named_parameters())
    missing = [name for name in snapshot if name not in parameters]
    if missing:
        raise KeyError(f"Unknown parameters in restore: {missing[:12]}")
    with torch.no_grad():
        for name, value in snapshot.items():
            parameter = parameters[name]
            if parameter.shape != value.shape:
                raise RuntimeError(
                    f"Restore shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def audit_active_context_parameters(
    network: torch.nn.Module,
    *,
    scene_name: str,
    trainable_names: Sequence[str],
    audit_config: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    parameters = dict(network.named_parameters())
    missing = [name for name in trainable_names if name not in parameters]
    if missing:
        raise KeyError(f"Missing active residual parameters: {missing[:12]}")
    weight_names = [
        name for name in trainable_names
        if "scene_weight_bank.params." in name
    ]
    active_numel = sum(parameters[name].numel() for name in trainable_names)
    weight_numel = sum(parameters[name].numel() for name in weight_names)
    require_weight = bool(audit_config.get("REQUIRE_SCENE_WEIGHT_BANK", False))
    minimum_numel = int(audit_config.get("MIN_ACTIVE_PARAMETER_NUMEL", 0))
    status = "pass"
    if (require_weight and not weight_names) or active_numel < minimum_numel:
        status = "fail"
    row = {
        "scene": scene_name,
        "status": status,
        "active_tensor_count": len(trainable_names),
        "active_parameter_numel": active_numel,
        "scene_weight_tensor_count": len(weight_names),
        "scene_weight_parameter_numel": weight_numel,
        "minimum_required_numel": minimum_numel,
        "require_scene_weight_bank": require_weight,
        "active_preview": ";".join(trainable_names[:12]),
    }
    append_csv(
        run_dir / "metrics" / "residual_capacity_audit.csv",
        tuple(row),
        row,
    )
    print(
        f"* Residual capacity scene={scene_name}: status={status}, "
        f"tensors={len(trainable_names)}, parameters={active_numel}, "
        f"scene_weight_tensors={len(weight_names)}, "
        f"scene_weight_parameters={weight_numel}"
    )
    if bool(audit_config.get("STRICT", True)) and status != "pass":
        raise RuntimeError(f"Residual capacity audit failed for {scene_name}: {row}")
    return row

def audit_context_initialization(
    *,
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    scene_name: str,
    parent_context: str | None,
    initialization_mode: str,
    inheritance_audit: Any | None,
    audit_config: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    old_max_diff = 0.0
    changed_old = []
    for name, old_value in before.items():
        if name not in after:
            changed_old.append(name)
            old_max_diff = float("inf")
            continue
        new_value = after[name].detach().cpu()
        if old_value.shape != new_value.shape:
            changed_old.append(name)
            old_max_diff = float("inf")
        elif old_value.is_floating_point() or old_value.is_complex():
            difference = float((new_value - old_value).abs().max().item()) if old_value.numel() else 0.0
            old_max_diff = max(old_max_diff, difference)
            if difference > float(audit_config.get("ATOL", 0.0)):
                changed_old.append(name)
        elif not torch.equal(old_value, new_value):
            changed_old.append(name)
            old_max_diff = float("inf")

    effective_diff = (
        ""
        if inheritance_audit is None
        else inheritance_audit.max_effective_parameter_diff
    )
    handled_count = (
        0 if inheritance_audit is None else len(inheritance_audit.handled_bank_names)
    )
    compared_count = (
        0 if inheritance_audit is None else inheritance_audit.compared_tensor_count
    )
    inherited_ok = (
        inheritance_audit is None
        or inheritance_audit.max_effective_parameter_diff
        <= float(audit_config.get("INHERITANCE_ATOL", 1e-6))
    )
    status = "pass" if not changed_old and inherited_ok else "fail"
    row = {
        "scene": scene_name,
        "parent_context": "" if parent_context is None else parent_context,
        "initialization_mode": initialization_mode,
        "status": status,
        "handled_bank_count": handled_count,
        "compared_tensor_count": compared_count,
        "max_effective_parameter_diff": effective_diff,
        "old_parameter_max_diff": old_max_diff,
        "changed_old_tensor_count": len(changed_old),
        "changed_old_preview": ";".join(changed_old[:12]),
    }
    append_csv(
        run_dir / "metrics" / "inheritance_audit.csv",
        tuple(row),
        row,
    )
    if bool(audit_config.get("STRICT", True)) and status != "pass":
        raise RuntimeError(f"Context initialization audit failed for {scene_name}: {row}")
    return row


def audit_update(
    *,
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    allowed_names: set[str],
    all_parameter_names: set[str],
    scene_name: str,
    audit_config: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    atol = float(audit_config.get("ATOL", 0.0))
    check_buffers = bool(audit_config.get("CHECK_BUFFERS", True))
    parameter_names = set(allowed_names)
    changed_allowed = []
    changed_forbidden = []
    max_allowed = 0.0
    max_forbidden = 0.0
    for name, old_value in before.items():
        new_value = after[name].detach().cpu()
        if old_value.shape != new_value.shape:
            changed_forbidden.append(name)
            max_forbidden = float("inf")
            continue
        if old_value.numel() == 0:
            difference = 0.0
        elif old_value.is_floating_point() or old_value.is_complex():
            difference = float((new_value - old_value).abs().max().item())
        else:
            difference = 0.0 if torch.equal(old_value, new_value) else float("inf")
        if name in parameter_names:
            max_allowed = max(max_allowed, difference)
            if difference > atol:
                changed_allowed.append(name)
        elif difference > atol:
            if check_buffers or name in all_parameter_names:
                changed_forbidden.append(name)
                max_forbidden = max(max_forbidden, difference)

    status = "pass" if not changed_forbidden and changed_allowed else "fail"
    row = {
        "scene": scene_name,
        "status": status,
        "allowed_tensor_count": len(allowed_names),
        "changed_current_tensors": len(changed_allowed),
        "changed_forbidden_tensors": len(changed_forbidden),
        "current_max_abs_diff": max_allowed,
        "forbidden_max_abs_diff": max_forbidden,
        "forbidden_preview": ";".join(changed_forbidden[:12]),
    }
    append_csv(
        run_dir / "metrics" / "isolation_audit.csv",
        tuple(row),
        row,
    )
    if bool(audit_config.get("STRICT", True)) and status != "pass":
        raise RuntimeError(f"Parameter isolation audit failed for {scene_name}: {row}")
    return row


def build_optimizer(
    parameters: Sequence[torch.nn.Parameter], online: Mapping[str, Any]
) -> torch.optim.Optimizer:
    name = str(online.get("OPTIMIZER", "adamw")).lower()
    lr = float(online.get("LR", 1e-4))
    weight_decay = float(online.get("WEIGHT_DECAY", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(online.get("MOMENTUM", 0.9)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def run_online_scene(
    pipeline: Any,
    *,
    base_config: Path,
    scene: Scene,
    online: Mapping[str, Any],
    num_workers: int,
    max_steps: int,
    eval_every_updates: int = 0,
    evaluation_callback: Callable[[int, float], None] | None = None,
) -> tuple[int, float, int]:
    dataset = make_dataset(base_config, scene, "train")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(online.get("BATCH_SIZE", 1)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )
    scope = str(online.get("TRAINABLE_SCOPE", "fuser_head"))
    configure_residual_only_training(pipeline.network, trainable_scope=scope)
    active_names, active_parameters = activate_context_residuals(
        pipeline.network, scene.name, trainable_scope=scope
    )
    optimizer = build_optimizer(active_parameters, online)
    adapter = ContextAwareModelAdapter(pipeline.network)
    grad_clip = float(online.get("GRAD_CLIP", 0.0))
    updates = 0
    loss_sum = 0.0
    interval_loss_sum = 0.0
    interval_updates = 0
    last_periodic_update = 0
    progress = tqdm(loader, desc=f"* Oracle online {scene.name}")
    for batch in progress:
        if max_steps > 0 and updates >= max_steps:
            clear_batch(batch)
            break
        pipeline.network.train()
        freeze_encoder_batch_stats(pipeline.network)
        optimizer.zero_grad(set_to_none=True)
        encoded = adapter.encode(batch)
        output = adapter.forward_with_context(encoded, scene.name)
        loss = adapter.loss(output)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in {scene.name} at update {updates + 1}.")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(active_parameters, grad_clip)
        optimizer.step()
        updates += 1
        loss_value = float(loss.detach().cpu().item())
        loss_sum += loss_value
        interval_loss_sum += loss_value
        interval_updates += 1
        progress.set_postfix(loss=f"{loss_value:.4f}")
        clear_batch(batch)

        if eval_every_updates > 0 and updates % eval_every_updates == 0:
            if evaluation_callback is None:
                raise RuntimeError("Periodic evaluation requires an evaluation callback.")
            evaluation_callback(updates, interval_loss_sum / interval_updates)
            last_periodic_update = updates
            interval_loss_sum = 0.0
            interval_updates = 0
    if updates <= 0:
        raise RuntimeError(f"No online updates were performed for {scene.name}.")
    print(f"* Active residual tensors for {scene.name}: {len(active_names)}")
    return updates, loss_sum / updates, last_periodic_update


def summary_fields() -> tuple[str, ...]:
    return (
        "phase", "stage", "scene", "context", "parent_context",
        "initialization_mode", "score", "pre_score", "best_score", "best_update",
        "last_score", "selected_score", "selected_update", "selection_policy",
        "updates", "avg_loss", "checkpoint", "elapsed_sec",
    )


def record_summary(run_dir: Path, **row: Any) -> None:
    append_csv(run_dir / "metrics" / "stage_summary.csv", summary_fields(), row)


def online_curve_fields() -> tuple[str, ...]:
    return (
        "scene", "event", "local_update", "global_update",
        "score", "recent_avg_loss", "elapsed_sec",
    )


def record_online_curve(run_dir: Path, **row: Any) -> None:
    append_csv(
        run_dir / "metrics" / "online_curve.csv",
        online_curve_fields(),
        row,
    )


def run_baseline(
    pipeline: Any,
    *,
    base_config: Path,
    base_scene: Scene,
    scenes: Sequence[Scene],
    run_dir: Path,
    metric: Mapping[str, Any],
) -> None:
    print(f"* Running fixed {base_scene.name}-context baseline on every scene.")
    for index, scene in enumerate((base_scene, *scenes), start=1):
        start = time.time()
        score, _ = evaluate_scene(
            pipeline,
            base_config=base_config,
            scene=scene,
            context_name=base_scene.name,
            phase="baseline",
            run_dir=run_dir,
            metric=metric,
            evaluation_index=index,
        )
        record_summary(
            run_dir,
            phase="baseline",
            stage="base",
            scene=scene.name,
            context=base_scene.name,
            score="" if score is None else score,
            elapsed_sec=time.time() - start,
        )


def run_adaptation(
    pipeline: Any,
    *,
    experiment_path: Path,
    base_config: Path,
    base_scene: Scene,
    scenes: Sequence[Scene],
    run_dir: Path,
    config: Mapping[str, Any],
    resume_payload: Mapping[str, Any] | None,
    max_target_scenes: int,
    max_steps: int,
    num_workers: int,
    backtest_policy: str,
    eval_every_updates: int,
    scene_pre_eval: bool,
    context_initialization: str,
    scene_final_eval: bool,
) -> Path:
    online = require_mapping(config.get("ONLINE"), "ONLINE")
    metric = require_mapping(config.get("EVALUATION"), "EVALUATION")
    audit = require_mapping(config.get("AUDIT", {}), "AUDIT")
    scope = str(online.get("TRAINABLE_SCOPE", "fuser_head"))
    initialization_mode = str(context_initialization)
    selection_policy = str(online.get("STAGE_SELECTION", "last")).lower()
    if selection_policy not in {"best", "last"}:
        raise ValueError(f"Unsupported ONLINE.STAGE_SELECTION: {selection_policy}")
    if initialization_mode not in {"inherit_previous", "inherit_base", "zero_random"}:
        raise ValueError(f"Unsupported context initialization: {initialization_mode}")
    if resume_payload is not None:
        checkpoint_mode = str(resume_payload.get("context_initialization", ""))
        if checkpoint_mode != initialization_mode:
            raise ValueError(
                f"Resume initialization mismatch: checkpoint={checkpoint_mode}, "
                f"requested={initialization_mode}"
            )
        context_parents = {
            str(key): str(value)
            for key, value in require_mapping(
                resume_payload.get("context_parents", {}), "context_parents"
            ).items()
        }
    else:
        context_parents = {}
    processed = list(resume_payload.get("processed_scenes", [])) if resume_payload else []
    resume_metadata = (
        require_mapping(resume_payload.get("metadata", {}), "metadata")
        if resume_payload else {}
    )
    selection_records = {
        str(name): dict(require_mapping(value, f"stage_selection.{name}"))
        for name, value in require_mapping(
            resume_metadata.get("stage_selection", {}), "metadata.stage_selection"
        ).items()
    }
    step_idx = int(resume_payload.get("step_idx", 0)) if resume_payload else 0
    update_idx = int(resume_payload.get("update_idx", 0)) if resume_payload else 0
    expected_prefix = [scene.name for scene in scenes[: len(processed)]]
    if processed != expected_prefix:
        raise ValueError(
            f"Resume processed scenes are not an experiment prefix: {processed}."
        )
    selected = list(scenes[len(processed) :])
    if max_target_scenes > 0:
        selected = selected[:max_target_scenes]
    scene_by_name = {scene.name: scene for scene in (base_scene, *scenes)}
    scene_to_sequence = {
        scene.name: scene.sequence for scene in (base_scene, *scenes)
    }
    evaluation_index = 1000 + len(processed) * 100

    configure_residual_only_training(pipeline.network, trainable_scope=scope)
    for scene in selected:
        stage_start = time.time()
        stage_global_start = update_idx
        print(f"\n* Oracle stage start: {scene.name} (Sequence {scene.sequence})")
        initialization_before = state_snapshot(pipeline.network)
        parent_context = None
        inheritance_result = None
        if initialization_mode == "zero_random":
            result = add_dynamic_context(
                pipeline.network,
                scene.name,
                trainable_scope=scope,
            )
            if any(
                torch.count_nonzero(parameter).item()
                for parameter in result.trainable_parameters
            ):
                raise RuntimeError(
                    f"New zero_random context {scene.name} was not zero initialized."
                )
            activate_context_residuals(
                pipeline.network, scene.name, trainable_scope=scope
            )
        else:
            manifest_before = get_context_manifest(pipeline.network)
            parent_context = (
                manifest_before[-1]
                if initialization_mode == "inherit_previous"
                else base_scene.name
            )
            result, inheritance_result = add_inherited_dynamic_context(
                pipeline.network,
                scene.name,
                parent_context,
                trainable_scope=scope,
                atol=float(audit.get("INHERITANCE_ATOL", 1e-6)),
            )
            context_parents[scene.name] = parent_context
        audit_context_initialization(
            before=initialization_before,
            after=pipeline.network.state_dict(),
            scene_name=scene.name,
            parent_context=parent_context,
            initialization_mode=initialization_mode,
            inheritance_audit=inheritance_result,
            audit_config=audit,
            run_dir=run_dir,
        )
        print(
            f"* Context initialized: scene={scene.name}, "
            f"mode={initialization_mode}, parent={parent_context}"
        )

        active_names, _ = activate_context_residuals(
            pipeline.network, scene.name, trainable_scope=scope
        )
        audit_active_context_parameters(
            pipeline.network,
            scene_name=scene.name,
            trainable_names=active_names,
            audit_config=audit,
            run_dir=run_dir,
        )
        best_score: float | None = None
        best_update: int | None = None
        best_snapshot: dict[str, torch.Tensor] | None = None

        def consider_candidate(score: float | None, local_update: int) -> None:
            nonlocal best_score, best_update, best_snapshot
            if score is None:
                return
            if best_score is None or score > best_score:
                best_score = float(score)
                best_update = int(local_update)
                best_snapshot = parameter_snapshot(
                    pipeline.network, result.trainable_parameter_names
                )
                print(
                    f"* New stage best scene={scene.name}: "
                    f"update={best_update}, score={best_score:.6f}"
                )

        pre_score = None
        if scene_pre_eval:
            evaluation_index += 1
            pre_score, _ = evaluate_scene(
                pipeline,
                base_config=base_config,
                scene=scene,
                context_name=scene.name,
                phase=f"pre_update/{scene.name}",
                run_dir=run_dir,
                metric=metric,
                evaluation_index=evaluation_index,
            )
            consider_candidate(pre_score, 0)
            record_summary(
                run_dir,
                phase="pre_update",
                stage=scene.name,
                scene=scene.name,
                context=scene.name,
                parent_context="" if parent_context is None else parent_context,
                initialization_mode=initialization_mode,
                score="" if pre_score is None else pre_score,
            )
            record_online_curve(
                run_dir,
                scene=scene.name,
                event="pre_update",
                local_update=0,
                global_update=update_idx,
                score="" if pre_score is None else pre_score,
                elapsed_sec=time.time() - stage_start,
            )

        periodic_scores: dict[int, float | None] = {}

        def evaluate_periodic(local_update: int, recent_avg_loss: float) -> None:
            nonlocal evaluation_index
            evaluation_index += 1
            score, _ = evaluate_scene(
                pipeline,
                base_config=base_config,
                scene=scene,
                context_name=scene.name,
                phase=f"periodic/{scene.name}/update_{local_update:06d}",
                run_dir=run_dir,
                metric=metric,
                evaluation_index=evaluation_index,
            )
            periodic_scores[local_update] = score
            consider_candidate(score, local_update)
            record_online_curve(
                run_dir,
                scene=scene.name,
                event="periodic_eval",
                local_update=local_update,
                global_update=update_idx + local_update,
                score="" if score is None else score,
                recent_avg_loss=recent_avg_loss,
                elapsed_sec=time.time() - stage_start,
            )

        before = state_snapshot(pipeline.network)
        updates, avg_loss, last_periodic_update = run_online_scene(
            pipeline,
            base_config=base_config,
            scene=scene,
            online={
                **dict(online),
                "BATCH_SIZE": int(
                    require_mapping(config.get("DATA"), "DATA").get(
                        "ONLINE_BATCH_SIZE", 1
                    )
                ),
            },
            num_workers=num_workers,
            max_steps=max_steps,
            eval_every_updates=eval_every_updates,
            evaluation_callback=(
                evaluate_periodic if eval_every_updates > 0 else None
            ),
        )
        after = pipeline.network.state_dict()
        if bool(audit.get("ENABLED", True)):
            audit_update(
                before=before,
                after=after,
                allowed_names=set(result.trainable_parameter_names),
                all_parameter_names={name for name, _ in pipeline.network.named_parameters()},
                scene_name=scene.name,
                audit_config=audit,
                run_dir=run_dir,
            )
        step_idx += updates
        update_idx += updates
        processed.append(scene.name)

        post_score = None
        if scene_final_eval:
            if last_periodic_update == updates:
                post_score = periodic_scores.get(updates)
                final_event = "final_reuse_periodic"
            else:
                evaluation_index += 1
                post_score, _ = evaluate_scene(
                    pipeline,
                    base_config=base_config,
                    scene=scene,
                    context_name=scene.name,
                    phase=f"post_update/{scene.name}",
                    run_dir=run_dir,
                    metric=metric,
                    evaluation_index=evaluation_index,
                )
                final_event = "final_eval"
            consider_candidate(post_score, updates)
            record_online_curve(
                run_dir,
                scene=scene.name,
                event=final_event,
                local_update=updates,
                global_update=update_idx,
                score="" if post_score is None else post_score,
                recent_avg_loss=avg_loss,
                elapsed_sec=time.time() - stage_start,
            )
        last_score = post_score
        if selection_policy == "best":
            if best_snapshot is None or best_score is None or best_update is None:
                raise RuntimeError(
                    f"No valid evaluation score was available to select for {scene.name}."
                )
            restore_parameter_snapshot(pipeline.network, best_snapshot)
            selected_score = best_score
            selected_update = best_update
        else:
            selected_score = last_score
            selected_update = updates
        selection_records[scene.name] = {
            "pre_score": pre_score,
            "best_score": best_score,
            "best_update": best_update,
            "last_score": last_score,
            "selected_score": selected_score,
            "selected_update": selected_update,
            "selection_policy": selection_policy,
        }
        print(
            f"* Stage selected scene={scene.name}: policy={selection_policy}, "
            f"pre={pre_score}, best={best_score}@{best_update}, "
            f"last={last_score}, selected={selected_score}@{selected_update}"
        )
        record_online_curve(
            run_dir,
            scene=scene.name,
            event="selected_context",
            local_update=selected_update,
            global_update=stage_global_start + selected_update,
            score="" if selected_score is None else selected_score,
            recent_avg_loss=avg_loss,
            elapsed_sec=time.time() - stage_start,
        )
        record_summary(
            run_dir,
            phase="post_update" if scene_final_eval else "adapt_complete",
            stage=scene.name,
            scene=scene.name,
            context=scene.name,
            parent_context="" if parent_context is None else parent_context,
            initialization_mode=initialization_mode,
            score="" if selected_score is None else selected_score,
            pre_score="" if pre_score is None else pre_score,
            best_score="" if best_score is None else best_score,
            best_update="" if best_update is None else best_update,
            last_score="" if last_score is None else last_score,
            selected_score="" if selected_score is None else selected_score,
            selected_update=selected_update,
            selection_policy=selection_policy,
            updates=updates,
            avg_loss=avg_loss,
            elapsed_sec=time.time() - stage_start,
        )

        checkpoint = run_dir / "checkpoints" / f"after_{scene.name}.oracle.checkpoint"
        save_oracle_checkpoint(
            checkpoint,
            network=pipeline.network,
            base_scene=base_scene,
            processed_scenes=processed,
            context_initialization=initialization_mode,
            context_parents=context_parents,
            scene_to_sequence=scene_to_sequence,
            step_idx=step_idx,
            update_idx=update_idx,
            experiment_config=experiment_path,
            metadata={
                "last_scene": scene.name,
                "avg_loss": avg_loss,
                "selection_policy": selection_policy,
                "pre_score": pre_score,
                "best_score": best_score,
                "best_update": best_update,
                "last_score": last_score,
                "selected_score": selected_score,
                "selected_update": selected_update,
                "stage_selection": dict(selection_records),
            },
        )

        if backtest_policy == "all":
            learned = [base_scene, *(scene_by_name[name] for name in processed)]
            for learned_scene in learned:
                evaluation_index += 1
                score, _ = evaluate_scene(
                    pipeline,
                    base_config=base_config,
                    scene=learned_scene,
                    context_name=learned_scene.name,
                    phase=f"backtest/{scene.name}",
                    run_dir=run_dir,
                    metric=metric,
                    evaluation_index=evaluation_index,
                )
                record_summary(
                    run_dir,
                    phase="backtest",
                    stage=scene.name,
                    scene=learned_scene.name,
                    context=learned_scene.name,
                    score="" if score is None else score,
                    checkpoint=str(checkpoint),
                )

    final_path = run_dir / "checkpoints" / "final.oracle.checkpoint"
    save_oracle_checkpoint(
        final_path,
        network=pipeline.network,
        base_scene=base_scene,
        processed_scenes=processed,
        context_initialization=initialization_mode,
        context_parents=context_parents,
        scene_to_sequence=scene_to_sequence,
        step_idx=step_idx,
        update_idx=update_idx,
        experiment_config=experiment_path,
        metadata={
            "complete": len(processed) == len(scenes),
            "stage_selection": dict(selection_records),
        },
    )
    print(f"* Final Oracle checkpoint: {final_path}")
    return final_path


def run_final_evaluation(
    pipeline: Any,
    *,
    base_config: Path,
    base_scene: Scene,
    scenes: Sequence[Scene],
    run_dir: Path,
    metric: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    manifest = get_context_manifest(pipeline.network)
    expected = tuple(scene.name for scene in (base_scene, *scenes))
    missing = [name for name in expected if name not in manifest]
    if missing:
        raise RuntimeError(
            f"Final evaluation checkpoint lacks contexts {missing}; manifest={manifest}."
        )
    for index, scene in enumerate((base_scene, *scenes), start=20001):
        start = time.time()
        score, _ = evaluate_scene(
            pipeline,
            base_config=base_config,
            scene=scene,
            context_name=scene.name,
            phase="final",
            run_dir=run_dir,
            metric=metric,
            evaluation_index=index,
        )
        record_summary(
            run_dir,
            phase="final",
            stage="final",
            scene=scene.name,
            context=scene.name,
            score="" if score is None else score,
            checkpoint=str(checkpoint_path),
            elapsed_sec=time.time() - start,
        )


def main() -> None:
    args = parse_args()
    experiment_path = resolve_path(args.experiment_config)
    config, base_scene, scenes, source_base_config, result_root = load_experiment(experiment_path)
    experiment = require_mapping(config["EXPERIMENT"], "EXPERIMENT")
    run_dir = resolve_path(
        args.output_dir or (result_root / str(experiment["NAME"]))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots = run_dir / "config_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    shutil.copy2(experiment_path, snapshots / "oracle_config.yml")
    shutil.copy2(source_base_config, snapshots / "base_model_config.yml")

    data = require_mapping(config.get("DATA"), "DATA")
    base_config = materialize_effective_base_config(
        source_base_config,
        base_scene=base_scene,
        data_config=data,
        output_path=snapshots / "effective_base_model_config.yml",
    )
    validate_base_config(base_config, base_scene)
    data_protocol_rows = audit_data_protocol(
        base_config, scenes=(base_scene, *scenes), run_dir=run_dir
    )
    metric = require_mapping(config.get("EVALUATION"), "EVALUATION")
    online = require_mapping(config.get("ONLINE"), "ONLINE")
    audit = require_mapping(config.get("AUDIT", {}), "AUDIT")
    base_options = require_mapping(config.get("BASE"), "BASE")
    checkpoint_load_policy = str(
        base_options.get("CHECKPOINT_LOAD_POLICY", "strict")
    )
    num_workers = int(
        args.num_workers if args.num_workers is not None else data.get("NUM_WORKERS", 0)
    )
    max_scenes = int(
        args.max_target_scenes if args.max_target_scenes is not None else -1
    )
    max_steps = int(
        args.max_steps_per_scene
        if args.max_steps_per_scene is not None
        else online.get("MAX_STEPS_PER_SCENE", -1)
    )
    context_initialization = str(
        args.context_initialization
        or online.get("CONTEXT_INITIALIZATION", "inherit_previous")
    )
    if context_initialization not in {"inherit_previous", "inherit_base", "zero_random"}:
        raise ValueError(
            f"Unsupported context initialization: {context_initialization}"
        )
    backtest_policy = str(
        args.backtest_policy or metric.get("BACKTEST_POLICY", "final")
    )
    eval_every_updates = int(
        args.eval_every_updates
        if args.eval_every_updates is not None
        else metric.get("EVAL_EVERY_UPDATES", 0)
    )
    scene_pre_eval = bool(
        args.scene_pre_eval
        if args.scene_pre_eval is not None
        else metric.get("SCENE_PRE_EVAL", True)
    )
    scene_final_eval = bool(
        args.scene_final_eval
        if args.scene_final_eval is not None
        else metric.get("SCENE_FINAL_EVAL", True)
    )
    if num_workers < 0 or max_scenes == 0 or max_scenes < -1 or max_steps == 0 or max_steps < -1 or eval_every_updates < 0:
        raise ValueError("Invalid workers/scenes/steps override.")

    initial_path = resolve_path(args.init_model) if args.init_model else None
    if args.action in {"baseline", "all"} and initial_path is None:
        raise ValueError("--init-model is required for baseline, adapt, and all.")
    if args.action == "adapt" and initial_path is None and not args.resume_checkpoint:
        raise ValueError("adapt requires --init-model or --resume-checkpoint.")
    resume_path = resolve_path(args.resume_checkpoint) if args.resume_checkpoint else None
    final_path = resolve_path(
        args.final_checkpoint or (run_dir / "checkpoints" / "final.oracle.checkpoint")
    )
    model_source = final_path if args.action == "eval" else (resume_path or initial_path)
    if model_source is None or not model_source.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_source}")

    resolved_meta = {
        "action": args.action,
        "experiment_config": str(experiment_path),
        "source_base_config": str(source_base_config),
        "effective_base_config": str(base_config),
        "checkpoint_load_policy": checkpoint_load_policy,
        "init_model": None if initial_path is None else str(initial_path),
        "model_source": str(model_source),
        "run_dir": str(run_dir),
        "base_scene": base_scene.__dict__,
        "target_scenes": [scene.__dict__ for scene in scenes],
        "max_target_scenes": max_scenes,
        "max_steps_per_scene": max_steps,
        "context_initialization": context_initialization,
        "backtest_policy": backtest_policy,
        "eval_every_updates": eval_every_updates,
        "scene_pre_eval": scene_pre_eval,
        "scene_final_eval": scene_final_eval,
        "num_workers": num_workers,
        "data_protocol": data_protocol_rows,
    }
    write_yaml(snapshots / "resolved_run.yml", resolved_meta)

    pipeline = None
    runtime_path = None
    try:
        pipeline, runtime_path, runtime_config = build_pipeline(
            base_config=base_config,
            base_scene=base_scene,
            run_dir=run_dir,
            num_workers=num_workers,
            best_metric=metric,
        )
        shutil.copy2(runtime_path, snapshots / "runtime_model_config.yml")
        payload = load_raw_or_oracle_checkpoint(
            pipeline.network,
            model_source,
            base_scene=base_scene,
            trainable_scope=str(online.get("TRAINABLE_SCOPE", "fuser_head")),
            checkpoint_load_policy=checkpoint_load_policy,
        )
        print(f"* Loaded model: {model_source}")
        print(f"* Context manifest: {list(get_context_manifest(pipeline.network))}")
        startup_active_names, _ = activate_context_residuals(
            pipeline.network,
            base_scene.name,
            trainable_scope=str(online.get("TRAINABLE_SCOPE", "fuser_head")),
        )
        audit_active_context_parameters(
            pipeline.network,
            scene_name=f"startup:{base_scene.name}",
            trainable_names=startup_active_names,
            audit_config=audit,
            run_dir=run_dir,
        )

        if args.action in {"baseline", "all"}:
            if payload is not None:
                raise ValueError("Baseline must start from the raw base-scene checkpoint.")
            run_baseline(
                pipeline,
                base_config=base_config,
                base_scene=base_scene,
                scenes=scenes,
                run_dir=run_dir,
                metric=metric,
            )

        if args.action in {"adapt", "all"}:
            final_path = run_adaptation(
                pipeline,
                experiment_path=experiment_path,
                base_config=base_config,
                base_scene=base_scene,
                scenes=scenes,
                run_dir=run_dir,
                config=config,
                resume_payload=payload,
                max_target_scenes=max_scenes,
                max_steps=max_steps,
                num_workers=num_workers,
                backtest_policy=backtest_policy,
                eval_every_updates=eval_every_updates,
                scene_pre_eval=scene_pre_eval,
                context_initialization=context_initialization,
                scene_final_eval=scene_final_eval,
            )

        if args.action == "all":
            run_final_evaluation(
                pipeline,
                base_config=base_config,
                base_scene=base_scene,
                scenes=scenes,
                run_dir=run_dir,
                metric=metric,
                checkpoint_path=final_path,
            )
        elif args.action == "eval":
            run_final_evaluation(
                pipeline,
                base_config=base_config,
                base_scene=base_scene,
                scenes=scenes,
                run_dir=run_dir,
                metric=metric,
                checkpoint_path=model_source,
            )
    finally:
        if pipeline is not None:
            close_writers(pipeline)
        if runtime_path is not None:
            try:
                runtime_path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
