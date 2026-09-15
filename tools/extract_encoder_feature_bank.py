#!/usr/bin/env python3
"""Extract camera/LiDAR/radar encoder BEV features into resumable FP16 banks.

The output unit is one ``sequence/split`` partition.  Each modality is stored as
one NumPy ``.npy`` array in NCHW order so later analysis can use mmap without
loading an entire scene into RAM.  A validity bitmap makes interrupted runs
safe to resume, while CSV/JSONL manifests preserve the row-to-frame mapping.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader, Subset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import datasets  # noqa: E402
from models.skeletons import build_skeleton  # noqa: E402
from utils.util_config import cfg_from_yaml_file  # noqa: E402


SCHEMA_VERSION = "kradar-encoder-feature-bank/v1"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/baselines/ASF_v2_0_10scenes_offline_60_40.yml"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/EncoderFeatureBank/10scenes_fp16"
DEFAULT_SEQUENCES = ("1", "35", "46", "19", "58", "5", "22", "34", "9", "38")
MODALITY_ATTRS = {"camera": "cam", "lidar": "ldr", "radar": "rdr"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_path(value: str, base: Path = REPOSITORY_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def normalize_sequence(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("seq"):
        text = text[3:]
    return str(int(text))


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, include_hash: bool = True) -> Dict[str, Any]:
    path = path.resolve()
    record: Dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        record.update({"size_bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
        if include_hash:
            record["sha256"] = file_sha256(path)
    return record


def jsonable(value: Any) -> Any:
    if isinstance(value, EasyDict):
        value = dict(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def load_config(path: Path) -> EasyDict:
    return cfg_from_yaml_file(str(path), EasyDict())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_revision() -> Dict[str, Any]:
    git_dir = REPOSITORY_ROOT / ".git"
    head = git_dir / "HEAD"
    result: Dict[str, Any] = {"head": None, "reference": None}
    if not head.is_file():
        return result
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        reference = value[5:]
        ref_path = git_dir / reference
        result["reference"] = reference
        if ref_path.is_file():
            result["head"] = ref_path.read_text(encoding="utf-8").strip()
        else:
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith("#") and line.endswith(" " + reference):
                        result["head"] = line.split(" ", 1)[0]
                        break
    else:
        result["head"] = value
    return result


def environment_metadata() -> Dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append({
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            })
    return {
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_devices": cuda_devices,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def encoder_configuration(cfg: EasyDict) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for modality, config_name in (("camera", "CAMERA"), ("lidar", "LIDAR"), ("radar", "RADAR")):
        item = cfg.MODEL[config_name]
        module_config = resolve_path(str(item.CFG))
        pretrained = resolve_path(str(item.PRETRAINED)) if item.get("PRETRAINED") else None
        records[modality] = {
            "network_attribute": MODALITY_ATTRS[modality],
            "feature_key": str(item.KEY),
            "configured_channels": int(item.CHANNEL),
            "module_config": file_record(module_config),
            "pretrained_checkpoint": None if pretrained is None else file_record(pretrained),
        }
    return records


def extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    normalized = {}
    for key, value in payload.items():
        name = str(key)
        for prefix in ("module.", "network."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        normalized[name] = value
    return normalized


def load_encoder_checkpoint(
    network: torch.nn.Module,
    checkpoint: Path,
    checkpoint_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    print("Loading encoder tensors from checkpoint: {}".format(checkpoint), flush=True)
    checkpoint_info = dict(checkpoint_info or file_record(checkpoint))
    payload = torch.load(str(checkpoint), map_location="cpu")
    state = extract_state_dict(payload)
    prefixes = tuple(name + "." for name in MODALITY_ATTRS.values())
    selected = {name: value for name, value in state.items() if name.startswith(prefixes)}
    target_state = network.state_dict()
    expected = {name for name in target_state if name.startswith(prefixes)}
    missing = sorted(expected - set(selected))
    unexpected = sorted(set(selected) - expected)
    shape_mismatch = sorted(
        name for name in selected
        if name in expected and tuple(selected[name].shape) != tuple(target_state[name].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Encoder checkpoint is not an exact match: missing={}, unexpected={}, shape_mismatch={}".format(
                missing[:10], unexpected[:10], shape_mismatch[:10]
            )
        )
    incompatible = network.load_state_dict(selected, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("Unexpected encoder keys: {}".format(incompatible.unexpected_keys))
    checkpoint_info.update({
        "loading_policy": "exact encoder-only load",
        "loaded_tensor_count": len(selected),
        "ignored_non_encoder_tensor_count": len(state) - len(selected),
    })
    del payload, state, selected
    return checkpoint_info


def parameter_metadata(network: torch.nn.Module) -> Dict[str, Any]:
    result = {}
    for modality, attr in MODALITY_ATTRS.items():
        module = getattr(network, attr)
        total = sum(parameter.numel() for parameter in module.parameters())
        trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        result[modality] = {
            "class": "{}.{}".format(module.__class__.__module__, module.__class__.__name__),
            "parameter_count": total,
            "trainable_parameter_count_at_extraction": trainable,
            "training_mode_at_extraction": module.training,
        }
    return result


def build_dataset(cfg: EasyDict, sequence: str, split: str) -> Any:
    partition_cfg = EasyDict(jsonable(cfg))
    partition_cfg.DATASET.portion = [sequence]
    name = str(partition_cfg.DATASET.NAME)
    if name not in datasets.__all__:
        raise KeyError("Unknown dataset implementation: {}".format(name))
    dataset = datasets.__all__[name](cfg=partition_cfg, split=split)
    if not hasattr(dataset, "collate_fn"):
        raise AttributeError("Dataset has no collate_fn")
    # Point order must be stable even if an encoder config enables shuffling.
    if hasattr(dataset, "shuffle_points"):
        dataset.shuffle_points = False
    return dataset


def run_encoders(network: torch.nn.Module, batch: MutableMapping[str, Any], feature_keys: Mapping[str, str]) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        batch = network.cam(batch)
        batch = network.ldr(batch)
        batch = network.rdr(batch)
        outputs = {}
        for modality, key in feature_keys.items():
            if key not in batch:
                raise KeyError("{} encoder did not produce feature key {!r}".format(modality, key))
            feature = batch[key]
            if not torch.is_tensor(feature) or feature.ndim != 4:
                raise ValueError("{} must be a 4D NCHW tensor; got {}".format(key, type(feature)))
            outputs[modality] = feature.detach().to(device="cpu", dtype=torch.float16).contiguous()
    return outputs


def sample_record(row_index: int, sequence: str, split: str, meta: Mapping[str, Any]) -> Dict[str, Any]:
    clean = jsonable(meta)
    indexes = clean.get("idx", {}) if isinstance(clean, dict) else {}
    radar_index = indexes.get("rdr")
    labels = clean.get("label", []) if isinstance(clean, dict) else []
    class_counts = Counter()
    for label in labels or []:
        if isinstance(label, (list, tuple)) and label:
            class_counts[str(label[0])] += 1
    return {
        "row_index": row_index,
        "sample_id": "seq{}_{}_rdr{}".format(sequence, split, radar_index if radar_index is not None else row_index),
        "sequence": sequence,
        "split": split,
        "radar_index": radar_index,
        "lidar64_index": indexes.get("ldr64"),
        "camera_front_index": indexes.get("camf"),
        "lidar128_index": indexes.get("ldr128"),
        "camera_rear_index": indexes.get("camr"),
        "timestamp": indexes.get("tstamp"),
        "num_objects": clean.get("num_obj") if isinstance(clean, dict) else None,
        "class_counts": dict(sorted(class_counts.items())),
        "description": clean.get("desc") if isinstance(clean, dict) else None,
        "label_path": clean.get("label_v2_0") if isinstance(clean, dict) else None,
        "source_paths": clean.get("path") if isinstance(clean, dict) else None,
        "meta": clean,
    }


def write_manifests(partition_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    jsonl = "".join(json.dumps(jsonable(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    atomic_write_text(partition_dir / "sample_metadata.jsonl", jsonl)
    fields = [
        "row_index", "sample_id", "sequence", "split", "radar_index", "lidar64_index",
        "camera_front_index", "lidar128_index", "camera_rear_index", "timestamp",
        "num_objects", "class_counts", "description", "label_path", "source_paths",
    ]
    temporary = partition_dir / "manifest.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {name: record.get(name) for name in fields}
            for name in ("class_counts", "description", "source_paths"):
                row[name] = json.dumps(jsonable(row[name]), ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(partition_dir / "manifest.csv"))


def open_existing_arrays(partition_dir: Path, count: int) -> Tuple[Dict[str, np.memmap], np.memmap]:
    arrays = {}
    for modality in MODALITY_ATTRS:
        path = partition_dir / (modality + ".npy")
        if not path.is_file():
            raise RuntimeError("Incomplete partition is missing {}".format(path))
        array = np.load(str(path), mmap_mode="r+")
        if array.dtype != np.float16 or array.ndim != 4 or array.shape[0] != count:
            raise RuntimeError("Invalid resumable array {}: shape={}, dtype={}".format(path, array.shape, array.dtype))
        arrays[modality] = array
    valid_path = partition_dir / "valid.npy"
    if not valid_path.is_file():
        raise RuntimeError("Incomplete partition is missing {}".format(valid_path))
    valid = np.load(str(valid_path), mmap_mode="r+")
    if valid.shape != (count,) or valid.dtype != np.bool_:
        raise RuntimeError("Invalid validity bitmap: shape={}, dtype={}".format(valid.shape, valid.dtype))
    return arrays, valid


def create_arrays(partition_dir: Path, count: int, features: Mapping[str, torch.Tensor], reserve_gib: float) -> Tuple[Dict[str, np.memmap], np.memmap]:
    required = sum(count * int(feature[0].numel()) * np.dtype(np.float16).itemsize for feature in features.values())
    free = shutil.disk_usage(str(partition_dir)).free
    reserve = int(reserve_gib * (1024 ** 3))
    if free < required + reserve:
        raise RuntimeError(
            "Insufficient disk space for partition: need {:.2f} GiB plus {:.2f} GiB reserve, free {:.2f} GiB".format(
                required / (1024 ** 3), reserve_gib, free / (1024 ** 3)
            )
        )
    arrays = {}
    for modality, feature in features.items():
        shape = (count,) + tuple(int(value) for value in feature.shape[1:])
        arrays[modality] = np.lib.format.open_memmap(
            str(partition_dir / (modality + ".npy")), mode="w+", dtype=np.float16, shape=shape
        )
    valid = np.lib.format.open_memmap(
        str(partition_dir / "valid.npy"), mode="w+", dtype=np.bool_, shape=(count,)
    )
    valid[:] = False
    valid.flush()
    return arrays, valid


def array_file_metadata(path: Path, array: np.ndarray) -> Dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "format": "NumPy .npy v1/v2 (memory-mappable)",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "layout": "NCHW",
        "size_bytes": path.stat().st_size,
        "row_axis": "manifest.csv row_index",
    }


def flush_state(partition_dir: Path, arrays: Mapping[str, np.memmap], valid: np.memmap, records: Sequence[Optional[Mapping[str, Any]]]) -> None:
    for array in arrays.values():
        array.flush()
    valid.flush()
    atomic_write_json(partition_dir / "records.state.json", records)


def extract_partition(
    network: torch.nn.Module,
    cfg: EasyDict,
    output_dir: Path,
    sequence: str,
    split: str,
    feature_keys: Mapping[str, str],
    batch_size: int,
    num_workers: int,
    flush_every: int,
    reserve_gib: float,
    max_samples: int,
) -> Dict[str, Any]:
    partition_dir = output_dir / ("seq" + sequence) / split
    partition_dir.mkdir(parents=True, exist_ok=True)
    complete_path = partition_dir / "COMPLETE.json"
    if complete_path.is_file():
        result = json.loads(complete_path.read_text(encoding="utf-8"))
        print("[skip] seq{} {} is complete ({} samples)".format(sequence, split, result["sample_count"]), flush=True)
        return result

    print("[dataset] loading seq{} {}".format(sequence, split), flush=True)
    dataset = build_dataset(cfg, sequence, split)
    total_available = len(dataset)
    count = total_available if max_samples < 0 else min(total_available, max_samples)
    if count <= 0:
        raise RuntimeError("seq{} {} contains no samples".format(sequence, split))

    state_path = partition_dir / "records.state.json"
    existing = (partition_dir / "valid.npy").is_file()
    if existing:
        arrays, valid = open_existing_arrays(partition_dir, count)
        if not state_path.is_file():
            raise RuntimeError("Resume state is missing: {}".format(state_path))
        records = json.loads(state_path.read_text(encoding="utf-8"))
        if len(records) != count:
            raise RuntimeError("Resume record count does not match dataset")
    else:
        stray = [partition_dir / (name + ".npy") for name in MODALITY_ATTRS if (partition_dir / (name + ".npy")).exists()]
        if stray or state_path.exists():
            raise RuntimeError("Partition has inconsistent partial files; move it aside and retry: {}".format(partition_dir))
        arrays = {}
        valid = None
        records: List[Optional[Mapping[str, Any]]] = [None] * count

    pending = [index for index in range(count) if valid is None or not bool(valid[index]) or records[index] is None]
    if not pending:
        raise RuntimeError("Partition has no pending rows but no COMPLETE marker: {}".format(partition_dir))

    subset = Subset(dataset, pending)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=dataset.collate_fn, drop_last=False, pin_memory=False,
    )
    cursor = 0
    batches_since_flush = 0
    started = time.time()
    for batch in loader:
        if batch is None:
            raise RuntimeError("Dataset collate_fn returned None")
        batch_count = int(batch.get("batch_size", len(batch.get("meta", []))))
        rows = pending[cursor:cursor + batch_count]
        if len(rows) != batch_count or len(batch.get("meta", [])) != batch_count:
            raise RuntimeError("Batch size and metadata size disagree")
        features = run_encoders(network, batch, feature_keys)
        if not arrays:
            arrays, valid = create_arrays(partition_dir, count, features, reserve_gib)
            atomic_write_json(partition_dir / "records.state.json", records)
        assert valid is not None
        for modality, feature in features.items():
            if tuple(feature.shape[1:]) != tuple(arrays[modality].shape[1:]):
                raise RuntimeError("Feature shape changed for {}: {} vs {}".format(modality, feature.shape, arrays[modality].shape))
            arrays[modality][rows] = feature.numpy()
        for offset, row in enumerate(rows):
            records[row] = sample_record(row, sequence, split, batch["meta"][offset])
        # Mark rows valid only after every modality and metadata record are assigned.
        valid[rows] = True
        cursor += batch_count
        batches_since_flush += 1
        if batches_since_flush >= flush_every:
            flush_state(partition_dir, arrays, valid, records)
            batches_since_flush = 0
        elapsed = max(time.time() - started, 1e-6)
        done = cursor
        print(
            "[seq{} {}] {}/{} pending rows ({:.2f} frames/s)".format(
                sequence, split, done, len(pending), done / elapsed
            ), flush=True,
        )
        del batch, features

    assert valid is not None
    flush_state(partition_dir, arrays, valid, records)
    if not bool(np.all(valid)) or any(record is None for record in records):
        raise RuntimeError("Partition finished with invalid rows")
    final_records = [record for record in records if record is not None]
    sample_ids = [str(record["sample_id"]) for record in final_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample_id values detected")
    write_manifests(partition_dir, final_records)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_utc": utc_now(),
        "sequence": sequence,
        "split": split,
        "sample_count": count,
        "total_available_before_max_samples": total_available,
        "is_complete_dataset_partition": count == total_available,
        "feature_files": {
            modality: array_file_metadata(partition_dir / (modality + ".npy"), array)
            for modality, array in arrays.items()
        },
        "validity_file": file_record(partition_dir / "valid.npy", include_hash=False),
        "manifest_csv": file_record(partition_dir / "manifest.csv", include_hash=True),
        "sample_metadata_jsonl": file_record(partition_dir / "sample_metadata.jsonl", include_hash=True),
        "elapsed_seconds_this_process": time.time() - started,
    }
    atomic_write_json(partition_dir / "partition_meta.json", result)
    atomic_write_json(complete_path, result)
    try:
        state_path.unlink()
    except FileNotFoundError:
        pass
    del loader, subset, dataset, arrays, valid
    print("[done] seq{} {}: {} samples".format(sequence, split, count), flush=True)
    return result


def build_global_metadata(
    args: argparse.Namespace,
    cfg: EasyDict,
    config_path: Path,
    output_dir: Path,
    checkpoint: Optional[Path],
) -> Dict[str, Any]:
    split_files = [resolve_path(str(value)) for value in cfg.DATASET.path_data.split]
    script_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "purpose": "Full per-frame, pre-fusion BEV feature bank for scene characterization and automatic scene identification research.",
        "content_scope": {
            "included": ["camera encoder output", "LiDAR encoder output", "radar encoder output", "per-frame source metadata"],
            "excluded": ["fused feature", "detection-head output", "optimizer state", "gradients"],
        },
        "repository": {"root": str(REPOSITORY_ROOT), "git": git_revision()},
        "extractor": file_record(script_path),
        "source_config": file_record(config_path),
        "source_config_resolved": jsonable(cfg),
        "dataset": {
            "name": str(cfg.DATASET.NAME),
            "label_version": str(cfg.DATASET.get("label_version", "unknown")),
            "sequences": list(args.sequences),
            "splits": list(args.splits),
            "data_roots": [str(resolve_path(str(value))) for value in cfg.DATASET.path_data.list_dir_kradar],
            "split_files": [file_record(path) for path in split_files],
            "item": jsonable(cfg.DATASET.item),
            "roi": jsonable(cfg.DATASET.roi),
            "camera": jsonable(cfg.DATASET.get("cam")),
            "camera_processing": jsonable(cfg.DATASET.get("cam_process")),
            "remove_zero_object_frames": bool(cfg.DATASET.label.get("remove_0_obj", False)),
        },
        "model": {
            "skeleton": str(cfg.MODEL.SKELETON),
            "encoders": encoder_configuration(cfg),
            "external_checkpoint": None if checkpoint is None else file_record(checkpoint),
        },
        "storage": {
            "root": str(output_dir),
            "feature_format": "one NumPy .npy memory map per sequence/split/modality",
            "dtype": "float16",
            "layout": "NCHW",
            "compression": "none",
            "random_access": True,
            "integrity_protocol": "valid.npy plus atomic state/manifest/COMPLETE metadata",
            "feature_file_hashes": "not computed because hashing hundreds of GiB would add a full extra read pass",
        },
        "extraction": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "flush_every_batches": args.flush_every,
            "seed": args.seed,
            "max_samples_per_partition": args.max_samples_per_partition,
            "inference": "torch.no_grad",
            "module_mode": "eval",
            "point_shuffle": False,
            "data_order": "dataset order, shuffle=False",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "environment": environment_metadata(),
        "partitions": {},
    }


def validate_existing_identity(existing: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    checks = (
        ("schema_version", existing.get("schema_version"), current.get("schema_version")),
        ("source config SHA256", existing.get("source_config", {}).get("sha256"), current.get("source_config", {}).get("sha256")),
        ("sequences", existing.get("dataset", {}).get("sequences"), current.get("dataset", {}).get("sequences")),
        ("splits", existing.get("dataset", {}).get("splits"), current.get("dataset", {}).get("splits")),
        ("max_samples", existing.get("extraction", {}).get("max_samples_per_partition"), current.get("extraction", {}).get("max_samples_per_partition")),
        ("external checkpoint SHA256", existing.get("model", {}).get("external_checkpoint", {}).get("sha256") if existing.get("model", {}).get("external_checkpoint") else None, current.get("model", {}).get("external_checkpoint", {}).get("sha256") if current.get("model", {}).get("external_checkpoint") else None),
    )
    mismatches = ["{}: {!r} != {!r}".format(name, old, new) for name, old, new in checks if old != new]
    if mismatches:
        raise RuntimeError("Output directory belongs to a different extraction:\n" + "\n".join(mismatches))


def write_readme(output_dir: Path) -> None:
    text = """# K-Radar encoder feature bank

This directory contains pre-fusion Camera, LiDAR, and Radar BEV encoder outputs.
Features are uncompressed FP16 NumPy arrays with shape `[N,C,H,W]`, split by
`seq<id>/<train|test>/`. Use `np.load(path, mmap_mode='r')` for zero-copy,
row-wise access. `manifest.csv` maps each array row to its source frame;
`sample_metadata.jsonl` preserves the complete dataset metadata. Only trust a
partition with `COMPLETE.json`, or consult `valid.npy` when resuming extraction.

`dataset_meta.json` is the authoritative global provenance record.
"""
    atomic_write_text(output_dir / "README.md", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Model/dataset YAML used to build the three encoders")
    parser.add_argument("--checkpoint", default=None, help="Optional full model checkpoint; only cam/ldr/rdr tensors are loaded")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=16, help="Flush arrays and resume state every N batches")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--reserve-free-gib", type=float, default=20.0)
    parser.add_argument("--max-samples-per-partition", type=int, default=-1, help="Smoke-test limit; -1 extracts everything")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.flush_every <= 0:
        raise ValueError("batch-size/flush-every must be positive and num-workers nonnegative")
    if args.max_samples_per_partition == 0 or args.max_samples_per_partition < -1:
        raise ValueError("max-samples-per-partition must be -1 or positive")
    args.sequences = [normalize_sequence(value) for value in args.sequences]
    if len(set(args.sequences)) != len(args.sequences):
        raise ValueError("Duplicate sequences are not allowed")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("Duplicate splits are not allowed")
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else None
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the configured encoders")

    os.chdir(str(REPOSITORY_ROOT))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    cfg = load_config(config_path)
    metadata = build_global_metadata(args, cfg, config_path, output_dir, checkpoint)
    global_path = output_dir / "dataset_meta.json"
    if global_path.is_file():
        existing = json.loads(global_path.read_text(encoding="utf-8"))
        validate_existing_identity(existing, metadata)
        metadata["created_utc"] = existing.get("created_utc", metadata["created_utc"])
        metadata["partitions"] = existing.get("partitions", {})
    write_readme(output_dir)
    shutil.copy2(str(config_path), str(output_dir / "source_config.yml"))
    atomic_write_json(global_path, metadata)

    print("Building {} and loading three encoders...".format(cfg.MODEL.SKELETON), flush=True)
    network = build_skeleton(cfg).cuda()
    if checkpoint is not None:
        metadata["model"]["external_checkpoint"] = load_encoder_checkpoint(
            network, checkpoint, metadata["model"]["external_checkpoint"]
        )
    network.eval()
    for attr in MODALITY_ATTRS.values():
        getattr(network, attr).eval()
        for parameter in getattr(network, attr).parameters():
            parameter.requires_grad_(False)
    metadata["model"]["runtime_encoders"] = parameter_metadata(network)
    feature_keys = {
        modality: metadata["model"]["encoders"][modality]["feature_key"]
        for modality in MODALITY_ATTRS
    }
    atomic_write_json(global_path, metadata)

    try:
        for sequence in args.sequences:
            for split in args.splits:
                key = "seq{}/{}".format(sequence, split)
                result = extract_partition(
                    network=network,
                    cfg=cfg,
                    output_dir=output_dir,
                    sequence=sequence,
                    split=split,
                    feature_keys=feature_keys,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    flush_every=args.flush_every,
                    reserve_gib=args.reserve_free_gib,
                    max_samples=args.max_samples_per_partition,
                )
                metadata["partitions"][key] = result
                metadata["updated_utc"] = utc_now()
                atomic_write_json(global_path, metadata)
        metadata["status"] = "complete"
        metadata["completed_utc"] = utc_now()
    except BaseException as error:
        metadata["status"] = "interrupted_or_failed"
        metadata["last_error"] = {"type": type(error).__name__, "message": str(error), "utc": utc_now()}
        raise
    finally:
        metadata["updated_utc"] = utc_now()
        atomic_write_json(global_path, metadata)

    total_samples = sum(int(item["sample_count"]) for item in metadata["partitions"].values())
    total_bytes = sum(
        int(feature["size_bytes"])
        for item in metadata["partitions"].values()
        for feature in item["feature_files"].values()
    )
    print("All partitions complete: {} samples, {:.2f} GiB".format(total_samples, total_bytes / (1024 ** 3)), flush=True)


if __name__ == "__main__":
    # Some versions of this repository's Open3D/Numba dependencies can abort
    # during CPython's C-extension teardown ("free(): invalid pointer") even
    # after all work has completed.  All feature/state files are explicitly
    # flushed above, so use a hard process exit after flushing Python streams.
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        traceback.print_exc()
        exit_code = 130
    except SystemExit as error:
        exit_code = int(error.code or 0)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(exit_code)
