#!/usr/bin/env python3
"""Build resumable, compact scene descriptors from encoder BEV features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.common import atomic_write_json, file_sha256, utc_now  # noqa: E402
from scene_discovery.descriptors import DESCRIPTOR_KINDS, compute_descriptors, descriptor_dimension  # noqa: E402
from scene_discovery.feature_bank import (  # noqa: E402
    MODALITIES,
    SPLITS,
    DescriptorBank,
    FeatureBank,
    Partition,
)


INDEX_FIELDS = (
    "row_index",
    "sample_id",
    "sequence",
    "split",
    "timestamp",
    "radar_index",
    "lidar64_index",
    "camera_front_index",
    "capture_time",
    "climate",
    "road_type",
    "num_objects",
    "source_partition",
    "source_row_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-bank",
        type=Path,
        default=REPOSITORY_ROOT / "results/EncoderFeatureBank/10scenes_fp16",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/descriptors/v1",
    )
    parser.add_argument("--descriptors", nargs="+", choices=DESCRIPTOR_KINDS, default=list(DESCRIPTOR_KINDS))
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--spatial-grid", nargs=2, type=int, default=(2, 2), metavar=("ROWS", "COLS"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete this descriptor output and rebuild every partition. Resume is the default.",
    )
    return parser.parse_args()


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_description(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def csv_value(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def build_configuration(
    bank: FeatureBank,
    source_meta_sha256: str,
    modalities: Sequence[str],
    kinds: Sequence[str],
    splits: Sequence[str],
    spatial_grid: Tuple[int, int],
) -> Dict[str, Any]:
    return {
        "source_feature_bank": str(bank.root),
        "source_dataset_meta_sha256": source_meta_sha256,
        "sequences": list(bank.sequences),
        "splits": list(splits),
        "modalities": list(modalities),
        "descriptor_kinds": list(kinds),
        "descriptor_dtype": "float32",
        "spatial_grid": list(spatial_grid),
        "definitions_version": 1,
    }


def source_partition_signature(
    partition: Partition,
    build_signature: str,
    modalities: Sequence[str],
) -> str:
    files = []
    for path in [
        partition.path / "partition_meta.json",
        partition.path / "manifest.csv",
        partition.path / "COMPLETE.json",
        *[partition.array_path(modality) for modality in modalities],
    ]:
        stat = path.stat()
        files.append({
            "relative_path": str(path.relative_to(partition.path)),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return stable_hash({
        "build_signature": build_signature,
        "sequence": partition.sequence,
        "split": partition.split,
        "sample_count": partition.sample_count,
        "source_files": files,
    })


def partition_output_dir(output_root: Path, partition: Partition) -> Path:
    return output_root / "_partitions" / f"seq{partition.sequence}" / partition.split


def validate_partition_output(
    output_dir: Path,
    partition: Partition,
    signature: str,
    modalities: Sequence[str],
    kinds: Sequence[str],
) -> bool:
    complete_path = output_dir / "COMPLETE.json"
    index_path = output_dir / "index.csv"
    if not complete_path.is_file() or not index_path.is_file():
        return False
    try:
        with complete_path.open("r", encoding="utf-8") as stream:
            complete = json.load(stream)
        if complete.get("source_partition_signature") != signature:
            return False
        if int(complete.get("sample_count", -1)) != partition.sample_count:
            return False
        with index_path.open("r", encoding="utf-8", newline="") as stream:
            if sum(1 for _ in csv.DictReader(stream)) != partition.sample_count:
                return False
        for modality in modalities:
            for kind in kinds:
                array = np.load(str(output_dir / f"{modality}_{kind}.npy"), mmap_mode="r")
                if array.ndim != 2 or array.shape[0] != partition.sample_count or array.dtype != np.float32:
                    return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def build_index_rows(partition: Partition):
    manifest = partition.load_manifest()
    if len(manifest) != partition.sample_count:
        raise RuntimeError(f"Manifest mismatch in {partition.path}")
    for local_position, (_, row) in enumerate(manifest.iterrows()):
        description = parse_description(row.get("description", {}))
        yield {
            "row_index": local_position,
            "sample_id": csv_value(row.get("sample_id", "")),
            "sequence": partition.sequence,
            "split": partition.split,
            "timestamp": csv_value(row.get("timestamp", "")),
            "radar_index": csv_value(row.get("radar_index", "")),
            "lidar64_index": csv_value(row.get("lidar64_index", "")),
            "camera_front_index": csv_value(row.get("camera_front_index", "")),
            "capture_time": csv_value(description.get("capture_time", "")),
            "climate": csv_value(description.get("climate", "")),
            "road_type": csv_value(description.get("road_type", "")),
            "num_objects": csv_value(row.get("num_objects", "")),
            "source_partition": f"seq{partition.sequence}/{partition.split}",
            "source_row_index": int(row["row_index"]),
        }


def build_partition(
    partition: Partition,
    output_dir: Path,
    signature: str,
    modalities: Sequence[str],
    kinds: Sequence[str],
    chunk_size: int,
    spatial_grid: Tuple[int, int],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    writers: Dict[Tuple[str, str], np.memmap] = {}
    temporary_paths: Dict[Tuple[str, str], Path] = {}
    output_shapes: Dict[str, Dict[str, List[int]]] = {}

    for modality in modalities:
        feature = partition.load_array(modality)
        if feature.ndim != 4 or feature.shape[0] != partition.sample_count:
            raise RuntimeError(f"Unexpected {modality} shape in {partition.path}: {feature.shape}")
        channels = int(feature.shape[1])
        output_shapes[modality] = {}
        for kind in kinds:
            shape = (partition.sample_count, descriptor_dimension(channels, kind, spatial_grid))
            output_shapes[modality][kind] = list(shape)
            final = output_dir / f"{modality}_{kind}.npy"
            temporary = final.with_name(final.name + ".tmp")
            temporary_paths[(modality, kind)] = temporary
            writers[(modality, kind)] = np.lib.format.open_memmap(
                str(temporary), mode="w+", dtype=np.float32, shape=shape
            )
        for start in range(0, partition.sample_count, chunk_size):
            stop = min(start + chunk_size, partition.sample_count)
            values = compute_descriptors(feature[start:stop], kinds, spatial_grid)
            for kind, descriptor in values.items():
                writers[(modality, kind)][start:stop] = descriptor
        del feature

    for writer in writers.values():
        writer.flush()
    writers.clear()
    for key, temporary in temporary_paths.items():
        os.replace(str(temporary), str(output_dir / f"{key[0]}_{key[1]}.npy"))

    index_path = output_dir / "index.csv"
    index_temporary = index_path.with_name(index_path.name + ".tmp")
    with index_temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(build_index_rows(partition))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(index_temporary), str(index_path))

    complete = {
        "schema_version": "kradar-scene-descriptor-partition/v1",
        "status": "complete",
        "completed_utc": utc_now(),
        "sequence": partition.sequence,
        "split": partition.split,
        "sample_count": partition.sample_count,
        "source_partition": str(partition.path),
        "source_partition_signature": signature,
        "output_shapes": output_shapes,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output_dir / "COMPLETE.json", complete)
    return complete


def consolidate_split(
    bank: FeatureBank,
    split: str,
    output_root: Path,
    modalities: Sequence[str],
    kinds: Sequence[str],
) -> Dict[str, Any]:
    partitions = list(bank.partitions((split,)))
    total = sum(partition.sample_count for partition in partitions)
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: Dict[Tuple[str, str], np.memmap] = {}
    temporary_paths: Dict[Tuple[str, str], Path] = {}
    arrays_meta: Dict[str, Dict[str, Any]] = {}
    started = time.time()

    for modality in modalities:
        arrays_meta[modality] = {}
        for kind in kinds:
            first_path = partition_output_dir(output_root, partitions[0]) / f"{modality}_{kind}.npy"
            first = np.load(str(first_path), mmap_mode="r")
            shape = (total, int(first.shape[1]))
            final = output_dir / f"{modality}_{kind}.npy"
            temporary = final.with_name(final.name + ".tmp")
            temporary_paths[(modality, kind)] = temporary
            writers[(modality, kind)] = np.lib.format.open_memmap(
                str(temporary), mode="w+", dtype=np.float32, shape=shape
            )
            arrays_meta[modality][kind] = {"shape": list(shape), "dtype": "float32"}

    index_path = output_dir / "index.csv"
    index_temporary = index_path.with_name(index_path.name + ".tmp")
    offset = 0
    with index_temporary.open("w", encoding="utf-8", newline="") as output_stream:
        index_writer = csv.DictWriter(output_stream, fieldnames=INDEX_FIELDS)
        index_writer.writeheader()
        for partition in partitions:
            part_dir = partition_output_dir(output_root, partition)
            for modality in modalities:
                for kind in kinds:
                    source = np.load(str(part_dir / f"{modality}_{kind}.npy"), mmap_mode="r")
                    writers[(modality, kind)][offset : offset + partition.sample_count] = source
            with (part_dir / "index.csv").open("r", encoding="utf-8", newline="") as input_stream:
                for row in csv.DictReader(input_stream):
                    row["row_index"] = offset + int(row["row_index"])
                    index_writer.writerow(row)
            offset += partition.sample_count
        output_stream.flush()
        os.fsync(output_stream.fileno())

    if offset != total:
        raise RuntimeError(f"Consolidation row mismatch for {split}: {offset} != {total}")
    for writer in writers.values():
        writer.flush()
    writers.clear()
    for key, temporary in temporary_paths.items():
        os.replace(str(temporary), str(output_dir / f"{key[0]}_{key[1]}.npy"))
    os.replace(str(index_temporary), str(index_path))
    return {
        "status": "complete",
        "sample_count": total,
        "index_csv": str(index_path.resolve()),
        "arrays": arrays_meta,
        "elapsed_seconds": time.time() - started,
    }


def validate_complete_output(
    bank: FeatureBank,
    output_root: Path,
    metadata: Mapping[str, Any],
    build_signature: str,
    modalities: Sequence[str],
    kinds: Sequence[str],
    splits: Sequence[str],
) -> None:
    for partition in bank.partitions(splits):
        signature = source_partition_signature(partition, build_signature, modalities)
        if not validate_partition_output(
            partition_output_dir(output_root, partition),
            partition,
            signature,
            modalities,
            kinds,
        ):
            raise RuntimeError(
                f"Completed descriptor partition is stale or invalid: seq{partition.sequence}/{partition.split}"
            )
    descriptor_bank = DescriptorBank(output_root)
    for split in splits:
        descriptor_bank.index(split)
        for modality in modalities:
            for kind in kinds:
                descriptor_bank.array(split, modality, kind)


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if min(args.spatial_grid) < 1:
        raise ValueError("--spatial-grid values must be positive")

    output_root = args.output_root.expanduser().resolve()
    bank = FeatureBank(args.feature_bank)
    source_meta_path = bank.root / "dataset_meta.json"
    source_meta_sha256 = file_sha256(source_meta_path)
    configuration = build_configuration(
        bank,
        source_meta_sha256,
        args.modalities,
        args.descriptors,
        args.splits,
        tuple(args.spatial_grid),
    )
    build_signature = stable_hash(configuration)
    meta_path = output_root / "descriptor_meta.json"

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    if output_root.exists() and not meta_path.is_file():
        raise RuntimeError(
            f"Output exists without descriptor_meta.json; choose another path or pass --overwrite: {output_root}"
        )
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("build_signature") != build_signature:
            raise RuntimeError(
                "Existing descriptor configuration or source FeatureBank does not match this request. "
                "Use another output path, or pass --overwrite to rebuild explicitly."
            )
        if (output_root / "COMPLETE.json").is_file() and metadata.get("status") == "complete":
            validate_complete_output(
                bank, output_root, metadata, build_signature,
                args.modalities, args.descriptors, args.splits,
            )
            print(f"Descriptor bank is already complete and verified: {output_root}")
            return
        metadata["status"] = "building"
        metadata["resumed_utc"] = utc_now()
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": "kradar-scene-descriptor-bank/v2",
            "status": "building",
            "created_utc": utc_now(),
            **configuration,
            "build_signature": build_signature,
            "chunk_size": args.chunk_size,
            "definitions": {
                "mean": "per-channel mean over H,W",
                "mean_std": "concatenated per-channel mean and population standard deviation over H,W",
                "spatial": "global mean/std followed by row-major grid-cell means",
            },
            "partition_progress": {},
            "partitions": {},
        }
    complete_path = output_root / "COMPLETE.json"
    if complete_path.exists():
        complete_path.unlink()
    atomic_write_json(meta_path, metadata)
    started = time.time()

    try:
        for partition in bank.partitions(args.splits):
            part_dir = partition_output_dir(output_root, partition)
            signature = source_partition_signature(partition, build_signature, args.modalities)
            key = f"seq{partition.sequence}/{partition.split}"
            if validate_partition_output(
                part_dir, partition, signature, args.modalities, args.descriptors
            ):
                print(f"[skip] {key}: complete and source-matched", flush=True)
                with (part_dir / "COMPLETE.json").open("r", encoding="utf-8") as stream:
                    metadata["partition_progress"][key] = json.load(stream)
                continue
            print(f"[build] {key}: {partition.sample_count} samples", flush=True)
            metadata["partition_progress"][key] = build_partition(
                partition,
                part_dir,
                signature,
                args.modalities,
                args.descriptors,
                args.chunk_size,
                tuple(args.spatial_grid),
            )
            metadata["updated_utc"] = utc_now()
            atomic_write_json(meta_path, metadata)

        current_source_hash = file_sha256(source_meta_path)
        if current_source_hash != source_meta_sha256:
            raise RuntimeError("Source dataset_meta.json changed while descriptors were being built")
        for split in args.splits:
            print(f"[consolidate] {split}", flush=True)
            metadata["partitions"][split] = consolidate_split(
                bank, split, output_root, args.modalities, args.descriptors
            )
            metadata["updated_utc"] = utc_now()
            atomic_write_json(meta_path, metadata)

        metadata["status"] = "complete"
        metadata["completed_utc"] = utc_now()
        metadata["elapsed_seconds_this_process"] = time.time() - started
        atomic_write_json(meta_path, metadata)
        atomic_write_json(complete_path, {
            "schema_version": "kradar-scene-descriptor-bank/v2",
            "status": "complete",
            "completed_utc": metadata["completed_utc"],
            "build_signature": build_signature,
            "source_dataset_meta_sha256": source_meta_sha256,
            "sample_count": sum(item["sample_count"] for item in metadata["partitions"].values()),
        })
        validate_complete_output(
            bank, output_root, metadata, build_signature,
            args.modalities, args.descriptors, args.splits,
        )
    except Exception as error:
        metadata["status"] = "incomplete"
        metadata["interrupted_utc"] = utc_now()
        metadata["last_error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(meta_path, metadata)
        raise
    print(f"Descriptor bank complete: {output_root}")


if __name__ == "__main__":
    main()
