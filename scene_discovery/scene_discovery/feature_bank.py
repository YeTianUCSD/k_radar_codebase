from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from .common import file_sha256


DEFAULT_SEQUENCES = ("1", "35", "46", "19", "58", "5", "22", "34", "9", "38")
MODALITIES = ("camera", "lidar", "radar")
SPLITS = ("train", "test")
MANIFEST_STRING_COLUMNS = {
    "sample_id": str,
    "sequence": str,
    "split": str,
    "timestamp": str,
    "radar_index": str,
    "lidar64_index": str,
    "camera_front_index": str,
    "lidar128_index": str,
    "camera_rear_index": str,
}


def normalize_sequence(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("seq"):
        text = text[3:]
    return str(int(text))


@dataclass(frozen=True)
class Partition:
    sequence: str
    split: str
    path: Path
    sample_count: int

    def array_path(self, modality: str) -> Path:
        if modality not in MODALITIES:
            raise ValueError(f"Unknown modality: {modality}")
        return self.path / f"{modality}.npy"

    def load_array(self, modality: str, mmap_mode: str = "r") -> np.ndarray:
        return np.load(str(self.array_path(modality)), mmap_mode=mmap_mode)

    def load_valid(self) -> np.ndarray:
        return np.load(str(self.path / "valid.npy"), mmap_mode="r")

    def load_manifest(self) -> pd.DataFrame:
        frame = pd.read_csv(self.path / "manifest.csv", dtype=MANIFEST_STRING_COLUMNS)
        frame["sequence"] = frame["sequence"].map(normalize_sequence)
        frame["row_index"] = pd.to_numeric(frame["row_index"], errors="raise").astype(np.int64)
        return frame


class FeatureBank:
    def __init__(self, root: Path, sequences: Optional[Sequence[object]] = None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Feature bank does not exist: {self.root}")
        meta_path = self.root / "dataset_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing dataset_meta.json: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as stream:
            self.metadata = json.load(stream)
        configured = self.metadata.get("dataset", {}).get("sequences", DEFAULT_SEQUENCES)
        selected = configured if sequences is None else sequences
        self.sequences = tuple(normalize_sequence(value) for value in selected)

    def partition(self, sequence: object, split: str) -> Partition:
        sequence = normalize_sequence(sequence)
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        path = self.root / f"seq{sequence}" / split
        meta_path = path / "partition_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing partition metadata: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        return Partition(sequence, split, path, int(metadata["sample_count"]))

    def partitions(self, splits: Sequence[str] = SPLITS) -> Iterator[Partition]:
        for sequence in self.sequences:
            for split in splits:
                yield self.partition(sequence, split)

    def sample_counts(self) -> Dict[str, Dict[str, int]]:
        result: Dict[str, Dict[str, int]] = {}
        for partition in self.partitions():
            result.setdefault(partition.sequence, {})[partition.split] = partition.sample_count
        return result


class DescriptorBank:
    """Small, memory-mappable descriptors derived from a complete FeatureBank."""

    def __init__(self, root: Path, verify_source: bool = True):
        self.root = Path(root).expanduser().resolve()
        meta_path = self.root / "descriptor_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing descriptor_meta.json: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as stream:
            self.metadata = json.load(stream)
        complete_path = self.root / "COMPLETE.json"
        if not complete_path.is_file() or self.metadata.get("status") != "complete":
            raise RuntimeError(f"Descriptor bank is incomplete: {self.root}")
        with complete_path.open("r", encoding="utf-8") as stream:
            complete = json.load(stream)
        if (
            complete.get("status") != "complete"
            or complete.get("build_signature") != self.metadata.get("build_signature")
        ):
            raise RuntimeError(f"Descriptor completion marker does not match metadata: {self.root}")
        if verify_source:
            source_root = Path(self.metadata["source_feature_bank"])
            source_meta = source_root / "dataset_meta.json"
            if not source_meta.is_file():
                raise FileNotFoundError(f"Descriptor source metadata is unavailable: {source_meta}")
            expected = self.metadata.get("source_dataset_meta_sha256")
            actual = file_sha256(source_meta)
            if not expected or actual != expected:
                raise RuntimeError(
                    "Source FeatureBank metadata changed after descriptor creation: "
                    f"expected {expected}, got {actual}"
                )
        self._indices: Dict[str, pd.DataFrame] = {}

    def index(self, split: str) -> pd.DataFrame:
        if split not in self.metadata.get("splits", []):
            raise ValueError(f"Split {split!r} is not present in this descriptor bank")
        if split not in self._indices:
            dtype = {
                key: value for key, value in MANIFEST_STRING_COLUMNS.items()
                if key not in {"lidar128_index", "camera_rear_index"}
            }
            frame = pd.read_csv(self.root / split / "index.csv", dtype=dtype)
            frame["sequence"] = frame["sequence"].map(normalize_sequence)
            frame["row_index"] = pd.to_numeric(frame["row_index"], errors="raise").astype(np.int64)
            expected = int(self.metadata["partitions"][split]["sample_count"])
            if len(frame) != expected or not np.array_equal(frame["row_index"], np.arange(expected)):
                raise RuntimeError(f"Descriptor index is not aligned for split {split}")
            self._indices[split] = frame
        return self._indices[split].copy()

    def array(self, split: str, modality: str, descriptor: str) -> np.ndarray:
        if modality not in self.metadata.get("modalities", []):
            raise ValueError(f"Modality {modality!r} is not present in this descriptor bank")
        if descriptor not in self.metadata.get("descriptor_kinds", []):
            raise ValueError(f"Descriptor {descriptor!r} is not present in this descriptor bank")
        array = np.load(str(self.root / split / f"{modality}_{descriptor}.npy"), mmap_mode="r")
        expected = int(self.metadata["partitions"][split]["sample_count"])
        if array.ndim != 2 or array.shape[0] != expected or array.dtype != np.float32:
            raise RuntimeError(
                f"Invalid descriptor array {modality}/{descriptor}/{split}: "
                f"shape={array.shape}, dtype={array.dtype}, expected rows={expected}"
            )
        return array
