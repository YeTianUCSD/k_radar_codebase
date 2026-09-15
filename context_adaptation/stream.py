"""Sequential dataset stream utilities that never expose labels to routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import torch
import yaml
from easydict import EasyDict

import datasets
from utils.util_config import cfg_from_yaml_file


PathLike = Union[str, Path]


@dataclass(frozen=True)
class StreamSegment:
    config_path: str
    split: str = "train"
    name_for_metrics_only: str = "unknown"
    start: int = 0
    stop: int = -1
    max_steps: int = -1
    sequences: tuple[str, ...] = ()


@dataclass(frozen=True)
class StreamBatch:
    segment_index: int
    segment_name_for_metrics_only: str
    local_step: int
    batch: Any


def _resolve_path(path: str, repository_root: Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    return str(candidate.resolve())


def load_stream_manifest(
    path: PathLike,
    *,
    repository_root: Optional[PathLike] = None,
) -> list[StreamSegment]:
    manifest_path = Path(path).expanduser().resolve()
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    with manifest_path.open("r") as stream_file:
        payload = yaml.safe_load(stream_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("STREAM"), list):
        raise ValueError("Stream manifest must contain a top-level STREAM list.")

    segments = []
    for index, item in enumerate(payload["STREAM"]):
        if not isinstance(item, dict) or "CONFIG" not in item:
            raise ValueError(f"STREAM[{index}] must contain CONFIG.")
        split = str(item.get("SPLIT", "train")).lower()
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split in STREAM[{index}]: {split}")
        start = int(item.get("START", 0))
        stop = int(item.get("STOP", -1))
        max_steps = int(item.get("MAX_STEPS", -1))
        raw_sequences = item.get("SEQUENCES", ())
        if raw_sequences is None:
            raw_sequences = ()
        if not isinstance(raw_sequences, (list, tuple)):
            raise ValueError(f"STREAM[{index}].SEQUENCES must be a list.")
        sequences = tuple(str(value) for value in raw_sequences)
        if any(not value for value in sequences):
            raise ValueError(
                f"STREAM[{index}].SEQUENCES must not contain empty values."
            )
        if len(set(sequences)) != len(sequences):
            raise ValueError(
                f"STREAM[{index}].SEQUENCES must not contain duplicates."
            )
        if start < 0:
            raise ValueError("START must be nonnegative.")
        if stop != -1 and stop <= start:
            raise ValueError("STOP must be -1 or greater than START.")
        if max_steps == 0 or max_steps < -1:
            raise ValueError("MAX_STEPS must be -1 or a positive integer.")
        segments.append(
            StreamSegment(
                config_path=_resolve_path(str(item["CONFIG"]), root),
                split=split,
                name_for_metrics_only=str(
                    item.get("NAME_FOR_METRICS_ONLY", f"segment_{index:04d}")
                ),
                start=start,
                stop=stop,
                max_steps=max_steps,
                sequences=sequences,
            )
        )
    if not segments:
        raise ValueError("Stream manifest must contain at least one segment.")
    return segments


def load_fresh_config(path: PathLike) -> EasyDict:
    """Load a config without mutating the repository-wide global cfg object."""
    return cfg_from_yaml_file(str(path), EasyDict())


def build_segment_dataset(segment: StreamSegment) -> Any:
    cfg = load_fresh_config(segment.config_path)
    if segment.sequences:
        cfg.DATASET.portion = list(segment.sequences)
    dataset_name = str(cfg.DATASET.NAME)
    if dataset_name not in datasets.__all__:
        raise KeyError(f"Unknown dataset implementation: {dataset_name}")
    return datasets.__all__[dataset_name](cfg=cfg, split=segment.split)


def iter_stream_batches(
    segments: Sequence[StreamSegment],
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    dataset_overrides: Optional[Mapping[int, Any]] = None,
    segment_index_offset: int = 0,
    first_segment_local_step_offset: int = 0,
) -> Iterator[StreamBatch]:
    """Yield ordered batches while keeping metric-only names outside batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")
    if segment_index_offset < 0:
        raise ValueError("segment_index_offset must be nonnegative.")
    if first_segment_local_step_offset < 0:
        raise ValueError("first_segment_local_step_offset must be nonnegative.")
    overrides = dict(dataset_overrides or {})
    unknown_overrides = set(overrides) - set(range(len(segments)))
    if unknown_overrides:
        raise ValueError(f"Invalid dataset override indices: {sorted(unknown_overrides)}")

    for segment_index, segment in enumerate(segments):
        dataset = overrides.get(segment_index)
        owns_dataset = dataset is None
        if dataset is None:
            dataset = build_segment_dataset(segment)
        collate_fn = getattr(dataset, "collate_fn", None)
        if collate_fn is None:
            raise AttributeError(
                f"Dataset for segment {segment_index} has no collate_fn."
            )
        dataset_size = len(dataset)
        stop = dataset_size if segment.stop == -1 else segment.stop
        if segment.start >= dataset_size or stop > dataset_size:
            raise ValueError(
                f"Segment {segment_index} range [{segment.start}, {stop}) exceeds "
                f"dataset size {dataset_size}."
            )
        selected_dataset = torch.utils.data.Subset(
            dataset,
            range(segment.start, stop),
        )
        loader = torch.utils.data.DataLoader(
            selected_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        for local_step, batch in enumerate(loader, start=1):
            if segment.max_steps > 0 and local_step > segment.max_steps:
                break
            yield StreamBatch(
                segment_index=segment_index + segment_index_offset,
                segment_name_for_metrics_only=segment.name_for_metrics_only,
                local_step=(
                    local_step
                    if segment_index > 0
                    else local_step + first_segment_local_step_offset
                ),
                batch=batch,
            )
        del loader, selected_dataset
        if owns_dataset:
            del dataset
