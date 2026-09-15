"""Build causal train/test manifests using the dataset's usable samples.

All raw frames are assigned to exactly one manifest so the K-Radar dataset can
look up every label.  The temporal boundary is chosen after ``ratio`` of the
samples that survive the configured label, class, ROI, and remove-zero-object
filters.  Consequently the effective dataset, rather than the raw frame list,
has the requested train/test ratio.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from easydict import EasyDict


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets  # noqa: E402
from utils.util_config import cfg_from_yaml_file  # noqa: E402


DEFAULT_SEQUENCES = ("1", "58", "5", "22", "34", "9", "38", "35", "46", "19")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a causal split with a target ratio of usable samples."
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--source-train", required=True)
    parser.add_argument("--source-test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--train-ratio", type=float, default=0.6)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def temporal_key(label: str) -> tuple[int, ...]:
    stem = Path(label).stem
    try:
        return tuple(int(part) for part in stem.split("_"))
    except ValueError as error:
        raise ValueError(f"Non-numeric K-Radar label name: {label}") from error


def read_raw_universe(
    paths: Iterable[Path], sequences: tuple[str, ...]
) -> dict[str, list[str]]:
    selected = set(sequences)
    values = {sequence: set() for sequence in sequences}
    for path in paths:
        with path.open() as source:
            for line_number, line in enumerate(source, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split(",")
                if len(parts) != 2:
                    raise ValueError(f"Malformed split row {path}:{line_number}: {stripped}")
                sequence, label = parts
                if sequence in selected:
                    values[sequence].add(label)
    missing = [sequence for sequence, labels in values.items() if not labels]
    if missing:
        raise RuntimeError(f"Source manifests contain no frames for: {missing}")
    return {
        sequence: sorted(labels, key=temporal_key)
        for sequence, labels in values.items()
    }


def usable_labels_by_sequence(
    *,
    base_config: Path,
    source_train: Path,
    source_test: Path,
    data_root: Path,
    sequences: tuple[str, ...],
) -> dict[str, list[str]]:
    config = cfg_from_yaml_file(str(base_config), EasyDict())
    config.DATASET.path_data.list_dir_kradar = [str(data_root)]
    config.DATASET.path_data.split = [str(source_train), str(source_test)]
    config.DATASET.portion = list(sequences)
    dataset_name = str(config.DATASET.NAME)
    if dataset_name not in datasets.__all__:
        raise KeyError(f"Unknown dataset implementation: {dataset_name}")
    dataset = datasets.__all__[dataset_name](cfg=config, split="all")
    values = {sequence: set() for sequence in sequences}
    for item in dataset.list_dict_item:
        meta = item["meta"]
        sequence = str(meta["seq"])
        if sequence not in values:
            continue
        values[sequence].add(Path(meta["label_v1_0"]).name)
    del dataset
    return {
        sequence: sorted(labels, key=temporal_key)
        for sequence, labels in values.items()
    }


def write_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write("\n".join(lines))
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    args = parse_args()
    ratio = float(args.train_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("--train-ratio must be strictly between 0 and 1.")
    sequences = tuple(str(value) for value in args.sequences)
    if not sequences or len(set(sequences)) != len(sequences):
        raise ValueError("--sequences must be non-empty and unique.")

    base_config = resolve_path(args.base_config)
    source_train = resolve_path(args.source_train)
    source_test = resolve_path(args.source_test)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    for path in (base_config, source_train, source_test):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)

    raw = read_raw_universe((source_train, source_test), sequences)
    usable = usable_labels_by_sequence(
        base_config=base_config,
        source_train=source_train,
        source_test=source_test,
        data_root=data_root,
        sequences=sequences,
    )
    train_lines: list[str] = []
    test_lines: list[str] = []
    for sequence in sequences:
        raw_labels = raw[sequence]
        usable_labels = usable[sequence]
        usable_set = set(usable_labels)
        unknown = sorted(usable_set - set(raw_labels), key=temporal_key)
        if unknown:
            raise RuntimeError(
                f"Usable labels are absent from source manifests for seq{sequence}: "
                f"{unknown[:8]}"
            )
        cutoff = int(len(usable_labels) * ratio)
        if cutoff <= 0 or cutoff >= len(usable_labels):
            raise RuntimeError(
                f"Cannot split seq{sequence}: usable={len(usable_labels)}, ratio={ratio}"
            )
        boundary = usable_labels[cutoff - 1]
        boundary_key = temporal_key(boundary)
        raw_train = [label for label in raw_labels if temporal_key(label) <= boundary_key]
        raw_test = [label for label in raw_labels if temporal_key(label) > boundary_key]
        valid_train = sum(label in usable_set for label in raw_train)
        valid_test = sum(label in usable_set for label in raw_test)
        if valid_train != cutoff or valid_test != len(usable_labels) - cutoff:
            raise RuntimeError(f"Effective split mismatch for seq{sequence}.")
        train_lines.extend(f"{sequence},{label}" for label in raw_train)
        test_lines.extend(f"{sequence},{label}" for label in raw_test)
        print(
            f"seq{sequence}: boundary={boundary}, "
            f"raw={len(raw_train)}/{len(raw_test)}, "
            f"usable={valid_train}/{valid_test}, total_usable={len(usable_labels)}"
        )

    write_atomic(output_dir / "train.txt", train_lines)
    write_atomic(output_dir / "test.txt", test_lines)
    print(f"train_manifest={output_dir / 'train.txt'} rows={len(train_lines)}")
    print(f"test_manifest={output_dir / 'test.txt'} rows={len(test_lines)}")


if __name__ == "__main__":
    main()
