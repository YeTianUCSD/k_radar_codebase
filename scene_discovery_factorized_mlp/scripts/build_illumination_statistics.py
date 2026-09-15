#!/usr/bin/env python3
"""Build row-aligned spatial Camera illumination statistics for Train/Test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factorized_mlp.illumination import (  # noqa: E402
    camera_image_path, extract_spatial_illumination, statistic_names,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/spatial_lighting_v6.yml",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def build_split(
    descriptor_bank: Path, dataset_root: Path, destination: Path,
    split: str, output_size,
):
    source_index = pd.read_csv(descriptor_bank / split / "index.csv")
    required = {"sample_id", "sequence", "camera_front_index"}
    missing = required.difference(source_index.columns)
    if missing:
        raise ValueError(f"{split} index is missing {sorted(missing)}")
    destination.mkdir(parents=True)
    temporary = destination / "illumination.npy.tmp"
    values = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32,
        shape=(len(source_index), len(statistic_names())),
    )
    started = time.monotonic()
    for position, (_, row) in enumerate(source_index.iterrows()):
        image_path = camera_image_path(dataset_root, row)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"failed to read {image_path} for {row['sample_id']}"
            )
        values[position] = extract_spatial_illumination(image, output_size)
        if (position + 1) % 250 == 0 or position + 1 == len(source_index):
            elapsed = time.monotonic() - started
            print(
                f"[{split}] {position + 1}/{len(source_index)} "
                f"({elapsed:.1f}s)", flush=True,
            )
    values.flush()
    del values
    os.replace(str(temporary), str(destination / "illumination.npy"))
    source_index[[
        "sample_id", "sequence", "camera_front_index", "row_index"
    ]].to_csv(destination / "index.csv", index=False)
    return len(source_index)


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    descriptor_bank = Path(config["descriptor_bank"]).expanduser().resolve()
    illumination = config["illumination"]
    dataset_root = Path(illumination["dataset_root"]).expanduser().resolve()
    output_root = Path(illumination["bank"]).expanduser().resolve()
    output_size = (
        int(illumination.get("image_width", 320)),
        int(illumination.get("image_height", 90)),
    )
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_root} already exists; pass --force to replace it"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    counts = {
        split: build_split(
            descriptor_bank, dataset_root, output_root / split,
            split, output_size,
        ) for split in ("train", "test")
    }
    atomic_json(output_root / "metadata.json", {
        "schema_version": "kradar-spatial-illumination/v1",
        "status": "complete",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "descriptor_bank": str(descriptor_bank),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "image_size": {"width": output_size[0], "height": output_size[1]},
        "feature_dimension": len(statistic_names()),
        "feature_names": list(statistic_names()),
        "split_counts": counts,
        "dtype": "float32",
        "alignment_key": "sample_id",
    })
    print(f"Completed: {output_root}", flush=True)


if __name__ == "__main__":
    main()
