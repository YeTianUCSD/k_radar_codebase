"""Compact spatial illumination statistics aligned with descriptor-bank rows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import pandas as pd


LUMA_STATISTICS = (
    "mean", "std", "p10", "p25", "p50", "p75", "p90",
    "dark_ratio", "bright_ratio",
)
REGIONS = ("global", "top", "middle", "bottom")
COLOR_CHANNELS = ("r", "g", "b", "h", "s", "v")


def statistic_names() -> Tuple[str, ...]:
    names = [
        f"{region}_luma_{statistic}"
        for region in REGIONS
        for statistic in LUMA_STATISTICS
    ]
    names.extend(
        f"global_{channel}_{statistic}"
        for channel in COLOR_CHANNELS
        for statistic in ("mean", "std")
    )
    return tuple(names)


def _luma_statistics(values: np.ndarray) -> Iterable[float]:
    values = np.asarray(values, dtype=np.float32)
    percentiles = np.percentile(values, (10, 25, 50, 75, 90))
    return (
        float(values.mean()), float(values.std()),
        *(float(value) for value in percentiles),
        float(np.mean(values <= 32.0)),
        float(np.mean(values >= 192.0)),
    )


def extract_spatial_illumination(
    image: np.ndarray, output_size=(320, 90)
) -> np.ndarray:
    """Return global/color and vertically localized illumination statistics.

    The vertical regions retain the distinction between a dark sky and a road
    brightly illuminated by street lamps, which a single global mean loses.
    Ratios are represented on [0, 1]; all values are standardized later using
    support-training rows only.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a non-empty BGR image")
    width, height = (int(output_size[0]), int(output_size[1]))
    if width < 1 or height < 3:
        raise ValueError("output_size must have positive width and height >= 3")
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    luma = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    boundaries = np.linspace(0, height, 4, dtype=int)
    region_values = (
        luma,
        luma[boundaries[0]:boundaries[1]],
        luma[boundaries[1]:boundaries[2]],
        luma[boundaries[2]:boundaries[3]],
    )
    features = []
    for values in region_values:
        features.extend(_luma_statistics(values))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    color = np.concatenate((rgb, hsv), axis=2)
    for channel in range(color.shape[2]):
        values = color[:, :, channel]
        features.extend((float(values.mean()), float(values.std())))

    output = np.asarray(features, dtype=np.float32)
    if output.shape != (len(statistic_names()),) or not np.isfinite(output).all():
        raise RuntimeError("invalid illumination-statistics output")
    return output


def camera_image_path(dataset_root: Path, row: pd.Series) -> Path:
    sequence = str(int(row["sequence"]))
    camera_index = int(row["camera_front_index"])
    return (
        Path(dataset_root) / sequence / "cam-front" /
        f"cam-front_{camera_index:05d}.png"
    )


def load_aligned_illumination_bank(
    bank_root: Path, split: str, expected_index: pd.DataFrame
) -> np.ndarray:
    """Load an illumination array and reject silent row-order mismatches."""
    root = Path(bank_root) / str(split)
    stored_index = pd.read_csv(root / "index.csv")
    values = np.load(root / "illumination.npy", mmap_mode="r")
    if len(values) != len(stored_index) or len(values) != len(expected_index):
        raise ValueError(f"{split} illumination feature-index length mismatch")
    expected_ids = expected_index["sample_id"].astype(str).to_numpy()
    stored_ids = stored_index["sample_id"].astype(str).to_numpy()
    if not np.array_equal(expected_ids, stored_ids):
        raise ValueError(f"{split} illumination sample_id order mismatch")
    if values.ndim != 2 or values.shape[1] != len(statistic_names()):
        raise ValueError(
            f"{split} illumination has shape {values.shape}, expected "
            f"(*, {len(statistic_names())})"
        )
    return values
