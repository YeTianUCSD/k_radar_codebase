from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


DESCRIPTOR_KINDS = ("mean", "mean_std", "spatial")


def descriptor_dimension(channels: int, kind: str, spatial_grid: Tuple[int, int] = (2, 2)) -> int:
    if kind == "mean":
        return channels
    if kind == "mean_std":
        return channels * 2
    if kind == "spatial":
        return channels * (2 + spatial_grid[0] * spatial_grid[1])
    raise ValueError(f"Unknown descriptor kind: {kind}")


def compute_descriptors(
    feature: np.ndarray,
    kinds: Sequence[str],
    spatial_grid: Tuple[int, int] = (2, 2),
) -> Dict[str, np.ndarray]:
    """Compute FP32 channel statistics from an NCHW feature chunk."""
    if feature.ndim != 4:
        raise ValueError(f"Expected NCHW feature tensor, got shape {feature.shape}")
    unknown = set(kinds) - set(DESCRIPTOR_KINDS)
    if unknown:
        raise ValueError(f"Unknown descriptor kinds: {sorted(unknown)}")
    values = np.asarray(feature, dtype=np.float32)
    mean = values.mean(axis=(2, 3), dtype=np.float32)
    result: Dict[str, np.ndarray] = {}
    if "mean" in kinds:
        result["mean"] = mean
    std = None
    if "mean_std" in kinds or "spatial" in kinds:
        std = values.std(axis=(2, 3), dtype=np.float32)
    if "mean_std" in kinds:
        result["mean_std"] = np.concatenate((mean, std), axis=1)
    if "spatial" in kinds:
        rows, columns = spatial_grid
        h_edges = np.linspace(0, values.shape[2], rows + 1, dtype=int)
        w_edges = np.linspace(0, values.shape[3], columns + 1, dtype=int)
        pooled = [mean, std]
        for row in range(rows):
            for column in range(columns):
                cell = values[:, :, h_edges[row] : h_edges[row + 1], w_edges[column] : w_edges[column + 1]]
                pooled.append(cell.mean(axis=(2, 3), dtype=np.float32))
        result["spatial"] = np.concatenate(pooled, axis=1)
    return result

