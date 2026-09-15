"""Seq1-only preprocessing and distance calibration shared by online studies."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from .temporal import aggregate_windows


def chronological_masks(
    index: pd.DataFrame,
    scenes: Sequence[str],
    calibration_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = index["sequence"].astype(str).to_numpy()
    fit = np.zeros(len(index), dtype=bool)
    calibration = np.zeros(len(index), dtype=bool)
    for scene in scenes:
        positions = np.flatnonzero(sequences == str(scene))
        if len(positions) < 2:
            raise ValueError(f"Seq{scene} needs at least two initialization rows")
        cut = max(1, min(
            len(positions) - 1,
            int(round(len(positions) * (1.0 - calibration_fraction))),
        ))
        fit[positions[:cut]] = True
        calibration[positions[cut:]] = True
    return fit, calibration


def subset_windows(features, index, mask, window, stride):
    return aggregate_windows(
        np.asarray(features[mask], dtype=np.float32),
        index.loc[mask].reset_index(drop=True),
        window,
        1 if window == 1 else stride,
    )


def prepare_initial_contexts(
    features: np.ndarray,
    index: pd.DataFrame,
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
    scenes: Sequence[str],
    window: int,
    stride: int,
):
    sequences = index["sequence"].astype(str).to_numpy()
    contexts = []
    for scene in scenes:
        fit_x, _ = subset_windows(
            features, index, fit_mask & (sequences == str(scene)), window, stride
        )
        calibration_x, _ = subset_windows(
            features, index, calibration_mask & (sequences == str(scene)), window, 1
        )
        if not len(fit_x) or not len(calibration_x):
            raise ValueError(f"Seq{scene} has too few rows for window={window}")
        contexts.append({
            "scene": str(scene),
            "fit_count": len(fit_x),
            "fit_features": fit_x,
            "calibration_features": calibration_x,
        })
    return contexts


def calibrate_modality_scales(
    projected_train,
    index,
    fit_mask,
    calibration_mask,
    scenes,
    modalities,
    window,
    stride,
    quantile,
    fusion_mode,
):
    sequence_values = index["sequence"].astype(str).to_numpy()
    scales = {}
    rows = []
    for modality in modalities:
        fit_values = []
        calibration_values = []
        for scene in scenes:
            fit_x, _ = subset_windows(
                projected_train[modality], index,
                fit_mask & (sequence_values == str(scene)), window, stride,
            )
            calibration_x, _ = subset_windows(
                projected_train[modality], index,
                calibration_mask & (sequence_values == str(scene)), window, 1,
            )
            prototype = fit_x.mean(axis=0)
            fit_values.append(np.linalg.norm(fit_x - prototype[None, :], axis=1))
            calibration_values.append(
                np.linalg.norm(calibration_x - prototype[None, :], axis=1)
            )
        fit_distances = np.concatenate(fit_values)
        calibration_distances = np.concatenate(calibration_values)
        radius = float(np.quantile(calibration_distances, quantile))
        scale = max(radius, 1e-6) if fusion_mode == "seq1_calibrated_late_distance" else 1.0
        scales[modality] = scale
        rows.append({
            "modality": modality,
            "fit_distance_mean": float(fit_distances.mean()),
            "calibration_distance_mean": float(calibration_distances.mean()),
            "calibration_distance_std": float(calibration_distances.std()),
            "calibrated_radius": radius,
            "applied_scale": scale,
        })
    return scales, pd.DataFrame(rows)


def compressed_memory(values: np.ndarray, memory_size: int) -> np.ndarray:
    if len(values) <= memory_size:
        return values
    positions = np.linspace(0, len(values) - 1, memory_size).round().astype(int)
    return values[positions]


def memory_calibration_distances(contexts, memory_size: int, neighbors: int):
    distances = []
    for context in contexts:
        memory = compressed_memory(context["fit_features"], memory_size)
        calibration = context["calibration_features"]
        pairwise = np.linalg.norm(
            calibration[:, None, :] - memory[None, :, :], axis=2
        )
        count = min(neighbors, memory.shape[0])
        distances.append(np.sort(pairwise, axis=1)[:, :count].mean(axis=1))
    return np.concatenate(distances)
