from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def aggregate_windows(
    features: np.ndarray,
    index: pd.DataFrame,
    window_size: int,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Average consecutive rows without allowing a window to cross a sequence."""
    if len(features) != len(index):
        raise ValueError(f"Feature/index mismatch: {len(features)} != {len(index)}")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if window_size == 1:
        result = index.copy().reset_index(drop=True)
        result["window_start_row"] = result["row_index"]
        result["window_end_row"] = result["row_index"]
        result["window_size"] = 1
        return np.asarray(features, dtype=np.float32), result
    stride = window_size if stride is None else stride
    if stride < 1:
        raise ValueError("stride must be positive")
    output: List[np.ndarray] = []
    rows = []
    sequences = index["sequence"].astype(str).to_numpy()
    for sequence in index["sequence"].astype(str).drop_duplicates():
        positions = np.flatnonzero(sequences == sequence)
        for local_start in range(0, len(positions) - window_size + 1, stride):
            selected = positions[local_start : local_start + window_size]
            if not np.array_equal(selected, np.arange(selected[0], selected[-1] + 1)):
                raise ValueError(f"Rows for sequence {sequence} are not contiguous")
            output.append(np.asarray(features[selected], dtype=np.float32).mean(axis=0))
            center = selected[len(selected) // 2]
            row = index.iloc[center].to_dict()
            row["window_start_row"] = int(index.iloc[selected[0]]["row_index"])
            row["window_end_row"] = int(index.iloc[selected[-1]]["row_index"])
            row["window_size"] = window_size
            rows.append(row)
    if not output:
        return np.empty((0, features.shape[1]), dtype=np.float32), pd.DataFrame(columns=list(index.columns))
    return np.stack(output).astype(np.float32), pd.DataFrame(rows).reset_index(drop=True)


def concatenate_scene_stream(
    features: np.ndarray,
    index: pd.DataFrame,
    scene_order: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    parts = []
    frames = []
    for block_id, sequence in enumerate(scene_order):
        mask = index["sequence"].astype(str).to_numpy() == str(sequence)
        if not mask.any():
            raise ValueError(f"Sequence {sequence} is absent from the descriptor split")
        parts.append(np.asarray(features[mask], dtype=np.float32))
        frame = index.loc[mask].copy()
        frame["stream_block"] = block_id
        frame["stream_true_sequence"] = str(sequence)
        frames.append(frame)
    combined = np.concatenate(parts, axis=0)
    combined_index = pd.concat(frames, ignore_index=True)
    combined_index["stream_row"] = np.arange(len(combined_index))
    return combined, combined_index


def aggregate_stream_windows(
    features: np.ndarray,
    index: pd.DataFrame,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Create causal windows, including windows that cross scene boundaries."""
    if len(features) != len(index):
        raise ValueError(f"Feature/index mismatch: {len(features)} != {len(index)}")
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")
    output = []
    rows = []
    for stop in range(window_size, len(features) + 1, stride):
        start = stop - window_size
        output.append(np.asarray(features[start:stop], dtype=np.float32).mean(axis=0))
        row = index.iloc[stop - 1].to_dict()
        row.update({"window_start_row": start, "window_end_row": stop - 1, "window_size": window_size})
        rows.append(row)
    if not output:
        return np.empty((0, features.shape[1]), dtype=np.float32), pd.DataFrame(columns=list(index.columns))
    return np.stack(output).astype(np.float32), pd.DataFrame(rows).reset_index(drop=True)

