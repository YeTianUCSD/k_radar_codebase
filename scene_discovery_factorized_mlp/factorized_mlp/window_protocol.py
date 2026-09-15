"""Blocked temporal protocol with paired causal-window endpoints."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from scene_discovery.attributes import attribute_frame
from scene_discovery.online_protocol import normalize_sequence

from .fewshot_protocol import _nested_positions


def build_blocked_window_manifest(
    train_index: pd.DataFrame,
    test_index: pd.DataFrame,
    support_budget: int,
    window_sizes: Iterable[int],
    *,
    validation_fraction: float = 0.2,
    gap_frames: Optional[int] = None,
) -> pd.DataFrame:
    """Create common endpoints for every window without temporal overlap.

    Support endpoints are spread over the early portion of each train sequence.
    Validation is the final contiguous fraction.  The gap is large enough that a
    maximum-size support window and validation window never share raw frames.
    """
    windows = tuple(sorted(set(int(value) for value in window_sizes)))
    if not windows or windows[0] < 1:
        raise ValueError("window sizes must be positive")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    maximum_window = windows[-1]
    gap_frames = maximum_window if gap_frames is None else int(gap_frames)
    if gap_frames < maximum_window:
        raise ValueError("gap_frames must be at least the maximum window size")
    required = {"sequence", "sample_id", "row_index"}
    for name, frame in (("train", train_index), ("test", test_index)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} index is missing columns: {sorted(missing)}")

    outputs = []
    for source_split, source in (("train", train_index), ("test", test_index)):
        source = source.reset_index(drop=True).copy()
        source["split_position"] = np.arange(len(source))
        for sequence, group in source.groupby("sequence", sort=False):
            values = group.sort_values("row_index").copy()
            count = len(values)
            roles = np.full(count, "excluded", dtype=object)
            endpoint = np.zeros(count, dtype=bool)
            if source_split == "train":
                validation_start = int(np.floor(count * (1.0 - validation_fraction)))
                validation_start = max(maximum_window, min(count - 1, validation_start))
                support_stop = validation_start - gap_frames
                candidates = np.arange(maximum_window - 1, support_stop + 1)
                if len(candidates) < support_budget:
                    raise ValueError(
                        f"Seq{sequence} has only {len(candidates)} valid support endpoints "
                        f"for budget={support_budget}, max_window={maximum_window}, "
                        f"validation_fraction={validation_fraction}, gap={gap_frames}"
                    )
                chosen = candidates[_nested_positions(len(candidates), support_budget)]
                roles[:maximum_window - 1] = "warmup"
                roles[candidates] = "support_candidate"
                roles[chosen] = "support_train"
                roles[support_stop + 1:validation_start] = "temporal_gap"
                roles[validation_start:] = "support_validation"
                endpoint[chosen] = True
                endpoint[validation_start:] = True
            else:
                roles[:maximum_window - 1] = "warmup"
                roles[maximum_window - 1:] = "official_test"
                endpoint[maximum_window - 1:] = True
            values["role"] = roles
            values["is_common_endpoint"] = endpoint
            values["source_split"] = source_split
            values["sequence"] = values["sequence"].map(normalize_sequence)
            outputs.append(values)

    manifest = pd.concat(outputs, ignore_index=True)
    attributes = attribute_frame(manifest)
    for column in ("weather", "road", "lighting", "semantic_key"):
        manifest[column] = attributes[column]
    manifest["support_budget"] = int(support_budget)
    manifest["maximum_window"] = maximum_window
    manifest["validation_fraction"] = float(validation_fraction)
    manifest["gap_frames"] = gap_frames
    validate_blocked_manifest(manifest, support_budget, maximum_window)
    return manifest


def validate_blocked_manifest(manifest, support_budget, maximum_window):
    train = manifest[manifest.source_split.eq("train")]
    counts = train[train.role.eq("support_train")].groupby("sequence").size()
    if counts.empty or not counts.eq(int(support_budget)).all():
        raise ValueError("every sequence must have the requested support budget")
    if not train.loc[train.role.eq("support_train"), "is_common_endpoint"].all():
        raise ValueError("all support rows must be common endpoints")
    test = manifest[manifest.source_split.eq("test")]
    if not test.loc[test.role.eq("official_test"), "is_common_endpoint"].all():
        raise ValueError("all official test rows must be common endpoints")
    for sequence, group in train.groupby("sequence", sort=False):
        local = group.sort_values("row_index").reset_index(drop=True)
        support_end = np.flatnonzero(local.role.eq("support_train"))[-1]
        validation_end = np.flatnonzero(local.role.eq("support_validation"))[0]
        if support_end > validation_end - maximum_window:
            raise ValueError(f"Seq{sequence} support and validation windows overlap")


def endpoint_positions(manifest: pd.DataFrame, split: str, role: str) -> np.ndarray:
    rows = manifest[
        manifest.source_split.eq(split)
        & manifest.role.eq(role)
        & manifest.is_common_endpoint
    ]
    return rows.split_position.astype(int).to_numpy()


def aggregate_causal_endpoints(
    features: np.ndarray,
    index: pd.DataFrame,
    endpoints: Iterable[int],
    window_size: int,
):
    """Average windows ending at fixed split positions, never crossing a sequence."""
    values = np.asarray(features, dtype=np.float32)
    index = index.reset_index(drop=True)
    endpoints = np.asarray(tuple(endpoints), dtype=int)
    output, rows = [], []
    sequence_values = index["sequence"].map(normalize_sequence).to_numpy()
    for stop in endpoints:
        start = int(stop) - int(window_size) + 1
        if start < 0:
            raise ValueError(f"endpoint {stop} lacks {window_size} preceding frames")
        if not np.all(sequence_values[start:stop + 1] == sequence_values[stop]):
            raise ValueError(f"window ending at {stop} crosses a sequence boundary")
        output.append(values[start:stop + 1].mean(axis=0))
        row = index.iloc[stop].to_dict()
        row.update({
            "split_position": int(stop),
            "window_start_position": start,
            "window_end_position": int(stop),
            "window_size": int(window_size),
        })
        rows.append(row)
    if not output:
        return np.empty((0, values.shape[1]), dtype=np.float32), pd.DataFrame()
    return np.stack(output).astype(np.float32), pd.DataFrame(rows)
