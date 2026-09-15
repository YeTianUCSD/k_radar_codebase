"""Leakage-resistant few-shot partitions built from descriptor indices."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from scene_discovery.attributes import attribute_frame
from scene_discovery.online_protocol import normalize_sequence


ROLES = (
    "support_train", "support_validation", "guard", "train_remainder",
    "official_test",
)


def _nested_positions(pool_size: int, budget: int) -> np.ndarray:
    if not 1 <= budget <= pool_size:
        raise ValueError(f"support budget must be in [1, {pool_size}]")
    # A deterministic nested order spreads early additions across the pool.
    order = []
    remaining = set(range(pool_size))
    while remaining:
        if not order:
            chosen = 0
        else:
            chosen = max(
                remaining,
                key=lambda value: (
                    min(abs(value - selected) for selected in order), -value
                ),
            )
        order.append(chosen)
        remaining.remove(chosen)
    return np.asarray(sorted(order[:budget]), dtype=int)


def build_fewshot_manifest(
    train_index: pd.DataFrame,
    test_index: pd.DataFrame,
    support_budget: int,
    support_pool: int = 30,
    validation_frames: int = 20,
    guard_frames: int = 10,
) -> pd.DataFrame:
    """Assign each descriptor row a role without reusing official test data."""
    required = {"sequence", "sample_id", "row_index"}
    for name, frame in (("train", train_index), ("test", test_index)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} index is missing columns: {sorted(missing)}")
    if min(support_pool, validation_frames) < 1 or guard_frames < 0:
        raise ValueError("invalid few-shot partition sizes")

    train_index = train_index.reset_index(drop=True).copy()
    train_index["split_position"] = np.arange(len(train_index))
    test_index = test_index.reset_index(drop=True).copy()
    test_index["split_position"] = np.arange(len(test_index))
    frames = []
    selected_local = set(_nested_positions(support_pool, support_budget).tolist())
    for sequence, group in train_index.groupby("sequence", sort=False):
        values = group.sort_values("row_index").copy()
        required_rows = support_pool + validation_frames + guard_frames + 1
        if len(values) < required_rows:
            raise ValueError(
                f"Seq{sequence} has {len(values)} train rows; needs at least {required_rows}"
            )
        roles = np.full(len(values), "excluded_support_pool", dtype=object)
        for position in selected_local:
            roles[position] = "support_train"
        validation_stop = support_pool + validation_frames
        guard_stop = validation_stop + guard_frames
        roles[support_pool:validation_stop] = "support_validation"
        roles[validation_stop:guard_stop] = "guard"
        roles[guard_stop:] = "train_remainder"
        values["role"] = roles
        values["source_split"] = "train"
        values["sequence"] = values["sequence"].map(normalize_sequence)
        frames.append(values)

    test = test_index.sort_values(["sequence", "row_index"]).copy()
    test["role"] = "official_test"
    test["source_split"] = "test"
    test["sequence"] = test["sequence"].map(normalize_sequence)
    frames.append(test)
    manifest = pd.concat(frames, ignore_index=True)
    attributes = attribute_frame(manifest)
    for column in ("weather", "road", "lighting", "semantic_key"):
        manifest[column] = attributes[column]
    manifest["support_budget"] = int(support_budget)
    manifest["manifest_row"] = np.arange(len(manifest))
    validate_manifest(manifest, support_budget)
    return manifest


def validate_manifest(manifest: pd.DataFrame, support_budget: int) -> None:
    if manifest[["source_split", "sample_id"]].duplicated().any():
        raise ValueError("manifest contains duplicate split/sample pairs")
    support = manifest[manifest["role"].eq("support_train")]
    counts = support.groupby("sequence").size()
    if counts.empty or not counts.eq(support_budget).all():
        raise ValueError("every sequence must have the requested support budget")
    if not manifest.loc[manifest.source_split.eq("test"), "role"].eq("official_test").all():
        raise ValueError("official test rows cannot enter training roles")
    overlap = set(support.manifest_row) & set(
        manifest.loc[manifest.role.eq("train_remainder"), "manifest_row"]
    )
    if overlap:
        raise ValueError("support and train remainder overlap")


def role_mask(manifest: pd.DataFrame, roles: Iterable[str]) -> np.ndarray:
    return manifest["role"].isin(tuple(roles)).to_numpy()
