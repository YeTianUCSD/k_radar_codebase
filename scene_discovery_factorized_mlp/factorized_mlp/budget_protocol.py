"""Total-label-budget protocol for sample-efficiency experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from scene_discovery.attributes import attribute_frame
from scene_discovery.online_protocol import normalize_sequence

from .fewshot_protocol import _nested_positions


def build_budget_manifest(
    train_index: pd.DataFrame,
    test_index: pd.DataFrame,
    support_budget: int,
    window_sizes: Iterable[int],
):
    """Select nested support endpoints with no separately labelled validation set."""
    windows = tuple(sorted(set(int(value) for value in window_sizes)))
    if not windows or windows[0] < 1:
        raise ValueError("window sizes must be positive")
    maximum_window = windows[-1]
    required = {"sequence", "sample_id", "row_index"}
    outputs = []
    for source_split, source in (("train", train_index), ("test", test_index)):
        missing = required.difference(source.columns)
        if missing:
            raise ValueError(f"{source_split} index missing {sorted(missing)}")
        source = source.reset_index(drop=True).copy()
        source["split_position"] = np.arange(len(source))
        for sequence, group in source.groupby("sequence", sort=False):
            values = group.sort_values("row_index").copy()
            count = len(values)
            candidates = np.arange(maximum_window - 1, count)
            if len(candidates) < support_budget and source_split == "train":
                raise ValueError(
                    f"Seq{sequence} has {len(candidates)} valid endpoints, "
                    f"less than budget {support_budget}"
                )
            roles = np.full(count, "unused", dtype=object)
            common = np.zeros(count, dtype=bool)
            roles[:maximum_window - 1] = "warmup"
            if source_split == "train":
                selected = candidates[
                    _nested_positions(len(candidates), int(support_budget))
                ]
                roles[candidates] = "support_candidate"
                roles[selected] = "support_train"
                common[selected] = True
            else:
                roles[candidates] = "official_test"
                common[candidates] = True
            values["role"] = roles
            values["is_common_endpoint"] = common
            values["source_split"] = source_split
            values["sequence"] = values["sequence"].map(normalize_sequence)
            outputs.append(values)
    manifest = pd.concat(outputs, ignore_index=True)
    attributes = attribute_frame(manifest)
    for column in ("weather", "road", "lighting", "semantic_key"):
        manifest[column] = attributes[column]
    manifest["support_budget"] = int(support_budget)
    manifest["maximum_window"] = maximum_window
    support = manifest[
        manifest.source_split.eq("train") & manifest.role.eq("support_train")
    ]
    counts = support.groupby("sequence").size()
    if counts.empty or not counts.eq(int(support_budget)).all():
        raise ValueError("incorrect support count")
    if manifest[
        manifest.source_split.eq("test")
    ].role.isin(["support_train", "support_validation"]).any():
        raise ValueError("test data leaked into support")
    return manifest


def endpoint_positions(manifest, split, role):
    rows = manifest[
        manifest.source_split.eq(split)
        & manifest.role.eq(role)
        & manifest.is_common_endpoint
    ]
    return rows.split_position.astype(int).to_numpy()


def unique_window_frame_count(endpoints, window_size):
    used = set()
    for endpoint in np.asarray(endpoints, dtype=int):
        used.update(range(int(endpoint) - int(window_size) + 1, int(endpoint) + 1))
    return len(used)
