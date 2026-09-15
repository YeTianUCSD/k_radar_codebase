"""Build a three-pass stream while keeping truth hidden from the controller."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from scene_discovery.attributes import attribute_frame


def _sequence(value):
    text = str(value).strip().lower()
    if text.startswith("seq"):
        text = text[3:]
    return str(int(text))


def _permutation(values, rng, forbidden_first=None):
    values = list(values)
    if not values:
        return values
    for _ in range(100):
        result = [str(value) for value in rng.permutation(values)]
        if forbidden_first is None or result[0] != str(forbidden_first):
            return result
    raise RuntimeError("could not draw a phase order with a different first scene")


def build_three_pass_stream(
    train_index: pd.DataFrame,
    test_index: pd.DataFrame,
    support_manifest: pd.DataFrame,
    sequences: Sequence[object],
    base_sequence: object,
    seed: int,
):
    """Return frame references for bootstrap, discovery, replay, and test.

    Ground-truth columns are retained solely for the evaluator. The controller
    receives only descriptor rows extracted from ``source_position``.
    """
    sequence_values = [_sequence(value) for value in sequences]
    if len(sequence_values) != len(set(sequence_values)):
        raise ValueError("sequences must be unique")
    base = _sequence(base_sequence)
    if base not in sequence_values:
        raise ValueError("base_sequence is absent from sequences")
    support = support_manifest[
        support_manifest["source_split"].astype(str).eq("train")
        & support_manifest["role"].astype(str).eq("support_train")
    ]
    support_ids = set(support["sample_id"].astype(str))
    if len(support_ids) != len(sequence_values) * int(
        support_manifest["support_budget"].iloc[0]
    ):
        raise ValueError("support manifest count does not match sequence budget")

    prepared = {}
    for split, source in (("train", train_index), ("test", test_index)):
        values = source.reset_index(drop=True).copy()
        values["source_position"] = np.arange(len(values))
        values["sequence"] = values["sequence"].map(_sequence)
        labels = attribute_frame(values)
        for column in ("weather", "road", "lighting", "semantic_key"):
            values[f"true_{column}"] = labels[column].to_numpy()
        if split == "train":
            values = values[~values["sample_id"].astype(str).isin(support_ids)]
        prepared[split] = values

    rng = np.random.RandomState(int(seed))
    remaining = [value for value in sequence_values if value != base]
    first_order = _permutation(remaining, rng, forbidden_first=base)
    second_order = _permutation(
        sequence_values, rng, forbidden_first=first_order[-1]
    )
    third_order = _permutation(
        sequence_values, rng, forbidden_first=second_order[-1]
    )
    blocks = [("bootstrap", "train", base, "initial", False)]
    blocks.extend(
        ("first_visit_train", "train", value, "first", True)
        for value in first_order
    )
    blocks.extend(
        ("second_visit_train", "train", value, "revisit", False)
        for value in second_order
    )
    blocks.extend(
        ("third_visit_test", "test", value, "test", False)
        for value in third_order
    )

    parts = []
    for block_id, (phase, split, sequence, visit, expected_novel) in enumerate(blocks):
        source = prepared[split]
        block = source[source["sequence"].eq(sequence)].copy()
        if block.empty:
            raise ValueError(f"Seq{sequence}/{split} has no stream frames")
        order_column = "source_row_index" if "source_row_index" in block else "row_index"
        block = block.sort_values(order_column).reset_index(drop=True)
        block["phase"] = phase
        block["visit"] = visit
        block["expected_novel"] = bool(expected_novel)
        block["block_id"] = int(block_id)
        block["block_local_frame"] = np.arange(len(block))
        block["true_boundary"] = False
        if block_id > 0:
            block.loc[0, "true_boundary"] = True
        block["source_split"] = split
        parts.append(block)
    stream = pd.concat(parts, ignore_index=True)
    stream["stream_frame"] = np.arange(len(stream))
    if stream["sample_id"].astype(str).isin(support_ids).any():
        raise RuntimeError("support sample leaked into online stream")
    orders = {
        "bootstrap": [base],
        "first_visit_train": first_order,
        "second_visit_train": second_order,
        "third_visit_test": third_order,
    }
    return stream, orders, support_ids


def materialize_stream_features(
    stream: pd.DataFrame,
    arrays_by_split: Mapping[str, Mapping[str, np.ndarray]],
    modalities: Sequence[str],
):
    """Gather descriptors in stream order without exposing truth to inference."""
    output = {}
    split_values = stream["source_split"].astype(str).to_numpy()
    positions = stream["source_position"].astype(int).to_numpy()
    for modality in modalities:
        shape = next(iter(arrays_by_split.values()))[modality].shape[1:]
        values = np.empty((len(stream),) + shape, dtype=np.float32)
        for split in ("train", "test"):
            mask = split_values == split
            values[mask] = np.asarray(
                arrays_by_split[split][modality][positions[mask]], dtype=np.float32
            )
        output[modality] = values
    return output
