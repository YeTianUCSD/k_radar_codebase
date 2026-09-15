"""Leakage-resistant construction and scoring of online context streams."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def normalize_sequence(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("seq"):
        text = text[3:]
    return str(int(text))


def build_disjoint_orders(
    base_order: Sequence[object],
    initial_scenes: Sequence[object],
    revisit_order: Sequence[object],
    random_orders: int,
    seed: int,
):
    """Build train-first-visit/test-revisit orders without repeated samples."""
    if random_orders < 0:
        raise ValueError("random_orders must be non-negative")
    base = [normalize_sequence(value) for value in base_order]
    initial = [normalize_sequence(value) for value in initial_scenes]
    revisit = [normalize_sequence(value) for value in revisit_order]
    if not base or len(base) != len(set(base)):
        raise ValueError("base_order must contain unique scenes")
    if not initial or len(initial) != len(set(initial)):
        raise ValueError("initial_scenes must contain unique scenes")
    if not set(initial).issubset(base):
        raise ValueError("initial_scenes must be included in base_order")
    if len(revisit) != len(set(revisit)) or not set(revisit).issubset(base):
        raise ValueError("revisit_order must be unique and contained in base_order")

    discovery = [scene for scene in base if scene not in set(initial)]

    def blocks(first_scenes, revisit_scenes):
        rows = []
        for scene in first_scenes:
            rows.append({
                "sequence": scene,
                "split": "train",
                "visit": "first",
                "expected_novel": True,
            })
        for scene in revisit_scenes:
            rows.append({
                "sequence": scene,
                "split": "test",
                "visit": "revisit",
                "expected_novel": False,
            })
        return rows

    orders = [("canonical", blocks(discovery, revisit))]
    rng = np.random.RandomState(seed)
    seen = {(tuple(discovery), tuple(revisit))}
    attempts = 0
    while len(orders) < random_orders + 1:
        attempts += 1
        first = list(rng.permutation(discovery))
        repeated = list(rng.permutation(revisit))
        key = (tuple(first), tuple(repeated))
        if key in seen:
            if attempts > 1000:
                raise RuntimeError("Could not generate enough unique stream orders")
            continue
        seen.add(key)
        orders.append((f"random_{len(orders):02d}", blocks(first, repeated)))
    return orders


def concatenate_partition_stream(
    arrays_by_split: Mapping[str, Mapping[str, np.ndarray]],
    indices_by_split: Mapping[str, pd.DataFrame],
    blocks: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Concatenate disjoint split/scene blocks and retain labels only for scoring."""
    if not blocks:
        raise ValueError("blocks cannot be empty")
    modalities = tuple(arrays_by_split["train"])
    if not modalities or set(arrays_by_split) != {"train", "test"}:
        raise ValueError("arrays_by_split must contain train and test modalities")
    if set(indices_by_split) != {"train", "test"}:
        raise ValueError("indices_by_split must contain train and test")
    output_parts = {modality: [] for modality in modalities}
    frames = []
    used_partitions = set()
    for block_id, block in enumerate(blocks):
        sequence = normalize_sequence(block["sequence"])
        split = str(block["split"])
        visit = str(block["visit"])
        if split not in arrays_by_split:
            raise ValueError(f"Unknown stream split {split!r}")
        partition_key = (split, sequence)
        if partition_key in used_partitions:
            raise ValueError(f"Partition appears twice in stream: {partition_key}")
        used_partitions.add(partition_key)
        index = indices_by_split[split]
        mask = index["sequence"].astype(str).eq(sequence).to_numpy()
        if not mask.any():
            raise ValueError(f"Seq{sequence}/{split} is absent from descriptor bank")
        for modality in modalities:
            values = np.asarray(arrays_by_split[split][modality])
            if len(values) != len(index):
                raise ValueError(f"{split}/{modality} feature-index mismatch")
            output_parts[modality].append(np.asarray(values[mask], dtype=np.float32))
        frame = index.loc[mask].copy()
        frame["stream_block"] = block_id
        frame["stream_split"] = split
        frame["stream_visit"] = visit
        frame["stream_expected_novel"] = bool(block["expected_novel"])
        frame["stream_true_sequence"] = sequence
        frames.append(frame)
    combined_index = pd.concat(frames, ignore_index=True)
    combined_index["stream_row"] = np.arange(len(combined_index))
    sample_keys = (
        combined_index["stream_split"].astype(str)
        + "|"
        + combined_index["sample_id"].astype(str)
    )
    if sample_keys.duplicated().any():
        raise ValueError("The online stream reuses descriptor samples")
    combined = {
        modality: np.concatenate(parts, axis=0).astype(np.float32)
        for modality, parts in output_parts.items()
    }
    return combined, combined_index


def summarize_online_blocks(
    frame: pd.DataFrame,
    context_ids: Sequence[int],
    is_new_context: Sequence[bool],
    mapped_labels: Sequence[object],
    initial_scene_contexts: Mapping[str, int],
):
    """Score first-visit creation and disjoint revisit reuse per scene block."""
    values = frame.copy().reset_index(drop=True)
    values["context_id"] = np.asarray(context_ids, dtype=int)
    values["is_new_context"] = np.asarray(is_new_context, dtype=bool)
    values["mapped_sequence"] = np.asarray(mapped_labels).astype(str)
    learned = {str(scene): int(context) for scene, context in initial_scene_contexts.items()}
    rows = []
    for block_id, block in values.groupby("stream_block", sort=True):
        scene = str(block["stream_true_sequence"].iloc[-1])
        visit = str(block["stream_visit"].iloc[-1])
        ids, counts = np.unique(block["context_id"], return_counts=True)
        dominant = int(ids[counts.argmax()])
        creations = np.flatnonzero(block["is_new_context"].to_numpy())
        mapped_correct = block["mapped_sequence"].eq(scene).to_numpy()
        first_correct = np.flatnonzero(mapped_correct)
        expected_context = learned.get(scene)
        committed_context = int(block["context_id"].iloc[-1])
        registry_collision = (
            visit == "first"
            and committed_context in set(learned.values())
        )
        reuse_correct = (
            bool(dominant == expected_context)
            if visit == "revisit" and expected_context is not None
            else np.nan
        )
        rows.append({
            "stream_block": int(block_id),
            "sequence": scene,
            "split": str(block["stream_split"].iloc[-1]),
            "visit": visit,
            "expected_novel": bool(block["stream_expected_novel"].iloc[-1]),
            "windows": len(block),
            "dominant_context": dominant,
            "dominant_fraction": float(counts.max() / counts.sum()),
            "creation_events": int(len(creations)),
            "duplicate_creation_events": max(0, int(len(creations)) - 1),
            "registry_collision": bool(registry_collision),
            "created_in_block": bool(len(creations)),
            "creation_latency_windows": int(creations[0]) if len(creations) else len(block),
            "mapped_accuracy": float(mapped_correct.mean()),
            "first_correct_latency_windows": (
                int(first_correct[0]) if len(first_correct) else len(block)
            ),
            "expected_reuse_context": expected_context,
            "reuse_correct": reuse_correct,
        })
        if visit == "first":
            # Early boundary windows may still carry the previous context while
            # persistence is accumulating; the committed end-of-block context
            # is the identity that should be reused later.
            learned[scene] = committed_context
    return pd.DataFrame(rows), learned
