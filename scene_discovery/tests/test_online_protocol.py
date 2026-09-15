import numpy as np
import pandas as pd

from scene_discovery.online_protocol import (
    build_disjoint_orders,
    concatenate_partition_stream,
    summarize_online_blocks,
)


def test_disjoint_order_uses_train_for_discovery_and_test_for_revisit():
    orders = build_disjoint_orders([1, 5, 9], [1], [1, 5, 9], 0, 7)
    _, blocks = orders[0]
    assert [(row["sequence"], row["split"], row["visit"]) for row in blocks] == [
        ("5", "train", "first"),
        ("9", "train", "first"),
        ("1", "test", "revisit"),
        ("5", "test", "revisit"),
        ("9", "test", "revisit"),
    ]


def test_partition_stream_never_reuses_a_split_scene():
    index = {
        "train": pd.DataFrame({
            "sequence": ["5", "5", "9"],
            "sample_id": ["a", "b", "c"],
            "row_index": [0, 1, 2],
        }),
        "test": pd.DataFrame({
            "sequence": ["5", "9", "9"],
            "sample_id": ["d", "e", "f"],
            "row_index": [0, 1, 2],
        }),
    }
    arrays = {
        split: {"camera": np.arange(6, dtype=np.float32).reshape(3, 2)}
        for split in ("train", "test")
    }
    blocks = [
        {"sequence": "5", "split": "train", "visit": "first", "expected_novel": True},
        {"sequence": "5", "split": "test", "visit": "revisit", "expected_novel": False},
    ]
    combined, frame = concatenate_partition_stream(arrays, index, blocks)
    assert combined["camera"].shape == (3, 2)
    assert frame.stream_split.tolist() == ["train", "train", "test"]
    assert frame.stream_visit.tolist() == ["first", "first", "revisit"]


def test_block_summary_measures_creation_and_reuse_separately():
    frame = pd.DataFrame({
        "stream_block": [0, 0, 1, 1, 2, 2],
        "stream_true_sequence": ["5", "5", "1", "1", "5", "5"],
        "stream_split": ["train", "train", "test", "test", "test", "test"],
        "stream_visit": ["first", "first", "revisit", "revisit", "revisit", "revisit"],
        "stream_expected_novel": [True, True, False, False, False, False],
    })
    blocks, learned = summarize_online_blocks(
        frame,
        context_ids=[0, 2, 0, 0, 2, 2],
        is_new_context=[False, True, False, False, False, False],
        mapped_labels=["1", "5", "1", "1", "5", "5"],
        initial_scene_contexts={"1": 0},
    )
    assert blocks.loc[0, "created_in_block"]
    assert blocks.loc[1, "reuse_correct"]
    assert blocks.loc[2, "reuse_correct"]
    assert learned == {"1": 0, "5": 2}
