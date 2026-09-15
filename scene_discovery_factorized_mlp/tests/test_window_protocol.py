import numpy as np
import pandas as pd

from factorized_mlp.window_protocol import (
    aggregate_causal_endpoints, build_blocked_window_manifest, endpoint_positions,
)


def _index(split, rows_per_sequence=100):
    rows = []
    global_row = 0
    for sequence, lighting in ((1, "night"), (5, "day")):
        for local in range(rows_per_sequence):
            rows.append({
                "row_index": global_row, "sample_id": f"{split}_{sequence}_{local}",
                "sequence": sequence, "climate": "normal", "road_type": "urban",
                "capture_time": lighting,
            })
            global_row += 1
    return pd.DataFrame(rows)


def test_blocked_protocol_has_equal_support_and_disjoint_max_windows():
    manifest = build_blocked_window_manifest(
        _index("train"), _index("test", 60), 30, [1, 5, 10, 20],
        validation_fraction=0.2, gap_frames=20,
    )
    train = manifest[manifest.source_split.eq("train")]
    assert train[train.role.eq("support_train")].groupby("sequence").size().eq(30).all()
    for _, group in train.groupby("sequence"):
        local = group.sort_values("row_index").reset_index(drop=True)
        support_stop = np.flatnonzero(local.role.eq("support_train"))[-1]
        validation_start = np.flatnonzero(local.role.eq("support_validation"))[0]
        assert support_stop <= validation_start - 20


def test_all_windows_use_identical_endpoints_and_are_causal():
    train, test = _index("train"), _index("test", 60)
    manifest = build_blocked_window_manifest(train, test, 10, [1, 5, 10, 20])
    endpoints = endpoint_positions(manifest, "test", "official_test")
    features = np.arange(len(test), dtype=np.float32)[:, None]
    observed = []
    for window in (1, 5, 10, 20):
        values, index = aggregate_causal_endpoints(features, test, endpoints, window)
        observed.append(index["window_end_position"].tolist())
        first = endpoints[0]
        expected = features[first - window + 1:first + 1].mean()
        assert np.isclose(values[0, 0], expected)
        assert index.iloc[0]["sequence"] == test.iloc[first]["sequence"]
    assert all(value == observed[0] for value in observed[1:])


def test_official_test_never_enters_support_or_validation():
    manifest = build_blocked_window_manifest(
        _index("train"), _index("test", 60), 10, [1, 20]
    )
    test = manifest[manifest.source_split.eq("test")]
    assert not test.role.isin(["support_train", "support_validation"]).any()


def test_blocked_support_sets_are_nested_across_budgets():
    train, test = _index("train"), _index("test", 60)
    supports = {}
    for budget in (10, 20, 30):
        manifest = build_blocked_window_manifest(train, test, budget, [1, 5, 10, 20])
        supports[budget] = set(
            manifest.loc[manifest.role.eq("support_train"), "sample_id"]
        )
    assert supports[10] < supports[20] < supports[30]
