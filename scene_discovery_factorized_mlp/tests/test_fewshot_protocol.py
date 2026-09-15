import pandas as pd

from factorized_mlp.fewshot_protocol import build_fewshot_manifest


def _index(split, rows_per_sequence):
    descriptions = {
        1: ("normal", "urban", "night"),
        5: ("normal", "urban", "day"),
    }
    rows = []
    row_index = 0
    for sequence, (weather, road, lighting) in descriptions.items():
        for local in range(rows_per_sequence):
            rows.append({
                "row_index": row_index,
                "sample_id": f"seq{sequence}_{split}_{local}",
                "sequence": sequence,
                "split": split,
                "climate": weather,
                "road_type": road,
                "capture_time": lighting,
            })
            row_index += 1
    return pd.DataFrame(rows)


def test_partitions_have_expected_sizes_and_no_test_leakage():
    manifest = build_fewshot_manifest(
        _index("train", 70), _index("test", 11), 10,
        support_pool=30, validation_frames=20, guard_frames=10,
    )
    train = manifest[manifest.source_split.eq("train")]
    counts = train.groupby(["sequence", "role"]).size().unstack(fill_value=0)
    assert counts["support_train"].eq(10).all()
    assert counts["excluded_support_pool"].eq(20).all()
    assert counts["support_validation"].eq(20).all()
    assert counts["guard"].eq(10).all()
    assert counts["train_remainder"].eq(10).all()
    assert manifest.loc[manifest.source_split.eq("test"), "role"].eq("official_test").all()


def test_support_sets_are_nested_across_budgets():
    train, test = _index("train", 70), _index("test", 11)
    manifests = {
        budget: build_fewshot_manifest(train, test, budget)
        for budget in (10, 20, 30)
    }
    supports = {
        budget: set(frame.loc[frame.role.eq("support_train"), "sample_id"])
        for budget, frame in manifests.items()
    }
    assert supports[10] < supports[20] < supports[30]


def test_manifest_preserves_original_split_positions():
    train, test = _index("train", 70), _index("test", 11)
    manifest = build_fewshot_manifest(train, test, 10)
    for split, original in (("train", train), ("test", test)):
        rows = manifest[manifest.source_split.eq(split)]
        recovered = rows.sort_values("split_position")["sample_id"].tolist()
        assert recovered == original["sample_id"].tolist()
