import numpy as np
import pandas as pd
import torch

from factorized_mlp.budget_protocol import (
    build_budget_manifest, endpoint_positions, unique_window_frame_count,
)
from factorized_mlp.hybrid_model import (
    HybridAttributeMLP, WeatherMultimodalAttributeMLP,
)


def _index(split, rows_per_sequence=80):
    rows = []
    row_index = 0
    for sequence, lighting in ((1, "night"), (5, "day")):
        for local in range(rows_per_sequence):
            rows.append({
                "row_index": row_index,
                "sample_id": f"{split}_{sequence}_{local}",
                "sequence": sequence,
                "climate": "normal",
                "road_type": "urban",
                "capture_time": lighting,
            })
            row_index += 1
    return pd.DataFrame(rows)


def test_v4_budget_is_total_label_budget_without_validation_labels():
    manifest = build_budget_manifest(
        _index("train"), _index("test"), 5, [1, 3, 5, 10, 20]
    )
    train = manifest[manifest.source_split.eq("train")]
    assert train[train.role.eq("support_train")].groupby("sequence").size().eq(5).all()
    assert not manifest.role.eq("support_validation").any()
    assert not manifest.loc[
        manifest.source_split.eq("test"), "role"
    ].eq("support_train").any()


def test_v4_support_budgets_are_nested():
    train, test = _index("train"), _index("test")
    supports = {}
    for budget in (3, 5, 10, 20, 30):
        manifest = build_budget_manifest(train, test, budget, [1, 3, 5, 10, 20])
        supports[budget] = set(
            manifest.loc[manifest.role.eq("support_train"), "sample_id"]
        )
    assert supports[3] < supports[5] < supports[10] < supports[20] < supports[30]


def test_unique_frame_count_handles_overlapping_windows():
    assert unique_window_frame_count([4, 5], 5) == 6
    assert unique_window_frame_count([4, 10], 5) == 10


def test_hybrid_lighting_is_independent_of_lidar_and_radar():
    model = HybridAttributeMLP(
        {"camera": 7, "lidar": 5, "radar": 9},
        {"weather": 6, "road": 6, "lighting": 2},
        embedding_dim=4, head_hidden_dim=8, dropout=0.0, fusion="gated",
    ).eval()
    inputs = {
        "camera": torch.randn(3, 7),
        "lidar": torch.randn(3, 5),
        "radar": torch.randn(3, 9),
    }
    changed = {
        "camera": inputs["camera"],
        "lidar": inputs["lidar"] + 100.0,
        "radar": inputs["radar"] - 100.0,
    }
    first = model(inputs)["logits"]
    second = model(changed)["logits"]
    assert torch.allclose(first["lighting"], second["lighting"])
    for weights in model.gate_weights().values():
        assert torch.allclose(weights.sum(), torch.tensor(1.0))


def test_lighting_gradient_only_reaches_dedicated_camera_path():
    model = HybridAttributeMLP(
        {"camera": 7, "lidar": 5, "radar": 9},
        {"weather": 6, "road": 6, "lighting": 2},
        embedding_dim=4, head_hidden_dim=8, dropout=0.0, fusion="gated",
    )
    inputs = {
        "camera": torch.randn(3, 7),
        "lidar": torch.randn(3, 5),
        "radar": torch.randn(3, 9),
    }
    model(inputs)["logits"]["lighting"].sum().backward()
    assert next(model.lighting_camera_projector.parameters()).grad is not None
    assert next(model.multimodal_projectors["camera"].parameters()).grad is None


def test_v5_road_and_lighting_ignore_lidar_and_radar():
    model = WeatherMultimodalAttributeMLP(
        {"camera": 7, "lidar": 5, "radar": 9},
        {"weather": 6, "road": 6, "lighting": 2},
        embedding_dim=4, head_hidden_dim=8, dropout=0.0, fusion="gated",
    ).eval()
    inputs = {
        "camera": torch.randn(3, 7), "lidar": torch.randn(3, 5),
        "radar": torch.randn(3, 9),
    }
    changed = {
        "camera": inputs["camera"], "lidar": inputs["lidar"] + 100.0,
        "radar": inputs["radar"] - 100.0,
    }
    first, second = model(inputs)["logits"], model(changed)["logits"]
    assert torch.allclose(first["road"], second["road"])
    assert torch.allclose(first["lighting"], second["lighting"])


def test_v5_camera_context_gradients_do_not_reach_weather_projectors():
    model = WeatherMultimodalAttributeMLP(
        {"camera": 7, "lidar": 5, "radar": 9},
        {"weather": 6, "road": 6, "lighting": 2},
        embedding_dim=4, head_hidden_dim=8, dropout=0.0, fusion="gated",
    )
    inputs = {
        "camera": torch.randn(3, 7), "lidar": torch.randn(3, 5),
        "radar": torch.randn(3, 9),
    }
    logits = model(inputs)["logits"]
    (logits["road"].sum() + logits["lighting"].sum()).backward()
    assert next(model.camera_context_projector.parameters()).grad is not None
    for projector in model.weather_projectors.values():
        assert next(projector.parameters()).grad is None
    assert set(model.gate_weights()) == {"weather"}
