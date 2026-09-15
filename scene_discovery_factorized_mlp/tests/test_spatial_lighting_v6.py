from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from factorized_mlp.illumination import (
    extract_spatial_illumination, load_aligned_illumination_bank,
    statistic_names,
)
from factorized_mlp.spatial_lighting_model import (
    WeatherMultimodalSpatialLightingMLP,
)


def _model():
    return WeatherMultimodalSpatialLightingMLP(
        {"camera": 7, "lidar": 5, "radar": 9, "illumination": 48},
        {"weather": 6, "road": 6, "lighting": 2},
        embedding_dim=4, illumination_embedding_dim=3,
        head_hidden_dim=8, dropout=0.0, fusion="gated",
    )


def _inputs():
    return {
        "camera": torch.randn(3, 7), "lidar": torch.randn(3, 5),
        "radar": torch.randn(3, 9), "illumination": torch.randn(3, 48),
    }


def test_spatial_statistics_are_finite_and_retain_vertical_layout():
    dark_sky_bright_road = np.zeros((90, 120, 3), dtype=np.uint8)
    dark_sky_bright_road[60:] = 220
    bright_sky_dark_road = np.zeros((90, 120, 3), dtype=np.uint8)
    bright_sky_dark_road[:30] = 220
    first = extract_spatial_illumination(dark_sky_bright_road)
    second = extract_spatial_illumination(bright_sky_dark_road)
    names = statistic_names()
    assert first.shape == (48,)
    assert np.isfinite(first).all()
    assert first[names.index("top_luma_mean")] < first[
        names.index("bottom_luma_mean")
    ]
    assert second[names.index("top_luma_mean")] > second[
        names.index("bottom_luma_mean")
    ]


def test_road_and_lighting_projectors_have_isolated_gradients():
    model = _model()
    inputs = _inputs()
    model(inputs)["logits"]["road"].sum().backward()
    assert next(model.road_camera_projector.parameters()).grad is not None
    assert next(model.lighting_camera_projector.parameters()).grad is None
    assert next(model.lighting_illumination_projector.parameters()).grad is None
    for projector in model.weather_projectors.values():
        assert next(projector.parameters()).grad is None

    model.zero_grad(set_to_none=True)
    model(inputs)["logits"]["lighting"].sum().backward()
    assert next(model.road_camera_projector.parameters()).grad is None
    assert next(model.lighting_camera_projector.parameters()).grad is not None
    assert next(model.lighting_illumination_projector.parameters()).grad is not None
    for projector in model.weather_projectors.values():
        assert next(projector.parameters()).grad is None


def test_illumination_input_only_changes_lighting_logits():
    model = _model().eval()
    inputs = _inputs()
    changed = dict(inputs)
    changed["illumination"] = inputs["illumination"] + 100.0
    first = model(inputs)["logits"]
    second = model(changed)["logits"]
    assert torch.allclose(first["weather"], second["weather"])
    assert torch.allclose(first["road"], second["road"])
    assert not torch.allclose(first["lighting"], second["lighting"])


def test_aligned_bank_rejects_reordered_sample_ids():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        split = root / "train"
        split.mkdir()
        np.save(split / "illumination.npy", np.zeros((2, 48), dtype=np.float32))
        pd.DataFrame({"sample_id": ["a", "b"]}).to_csv(
            split / "index.csv", index=False
        )
        expected = pd.DataFrame({"sample_id": ["b", "a"]})
        try:
            load_aligned_illumination_bank(root, "train", expected)
        except ValueError as error:
            assert "sample_id order mismatch" in str(error)
        else:
            raise AssertionError("reordered sample IDs were silently accepted")
