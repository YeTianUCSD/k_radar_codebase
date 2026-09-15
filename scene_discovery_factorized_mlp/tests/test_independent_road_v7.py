import unittest

import numpy as np
import torch

from factorized_mlp.independent_model import (
    IndependentGatedSpatialAttributeMLP,
)
from factorized_mlp.training import class_weights


class IndependentRoadModelTests(unittest.TestCase):
    @staticmethod
    def model():
        return IndependentGatedSpatialAttributeMLP(
            {"camera": 8, "lidar": 6, "radar": 4, "illumination": 12},
            {"weather": 6, "road": 6, "lighting": 2},
            embedding_dim=5,
            illumination_embedding_dim=3,
            head_hidden_dim=7,
            dropout=0.0,
        )

    @staticmethod
    def inputs():
        return {
            "camera": torch.randn(3, 8),
            "lidar": torch.randn(3, 6),
            "radar": torch.randn(3, 4),
            "illumination": torch.randn(3, 12),
        }

    def test_attribute_paths_use_requested_modalities(self):
        model = self.model().eval()
        inputs = self.inputs()
        first = model(inputs)["logits"]

        radar_changed = dict(inputs)
        radar_changed["radar"] = inputs["radar"] + 10.0
        second = model(radar_changed)["logits"]
        self.assertFalse(torch.allclose(first["weather"], second["weather"]))
        self.assertTrue(torch.allclose(first["road"], second["road"]))
        self.assertTrue(torch.allclose(first["lighting"], second["lighting"]))

        lidar_changed = dict(inputs)
        lidar_changed["lidar"] = inputs["lidar"] + 10.0
        third = model(lidar_changed)["logits"]
        self.assertFalse(torch.allclose(first["weather"], third["weather"]))
        self.assertFalse(torch.allclose(first["road"], third["road"]))
        self.assertTrue(torch.allclose(first["lighting"], third["lighting"]))

    def test_road_gradient_is_isolated(self):
        model = self.model()
        model(self.inputs())["logits"]["road"].sum().backward()
        for projector in model.road_projectors.values():
            self.assertIsNotNone(next(projector.parameters()).grad)
        for projector in model.weather_projectors.values():
            self.assertIsNone(next(projector.parameters()).grad)
        self.assertIsNone(
            next(model.lighting_camera_projector.parameters()).grad
        )
        self.assertIsNone(
            next(model.lighting_illumination_projector.parameters()).grad
        )

    def test_road_gate_contains_camera_and_lidar_only(self):
        model = self.model()
        self.assertEqual(model.gate_modalities()["road"], ("camera", "lidar"))
        self.assertEqual(len(model.gate_weights()["road"]), 2)

    def test_sqrt_inverse_frequency_is_milder(self):
        labels = np.asarray([0, 0, 0, 0, 1])
        full = class_weights(labels, 2, exponent=1.0).numpy()
        mild = class_weights(labels, 2, exponent=0.5).numpy()
        self.assertAlmostEqual(full[1] / full[0], 4.0, places=5)
        self.assertAlmostEqual(mild[1] / mild[0], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
