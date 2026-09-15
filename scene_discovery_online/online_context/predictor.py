"""Load the deployable V5 checkpoint and infer per-frame attribute probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from factorized_mlp.hybrid_model import WeatherMultimodalAttributeMLP
from factorized_mlp.spatial_lighting_model import (
    WeatherMultimodalSpatialLightingMLP,
)
from factorized_mlp.independent_model import (
    IndependentGatedSpatialAttributeMLP,
)


class AttributePredictor:
    def __init__(self, checkpoint_path, device="auto"):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.architecture = str(checkpoint["architecture"])
        allowed = {
            "weather_multimodal_gated",
            "weather_multimodal_gated_spatial_lighting",
            "independent_gated_spatial_lighting",
        }
        if self.architecture not in allowed:
            raise ValueError(f"online V1 does not support {self.architecture}")
        if int(checkpoint["window"]) != 1:
            raise ValueError("online V1 requires a feature-window-1 checkpoint")
        self.modalities = tuple(checkpoint["modalities"])
        self.sensor_modalities = tuple(
            checkpoint.get("sensor_modalities", self.modalities)
        )
        self.label_names = {
            key: tuple(value) for key, value in checkpoint["label_names"].items()
        }
        standardizer = checkpoint["standardizer"]
        self.means = {
            key: np.asarray(standardizer["means"][key], dtype=np.float32)
            for key in self.modalities
        }
        self.scales = {
            key: np.asarray(standardizer["scales"][key], dtype=np.float32)
            for key in self.modalities
        }
        input_dims = {key: len(self.means[key]) for key in self.modalities}
        class_counts = {key: len(value) for key, value in self.label_names.items()}
        config = checkpoint["model_config"]
        common = {
            "embedding_dim": int(config["embedding_dim"]),
            "head_hidden_dim": int(config["head_hidden_dim"]),
            "dropout": float(config["dropout"]),
        }
        if self.architecture == "weather_multimodal_gated":
            self.model = WeatherMultimodalAttributeMLP(
                input_dims, class_counts, modalities=self.sensor_modalities,
                fusion="gated", **common,
            )
        elif self.architecture == "weather_multimodal_gated_spatial_lighting":
            self.model = WeatherMultimodalSpatialLightingMLP(
                input_dims, class_counts, modalities=self.sensor_modalities,
                illumination_embedding_dim=int(
                    config.get("illumination_embedding_dim", 16)
                ), fusion="gated", **common,
            )
        else:
            self.model = IndependentGatedSpatialAttributeMLP(
                input_dims, class_counts,
                weather_modalities=self.sensor_modalities,
                road_modalities=tuple(config.get(
                    "road_modalities", ("camera", "lidar")
                )),
                illumination_embedding_dim=int(
                    config.get("illumination_embedding_dim", 16)
                ), **common,
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.checkpoint_metadata = {
            key: checkpoint[key] for key in (
                "architecture", "support_budget", "window", "seed", "epochs"
            )
        }

    def predict_probabilities(self, features: Mapping[str, np.ndarray], batch_size=256):
        lengths = {len(features[key]) for key in self.modalities}
        if len(lengths) != 1:
            raise ValueError("modality arrays have different lengths")
        count = lengths.pop()
        output = {key: [] for key in self.label_names}
        with torch.no_grad():
            for start in range(0, count, int(batch_size)):
                stop = min(start + int(batch_size), count)
                batch = {
                    modality: torch.from_numpy(
                        ((np.asarray(features[modality][start:stop], dtype=np.float32)
                          - self.means[modality]) / self.scales[modality]).astype(
                            np.float32, copy=False
                        )
                    ).to(self.device)
                    for modality in self.modalities
                }
                logits = self.model(batch)["logits"]
                for attribute, values in logits.items():
                    output[attribute].append(
                        torch.softmax(values, dim=1).cpu().numpy()
                    )
        return {
            key: np.concatenate(parts, axis=0) for key, parts in output.items()
        }
