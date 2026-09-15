"""Independent attribute paths for weather, road, and spatial lighting."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .model import ModalityProjector


class IndependentGatedSpatialAttributeMLP(nn.Module):
    """Three isolated heads with attribute-specific sensor inputs."""

    attributes = ("weather", "road", "lighting")

    def __init__(
        self,
        input_dims: Mapping[str, int],
        class_counts: Mapping[str, int],
        weather_modalities: Sequence[str] = ("camera", "lidar", "radar"),
        road_modalities: Sequence[str] = ("camera", "lidar"),
        embedding_dim: int = 32,
        illumination_embedding_dim: int = 16,
        head_hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.weather_modalities = tuple(weather_modalities)
        self.road_modalities = tuple(road_modalities)
        if not self.weather_modalities:
            raise ValueError("weather requires at least one modality")
        if not self.road_modalities:
            raise ValueError("road requires at least one modality")
        required = (
            set(self.weather_modalities)
            | set(self.road_modalities)
            | {"camera", "illumination"}
        )
        missing = required.difference(input_dims)
        if missing:
            raise ValueError(f"missing input dimensions: {sorted(missing)}")
        if set(class_counts) != set(self.attributes):
            raise ValueError(
                "class_counts must contain weather, road, and lighting"
            )

        self.weather_projectors = nn.ModuleDict({
            modality: ModalityProjector(
                input_dims[modality], embedding_dim, dropout
            )
            for modality in self.weather_modalities
        })
        self.road_projectors = nn.ModuleDict({
            modality: ModalityProjector(
                input_dims[modality], embedding_dim, dropout
            )
            for modality in self.road_modalities
        })
        self.lighting_camera_projector = ModalityProjector(
            input_dims["camera"], embedding_dim, dropout
        )
        self.lighting_illumination_projector = ModalityProjector(
            input_dims["illumination"], illumination_embedding_dim, dropout
        )

        self.weather_gate_logits = nn.Parameter(
            torch.zeros(len(self.weather_modalities))
        )
        self.road_gate_logits = nn.Parameter(
            torch.zeros(len(self.road_modalities))
        )
        head_inputs = {
            "weather": embedding_dim,
            "road": embedding_dim,
            "lighting": embedding_dim + illumination_embedding_dim,
        }
        self.heads = nn.ModuleDict({
            attribute: nn.Sequential(
                nn.Linear(head_inputs[attribute], head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    head_hidden_dim, int(class_counts[attribute])
                ),
            )
            for attribute in self.attributes
        })

    def gate_modalities(self):
        return {
            "weather": self.weather_modalities,
            "road": self.road_modalities,
        }

    def gate_weights(self):
        return {
            "weather": torch.softmax(self.weather_gate_logits, dim=0),
            "road": torch.softmax(self.road_gate_logits, dim=0),
        }

    @staticmethod
    def _fuse(inputs, projectors, modalities, weights):
        return sum(
            weights[position] * projectors[modality](inputs[modality])
            for position, modality in enumerate(modalities)
        )

    def forward(self, inputs):
        gates = self.gate_weights()
        weather = self._fuse(
            inputs, self.weather_projectors, self.weather_modalities,
            gates["weather"],
        )
        road = self._fuse(
            inputs, self.road_projectors, self.road_modalities,
            gates["road"],
        )
        lighting_camera = self.lighting_camera_projector(inputs["camera"])
        lighting_statistics = self.lighting_illumination_projector(
            inputs["illumination"]
        )
        lighting = torch.cat((lighting_camera, lighting_statistics), dim=1)
        embeddings = {
            "weather": weather,
            "road": road,
            "lighting": lighting,
        }
        logits = {
            attribute: self.heads[attribute](embeddings[attribute])
            for attribute in self.attributes
        }
        return {
            "logits": logits,
            "embeddings": embeddings,
            "gates": gates,
        }
