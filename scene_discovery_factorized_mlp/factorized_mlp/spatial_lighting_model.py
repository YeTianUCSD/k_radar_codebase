"""Weather-multimodal model with isolated Road and spatial Lighting paths."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .model import ModalityProjector


class WeatherMultimodalSpatialLightingMLP(nn.Module):
    """Weather uses all sensors; Road and Lighting have isolated Camera paths."""

    attributes = ("weather", "road", "lighting")

    def __init__(
        self, input_dims: Mapping[str, int], class_counts: Mapping[str, int],
        modalities: Sequence[str] = ("camera", "lidar", "radar"),
        embedding_dim: int = 32, illumination_embedding_dim: int = 16,
        head_hidden_dim: int = 64, dropout: float = 0.3,
        fusion: str = "gated",
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.fusion = str(fusion)
        if "camera" not in self.modalities:
            raise ValueError("spatial-lighting model requires camera")
        if "illumination" not in input_dims:
            raise ValueError("spatial-lighting model requires illumination features")
        if self.fusion not in {"concat", "gated"}:
            raise ValueError("fusion must be concat or gated")
        self.weather_projectors = nn.ModuleDict({
            modality: ModalityProjector(
                input_dims[modality], embedding_dim, dropout
            ) for modality in self.modalities
        })
        # Road and Lighting intentionally do not share trainable parameters.
        self.road_camera_projector = ModalityProjector(
            input_dims["camera"], embedding_dim, dropout
        )
        self.lighting_camera_projector = ModalityProjector(
            input_dims["camera"], embedding_dim, dropout
        )
        self.lighting_illumination_projector = ModalityProjector(
            input_dims["illumination"], illumination_embedding_dim, dropout
        )
        self.weather_gate_logits = nn.Parameter(
            torch.zeros(len(self.modalities))
        )
        weather_input = (
            embedding_dim * len(self.modalities)
            if self.fusion == "concat" else embedding_dim
        )
        head_inputs = {
            "weather": weather_input,
            "road": embedding_dim,
            "lighting": embedding_dim + illumination_embedding_dim,
        }
        self.heads = nn.ModuleDict({
            attribute: nn.Sequential(
                nn.Linear(head_inputs[attribute], head_hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, int(class_counts[attribute])),
            ) for attribute in self.attributes
        })

    def gate_weights(self):
        if self.fusion == "concat":
            return {}
        return {"weather": torch.softmax(self.weather_gate_logits, dim=0)}

    def forward(self, inputs):
        projected = {
            modality: self.weather_projectors[modality](inputs[modality])
            for modality in self.modalities
        }
        gates = self.gate_weights()
        if self.fusion == "concat":
            weather = torch.cat(
                [projected[modality] for modality in self.modalities], dim=1
            )
        else:
            weather = sum(
                gates["weather"][position] * projected[modality]
                for position, modality in enumerate(self.modalities)
            )
        road = self.road_camera_projector(inputs["camera"])
        lighting_camera = self.lighting_camera_projector(inputs["camera"])
        lighting_statistics = self.lighting_illumination_projector(
            inputs["illumination"]
        )
        lighting = torch.cat((lighting_camera, lighting_statistics), dim=1)
        embeddings = {
            "weather": weather, "road": road, "lighting": lighting,
        }
        logits = {
            attribute: self.heads[attribute](embeddings[attribute])
            for attribute in self.attributes
        }
        return {"logits": logits, "embeddings": embeddings, "gates": gates}
