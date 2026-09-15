"""Hybrid heads: multimodal weather/road and an isolated Camera lighting path."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .model import ModalityProjector


class HybridAttributeMLP(nn.Module):
    """Use three modalities for weather/road and Camera only for lighting."""

    attributes = ("weather", "road", "lighting")
    multimodal_attributes = ("weather", "road")

    def __init__(
        self,
        input_dims: Mapping[str, int],
        class_counts: Mapping[str, int],
        modalities: Sequence[str] = ("camera", "lidar", "radar"),
        embedding_dim: int = 32,
        head_hidden_dim: int = 64,
        dropout: float = 0.3,
        fusion: str = "gated",
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.fusion = str(fusion)
        if "camera" not in self.modalities:
            raise ValueError("hybrid model requires camera")
        if self.fusion not in {"concat", "gated"}:
            raise ValueError("fusion must be concat or gated")
        self.multimodal_projectors = nn.ModuleDict({
            modality: ModalityProjector(input_dims[modality], embedding_dim, dropout)
            for modality in self.modalities
        })
        # Keep the lighting representation isolated from weather/road gradients.
        self.lighting_camera_projector = ModalityProjector(
            input_dims["camera"], embedding_dim, dropout
        )
        self.gate_logits = nn.ParameterDict({
            attribute: nn.Parameter(torch.zeros(len(self.modalities)))
            for attribute in self.multimodal_attributes
        })
        multimodal_input = (
            embedding_dim * len(self.modalities)
            if self.fusion == "concat" else embedding_dim
        )
        head_inputs = {
            "weather": multimodal_input,
            "road": multimodal_input,
            "lighting": embedding_dim,
        }
        self.heads = nn.ModuleDict({
            attribute: nn.Sequential(
                nn.Linear(head_inputs[attribute], head_hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, int(class_counts[attribute])),
            )
            for attribute in self.attributes
        })

    def gate_weights(self):
        if self.fusion == "concat":
            return {}
        return {
            attribute: torch.softmax(self.gate_logits[attribute], dim=0)
            for attribute in self.multimodal_attributes
        }

    def forward(self, inputs):
        projected = {
            modality: self.multimodal_projectors[modality](inputs[modality])
            for modality in self.modalities
        }
        gates = self.gate_weights()
        embeddings = {}
        for attribute in self.multimodal_attributes:
            if self.fusion == "concat":
                embeddings[attribute] = torch.cat(
                    [projected[modality] for modality in self.modalities], dim=1
                )
            else:
                embeddings[attribute] = sum(
                    gates[attribute][position] * projected[modality]
                    for position, modality in enumerate(self.modalities)
                )
        embeddings["lighting"] = self.lighting_camera_projector(inputs["camera"])
        logits = {
            attribute: self.heads[attribute](embeddings[attribute])
            for attribute in self.attributes
        }
        return {"logits": logits, "embeddings": embeddings, "gates": gates}


class WeatherMultimodalAttributeMLP(nn.Module):
    """Use all modalities only for weather and Camera for road/lighting."""

    attributes = ("weather", "road", "lighting")

    def __init__(
        self, input_dims: Mapping[str, int], class_counts: Mapping[str, int],
        modalities: Sequence[str] = ("camera", "lidar", "radar"),
        embedding_dim: int = 32, head_hidden_dim: int = 64,
        dropout: float = 0.3, fusion: str = "gated",
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.fusion = str(fusion)
        if "camera" not in self.modalities:
            raise ValueError("weather-multimodal model requires camera")
        if self.fusion not in {"concat", "gated"}:
            raise ValueError("fusion must be concat or gated")
        self.weather_projectors = nn.ModuleDict({
            modality: ModalityProjector(input_dims[modality], embedding_dim, dropout)
            for modality in self.modalities
        })
        # Road and lighting share a Camera-only representation that is isolated
        # from every weather projector and all LiDAR/Radar inputs.
        self.camera_context_projector = ModalityProjector(
            input_dims["camera"], embedding_dim, dropout
        )
        self.weather_gate_logits = nn.Parameter(torch.zeros(len(self.modalities)))
        weather_input = (embedding_dim * len(self.modalities)
                         if self.fusion == "concat" else embedding_dim)
        head_inputs = {
            "weather": weather_input, "road": embedding_dim,
            "lighting": embedding_dim,
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
        camera = self.camera_context_projector(inputs["camera"])
        embeddings = {"weather": weather, "road": camera, "lighting": camera}
        logits = {
            attribute: self.heads[attribute](embeddings[attribute])
            for attribute in self.attributes
        }
        return {"logits": logits, "embeddings": embeddings, "gates": gates}
