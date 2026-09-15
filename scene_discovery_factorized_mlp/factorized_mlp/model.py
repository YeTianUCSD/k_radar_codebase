"""PyTorch models for factorized weather, road, and lighting prediction."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn


ATTRIBUTES = ("weather", "road", "lighting")


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, value):
        return self.network(value)


class FactorizedGatedMLP(nn.Module):
    """Independent static modality gates followed by three classification heads."""

    def __init__(
        self,
        input_dims: Mapping[str, int],
        class_counts: Mapping[str, int],
        modalities: Sequence[str] = ("camera", "lidar", "radar"),
        embedding_dim: int = 64,
        head_hidden_dim: int = 64,
        dropout: float = 0.2,
        fusion: str = "gated",
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.attributes = ATTRIBUTES
        self.fusion = str(fusion)
        if not self.modalities:
            raise ValueError("at least one modality is required")
        if self.fusion not in {"gated", "concat"}:
            raise ValueError("fusion must be gated or concat")
        missing = set(self.modalities).difference(input_dims)
        if missing:
            raise ValueError(f"missing input dimensions: {sorted(missing)}")
        if set(class_counts) != set(self.attributes):
            raise ValueError("class_counts must contain weather, road, and lighting")
        self.projectors = nn.ModuleDict({
            modality: ModalityProjector(input_dims[modality], embedding_dim, dropout)
            for modality in self.modalities
        })
        self.gate_logits = nn.ParameterDict({
            attribute: nn.Parameter(torch.zeros(len(self.modalities)))
            for attribute in self.attributes
        })
        head_input = embedding_dim if self.fusion == "gated" else embedding_dim * len(self.modalities)
        self.heads = nn.ModuleDict({
            attribute: nn.Sequential(
                nn.Linear(head_input, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, int(class_counts[attribute])),
            )
            for attribute in self.attributes
        })

    def gate_weights(self):
        if self.fusion == "concat":
            uniform = 1.0 / len(self.modalities)
            return {
                attribute: torch.full_like(self.gate_logits[attribute], uniform)
                for attribute in self.attributes
            }
        return {
            attribute: torch.softmax(self.gate_logits[attribute], dim=0)
            for attribute in self.attributes
        }

    def forward(self, inputs):
        projected = {
            modality: self.projectors[modality](inputs[modality])
            for modality in self.modalities
        }
        weights = self.gate_weights()
        logits = {}
        embeddings = {}
        for attribute in self.attributes:
            if self.fusion == "gated":
                embedding = sum(
                    weights[attribute][position] * projected[modality]
                    for position, modality in enumerate(self.modalities)
                )
            else:
                embedding = torch.cat(
                    [projected[modality] for modality in self.modalities], dim=1
                )
            embeddings[attribute] = embedding
            logits[attribute] = self.heads[attribute](embedding)
        return {"logits": logits, "embeddings": embeddings, "gates": weights}
