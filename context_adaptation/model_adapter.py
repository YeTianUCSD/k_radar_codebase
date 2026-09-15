"""Adapter that inserts context routing between encoders and PSP fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping

import torch

from models.superposition import (
    freeze_shared_scene_specific_anchors,
    keep_only_scene_specific_residuals,
)


@dataclass(frozen=True)
class ResidualTrainingInfo:
    total_parameters: int
    trainable_parameters: int
    kept_residual_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    frozen_shared_anchors: tuple[str, ...]


class ContextAwareModelAdapter:
    """Run frozen encoders before selecting a scene-specific PSP context."""

    def __init__(self, network: torch.nn.Module) -> None:
        self.network = network
        for module_name in ("cam", "ldr", "rdr", "fuser", "head"):
            if not hasattr(network, module_name):
                raise AttributeError(f"Network is missing module: {module_name}")

    def encode(
        self,
        batch_dict: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        """Run context-independent encoders without building gradients."""
        with torch.no_grad():
            batch_dict = self.network.cam(batch_dict)
            batch_dict = self.network.ldr(batch_dict)
            batch_dict = self.network.rdr(batch_dict)
        return batch_dict

    def forward_with_context(
        self,
        encoded_batch: MutableMapping[str, Any],
        context_name: str,
    ) -> MutableMapping[str, Any]:
        """Run PSP fusion and detection after routing selected a context."""
        resolved_name = str(context_name)
        if not resolved_name:
            raise ValueError("context_name must not be empty.")
        encoded_batch["scene_context"] = resolved_name
        if hasattr(self.network, "default_scene_context"):
            self.network.default_scene_context = resolved_name
        output = self.network.fuser(encoded_batch)
        output = self.network.head(output)
        return output

    def loss(self, output: MutableMapping[str, Any]) -> torch.Tensor:
        loss = self.network.loss(output)
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise RuntimeError("Network loss must be a scalar tensor.")
        return loss

    def infer_with_context(
        self,
        encoded_batch: MutableMapping[str, Any],
        context_name: str,
    ) -> MutableMapping[str, Any]:
        with torch.no_grad():
            return self.forward_with_context(encoded_batch, context_name)


def freeze_encoder_batch_stats(network: torch.nn.Module) -> tuple[str, ...]:
    """Keep frozen encoder normalization layers in evaluation mode."""
    norm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.LayerNorm,
    )
    frozen = []
    for name in ("cam", "ldr", "rdr"):
        module = getattr(network, name, None)
        if module is None:
            continue
        module.eval()
        frozen.append(name)
        for child in module.modules():
            if isinstance(child, norm_types):
                child.eval()
    return tuple(frozen)


def configure_residual_only_training(
    network: torch.nn.Module,
    *,
    trainable_scope: str = "fuser_head",
) -> ResidualTrainingInfo:
    """Freeze shared weights and retain only scene-specific residual updates."""
    if trainable_scope not in {"cls", "head", "fuser_head", "full"}:
        raise ValueError(f"Unsupported trainable scope: {trainable_scope}")

    total_parameters = sum(parameter.numel() for parameter in network.parameters())
    for parameter in network.parameters():
        parameter.requires_grad = False

    if trainable_scope == "cls":
        modules = [getattr(getattr(network, "head"), "conv_cls")]
    elif trainable_scope == "head":
        modules = [getattr(network, "head")]
    elif trainable_scope == "fuser_head":
        modules = [getattr(network, "fuser"), getattr(network, "head")]
    else:
        modules = [network]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = parameter.is_floating_point() or parameter.is_complex()

    frozen_shared_anchors = freeze_shared_scene_specific_anchors(network)
    kept, frozen = keep_only_scene_specific_residuals(
        network,
        scope=trainable_scope,
    )
    freeze_encoder_batch_stats(network)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters <= 0:
        raise RuntimeError(
            f"No residual parameters remain trainable for scope={trainable_scope}."
        )
    return ResidualTrainingInfo(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        kept_residual_names=tuple(kept),
        frozen_parameter_names=tuple(frozen),
        frozen_shared_anchors=tuple(frozen_shared_anchors),
    )
