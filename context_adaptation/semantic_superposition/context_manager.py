"""Bind semantic IDs to dynamic PSP contexts and optimizer parameter groups."""

from __future__ import annotations

from typing import Mapping

import torch

from context_adaptation.dynamic_psp import (
    activate_context_residuals,
    add_context_optimizer_group,
    add_inherited_dynamic_context,
    get_context_manifest,
    remove_context_optimizer_group,
    remove_dynamic_context,
)
from models.superposition import PSPLayerNorm


class SemanticContextManager:
    """Transactional lifecycle for semantic IDs and PSP model contexts."""

    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        base_context_name: str,
        trainable_scope: str = "fuser_head",
        learning_rate: float | None = None,
        inheritance_atol: float = 1e-6,
        restored_state: Mapping | None = None,
    ) -> None:
        manifest = get_context_manifest(network)
        if restored_state is None and tuple(manifest) != (str(base_context_name),):
            raise ValueError(
                "automatic semantic runs must start with exactly one model context"
            )
        self.network = network
        self.optimizer = optimizer
        self.trainable_scope = str(trainable_scope)
        self.learning_rate = learning_rate
        self.inheritance_atol = float(inheritance_atol)
        if restored_state is None:
            self._model_names = {0: str(base_context_name)}
            self._parents = {}
        else:
            names = {
                int(key): str(value)
                for key, value in restored_state["model_names"].items()
            }
            if tuple(sorted(names)) != tuple(range(len(names))):
                raise ValueError("restored model Context IDs are not contiguous")
            if names.get(0) != str(base_context_name):
                raise ValueError("restored base Context differs from configuration")
            if tuple(names[index] for index in range(len(names))) != tuple(manifest):
                raise ValueError("restored Context names differ from model manifest")
            if str(restored_state.get("trainable_scope")) != self.trainable_scope:
                raise ValueError("restored trainable scope differs from configuration")
            self._model_names = names
            self._parents = {
                str(key): str(value)
                for key, value in restored_state.get("parents", {}).items()
            }

    @property
    def model_names(self) -> dict[int, str]:
        return dict(self._model_names)

    @property
    def parents(self) -> dict[str, str]:
        return dict(self._parents)

    def name(self, context_id: int) -> str:
        return self._model_names[int(context_id)]

    def activate(self, context_id: int):
        return activate_context_residuals(
            self.network,
            self.name(context_id),
            trainable_scope=self.trainable_scope,
        )

    def create_from_parent(self, context_id: int, parent_context_id: int):
        context_id = int(context_id)
        parent_context_id = int(parent_context_id)
        if context_id in self._model_names:
            raise ValueError(f"context ID already exists: {context_id}")
        if context_id != len(self._model_names):
            raise ValueError("context IDs must be allocated contiguously")
        parent_name = self.name(parent_context_id)
        context_name = f"context_{context_id:04d}"
        result = None
        group_added = False
        try:
            result, audit = add_inherited_dynamic_context(
                self.network,
                context_name,
                parent_name,
                trainable_scope=self.trainable_scope,
                atol=self.inheritance_atol,
            )
            count = add_context_optimizer_group(
                self.optimizer, result, learning_rate=self.learning_rate
            )
            if count != len(result.trainable_parameters):
                raise RuntimeError("new optimizer context group is incomplete")
            group_added = True
            self._model_names[context_id] = context_name
            self._parents[context_name] = parent_name
            return result, audit
        except Exception:
            if group_added:
                remove_context_optimizer_group(self.optimizer, context_name)
            if result is not None and get_context_manifest(self.network)[-1:] == (context_name,):
                for module in self.network.modules():
                    if isinstance(module, PSPLayerNorm):
                        try:
                            module.scene_context_registry.remove_alias(context_name)
                        except (KeyError, ValueError):
                            pass
                remove_dynamic_context(self.network, context_name)
            raise

    def state_dict(self) -> dict:
        return {
            "model_names": {str(key): value for key, value in self._model_names.items()},
            "parents": dict(self._parents),
            "trainable_scope": self.trainable_scope,
        }
