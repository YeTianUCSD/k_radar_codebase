"""Atomic coordination for dynamically creating a model context."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .context_memory import ContextEntry, ContextMemory
from .context_router import ContextRouter
from .dynamic_psp import (
    DynamicContextResult,
    add_context_optimizer_group,
    add_dynamic_context,
    remove_context_optimizer_group,
    remove_dynamic_context,
)


def create_context_transaction(
    *,
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    memory: ContextMemory,
    router: ContextRouter,
    initial_features: Any,
    created_step: int,
    trainable_scope: str,
    learning_rate: Optional[float] = None,
) -> tuple[ContextEntry, DynamicContextResult]:
    """Commit model, optimizer, memory, and router state or roll back all of them."""
    proposed_id, context_name, model_context_name = memory.propose_context_identity()
    previous_memory_active = memory.active_context_id
    previous_router_active = router.active_context_id
    dynamic_result = None
    optimizer_group_added = False
    memory_entry = None
    try:
        dynamic_result = add_dynamic_context(
            network,
            model_context_name,
            trainable_scope=trainable_scope,
        )
        added_parameters = add_context_optimizer_group(
            optimizer,
            dynamic_result,
            learning_rate=learning_rate,
        )
        if added_parameters != len(dynamic_result.trainable_parameters):
            raise RuntimeError("The dynamic optimizer group is incomplete.")
        optimizer_group_added = True
        memory_entry = memory.create_context(
            initial_features,
            context_name=context_name,
            model_context_name=model_context_name,
            created_step=created_step,
            activate=True,
            model_update_enabled=True,
            lifecycle="provisional",
        )
        if memory_entry.context_id != proposed_id:
            raise RuntimeError("Context ID changed between proposal and commit.")
        router.confirm_created_context(memory_entry.context_id)
        return memory_entry, dynamic_result
    except Exception:
        router.abort_context_creation()
        if memory_entry is not None:
            memory.remove_last_context(memory_entry.context_id)
        memory.active_context_id = previous_memory_active
        if optimizer_group_added:
            remove_context_optimizer_group(optimizer, model_context_name)
        if dynamic_result is not None:
            remove_dynamic_context(network, model_context_name)
        if previous_router_active is not None:
            router.set_active_context(previous_router_active)
        else:
            router.active_context_id = None
        raise
