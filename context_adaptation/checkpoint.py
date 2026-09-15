"""Checkpoint helpers for dynamically growing context-adaptation runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch

from .context_memory import ContextMemory
from .context_router import ContextRouter
from .dynamic_psp import get_context_manifest, restore_dynamic_contexts
from .feature_projector import RandomFeatureProjector


PathLike = Union[str, Path]
CHECKPOINT_VERSION = 3
SUPPORTED_CHECKPOINT_VERSIONS = (1, 2, CHECKPOINT_VERSION)


def _validate_cross_component_contexts(
    model_manifest: tuple[str, ...],
    context_memory: ContextMemory,
) -> None:
    if not model_manifest:
        raise RuntimeError("The model context manifest must not be empty.")
    if len(set(model_manifest)) != len(model_manifest):
        raise RuntimeError(f"The model context manifest has duplicates: {model_manifest}.")
    memory_manifest = tuple(
        entry.model_context_name
        for entry in context_memory.contexts.values()
    )
    if memory_manifest != model_manifest:
        raise RuntimeError(
            "Model and Context Memory manifests differ: "
            f"model={model_manifest}, memory={memory_manifest}."
        )


def _validate_runtime_state(
    context_memory: ContextMemory,
    router: ContextRouter,
    projector: RandomFeatureProjector,
) -> None:
    if not projector.finalized:
        raise RuntimeError("The context feature projector must be finalized.")
    dimensions = (
        projector.projection_dim,
        context_memory.input_dim,
        router.input_dim,
    )
    if len(set(dimensions)) != 1:
        raise RuntimeError(
            "Projector, Context Memory, and router dimensions differ: "
            f"projector={dimensions[0]}, memory={dimensions[1]}, router={dimensions[2]}."
        )
    valid_ids = set(context_memory.contexts)
    if not valid_ids:
        raise RuntimeError("Context Memory must contain at least one context.")
    if context_memory.active_context_id is None or router.active_context_id is None:
        raise RuntimeError("Context Memory and router must both have an active context.")
    if router._awaiting_context_creation:
        raise RuntimeError(
            "Cannot checkpoint an uncommitted context-creation transaction."
        )
    for label, context_id in (
        ("memory active", context_memory.active_context_id),
        ("router active", router.active_context_id),
    ):
        if context_id is not None and context_id not in valid_ids:
            raise RuntimeError(f"{label} context ID does not exist: {context_id}.")
    unknown_evidence_ids = set(router._context_evidence) - valid_ids
    if unknown_evidence_ids:
        raise RuntimeError(
            "Router evidence references unknown context IDs: "
            f"{sorted(unknown_evidence_ids)}."
        )


def optimizer_schema(
    optimizer: Optional[torch.optim.Optimizer],
) -> Optional[dict[str, Any]]:
    """Return the optimizer structure needed for safe resume validation."""
    if optimizer is None:
        return None
    keys = ("lr", "weight_decay", "momentum", "betas", "eps", "amsgrad")
    return {
        "class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__name__}",
        "groups": [
            {
                "context_name": group.get("context_name"),
                "num_parameters": len(group["params"]),
                "options": {key: group[key] for key in keys if key in group},
            }
            for group in optimizer.param_groups
        ],
    }


def validate_optimizer_resume(
    optimizer: torch.optim.Optimizer,
    payload: Mapping[str, Any],
) -> None:
    """Reject resumes whose optimizer type or parameter-group schema changed."""
    saved_state = payload.get("optimizer_state_dict")
    if saved_state is None:
        return
    saved_schema = payload.get("optimizer_schema")
    if saved_schema is None:
        raise ValueError(
            "This checkpoint has optimizer state but no optimizer schema; "
            "restart from a bootstrap checkpoint or use a version-2-or-newer context checkpoint."
        )
    current_schema = optimizer_schema(optimizer)
    if current_schema != saved_schema:
        raise ValueError(
            "Optimizer configuration or context parameter groups differ from the "
            f"checkpoint: saved={saved_schema}, current={current_schema}."
        )


def build_checkpoint_payload(
    *,
    network: torch.nn.Module,
    context_memory: ContextMemory,
    router: ContextRouter,
    projector: RandomFeatureProjector,
    optimizer: Optional[torch.optim.Optimizer],
    step_idx: int,
    update_idx: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if int(step_idx) < 0 or int(update_idx) < 0:
        raise ValueError("step_idx and update_idx must be nonnegative.")
    manifest = get_context_manifest(network)
    _validate_cross_component_contexts(manifest, context_memory)
    _validate_runtime_state(context_memory, router, projector)
    if router.active_context_id != context_memory.active_context_id:
        raise RuntimeError(
            "Router and Context Memory active context IDs are inconsistent: "
            f"router={router.active_context_id}, "
            f"memory={context_memory.active_context_id}."
        )
    if optimizer is not None:
        group_contexts = tuple(
            group.get("context_name") for group in optimizer.param_groups
        )
        if group_contexts != manifest:
            raise RuntimeError(
                "Optimizer context groups do not match the context manifest: "
                f"optimizer={group_contexts}, manifest={manifest}."
            )
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "component_state_versions": {
            "context_memory": ContextMemory.STATE_VERSION,
            "router": ContextRouter.STATE_VERSION,
            "projector": RandomFeatureProjector.STATE_VERSION,
        },
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": (
            None if optimizer is None else optimizer.state_dict()
        ),
        "optimizer_schema": optimizer_schema(optimizer),
        "context_manifest": list(manifest),
        "context_memory_state": context_memory.state_dict(),
        "router_state": router.state_dict(),
        "projector_state": projector.state_dict(),
        "step_idx": int(step_idx),
        "update_idx": int(update_idx),
        "metadata": dict(metadata or {}),
    }


def save_context_checkpoint(
    path: PathLike,
    *,
    network: torch.nn.Module,
    context_memory: ContextMemory,
    router: ContextRouter,
    projector: RandomFeatureProjector,
    optimizer: Optional[torch.optim.Optimizer],
    step_idx: int,
    update_idx: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically save model, dynamic contexts, memory, and routing state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_payload(
        network=network,
        context_memory=context_memory,
        router=router,
        projector=projector,
        optimizer=optimizer,
        step_idx=step_idx,
        update_idx=update_idx,
        metadata=metadata,
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_context_checkpoint(
    path: PathLike,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError("Context checkpoint must contain a dictionary payload.")
    version = int(payload.get("checkpoint_version", -1))
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(f"Unsupported context checkpoint version: {version}")
    required = {
        "model_state_dict",
        "context_manifest",
        "context_memory_state",
        "router_state",
        "projector_state",
        "step_idx",
        "update_idx",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"Context checkpoint is missing keys: {missing}")
    manifest = payload["context_manifest"]
    if not isinstance(manifest, (list, tuple)) or not manifest:
        raise ValueError("Context checkpoint manifest must be a non-empty list or tuple.")
    resolved_manifest = tuple(str(name) for name in manifest)
    if any(not name for name in resolved_manifest):
        raise ValueError("Context checkpoint manifest names must not be empty.")
    if len(set(resolved_manifest)) != len(resolved_manifest):
        raise ValueError("Context checkpoint manifest names must be unique.")
    for key in ("context_memory_state", "router_state", "projector_state"):
        if not isinstance(payload[key], Mapping):
            raise TypeError(f"{key} must contain a mapping state.")
    for key in ("step_idx", "update_idx"):
        if int(payload[key]) < 0:
            raise ValueError(f"{key} must be nonnegative.")
    if version >= 3:
        component_versions = payload.get("component_state_versions")
        if not isinstance(component_versions, Mapping):
            raise KeyError("Version-3 checkpoint is missing component_state_versions.")
        state_keys = {
            "context_memory": "context_memory_state",
            "router": "router_state",
            "projector": "projector_state",
        }
        for component, state_key in state_keys.items():
            if component not in component_versions:
                raise KeyError(f"Missing component state version: {component}.")
            declared = int(component_versions[component])
            actual = int(payload[state_key].get("version", 1))
            if declared != actual:
                raise ValueError(
                    f"Component state version mismatch for {component}: "
                    f"declared={declared}, actual={actual}."
                )
    return payload


def restore_model_and_context_state(
    network: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    trainable_scope: str = "fuser_head",
) -> tuple[ContextMemory, ContextRouter, RandomFeatureProjector]:
    """Rebuild dynamic banks, load model weights, and restore routing state."""
    manifest = tuple(str(name) for name in payload["context_manifest"])
    restore_dynamic_contexts(
        network,
        manifest,
        trainable_scope=trainable_scope,
    )
    missing, unexpected = network.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Strict model restore failed: missing={missing}, unexpected={unexpected}."
        )
    memory = ContextMemory.from_state_dict(payload["context_memory_state"])
    router = ContextRouter.from_state_dict(payload["router_state"])
    projector = RandomFeatureProjector.from_state_dict(payload["projector_state"])
    _validate_cross_component_contexts(get_context_manifest(network), memory)
    _validate_runtime_state(memory, router, projector)
    if router.active_context_id != memory.active_context_id:
        raise RuntimeError("Restored router and Context Memory active IDs differ.")
    return memory, router, projector
