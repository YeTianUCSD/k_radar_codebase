"""Runtime growth utilities for existing PSP scene residual banks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from models.superposition import (
    PSPConv2d,
    PSPLinear,
    PSPLayerNorm,
    SceneResidualBank,
)


@dataclass(frozen=True)
class DynamicContextResult:
    context_name: str
    scene_key: str
    all_parameter_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameters: tuple[nn.Parameter, ...]


@dataclass(frozen=True)
class InheritanceAuditResult:
    parent_context: str
    context_name: str
    handled_bank_names: tuple[str, ...]
    compared_tensor_count: int
    max_effective_parameter_diff: float


def _scope_matches(parameter_name: str, scope: str) -> bool:
    scope = str(scope)
    if scope == "cls":
        return parameter_name.startswith("head.conv_cls.")
    if scope == "head":
        return parameter_name.startswith("head.")
    if scope == "fuser_head":
        return parameter_name.startswith("fuser.") or parameter_name.startswith(
            "head."
        )
    if scope == "full":
        return True
    raise ValueError(f"Unsupported trainable scope: {scope}")


def iter_scene_residual_banks(
    network: nn.Module,
) -> Iterable[tuple[str, SceneResidualBank]]:
    """Yield every residual bank once, including weight-bank subclasses."""
    seen: set[int] = set()
    for module_name, module in network.named_modules():
        if not isinstance(module, SceneResidualBank) or id(module) in seen:
            continue
        seen.add(id(module))
        yield module_name, module


def get_context_manifest(network: nn.Module) -> tuple[str, ...]:
    """Return the common ordered context manifest across all residual banks."""
    manifests = [tuple(bank.scene_names) for _, bank in iter_scene_residual_banks(network)]
    if not manifests:
        raise RuntimeError("No SceneResidualBank modules were found in the network.")
    reference = manifests[0]
    inconsistent = [manifest for manifest in manifests[1:] if manifest != reference]
    if inconsistent:
        raise RuntimeError("PSP residual banks have inconsistent context manifests.")
    return reference


def _reference_device_dtype(bank: SceneResidualBank) -> tuple[torch.device, torch.dtype]:
    for parameter in bank.params.values():
        return parameter.device, parameter.dtype
    return torch.device("cpu"), torch.get_default_dtype()


def add_dynamic_context(
    network: nn.Module,
    context_name: str,
    *,
    trainable_scope: str = "fuser_head",
) -> DynamicContextResult:
    """Register a zero residual for a new context in every existing bank."""
    resolved_name = str(context_name)
    if not resolved_name:
        raise ValueError("context_name must not be empty.")
    banks = list(iter_scene_residual_banks(network))
    manifest = get_context_manifest(network)
    presence = [bank.has_scene(resolved_name) for _, bank in banks]
    if any(presence):
        if not all(presence):
            raise RuntimeError(
                f"Context {resolved_name} exists in only a subset of PSP residual banks."
            )
        raise ValueError(f"Context already exists in the model: {resolved_name}")

    scene_index = len(manifest)
    scene_key = f"scene_{scene_index:04d}"
    for bank_name, bank in banks:
        if scene_key in bank.params:
            raise RuntimeError(
                f"Residual key collision in {bank_name}: {scene_key}"
            )

    all_names: list[str] = []
    trainable_names: list[str] = []
    trainable_parameters: list[nn.Parameter] = []
    modified_banks: list[SceneResidualBank] = []
    try:
        for bank_name, bank in banks:
            device, dtype = _reference_device_dtype(bank)
            parameter = nn.Parameter(
                torch.zeros(bank.shape, device=device, dtype=dtype),
                requires_grad=False,
            )
            bank.params[scene_key] = parameter
            bank.scene_names = tuple(bank.scene_names) + (resolved_name,)
            bank.scene_to_key[resolved_name] = scene_key
            modified_banks.append(bank)

            parameter_name = f"{bank_name}.params.{scene_key}"
            all_names.append(parameter_name)
            if _scope_matches(parameter_name, trainable_scope):
                parameter.requires_grad = True
                trainable_names.append(parameter_name)
                trainable_parameters.append(parameter)

        updated_manifest = get_context_manifest(network)
        if updated_manifest != manifest + (resolved_name,):
            raise RuntimeError("Dynamic context registration produced an invalid manifest.")
        if not trainable_parameters:
            raise RuntimeError(
                f"No dynamic residual parameters matched scope={trainable_scope}."
            )
    except Exception:
        for bank in reversed(modified_banks):
            if scene_key in bank.params:
                del bank.params[scene_key]
            bank.scene_to_key.pop(resolved_name, None)
            bank.scene_names = manifest
        raise
    return DynamicContextResult(
        context_name=resolved_name,
        scene_key=scene_key,
        all_parameter_names=tuple(all_names),
        trainable_parameter_names=tuple(trainable_names),
        trainable_parameters=tuple(trainable_parameters),
    )


def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise RuntimeError(
            f"Inheritance tensor shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    if left.numel() == 0:
        return 0.0
    return float((left.detach() - right.detach()).abs().max().item())


def audit_inherited_context(
    network: nn.Module,
    parent_context: str,
    context_name: str,
    *,
    handled_bank_names: Sequence[str] = (),
) -> InheritanceAuditResult:
    """Compare effective PSP tensors for a parent and inherited context."""
    parent = str(parent_context)
    child = str(context_name)
    compared = 0
    max_diff = 0.0

    with torch.no_grad():
        for module in network.modules():
            if isinstance(module, PSPLinear):
                weight = module.weight.detach()
                parent_sign = module.scene_context_registry.get_vector(
                    parent, module.key_name, module.in_features, weight.device, weight.dtype
                ).view(1, -1)
                child_sign = module.scene_context_registry.get_vector(
                    child, module.key_name, module.in_features, weight.device, weight.dtype
                ).view(1, -1)
                parent_residual = module.scene_weight_bank.get(parent)
                child_residual = module.scene_weight_bank.get(child)
                parent_effective = (weight + parent_residual) * parent_sign
                child_effective = (weight + child_residual) * child_sign
                max_diff = max(max_diff, _max_abs_difference(parent_effective, child_effective))
                compared += 1
                if module.bias is not None and module.scene_bias_bank is not None:
                    parent_bias = module.bias.detach() + module.scene_bias_bank.get(parent)
                    child_bias = module.bias.detach() + module.scene_bias_bank.get(child)
                    max_diff = max(max_diff, _max_abs_difference(parent_bias, child_bias))
                    compared += 1

            elif isinstance(module, PSPConv2d):
                weight = module.weight.detach()
                shape = weight.shape[1:]
                parent_sign = module.scene_context_registry.get_tensor(
                    parent, module.key_name, shape, weight.device, weight.dtype
                ).unsqueeze(0)
                child_sign = module.scene_context_registry.get_tensor(
                    child, module.key_name, shape, weight.device, weight.dtype
                ).unsqueeze(0)
                parent_effective = weight * parent_sign + module.scene_weight_bank.get(parent)
                child_effective = weight * child_sign + module.scene_weight_bank.get(child)
                max_diff = max(max_diff, _max_abs_difference(parent_effective, child_effective))
                compared += 1
                if module.bias is not None and module.scene_bias_bank is not None:
                    parent_bias = module.bias.detach() + module.scene_bias_bank.get(parent)
                    child_bias = module.bias.detach() + module.scene_bias_bank.get(child)
                    max_diff = max(max_diff, _max_abs_difference(parent_bias, child_bias))
                    compared += 1

            elif isinstance(module, PSPLayerNorm) and module.elementwise_affine:
                weight = module.weight.detach()
                parent_weight_sign = module.scene_context_registry.get_tensor(
                    parent,
                    f"{module.key_name}.weight",
                    module.normalized_shape,
                    weight.device,
                    weight.dtype,
                )
                child_weight_sign = module.scene_context_registry.get_tensor(
                    child,
                    f"{module.key_name}.weight",
                    module.normalized_shape,
                    weight.device,
                    weight.dtype,
                )
                max_diff = max(
                    max_diff,
                    _max_abs_difference(
                        weight * parent_weight_sign, weight * child_weight_sign
                    ),
                )
                compared += 1
                bias = module.bias.detach()
                parent_bias_sign = module.scene_context_registry.get_tensor(
                    parent,
                    f"{module.key_name}.bias",
                    module.normalized_shape,
                    bias.device,
                    bias.dtype,
                )
                child_bias_sign = module.scene_context_registry.get_tensor(
                    child,
                    f"{module.key_name}.bias",
                    module.normalized_shape,
                    bias.device,
                    bias.dtype,
                )
                max_diff = max(
                    max_diff,
                    _max_abs_difference(bias * parent_bias_sign, bias * child_bias_sign),
                )
                compared += 1

        aware_bank = getattr(getattr(network, "fuser", None), "aware_query_scene_bank", None)
        aware_query = getattr(getattr(network, "fuser", None), "aware_query", None)
        if aware_bank is not None and aware_query is not None:
            parent_query = aware_query.detach() + aware_bank.get(parent)
            child_query = aware_query.detach() + aware_bank.get(child)
            max_diff = max(max_diff, _max_abs_difference(parent_query, child_query))
            compared += 1

    if compared <= 0:
        raise RuntimeError("No effective PSP tensors were compared for inheritance.")
    return InheritanceAuditResult(
        parent_context=parent,
        context_name=child,
        handled_bank_names=tuple(sorted(str(name) for name in handled_bank_names)),
        compared_tensor_count=compared,
        max_effective_parameter_diff=max_diff,
    )


def add_inherited_dynamic_context(
    network: nn.Module,
    context_name: str,
    parent_context: str,
    *,
    trainable_scope: str = "fuser_head",
    atol: float = 1e-6,
) -> tuple[DynamicContextResult, InheritanceAuditResult]:
    """Create a context whose effective PSP function initially matches its parent."""
    child = str(context_name)
    parent = str(parent_context)
    manifest = get_context_manifest(network)
    if parent not in manifest:
        raise KeyError(f"Unknown parent context {parent}; available={manifest}")
    if child in manifest:
        raise ValueError(f"Context already exists in the model: {child}")
    requires_grad_before = [(parameter, parameter.requires_grad) for parameter in network.parameters()]
    registered_aliases = []
    result = None

    try:
        result = add_dynamic_context(
            network,
            child,
            trainable_scope=trainable_scope,
        )
        banks = list(iter_scene_residual_banks(network))
        handled = {}

        def mark(bank: SceneResidualBank, bank_name: str) -> None:
            if id(bank) in handled:
                raise RuntimeError(f"Residual bank handled twice: {bank_name}")
            handled[id(bank)] = bank_name

        with torch.no_grad():
            for module_name, module in network.named_modules():
                if isinstance(module, PSPLinear):
                    weight_bank = module.scene_weight_bank
                    if weight_bank is None:
                        raise RuntimeError(f"Enabled PSPLinear has no weight bank: {module_name}")
                    weight = module.weight.detach()
                    parent_sign = module.scene_context_registry.get_vector(
                        parent, module.key_name, module.in_features, weight.device, weight.dtype
                    ).view(1, -1)
                    child_sign = module.scene_context_registry.get_vector(
                        child, module.key_name, module.in_features, weight.device, weight.dtype
                    ).view(1, -1)
                    parent_residual = weight_bank.get(parent)
                    weight_bank.get(child).copy_(
                        (weight + parent_residual) * (parent_sign * child_sign) - weight
                    )
                    mark(weight_bank, f"{module_name}.scene_weight_bank")
                    if module.bias is not None and module.scene_bias_bank is not None:
                        module.scene_bias_bank.get(child).copy_(
                            module.scene_bias_bank.get(parent)
                        )
                        mark(module.scene_bias_bank, f"{module_name}.scene_bias_bank")

                elif isinstance(module, PSPConv2d):
                    weight_bank = module.scene_weight_bank
                    if weight_bank is None:
                        raise RuntimeError(f"Enabled PSPConv2d has no weight bank: {module_name}")
                    weight = module.weight.detach()
                    shape = weight.shape[1:]
                    parent_sign = module.scene_context_registry.get_tensor(
                        parent, module.key_name, shape, weight.device, weight.dtype
                    ).unsqueeze(0)
                    child_sign = module.scene_context_registry.get_tensor(
                        child, module.key_name, shape, weight.device, weight.dtype
                    ).unsqueeze(0)
                    parent_residual = weight_bank.get(parent)
                    weight_bank.get(child).copy_(
                        weight * parent_sign + parent_residual - weight * child_sign
                    )
                    mark(weight_bank, f"{module_name}.scene_weight_bank")
                    if module.bias is not None and module.scene_bias_bank is not None:
                        module.scene_bias_bank.get(child).copy_(
                            module.scene_bias_bank.get(parent)
                        )
                        mark(module.scene_bias_bank, f"{module_name}.scene_bias_bank")

                elif isinstance(module, PSPLayerNorm):
                    registry = module.scene_context_registry
                    registry.register_alias(child, parent)
                    registered_aliases.append(registry)

            for bank_name, bank in banks:
                if id(bank) in handled:
                    continue
                if bank_name.endswith("aware_query_scene_bank"):
                    bank.get(child).copy_(bank.get(parent))
                    mark(bank, bank_name)
                    continue
                raise RuntimeError(
                    f"Inherited context does not know how to initialize bank: {bank_name}"
                )

        activate_context_residuals(
            network,
            child,
            trainable_scope=trainable_scope,
        )
        audit = audit_inherited_context(
            network,
            parent,
            child,
            handled_bank_names=tuple(handled.values()),
        )
        if audit.max_effective_parameter_diff > float(atol):
            raise RuntimeError(
                f"Inherited context mismatch for {child}: "
                f"max_diff={audit.max_effective_parameter_diff} > atol={atol}"
            )
        return result, audit
    except Exception:
        for registry in reversed(registered_aliases):
            registry.remove_alias(child)
        if result is not None and get_context_manifest(network)[-1:] == (child,):
            remove_dynamic_context(network, child)
        for parameter, requires_grad in requires_grad_before:
            parameter.requires_grad = requires_grad
        raise


def remove_dynamic_context(
    network: nn.Module,
    context_name: str,
) -> tuple[nn.Parameter, ...]:
    """Remove the newest context from every bank during transaction rollback."""
    resolved_name = str(context_name)
    manifest = get_context_manifest(network)
    if not manifest or manifest[-1] != resolved_name:
        raise RuntimeError("Only the newest dynamic model context can be removed.")
    banks = list(iter_scene_residual_banks(network))
    scene_key = f"scene_{len(manifest) - 1:04d}"
    for bank_name, bank in banks:
        if bank.scene_to_key.get(resolved_name) != scene_key or scene_key not in bank.params:
            raise RuntimeError(
                f"Dynamic context rollback validation failed in {bank_name}."
            )
    removed = []
    previous_manifest = manifest[:-1]
    for _, bank in banks:
        removed.append(bank.params[scene_key])
        del bank.params[scene_key]
        bank.scene_to_key.pop(resolved_name)
        bank.scene_names = previous_manifest
    if get_context_manifest(network) != previous_manifest:
        raise RuntimeError("Dynamic context rollback produced an invalid manifest.")
    return tuple(removed)


def remove_context_optimizer_group(
    optimizer: torch.optim.Optimizer,
    context_name: str,
) -> int:
    """Remove one dynamically appended context group and its optimizer state."""
    matches = [
        index
        for index, group in enumerate(optimizer.param_groups)
        if group.get("context_name") == str(context_name)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one optimizer group for context={context_name}; got {matches}."
        )
    index = matches[0]
    if index != len(optimizer.param_groups) - 1:
        raise RuntimeError("Only the newest optimizer context group can be removed.")
    group = optimizer.param_groups.pop(index)
    for parameter in group["params"]:
        optimizer.state.pop(parameter, None)
    return len(group["params"])


def restore_dynamic_contexts(
    network: nn.Module,
    context_manifest: Sequence[str],
    *,
    trainable_scope: str = "fuser_head",
) -> list[DynamicContextResult]:
    """Recreate checkpoint contexts in their original order before loading weights."""
    target = tuple(str(name) for name in context_manifest)
    current = get_context_manifest(network)
    if target[: len(current)] != current:
        raise ValueError(
            f"Checkpoint context manifest {target} does not extend model manifest {current}."
        )
    results = []
    for context_name in target[len(current) :]:
        results.append(
            add_dynamic_context(
                network,
                context_name,
                trainable_scope=trainable_scope,
            )
        )
    return results


def restore_inherited_contexts(
    network: nn.Module,
    context_manifest: Sequence[str],
    context_parents: Mapping[str, str],
    *,
    trainable_scope: str = "fuser_head",
    atol: float = 1e-6,
) -> list[tuple[DynamicContextResult, InheritanceAuditResult]]:
    """Recreate inherited contexts and LayerNorm aliases before strict loading."""
    target = tuple(str(name) for name in context_manifest)
    current = get_context_manifest(network)
    if target[: len(current)] != current:
        raise ValueError(
            f"Checkpoint context manifest {target} does not extend model manifest {current}."
        )
    normalized_parents = {str(key): str(value) for key, value in context_parents.items()}
    results = []
    for context_name in target[len(current) :]:
        parent = normalized_parents.get(context_name)
        if parent is None:
            raise ValueError(f"Missing parent for inherited context: {context_name}")
        if parent not in get_context_manifest(network):
            raise ValueError(
                f"Parent {parent} must be restored before context {context_name}."
            )
        results.append(
            add_inherited_dynamic_context(
                network,
                context_name,
                parent,
                trainable_scope=trainable_scope,
                atol=atol,
            )
        )
    expected_children = set(target[len(current) :])
    extra_parents = set(normalized_parents) - expected_children
    if extra_parents:
        raise ValueError(f"Unexpected context parent entries: {sorted(extra_parents)}")
    return results


def activate_context_residuals(
    network: nn.Module,
    context_name: str,
    *,
    trainable_scope: str = "fuser_head",
) -> tuple[tuple[str, ...], tuple[nn.Parameter, ...]]:
    """Enable only the selected context residuals inside the requested scope."""
    resolved_name = str(context_name)
    names: list[str] = []
    parameters: list[nn.Parameter] = []
    found = False
    for bank_name, bank in iter_scene_residual_banks(network):
        if not bank.has_scene(resolved_name):
            available = ", ".join(bank.scene_names)
            raise KeyError(
                f"Unknown context {resolved_name} in {bank_name}; available: {available}"
            )
        scene_key = bank.scene_to_key[resolved_name]
        for key, parameter in bank.params.items():
            parameter_name = f"{bank_name}.params.{key}"
            should_train = key == scene_key and _scope_matches(
                parameter_name,
                trainable_scope,
            )
            parameter.requires_grad = should_train
            if should_train:
                found = True
                names.append(parameter_name)
                parameters.append(parameter)
    if not found:
        raise RuntimeError(
            f"No residual parameters were activated for context={resolved_name}, "
            f"scope={trainable_scope}."
        )
    if hasattr(network, "default_scene_context"):
        network.default_scene_context = resolved_name
    return tuple(names), tuple(parameters)


def add_context_optimizer_group(
    optimizer: torch.optim.Optimizer,
    result: DynamicContextResult,
    *,
    learning_rate: Optional[float] = None,
) -> int:
    """Append new context parameters without disturbing existing optimizer state."""
    existing_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    new_parameters = [
        parameter
        for parameter in result.trainable_parameters
        if id(parameter) not in existing_ids
    ]
    if not new_parameters:
        return 0
    group = {
        "params": new_parameters,
        "context_name": result.context_name,
    }
    if learning_rate is not None:
        group["lr"] = float(learning_rate)
    optimizer.add_param_group(group)
    return len(new_parameters)
