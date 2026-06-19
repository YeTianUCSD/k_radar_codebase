import torch


def _validate_scene_name(scene_name):
    if scene_name is None:
        raise RuntimeError('scene_context is required for context-aware superposition')
    return str(scene_name)


def apply_channel_sign_modulation_2d(x, registry, scene_name, key_name):
    if registry is None or not getattr(registry, 'enabled', False):
        return x
    scene_name = _validate_scene_name(scene_name)
    context = registry.get_channel_context(
        scene_name=scene_name,
        key_name=key_name,
        channels=x.shape[1],
        device=x.device,
        dtype=x.dtype,
    )
    return x * context.view(1, -1, 1, 1)


def apply_token_sign_modulation(x, registry, scene_name, key_name):
    if registry is None or not getattr(registry, 'enabled', False):
        return x
    scene_name = _validate_scene_name(scene_name)
    context = registry.get_token_context(
        scene_name=scene_name,
        key_name=key_name,
        dim=x.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )
    return x * context.view(1, 1, -1)
