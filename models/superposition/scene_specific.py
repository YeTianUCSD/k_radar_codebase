import torch
import torch.nn as nn


def resolve_superposition_scene_names(superposition_cfg=None, fallback_scene_name=None):
    names = []
    if superposition_cfg is not None:
        explicit = getattr(superposition_cfg, 'SCENE_LIST', None)
        if explicit is None and isinstance(superposition_cfg, dict):
            explicit = superposition_cfg.get('SCENE_LIST', None)
        if explicit is not None:
            if isinstance(explicit, (list, tuple)):
                names.extend(str(x) for x in explicit if x is not None and str(x) != '')
            else:
                names.append(str(explicit))

        for key in ('BASE_SCENE', 'ACTIVE_SCENE'):
            value = getattr(superposition_cfg, key, None)
            if value is None and isinstance(superposition_cfg, dict):
                value = superposition_cfg.get(key, None)
            if value is not None and str(value) != '':
                names.append(str(value))

    if fallback_scene_name is not None and str(fallback_scene_name) != '':
        names.append(str(fallback_scene_name))

    deduped = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


class SceneResidualBank(nn.Module):
    def __init__(self, shape, scene_names):
        super().__init__()
        self.shape = tuple(int(x) for x in shape)
        self.scene_names = tuple(str(x) for x in scene_names)
        self.scene_to_key = {scene_name: f'scene_{idx:04d}' for idx, scene_name in enumerate(self.scene_names)}
        self.params = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(*self.shape))
            for key in self.scene_to_key.values()
        })

    def has_scene(self, scene_name):
        return str(scene_name) in self.scene_to_key

    def get(self, scene_name, device=None, dtype=None):
        scene_name = str(scene_name)
        if scene_name not in self.scene_to_key:
            available = ', '.join(self.scene_names) if self.scene_names else 'none'
            raise RuntimeError(f'Unknown scene_context={scene_name} for SceneResidualBank. Available scenes: {available}')
        value = self.params[self.scene_to_key[scene_name]]
        if device is not None or dtype is not None:
            value = value.to(device=device if device is not None else value.device, dtype=dtype if dtype is not None else value.dtype)
        return value


class SceneWeightResidualBank(SceneResidualBank):
    pass


def freeze_shared_scene_specific_anchors(module):
    frozen = []
    for module_name, child in module.named_modules():
        if hasattr(child, 'aware_query_scene_bank') and getattr(child, 'aware_query', None) is not None:
            aware_query = child.aware_query
            if aware_query.requires_grad:
                aware_query.requires_grad = False
                frozen.append(f'{module_name}.aware_query' if module_name else 'aware_query')
        if hasattr(child, 'scene_bias_bank') and getattr(child, 'bias', None) is not None:
            bias = child.bias
            if bias is not None and bias.requires_grad:
                bias.requires_grad = False
                frozen.append(f'{module_name}.bias' if module_name else 'bias')
    return frozen


def keep_only_scene_specific_residuals(module, scope='fuser_head'):
    scope = str(scope)
    allowed_prefixes = []
    if scope == 'cls':
        allowed_prefixes = [
            'head.conv_cls.',
        ]
    elif scope == 'head':
        allowed_prefixes = [
            'head.',
        ]
    elif scope == 'fuser_head':
        allowed_prefixes = [
            'fuser.aware_query_scene_bank.params.',
            'fuser.',
            'head.',
        ]
    elif scope == 'full':
        allowed_prefixes = [
            '',
        ]
    else:
        raise RuntimeError(f'Unsupported residual-only scope: {scope}')

    frozen = []
    kept = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        is_scene_residual = (
            'aware_query_scene_bank.params.' in name
            or 'scene_bias_bank.params.' in name
            or 'scene_weight_bank.params.' in name
        )
        in_scope = any(name.startswith(prefix) for prefix in allowed_prefixes)
        keep = is_scene_residual and in_scope

        if keep:
            kept.append(name)
        else:
            param.requires_grad = False
            frozen.append(name)

    return kept, frozen
