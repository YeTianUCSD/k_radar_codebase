import hashlib

import torch


def _stable_seed(seed, scene_name, key_name):
    digest = hashlib.sha256(f'{seed}:{scene_name}:{key_name}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big', signed=False)


class SceneContextRegistry:
    def __init__(self, seed=20260619, enabled=True):
        self.seed = int(seed)
        self.enabled = bool(enabled)
        self._cache = {}
        self._aliases = {}

    def resolve_scene_name(self, scene_name):
        current = str(scene_name)
        visited = set()
        while current in self._aliases:
            if current in visited:
                raise RuntimeError(f'Cycle in scene context aliases at {current}')
            visited.add(current)
            current = self._aliases[current]
        return current

    def register_alias(self, scene_name, parent_scene_name):
        scene_name = str(scene_name)
        parent_scene_name = str(parent_scene_name)
        if not scene_name or not parent_scene_name:
            raise ValueError('Scene context aliases must not be empty')
        if scene_name == parent_scene_name:
            raise ValueError('A scene context cannot alias itself')
        existing = self._aliases.get(scene_name)
        if existing is not None and existing != parent_scene_name:
            raise RuntimeError(
                f'Conflicting scene context alias for {scene_name}: '
                f'{existing} != {parent_scene_name}'
            )
        self._aliases[scene_name] = parent_scene_name
        try:
            self.resolve_scene_name(scene_name)
        except Exception:
            if existing is None:
                self._aliases.pop(scene_name, None)
            else:
                self._aliases[scene_name] = existing
            raise
        self._drop_cached_scene(scene_name)

    def remove_alias(self, scene_name):
        scene_name = str(scene_name)
        removed = self._aliases.pop(scene_name, None)
        self._drop_cached_scene(scene_name)
        return removed

    def _drop_cached_scene(self, scene_name):
        scene_name = str(scene_name)
        for key in list(self._cache):
            if key[0] == scene_name:
                del self._cache[key]

    def _make_binary_sign_tensor(self, scene_name, key_name, shape, device, dtype):
        scene_name = self.resolve_scene_name(scene_name)
        shape = tuple(int(x) for x in shape)
        cache_key = (str(scene_name), str(key_name), shape, str(device), str(dtype))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        generator = torch.Generator(device='cpu')
        generator.manual_seed(_stable_seed(self.seed, scene_name, key_name))
        bits = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
        signs = bits.to(torch.float32).mul_(2.0).sub_(1.0)
        context = signs.to(device=device, dtype=dtype)
        self._cache[cache_key] = context
        return context

    def get_tensor(self, scene_name, key_name, shape, device, dtype):
        if not self.enabled:
            return torch.ones(tuple(int(x) for x in shape), device=device, dtype=dtype)
        return self._make_binary_sign_tensor(scene_name, key_name, shape, device, dtype)

    def get_vector(self, scene_name, key_name, size, device, dtype):
        return self.get_tensor(scene_name, key_name, (int(size),), device, dtype)

    def get_channel_context(self, scene_name, key_name, channels, device, dtype):
        return self.get_vector(scene_name, key_name, channels, device, dtype)

    def get_token_context(self, scene_name, key_name, dim, device, dtype):
        return self.get_vector(scene_name, key_name, dim, device, dtype)
