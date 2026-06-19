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

    def _make_binary_sign_tensor(self, scene_name, key_name, shape, device, dtype):
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
