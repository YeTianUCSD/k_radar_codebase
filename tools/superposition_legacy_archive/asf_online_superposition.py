import hashlib
import os
from collections import OrderedDict
from datetime import datetime

import torch
import yaml


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_state_dict(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def save_checkpoint(path, state_dict, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    torch.save(meta, f"{path}.state")


def is_selected_key(key, modules):
    return any(key.startswith(f"{prefix}.") for prefix in modules)


def stable_seed(name, seed):
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_sign_context_like(tensor, key, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(key, seed))
    bits = torch.randint(0, 2, tensor.shape, generator=generator, dtype=torch.int64)
    signs = bits.to(torch.float32).mul_(2.0).sub_(1.0)
    return signs.to(dtype=tensor.dtype)


def clone_state_dict(state_dict):
    cloned = OrderedDict()
    for key, value in state_dict.items():
        cloned[key] = value.detach().cpu().clone() if torch.is_tensor(value) else value
    return cloned


class ASFOnlineSuperpositionManager:
    FORMAT_VERSION = 1
    METHOD = "binary_sign_binding"
    LEGACY_METHODS = {"binary_sign_binding_delta_cache"}

    def __init__(
        self,
        base_state,
        base_model_path,
        modules,
        base_scene,
        active_scene,
        seed,
        output_dir,
        store_scene_deltas=False,
    ):
        self.base_state = clone_state_dict(base_state)
        self.base_model_path = os.path.abspath(base_model_path) if base_model_path else ""
        self.modules = list(modules)
        self.base_scene = str(base_scene)
        self.active_scene = str(active_scene)
        self.seed = int(seed)
        self.store_scene_deltas = bool(store_scene_deltas)
        self.output_dir = os.path.abspath(output_dir)
        self.bundle_dir = os.path.join(self.output_dir, "bundles")
        self.meta_dir = os.path.join(self.output_dir, "meta")
        self.materialized_dir = os.path.join(self.output_dir, "materialized")
        self.scene_order = []
        self.context_table = OrderedDict()
        self.scene_deltas = OrderedDict()
        self.selected_keys = [
            key for key in sorted(self.base_state.keys())
            if is_selected_key(key, self.modules) and torch.is_floating_point(self.base_state[key])
        ]
        if len(self.selected_keys) == 0:
            raise RuntimeError(f"No floating-point parameters matched superposition modules={self.modules}")
        self.storage_state = OrderedDict(
            (key, torch.zeros_like(self.base_state[key]).detach().cpu())
            for key in self.selected_keys
        )
        self.active_scene_term = OrderedDict(
            (key, torch.zeros_like(self.base_state[key]).detach().cpu())
            for key in self.selected_keys
        )
        self.frozen_storage_state = OrderedDict(
            (key, torch.zeros_like(self.base_state[key]).detach().cpu())
            for key in self.selected_keys
        )
        self.ensure_scene(self.active_scene)
        self.reset_active_scene_tracking()

    @classmethod
    def from_bundle(cls, bundle_path, active_scene=None, output_dir=None):
        payload = torch.load(bundle_path, map_location="cpu")
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported superposition bundle version: {payload.get('format_version')} != {cls.FORMAT_VERSION}"
            )
        method = payload.get("method")
        if method != cls.METHOD and method not in cls.LEGACY_METHODS:
            raise RuntimeError(
                f"Unsupported superposition bundle method: {method} != {cls.METHOD}"
            )
        manager = cls(
            base_state=payload["base_state_dict"],
            base_model_path=payload.get("base_model_path", ""),
            modules=payload["modules"],
            base_scene=payload["base_scene"],
            active_scene=active_scene or payload.get("active_scene") or payload["base_scene"],
            seed=payload["seed"],
            output_dir=output_dir or os.path.join(os.path.dirname(bundle_path), ".."),
            store_scene_deltas=bool(payload.get("store_scene_deltas", False) or payload.get("scene_deltas")),
        )
        manager.scene_order = list(payload.get("scene_order", []))
        manager.context_table = OrderedDict()
        for scene_name, scene_context in payload.get("context_table", {}).items():
            manager.context_table[scene_name] = OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in scene_context.items()
            )
        manager.scene_deltas = OrderedDict()
        for scene_name, scene_delta in payload.get("scene_deltas", {}).items():
            manager.scene_deltas[scene_name] = OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in scene_delta.items()
            )
        manager.storage_state = OrderedDict(
            (key, payload["storage_state"][key].detach().cpu().clone()) for key in manager.selected_keys
        )
        manager.ensure_scene(manager.active_scene)
        manager.reset_active_scene_tracking()
        return manager

    def has_scene(self, scene_name):
        scene_name = str(scene_name)
        return scene_name == self.base_scene or scene_name in self.scene_order

    def has_exact_delta(self, scene_name):
        scene_name = str(scene_name)
        return scene_name in self.scene_deltas

    def expected_init_state(self, scene_name):
        scene_name = str(scene_name)
        if scene_name == self.base_scene:
            return self.materialize_full_state(self.base_scene, exact=True)
        if scene_name in self.scene_order:
            return self.materialize_full_state(scene_name, exact=self.has_exact_delta(scene_name))
        return self.materialize_full_state(self.base_scene, exact=True)

    def ensure_scene(self, scene_name):
        scene_name = str(scene_name)
        if scene_name == self.base_scene:
            return
        if scene_name not in self.scene_order:
            self.scene_order.append(scene_name)
        if scene_name not in self.context_table:
            self.context_table[scene_name] = OrderedDict(
                (key, make_sign_context_like(self.base_state[key], f"{scene_name}:{key}", self.seed))
                for key in self.selected_keys
            )
        if self.store_scene_deltas and scene_name not in self.scene_deltas:
            self.scene_deltas[scene_name] = OrderedDict(
                (key, torch.zeros_like(self.base_state[key]).detach().cpu())
                for key in self.selected_keys
            )

    def reset_active_scene_tracking(self):
        self.active_scene_term = OrderedDict(
            (key, torch.zeros_like(self.base_state[key]).detach().cpu())
            for key in self.selected_keys
        )
        self.frozen_storage_state = OrderedDict(
            (key, self.storage_state[key].detach().cpu().clone())
            for key in self.selected_keys
        )

    def _compute_scene_delta(self, state_dict):
        delta = OrderedDict()
        for key in self.selected_keys:
            target_tensor = state_dict[key].detach().cpu()
            delta[key] = (target_tensor - self.base_state[key]).detach().cpu()
        return delta

    def _bind_delta(self, delta, scene_name):
        scene_name = str(scene_name)
        self.ensure_scene(scene_name)
        return OrderedDict(
            (key, (delta[key] * self.context_table[scene_name][key]).detach().cpu())
            for key in self.selected_keys
        )

    def initialize_active_scene_from_state_dict(self, state_dict, scene_name=None):
        scene_name = str(scene_name or self.active_scene)
        self.ensure_scene(scene_name)
        delta = self._compute_scene_delta(state_dict)
        self.active_scene_term = self._bind_delta(delta, scene_name)
        self.frozen_storage_state = OrderedDict(
            (key, (self.storage_state[key] - self.active_scene_term[key]).detach().cpu())
            for key in self.selected_keys
        )
        self.storage_state = OrderedDict(
            (key, (self.frozen_storage_state[key] + self.active_scene_term[key]).detach().cpu())
            for key in self.selected_keys
        )
        if self.store_scene_deltas:
            self.scene_deltas[scene_name] = delta

    def capture_scene_delta_from_state_dict(self, state_dict, scene_name=None):
        scene_name = str(scene_name or self.active_scene)
        self.ensure_scene(scene_name)
        delta = self._compute_scene_delta(state_dict)
        self.active_scene_term = self._bind_delta(delta, scene_name)
        self.storage_state = OrderedDict(
            (key, (self.frozen_storage_state[key] + self.active_scene_term[key]).detach().cpu())
            for key in self.selected_keys
        )
        if self.store_scene_deltas:
            self.scene_deltas[scene_name] = delta

    def rebuild_storage(self):
        self.storage_state = OrderedDict(
            (key, torch.zeros_like(self.base_state[key]).detach().cpu())
            for key in self.selected_keys
        )
        for scene_name in self.scene_order:
            scene_delta = self.scene_deltas.get(scene_name)
            scene_context = self.context_table.get(scene_name)
            if scene_delta is None or scene_context is None:
                continue
            for key in self.selected_keys:
                self.storage_state[key] = self.storage_state[key] + scene_delta[key] * scene_context[key]
        self.reset_active_scene_tracking()

    def recover_scene_delta(self, scene_name, exact=True):
        scene_name = str(scene_name)
        if scene_name == self.base_scene:
            return OrderedDict((key, torch.zeros_like(self.base_state[key])) for key in self.selected_keys)
        self.ensure_scene(scene_name)
        if exact and self.has_exact_delta(scene_name):
            return OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in self.scene_deltas[scene_name].items()
            )
        return OrderedDict(
            (key, (self.storage_state[key] * self.context_table[scene_name][key]).detach().cpu().clone())
            for key in self.selected_keys
        )

    def materialize_full_state(self, scene_name, exact=True):
        recovered = clone_state_dict(self.base_state)
        if str(scene_name) == self.base_scene:
            return recovered
        if exact and not self.has_exact_delta(scene_name):
            raise RuntimeError(f"Exact delta is unavailable for scene '{scene_name}'")
        delta = self.recover_scene_delta(scene_name, exact=exact)
        for key in self.selected_keys:
            recovered[key] = recovered[key] + delta[key].to(dtype=recovered[key].dtype)
        return recovered

    def export(self, tag, step_idx, update_idx, score=None, best_score=None, source_checkpoint=None, materialize_scene_names=None, materialize_exact=False):
        os.makedirs(self.bundle_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.materialized_dir, exist_ok=True)

        bundle_path = os.path.join(self.bundle_dir, f"{tag}.bundle.pt")
        bundle_payload = {
            "format_version": self.FORMAT_VERSION,
            "method": self.METHOD,
            "timestamp": timestamp_now(),
            "base_model_path": self.base_model_path,
            "base_scene": self.base_scene,
            "active_scene": self.active_scene,
            "modules": list(self.modules),
            "seed": self.seed,
            "store_scene_deltas": self.store_scene_deltas,
            "selected_keys": list(self.selected_keys),
            "scene_order": list(self.scene_order),
            "base_state_dict": clone_state_dict(self.base_state),
            "storage_state": OrderedDict((key, value.detach().cpu().clone()) for key, value in self.storage_state.items()),
            "context_table": {
                scene_name: OrderedDict((key, value.detach().cpu().clone()) for key, value in scene_context.items())
                for scene_name, scene_context in self.context_table.items()
            },
            "scene_deltas": (
                {
                    scene_name: OrderedDict((key, value.detach().cpu().clone()) for key, value in scene_delta.items())
                    for scene_name, scene_delta in self.scene_deltas.items()
                }
                if self.store_scene_deltas else {}
            ),
            "export_meta": {
                "tag": tag,
                "step_idx": int(step_idx),
                "update_idx": int(update_idx),
                "score": None if score is None else float(score),
                "best_score": None if best_score is None else float(best_score),
                "source_checkpoint": source_checkpoint,
            },
        }
        torch.save(bundle_payload, bundle_path)

        manifest = {
            "timestamp": timestamp_now(),
            "tag": tag,
            "base_model_path": self.base_model_path,
            "base_scene": self.base_scene,
            "active_scene": self.active_scene,
            "modules": list(self.modules),
            "seed": self.seed,
            "store_scene_deltas": self.store_scene_deltas,
            "scene_order": list(self.scene_order),
            "bundle_path": bundle_path,
            "step_idx": int(step_idx),
            "update_idx": int(update_idx),
            "score": None if score is None else float(score),
            "best_score": None if best_score is None else float(best_score),
            "source_checkpoint": source_checkpoint,
        }
        manifest_path = os.path.join(self.meta_dir, f"{tag}.bundle.yml")
        with open(manifest_path, "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)

        scenes = list(materialize_scene_names) if materialize_scene_names else [self.base_scene, self.active_scene]
        unique_scenes = []
        for scene_name in scenes:
            if scene_name not in unique_scenes:
                unique_scenes.append(scene_name)
        scene_paths = {}
        scene_root = os.path.join(self.materialized_dir, tag)
        os.makedirs(scene_root, exist_ok=True)
        for scene_name in unique_scenes:
            scene_state = self.materialize_full_state(scene_name, exact=materialize_exact)
            ckpt_path = os.path.join(scene_root, f"{scene_name}.checkpoint")
            save_checkpoint(ckpt_path, scene_state, {
                "context": scene_name,
                "tag": tag,
                "bundle_path": bundle_path,
                "base_scene": self.base_scene,
                "active_scene": self.active_scene,
                "source_checkpoint": source_checkpoint,
                "materialization_mode": "exact" if materialize_exact else "approximate",
            })
            scene_paths[scene_name] = ckpt_path

        return {
            "bundle_path": bundle_path,
            "manifest_path": manifest_path,
            "materialized_dir": scene_root,
            "scene_paths": scene_paths,
        }
