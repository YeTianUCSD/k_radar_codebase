"""Build context descriptors from frozen multimodal BEV encoder features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FeatureProjectorInfo:
    feature_keys: tuple[str, ...]
    pooling: tuple[str, ...]
    input_dim: int
    projection_dim: int
    fitted_samples: int
    finalized: bool


class RandomFeatureProjector:
    """Pool BEV features, normalize them, and apply a fixed projection."""

    STATE_VERSION = 1
    SUPPORTED_POOLING = {"mean", "std"}

    def __init__(
        self,
        feature_keys: Sequence[str],
        projection_dim: int = 2,
        *,
        pooling: Sequence[str] = ("mean", "std"),
        random_state: int = 20260812,
        eps: float = 1e-6,
        dtype: Any = np.float32,
    ) -> None:
        keys = tuple(str(key) for key in feature_keys)
        pooling_names = tuple(str(name).lower() for name in pooling)
        if not keys:
            raise ValueError("feature_keys must not be empty.")
        if len(set(keys)) != len(keys):
            raise ValueError("feature_keys must be unique.")
        if not pooling_names:
            raise ValueError("pooling must not be empty.")
        unsupported = set(pooling_names) - self.SUPPORTED_POOLING
        if unsupported:
            raise ValueError(f"Unsupported pooling methods: {sorted(unsupported)}")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.feature_keys = keys
        self.pooling = pooling_names
        self.projection_dim = int(projection_dim)
        self.random_state = int(random_state)
        self.eps = float(eps)
        self.dtype = np.dtype(dtype)

        self.input_dim = 0
        self.fitted_samples = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._projection: np.ndarray | None = None
        self._finalized = False

    def extract_descriptor(self, batch_dict: Mapping[str, Any]) -> np.ndarray:
        """Return a ``[batch, descriptor_dim]`` array from BEV tensors."""
        pooled = []
        batch_size = None
        for key in self.feature_keys:
            if key not in batch_dict:
                raise KeyError(f"Missing context feature: {key}")
            feature = batch_dict[key]
            if not torch.is_tensor(feature):
                raise TypeError(f"Feature {key} must be a torch.Tensor.")
            if feature.ndim != 4:
                raise ValueError(
                    f"Feature {key} must have shape [B, C, H, W]; "
                    f"got {tuple(feature.shape)}."
                )
            if batch_size is None:
                batch_size = int(feature.shape[0])
            elif int(feature.shape[0]) != batch_size:
                raise ValueError("All context features must use the same batch size.")

            detached = feature.detach().to(dtype=torch.float32)
            for method in self.pooling:
                if method == "mean":
                    value = detached.mean(dim=(-2, -1))
                elif method == "std":
                    value = detached.std(dim=(-2, -1), unbiased=False)
                else:
                    raise RuntimeError(f"Unhandled pooling method: {method}")
                pooled.append(value)

        descriptor = torch.cat(pooled, dim=1)
        output = descriptor.cpu().numpy().astype(self.dtype, copy=False)
        if not np.all(np.isfinite(output)):
            raise ValueError("Extracted descriptor contains NaN or infinite values.")
        return output

    def update_normalization(self, descriptors: Any) -> None:
        """Accumulate normal-context mean and variance with Welford updates."""
        if self._finalized:
            raise RuntimeError("The projector is finalized and cannot be refitted.")
        values = np.asarray(descriptors, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("descriptors must have shape [N, D] with N > 0.")
        if not np.all(np.isfinite(values)):
            raise ValueError("descriptors contain NaN or infinite values.")

        current_dim = int(values.shape[1])
        if self.input_dim == 0:
            self.input_dim = current_dim
            self._mean = np.zeros(current_dim, dtype=np.float64)
            self._m2 = np.zeros(current_dim, dtype=np.float64)
        elif current_dim != self.input_dim:
            raise ValueError(
                f"Descriptor dimension changed from {self.input_dim} to {current_dim}."
            )

        batch_count = int(values.shape[0])
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        if self.fitted_samples == 0:
            self._mean = batch_mean
            self._m2 = batch_m2
            self.fitted_samples = batch_count
            return

        assert self._mean is not None and self._m2 is not None
        old_count = self.fitted_samples
        total_count = old_count + batch_count
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (batch_count / total_count)
        self._m2 = (
            self._m2
            + batch_m2
            + np.square(delta) * old_count * batch_count / total_count
        )
        self.fitted_samples = total_count

    def finalize(self) -> None:
        """Freeze normalization statistics and generate the projection matrix."""
        if self._finalized:
            return
        if self.fitted_samples <= 0 or self.input_dim <= 0:
            raise RuntimeError("No normal-context descriptors were fitted.")
        assert self._m2 is not None
        variance = self._m2 / max(1, self.fitted_samples)
        self._scale = np.sqrt(np.maximum(variance, 0.0))
        self._scale[self._scale < self.eps] = 1.0

        rng = np.random.default_rng(self.random_state)
        self._projection = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(self.input_dim),
            size=(self.input_dim, self.projection_dim),
        ).astype(self.dtype)
        self._mean = np.asarray(self._mean, dtype=self.dtype)
        self._scale = np.asarray(self._scale, dtype=self.dtype)
        self._m2 = None
        self._finalized = True

    def transform_descriptor(self, descriptors: Any) -> np.ndarray:
        """Project descriptors into the fixed KDE coordinate system."""
        if not self._finalized:
            raise RuntimeError("Call finalize() before transforming descriptors.")
        values = np.asarray(descriptors, dtype=self.dtype)
        was_single = values.ndim == 1
        if was_single:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected descriptor shape ({self.input_dim},) or [N, {self.input_dim}]; "
                f"got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("descriptors contain NaN or infinite values.")
        assert self._mean is not None
        assert self._scale is not None
        assert self._projection is not None
        normalized = (values - self._mean) / self._scale
        projected = normalized @ self._projection
        projected = projected.astype(self.dtype, copy=False)
        return projected[0] if was_single else projected

    def transform_batch(self, batch_dict: Mapping[str, Any]) -> np.ndarray:
        return self.transform_descriptor(self.extract_descriptor(batch_dict))

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def info(self) -> FeatureProjectorInfo:
        return FeatureProjectorInfo(
            feature_keys=self.feature_keys,
            pooling=self.pooling,
            input_dim=self.input_dim,
            projection_dim=self.projection_dim,
            fitted_samples=self.fitted_samples,
            finalized=self.finalized,
        )

    def state_dict(self) -> dict[str, Any]:
        if not self._finalized:
            raise RuntimeError("Only a finalized projector can be serialized.")
        assert self._mean is not None
        assert self._scale is not None
        assert self._projection is not None
        return {
            "version": self.STATE_VERSION,
            "feature_keys": list(self.feature_keys),
            "pooling": list(self.pooling),
            "projection_dim": self.projection_dim,
            "random_state": self.random_state,
            "eps": self.eps,
            "dtype": self.dtype.str,
            "input_dim": self.input_dim,
            "fitted_samples": self.fitted_samples,
            "mean": self._mean.copy(),
            "scale": self._scale.copy(),
            "projection": self._projection.copy(),
            "finalized": True,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RandomFeatureProjector":
        version = int(state.get("version", 1))
        if version != cls.STATE_VERSION:
            raise ValueError(f"Unsupported projector state version: {version}")
        obj = cls(
            feature_keys=state["feature_keys"],
            projection_dim=int(state["projection_dim"]),
            pooling=state["pooling"],
            random_state=int(state["random_state"]),
            eps=float(state["eps"]),
            dtype=np.dtype(state["dtype"]),
        )
        obj.input_dim = int(state["input_dim"])
        obj.fitted_samples = int(state["fitted_samples"])
        obj._mean = np.asarray(state["mean"], dtype=obj.dtype).copy()
        obj._scale = np.asarray(state["scale"], dtype=obj.dtype).copy()
        obj._projection = np.asarray(state["projection"], dtype=obj.dtype).copy()
        if obj._mean.shape != (obj.input_dim,):
            raise ValueError(f"Invalid projector mean shape: {obj._mean.shape}")
        if obj._scale.shape != (obj.input_dim,):
            raise ValueError(f"Invalid projector scale shape: {obj._scale.shape}")
        if obj._projection.shape != (obj.input_dim, obj.projection_dim):
            raise ValueError(
                f"Invalid projection matrix shape: {obj._projection.shape}"
            )
        if np.any(obj._scale <= 0):
            raise ValueError("Projector scale must be strictly positive.")
        obj._m2 = None
        obj._finalized = bool(state.get("finalized", True))
        if not obj._finalized:
            raise ValueError("Serialized projector state must be finalized.")
        return obj
