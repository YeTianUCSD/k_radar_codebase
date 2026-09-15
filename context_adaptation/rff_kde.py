"""Online Gaussian KDE backed by a random Fourier feature sketch.

This module is adapted from ``/home/code/hyperradar/Density_estimation/ODE.py``.
It keeps the same cumulative estimator while exposing in-memory state methods
used by the context-adaptation checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


PathLike = Union[str, Path]


@dataclass(frozen=True)
class RFFKDEInfo:
    input_dim: int
    n_features: int
    bandwidth: float
    n_samples: int
    total_weight: float
    clip_min: float


class OnlineRFFKDE:
    """Cumulative Gaussian KDE using real-valued random Fourier features."""

    STATE_VERSION = 1

    def __init__(
        self,
        input_dim: int,
        n_features: int = 1024,
        bandwidth: float = 1.0,
        *,
        random_state: Optional[int] = 0,
        clip_min: float = 0.0,
        dtype: Any = np.float64,
        default_batch_size: int = 8192,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if clip_min < 0:
            raise ValueError("clip_min must be nonnegative.")
        if default_batch_size <= 0:
            raise ValueError("default_batch_size must be positive.")

        self.input_dim = int(input_dim)
        self.n_features = int(n_features)
        self.bandwidth = float(bandwidth)
        self.clip_min = float(clip_min)
        self.dtype = np.dtype(dtype)
        self.default_batch_size = int(default_batch_size)

        rng = np.random.default_rng(random_state)
        self.omega: NDArray[np.floating] = rng.normal(
            loc=0.0,
            scale=1.0 / self.bandwidth,
            size=(self.n_features, self.input_dim),
        ).astype(self.dtype, copy=False)
        self.phase: NDArray[np.floating] = rng.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=self.n_features,
        ).astype(self.dtype, copy=False)

        self._feature_scale = self.dtype.type(np.sqrt(2.0 / self.n_features))
        self._feature_sum: NDArray[np.floating] = np.zeros(
            self.n_features,
            dtype=self.dtype,
        )
        self.n_samples = 0
        self.total_weight = 0.0
        self._log_normalizer = 0.5 * self.input_dim * np.log(
            2.0 * np.pi * self.bandwidth**2
        )

    def _validate_X(
        self,
        X: ArrayLike,
    ) -> tuple[NDArray[np.floating], bool]:
        arr = np.asarray(X, dtype=self.dtype)
        was_single = arr.ndim == 1
        if was_single:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected shape ({self.input_dim},) or "
                f"(n, {self.input_dim}); got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("X contains NaN or infinite values.")
        return arr, was_single

    def _validate_weights(
        self,
        sample_weight: Optional[ArrayLike],
        n: int,
    ) -> NDArray[np.floating]:
        if sample_weight is None:
            return np.ones(n, dtype=self.dtype)
        weights = np.asarray(sample_weight, dtype=self.dtype)
        if weights.ndim == 0:
            weights = np.full(n, weights.item(), dtype=self.dtype)
        if weights.shape != (n,):
            raise ValueError(
                f"sample_weight must be scalar or shape ({n},); "
                f"got {weights.shape}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight contains NaN or infinite values.")
        if np.any(weights < 0):
            raise ValueError("sample_weight must be nonnegative.")
        return weights

    def transform(
        self,
        X: ArrayLike,
        *,
        batch_size: Optional[int] = None,
    ) -> NDArray[np.floating]:
        """Encode one point or a batch with the fixed RFF map."""
        arr, was_single = self._validate_X(X)
        chunk_size = int(batch_size or self.default_batch_size)
        output = np.empty((arr.shape[0], self.n_features), dtype=self.dtype)
        for start in range(0, arr.shape[0], chunk_size):
            stop = min(start + chunk_size, arr.shape[0])
            output[start:stop] = self._feature_scale * np.cos(
                arr[start:stop] @ self.omega.T + self.phase
            )
        return output[0] if was_single else output

    def update(
        self,
        X: ArrayLike,
        *,
        sample_weight: Optional[ArrayLike] = None,
        batch_size: Optional[int] = None,
    ) -> "OnlineRFFKDE":
        """Add observations without retaining the original samples."""
        arr, _ = self._validate_X(X)
        weights = self._validate_weights(sample_weight, arr.shape[0])
        chunk_size = int(batch_size or self.default_batch_size)
        for start in range(0, arr.shape[0], chunk_size):
            stop = min(start + chunk_size, arr.shape[0])
            phi = self.transform(arr[start:stop], batch_size=chunk_size)
            self._feature_sum += weights[start:stop] @ phi
        self.n_samples += arr.shape[0]
        self.total_weight += float(np.sum(weights, dtype=np.float64))
        return self

    partial_fit = update

    def update_one(self, x: ArrayLike, *, weight: float = 1.0) -> "OnlineRFFKDE":
        return self.update(x, sample_weight=weight)

    def query_kernel_mean(
        self,
        X: ArrayLike,
        *,
        clip: bool = True,
        batch_size: Optional[int] = None,
    ) -> Union[float, NDArray[np.floating]]:
        """Estimate the mean Gaussian-kernel similarity to stored samples."""
        arr, was_single = self._validate_X(X)
        if self.total_weight <= 0:
            result = np.zeros(arr.shape[0], dtype=self.dtype)
        else:
            chunk_size = int(batch_size or self.default_batch_size)
            result = np.empty(arr.shape[0], dtype=self.dtype)
            for start in range(0, arr.shape[0], chunk_size):
                stop = min(start + chunk_size, arr.shape[0])
                phi = self.transform(arr[start:stop], batch_size=chunk_size)
                result[start:stop] = phi @ self._feature_sum / self.total_weight
        if clip:
            result = np.maximum(result, self.clip_min)
        return float(result[0]) if was_single else result

    def query_density(
        self,
        X: ArrayLike,
        *,
        clip: bool = True,
        batch_size: Optional[int] = None,
    ) -> Union[float, NDArray[np.floating]]:
        kernel_mean = self.query_kernel_mean(
            X,
            clip=False,
            batch_size=batch_size,
        )
        density = np.asarray(kernel_mean, dtype=self.dtype) * np.exp(
            -self._log_normalizer
        )
        if clip:
            density = np.maximum(density, self.clip_min)
        return float(density) if np.ndim(kernel_mean) == 0 else density

    density = query_density

    def query_log_density(
        self,
        X: ArrayLike,
        *,
        floor: float = 1e-300,
        batch_size: Optional[int] = None,
    ) -> Union[float, NDArray[np.floating]]:
        if floor <= 0:
            raise ValueError("floor must be strictly positive.")
        density = self.query_density(X, clip=True, batch_size=batch_size)
        log_density = np.log(np.maximum(density, floor))
        return float(log_density) if np.ndim(density) == 0 else log_density

    def score_then_update(
        self,
        x: ArrayLike,
        *,
        weight: float = 1.0,
        log_score: bool = False,
        log_floor: float = 1e-300,
    ) -> float:
        """Score a point against the old sketch before inserting it."""
        if log_score:
            value = -self.query_log_density(x, floor=log_floor)
        else:
            value = self.query_density(x)
        self.update_one(x, weight=weight)
        return float(value)

    @property
    def feature_sum(self) -> NDArray[np.floating]:
        return self._feature_sum.copy()

    @property
    def feature_mean(self) -> NDArray[np.floating]:
        if self.total_weight <= 0:
            return np.zeros_like(self._feature_sum)
        return self._feature_sum / self.total_weight

    @property
    def is_fitted(self) -> bool:
        return self.total_weight > 0

    @property
    def info(self) -> RFFKDEInfo:
        return RFFKDEInfo(
            input_dim=self.input_dim,
            n_features=self.n_features,
            bandwidth=self.bandwidth,
            n_samples=self.n_samples,
            total_weight=self.total_weight,
            clip_min=self.clip_min,
        )

    def reset(self) -> "OnlineRFFKDE":
        self._feature_sum.fill(0)
        self.n_samples = 0
        self.total_weight = 0.0
        return self

    def merge(self, other: "OnlineRFFKDE") -> "OnlineRFFKDE":
        if not isinstance(other, OnlineRFFKDE):
            raise TypeError("other must be an OnlineRFFKDE instance.")
        same_config = (
            self.input_dim == other.input_dim
            and self.n_features == other.n_features
            and self.bandwidth == other.bandwidth
            and self.clip_min == other.clip_min
            and self.dtype == other.dtype
        )
        same_map = (
            np.array_equal(self.omega, other.omega)
            and np.array_equal(self.phase, other.phase)
        )
        if not (same_config and same_map):
            raise ValueError("KDE sketches must use identical configurations and RFF maps.")
        self._feature_sum += other._feature_sum
        self.n_samples += other.n_samples
        self.total_weight += other.total_weight
        return self

    def state_dict(self) -> dict[str, Any]:
        """Return a pickle-safe state used by pipeline checkpoints."""
        return {
            "version": self.STATE_VERSION,
            "input_dim": self.input_dim,
            "n_features": self.n_features,
            "bandwidth": self.bandwidth,
            "clip_min": self.clip_min,
            "dtype": self.dtype.str,
            "default_batch_size": self.default_batch_size,
            "omega": self.omega.copy(),
            "phase": self.phase.copy(),
            "feature_sum": self._feature_sum.copy(),
            "n_samples": self.n_samples,
            "total_weight": self.total_weight,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "OnlineRFFKDE":
        def scalar(value: Any) -> Any:
            array = np.asarray(value)
            return array.item() if array.ndim == 0 else value

        version = int(scalar(state.get("version", 1)))
        if version != cls.STATE_VERSION:
            raise ValueError(f"Unsupported OnlineRFFKDE state version: {version}")
        obj = cls(
            input_dim=int(scalar(state["input_dim"])),
            n_features=int(scalar(state["n_features"])),
            bandwidth=float(scalar(state["bandwidth"])),
            random_state=0,
            clip_min=float(scalar(state["clip_min"])),
            dtype=np.dtype(scalar(state["dtype"])),
            default_batch_size=int(scalar(state["default_batch_size"])),
        )
        omega = np.asarray(state["omega"], dtype=obj.dtype)
        phase = np.asarray(state["phase"], dtype=obj.dtype)
        feature_sum = np.asarray(state["feature_sum"], dtype=obj.dtype)
        if omega.shape != (obj.n_features, obj.input_dim):
            raise ValueError(f"Invalid omega shape: {omega.shape}")
        if phase.shape != (obj.n_features,):
            raise ValueError(f"Invalid phase shape: {phase.shape}")
        if feature_sum.shape != (obj.n_features,):
            raise ValueError(f"Invalid feature_sum shape: {feature_sum.shape}")
        obj.omega = omega.copy()
        obj.phase = phase.copy()
        obj._feature_sum = feature_sum.copy()
        obj.n_samples = int(scalar(state["n_samples"]))
        obj.total_weight = float(scalar(state["total_weight"]))
        return obj

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.state_dict())

    @classmethod
    def load(cls, path: PathLike) -> "OnlineRFFKDE":
        with np.load(path, allow_pickle=False) as data:
            state = {key: data[key] for key in data.files}
        return cls.from_state_dict(state)

    def __len__(self) -> int:
        return self.n_samples
