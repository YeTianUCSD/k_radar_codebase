"""Dynamic context memory containing one online KDE per discovered scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .rff_kde import OnlineRFFKDE


@dataclass
class ContextEntry:
    context_id: int
    context_name: str
    model_context_name: str
    kde_x: OnlineRFFKDE
    density_threshold: float
    lifecycle: str = "stable"
    provisional_samples: int = 0
    provisional_frames: int = 0
    consecutive_mismatches: int = 0
    created_step: int = 0
    last_seen_step: int = 0
    model_update_enabled: bool = True
    calibration_scores: list[float] = field(default_factory=list)

    VALID_LIFECYCLES = {"provisional", "stable", "retired"}

    @property
    def num_samples(self) -> int:
        return self.kde_x.n_samples

    @property
    def threshold(self) -> float:
        """Backward-compatible alias for the calibrated density threshold."""
        return self.density_threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.density_threshold = float(value)

    @property
    def is_provisional(self) -> bool:
        return self.lifecycle == "provisional"

    @property
    def is_stable(self) -> bool:
        return self.lifecycle == "stable"

    @property
    def is_retired(self) -> bool:
        return self.lifecycle == "retired"


class ContextMemory:
    """Manage discovered contexts, lifecycles, and cumulative density sketches."""

    STATE_VERSION = 5

    def __init__(
        self,
        input_dim: int,
        *,
        n_rff_features: int = 2048,
        bandwidth: float = 1.0,
        random_state: int = 20260812,
        threshold_quantile: float = 0.05,
        update_threshold_quantile: Optional[float] = None,
        default_threshold: float = 0.1,
        calibration_history_size: int = 512,
        min_calibration_samples: int = 8,
        provisional_min_samples: int = 24,
        provisional_max_frames: int = 48,
        dtype: Any = np.float32,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if n_rff_features <= 0:
            raise ValueError("n_rff_features must be positive.")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if not 0.0 <= threshold_quantile <= 1.0:
            raise ValueError("threshold_quantile must be in [0, 1].")
        resolved_update_quantile = (
            threshold_quantile
            if update_threshold_quantile is None
            else float(update_threshold_quantile)
        )
        if not 0.0 <= resolved_update_quantile <= 1.0:
            raise ValueError("update_threshold_quantile must be in [0, 1].")
        if default_threshold < 0:
            raise ValueError("default_threshold must be nonnegative.")
        if calibration_history_size <= 0:
            raise ValueError("calibration_history_size must be positive.")
        if min_calibration_samples <= 0:
            raise ValueError("min_calibration_samples must be positive.")
        if provisional_min_samples <= 0:
            raise ValueError("provisional_min_samples must be positive.")
        if provisional_max_frames <= 0:
            raise ValueError("provisional_max_frames must be positive.")

        self.input_dim = int(input_dim)
        self.n_rff_features = int(n_rff_features)
        self.bandwidth = float(bandwidth)
        self.random_state = int(random_state)
        self.threshold_quantile = float(threshold_quantile)
        self.update_threshold_quantile = float(resolved_update_quantile)
        self.default_threshold = float(default_threshold)
        self.calibration_history_size = int(calibration_history_size)
        self.min_calibration_samples = int(min_calibration_samples)
        self.provisional_min_samples = int(provisional_min_samples)
        self.provisional_max_frames = int(provisional_max_frames)
        self.dtype = np.dtype(dtype)

        self.contexts: dict[int, ContextEntry] = {}
        self.active_context_id: Optional[int] = None
        self.next_context_id = 0

    def _validate_features(self, features: ArrayLike) -> np.ndarray:
        values = np.asarray(features, dtype=self.dtype)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected features with shape [N, {self.input_dim}]; "
                f"got {values.shape}."
            )
        if values.shape[0] == 0:
            raise ValueError("At least one feature is required.")
        if not np.all(np.isfinite(values)):
            raise ValueError("features contain NaN or infinite values.")
        return values

    @staticmethod
    def _validate_scale(scale: float, name: str) -> float:
        value = float(scale)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
        return value

    def _new_kde(self) -> OnlineRFFKDE:
        return OnlineRFFKDE(
            input_dim=self.input_dim,
            n_features=self.n_rff_features,
            bandwidth=self.bandwidth,
            random_state=self.random_state,
            dtype=self.dtype,
        )

    def _threshold_from_scores(self, scores: Sequence[float]) -> float:
        if len(scores) < self.min_calibration_samples:
            return self.default_threshold
        value = float(np.quantile(np.asarray(scores), self.threshold_quantile))
        return max(0.0, value)

    def _append_calibration_score(self, entry: ContextEntry, score: float) -> None:
        resolved_score = float(score)
        if not np.isfinite(resolved_score):
            raise ValueError("Calibration score must be finite.")
        entry.calibration_scores.append(resolved_score)
        if len(entry.calibration_scores) > self.calibration_history_size:
            del entry.calibration_scores[: -self.calibration_history_size]

    def _commit_update(
        self,
        entry: ContextEntry,
        values: np.ndarray,
        score: float,
        *,
        step: Optional[int],
        recalibrate: bool,
    ) -> float:
        self._append_calibration_score(entry, score)
        entry.kde_x.update(values)
        if recalibrate:
            entry.density_threshold = self._threshold_from_scores(
                entry.calibration_scores
            )
        if step is not None:
            entry.last_seen_step = int(step)
        return float(score)

    def propose_context_identity(
        self,
        *,
        context_name: Optional[str] = None,
        model_context_name: Optional[str] = None,
    ) -> tuple[int, str, str]:
        """Return the next identity without modifying the memory."""
        context_id = self.next_context_id
        resolved_name = context_name or (
            "normal" if context_id == 0 else f"context_{context_id:04d}"
        )
        resolved_model_name = model_context_name or resolved_name
        if any(entry.context_name == resolved_name for entry in self.contexts.values()):
            raise ValueError(f"Duplicate context_name: {resolved_name}")
        if any(
            entry.model_context_name == resolved_model_name
            for entry in self.contexts.values()
        ):
            raise ValueError(f"Duplicate model_context_name: {resolved_model_name}")
        return context_id, str(resolved_name), str(resolved_model_name)

    def create_context(
        self,
        initial_features: ArrayLike,
        *,
        context_name: Optional[str] = None,
        model_context_name: Optional[str] = None,
        created_step: int = 0,
        activate: bool = True,
        model_update_enabled: Optional[bool] = None,
        lifecycle: Optional[str] = None,
    ) -> ContextEntry:
        """Commit a stable base context or a provisional discovered context."""
        values = self._validate_features(initial_features)
        context_id, resolved_name, resolved_model_name = self.propose_context_identity(
            context_name=context_name,
            model_context_name=model_context_name,
        )
        resolved_lifecycle = str(
            lifecycle or ("stable" if context_id == 0 else "provisional")
        )
        if resolved_lifecycle not in {"stable", "provisional"}:
            raise ValueError(f"Unsupported context lifecycle for creation: {resolved_lifecycle}")
        if context_id == 0 and resolved_lifecycle != "stable":
            raise ValueError("The base normal context must be stable.")

        kde_x = self._new_kde()
        split = max(1, values.shape[0] // 2)
        kde_x.update(values[:split])
        calibration_scores: list[float] = []
        for value in values[split:]:
            score = float(kde_x.query_kernel_mean(value))
            calibration_scores.append(score)
            kde_x.update_one(value)
        density_threshold = self._threshold_from_scores(calibration_scores)
        entry = ContextEntry(
            context_id=context_id,
            context_name=resolved_name,
            model_context_name=resolved_model_name,
            kde_x=kde_x,
            density_threshold=density_threshold,
            lifecycle=resolved_lifecycle,
            provisional_samples=(
                int(values.shape[0]) if resolved_lifecycle == "provisional" else 0
            ),
            created_step=int(created_step),
            last_seen_step=int(created_step),
            model_update_enabled=(
                context_id != 0
                if model_update_enabled is None
                else bool(model_update_enabled)
            ),
            calibration_scores=calibration_scores[-self.calibration_history_size :],
        )
        self.contexts[context_id] = entry
        self.next_context_id += 1
        if activate:
            self.active_context_id = context_id
        return entry

    def score_all(self, features: ArrayLike) -> dict[int, float]:
        """Return mean pre-update similarity for every committed context."""
        values = self._validate_features(features)
        return {
            context_id: float(np.mean(entry.kde_x.query_kernel_mean(values)))
            for context_id, entry in self.contexts.items()
            if not entry.is_retired
        }

    def density_thresholds(self) -> dict[int, float]:
        return {
            context_id: float(entry.density_threshold)
            for context_id, entry in self.contexts.items()
            if not entry.is_retired
        }

    def thresholds(self) -> dict[int, float]:
        """Backward-compatible alias for calibrated density thresholds."""
        return self.density_thresholds()

    def _scaled_thresholds(self, scale: float, name: str) -> dict[int, float]:
        resolved_scale = self._validate_scale(scale, name)
        return {
            context_id: float(entry.density_threshold * resolved_scale)
            for context_id, entry in self.contexts.items()
            if not entry.is_retired
        }

    def routing_thresholds(self, scale: float = 0.85) -> dict[int, float]:
        return self._scaled_thresholds(scale, "routing threshold scale")

    def near_thresholds(self, scale: float = 0.60) -> dict[int, float]:
        return self._scaled_thresholds(scale, "near threshold scale")

    def update_thresholds(self, scale: float = 1.0) -> dict[int, float]:
        resolved_scale = self._validate_scale(scale, "update threshold scale")
        output = {}
        for context_id, entry in self.contexts.items():
            if entry.is_retired:
                continue
            if len(entry.calibration_scores) < self.min_calibration_samples:
                threshold = max(entry.density_threshold, self.default_threshold)
            else:
                threshold = float(np.quantile(
                    np.asarray(entry.calibration_scores),
                    self.update_threshold_quantile,
                ))
            output[context_id] = max(0.0, threshold) * resolved_scale
        return output

    def context_lifecycles(self) -> dict[int, str]:
        return {
            context_id: entry.lifecycle
            for context_id, entry in self.contexts.items()
            if not entry.is_retired
        }

    def update(
        self,
        context_id: int,
        features: ArrayLike,
        *,
        step: Optional[int] = None,
        pre_update_score: Optional[float] = None,
    ) -> float:
        """Backward-compatible unconditional update for an accepted observation."""
        entry = self.get(context_id)
        values = self._validate_features(features)
        score = (
            float(pre_update_score)
            if pre_update_score is not None
            else float(np.mean(entry.kde_x.query_kernel_mean(values)))
        )
        return self._commit_update(
            entry,
            values,
            score,
            step=step,
            recalibrate=True,
        )

    def update_provisional(
        self,
        context_id: int,
        features: ArrayLike,
        *,
        step: Optional[int] = None,
        pre_update_score: Optional[float] = None,
    ) -> bool:
        """Expand a provisional KDE before enforcing its final stable threshold."""
        entry = self.get(context_id)
        if not entry.is_provisional:
            raise ValueError(f"Context {context_id} is not provisional.")
        values = self._validate_features(features)
        score = (
            float(pre_update_score)
            if pre_update_score is not None
            else float(np.mean(entry.kde_x.query_kernel_mean(values)))
        )
        self._commit_update(
            entry,
            values,
            score,
            step=step,
            recalibrate=False,
        )
        entry.provisional_samples += int(values.shape[0])
        entry.provisional_frames += 1
        entry.consecutive_mismatches = 0
        if self.should_promote(context_id):
            self.promote_context(context_id)
            return True
        return False

    def mark_provisional_mismatch(self, context_id: int) -> int:
        entry = self.get(context_id)
        if not entry.is_provisional:
            raise ValueError(f"Context {context_id} is not provisional.")
        entry.provisional_frames += 1
        entry.consecutive_mismatches += 1
        return entry.consecutive_mismatches

    def should_promote(self, context_id: int) -> bool:
        entry = self.get(context_id)
        if not entry.is_provisional:
            return False
        return entry.provisional_samples >= self.provisional_min_samples

    def should_retire(self, context_id: int) -> bool:
        entry = self.get(context_id)
        if not entry.is_provisional:
            return False
        return (
            entry.provisional_frames >= self.provisional_max_frames
            and entry.provisional_samples < self.provisional_min_samples
        )

    def retire_context(
        self, context_id: int, *, fallback_context_id: int = 0
    ) -> ContextEntry:
        entry = self.get(context_id)
        if not entry.is_provisional:
            raise ValueError(f"Context {context_id} is not provisional.")
        fallback = self.get(fallback_context_id)
        if fallback.context_id == entry.context_id or fallback.is_retired:
            raise ValueError("Fallback context must be a different routable context.")
        entry.lifecycle = "retired"
        entry.model_update_enabled = False
        if self.active_context_id == entry.context_id:
            self.active_context_id = fallback.context_id
        return entry

    def promote_context(self, context_id: int) -> ContextEntry:
        entry = self.get(context_id)
        if not entry.is_provisional:
            raise ValueError(f"Context {context_id} is not provisional.")
        entry.density_threshold = self._threshold_from_scores(
            entry.calibration_scores
        )
        entry.lifecycle = "stable"
        entry.consecutive_mismatches = 0
        return entry

    def update_stable_if_confident(
        self,
        context_id: int,
        features: ArrayLike,
        *,
        threshold_scale: float = 1.0,
        step: Optional[int] = None,
        pre_update_score: Optional[float] = None,
    ) -> bool:
        """Update a stable KDE only when its stricter update gate is passed."""
        entry = self.get(context_id)
        if not entry.is_stable:
            raise ValueError(f"Context {context_id} is not stable.")
        values = self._validate_features(features)
        score = (
            float(pre_update_score)
            if pre_update_score is not None
            else float(np.mean(entry.kde_x.query_kernel_mean(values)))
        )
        scale = self._validate_scale(threshold_scale, "update threshold scale")
        if score < self.update_thresholds(scale)[context_id]:
            return False
        self._commit_update(
            entry,
            values,
            score,
            step=step,
            recalibrate=True,
        )
        return True

    def activate(self, context_id: int, *, step: Optional[int] = None) -> ContextEntry:
        entry = self.get(context_id)
        if entry.is_retired:
            raise ValueError(f"Context {context_id} is retired and cannot be activated.")
        self.active_context_id = entry.context_id
        if step is not None:
            entry.last_seen_step = int(step)
        return entry

    def remove_last_context(self, context_id: int) -> ContextEntry:
        """Remove only the newest context as part of a failed transaction rollback."""
        resolved_id = int(context_id)
        if resolved_id != self.next_context_id - 1 or resolved_id not in self.contexts:
            raise RuntimeError("Only the newest Context Memory entry can be removed.")
        entry = self.contexts.pop(resolved_id)
        self.next_context_id -= 1
        if self.active_context_id == resolved_id:
            self.active_context_id = None
        return entry

    def get(self, context_id: int) -> ContextEntry:
        resolved_id = int(context_id)
        if resolved_id not in self.contexts:
            raise KeyError(f"Unknown context_id: {resolved_id}")
        return self.contexts[resolved_id]

    @property
    def active(self) -> Optional[ContextEntry]:
        if self.active_context_id is None:
            return None
        return self.get(self.active_context_id)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "config": {
                "input_dim": self.input_dim,
                "n_rff_features": self.n_rff_features,
                "bandwidth": self.bandwidth,
                "random_state": self.random_state,
                "threshold_quantile": self.threshold_quantile,
                "update_threshold_quantile": self.update_threshold_quantile,
                "default_threshold": self.default_threshold,
                "calibration_history_size": self.calibration_history_size,
                "min_calibration_samples": self.min_calibration_samples,
                "provisional_min_samples": self.provisional_min_samples,
                "provisional_max_frames": self.provisional_max_frames,
                "dtype": self.dtype.str,
            },
            "active_context_id": self.active_context_id,
            "next_context_id": self.next_context_id,
            "contexts": [
                {
                    "context_id": entry.context_id,
                    "context_name": entry.context_name,
                    "model_context_name": entry.model_context_name,
                    "density_threshold": entry.density_threshold,
                    "lifecycle": entry.lifecycle,
                    "provisional_samples": entry.provisional_samples,
                    "provisional_frames": entry.provisional_frames,
                    "consecutive_mismatches": entry.consecutive_mismatches,
                    "created_step": entry.created_step,
                    "last_seen_step": entry.last_seen_step,
                    "model_update_enabled": entry.model_update_enabled,
                    "calibration_scores": list(entry.calibration_scores),
                    "kde_x": entry.kde_x.state_dict(),
                }
                for entry in self.contexts.values()
            ],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ContextMemory":
        version = int(state.get("version", 1))
        if version not in (1, 2, 3, 4, cls.STATE_VERSION):
            raise ValueError(f"Unsupported ContextMemory state version: {version}")
        config = dict(state["config"])
        obj = cls(**config)
        expected_id = 0
        for item in state.get("contexts", []):
            context_id = int(item["context_id"])
            if context_id != expected_id:
                raise ValueError(
                    "Context IDs must be contiguous and preserve creation order; "
                    f"expected {expected_id}, got {context_id}."
                )
            kde_x = OnlineRFFKDE.from_state_dict(item["kde_x"])
            expected_config = (
                obj.input_dim,
                obj.n_rff_features,
                obj.bandwidth,
                obj.dtype,
            )
            actual_config = (
                kde_x.input_dim,
                kde_x.n_features,
                kde_x.bandwidth,
                kde_x.dtype,
            )
            if actual_config != expected_config:
                raise ValueError(
                    "Context KDE configuration does not match Context Memory: "
                    f"expected={expected_config}, actual={actual_config}."
                )
            if obj.contexts:
                reference = next(iter(obj.contexts.values())).kde_x
                if not np.array_equal(kde_x.omega, reference.omega) or not np.array_equal(
                    kde_x.phase, reference.phase
                ):
                    raise ValueError("All Context KDEs must share the same RFF map.")

            threshold_key = (
                "density_threshold" if "density_threshold" in item else "threshold"
            )
            density_threshold = float(item[threshold_key])
            calibration_scores = [float(x) for x in item["calibration_scores"]]
            lifecycle = str(item.get("lifecycle", "stable"))
            if lifecycle not in ContextEntry.VALID_LIFECYCLES:
                raise ValueError(f"Unsupported context lifecycle: {lifecycle}")
            if context_id == 0 and lifecycle != "stable":
                raise ValueError("The base normal context must be stable.")
            if not np.isfinite(density_threshold) or density_threshold < 0:
                raise ValueError("Context threshold must be finite and nonnegative.")
            if not np.all(np.isfinite(calibration_scores)):
                raise ValueError("Context calibration scores must be finite.")
            if len(calibration_scores) > obj.calibration_history_size:
                raise ValueError("Context calibration history exceeds its configured size.")

            context_name = str(item["context_name"])
            model_context_name = str(item["model_context_name"])
            if any(entry.context_name == context_name for entry in obj.contexts.values()):
                raise ValueError(f"Duplicate context_name: {context_name}")
            if any(
                entry.model_context_name == model_context_name
                for entry in obj.contexts.values()
            ):
                raise ValueError(f"Duplicate model_context_name: {model_context_name}")

            provisional_samples = int(item.get("provisional_samples", 0))
            provisional_frames = int(item.get("provisional_frames", 0))
            consecutive_mismatches = int(item.get("consecutive_mismatches", 0))
            if min(
                provisional_samples,
                provisional_frames,
                consecutive_mismatches,
            ) < 0:
                raise ValueError("Context lifecycle counters must be nonnegative.")
            entry = ContextEntry(
                context_id=context_id,
                context_name=context_name,
                model_context_name=model_context_name,
                kde_x=kde_x,
                density_threshold=density_threshold,
                lifecycle=lifecycle,
                provisional_samples=provisional_samples,
                provisional_frames=provisional_frames,
                consecutive_mismatches=consecutive_mismatches,
                created_step=int(item["created_step"]),
                last_seen_step=int(item["last_seen_step"]),
                model_update_enabled=bool(
                    item.get("model_update_enabled", context_id != 0)
                ),
                calibration_scores=calibration_scores,
            )
            obj.contexts[context_id] = entry
            expected_id += 1

        obj.next_context_id = int(state.get("next_context_id", expected_id))
        if obj.next_context_id != expected_id:
            raise ValueError("next_context_id does not match the context manifest.")
        active_id = state.get("active_context_id", None)
        obj.active_context_id = None if active_id is None else int(active_id)
        if obj.active_context_id is not None:
            obj.get(obj.active_context_id)
            if obj.get(obj.active_context_id).is_retired:
                raise ValueError("The active context cannot be retired.")
        return obj

    def __len__(self) -> int:
        return len(self.contexts)
