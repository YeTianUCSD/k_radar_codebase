from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class AgglomerativeCentroidPredictor:
    """Serializable out-of-sample predictor for agglomerative clustering."""

    estimator: object
    cluster_ids: np.ndarray
    centroids: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        centroids = np.asarray(self.centroids, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"features must be a 2D array, got shape {values.shape}")
        if centroids.ndim != 2 or values.shape[1] != centroids.shape[1]:
            saved_dimension = centroids.shape[1] if centroids.ndim == 2 else "invalid"
            raise ValueError(
                "feature dimension does not match saved centroids: "
                f"{values.shape[1]} != {saved_dimension}"
            )
        distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return np.asarray(self.cluster_ids)[distances.argmin(axis=1)]


@dataclass
class ContextDecision:
    context_id: int
    is_new: bool
    accepted_distance: float
    nearest_context: int
    nearest_distance: float
    acceptance_radius: float = 0.0
    candidate_size: int = 0


class StreamingContextManager:
    """DP-Means-style online context discovery with persistence and merging.

    A sample must be outside every context-specific radius before it can enter a
    candidate buffer. Candidate samples must also remain mutually consistent.
    Context radii adapt from accepted-distance statistics, and nearby prototypes
    are merged while retaining canonical context IDs.
    """

    def __init__(
        self,
        threshold: float,
        create_persistence: int = 5,
        switch_persistence: int = 3,
        prototype_alpha: float = 0.02,
        candidate_threshold: Optional[float] = None,
        merge_threshold: Optional[float] = None,
        radius_alpha: float = 0.05,
        radius_std_scale: float = 3.0,
        radius_min_scale: float = 0.75,
        radius_max_scale: float = 1.25,
        radius_warmup: int = 10,
    ):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if create_persistence < 1 or switch_persistence < 1:
            raise ValueError("persistence values must be positive")
        if not 0.0 < prototype_alpha <= 1.0:
            raise ValueError("prototype_alpha must be in (0, 1]")
        if not 0.0 < radius_alpha <= 1.0:
            raise ValueError("radius_alpha must be in (0, 1]")
        if radius_std_scale < 0:
            raise ValueError("radius_std_scale must be non-negative")
        if not 0.0 < radius_min_scale <= radius_max_scale:
            raise ValueError("invalid radius scale bounds")
        if radius_warmup < 1:
            raise ValueError("radius_warmup must be positive")

        self.threshold = float(threshold)
        self.create_persistence = int(create_persistence)
        self.switch_persistence = int(switch_persistence)
        self.prototype_alpha = float(prototype_alpha)
        self.candidate_threshold = float(
            candidate_threshold if candidate_threshold is not None else threshold * 0.75
        )
        self.merge_threshold = float(
            merge_threshold if merge_threshold is not None else threshold * 0.65
        )
        if self.candidate_threshold <= 0 or self.merge_threshold <= 0:
            raise ValueError("candidate and merge thresholds must be positive")
        self.radius_alpha = float(radius_alpha)
        self.radius_std_scale = float(radius_std_scale)
        self.radius_min_scale = float(radius_min_scale)
        self.radius_max_scale = float(radius_max_scale)
        self.radius_warmup = int(radius_warmup)

        self.prototypes: Dict[int, np.ndarray] = {}
        self.counts: Dict[int, int] = {}
        self.radii: Dict[int, float] = {}
        self.distance_counts: Dict[int, int] = {}
        self.distance_means: Dict[int, float] = {}
        self.distance_m2: Dict[int, float] = {}
        self.aliases: Dict[int, int] = {}
        self.current_context: Optional[int] = None
        self._unknown_buffer: List[np.ndarray] = []
        self._pending_context: Optional[int] = None
        self._pending_count = 0
        self._next_context_id = 0

    def canonical_id(self, context_id: int) -> int:
        context_id = int(context_id)
        trail = []
        while context_id in self.aliases:
            trail.append(context_id)
            context_id = int(self.aliases[context_id])
        for alias in trail:
            self.aliases[alias] = context_id
        return context_id

    def canonicalize(self, context_ids: Iterable[int]) -> np.ndarray:
        return np.asarray([self.canonical_id(value) for value in context_ids], dtype=int)

    def _reset_pending(self) -> None:
        self._pending_context = None
        self._pending_count = 0

    def _set_distance_statistics(self, context_id: int, distances: np.ndarray) -> None:
        values = np.asarray(distances, dtype=np.float64).reshape(-1)
        if not len(values):
            self.distance_counts[context_id] = 0
            self.distance_means[context_id] = 0.0
            self.distance_m2[context_id] = 0.0
            return
        self.distance_counts[context_id] = int(len(values))
        self.distance_means[context_id] = float(values.mean())
        self.distance_m2[context_id] = float(((values - values.mean()) ** 2).sum())

    def initialize_context(
        self,
        prototype: np.ndarray,
        count: int = 1,
        distances: Optional[np.ndarray] = None,
    ) -> int:
        value = np.asarray(prototype, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("prototype must be a 1D array")
        if self.prototypes and value.shape != next(iter(self.prototypes.values())).shape:
            raise ValueError("prototype dimension mismatch")
        context_id = self._next_context_id
        self._next_context_id += 1
        self.prototypes[context_id] = value.copy()
        self.counts[context_id] = max(1, int(count))
        self.radii[context_id] = self.threshold
        self._set_distance_statistics(
            context_id,
            np.empty(0, dtype=np.float32) if distances is None else distances,
        )
        self.current_context = context_id
        return context_id

    def _adaptive_radius(self, context_id: int) -> float:
        count = self.distance_counts.get(context_id, 0)
        if count < self.radius_warmup:
            return self.radii.get(context_id, self.threshold)
        mean = self.distance_means[context_id]
        variance = self.distance_m2[context_id] / max(1, count - 1)
        proposed = mean + self.radius_std_scale * float(np.sqrt(max(variance, 0.0)))
        lower = self.threshold * self.radius_min_scale
        upper = self.threshold * self.radius_max_scale
        return float(np.clip(proposed, lower, upper))

    def _register_distance(self, context_id: int, distance: float) -> None:
        count = self.distance_counts.get(context_id, 0) + 1
        old_mean = self.distance_means.get(context_id, 0.0)
        delta = distance - old_mean
        new_mean = old_mean + delta / count
        new_m2 = self.distance_m2.get(context_id, 0.0) + delta * (distance - new_mean)
        self.distance_counts[context_id] = count
        self.distance_means[context_id] = float(new_mean)
        self.distance_m2[context_id] = float(new_m2)
        target = self._adaptive_radius(context_id)
        old_radius = self.radii.get(context_id, self.threshold)
        self.radii[context_id] = float(
            (1.0 - self.radius_alpha) * old_radius + self.radius_alpha * target
        )

    def _update(self, context_id: int, value: np.ndarray, distance: float) -> int:
        context_id = self.canonical_id(context_id)
        alpha = self.prototype_alpha
        self.prototypes[context_id] = (
            (1.0 - alpha) * self.prototypes[context_id] + alpha * value
        ).astype(np.float32)
        self.counts[context_id] += 1
        self._register_distance(context_id, float(distance))
        return self._merge_nearby(context_id)

    def _combine_statistics(self, keep: int, drop: int) -> None:
        n1 = self.distance_counts.get(keep, 0)
        n2 = self.distance_counts.get(drop, 0)
        if not n2:
            return
        if not n1:
            self.distance_counts[keep] = n2
            self.distance_means[keep] = self.distance_means.get(drop, 0.0)
            self.distance_m2[keep] = self.distance_m2.get(drop, 0.0)
            return
        mean1 = self.distance_means[keep]
        mean2 = self.distance_means[drop]
        total = n1 + n2
        delta = mean2 - mean1
        self.distance_counts[keep] = total
        self.distance_means[keep] = mean1 + delta * n2 / total
        self.distance_m2[keep] = (
            self.distance_m2[keep]
            + self.distance_m2[drop]
            + delta * delta * n1 * n2 / total
        )

    def _merge_pair(self, first: int, second: int) -> int:
        first = self.canonical_id(first)
        second = self.canonical_id(second)
        if first == second:
            return first
        keep, drop = sorted((first, second))
        keep_count = self.counts[keep]
        drop_count = self.counts[drop]
        total = keep_count + drop_count
        self.prototypes[keep] = (
            (self.prototypes[keep] * keep_count + self.prototypes[drop] * drop_count) / total
        ).astype(np.float32)
        self.counts[keep] = total
        self._combine_statistics(keep, drop)
        self.radii[keep] = max(self.radii.get(keep, self.threshold), self.radii.get(drop, self.threshold))
        self.aliases[drop] = keep
        for alias, target in list(self.aliases.items()):
            self.aliases[alias] = self.canonical_id(target)
        for store in (
            self.prototypes,
            self.counts,
            self.radii,
            self.distance_counts,
            self.distance_means,
            self.distance_m2,
        ):
            store.pop(drop, None)
        if self.current_context is not None:
            self.current_context = self.canonical_id(self.current_context)
        if self._pending_context is not None:
            self._pending_context = self.canonical_id(self._pending_context)
        return keep

    def _merge_nearby(self, context_id: int) -> int:
        context_id = self.canonical_id(context_id)
        while len(self.prototypes) > 1:
            others = [key for key in self.prototypes if key != context_id]
            distances = np.asarray(
                [np.linalg.norm(self.prototypes[key] - self.prototypes[context_id]) for key in others]
            )
            position = int(distances.argmin())
            if float(distances[position]) > self.merge_threshold:
                break
            context_id = self._merge_pair(context_id, others[position])
        return context_id

    def _create(self, values: List[np.ndarray]) -> tuple:
        stacked = np.stack(values).astype(np.float32)
        prototype = stacked.mean(axis=0).astype(np.float32)
        context_id = self.initialize_context(
            prototype,
            count=len(values),
            distances=np.linalg.norm(stacked - prototype[None, :], axis=1),
        )
        merged_id = self._merge_nearby(context_id)
        is_new = merged_id == context_id
        self.current_context = merged_id
        self._unknown_buffer.clear()
        self._reset_pending()
        return merged_id, is_new

    def _append_candidate(self, value: np.ndarray) -> None:
        if not self._unknown_buffer:
            self._unknown_buffer.append(value.copy())
            return
        candidate = np.stack(self._unknown_buffer).mean(axis=0)
        if float(np.linalg.norm(candidate - value)) <= self.candidate_threshold:
            self._unknown_buffer.append(value.copy())
        else:
            self._unknown_buffer = [value.copy()]

    def observe(self, value: np.ndarray) -> ContextDecision:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("observations must be 1D arrays")
        if not self.prototypes:
            context_id, _ = self._create([value])
            return ContextDecision(context_id, True, 0.0, context_id, 0.0, self.threshold, 0)
        if value.shape != next(iter(self.prototypes.values())).shape:
            raise ValueError("observation dimension mismatch")

        ids = np.asarray(sorted(self.prototypes))
        prototypes = np.stack([self.prototypes[int(key)] for key in ids])
        distances = np.linalg.norm(prototypes - value[None, :], axis=1)
        nearest_position = int(distances.argmin())
        nearest = int(ids[nearest_position])
        nearest_distance = float(distances[nearest_position])
        acceptance_radius = float(self.radii.get(nearest, self.threshold))

        if nearest_distance > acceptance_radius:
            self._append_candidate(value)
            self._reset_pending()
            candidate_size = len(self._unknown_buffer)
            if candidate_size >= self.create_persistence:
                context_id, is_new = self._create(self._unknown_buffer.copy())
                return ContextDecision(
                    context_id, is_new, nearest_distance, nearest,
                    nearest_distance, acceptance_radius, candidate_size,
                )
            return ContextDecision(
                int(self.current_context), False, nearest_distance, nearest,
                nearest_distance, acceptance_radius, candidate_size,
            )

        self._unknown_buffer.clear()
        if nearest == self.current_context:
            self._reset_pending()
            nearest = self._update(nearest, value, nearest_distance)
            self.current_context = nearest
            return ContextDecision(
                nearest, False, nearest_distance, nearest,
                nearest_distance, acceptance_radius, 0,
            )

        if self._pending_context == nearest:
            self._pending_count += 1
        else:
            self._pending_context = nearest
            self._pending_count = 1
        if self._pending_count >= self.switch_persistence:
            self.current_context = nearest
            self._reset_pending()
            nearest = self._update(nearest, value, nearest_distance)
            self.current_context = nearest
        return ContextDecision(
            int(self.current_context), False, nearest_distance, nearest,
            nearest_distance, acceptance_radius, 0,
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "threshold": self.threshold,
            "candidate_threshold": self.candidate_threshold,
            "merge_threshold": self.merge_threshold,
            "current_context": self.current_context,
            "prototypes": {str(k): v.tolist() for k, v in self.prototypes.items()},
            "counts": {str(k): int(v) for k, v in self.counts.items()},
            "radii": {str(k): float(v) for k, v in self.radii.items()},
            "distance_counts": {str(k): int(v) for k, v in self.distance_counts.items()},
            "distance_means": {str(k): float(v) for k, v in self.distance_means.items()},
            "distance_variances": {
                str(k): float(self.distance_m2[k] / max(1, self.distance_counts[k] - 1))
                for k in self.distance_counts
            },
            "aliases": {str(k): int(v) for k, v in self.aliases.items()},
        }
