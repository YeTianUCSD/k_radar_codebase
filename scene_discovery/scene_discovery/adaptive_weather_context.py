"""Adaptive causal context manager for Seq1-only weather discovery.

The manager consumes feature vectors only.  Sequence boundaries and semantic
labels are deliberately absent from this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class AdaptiveWeatherDecision:
    observation_index: int
    context_id: int
    state: str
    boundary_event: bool
    estimated_boundary_index: int
    action: str
    previous_context: int
    created_context: int
    nearest_context: int
    current_distance: float
    nearest_distance: float
    candidate_size: int
    change_score: float
    current_radius: float
    best_coverage: float
    provisional_age: int
    cooldown_remaining: int


class AdaptiveWeatherContextManager:
    """Separate departure detection from delayed REUSE/CREATE commitment."""

    def __init__(
        self,
        change_threshold: float,
        match_threshold_ratio: float = 1.1,
        change_persistence: int = 4,
        candidate_min_windows: int = 8,
        candidate_max_windows: int = 24,
        provisional_windows: int = 4,
        cooldown_windows: int = 12,
        reference_windows: int = 32,
        recent_windows: int = 6,
        distribution_threshold_ratio: float = 0.6,
        strong_departure_ratio: float = 1.5,
        coverage_threshold: float = 0.7,
        match_margin: float = 0.1,
        context_quantile: float = 0.95,
        min_radius_ratio: float = 0.8,
        max_radius_ratio: float = 2.0,
        memory_size: int = 64,
        neighbors: int = 7,
        update_threshold_ratio: float = 0.8,
        memory_min_separation_ratio: float = 0.1,
    ):
        if change_threshold <= 0.0 or match_threshold_ratio < 1.0:
            raise ValueError("change threshold must be positive and match ratio >= 1")
        if not 1 <= change_persistence <= candidate_min_windows <= candidate_max_windows:
            raise ValueError("require persistence <= candidate_min <= candidate_max")
        if provisional_windows < 1 or cooldown_windows < 0:
            raise ValueError("provisional must be positive and cooldown non-negative")
        if recent_windows < 1 or reference_windows < recent_windows:
            raise ValueError("reference_windows must be >= recent_windows >= 1")
        if distribution_threshold_ratio <= 0.0 or strong_departure_ratio <= 1.0:
            raise ValueError("invalid departure ratios")
        if not 0.0 < coverage_threshold <= 1.0 or match_margin < 0.0:
            raise ValueError("invalid matching criteria")
        if not 0.5 <= context_quantile < 1.0:
            raise ValueError("context_quantile must be in [0.5, 1)")
        if not 0.0 < min_radius_ratio <= max_radius_ratio:
            raise ValueError("invalid adaptive radius bounds")
        if memory_size < 1 or neighbors < 1:
            raise ValueError("memory size and neighbors must be positive")
        if not 0.0 < update_threshold_ratio <= 1.0:
            raise ValueError("update_threshold_ratio must be in (0, 1]")
        if not 0.0 <= memory_min_separation_ratio <= 1.0:
            raise ValueError("memory_min_separation_ratio must be in [0, 1]")

        self.change_threshold = float(change_threshold)
        self.match_threshold_ratio = float(match_threshold_ratio)
        self.change_persistence = int(change_persistence)
        self.candidate_min_windows = int(candidate_min_windows)
        self.candidate_max_windows = int(candidate_max_windows)
        self.provisional_windows = int(provisional_windows)
        self.cooldown_windows = int(cooldown_windows)
        self.reference_windows = int(reference_windows)
        self.recent_windows = int(recent_windows)
        self.distribution_threshold_ratio = float(distribution_threshold_ratio)
        self.strong_departure_ratio = float(strong_departure_ratio)
        self.coverage_threshold = float(coverage_threshold)
        self.match_margin = float(match_margin)
        self.context_quantile = float(context_quantile)
        self.min_radius_ratio = float(min_radius_ratio)
        self.max_radius_ratio = float(max_radius_ratio)
        self.memory_size = int(memory_size)
        self.neighbors = int(neighbors)
        self.update_threshold_ratio = float(update_threshold_ratio)
        self.memory_min_separation_ratio = float(memory_min_separation_ratio)

        self.memories: Dict[int, np.ndarray] = {}
        self.counts: Dict[int, int] = {}
        self.distance_histories: Dict[int, Deque[float]] = {}
        self.current_context: Optional[int] = None
        self._next_context_id = 0
        self._observation_index = -1
        self._stable_history: Deque[np.ndarray] = deque(maxlen=self.reference_windows)
        self._candidate_start = -1
        self._candidate_buffer = []
        self._provisional_age = 0
        self._cooldown_remaining = 0
        self.provisional_entries = 0
        self.provisional_cancellations = 0

    @property
    def prototypes(self):
        return {
            context: memory.mean(axis=0).astype(np.float32)
            for context, memory in self.memories.items()
        }

    def _validate_samples(self, samples):
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not len(values):
            raise ValueError("context samples must be a non-empty 2D array")
        if self.memories and values.shape[1] != next(iter(self.memories.values())).shape[1]:
            raise ValueError("context sample dimension mismatch")
        return values

    def _compress(self, values):
        if len(values) <= self.memory_size:
            return values.copy()
        positions = np.linspace(0, len(values) - 1, self.memory_size).round().astype(int)
        return values[positions].copy()

    def _distance(self, value, memory):
        distances = np.linalg.norm(memory - value[None, :], axis=1)
        count = min(self.neighbors, len(memory))
        return float(np.sort(distances)[:count].mean())

    def _initial_distance_history(self, values):
        if len(values) < 2:
            return []
        pairwise = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        count = min(self.neighbors, len(values) - 1)
        return np.sort(pairwise, axis=1)[:, :count].mean(axis=1).astype(float).tolist()

    def initialize_context(self, samples, count=None):
        values = self._validate_samples(samples)
        context = self._next_context_id
        self._next_context_id += 1
        memory = self._compress(values)
        self.memories[context] = memory
        self.counts[context] = int(len(values) if count is None else count)
        self.distance_histories[context] = deque(
            self._initial_distance_history(memory), maxlen=max(64, self.memory_size * 4)
        )
        self.current_context = context
        self._stable_history.clear()
        for value in memory[-self.reference_windows:]:
            self._stable_history.append(value.copy())
        return context

    def _context_radius(self, context):
        history = np.asarray(self.distance_histories[context], dtype=np.float64)
        adaptive = (
            float(np.quantile(history, self.context_quantile))
            if len(history) else self.change_threshold
        )
        lower = self.change_threshold * self.min_radius_ratio
        upper = self.change_threshold * self.max_radius_ratio
        return float(np.clip(adaptive, lower, upper))

    def _all_distances(self, value):
        ids = np.asarray(sorted(self.memories), dtype=int)
        values = np.asarray([
            self._distance(value, self.memories[int(context)]) for context in ids
        ], dtype=np.float64)
        return ids, values

    def _change_score(self, value):
        history = list(self._stable_history)
        if len(history) < self.recent_windows:
            return 0.0
        recent = np.stack((history + [value])[-self.recent_windows:]).mean(axis=0)
        reference_values = history[:-self.recent_windows]
        if not reference_values:
            reference_values = history
        reference = np.stack(reference_values).mean(axis=0)
        return float(np.linalg.norm(recent - reference))

    def _segment_statistics(self, values):
        contexts = sorted(self.memories)
        rows = []
        for context in contexts:
            distances = np.asarray([
                self._distance(value, self.memories[context]) for value in values
            ])
            radius = self._context_radius(context) * self.match_threshold_ratio
            rows.append((
                context,
                float(np.median(distances)),
                float(np.quantile(distances, 0.9)),
                float(np.mean(distances <= radius)),
                radius,
            ))
        return rows

    def _best_segment_match(self, values):
        rows = self._segment_statistics(values)
        ranked = sorted(rows, key=lambda row: (-row[3], row[1]))
        best = ranked[0]
        second_coverage = ranked[1][3] if len(ranked) > 1 else 0.0
        second_distance = ranked[1][1] if len(ranked) > 1 else float("inf")
        separated = (
            best[3] - second_coverage >= self.match_margin
            or best[1] + self.change_threshold * self.match_margin < second_distance
        )
        confident = (
            best[3] >= self.coverage_threshold
            and best[1] <= best[4]
            and (len(ranked) == 1 or separated)
        )
        return best, confident

    def _candidate_is_compact(self, values):
        centroid = values.mean(axis=0)
        dispersion = np.linalg.norm(values - centroid[None, :], axis=1)
        return float(np.quantile(dispersion, 0.9)) <= (
            self.change_threshold * self.max_radius_ratio
        )

    def _add_representative(self, context, value):
        memory = self.memories[context]
        separation = np.linalg.norm(memory - value[None, :], axis=1).min()
        if separation < self.change_threshold * self.memory_min_separation_ratio:
            return
        if len(memory) < self.memory_size:
            self.memories[context] = np.concatenate([memory, value[None, :]], axis=0)
            return
        pairwise = np.linalg.norm(memory[:, None, :] - memory[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        redundancy = pairwise.min(axis=1)
        replace = int(redundancy.argmin())
        if separation > redundancy[replace]:
            updated = memory.copy()
            updated[replace] = value
            self.memories[context] = updated

    def _remember(self, context, value, distance=None):
        distance = self._distance(value, self.memories[context]) if distance is None else float(distance)
        self.counts[context] += 1
        radius = self._context_radius(context)
        if distance <= radius * self.match_threshold_ratio:
            self.distance_histories[context].append(distance)
        if distance <= radius * self.update_threshold_ratio:
            self._add_representative(context, value)

    def _remember_segment(self, context, values):
        for value in values:
            self._remember(context, value)

    def _clear_candidate(self):
        self._candidate_start = -1
        self._candidate_buffer = []
        self._provisional_age = 0

    def _decision(self, state, boundary, estimated, action, previous, created,
                  nearest, current_distance, nearest_distance, candidate_size,
                  change_score, current_radius, best_coverage=0.0):
        return AdaptiveWeatherDecision(
            observation_index=self._observation_index,
            context_id=int(self.current_context), state=state,
            boundary_event=bool(boundary), estimated_boundary_index=int(estimated),
            action=action, previous_context=int(previous), created_context=int(created),
            nearest_context=int(nearest), current_distance=float(current_distance),
            nearest_distance=float(nearest_distance), candidate_size=int(candidate_size),
            change_score=float(change_score), current_radius=float(current_radius),
            best_coverage=float(best_coverage), provisional_age=int(self._provisional_age),
            cooldown_remaining=int(self._cooldown_remaining),
        )

    def observe(self, value):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("observations must be 1D vectors")
        if not self.memories or self.current_context is None:
            raise RuntimeError("initialize_context must be called before observe")
        if value.shape[0] != next(iter(self.memories.values())).shape[1]:
            raise ValueError("observation dimension mismatch")

        self._observation_index += 1
        previous = int(self.current_context)
        ids, distances = self._all_distances(value)
        nearest_position = int(distances.argmin())
        nearest = int(ids[nearest_position])
        nearest_distance = float(distances[nearest_position])
        current_distance = float(distances[np.flatnonzero(ids == previous)[0]])
        current_radius = self._context_radius(previous)
        change_score = self._change_score(value)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._remember(previous, value, current_distance)
            self._stable_history.append(value.copy())
            return self._decision(
                "cooldown", False, -1, "cooldown", previous, -1, nearest,
                current_distance, nearest_distance, 0, change_score, current_radius,
            )

        competitor_radius = self._context_radius(nearest) * self.match_threshold_ratio
        competitor_wins = (
            nearest != previous and nearest_distance <= competitor_radius
            and nearest_distance + self.change_threshold * self.match_margin < current_distance
        )
        distribution_shift = (
            change_score > self.change_threshold * self.distribution_threshold_ratio
        )
        strong_departure = current_distance > current_radius * self.strong_departure_ratio
        departed = competitor_wins or (
            current_distance > current_radius and (distribution_shift or strong_departure)
        )

        if not self._candidate_buffer and not departed:
            self._remember(previous, value, current_distance)
            self._stable_history.append(value.copy())
            return self._decision(
                "stable", False, -1, "stay", previous, -1, nearest,
                current_distance, nearest_distance, 0, change_score, current_radius,
            )

        if not self._candidate_buffer:
            self._candidate_start = self._observation_index
        self._candidate_buffer.append(value.copy())
        candidate_size = len(self._candidate_buffer)
        candidate = np.stack(self._candidate_buffer)

        if candidate_size < self.change_persistence:
            return self._decision(
                "suspect", False, self._candidate_start, "pending", previous, -1,
                nearest, current_distance, nearest_distance, candidate_size,
                change_score, current_radius,
            )

        best, confident = self._best_segment_match(candidate)
        best_context, best_distance, _, best_coverage, _ = best
        if confident and best_context == previous:
            self.provisional_cancellations += int(self._provisional_age > 0)
            self._remember_segment(previous, candidate)
            for sample in candidate[-self.reference_windows:]:
                self._stable_history.append(sample.copy())
            start = self._candidate_start
            self._clear_candidate()
            return self._decision(
                "stable", False, start, "cancel", previous, -1, best_context,
                current_distance, best_distance, candidate_size, change_score,
                current_radius, best_coverage,
            )

        if candidate_size < self.candidate_min_windows:
            return self._decision(
                "collecting", False, self._candidate_start, "collect", previous, -1,
                best_context, current_distance, best_distance, candidate_size,
                change_score, current_radius, best_coverage,
            )

        if confident:
            start = self._candidate_start
            self._remember_segment(best_context, candidate)
            self.current_context = best_context
            self._cooldown_remaining = self.cooldown_windows
            self._stable_history.clear()
            for sample in candidate[-self.reference_windows:]:
                self._stable_history.append(sample.copy())
            self._clear_candidate()
            return self._decision(
                "cooldown", True, start, "reuse", previous, -1, best_context,
                current_distance, best_distance, candidate_size, change_score,
                current_radius, best_coverage,
            )

        if self._provisional_age == 0:
            self.provisional_entries += 1
        self._provisional_age += 1
        compact = self._candidate_is_compact(candidate)
        should_create = compact and self._provisional_age >= self.provisional_windows
        forced = candidate_size >= self.candidate_max_windows
        if not should_create and not forced:
            return self._decision(
                "provisional", False, self._candidate_start, "provisional", previous,
                -1, best_context, current_distance, best_distance, candidate_size,
                change_score, current_radius, best_coverage,
            )

        if forced and not compact:
            self.provisional_cancellations += 1
            start = self._candidate_start
            self._clear_candidate()
            self._stable_history.append(value.copy())
            return self._decision(
                "stable", False, start, "reject_unstable", previous, -1,
                best_context, current_distance, best_distance, candidate_size,
                change_score, current_radius, best_coverage,
            )

        start = self._candidate_start
        created = self.initialize_context(candidate, count=candidate_size)
        self._cooldown_remaining = self.cooldown_windows
        self._clear_candidate()
        return self._decision(
            "cooldown", True, start, "create", previous, created, best_context,
            current_distance, best_distance, candidate_size, change_score,
            current_radius, best_coverage,
        )

    def state_dict(self):
        return {
            "kind": "adaptive_weather_context_v51",
            "change_threshold": self.change_threshold,
            "match_threshold_ratio": self.match_threshold_ratio,
            "change_persistence": self.change_persistence,
            "candidate_min_windows": self.candidate_min_windows,
            "candidate_max_windows": self.candidate_max_windows,
            "provisional_windows": self.provisional_windows,
            "cooldown_windows": self.cooldown_windows,
            "reference_windows": self.reference_windows,
            "recent_windows": self.recent_windows,
            "distribution_threshold_ratio": self.distribution_threshold_ratio,
            "coverage_threshold": self.coverage_threshold,
            "current_context": self.current_context,
            "provisional_entries": self.provisional_entries,
            "provisional_cancellations": self.provisional_cancellations,
            "counts": {str(key): int(value) for key, value in self.counts.items()},
            "memory_counts": {str(key): int(len(value)) for key, value in self.memories.items()},
            "context_radii": {
                str(key): self._context_radius(key) for key in self.memories
            },
            "prototypes": {
                str(key): value.mean(axis=0).astype(float).tolist()
                for key, value in self.memories.items()
            },
        }
