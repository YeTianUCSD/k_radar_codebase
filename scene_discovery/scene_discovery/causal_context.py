"""Causal change detection and hierarchical context assignment.

The manager consumes descriptor vectors only.  It never receives sequence IDs,
split names, or ground-truth block boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CausalContextDecision:
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


class CausalContextManager:
    """Detect sustained departures, then reuse/create from a buffered segment.

    A stricter ``change_threshold`` starts a candidate transition.  After
    ``change_persistence`` consecutive departures, the entire buffered segment
    is compared with every context using a looser ``match_threshold``.  A match
    to the current context expands its representative memory without emitting a
    boundary; a different match emits REUSE; no match emits CREATE.
    """

    def __init__(
        self,
        change_threshold: float,
        match_threshold: float,
        change_persistence: int = 5,
        memory_size: int = 64,
        neighbors: int = 7,
        switch_margin: float = 0.05,
        update_threshold_ratio: float = 0.8,
        memory_min_separation_ratio: float = 0.1,
    ):
        if change_threshold <= 0.0 or match_threshold <= 0.0:
            raise ValueError("thresholds must be positive")
        if match_threshold < change_threshold:
            raise ValueError("match_threshold must be >= change_threshold")
        if change_persistence < 1 or memory_size < 1 or neighbors < 1:
            raise ValueError("persistence, memory size, and neighbors must be positive")
        if switch_margin < 0.0:
            raise ValueError("switch_margin must be non-negative")
        if not 0.0 < update_threshold_ratio <= 1.0:
            raise ValueError("update_threshold_ratio must be in (0, 1]")
        if not 0.0 <= memory_min_separation_ratio <= 1.0:
            raise ValueError("memory_min_separation_ratio must be in [0, 1]")
        self.change_threshold = float(change_threshold)
        self.match_threshold = float(match_threshold)
        self.change_persistence = int(change_persistence)
        self.memory_size = int(memory_size)
        self.neighbors = int(neighbors)
        self.switch_margin = float(switch_margin)
        self.update_threshold_ratio = float(update_threshold_ratio)
        self.memory_min_separation_ratio = float(memory_min_separation_ratio)
        self.memories: Dict[int, np.ndarray] = {}
        self.counts: Dict[int, int] = {}
        self.current_context: Optional[int] = None
        self._next_context_id = 0
        self._observation_index = -1
        self._candidate_start = -1
        self._candidate_buffer = []

    @property
    def prototypes(self) -> Dict[int, np.ndarray]:
        return {
            context: memory.mean(axis=0).astype(np.float32)
            for context, memory in self.memories.items()
        }

    def _validate_samples(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not len(values):
            raise ValueError("context samples must be a non-empty 2D array")
        if self.memories and values.shape[1] != next(iter(self.memories.values())).shape[1]:
            raise ValueError("context sample dimension mismatch")
        return values

    def _compress(self, values: np.ndarray) -> np.ndarray:
        if len(values) <= self.memory_size:
            return values.copy()
        positions = np.linspace(0, len(values) - 1, self.memory_size).round().astype(int)
        return values[positions].copy()

    def initialize_context(self, samples: np.ndarray, count: Optional[int] = None) -> int:
        values = self._validate_samples(samples)
        context_id = self._next_context_id
        self._next_context_id += 1
        self.memories[context_id] = self._compress(values)
        self.counts[context_id] = int(len(values) if count is None else count)
        self.current_context = context_id
        return context_id

    def _distance(self, value: np.ndarray, memory: np.ndarray) -> float:
        distances = np.linalg.norm(memory - value[None, :], axis=1)
        count = min(self.neighbors, len(memory))
        return float(np.sort(distances)[:count].mean())

    def _distances(self, value: np.ndarray):
        ids = np.asarray(sorted(self.memories), dtype=int)
        distances = np.asarray(
            [self._distance(value, self.memories[int(context)]) for context in ids],
            dtype=np.float64,
        )
        return ids, distances

    def _segment_scores(self, values: np.ndarray):
        ids = np.asarray(sorted(self.memories), dtype=int)
        scores = np.asarray([
            np.median([self._distance(value, self.memories[int(context)]) for value in values])
            for context in ids
        ], dtype=np.float64)
        return ids, scores

    def _add_representative(self, context_id: int, value: np.ndarray) -> None:
        memory = self.memories[context_id]
        separation = np.linalg.norm(memory - value[None, :], axis=1).min()
        if separation < self.change_threshold * self.memory_min_separation_ratio:
            return
        if len(memory) < self.memory_size:
            self.memories[context_id] = np.concatenate([memory, value[None, :]], axis=0)
            return
        pairwise = np.linalg.norm(memory[:, None, :] - memory[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        redundancy = pairwise.min(axis=1)
        replace = int(redundancy.argmin())
        if separation > redundancy[replace]:
            updated = memory.copy()
            updated[replace] = value
            self.memories[context_id] = updated

    def _remember(self, context_id: int, value: np.ndarray, distance: float) -> None:
        self.counts[context_id] += 1
        if distance <= self.change_threshold * self.update_threshold_ratio:
            self._add_representative(context_id, value)

    def _remember_segment(self, context_id: int, values: np.ndarray) -> None:
        for value in values:
            distance = self._distance(value, self.memories[context_id])
            self.counts[context_id] += 1
            if distance <= self.match_threshold:
                self._add_representative(context_id, value)

    def _clear_candidate(self) -> None:
        self._candidate_start = -1
        self._candidate_buffer = []

    def _decision(
        self,
        *,
        state: str,
        boundary: bool,
        estimated: int,
        action: str,
        previous: int,
        created: int,
        nearest: int,
        current_distance: float,
        nearest_distance: float,
        candidate_size: int,
    ) -> CausalContextDecision:
        return CausalContextDecision(
            observation_index=self._observation_index,
            context_id=int(self.current_context),
            state=state,
            boundary_event=bool(boundary),
            estimated_boundary_index=int(estimated),
            action=action,
            previous_context=int(previous),
            created_context=int(created),
            nearest_context=int(nearest),
            current_distance=float(current_distance),
            nearest_distance=float(nearest_distance),
            candidate_size=int(candidate_size),
        )

    def observe(self, value: np.ndarray) -> CausalContextDecision:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("observations must be 1D arrays")
        if not self.memories or self.current_context is None:
            raise RuntimeError("initialize_context must be called before observe")
        if value.shape[0] != next(iter(self.memories.values())).shape[1]:
            raise ValueError("observation dimension mismatch")
        self._observation_index += 1
        previous = int(self.current_context)
        ids, distances = self._distances(value)
        nearest_position = int(distances.argmin())
        nearest = int(ids[nearest_position])
        nearest_distance = float(distances[nearest_position])
        current_position = int(np.flatnonzero(ids == previous)[0])
        current_distance = float(distances[current_position])
        competitor_wins = (
            nearest != previous
            and nearest_distance <= self.match_threshold
            and nearest_distance + self.switch_margin < current_distance
        )
        departed = current_distance > self.change_threshold or competitor_wins

        if not departed:
            self._clear_candidate()
            self._remember(previous, value, current_distance)
            return self._decision(
                state="stable", boundary=False, estimated=-1, action="stay",
                previous=previous, created=-1, nearest=nearest,
                current_distance=current_distance, nearest_distance=nearest_distance,
                candidate_size=0,
            )

        if not self._candidate_buffer:
            self._candidate_start = self._observation_index
        self._candidate_buffer.append(value.copy())
        candidate_size = len(self._candidate_buffer)
        if candidate_size < self.change_persistence:
            return self._decision(
                state="suspect", boundary=False,
                estimated=self._candidate_start, action="pending",
                previous=previous, created=-1, nearest=nearest,
                current_distance=current_distance, nearest_distance=nearest_distance,
                candidate_size=candidate_size,
            )

        candidate = np.stack(self._candidate_buffer)
        candidate_start = self._candidate_start
        segment_ids, segment_scores = self._segment_scores(candidate)
        best_position = int(segment_scores.argmin())
        best_context = int(segment_ids[best_position])
        best_score = float(segment_scores[best_position])
        self._clear_candidate()

        if best_score <= self.match_threshold:
            self._remember_segment(best_context, candidate)
            if best_context == previous:
                return self._decision(
                    state="stable", boundary=False, estimated=candidate_start,
                    action="expand", previous=previous, created=-1,
                    nearest=best_context, current_distance=current_distance,
                    nearest_distance=best_score, candidate_size=candidate_size,
                )
            self.current_context = best_context
            return self._decision(
                state="stable", boundary=True, estimated=candidate_start,
                action="reuse", previous=previous, created=-1,
                nearest=best_context, current_distance=current_distance,
                nearest_distance=best_score, candidate_size=candidate_size,
            )

        created = self.initialize_context(candidate, count=candidate_size)
        return self._decision(
            state="stable", boundary=True, estimated=candidate_start,
            action="create", previous=previous, created=created,
            nearest=best_context, current_distance=current_distance,
            nearest_distance=best_score, candidate_size=candidate_size,
        )

    def state_dict(self):
        return {
            "kind": "causal_hierarchical_memory",
            "change_threshold": self.change_threshold,
            "match_threshold": self.match_threshold,
            "change_persistence": self.change_persistence,
            "memory_size": self.memory_size,
            "neighbors": self.neighbors,
            "switch_margin": self.switch_margin,
            "current_context": self.current_context,
            "counts": {str(key): int(value) for key, value in self.counts.items()},
            "memory_counts": {str(key): int(len(value)) for key, value in self.memories.items()},
            "prototypes": {
                str(key): value.mean(axis=0).astype(float).tolist()
                for key, value in self.memories.items()
            },
        }
