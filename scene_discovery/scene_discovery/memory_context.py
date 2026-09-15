"""Online context registry backed by bounded representative memories."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .clustering import ContextDecision


class MemoryBankContextManager:
    """Create/reuse contexts using absolute nearest-memory distance and persistence."""

    def __init__(
        self,
        threshold: float,
        create_persistence: int = 5,
        switch_persistence: int = 3,
        candidate_threshold: Optional[float] = None,
        memory_size: int = 64,
        neighbors: int = 7,
        update_threshold_ratio: float = 0.8,
        memory_min_separation_ratio: float = 0.1,
    ):
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if create_persistence < 1 or switch_persistence < 1:
            raise ValueError("persistence values must be positive")
        if memory_size < 1:
            raise ValueError("memory_size must be positive")
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        if not 0.0 < update_threshold_ratio <= 1.0:
            raise ValueError("update_threshold_ratio must be in (0, 1]")
        if not 0.0 <= memory_min_separation_ratio <= 1.0:
            raise ValueError("memory_min_separation_ratio must be in [0, 1]")
        self.threshold = float(threshold)
        self.create_persistence = int(create_persistence)
        self.switch_persistence = int(switch_persistence)
        self.candidate_threshold = float(
            candidate_threshold
            if candidate_threshold is not None else threshold * 0.75
        )
        self.memory_size = int(memory_size)
        self.neighbors = int(neighbors)
        self.update_threshold_ratio = float(update_threshold_ratio)
        self.memory_min_separation_ratio = float(memory_min_separation_ratio)
        self.memories: Dict[int, np.ndarray] = {}
        self.counts: Dict[int, int] = {}
        self.current_context: Optional[int] = None
        self._unknown_buffer = []
        self._pending_context: Optional[int] = None
        self._pending_count = 0
        self._next_context_id = 0

    @property
    def prototypes(self):
        return {
            context: memory.mean(axis=0).astype(np.float32)
            for context, memory in self.memories.items()
        }

    @property
    def aliases(self):
        return {}

    def canonical_id(self, context_id: int) -> int:
        return int(context_id)

    def canonicalize(self, context_ids: Sequence[int]) -> np.ndarray:
        return np.asarray(context_ids, dtype=int)

    def _validate_samples(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not len(values):
            raise ValueError("context samples must be a non-empty 2D array")
        if self.memories and values.shape[1] != next(iter(self.memories.values())).shape[1]:
            raise ValueError("context sample dimension mismatch")
        return values

    def _compress_initial(self, values: np.ndarray) -> np.ndarray:
        if len(values) <= self.memory_size:
            return values.copy()
        positions = np.linspace(0, len(values) - 1, self.memory_size).round().astype(int)
        return values[positions].copy()

    def initialize_context(self, samples: np.ndarray, count: Optional[int] = None) -> int:
        values = self._validate_samples(samples)
        context_id = self._next_context_id
        self._next_context_id += 1
        self.memories[context_id] = self._compress_initial(values)
        self.counts[context_id] = int(len(values) if count is None else count)
        self.current_context = context_id
        return context_id

    def _context_distances(self, value: np.ndarray):
        ids = np.asarray(sorted(self.memories), dtype=int)
        distances = np.asarray([
            np.sort(np.linalg.norm(memory - value[None, :], axis=1))[
                : min(self.neighbors, len(memory))
            ].mean()
            for memory in (self.memories[int(context)] for context in ids)
        ], dtype=np.float64)
        return ids, distances

    def _reset_pending(self) -> None:
        self._pending_context = None
        self._pending_count = 0

    def _append_candidate(self, value: np.ndarray) -> None:
        if not self._unknown_buffer:
            self._unknown_buffer = [value.copy()]
            return
        center = np.stack(self._unknown_buffer).mean(axis=0)
        if np.linalg.norm(value - center) <= self.candidate_threshold:
            self._unknown_buffer.append(value.copy())
        else:
            self._unknown_buffer = [value.copy()]

    def _remember(self, context_id: int, value: np.ndarray, distance: float) -> None:
        self.counts[context_id] += 1
        if distance > self.threshold * self.update_threshold_ratio:
            return
        memory = self.memories[context_id]
        separation = np.linalg.norm(memory - value[None, :], axis=1).min()
        if separation < self.threshold * self.memory_min_separation_ratio:
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

    def observe(self, value: np.ndarray) -> ContextDecision:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("observations must be 1D arrays")
        if not self.memories:
            context = self.initialize_context(value)
            return ContextDecision(context, True, 0.0, context, 0.0, self.threshold, 0)
        if value.shape[0] != next(iter(self.memories.values())).shape[1]:
            raise ValueError("observation dimension mismatch")

        ids, distances = self._context_distances(value)
        position = int(distances.argmin())
        nearest = int(ids[position])
        nearest_distance = float(distances[position])
        if nearest_distance > self.threshold:
            self._append_candidate(value)
            self._reset_pending()
            size = len(self._unknown_buffer)
            if size >= self.create_persistence:
                context = self.initialize_context(np.stack(self._unknown_buffer))
                self._unknown_buffer.clear()
                return ContextDecision(
                    context, True, nearest_distance, nearest,
                    nearest_distance, self.threshold, size,
                )
            return ContextDecision(
                int(self.current_context), False, nearest_distance, nearest,
                nearest_distance, self.threshold, size,
            )

        self._unknown_buffer.clear()
        if nearest == self.current_context:
            self._reset_pending()
            self._remember(nearest, value, nearest_distance)
            return ContextDecision(
                nearest, False, nearest_distance, nearest,
                nearest_distance, self.threshold, 0,
            )

        if self._pending_context == nearest:
            self._pending_count += 1
        else:
            self._pending_context = nearest
            self._pending_count = 1
        if self._pending_count >= self.switch_persistence:
            self.current_context = nearest
            self._reset_pending()
            self._remember(nearest, value, nearest_distance)
        return ContextDecision(
            int(self.current_context), False, nearest_distance, nearest,
            nearest_distance, self.threshold, 0,
        )

    def state_dict(self):
        return {
            "kind": "memory_bank",
            "threshold": self.threshold,
            "candidate_threshold": self.candidate_threshold,
            "create_persistence": self.create_persistence,
            "switch_persistence": self.switch_persistence,
            "memory_size": self.memory_size,
            "neighbors": self.neighbors,
            "current_context": self.current_context,
            "counts": {str(key): int(value) for key, value in self.counts.items()},
            "memory_counts": {
                str(key): int(len(value)) for key, value in self.memories.items()
            },
            "memories": {
                str(key): value.tolist() for key, value in self.memories.items()
            },
        }
