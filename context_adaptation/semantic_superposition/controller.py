"""Checkpointable semantic controller used by automatic PSP adaptation."""

from __future__ import annotations

from collections import deque

import numpy as np

from online_context.controller import OnlineContextController
from online_context.registry import ContextRegistry, normalize_key


class CheckpointableSemanticController(OnlineContextController):
    """Add rollback/resume state without changing the evaluated V7 policy."""

    def state_dict(self):
        records = self.registry.to_records()
        buffer = []
        for key, confidence, normalized in self._buffer:
            buffer.append({
                "key": None if key is None else list(key),
                "confidence": dict(confidence),
                "normalized": {
                    name: np.asarray(value, dtype=np.float64).tolist()
                    for name, value in normalized.items()
                },
            })
        return {
            "policy": {
                "decision_window": self.decision_window,
                "confirmation_frames": self.confirmation_frames,
                "majority_ratio": self.majority_ratio,
                "new_context_confirmation_frames": self.new_context_confirmation_frames,
                "new_context_majority_ratio": self.new_context_majority_ratio,
                "confidence_thresholds": dict(self.confidence_thresholds),
                "confidence_policy": self.confidence_policy,
                "pause_update_when_pending": self.pause_update_when_pending,
            },
            "registry_records": records,
            "active_key": list(self.active_entry.key),
            "buffer": buffer,
            "pending_key": None if self._pending_key is None else list(self._pending_key),
            "pending_count": int(self._pending_count),
        }

    def load_state_dict(self, state):
        expected = self.state_dict()["policy"]
        actual = state.get("policy")
        if actual is not None and actual != expected:
            raise ValueError(
                f"semantic controller policy differs from checkpoint: "
                f"checkpoint={actual}, current={expected}"
            )
        records = sorted(
            list(state["registry_records"]), key=lambda item: int(item["context_id"])
        )
        if not records or int(records[0]["context_id"]) != 0:
            raise ValueError("semantic controller state has no base context")
        first = records[0]
        registry = ContextRegistry(
            (first["weather"], first["road"], first["lighting"]),
            created_step=int(first["created_step"]),
        )
        for expected, record in enumerate(records[1:], start=1):
            entry, created = registry.get_or_create(
                (record["weather"], record["road"], record["lighting"]),
                int(record["created_step"]),
            )
            if not created or entry.context_id != expected:
                raise ValueError("semantic controller context IDs are not contiguous")
        active = registry.find(state["active_key"])
        if active is None:
            raise ValueError("semantic controller active key is not registered")
        restored = deque(maxlen=self.decision_window)
        for item in state.get("buffer", ()):
            key = item.get("key")
            restored.append((
                None if key is None else normalize_key(key),
                {name: float(value) for name, value in item["confidence"].items()},
                {
                    name: np.asarray(value, dtype=np.float64)
                    for name, value in item["normalized"].items()
                },
            ))
        self.registry = registry
        self.active_entry = active
        self._buffer = restored
        pending = state.get("pending_key")
        self._pending_key = None if pending is None else normalize_key(pending)
        self._pending_count = int(state.get("pending_count", 0))
