from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass
class RegistryDecision:
    context_id: Optional[int]
    semantic_key: Optional[str]
    accepted: bool
    created: bool
    reused: bool
    pending_key: Optional[str]
    pending_count: int
    joint_confidence: float


class SemanticContextRegistry:
    """Confidence-gated registry keyed by factorized semantic attributes."""

    def __init__(
        self,
        confidence_thresholds: Mapping[str, float],
        create_persistence: int = 5,
        switch_persistence: int = 3,
    ):
        required = {"weather", "road", "lighting"}
        if set(confidence_thresholds) != required:
            raise ValueError(f"confidence_thresholds must contain exactly {sorted(required)}")
        if any(not 0.0 <= float(value) <= 1.0 for value in confidence_thresholds.values()):
            raise ValueError("confidence thresholds must be in [0, 1]")
        if create_persistence < 1 or switch_persistence < 1:
            raise ValueError("persistence values must be positive")
        self.confidence_thresholds = {
            key: float(value) for key, value in confidence_thresholds.items()
        }
        self.create_persistence = int(create_persistence)
        self.switch_persistence = int(switch_persistence)
        self.key_to_context: Dict[str, int] = {}
        self.context_to_key: Dict[int, str] = {}
        self.counts: Dict[int, int] = {}
        self.current_context: Optional[int] = None
        self._pending_key: Optional[str] = None
        self._pending_count = 0
        self._next_context_id = 0

    def register(self, semantic_key: str, count: int = 1) -> int:
        key = str(semantic_key)
        if key in self.key_to_context:
            context_id = self.key_to_context[key]
            self.counts[context_id] += max(0, int(count))
            self.current_context = context_id
            return context_id
        context_id = self._next_context_id
        self._next_context_id += 1
        self.key_to_context[key] = context_id
        self.context_to_key[context_id] = key
        self.counts[context_id] = max(1, int(count))
        self.current_context = context_id
        return context_id

    def _reset_pending(self) -> None:
        self._pending_key = None
        self._pending_count = 0

    def observe(
        self,
        semantic_key: str,
        confidences: Mapping[str, float],
    ) -> RegistryDecision:
        missing = set(self.confidence_thresholds) - set(confidences)
        if missing:
            raise ValueError(f"Missing confidence values: {sorted(missing)}")
        key = str(semantic_key)
        joint_confidence = min(float(confidences[name]) for name in self.confidence_thresholds)
        accepted = all(
            float(confidences[name]) >= threshold
            for name, threshold in self.confidence_thresholds.items()
        )
        if self.current_context is None:
            if not accepted:
                self._reset_pending()
                return RegistryDecision(
                    None, None, False, False, False, None, 0, joint_confidence
                )
            if self._pending_key == key:
                self._pending_count += 1
            else:
                self._pending_key = key
                self._pending_count = 1
            required = (
                self.switch_persistence
                if key in self.key_to_context
                else self.create_persistence
            )
            if self._pending_count < required:
                return RegistryDecision(
                    None, None, True, False, False,
                    self._pending_key, self._pending_count, joint_confidence,
                )
            existed = key in self.key_to_context
            context_id = self.register(key)
            self._reset_pending()
            return RegistryDecision(
                context_id, key, True, not existed, existed, None, 0, joint_confidence
            )

        current_key = self.context_to_key[self.current_context]
        if not accepted:
            self._reset_pending()
            return RegistryDecision(
                self.current_context, current_key, False, False, False,
                None, 0, joint_confidence,
            )
        if key == current_key:
            self.counts[self.current_context] += 1
            self._reset_pending()
            return RegistryDecision(
                self.current_context, current_key, True, False, False,
                None, 0, joint_confidence,
            )

        if self._pending_key == key:
            self._pending_count += 1
        else:
            self._pending_key = key
            self._pending_count = 1
        required = (
            self.switch_persistence
            if key in self.key_to_context
            else self.create_persistence
        )
        if self._pending_count < required:
            return RegistryDecision(
                self.current_context, current_key, True, False, False,
                self._pending_key, self._pending_count, joint_confidence,
            )

        existed = key in self.key_to_context
        context_id = self.register(key)
        self._reset_pending()
        return RegistryDecision(
            context_id,
            key,
            True,
            not existed,
            existed,
            None,
            0,
            joint_confidence,
        )

    def state_dict(self):
        return {
            "confidence_thresholds": self.confidence_thresholds,
            "create_persistence": self.create_persistence,
            "switch_persistence": self.switch_persistence,
            "current_context": self.current_context,
            "key_to_context": self.key_to_context,
            "context_to_key": {str(key): value for key, value in self.context_to_key.items()},
            "counts": {str(key): value for key, value in self.counts.items()},
            "pending_key": self._pending_key,
            "pending_count": self._pending_count,
        }
