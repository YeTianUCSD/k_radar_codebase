"""Boundary-blind temporal voting and semantic context routing."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from math import ceil
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .registry import ContextRegistry, SemanticKey, normalize_key


ATTRIBUTES = ("weather", "road", "lighting")


@dataclass(frozen=True)
class ContextDecision:
    step: int
    event: str
    context_id: int
    active_key: SemanticKey
    provisional_key: SemanticKey
    candidate_key: Optional[SemanticKey]
    candidate_votes: int
    candidate_ratio: float
    pending_count: int
    update_enabled: bool
    created_context_id: Optional[int]
    weather_confidence: float
    road_confidence: float
    lighting_confidence: float

    def to_dict(self):
        row = asdict(self)
        for field in ("active_key", "provisional_key", "candidate_key"):
            value = row[field]
            row[field] = "|".join(value) if value is not None else ""
        return row


class OnlineContextController:
    """Route every frame without receiving sequence IDs or boundary markers."""

    def __init__(
        self,
        label_names: Mapping[str, Sequence[str]],
        base_key: Sequence[str],
        decision_window: int,
        confirmation_frames: int,
        majority_ratio: float,
        confidence_thresholds: Optional[Mapping[str, float]] = None,
        confidence_policy: str = "observe_only",
        pause_update_when_pending: bool = True,
        new_context_confirmation_frames: Optional[int] = None,
        new_context_majority_ratio: Optional[float] = None,
    ):
        if int(decision_window) < 1 or int(confirmation_frames) < 1:
            raise ValueError("window and confirmation_frames must be positive")
        if not 0.5 <= float(majority_ratio) <= 1.0:
            raise ValueError("majority_ratio must be in [0.5, 1.0]")
        if new_context_confirmation_frames is None:
            new_context_confirmation_frames = confirmation_frames
        if new_context_majority_ratio is None:
            new_context_majority_ratio = majority_ratio
        if int(new_context_confirmation_frames) < 1:
            raise ValueError("new_context_confirmation_frames must be positive")
        if not 0.5 <= float(new_context_majority_ratio) <= 1.0:
            raise ValueError("new_context_majority_ratio must be in [0.5, 1.0]")
        if confidence_policy not in {"observe_only", "enforce"}:
            raise ValueError("confidence_policy must be observe_only or enforce")
        self.label_names = {
            key: tuple(str(value) for value in label_names[key])
            for key in ATTRIBUTES
        }
        self.decision_window = int(decision_window)
        self.confirmation_frames = int(confirmation_frames)
        self.majority_ratio = float(majority_ratio)
        self.required_votes = int(ceil(self.decision_window * self.majority_ratio))
        self.new_context_confirmation_frames = int(
            new_context_confirmation_frames
        )
        self.new_context_majority_ratio = float(new_context_majority_ratio)
        self.new_context_required_votes = int(ceil(
            self.decision_window * self.new_context_majority_ratio
        ))
        self.confidence_thresholds = {
            key: float((confidence_thresholds or {}).get(key, 0.0))
            for key in ATTRIBUTES
        }
        self.confidence_policy = confidence_policy
        self.pause_update_when_pending = bool(pause_update_when_pending)
        self.registry = ContextRegistry(base_key)
        self.active_entry = self.registry.find(base_key)
        self._buffer = deque(maxlen=self.decision_window)
        self._pending_key = None
        self._pending_count = 0

    def _decode(self, probabilities: Mapping[str, np.ndarray]):
        key, confidence = [], {}
        normalized = {}
        for attribute in ATTRIBUTES:
            values = np.asarray(probabilities[attribute], dtype=np.float64)
            if values.ndim != 1 or len(values) != len(self.label_names[attribute]):
                raise ValueError(f"invalid {attribute} probability vector")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite {attribute} probabilities")
            total = float(values.sum())
            if total <= 0:
                raise ValueError(f"non-positive {attribute} probability sum")
            values = values / total
            position = int(values.argmax())
            normalized[attribute] = values
            key.append(self.label_names[attribute][position])
            confidence[attribute] = float(values[position])
        return normalize_key(key), confidence, normalized

    def _candidate(self):
        if len(self._buffer) < self.decision_window:
            return None, 0, 0.0, {key: 0.0 for key in ATTRIBUTES}
        keys = [item[0] for item in self._buffer if item[0] is not None]
        if not keys:
            return None, 0, 0.0, {key: 0.0 for key in ATTRIBUTES}
        counts = Counter(keys)
        maximum = max(counts.values())
        # A tie is resolved in favour of the active key, then the newest key.
        tied = {key for key, count in counts.items() if count == maximum}
        candidate = (
            self.active_entry.key if self.active_entry.key in tied
            else next(key for key in reversed(keys) if key in tied)
        )
        confidence = {}
        for attribute_index, attribute in enumerate(ATTRIBUTES):
            class_index = self.label_names[attribute].index(candidate[attribute_index])
            confidence[attribute] = float(np.mean([
                item[2][attribute][class_index] for item in self._buffer
                if item[0] == candidate
            ]))
        ratio = maximum / self.decision_window
        is_new_context = self.registry.find(candidate) is None
        required_votes = (
            self.new_context_required_votes
            if is_new_context else self.required_votes
        )
        if maximum < required_votes:
            return None, maximum, ratio, confidence
        if self.confidence_policy == "enforce" and any(
            confidence[key] < self.confidence_thresholds[key] for key in ATTRIBUTES
        ):
            return None, maximum, ratio, confidence
        return candidate, maximum, ratio, confidence

    def observe(self, probabilities: Mapping[str, np.ndarray], step: int):
        provisional, frame_confidence, normalized = self._decode(probabilities)
        accepted = not (
            self.confidence_policy == "enforce" and any(
                frame_confidence[key] < self.confidence_thresholds[key]
                for key in ATTRIBUTES
            )
        )
        self._buffer.append((
            provisional if accepted else None, frame_confidence, normalized
        ))
        candidate, votes, ratio, _ = self._candidate()
        event = "stable"
        created_id = None

        if not accepted:
            self._pending_key = None
            self._pending_count = 0
            event = "abstain"
        elif candidate is None:
            self._pending_key = None
            self._pending_count = 0
            if provisional != self.active_entry.key:
                event = "pending"
        elif candidate == self.active_entry.key:
            self._pending_key = None
            self._pending_count = 0
            if provisional != self.active_entry.key:
                event = "pending"
        else:
            event = "pending"
            is_new_context = self.registry.find(candidate) is None
            required_confirmations = (
                self.new_context_confirmation_frames
                if is_new_context else self.confirmation_frames
            )
            if candidate == self._pending_key:
                self._pending_count += 1
            else:
                self._pending_key = candidate
                self._pending_count = 1
            if self._pending_count >= required_confirmations:
                entry, created = self.registry.get_or_create(candidate, step)
                self.active_entry = entry
                event = "create" if created else "switch"
                created_id = entry.context_id if created else None
                self._pending_key = None
                self._pending_count = 0

        update_enabled = (
            event == "stable" and provisional == self.active_entry.key
            if self.pause_update_when_pending else True
        )
        return ContextDecision(
            step=int(step), event=event, context_id=self.active_entry.context_id,
            active_key=self.active_entry.key, provisional_key=provisional,
            candidate_key=candidate, candidate_votes=int(votes),
            candidate_ratio=float(ratio), pending_count=int(self._pending_count),
            update_enabled=bool(update_enabled), created_context_id=created_id,
            weather_confidence=frame_confidence["weather"],
            road_confidence=frame_confidence["road"],
            lighting_confidence=frame_confidence["lighting"],
        )
