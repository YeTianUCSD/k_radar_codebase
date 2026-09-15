"""Evidence-based temporal routing for dynamic scene contexts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import ArrayLike

from .rff_kde import OnlineRFFKDE


def rank_eligible_contexts(
    scores: Mapping[int, float],
    thresholds: Mapping[int, float],
    *,
    eps: float = 1e-12,
) -> tuple[Optional[int], float, float, float, float]:
    """Rank contexts that pass their own threshold by relative log margin."""
    score_ids = {int(context_id) for context_id in scores}
    threshold_ids = {int(context_id) for context_id in thresholds}
    if score_ids != threshold_ids:
        raise ValueError("scores and thresholds must contain identical context IDs.")
    eligible = []
    for context_id, raw_score in scores.items():
        resolved_id = int(context_id)
        score = float(raw_score)
        threshold = float(thresholds[resolved_id])
        if not np.isfinite(score) or not np.isfinite(threshold):
            raise ValueError("Context scores and thresholds must be finite.")
        if threshold < 0:
            raise ValueError("Context thresholds must be nonnegative.")
        if score >= threshold:
            confidence = float(
                np.log(max(score, eps)) - np.log(max(threshold, eps))
            )
            eligible.append((resolved_id, score, confidence))
    if not eligible:
        return None, float("-inf"), float("-inf"), float("-inf"), float("-inf")
    eligible.sort(key=lambda item: item[2], reverse=True)
    best_id, best_score, best_confidence = eligible[0]
    if len(eligible) > 1:
        _, second_score, second_confidence = eligible[1]
    else:
        second_score = float("-inf")
        second_confidence = float("-inf")
    return best_id, best_score, second_score, best_confidence, second_confidence


@dataclass(frozen=True)
class RouteDecision:
    status: str
    context_id: Optional[int]
    reason: str
    best_context_id: Optional[int]
    best_score: float
    second_score: float
    best_confidence: float
    second_confidence: float
    historical_evidence: float
    novelty_strength: float
    novelty_evidence: float
    candidate_evidence: float
    should_update_model: bool
    should_update_memory: bool
    candidate_features: Optional[np.ndarray] = None


class ContextRouter:
    """Route frames using smoothed features and decaying temporal evidence."""

    STATE_VERSION = 4
    VALID_STATUS = {"stay", "switch", "pending", "create"}

    def __init__(
        self,
        input_dim: int,
        *,
        feature_ema_beta: float = 0.5,
        routing_threshold_scale: float = 0.85,
        near_threshold_scale: float = 0.60,
        min_score_margin: float = 0.02,
        evidence_decay: float = 0.7,
        evidence_miss_penalty: float = 0.2,
        evidence_confidence_clip: float = 1.0,
        switch_evidence_threshold: float = 0.35,
        switch_evidence_margin: float = 0.10,
        switch_cooldown: int = 2,
        novelty_decay: float = 0.8,
        novelty_drift: float = 0.2,
        novelty_evidence_threshold: float = 2.5,
        min_candidate_samples: int = 3,
        candidate_evidence_threshold: float = 1.5,
        candidate_similarity_threshold: float = 0.3,
        candidate_evidence_decay: float = 0.5,
        candidate_mismatch_patience: int = 2,
        candidate_n_rff_features: int = 1024,
        candidate_bandwidth: float = 1.0,
        random_state: int = 20260812,
        candidate_max_size: int = 64,
        ambiguous_creation_enabled: bool = False,
        near_creation_enabled: bool = False,
        boundary_creation_enabled: bool = False,
        split_min_candidate_samples: int = 8,
        split_evidence_threshold: float = 1.0,
        strict_update_gate_enabled: bool = False,
        update_min_confidence: float = 0.2,
        update_min_margin: float = 0.15,
        update_min_consecutive_accepts: int = 4,
        update_after_switch_warmup: int = 4,
        freeze_base_context_memory: bool = False,
        update_provisional_memory: bool = True,
        update_provisional_model: bool = False,
        dtype: Any = np.float32,
        switch_patience: Optional[int] = None,
        new_context_patience: Optional[int] = None,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not 0.0 <= feature_ema_beta < 1.0:
            raise ValueError("feature_ema_beta must be in [0, 1).")
        for name, value in (
            ("routing_threshold_scale", routing_threshold_scale),
            ("near_threshold_scale", near_threshold_scale),
            ("evidence_decay", evidence_decay),
            ("novelty_decay", novelty_decay),
            ("candidate_evidence_decay", candidate_evidence_decay),
        ):
            if name.endswith("_scale"):
                if value <= 0:
                    raise ValueError(f"{name} must be positive.")
            elif not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        for name, value in (
            ("min_score_margin", min_score_margin),
            ("evidence_miss_penalty", evidence_miss_penalty),
            ("evidence_confidence_clip", evidence_confidence_clip),
            ("switch_evidence_threshold", switch_evidence_threshold),
            ("switch_evidence_margin", switch_evidence_margin),
            ("novelty_drift", novelty_drift),
            ("novelty_evidence_threshold", novelty_evidence_threshold),
            ("candidate_evidence_threshold", candidate_evidence_threshold),
            ("candidate_similarity_threshold", candidate_similarity_threshold),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if evidence_confidence_clip <= 0:
            raise ValueError("evidence_confidence_clip must be positive.")
        if switch_cooldown < 0:
            raise ValueError("switch_cooldown must be nonnegative.")
        for name, value in (
            ("min_candidate_samples", min_candidate_samples),
            ("candidate_mismatch_patience", candidate_mismatch_patience),
            ("candidate_n_rff_features", candidate_n_rff_features),
            ("candidate_max_size", candidate_max_size),
            ("split_min_candidate_samples", split_min_candidate_samples),
            ("update_min_consecutive_accepts", update_min_consecutive_accepts),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if candidate_bandwidth <= 0:
            raise ValueError("candidate_bandwidth must be positive.")
        if candidate_max_size < min_candidate_samples:
            raise ValueError("candidate_max_size must be at least min_candidate_samples.")
        if (
            ambiguous_creation_enabled
            or near_creation_enabled
            or boundary_creation_enabled
        ) and (
            candidate_max_size < split_min_candidate_samples
        ):
            raise ValueError("candidate_max_size must cover split_min_candidate_samples.")
        if split_evidence_threshold < 0:
            raise ValueError("split_evidence_threshold must be nonnegative.")
        if update_min_confidence < 0 or update_min_margin < 0:
            raise ValueError("update confidence and margin must be nonnegative.")
        if update_after_switch_warmup < 0:
            raise ValueError("update_after_switch_warmup must be nonnegative.")

        self.input_dim = int(input_dim)
        self.feature_ema_beta = float(feature_ema_beta)
        self.routing_threshold_scale = float(routing_threshold_scale)
        self.near_threshold_scale = float(near_threshold_scale)
        self.min_score_margin = float(min_score_margin)
        self.evidence_decay = float(evidence_decay)
        self.evidence_miss_penalty = float(evidence_miss_penalty)
        self.evidence_confidence_clip = float(evidence_confidence_clip)
        self.switch_evidence_threshold = float(switch_evidence_threshold)
        self.switch_evidence_margin = float(switch_evidence_margin)
        self.switch_cooldown = int(switch_cooldown)
        self.novelty_decay = float(novelty_decay)
        self.novelty_drift = float(novelty_drift)
        self.novelty_evidence_threshold = float(novelty_evidence_threshold)
        self.min_candidate_samples = int(min_candidate_samples)
        self.candidate_evidence_threshold = float(candidate_evidence_threshold)
        self.candidate_similarity_threshold = float(candidate_similarity_threshold)
        self.candidate_evidence_decay = float(candidate_evidence_decay)
        self.candidate_mismatch_patience = int(candidate_mismatch_patience)
        self.candidate_n_rff_features = int(candidate_n_rff_features)
        self.candidate_bandwidth = float(candidate_bandwidth)
        self.random_state = int(random_state)
        self.candidate_max_size = int(candidate_max_size)
        self.ambiguous_creation_enabled = bool(ambiguous_creation_enabled)
        self.near_creation_enabled = bool(near_creation_enabled)
        self.boundary_creation_enabled = bool(boundary_creation_enabled)
        self.split_min_candidate_samples = int(split_min_candidate_samples)
        self.split_evidence_threshold = float(split_evidence_threshold)
        self.strict_update_gate_enabled = bool(strict_update_gate_enabled)
        self.update_min_confidence = float(update_min_confidence)
        self.update_min_margin = float(update_min_margin)
        self.update_min_consecutive_accepts = int(update_min_consecutive_accepts)
        self.update_after_switch_warmup = int(update_after_switch_warmup)
        self.freeze_base_context_memory = bool(freeze_base_context_memory)
        self.update_provisional_memory = bool(update_provisional_memory)
        self.update_provisional_model = bool(update_provisional_model)
        self.dtype = np.dtype(dtype)

        self.deprecated_switch_patience = (
            None if switch_patience is None else int(switch_patience)
        )
        self.deprecated_new_context_patience = (
            None if new_context_patience is None else int(new_context_patience)
        )

        self.active_context_id: Optional[int] = None
        self._context_evidence: dict[int, float] = {}
        self._novelty_evidence = 0.0
        self._candidate_evidence = 0.0
        self._candidate_mismatch_count = 0
        self._candidate_features: list[np.ndarray] = []
        self._candidate_kde: Optional[OnlineRFFKDE] = None
        self._candidate_kind: Optional[str] = None
        self._smoothed_feature: Optional[np.ndarray] = None
        self._switch_cooldown_remaining = 0
        self._awaiting_context_creation = False
        self._update_streak_context_id: Optional[int] = None
        self._update_streak = 0
        self._frames_since_switch = 0

    def _validate_x(self, x: ArrayLike) -> np.ndarray:
        value = np.asarray(x, dtype=self.dtype)
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (self.input_dim,):
            raise ValueError(
                f"Expected one projected feature with shape ({self.input_dim},); "
                f"got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("Projected feature contains NaN or infinite values.")
        return value

    def prepare_feature(self, x: ArrayLike) -> np.ndarray:
        """Apply checkpointed short-term EMA smoothing before density queries."""
        value = self._validate_x(x)
        if self._smoothed_feature is None:
            self._smoothed_feature = value.copy()
        else:
            beta = self.feature_ema_beta
            self._smoothed_feature = (
                beta * self._smoothed_feature + (1.0 - beta) * value
            ).astype(self.dtype, copy=False)
        return self._smoothed_feature.copy()

    def _new_candidate_kde(self) -> OnlineRFFKDE:
        return OnlineRFFKDE(
            input_dim=self.input_dim,
            n_features=self.candidate_n_rff_features,
            bandwidth=self.candidate_bandwidth,
            random_state=self.random_state,
            dtype=self.dtype,
        )

    def _reset_candidate(self) -> None:
        self._candidate_features = []
        self._candidate_kde = None
        self._candidate_evidence = 0.0
        self._candidate_mismatch_count = 0
        self._candidate_kind = None

    def _reset_temporal_evidence(self) -> None:
        self._context_evidence = {
            context_id: 0.0 for context_id in self._context_evidence
        }
        self._novelty_evidence = 0.0

    def _ensure_context_ids(self, context_ids: set[int]) -> None:
        stale = set(self._context_evidence) - context_ids
        for context_id in stale:
            del self._context_evidence[context_id]
        for context_id in context_ids:
            self._context_evidence.setdefault(context_id, 0.0)

    def _append_candidate(self, x: np.ndarray, *, kind: str = "novel") -> bool:
        if self._candidate_kind != kind:
            self._reset_candidate()
            self._candidate_kind = str(kind)
        if self._candidate_kde is None:
            self._candidate_kde = self._new_candidate_kde()
            self._candidate_kde.update_one(x)
            self._candidate_features = [x.copy()]
            self._candidate_evidence = 1.0
            self._candidate_mismatch_count = 0
            return True

        similarity = float(self._candidate_kde.query_kernel_mean(x))
        if similarity >= self.candidate_similarity_threshold:
            self._candidate_kde.update_one(x)
            self._candidate_features.append(x.copy())
            if len(self._candidate_features) > self.candidate_max_size:
                self._candidate_features = self._candidate_features[
                    -self.candidate_max_size :
                ]
            self._candidate_evidence = (
                self.candidate_evidence_decay * self._candidate_evidence
                + similarity
            )
            self._candidate_mismatch_count = 0
            return True

        self._candidate_evidence *= self.candidate_evidence_decay
        self._candidate_mismatch_count += 1
        if self._candidate_mismatch_count >= self.candidate_mismatch_patience:
            self._candidate_kde = self._new_candidate_kde()
            self._candidate_kde.update_one(x)
            self._candidate_features = [x.copy()]
            self._candidate_evidence = 1.0
            self._candidate_mismatch_count = 0
            return True
        return False

    @staticmethod
    def _validate_mapping_ids(
        scores: Mapping[int, float],
        *mappings: Mapping[int, Any],
    ) -> set[int]:
        score_ids = {int(context_id) for context_id in scores}
        for mapping in mappings:
            if {int(context_id) for context_id in mapping} != score_ids:
                raise ValueError("All routing mappings must contain identical context IDs.")
        return score_ids

    def _decision(
        self,
        *,
        status: str,
        context_id: Optional[int],
        reason: str,
        best_context_id: Optional[int],
        best_score: float,
        second_score: float,
        best_confidence: float,
        second_confidence: float,
        historical_evidence: float,
        novelty_strength: float,
        should_update_model: bool,
        should_update_memory: bool,
        candidate_features: Optional[np.ndarray] = None,
    ) -> RouteDecision:
        return RouteDecision(
            status=status,
            context_id=context_id,
            reason=reason,
            best_context_id=best_context_id,
            best_score=float(best_score),
            second_score=float(second_score),
            best_confidence=float(best_confidence),
            second_confidence=float(second_confidence),
            historical_evidence=float(historical_evidence),
            novelty_strength=float(novelty_strength),
            novelty_evidence=float(self._novelty_evidence),
            candidate_evidence=float(self._candidate_evidence),
            should_update_model=bool(should_update_model),
            should_update_memory=bool(should_update_memory),
            candidate_features=candidate_features,
        )

    def _split_candidate_decision(
        self,
        *,
        kind: str,
        value: np.ndarray,
        pending_reason: str,
        creation_reason: str,
        best_context_id: Optional[int],
        best_score: float,
        second_score: float,
        best_confidence: float,
        second_confidence: float,
        historical_evidence: float,
        allow_context_creation: bool,
    ) -> RouteDecision:
        coherent = self._append_candidate(value, kind=kind)
        ready = (
            coherent
            and len(self._candidate_features) >= self.split_min_candidate_samples
            and self._candidate_evidence >= self.split_evidence_threshold
        )
        if ready and allow_context_creation:
            self._awaiting_context_creation = True
            return self._decision(
                status="create", context_id=None, reason=creation_reason,
                best_context_id=best_context_id, best_score=best_score,
                second_score=second_score, best_confidence=best_confidence,
                second_confidence=second_confidence,
                historical_evidence=historical_evidence, novelty_strength=0.0,
                should_update_model=False, should_update_memory=False,
                candidate_features=np.stack(self._candidate_features, axis=0),
            )
        reason = (
            f"{kind}_context_creation_disabled"
            if ready and not allow_context_creation else pending_reason
        )
        return self._decision(
            status="pending", context_id=self.active_context_id, reason=reason,
            best_context_id=best_context_id, best_score=best_score,
            second_score=second_score, best_confidence=best_confidence,
            second_confidence=second_confidence,
            historical_evidence=historical_evidence, novelty_strength=0.0,
            should_update_model=False, should_update_memory=False,
        )

    def apply_update_gate(
        self,
        decision: RouteDecision,
        *,
        context_id: int,
        lifecycle: str,
        update_threshold: Optional[float] = None,
    ) -> RouteDecision:
        """Apply the checkpointed high-precision gate for KDE and PSP updates."""
        if not self.strict_update_gate_enabled:
            return decision
        accepted = decision.status in {"stay", "switch"}
        if not accepted:
            self._update_streak_context_id = None
            self._update_streak = 0
            self._frames_since_switch = 0
            return replace(
                decision, should_update_model=False, should_update_memory=False
            )
        resolved_id = int(context_id)
        if decision.status == "switch" or self._update_streak_context_id != resolved_id:
            self._update_streak_context_id = resolved_id
            self._update_streak = 1
            self._frames_since_switch = 0
        else:
            self._update_streak += 1
            self._frames_since_switch += 1
        margin = (
            float("inf")
            if not np.isfinite(decision.second_confidence)
            else decision.best_confidence - decision.second_confidence
        )
        density_safe = (
            True
            if update_threshold is None
            else decision.best_score >= float(update_threshold)
        )
        ready = (
            density_safe
            and decision.best_confidence >= self.update_min_confidence
            and margin >= self.update_min_margin
            and self._update_streak >= self.update_min_consecutive_accepts
            and self._frames_since_switch >= self.update_after_switch_warmup
        )
        memory_allowed = decision.should_update_memory and ready
        model_allowed = decision.should_update_model and ready
        if lifecycle == "provisional":
            memory_allowed = memory_allowed and self.update_provisional_memory
            model_allowed = model_allowed and self.update_provisional_model
        if resolved_id == 0 and self.freeze_base_context_memory:
            memory_allowed = False
        return replace(
            decision, should_update_model=bool(model_allowed),
            should_update_memory=bool(memory_allowed),
        )

    def step(
        self,
        x: ArrayLike,
        scores: Mapping[int, float],
        density_thresholds: Mapping[int, float],
        *,
        routing_thresholds: Optional[Mapping[int, float]] = None,
        near_thresholds: Optional[Mapping[int, float]] = None,
        context_lifecycles: Optional[Mapping[int, str]] = None,
        allow_context_creation: bool = True,
    ) -> RouteDecision:
        """Route one prepared feature without modifying committed Context Memory."""
        if self._awaiting_context_creation:
            raise RuntimeError(
                "confirm_created_context() must be called after a create decision."
            )
        value = self._validate_x(x)
        resolved_routing = (
            {
                int(context_id): float(threshold) * self.routing_threshold_scale
                for context_id, threshold in density_thresholds.items()
            }
            if routing_thresholds is None
            else routing_thresholds
        )
        resolved_near = (
            {
                int(context_id): float(threshold) * self.near_threshold_scale
                for context_id, threshold in density_thresholds.items()
            }
            if near_thresholds is None
            else near_thresholds
        )
        lifecycles = (
            {int(context_id): "stable" for context_id in scores}
            if context_lifecycles is None
            else {int(context_id): str(value) for context_id, value in context_lifecycles.items()}
        )
        context_ids = self._validate_mapping_ids(
            scores,
            density_thresholds,
            resolved_routing,
            resolved_near,
            lifecycles,
        )
        if any(value not in {"stable", "provisional"} for value in lifecycles.values()):
            raise ValueError("Unsupported context lifecycle in routing input.")
        if not context_ids:
            raise ValueError("At least one committed context is required for routing.")
        self._ensure_context_ids(context_ids)
        if self.active_context_id is not None and self.active_context_id not in context_ids:
            raise ValueError("The active context does not exist in routing inputs.")
        if self._switch_cooldown_remaining > 0:
            self._switch_cooldown_remaining -= 1

        (
            best_id,
            best_score,
            second_score,
            best_confidence,
            second_confidence,
        ) = rank_eligible_contexts(scores, resolved_routing)
        margin_ok = (
            best_id is not None
            and (
                not np.isfinite(second_confidence)
                or best_confidence - second_confidence >= self.min_score_margin
            )
        )

        for context_id in context_ids:
            if context_id == self.active_context_id:
                self._context_evidence[context_id] = 0.0
                continue
            score = float(scores[context_id])
            threshold = max(float(resolved_routing[context_id]), 1e-12)
            confidence = float(np.log(max(score, 1e-12) / threshold))
            previous = self._context_evidence[context_id]
            if confidence > 0:
                self._context_evidence[context_id] = (
                    self.evidence_decay * previous
                    + min(confidence, self.evidence_confidence_clip)
                )
            else:
                self._context_evidence[context_id] = max(
                    0.0,
                    self.evidence_decay * previous - self.evidence_miss_penalty,
                )

        if self.active_context_id is None and best_id is not None and margin_ok:
            self.active_context_id = best_id
            self._reset_temporal_evidence()
            self._reset_candidate()
            return self._decision(
                status="switch",
                context_id=best_id,
                reason="initialize_active_context",
                best_context_id=best_id,
                best_score=best_score,
                second_score=second_score,
                best_confidence=best_confidence,
                second_confidence=second_confidence,
                historical_evidence=0.0,
                novelty_strength=0.0,
                should_update_model=True,
                should_update_memory=True,
            )

        if best_id is not None and margin_ok and best_id == self.active_context_id:
            self._novelty_evidence *= self.novelty_decay
            update_margin = (
                float("inf")
                if not np.isfinite(second_confidence)
                else best_confidence - second_confidence
            )
            boundary_match = (
                self.strict_update_gate_enabled
                and self.boundary_creation_enabled
                and lifecycles[best_id] == "stable"
                and (
                    best_confidence < self.update_min_confidence
                    or update_margin < self.update_min_margin
                )
            )
            if boundary_match:
                return self._split_candidate_decision(
                    kind="boundary", value=value,
                    pending_reason="boundary_historical_match",
                    creation_reason="boundary_context_evidence_confirmed",
                    best_context_id=best_id, best_score=best_score,
                    second_score=second_score,
                    best_confidence=best_confidence,
                    second_confidence=second_confidence,
                    historical_evidence=0.0,
                    allow_context_creation=allow_context_creation,
                )
            self._reset_candidate()
            return self._decision(
                status="stay",
                context_id=best_id,
                reason="active_context_matched",
                best_context_id=best_id,
                best_score=best_score,
                second_score=second_score,
                best_confidence=best_confidence,
                second_confidence=second_confidence,
                historical_evidence=0.0,
                novelty_strength=0.0,
                should_update_model=True,
                should_update_memory=True,
            )

        if best_id is not None and margin_ok and best_id != self.active_context_id:
            evidence = self._context_evidence[best_id]
            competing = max(
                (
                    value
                    for context_id, value in self._context_evidence.items()
                    if context_id not in {best_id, self.active_context_id}
                ),
                default=0.0,
            )
            self._novelty_evidence *= self.novelty_decay
            self._reset_candidate()
            if (
                evidence >= self.switch_evidence_threshold
                and evidence - competing >= self.switch_evidence_margin
                and self._switch_cooldown_remaining == 0
            ):
                self.active_context_id = best_id
                self._reset_temporal_evidence()
                self._switch_cooldown_remaining = self.switch_cooldown
                return self._decision(
                    status="switch",
                    context_id=best_id,
                    reason="historical_context_evidence_confirmed",
                    best_context_id=best_id,
                    best_score=best_score,
                    second_score=second_score,
                    best_confidence=best_confidence,
                    second_confidence=second_confidence,
                    historical_evidence=evidence,
                    novelty_strength=0.0,
                    should_update_model=True,
                    should_update_memory=True,
                )
            return self._decision(
                status="pending",
                context_id=self.active_context_id,
                reason="historical_context_evidence_pending",
                best_context_id=best_id,
                best_score=best_score,
                second_score=second_score,
                best_confidence=best_confidence,
                second_confidence=second_confidence,
                historical_evidence=evidence,
                novelty_strength=0.0,
                should_update_model=False,
                should_update_memory=False,
            )

        if best_id is not None and not margin_ok:
            self._novelty_evidence *= self.novelty_decay
            if self.ambiguous_creation_enabled:
                return self._split_candidate_decision(
                    kind="ambiguous", value=value,
                    pending_reason="ambiguous_historical_match",
                    creation_reason="ambiguous_context_evidence_confirmed",
                    best_context_id=best_id, best_score=best_score,
                    second_score=second_score, best_confidence=best_confidence,
                    second_confidence=second_confidence,
                    historical_evidence=self._context_evidence.get(best_id, 0.0),
                    allow_context_creation=allow_context_creation,
                )
            self._reset_candidate()
            return self._decision(
                status="pending", context_id=self.active_context_id,
                reason="ambiguous_historical_match", best_context_id=best_id,
                best_score=best_score, second_score=second_score,
                best_confidence=best_confidence,
                second_confidence=second_confidence,
                historical_evidence=self._context_evidence.get(best_id, 0.0),
                novelty_strength=0.0, should_update_model=False,
                should_update_memory=False,
            )

        near_ratios = {
            context_id: float(scores[context_id])
            / max(float(resolved_near[context_id]), 1e-12)
            for context_id in context_ids
        }
        near_id = max(near_ratios, key=near_ratios.get)
        if near_ratios[near_id] >= 1.0:
            self._novelty_evidence *= self.novelty_decay
            if (
                self.near_creation_enabled
                and not (
                    near_id == self.active_context_id
                    and lifecycles[near_id] == "provisional"
                )
            ):
                return self._split_candidate_decision(
                    kind="near", value=value,
                    pending_reason="near_historical_context",
                    creation_reason="near_context_evidence_confirmed",
                    best_context_id=near_id, best_score=float(scores[near_id]),
                    second_score=float("-inf"),
                    best_confidence=float(np.log(max(near_ratios[near_id], 1e-12))),
                    second_confidence=float("-inf"),
                    historical_evidence=self._context_evidence.get(near_id, 0.0),
                    allow_context_creation=allow_context_creation,
                )
            self._reset_candidate()
            if (
                near_id == self.active_context_id
                and lifecycles[near_id] == "provisional"
            ):
                return self._decision(
                    status="stay",
                    context_id=near_id,
                    reason="provisional_context_warmup",
                    best_context_id=near_id,
                    best_score=float(scores[near_id]),
                    second_score=float("-inf"),
                    best_confidence=float(
                        np.log(max(near_ratios[near_id], 1e-12))
                    ),
                    second_confidence=float("-inf"),
                    historical_evidence=0.0,
                    novelty_strength=0.0,
                    should_update_model=True,
                    should_update_memory=True,
                )
            return self._decision(
                status="pending",
                context_id=self.active_context_id,
                reason="near_historical_context",
                best_context_id=near_id,
                best_score=float(scores[near_id]),
                second_score=float("-inf"),
                best_confidence=float(
                    np.log(max(near_ratios[near_id], 1e-12))
                ),
                second_confidence=float("-inf"),
                historical_evidence=self._context_evidence.get(near_id, 0.0),
                novelty_strength=0.0,
                should_update_model=False,
                should_update_memory=False,
            )

        if (
            self.active_context_id is not None
            and lifecycles[self.active_context_id] == "provisional"
        ):
            self._novelty_evidence *= self.novelty_decay
            self._candidate_evidence *= self.candidate_evidence_decay
            return self._decision(
                status="pending",
                context_id=self.active_context_id,
                reason="provisional_context_mismatch",
                best_context_id=near_id,
                best_score=float(scores[near_id]),
                second_score=float("-inf"),
                best_confidence=float("-inf"),
                second_confidence=float("-inf"),
                historical_evidence=0.0,
                novelty_strength=0.0,
                should_update_model=False,
                should_update_memory=False,
            )

        density_ratios = [
            float(scores[context_id])
            / max(float(density_thresholds[context_id]), 1e-12)
            for context_id in context_ids
        ]
        best_density_ratio = max(density_ratios)
        novelty_strength = float(np.clip(1.0 - best_density_ratio, 0.0, 1.0))
        self._novelty_evidence = max(
            0.0,
            self.novelty_decay * self._novelty_evidence
            + novelty_strength
            - self.novelty_drift,
        )
        candidate_coherent = self._append_candidate(value, kind="novel")
        if (
            candidate_coherent
            and len(self._candidate_features) >= self.min_candidate_samples
            and self._novelty_evidence >= self.novelty_evidence_threshold
            and self._candidate_evidence >= self.candidate_evidence_threshold
        ):
            if not allow_context_creation:
                return self._decision(
                    status="pending",
                    context_id=self.active_context_id,
                    reason="novel_context_creation_disabled",
                    best_context_id=None,
                    best_score=float("-inf"),
                    second_score=float("-inf"),
                    best_confidence=float("-inf"),
                    second_confidence=float("-inf"),
                    historical_evidence=0.0,
                    novelty_strength=novelty_strength,
                    should_update_model=False,
                    should_update_memory=False,
                )
            candidate_features = np.stack(self._candidate_features, axis=0)
            self._awaiting_context_creation = True
            return self._decision(
                status="create",
                context_id=None,
                reason="novel_context_evidence_confirmed",
                best_context_id=None,
                best_score=float("-inf"),
                second_score=float("-inf"),
                best_confidence=float("-inf"),
                second_confidence=float("-inf"),
                historical_evidence=0.0,
                novelty_strength=novelty_strength,
                should_update_model=False,
                should_update_memory=False,
                candidate_features=candidate_features,
            )
        return self._decision(
            status="pending",
            context_id=self.active_context_id,
            reason="novel_context_evidence_pending",
            best_context_id=None,
            best_score=float("-inf"),
            second_score=float("-inf"),
            best_confidence=float("-inf"),
            second_confidence=float("-inf"),
            historical_evidence=0.0,
            novelty_strength=novelty_strength,
            should_update_model=False,
            should_update_memory=False,
        )

    def confirm_created_context(self, context_id: int) -> None:
        if not self._awaiting_context_creation:
            raise RuntimeError("There is no pending context creation to confirm.")
        self.active_context_id = int(context_id)
        self._context_evidence[self.active_context_id] = 0.0
        self._awaiting_context_creation = False
        self._reset_candidate()
        self._reset_temporal_evidence()
        self._switch_cooldown_remaining = self.switch_cooldown
        self._update_streak_context_id = None
        self._update_streak = 0
        self._frames_since_switch = 0

    def abort_context_creation(self) -> None:
        """Release a pending creation after a failed transactional commit."""
        if not self._awaiting_context_creation:
            return
        self._awaiting_context_creation = False

    def retire_context(
        self, context_id: int, *, fallback_context_id: int
    ) -> None:
        if self._awaiting_context_creation:
            raise RuntimeError("Cannot retire a context during context creation.")
        resolved_id = int(context_id)
        fallback_id = int(fallback_context_id)
        if resolved_id == fallback_id:
            raise ValueError("Fallback context must differ from retired context.")
        self._context_evidence.pop(resolved_id, None)
        if self.active_context_id == resolved_id:
            self.set_active_context(fallback_id)

    def set_active_context(self, context_id: int) -> None:
        if self._awaiting_context_creation:
            raise RuntimeError("Cannot override active context during context creation.")
        self.active_context_id = int(context_id)
        self._context_evidence.setdefault(self.active_context_id, 0.0)
        self._reset_candidate()
        self._reset_temporal_evidence()
        self._update_streak_context_id = None
        self._update_streak = 0
        self._frames_since_switch = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "config": {
                "input_dim": self.input_dim,
                "feature_ema_beta": self.feature_ema_beta,
                "routing_threshold_scale": self.routing_threshold_scale,
                "near_threshold_scale": self.near_threshold_scale,
                "min_score_margin": self.min_score_margin,
                "evidence_decay": self.evidence_decay,
                "evidence_miss_penalty": self.evidence_miss_penalty,
                "evidence_confidence_clip": self.evidence_confidence_clip,
                "switch_evidence_threshold": self.switch_evidence_threshold,
                "switch_evidence_margin": self.switch_evidence_margin,
                "switch_cooldown": self.switch_cooldown,
                "novelty_decay": self.novelty_decay,
                "novelty_drift": self.novelty_drift,
                "novelty_evidence_threshold": self.novelty_evidence_threshold,
                "min_candidate_samples": self.min_candidate_samples,
                "candidate_evidence_threshold": self.candidate_evidence_threshold,
                "candidate_similarity_threshold": self.candidate_similarity_threshold,
                "candidate_evidence_decay": self.candidate_evidence_decay,
                "candidate_mismatch_patience": self.candidate_mismatch_patience,
                "candidate_n_rff_features": self.candidate_n_rff_features,
                "candidate_bandwidth": self.candidate_bandwidth,
                "random_state": self.random_state,
                "candidate_max_size": self.candidate_max_size,
                "ambiguous_creation_enabled": self.ambiguous_creation_enabled,
                "near_creation_enabled": self.near_creation_enabled,
                "boundary_creation_enabled": self.boundary_creation_enabled,
                "split_min_candidate_samples": self.split_min_candidate_samples,
                "split_evidence_threshold": self.split_evidence_threshold,
                "strict_update_gate_enabled": self.strict_update_gate_enabled,
                "update_min_confidence": self.update_min_confidence,
                "update_min_margin": self.update_min_margin,
                "update_min_consecutive_accepts": self.update_min_consecutive_accepts,
                "update_after_switch_warmup": self.update_after_switch_warmup,
                "freeze_base_context_memory": self.freeze_base_context_memory,
                "update_provisional_memory": self.update_provisional_memory,
                "update_provisional_model": self.update_provisional_model,
                "dtype": self.dtype.str,
            },
            "active_context_id": self.active_context_id,
            "context_evidence": dict(self._context_evidence),
            "novelty_evidence": self._novelty_evidence,
            "candidate_evidence": self._candidate_evidence,
            "candidate_kind": self._candidate_kind,
            "candidate_mismatch_count": self._candidate_mismatch_count,
            "candidate_features": [x.copy() for x in self._candidate_features],
            "candidate_kde": (
                None if self._candidate_kde is None else self._candidate_kde.state_dict()
            ),
            "smoothed_feature": (
                None if self._smoothed_feature is None else self._smoothed_feature.copy()
            ),
            "switch_cooldown_remaining": self._switch_cooldown_remaining,
            "awaiting_context_creation": self._awaiting_context_creation,
            "update_streak_context_id": self._update_streak_context_id,
            "update_streak": self._update_streak,
            "frames_since_switch": self._frames_since_switch,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ContextRouter":
        version = int(state.get("version", 1))
        if version not in (1, 2, 3, cls.STATE_VERSION):
            raise ValueError(f"Unsupported ContextRouter state version: {version}")
        config = dict(state["config"])
        obj = cls(**config)
        active_id = state.get("active_context_id", None)
        obj.active_context_id = None if active_id is None else int(active_id)
        if version == 1:
            switch_id = state.get("switch_target_id", None)
            obj._context_evidence = {}
            if switch_id is not None:
                obj._context_evidence[int(switch_id)] = 0.0
            obj._novelty_evidence = 0.0
            obj._candidate_evidence = 0.0
            obj._candidate_mismatch_count = 0
            obj._smoothed_feature = None
            obj._switch_cooldown_remaining = 0
        else:
            obj._context_evidence = {
                int(context_id): float(value)
                for context_id, value in state.get("context_evidence", {}).items()
            }
            if any(value < 0 or not np.isfinite(value) for value in obj._context_evidence.values()):
                raise ValueError("Context evidence must be finite and nonnegative.")
            obj._novelty_evidence = float(state.get("novelty_evidence", 0.0))
            obj._candidate_evidence = float(state.get("candidate_evidence", 0.0))
            obj._candidate_mismatch_count = int(
                state.get("candidate_mismatch_count", 0)
            )
            obj._candidate_kind = state.get("candidate_kind", None)
            smoothed = state.get("smoothed_feature", None)
            obj._smoothed_feature = (
                None if smoothed is None else obj._validate_x(smoothed).copy()
            )
            obj._switch_cooldown_remaining = int(
                state.get("switch_cooldown_remaining", 0)
            )
        if version == 1:
            obj._candidate_kind = None
        if obj._candidate_kind not in {
            None, "novel", "ambiguous", "near", "boundary"
        }:
            raise ValueError("Unsupported candidate kind in Router state.")
        obj._candidate_features = [
            obj._validate_x(value).copy()
            for value in state.get("candidate_features", [])
        ]
        candidate_kde = state.get("candidate_kde", None)
        obj._candidate_kde = (
            None
            if candidate_kde is None
            else OnlineRFFKDE.from_state_dict(candidate_kde)
        )
        if obj._candidate_kde is not None:
            if obj._candidate_kde.input_dim != obj.input_dim:
                raise ValueError("Candidate KDE input dimension does not match router.")
            if obj._candidate_kde.n_features != obj.candidate_n_rff_features:
                raise ValueError("Candidate KDE feature count does not match router.")
            if obj._candidate_kde.bandwidth != obj.candidate_bandwidth:
                raise ValueError("Candidate KDE bandwidth does not match router.")
            if obj._candidate_kde.dtype != obj.dtype:
                raise ValueError("Candidate KDE dtype does not match router.")
        for name, value in (
            ("novelty_evidence", obj._novelty_evidence),
            ("candidate_evidence", obj._candidate_evidence),
            ("candidate_mismatch_count", obj._candidate_mismatch_count),
            ("switch_cooldown_remaining", obj._switch_cooldown_remaining),
        ):
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative.")
        obj._awaiting_context_creation = bool(
            state.get("awaiting_context_creation", False)
        )
        streak_id = state.get("update_streak_context_id", None)
        obj._update_streak_context_id = (
            None if streak_id is None else int(streak_id)
        )
        obj._update_streak = int(state.get("update_streak", 0))
        obj._frames_since_switch = int(state.get("frames_since_switch", 0))
        if obj._update_streak < 0 or obj._frames_since_switch < 0:
            raise ValueError("Update gate counters must be nonnegative.")
        return obj
