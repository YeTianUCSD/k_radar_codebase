"""Automatic scene discovery for superposition-based online adaptation."""

from .checkpoint import (
    build_checkpoint_payload,
    load_context_checkpoint,
    restore_model_and_context_state,
    save_context_checkpoint,
    validate_optimizer_resume,
)
from .context_memory import ContextEntry, ContextMemory
from .context_router import ContextRouter, RouteDecision, rank_eligible_contexts
from .context_transaction import create_context_transaction
from .dynamic_psp import (
    DynamicContextResult,
    activate_context_residuals,
    add_context_optimizer_group,
    add_dynamic_context,
    get_context_manifest,
    restore_dynamic_contexts,
    remove_context_optimizer_group,
    remove_dynamic_context,
)
from .feature_projector import FeatureProjectorInfo, RandomFeatureProjector
from .model_adapter import (
    ContextAwareModelAdapter,
    ResidualTrainingInfo,
    configure_residual_only_training,
    freeze_encoder_batch_stats,
)
from .rff_kde import OnlineRFFKDE, RFFKDEInfo


__all__ = [
    "ContextAwareModelAdapter",
    "ContextEntry",
    "ContextMemory",
    "ContextRouter",
    "DynamicContextResult",
    "FeatureProjectorInfo",
    "OnlineRFFKDE",
    "RFFKDEInfo",
    "RandomFeatureProjector",
    "ResidualTrainingInfo",
    "RouteDecision",
    "activate_context_residuals",
    "add_context_optimizer_group",
    "add_dynamic_context",
    "build_checkpoint_payload",
    "configure_residual_only_training",
    "create_context_transaction",
    "freeze_encoder_batch_stats",
    "get_context_manifest",
    "load_context_checkpoint",
    "rank_eligible_contexts",
    "remove_context_optimizer_group",
    "remove_dynamic_context",
    "restore_dynamic_contexts",
    "restore_model_and_context_state",
    "save_context_checkpoint",
    "validate_optimizer_resume",
]
