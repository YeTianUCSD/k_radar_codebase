"""Shared configuration and construction helpers for the new pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
import yaml

from .context_memory import ContextMemory
from .context_router import ContextRouter
from .feature_projector import RandomFeatureProjector


PathLike = Union[str, Path]


def load_yaml(path: PathLike) -> dict[str, Any]:
    with Path(path).expanduser().open("r") as source:
        payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def write_single_context_runtime_config(
    base_config: PathLike,
    *,
    model_context_name: str,
    output_root: Optional[PathLike] = None,
    run_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    enable_logging: Optional[bool] = None,
    enable_validation: Optional[bool] = None,
) -> tuple[str, dict[str, Any]]:
    """Create a full temporary ASF config containing only the base context."""
    config = load_yaml(base_config)
    resolved_name = str(model_context_name)
    if not resolved_name:
        raise ValueError("model_context_name must not be empty.")
    try:
        superposition = config["MODEL"]["SUPERPOSITION"]
    except KeyError as error:
        raise KeyError("Base config has no MODEL.SUPERPOSITION section.") from error
    if not bool(superposition.get("ENABLED", False)):
        raise ValueError("Base config must enable MODEL.SUPERPOSITION.")
    superposition["SCENE_LIST"] = [resolved_name]
    superposition["BASE_SCENE"] = resolved_name
    superposition["ACTIVE_SCENE"] = resolved_name

    if output_root is not None:
        config["GENERAL"]["LOGGING"]["PATH_LOGGING"] = str(output_root)
    if run_name is not None:
        config["GENERAL"]["NAME"] = str(run_name)
    if batch_size is not None:
        config["OPTIMIZER"]["BATCH_SIZE"] = int(batch_size)
    if num_workers is not None:
        config["OPTIMIZER"]["NUM_WORKERS"] = int(num_workers)
    if enable_logging is not None:
        config["GENERAL"]["LOGGING"]["IS_LOGGING"] = bool(enable_logging)
        if not enable_logging:
            config["GENERAL"]["LOGGING"]["IS_SAVE_MODEL"] = False
    if enable_validation is not None:
        config["VAL"]["IS_VALIDATE"] = bool(enable_validation)

    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yml",
        prefix="kradar_auto_context_",
        delete=False,
    )
    with temporary:
        yaml.safe_dump(config, temporary, sort_keys=False)
    return temporary.name, config


def load_model_checkpoint(
    network: torch.nn.Module,
    checkpoint_path: PathLike,
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load raw, utility, or context checkpoint model weights."""
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a model state dictionary.")
    incompatible = network.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def build_projector(config: Mapping[str, Any]) -> RandomFeatureProjector:
    feature = config["FEATURE"]
    projection = config["PROJECTION"]
    return RandomFeatureProjector(
        feature_keys=feature["KEYS"],
        projection_dim=int(projection.get("DIM", 2)),
        pooling=feature.get("POOLING", ["mean", "std"]),
        random_state=int(projection.get("SEED", 20260812)),
        eps=float(projection.get("EPS", 1e-6)),
        dtype=np.dtype(projection.get("DTYPE", "float32")),
    )


def build_context_memory(
    config: Mapping[str, Any],
    *,
    input_dim: int,
    bandwidth_override: Optional[float] = None,
) -> ContextMemory:
    kde = config["KDE"]
    memory = config.get("CONTEXT_MEMORY", {})
    return ContextMemory(
        input_dim=input_dim,
        n_rff_features=int(kde.get("NUM_RFF_FEATURES", 2048)),
        bandwidth=(
            float(kde.get("BANDWIDTH", 1.0))
            if bandwidth_override is None
            else float(bandwidth_override)
        ),
        random_state=int(kde.get("SEED", 20260812)),
        threshold_quantile=float(kde.get("THRESHOLD_QUANTILE", 0.05)),
        update_threshold_quantile=float(
            kde.get("UPDATE_THRESHOLD_QUANTILE", kde.get("THRESHOLD_QUANTILE", 0.05))
        ),
        default_threshold=float(kde.get("DEFAULT_THRESHOLD", 0.1)),
        calibration_history_size=int(kde.get("CALIBRATION_HISTORY_SIZE", 512)),
        min_calibration_samples=int(kde.get("MIN_CALIBRATION_SAMPLES", 8)),
        provisional_min_samples=int(memory.get("PROVISIONAL_MIN_SAMPLES", 24)),
        provisional_max_frames=int(memory.get("PROVISIONAL_MAX_FRAMES", 48)),
        dtype=np.dtype(config["PROJECTION"].get("DTYPE", "float32")),
    )


def build_router(
    config: Mapping[str, Any],
    *,
    input_dim: int,
    candidate_bandwidth_override: Optional[float] = None,
) -> ContextRouter:
    router = config["ROUTER"]
    return ContextRouter(
        input_dim=input_dim,
        feature_ema_beta=float(router.get("FEATURE_EMA_BETA", 0.5)),
        routing_threshold_scale=float(
            router.get("ROUTING_THRESHOLD_SCALE", 0.85)
        ),
        near_threshold_scale=float(router.get("NEAR_THRESHOLD_SCALE", 0.60)),
        min_score_margin=float(router.get("MIN_SCORE_MARGIN", 0.02)),
        evidence_decay=float(router.get("EVIDENCE_DECAY", 0.7)),
        evidence_miss_penalty=float(
            router.get("EVIDENCE_MISS_PENALTY", 0.2)
        ),
        evidence_confidence_clip=float(
            router.get("EVIDENCE_CONFIDENCE_CLIP", 1.0)
        ),
        switch_evidence_threshold=float(
            router.get("SWITCH_EVIDENCE_THRESHOLD", 0.35)
        ),
        switch_evidence_margin=float(
            router.get("SWITCH_EVIDENCE_MARGIN", 0.10)
        ),
        switch_cooldown=int(router.get("SWITCH_COOLDOWN", 2)),
        novelty_decay=float(router.get("NOVELTY_DECAY", 0.8)),
        novelty_drift=float(router.get("NOVELTY_DRIFT", 0.2)),
        novelty_evidence_threshold=float(
            router.get("NOVELTY_EVIDENCE_THRESHOLD", 2.5)
        ),
        min_candidate_samples=int(router.get("MIN_CANDIDATE_SAMPLES", 3)),
        candidate_evidence_threshold=float(
            router.get("CANDIDATE_EVIDENCE_THRESHOLD", 1.5)
        ),
        candidate_similarity_threshold=float(
            router.get("CANDIDATE_SIMILARITY_THRESHOLD", 0.3)
        ),
        candidate_evidence_decay=float(
            router.get("CANDIDATE_EVIDENCE_DECAY", 0.5)
        ),
        candidate_mismatch_patience=int(
            router.get("CANDIDATE_MISMATCH_PATIENCE", 2)
        ),
        candidate_n_rff_features=int(
            router.get("CANDIDATE_NUM_RFF_FEATURES", 1024)
        ),
        candidate_bandwidth=(
            float(router.get("CANDIDATE_BANDWIDTH", 1.0))
            if candidate_bandwidth_override is None
            else float(candidate_bandwidth_override)
        ),
        random_state=int(router.get("SEED", 20260812)),
        candidate_max_size=int(router.get("CANDIDATE_MAX_SIZE", 64)),
        ambiguous_creation_enabled=bool(
            router.get("AMBIGUOUS_CREATION_ENABLED", False)
        ),
        near_creation_enabled=bool(router.get("NEAR_CREATION_ENABLED", False)),
        boundary_creation_enabled=bool(
            router.get("BOUNDARY_CREATION_ENABLED", False)
        ),
        split_min_candidate_samples=int(
            router.get("SPLIT_MIN_SAMPLES", 8)
        ),
        split_evidence_threshold=float(
            router.get("SPLIT_EVIDENCE_THRESHOLD", 1.0)
        ),
        strict_update_gate_enabled=bool(
            router.get("STRICT_UPDATE_GATE_ENABLED", False)
        ),
        update_min_confidence=float(router.get("UPDATE_MIN_CONFIDENCE", 0.2)),
        update_min_margin=float(router.get("UPDATE_MIN_MARGIN", 0.15)),
        update_min_consecutive_accepts=int(
            router.get("UPDATE_MIN_CONSECUTIVE_ACCEPTS", 4)
        ),
        update_after_switch_warmup=int(
            router.get("UPDATE_AFTER_SWITCH_WARMUP", 4)
        ),
        freeze_base_context_memory=bool(
            router.get("FREEZE_BASE_CONTEXT_MEMORY", False)
        ),
        update_provisional_memory=bool(
            router.get("UPDATE_PROVISIONAL_MEMORY", True)
        ),
        update_provisional_model=bool(
            router.get("UPDATE_PROVISIONAL_MODEL", False)
        ),
        dtype=np.dtype(config["PROJECTION"].get("DTYPE", "float32")),
    )
