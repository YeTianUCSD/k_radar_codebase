"""Data-driven Gaussian-kernel bandwidth selection."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def median_pairwise_distance(
    features: Any,
    *,
    max_samples: int = 512,
    random_state: int = 20260812,
) -> float:
    """Return the median nonzero pairwise Euclidean distance."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("features must have shape [N, D] with N >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("features contain NaN or infinite values")
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    if values.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(values.shape[0], max_samples, replace=False))
        values = values[indices]
    squared_norms = np.sum(np.square(values), axis=1)
    squared = (
        squared_norms[:, None] + squared_norms[None, :]
        - 2.0 * values @ values.T
    )
    upper = np.sqrt(np.maximum(squared[np.triu_indices(len(values), 1)], 0.0))
    positive = upper[upper > np.finfo(np.float64).eps]
    if positive.size == 0:
        raise ValueError("Cannot estimate bandwidth from identical features")
    return float(np.median(positive))


def resolve_bandwidth(
    kde_config: Mapping[str, Any],
    features: Any,
) -> tuple[float, dict[str, Any]]:
    """Resolve fixed or median-heuristic bandwidth and return diagnostics."""
    mode = str(kde_config.get("BANDWIDTH_MODE", "fixed")).lower()
    scale = float(kde_config.get("BANDWIDTH_SCALE", 1.0))
    minimum = float(kde_config.get("BANDWIDTH_MIN", 1e-3))
    maximum = float(kde_config.get("BANDWIDTH_MAX", 1e3))
    if scale <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("Invalid bandwidth scale or bounds")
    if mode == "fixed":
        base = float(kde_config.get("BANDWIDTH", 1.0))
    elif mode == "median":
        base = median_pairwise_distance(
            features,
            max_samples=int(kde_config.get("BANDWIDTH_MAX_SAMPLES", 512)),
            random_state=int(kde_config.get("SEED", 20260812)),
        )
    else:
        raise ValueError(f"Unsupported BANDWIDTH_MODE: {mode}")
    resolved = float(np.clip(base * scale, minimum, maximum))
    return resolved, {
        "mode": mode,
        "base_bandwidth": float(base),
        "scale": scale,
        "minimum": minimum,
        "maximum": maximum,
        "resolved_bandwidth": resolved,
    }
