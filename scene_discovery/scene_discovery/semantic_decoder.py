"""Probability-based decoding of factorized attributes into semantic contexts."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .attributes import ATTRIBUTES, ATTRIBUTE_CLASSES


def prepare_semantic_registry(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize a table of known semantic combinations."""
    required = {"sequence", *ATTRIBUTES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Semantic registry is missing columns: {sorted(missing)}")
    registry = frame[["sequence", *ATTRIBUTES]].copy()
    registry["sequence"] = registry["sequence"].astype(str)
    for attribute in ATTRIBUTES:
        registry[attribute] = registry[attribute].astype(str)
        invalid = set(registry[attribute]) - set(ATTRIBUTE_CLASSES[attribute])
        if invalid:
            raise ValueError(
                f"Semantic registry has invalid {attribute} labels: {sorted(invalid)}"
            )
    registry["semantic_key"] = registry[list(ATTRIBUTES)].agg("|".join, axis=1)
    grouped = registry.groupby("semantic_key", sort=False)["sequence"].agg(list)
    rows = []
    for semantic_key, sequences in grouped.items():
        source = registry[registry["semantic_key"].eq(semantic_key)].iloc[0]
        rows.append({
            "semantic_key": semantic_key,
            **{attribute: source[attribute] for attribute in ATTRIBUTES},
            "sequences": ",".join(sequences),
            "sequence_count": len(sequences),
        })
    if not rows:
        raise ValueError("Semantic registry is empty")
    return pd.DataFrame(rows)


def validate_probability_columns(predictions: pd.DataFrame) -> None:
    missing = []
    for attribute in ATTRIBUTES:
        for label in ATTRIBUTE_CLASSES[attribute]:
            column = f"{attribute}_prob_{label}"
            if column not in predictions:
                missing.append(column)
    if missing:
        raise ValueError(
            "Predictions do not contain complete class probabilities. Rerun "
            "evaluate_independent_attributes.py after the probability-export "
            f"update. Missing columns: {missing}"
        )


def score_known_combinations(
    predictions: pd.DataFrame,
    registry: pd.DataFrame,
    weights: Mapping[str, float],
    epsilon: float = 1e-12,
):
    """Return normalized joint scores and the best two registered contexts."""
    validate_probability_columns(predictions)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    raw_weights = np.asarray([float(weights[name]) for name in ATTRIBUTES])
    if np.any(raw_weights < 0.0) or raw_weights.sum() <= 0.0:
        raise ValueError("Attribute weights must be non-negative with positive sum")
    normalized_weights = raw_weights / raw_weights.sum()

    log_scores = np.empty((len(predictions), len(registry)), dtype=np.float64)
    for candidate_position, candidate in registry.reset_index(drop=True).iterrows():
        score = np.zeros(len(predictions), dtype=np.float64)
        for attribute, weight in zip(ATTRIBUTES, normalized_weights):
            probability = predictions[
                f"{attribute}_prob_{candidate[attribute]}"
            ].to_numpy(dtype=np.float64)
            score += weight * np.log(np.clip(probability, epsilon, 1.0))
        log_scores[:, candidate_position] = score

    # Weighted geometric likelihood is absolute evidence; unlike a softmax over
    # registered keys it remains comparable when contexts are added or removed.
    scores = np.exp(log_scores)
    order = np.argsort(-scores, axis=1, kind="stable")
    best_positions = order[:, 0]
    if scores.shape[1] > 1:
        second_positions = order[:, 1]
        second_scores = scores[np.arange(len(scores)), second_positions]
    else:
        second_positions = best_positions.copy()
        second_scores = np.zeros(len(scores), dtype=np.float64)
    best_scores = scores[np.arange(len(scores)), best_positions]
    keys = registry["semantic_key"].astype(str).to_numpy()
    return pd.DataFrame({
        "constrained_semantic_key": keys[best_positions],
        "second_semantic_key": keys[second_positions],
        "known_score": best_scores,
        "second_known_score": second_scores,
        "known_margin": best_scores - second_scores,
    }), scores


def decode_semantic_contexts(
    predictions: pd.DataFrame,
    registry: pd.DataFrame,
    weights: Mapping[str, float],
    minimum_score: float = 0.0,
    minimum_margin: float = 0.0,
    require_attribute_acceptance: bool = True,
    epsilon: float = 1e-12,
):
    """Decode legal contexts and reject uncertain samples as unknown."""
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be in [0, 1]")
    if not 0.0 <= minimum_margin <= 1.0:
        raise ValueError("minimum_margin must be in [0, 1]")
    decoded, scores = score_known_combinations(
        predictions, registry, weights, epsilon
    )
    accepted = (
        decoded["known_score"].ge(minimum_score)
        & decoded["known_margin"].ge(minimum_margin)
    )
    if require_attribute_acceptance:
        columns = [f"{attribute}_accepted" for attribute in ATTRIBUTES]
        missing = [column for column in columns if column not in predictions]
        if missing:
            raise ValueError(
                "Attribute acceptance was requested but columns are missing: "
                f"{missing}"
            )
        accepted &= predictions[columns].astype(bool).all(axis=1)
    decoded["known_accepted"] = accepted
    decoded["open_set_semantic_key"] = decoded["constrained_semantic_key"].where(
        accepted, "unknown"
    )
    return decoded, scores


def candidate_score_frame(
    predictions: pd.DataFrame,
    registry: pd.DataFrame,
    scores: np.ndarray,
    anchor_columns: Sequence[str] = (
        "held_out_sequence", "sequence", "window_end_row"
    ),
) -> pd.DataFrame:
    """Convert the score matrix into an auditable long-form table."""
    if scores.shape != (len(predictions), len(registry)):
        raise ValueError("Score matrix shape does not match predictions and registry")
    anchors = [column for column in anchor_columns if column in predictions]
    rows = []
    keys = registry["semantic_key"].astype(str).tolist()
    for position in range(len(predictions)):
        base = {column: predictions.iloc[position][column] for column in anchors}
        for key, score in zip(keys, scores[position]):
            rows.append({**base, "candidate_semantic_key": key, "score": float(score)})
    return pd.DataFrame(rows)
