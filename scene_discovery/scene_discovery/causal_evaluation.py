"""Event-level evaluation for causal scene-boundary and context decisions."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def true_transition_events(
    frame: pd.DataFrame,
    initial_scene: str,
) -> pd.DataFrame:
    """Return true transitions for scoring; these rows never enter the manager."""
    required = {
        "stream_block", "stream_true_sequence", "stream_visit",
        "stream_expected_novel",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"stream frame is missing columns: {sorted(missing)}")
    rows = []
    previous = str(initial_scene)
    for block_id, block in frame.reset_index(drop=True).groupby("stream_block", sort=True):
        scene = str(block["stream_true_sequence"].iloc[0])
        boundary_index = int(block.index[0])
        if scene != previous:
            rows.append({
                "true_event_id": len(rows),
                "stream_block": int(block_id),
                "boundary_index": boundary_index,
                "sequence": scene,
                "visit": str(block["stream_visit"].iloc[0]),
                "expected_novel": bool(block["stream_expected_novel"].iloc[0]),
            })
        previous = scene
    return pd.DataFrame(rows, columns=[
        "true_event_id", "stream_block", "boundary_index", "sequence",
        "visit", "expected_novel",
    ])


def predicted_transition_events(decisions: Sequence[object]) -> pd.DataFrame:
    rows = []
    for decision in decisions:
        if not decision.boundary_event:
            continue
        rows.append({
            "predicted_event_id": len(rows),
            "detected_index": int(decision.observation_index),
            "estimated_boundary_index": int(decision.estimated_boundary_index),
            "action": str(decision.action),
            "context_id": int(decision.context_id),
            "previous_context": int(decision.previous_context),
            "created_context": int(decision.created_context),
        })
    return pd.DataFrame(rows, columns=[
        "predicted_event_id", "detected_index", "estimated_boundary_index",
        "action", "context_id", "previous_context", "created_context",
    ])


def match_causal_boundaries(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    tolerance: int,
) -> Tuple[dict, set]:
    """Greedily match one prediction after each boundary within a causal window."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    matches = {}
    used = set()
    true_rows = list(truth.itertuples(index=False))
    for position, true_row in enumerate(true_rows):
        next_boundary = (
            int(true_rows[position + 1].boundary_index)
            if position + 1 < len(true_rows) else np.iinfo(np.int64).max
        )
        causal_stop = min(int(true_row.boundary_index) + tolerance, next_boundary - 1)
        candidates = predictions[
            predictions["detected_index"].between(
                int(true_row.boundary_index), causal_stop,
            )
            & ~predictions["predicted_event_id"].isin(used)
        ]
        if candidates.empty:
            continue
        chosen = candidates.sort_values("detected_index").iloc[0]
        prediction_id = int(chosen["predicted_event_id"])
        matches[int(true_row.true_event_id)] = prediction_id
        used.add(prediction_id)
    return matches, used


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def evaluate_causal_events(
    frame: pd.DataFrame,
    decisions: Sequence[object],
    initial_scene_contexts: Mapping[str, int],
    tolerance: int,
    final_context_count: int,
):
    """Score boundaries and strict one-to-one CREATE/REUSE event outcomes."""
    if len(initial_scene_contexts) != 1:
        raise ValueError("V4 event scoring currently requires one initial scene")
    initial_scene, initial_context = next(iter(initial_scene_contexts.items()))
    truth = true_transition_events(frame, str(initial_scene))
    predictions = predicted_transition_events(decisions)
    matches, used_predictions = match_causal_boundaries(truth, predictions, tolerance)
    registry = {str(initial_scene): int(initial_context)}
    reverse_registry = {int(initial_context): str(initial_scene)}
    rows = []
    successful_events = 0
    correct_creations = 0
    correct_reuses = 0
    registry_collisions = 0

    for position, true_row in enumerate(truth.itertuples(index=False)):
        event_id = int(true_row.true_event_id)
        matched_id = matches.get(event_id)
        matched = matched_id is not None
        predicted = (
            predictions[predictions["predicted_event_id"].eq(matched_id)].iloc[0]
            if matched else None
        )
        segment_stop = (
            int(truth.iloc[position + 1]["boundary_index"])
            if position + 1 < len(truth) else len(frame)
        )
        segment_creations = predictions[
            predictions["detected_index"].between(
                int(true_row.boundary_index), segment_stop - 1
            )
            & predictions["action"].eq("create")
        ]
        creation_count = len(segment_creations)
        action = str(predicted["action"]) if matched else "missed"
        context = int(predicted["context_id"]) if matched else -1
        delay = (
            int(predicted["detected_index"] - true_row.boundary_index)
            if matched else np.nan
        )
        scene = str(true_row.sequence)
        expected_context = registry.get(scene)
        collision = False
        exact_success = False

        if bool(true_row.expected_novel):
            unique_context = context not in reverse_registry if matched else False
            collision = bool(matched and not unique_context)
            exact_success = bool(
                matched and action == "create" and creation_count == 1
                and unique_context
            )
            if matched and action == "create" and unique_context:
                registry[scene] = context
                reverse_registry[context] = scene
            elif matched and context in reverse_registry:
                registry_collisions += 1
            correct_creations += int(exact_success)
        else:
            exact_success = bool(
                matched and action == "reuse" and expected_context is not None
                and context == expected_context and creation_count == 0
            )
            correct_reuses += int(exact_success)
        successful_events += int(exact_success)
        rows.append({
            **true_row._asdict(),
            "matched_boundary": matched,
            "predicted_event_id": matched_id if matched else -1,
            "detected_index": int(predicted["detected_index"]) if matched else -1,
            "detection_delay_windows": delay,
            "predicted_action": action,
            "predicted_context": context,
            "expected_context": expected_context if expected_context is not None else -1,
            "creation_events_in_true_segment": creation_count,
            "registry_collision": collision,
            "exact_event_success": exact_success,
        })

    event_frame = pd.DataFrame(rows)
    true_count = len(truth)
    predicted_count = len(predictions)
    matched_count = len(matches)
    precision = _safe_rate(matched_count, predicted_count)
    recall = _safe_rate(matched_count, true_count)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall > 0.0 else 0.0
    )
    novel_count = int(truth["expected_novel"].sum()) if len(truth) else 0
    known_count = true_count - novel_count
    matched_delays = event_frame.loc[
        event_frame["matched_boundary"], "detection_delay_windows"
    ].astype(float)
    true_scenes = set(frame["stream_true_sequence"].astype(str)) | {str(initial_scene)}
    metrics = {
        "true_boundary_events": true_count,
        "predicted_boundary_events": predicted_count,
        "matched_boundary_events": matched_count,
        "false_boundary_events": predicted_count - matched_count,
        "missed_boundary_events": true_count - matched_count,
        "change_precision": precision,
        "change_recall": recall,
        "change_f1": f1,
        "mean_detection_delay_windows": (
            float(matched_delays.mean()) if len(matched_delays) else float("nan")
        ),
        "median_detection_delay_windows": (
            float(matched_delays.median()) if len(matched_delays) else float("nan")
        ),
        "novel_transition_events": novel_count,
        "known_transition_events": known_count,
        "correct_creation_events": correct_creations,
        "correct_creation_rate": _safe_rate(correct_creations, novel_count),
        "correct_reuse_events": correct_reuses,
        "correct_reuse_rate": _safe_rate(correct_reuses, known_count),
        "end_to_end_success_events": successful_events,
        "end_to_end_success_rate": _safe_rate(successful_events, true_count),
        "registry_collisions": registry_collisions,
        "registered_scene_count": len(registry),
        "final_contexts": int(final_context_count),
        "true_contexts": len(true_scenes),
        "context_count_error": int(final_context_count - len(true_scenes)),
    }
    if len(predictions):
        predictions = predictions.copy()
        predictions["matched_true_event_id"] = predictions["predicted_event_id"].map(
            {prediction: truth_id for truth_id, prediction in matches.items()}
        ).fillna(-1).astype(int)
        predictions["is_false_boundary"] = ~predictions["predicted_event_id"].isin(
            used_predictions
        )
    return metrics, event_frame, predictions
