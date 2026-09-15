from types import SimpleNamespace

import numpy as np
import pandas as pd

from scene_discovery.causal_context import CausalContextManager
from scene_discovery.causal_evaluation import evaluate_causal_events, match_causal_boundaries


def test_sustained_unknown_creates_one_causal_boundary():
    manager = CausalContextManager(
        change_threshold=0.5,
        match_threshold=0.7,
        change_persistence=2,
        memory_size=8,
        neighbors=1,
    )
    first = manager.initialize_context(np.asarray([[0.0, 0.0]], dtype=np.float32))
    pending = manager.observe(np.asarray([5.0, 5.0], dtype=np.float32))
    created = manager.observe(np.asarray([5.1, 5.0], dtype=np.float32))
    assert not pending.boundary_event
    assert pending.state == "suspect"
    assert created.boundary_event
    assert created.action == "create"
    assert created.context_id != first
    assert created.estimated_boundary_index == 0


def test_candidate_matching_current_expands_without_false_boundary():
    manager = CausalContextManager(
        change_threshold=0.5,
        match_threshold=1.0,
        change_persistence=2,
        memory_size=8,
        neighbors=1,
    )
    context = manager.initialize_context(np.asarray([[0.0]], dtype=np.float32))
    manager.observe(np.asarray([0.8], dtype=np.float32))
    expanded = manager.observe(np.asarray([0.85], dtype=np.float32))
    assert expanded.action == "expand"
    assert not expanded.boundary_event
    assert expanded.context_id == context
    assert len(manager.memories) == 1


def test_sustained_known_context_switch_emits_reuse():
    manager = CausalContextManager(
        change_threshold=0.5,
        match_threshold=0.8,
        change_persistence=2,
        memory_size=8,
        neighbors=1,
    )
    first = manager.initialize_context(np.asarray([[0.0]], dtype=np.float32))
    second = manager.initialize_context(np.asarray([[5.0]], dtype=np.float32))
    manager.current_context = first
    manager.observe(np.asarray([5.1], dtype=np.float32))
    reused = manager.observe(np.asarray([4.9], dtype=np.float32))
    assert reused.boundary_event
    assert reused.action == "reuse"
    assert reused.context_id == second


def _decision(index, boundary=False, action="stay", context=0, created=-1):
    return SimpleNamespace(
        observation_index=index,
        boundary_event=boundary,
        estimated_boundary_index=index,
        action=action,
        context_id=context,
        previous_context=0,
        created_context=created,
    )


def _three_transition_frame():
    return pd.DataFrame({
        "stream_block": [0] * 5 + [1] * 5 + [2] * 5,
        "stream_true_sequence": ["5"] * 5 + ["1"] * 5 + ["5"] * 5,
        "stream_visit": ["first"] * 5 + ["revisit"] * 10,
        "stream_expected_novel": [True] * 5 + [False] * 10,
    })


def test_event_metrics_require_exact_create_and_correct_reuse():
    decisions = [_decision(index) for index in range(15)]
    decisions[1] = _decision(1, True, "create", 1, 1)
    decisions[6] = _decision(6, True, "reuse", 0)
    decisions[11] = _decision(11, True, "reuse", 1)
    metrics, events, predicted = evaluate_causal_events(
        _three_transition_frame(), decisions, {"1": 0}, tolerance=2,
        final_context_count=2,
    )
    assert metrics["change_f1"] == 1.0
    assert metrics["correct_creation_rate"] == 1.0
    assert metrics["correct_reuse_rate"] == 1.0
    assert metrics["end_to_end_success_rate"] == 1.0
    assert events["exact_event_success"].all()
    assert not predicted["is_false_boundary"].any()


def test_duplicate_creation_fails_strict_creation_metric():
    decisions = [_decision(index) for index in range(15)]
    decisions[1] = _decision(1, True, "create", 1, 1)
    decisions[3] = _decision(3, True, "create", 2, 2)
    metrics, events, _ = evaluate_causal_events(
        _three_transition_frame(), decisions, {"1": 0}, tolerance=2,
        final_context_count=3,
    )
    assert metrics["correct_creation_rate"] == 0.0
    assert not events.iloc[0]["exact_event_success"]
    assert events.iloc[0]["creation_events_in_true_segment"] == 2


def test_boundary_match_never_crosses_next_true_transition():
    truth = pd.DataFrame({
        "true_event_id": [0, 1],
        "boundary_index": [0, 5],
    })
    predictions = pd.DataFrame({
        "predicted_event_id": [0],
        "detected_index": [6],
    })
    matches, _ = match_causal_boundaries(truth, predictions, tolerance=10)
    assert 0 not in matches
    assert matches[1] == 0
