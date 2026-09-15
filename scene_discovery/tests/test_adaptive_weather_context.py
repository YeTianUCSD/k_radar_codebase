import numpy as np

from scene_discovery.adaptive_weather_context import AdaptiveWeatherContextManager


def _manager(**overrides):
    settings = dict(
        change_threshold=0.5,
        match_threshold_ratio=1.2,
        change_persistence=2,
        candidate_min_windows=4,
        candidate_max_windows=8,
        provisional_windows=2,
        cooldown_windows=3,
        reference_windows=6,
        recent_windows=2,
        distribution_threshold_ratio=0.2,
        strong_departure_ratio=1.2,
        coverage_threshold=0.75,
        match_margin=0.1,
        memory_size=8,
        neighbors=1,
    )
    settings.update(overrides)
    return AdaptiveWeatherContextManager(**settings)


def test_new_context_is_provisional_before_one_create():
    manager = _manager()
    initial = manager.initialize_context(np.asarray([[0.0], [0.1]], dtype=np.float32))
    decisions = [manager.observe(np.asarray([value], dtype=np.float32))
                 for value in [5.0, 5.1, 5.0, 5.1, 5.05, 5.0, 5.1]]
    creates = [decision for decision in decisions if decision.action == "create"]
    assert initial == 0
    assert len(creates) == 1
    assert creates[0].candidate_size >= manager.candidate_min_windows
    assert any(decision.state == "provisional" for decision in decisions)
    assert len(manager.memories) == 2


def test_known_context_is_reused_after_segment_evidence():
    manager = _manager(cooldown_windows=0)
    first = manager.initialize_context(np.asarray([[0.0], [0.1]], dtype=np.float32))
    second = manager.initialize_context(np.asarray([[5.0], [5.1]], dtype=np.float32))
    manager.current_context = first
    decisions = [manager.observe(np.asarray([value], dtype=np.float32))
                 for value in [5.0, 5.1, 5.0, 5.1]]
    assert decisions[-1].boundary_event
    assert decisions[-1].action == "reuse"
    assert decisions[-1].context_id == second
    assert len(manager.memories) == 2


def test_transient_departure_returns_to_current_without_create():
    manager = _manager(candidate_min_windows=5, candidate_max_windows=8)
    context = manager.initialize_context(np.asarray([[0.0], [0.1]], dtype=np.float32))
    decisions = [manager.observe(np.asarray([value], dtype=np.float32))
                 for value in [3.0, 0.05, 0.0, 0.1]]
    assert not any(decision.boundary_event for decision in decisions)
    assert manager.current_context == context
    assert len(manager.memories) == 1


def test_cooldown_prevents_immediate_duplicate_creation():
    manager = _manager(cooldown_windows=4)
    manager.initialize_context(np.asarray([[0.0], [0.1]], dtype=np.float32))
    decisions = [manager.observe(np.asarray([value], dtype=np.float32))
                 for value in [5.0, 5.1, 5.0, 5.1, 5.05, 8.0, 8.1, 8.0]]
    assert sum(decision.action == "create" for decision in decisions) == 1
    assert decisions[-1].action == "cooldown"
