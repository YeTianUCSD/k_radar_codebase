from scene_discovery.attribute_registry import SemanticContextRegistry


def confidences(value=0.95):
    return {"weather": value, "road": value, "lighting": value}


def test_registry_creates_only_after_persistence_and_reuses_known_key():
    registry = SemanticContextRegistry(
        confidence_thresholds=confidences(0.8),
        create_persistence=2,
        switch_persistence=1,
    )
    base = registry.register("normal|urban|night")
    pending = registry.observe("heavy_snow|highway|night", confidences())
    created = registry.observe("heavy_snow|highway|night", confidences())
    reused = registry.observe("normal|urban|night", confidences())
    assert pending.context_id == base
    assert pending.pending_count == 1
    assert created.created and not created.reused
    assert reused.reused and reused.context_id == base
    assert len(registry.key_to_context) == 2


def test_registry_rejects_low_confidence_without_creating_context():
    registry = SemanticContextRegistry(
        confidence_thresholds=confidences(0.8),
        create_persistence=2,
        switch_persistence=2,
    )
    base = registry.register("normal|urban|night")
    decision = registry.observe(
        "heavy_snow|highway|night",
        {"weather": 0.99, "road": 0.4, "lighting": 0.99},
    )
    assert not decision.accepted
    assert decision.context_id == base
    assert len(registry.key_to_context) == 1


def test_registry_resets_candidate_when_semantic_key_changes():
    registry = SemanticContextRegistry(
        confidence_thresholds=confidences(0.8),
        create_persistence=2,
        switch_persistence=1,
    )
    registry.register("normal|urban|night")
    first = registry.observe("heavy_snow|highway|night", confidences())
    second = registry.observe("fog|mountain|day", confidences())
    assert first.pending_count == 1
    assert second.pending_count == 1
    assert len(registry.key_to_context) == 1


def test_empty_registry_waits_for_confident_persistence_before_bootstrap():
    registry = SemanticContextRegistry(
        confidence_thresholds=confidences(0.8),
        create_persistence=2,
        switch_persistence=1,
    )
    rejected = registry.observe("normal|urban|night", confidences(0.4))
    pending = registry.observe("normal|urban|night", confidences())
    created = registry.observe("normal|urban|night", confidences())
    assert rejected.context_id is None and rejected.semantic_key is None
    assert not rejected.accepted
    assert pending.context_id is None and pending.pending_count == 1
    assert created.created and created.context_id == 0
    assert registry.current_context == 0
