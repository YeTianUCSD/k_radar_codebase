import numpy as np

from scene_discovery.memory_context import MemoryBankContextManager


def test_memory_manager_creates_and_reuses_disjoint_context():
    manager = MemoryBankContextManager(
        threshold=0.5,
        create_persistence=2,
        switch_persistence=1,
        memory_size=8,
    )
    first = manager.initialize_context(np.asarray([[0.0, 0.0], [0.1, 0.0]]))
    pending = manager.observe(np.asarray([5.0, 5.0], dtype=np.float32))
    created = manager.observe(np.asarray([5.1, 5.0], dtype=np.float32))
    reused = manager.observe(np.asarray([0.05, 0.0], dtype=np.float32))
    assert pending.candidate_size == 1
    assert created.is_new
    assert created.context_id != first
    assert reused.context_id == first
    assert len(manager.memories) == 2


def test_memory_manager_does_not_create_from_inconsistent_outliers():
    manager = MemoryBankContextManager(
        threshold=0.5,
        candidate_threshold=0.4,
        create_persistence=2,
    )
    manager.initialize_context(np.asarray([[0.0, 0.0]]))
    manager.observe(np.asarray([5.0, 5.0], dtype=np.float32))
    result = manager.observe(np.asarray([-5.0, -5.0], dtype=np.float32))
    assert result.candidate_size == 1
    assert len(manager.memories) == 1


def test_memory_manager_respects_memory_bound():
    manager = MemoryBankContextManager(
        threshold=10.0,
        memory_size=3,
        memory_min_separation_ratio=0.0,
    )
    context = manager.initialize_context(np.arange(10, dtype=np.float32)[:, None])
    assert len(manager.memories[context]) == 3
    for value in [0.5, 1.5, 2.5, 3.5]:
        manager.observe(np.asarray([value], dtype=np.float32))
    assert len(manager.memories[context]) <= 3
