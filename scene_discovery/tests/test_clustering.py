import numpy as np

from scene_discovery.clustering import AgglomerativeCentroidPredictor, StreamingContextManager
from scene_discovery.evaluation import apply_mapping, hungarian_mapping


def test_hungarian_mapping_recovers_permutation():
    truth = np.asarray(["1", "1", "2", "2"])
    clusters = np.asarray([7, 7, 3, 3])
    mapping = hungarian_mapping(truth, clusters)
    assert apply_mapping(clusters, mapping).tolist() == truth.tolist()


def test_stream_manager_creates_and_reuses_contexts():
    manager = StreamingContextManager(
        threshold=0.5,
        create_persistence=2,
        switch_persistence=1,
        prototype_alpha=0.01,
    )
    decisions = []
    for value in ([0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [0.0, 0.1]):
        decisions.append(manager.observe(np.asarray(value, dtype=np.float32)))
    assert len(manager.prototypes) == 2
    assert decisions[3].is_new
    assert decisions[-1].context_id == 0


def test_agglomerative_centroid_predictor_is_reusable():
    predictor = AgglomerativeCentroidPredictor(
        estimator={"kind": "test_stub"},
        cluster_ids=np.asarray([4, 9]),
        centroids=np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32),
    )
    features = np.asarray([[0.2, -0.1], [9.5, 10.2]], dtype=np.float32)
    assert predictor.predict(features).tolist() == [4, 9]


def test_stream_manager_requires_a_consistent_novel_candidate():
    manager = StreamingContextManager(
        threshold=0.5,
        create_persistence=2,
        switch_persistence=1,
        candidate_threshold=0.4,
        merge_threshold=0.2,
    )
    manager.observe(np.asarray([0.0, 0.0], dtype=np.float32))
    first_far = manager.observe(np.asarray([5.0, 5.0], dtype=np.float32))
    inconsistent_far = manager.observe(np.asarray([-5.0, -5.0], dtype=np.float32))
    assert first_far.candidate_size == 1
    assert inconsistent_far.candidate_size == 1
    assert len(manager.prototypes) == 1


def test_stream_manager_merges_nearby_contexts_and_canonicalizes_ids():
    manager = StreamingContextManager(
        threshold=1.0,
        create_persistence=2,
        switch_persistence=1,
        prototype_alpha=0.01,
        merge_threshold=0.3,
    )
    first = manager.initialize_context(np.asarray([0.0, 0.0], dtype=np.float32))
    second = manager.initialize_context(np.asarray([0.2, 0.0], dtype=np.float32))
    manager.observe(np.asarray([0.2, 0.0], dtype=np.float32))
    assert len(manager.prototypes) == 1
    assert manager.canonical_id(first) == manager.canonical_id(second)
    assert manager.canonicalize([first, second]).tolist() == [first, first]


def test_stream_manager_state_contains_adaptive_statistics():
    manager = StreamingContextManager(threshold=1.0, radius_warmup=2)
    context = manager.initialize_context(
        np.asarray([0.0, 0.0], dtype=np.float32),
        count=3,
        distances=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )
    manager.observe(np.asarray([0.2, 0.0], dtype=np.float32))
    state = manager.state_dict()
    assert state["distance_counts"][str(context)] == 4
    assert str(context) in state["distance_variances"]
    assert 0.75 <= state["radii"][str(context)] <= 1.25
