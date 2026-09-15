import numpy as np
import pandas as pd

from scene_discovery.preprocessing import PerModalityProjector, ProjectionSettings
from scene_discovery.temporal import aggregate_stream_windows, aggregate_windows


def test_projector_is_fit_on_supplied_train_only():
    train = {"camera": np.asarray([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32)}
    test = {"camera": np.asarray([[100.0, 200.0]], dtype=np.float32)}
    projector = PerModalityProjector(
        ("camera",), ProjectionSettings(pca_components=0, l2_normalize=False)
    ).fit(train)
    np.testing.assert_allclose(projector.scalers["camera"].mean_, [1.0, 2.0])
    transformed = projector.transform(test)
    assert transformed[0, 0] > 10.0


def test_scene_windows_never_cross_sequence():
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    index = pd.DataFrame({
        "row_index": np.arange(6),
        "sequence": ["1", "1", "1", "2", "2", "2"],
    })
    output, rows = aggregate_windows(features, index, window_size=2, stride=1)
    assert len(output) == 4
    assert rows["sequence"].tolist() == ["1", "1", "2", "2"]


def test_window_anchor_uses_stable_source_row_for_all_window_sizes():
    features = np.arange(6, dtype=np.float32).reshape(3, 2)
    index = pd.DataFrame({
        "row_index": [40, 41, 42],
        "sequence": ["5", "5", "5"],
    })
    _, single = aggregate_windows(features, index, window_size=1, stride=1)
    _, paired = aggregate_windows(features, index, window_size=2, stride=1)
    assert single["window_end_row"].tolist() == [40, 41, 42]
    assert paired["window_end_row"].tolist() == [41, 42]


def test_stream_windows_are_causal_and_can_cross_boundary():
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    index = pd.DataFrame({
        "row_index": np.arange(4),
        "sequence": ["1", "1", "2", "2"],
        "stream_true_sequence": ["1", "1", "2", "2"],
    })
    output, rows = aggregate_stream_windows(features, index, window_size=3, stride=1)
    assert len(output) == 2
    np.testing.assert_allclose(output[0], features[:3].mean(axis=0))
    assert rows.iloc[0]["stream_true_sequence"] == "2"
