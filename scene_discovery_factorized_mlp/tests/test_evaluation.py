import numpy as np

from factorized_mlp.evaluation import causal_smooth


def test_causal_smoothing_does_not_cross_sequence_boundaries():
    probabilities = np.asarray([
        [1.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0],
    ])
    result = causal_smooth(probabilities, ["1", "1", "5", "5"], window=10)
    assert np.allclose(result[0], [1.0, 0.0])
    assert np.allclose(result[1], [1.0, 0.0])
    assert np.allclose(result[2], [0.0, 1.0])
    assert np.allclose(result[3], [0.0, 1.0])
