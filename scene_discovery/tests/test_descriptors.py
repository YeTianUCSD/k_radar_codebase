import numpy as np

from scene_discovery.descriptors import compute_descriptors, descriptor_dimension


def test_descriptor_shapes_and_values():
    values = np.arange(2 * 3 * 4 * 6, dtype=np.float16).reshape(2, 3, 4, 6)
    result = compute_descriptors(values, ("mean", "mean_std", "spatial"), (2, 2))
    assert result["mean"].shape == (2, 3)
    assert result["mean_std"].shape == (2, 6)
    assert result["spatial"].shape == (2, 18)
    expected = values.astype(np.float32).mean(axis=(2, 3))
    np.testing.assert_allclose(result["mean"], expected)
    assert result["mean"].dtype == np.float32
    assert descriptor_dimension(3, "spatial", (2, 2)) == 18


def test_invalid_feature_rank_is_rejected():
    values = np.zeros((2, 3, 4), dtype=np.float16)
    try:
        compute_descriptors(values, ("mean",))
    except ValueError as error:
        assert "NCHW" in str(error)
    else:
        raise AssertionError("Expected ValueError")
