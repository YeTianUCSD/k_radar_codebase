import numpy as np
import pandas as pd

from scene_discovery.attribute_protocol import (
    align_attribute_predictions,
    build_attribute_support_matrix,
    eligible_sequences,
    sequence_balanced_metrics,
)


def scene_table():
    return pd.DataFrame({
        "sequence": [1, 5, 46, 58],
        "weather": ["normal", "normal", "heavy_snow", "heavy_snow"],
        "road": ["urban", "urban", "highway", "urban"],
        "lighting": ["night", "day", "night", "day"],
    })


def test_support_is_computed_independently_for_each_attribute():
    support = build_attribute_support_matrix(scene_table())
    seq46 = support[support.held_out_sequence.eq("46")].iloc[0]
    assert seq46.weather_supported
    assert not seq46.road_supported
    assert seq46.lighting_supported
    assert not seq46.joint_supported
    assert set(eligible_sequences(support, "weather")) == {"1", "5", "46", "58"}


def test_sequence_balanced_accuracy_does_not_favor_long_sequence():
    metrics = sequence_balanced_metrics(
        ["a"] * 10 + ["b"],
        ["a"] * 10 + ["a"],
        ["1"] * 10 + ["2"],
    )
    assert np.isclose(metrics["frame_accuracy"], 10 / 11)
    assert np.isclose(metrics["sequence_accuracy"], 0.5)


def prediction_frame(attribute, anchors, truth, prediction):
    return pd.DataFrame({
        "held_out_sequence": ["46"] * len(anchors),
        "sequence": ["46"] * len(anchors),
        "window_end_row": anchors,
        f"true_{attribute}": [truth] * len(anchors),
        f"predicted_{attribute}": prediction,
        f"{attribute}_confidence": [0.9] * len(anchors),
    })


def test_independent_windows_are_joined_on_common_causal_anchors():
    predictions = {
        "weather": prediction_frame(
            "weather", [2, 3, 4], "snow", ["snow"] * 3
        ),
        "road": prediction_frame(
            "road", [3, 4], "highway", ["urban", "highway"]
        ),
        "lighting": prediction_frame(
            "lighting", [1, 2, 3, 4], "night", ["night"] * 4
        ),
    }
    result = align_attribute_predictions(predictions)
    assert result.window_end_row.tolist() == [3, 4]
    assert result.exact_match.tolist() == [False, True]
    assert result.error_type.tolist() == ["road_only", "exact_match"]


def test_alignment_preserves_probability_and_acceptance_columns():
    predictions = {
        attribute: prediction_frame(
            attribute, [4], {"weather": "snow", "road": "highway", "lighting": "night"}[attribute],
            [{"weather": "snow", "road": "highway", "lighting": "night"}[attribute]],
        ).assign(**{
            f"{attribute}_accepted": [True],
            f"{attribute}_prob_example": [0.75],
        })
        for attribute in ("weather", "road", "lighting")
    }
    result = align_attribute_predictions(predictions)
    for attribute in predictions:
        assert result.loc[0, f"{attribute}_accepted"]
        assert np.isclose(result.loc[0, f"{attribute}_prob_example"], 0.75)
