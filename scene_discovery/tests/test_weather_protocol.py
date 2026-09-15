import pandas as pd

from scene_discovery.causal_evaluation import true_transition_events
from scene_discovery.weather_protocol import (
    annotate_weather_stream,
    build_weather_orders,
    sequence_weather_mapping,
)


def _attribute_index():
    return pd.DataFrame({
        "sequence": [1, 5, 46, 58, 35],
        "climate": ["Normal", "Normal", "Heavy Snow", "Heavy Snow", "Sleet"],
        "road_type": ["Urban", "Urban", "Highway", "Urban", "Parking Lot"],
        "capture_time": ["Night", "Day", "Night", "Day", "Night"],
    })


def test_weather_mapping_normalizes_descriptor_metadata():
    mapping = sequence_weather_mapping(_attribute_index())
    assert mapping == {
        "1": "normal", "5": "normal", "46": "heavy_snow",
        "58": "heavy_snow", "35": "sleet",
    }


def test_weather_order_marks_novelty_by_weather_not_sequence():
    mapping = sequence_weather_mapping(_attribute_index())
    _, blocks = build_weather_orders(
        [1, 5, 46, 58, 35], 1, mapping, random_orders=0, seed=7
    )[0]
    first_visits = blocks[:4]
    assert [block["sequence"] for block in first_visits] == ["5", "46", "58", "35"]
    assert [block["expected_novel"] for block in first_visits] == [False, True, False, True]
    assert all(not block["expected_novel"] for block in blocks[4:])


def test_annotation_preserves_scene_but_scores_weather():
    mapping = sequence_weather_mapping(_attribute_index())
    frame = pd.DataFrame({"stream_true_sequence": ["46", "58", "5"]})
    result = annotate_weather_stream(frame, mapping)
    assert result["stream_true_scene"].tolist() == ["46", "58", "5"]
    assert result["stream_true_sequence"].tolist() == [
        "heavy_snow", "heavy_snow", "normal",
    ]


def test_same_weather_scene_boundary_is_not_a_true_weather_transition():
    mapping = sequence_weather_mapping(_attribute_index())
    frame = pd.DataFrame({
        "stream_block": [0, 0, 1, 1, 2, 2, 3, 3],
        "stream_true_sequence": ["5", "5", "46", "46", "58", "58", "35", "35"],
        "stream_visit": ["known_weather"] * 2 + ["novel_weather"] * 2
                        + ["known_weather"] * 2 + ["novel_weather"] * 2,
        "stream_expected_novel": [False] * 2 + [True] * 2
                                 + [False] * 2 + [True] * 2,
    })
    weather_frame = annotate_weather_stream(frame, mapping)
    events = true_transition_events(weather_frame, initial_scene="normal")
    assert events["sequence"].tolist() == ["heavy_snow", "sleet"]
    assert events["boundary_index"].tolist() == [2, 6]
    assert events["expected_novel"].tolist() == [True, True]
