import numpy as np
import pandas as pd

from scene_discovery.attributes import ATTRIBUTE_CLASSES
from scene_discovery.semantic_decoder import (
    candidate_score_frame,
    decode_semantic_contexts,
    prepare_semantic_registry,
    score_known_combinations,
    validate_probability_columns,
)


def registry_source():
    return pd.DataFrame({
        "sequence": [38, 46, 58],
        "weather": ["fog", "heavy_snow", "heavy_snow"],
        "road": ["mountain", "highway", "urban"],
        "lighting": ["day", "night", "day"],
    })


def predictions():
    frame = pd.DataFrame({
        "held_out_sequence": [46, 58],
        "sequence": [46, 58],
        "window_end_row": [100, 200],
        "weather_accepted": [True, False],
        "road_accepted": [True, True],
        "lighting_accepted": [True, True],
    })
    values = {
        "weather": [
            {"fog": 0.50, "heavy_snow": 0.45, "normal": 0.05},
            {"heavy_snow": 0.80, "normal": 0.20},
        ],
        "road": [
            {"highway": 0.90, "mountain": 0.05, "urban": 0.05},
            {"urban": 0.90, "highway": 0.05, "mountain": 0.05},
        ],
        "lighting": [
            {"night": 0.90, "day": 0.10},
            {"day": 0.90, "night": 0.10},
        ],
    }
    for attribute, rows in values.items():
        for label in ATTRIBUTE_CLASSES[attribute]:
            frame[f"{attribute}_prob_{label}"] = [
                row.get(label, 0.0) for row in rows
            ]
    return frame


def test_registry_is_keyed_by_attributes_not_sequence_id():
    source = registry_source()
    duplicate = source.iloc[[2]].copy()
    duplicate["sequence"] = 99
    registry = prepare_semantic_registry(pd.concat([source, duplicate]))
    seq58 = registry[registry.semantic_key.eq("heavy_snow|urban|day")].iloc[0]
    assert seq58.sequence_count == 2
    assert seq58.sequences == "58,99"


def test_constrained_decoder_uses_other_attributes_to_rescue_weather_error():
    frame = predictions().iloc[[0]].reset_index(drop=True)
    registry = prepare_semantic_registry(registry_source())
    decoded, scores = score_known_combinations(
        frame, registry, {"weather": 1, "road": 1, "lighting": 1}
    )
    assert decoded.loc[0, "constrained_semantic_key"] == "heavy_snow|highway|night"
    assert decoded.loc[0, "known_score"] > decoded.loc[0, "second_known_score"]
    assert scores.shape == (1, 3)


def test_attribute_acceptance_can_reject_as_unknown():
    frame = predictions()
    registry = prepare_semantic_registry(registry_source())
    decoded, _ = decode_semantic_contexts(
        frame,
        registry,
        {"weather": 1, "road": 1, "lighting": 1},
        require_attribute_acceptance=True,
    )
    assert decoded.known_accepted.tolist() == [True, False]
    assert decoded.open_set_semantic_key.tolist() == [
        "heavy_snow|highway|night",
        "unknown",
    ]


def test_probability_schema_error_is_actionable():
    try:
        validate_probability_columns(pd.DataFrame({"weather_prob_normal": [1.0]}))
    except ValueError as error:
        assert "Rerun evaluate_independent_attributes" in str(error)
    else:
        raise AssertionError("Incomplete probabilities should fail")


def test_candidate_scores_keep_causal_anchor():
    frame = predictions().iloc[[0]].reset_index(drop=True)
    registry = prepare_semantic_registry(registry_source())
    _, scores = score_known_combinations(
        frame, registry, {"weather": 1, "road": 1, "lighting": 1}
    )
    long = candidate_score_frame(frame, registry, scores)
    assert len(long) == 3
    assert long.window_end_row.unique().tolist() == [100]
