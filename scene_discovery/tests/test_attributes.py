import numpy as np
import pandas as pd

from scene_discovery.attributes import (
    ATTRIBUTES,
    AttributeHead,
    FactorizedAttributeRecognizer,
    attribute_frame,
    chronological_fit_validation_masks,
    expected_calibration_error,
    normalize_attribute,
    semantic_key,
    sequence_attribute_table,
    select_confidence_threshold,
)


class ProbabilityStub:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict_proba(self, features):
        return np.repeat(self.probabilities[None, :], len(features), axis=0)


class ProjectorStub:
    def transform_modalities(self, arrays):
        return arrays


def sample_index():
    return pd.DataFrame({
        "sequence": [1, 1, 5, 5],
        "climate": ["normal\n", "normal", "Heavy Snow", "heavysnow"],
        "road_type": ["parkinglots", "Parking Lot", "urban", "urban"],
        "capture_time": ["night", "nighttime", "day", "day"],
    })


def test_attribute_taxonomy_normalizes_metadata_aliases():
    labels = attribute_frame(sample_index())
    assert labels.weather.tolist() == ["normal", "normal", "heavy_snow", "heavy_snow"]
    assert labels.road.tolist() == ["parking_lot", "parking_lot", "urban", "urban"]
    assert labels.lighting.tolist() == ["night", "night", "day", "day"]
    assert semantic_key("Heavy Snow", "Urban", "Day") == "heavy_snow|urban|day"
    assert normalize_attribute("road", "mountain-road") == "mountain"


def test_sequence_attribute_table_rejects_changing_attributes():
    values = sample_index()
    sequence_attribute_table(values)
    values.loc[1, "capture_time"] = "day"
    try:
        sequence_attribute_table(values)
    except ValueError as exc:
        assert "Seq1:lighting" in str(exc)
    else:
        raise AssertionError("Expected inconsistent sequence attributes to fail")


def test_chronological_fit_validation_masks_keep_every_sequence():
    index = pd.DataFrame({"sequence": [1] * 5 + [5] * 5})
    fit, validation = chronological_fit_validation_masks(index, 0.2)
    assert fit.sum() == 8
    assert validation.sum() == 2
    assert np.all(np.flatnonzero(fit) == np.asarray([0, 1, 2, 3, 5, 6, 7, 8]))


def test_factorized_recognizer_builds_semantic_key_and_joint_confidence():
    projected = {
        "camera": np.zeros((2, 3), dtype=np.float32),
        "radar": np.zeros((2, 2), dtype=np.float32),
    }
    heads = {
        "weather": AttributeHead(
            "weather", ("camera",), "stub",
            ProbabilityStub(["normal", "heavy_snow"], [0.1, 0.9]),
        ),
        "road": AttributeHead(
            "road", ("camera", "radar"), "stub",
            ProbabilityStub(["urban", "highway"], [0.8, 0.2]),
        ),
        "lighting": AttributeHead(
            "lighting", ("camera",), "stub",
            ProbabilityStub(["day", "night"], [0.25, 0.75]),
        ),
    }
    recognizer = FactorizedAttributeRecognizer(
        projector=ProjectorStub(), heads=heads, window=20, stride=5, taxonomy={}
    )
    output = recognizer.predict_projected(projected)
    assert output.predicted_semantic_key.tolist() == [
        "heavy_snow|urban|night",
        "heavy_snow|urban|night",
    ]
    assert np.allclose(output.joint_confidence_min, 0.75)
    assert set(heads) == set(ATTRIBUTES)


def test_expected_calibration_error_is_zero_for_matching_bin_accuracy():
    truth = ["yes", "no", "yes", "no"]
    prediction = ["yes", "yes", "yes", "yes"]
    confidence = [0.5, 0.5, 0.5, 0.5]
    assert expected_calibration_error(truth, prediction, confidence, bins=2) == 0.0


def make_recognizer(window=3, stride=2):
    heads = {
        "weather": AttributeHead(
            "weather", ("camera",), "stub",
            ProbabilityStub(["normal", "heavy_snow"], [0.9, 0.1]),
        ),
        "road": AttributeHead(
            "road", ("camera",), "stub",
            ProbabilityStub(["urban", "highway"], [0.8, 0.2]),
        ),
        "lighting": AttributeHead(
            "lighting", ("camera",), "stub",
            ProbabilityStub(["day", "night"], [0.7, 0.3]),
        ),
    }
    return FactorizedAttributeRecognizer(
        ProjectorStub(), heads, window=window, stride=stride, taxonomy={}
    )


def test_predict_arrays_requires_index_for_temporal_recognizer():
    recognizer = make_recognizer(window=3)
    try:
        recognizer.predict_arrays({"camera": np.zeros((6, 2), dtype=np.float32)})
    except ValueError as exc:
        assert "index is required" in str(exc)
    else:
        raise AssertionError("Expected temporal inference without index to fail")


def test_predict_arrays_applies_windows_without_crossing_sequences():
    recognizer = make_recognizer(window=3, stride=2)
    index = pd.DataFrame({
        "sequence": [1, 1, 1, 5, 5, 5],
        "row_index": list(range(6)),
    })
    output = recognizer.predict_arrays(
        {"camera": np.arange(12, dtype=np.float32).reshape(6, 2)}, index
    )
    assert len(output) == 2
    assert output["sequence"].astype(str).tolist() == ["1", "5"]
    assert output["predicted_semantic_key"].tolist() == [
        "normal|urban|day", "normal|urban|day"
    ]


def test_confidence_threshold_is_selected_from_operating_points():
    threshold, rows = select_confidence_threshold(
        ["a", "a", "b", "b"],
        ["a", "a", "a", "b"],
        [0.95, 0.8, 0.6, 0.9],
        [0.0, 0.7, 0.9],
        target_accuracy=1.0,
    )
    assert threshold == 0.7
    assert sum(row["selected"] for row in rows) == 1
