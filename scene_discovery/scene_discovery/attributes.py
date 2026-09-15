from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from .temporal import aggregate_windows


ATTRIBUTES = ("weather", "road", "lighting")
ATTRIBUTE_SOURCE_COLUMNS = {
    "weather": "climate",
    "road": "road_type",
    "lighting": "capture_time",
}
ATTRIBUTE_CLASSES = {
    "weather": ("normal", "overcast", "rain", "sleet", "fog", "heavy_snow"),
    "road": ("urban", "highway", "alleyway", "countryside", "parking_lot", "mountain"),
    "lighting": ("day", "night"),
}
ALIASES = {
    "weather": {
        "normal": "normal",
        "overcast": "overcast",
        "rain": "rain",
        "rainy": "rain",
        "sleet": "sleet",
        "fog": "fog",
        "foggy": "fog",
        "heavysnow": "heavy_snow",
    },
    "road": {
        "urban": "urban",
        "highway": "highway",
        "alleyway": "alleyway",
        "countryside": "countryside",
        "parkinglot": "parking_lot",
        "parkinglots": "parking_lot",
        "mountain": "mountain",
        "mountainroad": "mountain",
    },
    "lighting": {
        "day": "day",
        "daytime": "day",
        "night": "night",
        "nighttime": "night",
    },
}


def compact_token(value: object) -> str:
    token = str(value).strip().lower()
    return "".join(character for character in token if character.isalnum())


def normalize_attribute(attribute: str, value: object) -> str:
    if attribute not in ATTRIBUTES:
        raise ValueError(f"Unknown attribute {attribute!r}")
    token = compact_token(value)
    try:
        return ALIASES[attribute][token]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported {attribute} label {value!r} (normalized token {token!r})"
        ) from exc


def semantic_key(weather: object, road: object, lighting: object) -> str:
    return "|".join((
        normalize_attribute("weather", weather),
        normalize_attribute("road", road),
        normalize_attribute("lighting", lighting),
    ))


def attribute_frame(index: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ATTRIBUTE_SOURCE_COLUMNS.values() if column not in index]
    if missing:
        raise ValueError(f"Descriptor index is missing attribute columns: {missing}")
    result = pd.DataFrame(index=index.index)
    for attribute, source in ATTRIBUTE_SOURCE_COLUMNS.items():
        result[attribute] = [
            normalize_attribute(attribute, value) for value in index[source].to_numpy()
        ]
    result["semantic_key"] = (
        result["weather"] + "|" + result["road"] + "|" + result["lighting"]
    )
    return result


def sequence_attribute_table(index: pd.DataFrame) -> pd.DataFrame:
    if "sequence" not in index:
        raise ValueError("Descriptor index is missing sequence")
    values = pd.concat(
        [index[["sequence"]].reset_index(drop=True), attribute_frame(index).reset_index(drop=True)],
        axis=1,
    )
    inconsistent = []
    for sequence, group in values.groupby("sequence", sort=False):
        for attribute in ATTRIBUTES:
            if group[attribute].nunique() != 1:
                inconsistent.append(f"Seq{sequence}:{attribute}")
    if inconsistent:
        raise ValueError("Attributes change within sequences: " + ", ".join(inconsistent))
    return values.groupby("sequence", as_index=False, sort=False)[list(ATTRIBUTES) + ["semantic_key"]].first()


def chronological_fit_validation_masks(
    index: pd.DataFrame,
    validation_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    fit = np.zeros(len(index), dtype=bool)
    validation = np.zeros(len(index), dtype=bool)
    sequences = index["sequence"].astype(str).to_numpy()
    for sequence in index["sequence"].astype(str).drop_duplicates():
        positions = np.flatnonzero(sequences == sequence)
        if len(positions) < 2:
            raise ValueError(f"Seq{sequence} needs at least two samples")
        cut = max(1, min(len(positions) - 1, int(round(len(positions) * (1.0 - validation_fraction)))))
        fit[positions[:cut]] = True
        validation[positions[cut:]] = True
    return fit, validation


def make_classifier(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            multi_class="auto",
            random_state=seed,
        )
    if name == "linear_svm":
        return LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean")
    raise ValueError(f"Unknown classifier {name!r}")


def fit_probabilistic_classifier(
    name: str,
    seed: int,
    features: np.ndarray,
    labels: Sequence[object],
    index: Optional[pd.DataFrame] = None,
    calibration_fraction: float = 0.2,
):
    """Fit a classifier exposing predict_proba without frame-level CV leakage."""
    labels = np.asarray(labels)
    if name != "linear_svm":
        estimator = make_classifier(name, seed)
        estimator.fit(features, labels)
        return estimator
    if index is None:
        raise ValueError("linear_svm requires an index for chronological calibration")
    fit_mask, calibration_mask = chronological_fit_validation_masks(
        index.reset_index(drop=True), calibration_fraction
    )
    fit_classes = set(labels[fit_mask].astype(str))
    calibration_classes = set(labels[calibration_mask].astype(str))
    missing = calibration_classes - fit_classes
    if missing:
        raise ValueError(f"SVM calibration contains classes absent from fit split: {sorted(missing)}")
    base = make_classifier(name, seed)
    base.fit(features[fit_mask], labels[fit_mask])
    estimator = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    estimator.fit(features[calibration_mask], labels[calibration_mask])
    return estimator


@dataclass
class AttributeHead:
    attribute: str
    modalities: Tuple[str, ...]
    model_name: str
    estimator: object

    def feature_matrix(self, projected: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([projected[modality] for modality in self.modalities], axis=1)

    def predict(self, projected: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = self.feature_matrix(projected)
        probabilities = np.asarray(self.estimator.predict_proba(features), dtype=np.float32)
        positions = probabilities.argmax(axis=1)
        classes = np.asarray(self.estimator.classes_).astype(str)
        return classes[positions], probabilities.max(axis=1), probabilities


@dataclass
class FactorizedAttributeRecognizer:
    projector: object
    heads: Dict[str, AttributeHead]
    window: int
    stride: int
    taxonomy: Dict[str, Sequence[str]]
    confidence_thresholds: Dict[str, float] = field(default_factory=dict)
    training_sequences: Sequence[str] = field(default_factory=list)

    def predict_projected(self, projected: Mapping[str, np.ndarray]) -> pd.DataFrame:
        output = pd.DataFrame()
        confidences = []
        for attribute in ATTRIBUTES:
            labels, confidence, _ = self.heads[attribute].predict(projected)
            output[f"predicted_{attribute}"] = labels
            output[f"{attribute}_confidence"] = confidence
            confidences.append(confidence)
        output["predicted_semantic_key"] = (
            output["predicted_weather"]
            + "|"
            + output["predicted_road"]
            + "|"
            + output["predicted_lighting"]
        )
        output["joint_confidence_min"] = np.stack(confidences).min(axis=0)
        output["joint_confidence_product"] = np.stack(confidences).prod(axis=0)
        return output

    def predict_arrays(
        self,
        arrays: Mapping[str, np.ndarray],
        index: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Project raw descriptors and apply the recognizer training window."""
        projected = self.projector.transform_modalities(arrays)
        if index is None:
            if self.window != 1:
                raise ValueError("index is required when recognizer.window > 1")
            return self.predict_projected(projected)
        aggregated = {}
        window_index = None
        for modality, values in projected.items():
            current, current_index = aggregate_windows(
                values, index, self.window, self.stride
            )
            aggregated[modality] = current
            if window_index is None:
                window_index = current_index.reset_index(drop=True)
            elif not np.array_equal(
                window_index["window_end_row"].to_numpy(),
                current_index["window_end_row"].to_numpy(),
            ):
                raise RuntimeError("Modality windows are misaligned")
        predictions = self.predict_projected(aggregated)
        return pd.concat([window_index, predictions], axis=1)

    def metadata(self) -> Dict[str, object]:
        return {
            "window": self.window,
            "stride": self.stride,
            "taxonomy": self.taxonomy,
            "confidence_thresholds": getattr(self, "confidence_thresholds", {}),
            "training_sequences": list(getattr(self, "training_sequences", [])),
            "heads": {
                attribute: {
                    "modalities": list(head.modalities),
                    "model": head.model_name,
                    "classes": [str(value) for value in head.estimator.classes_],
                }
                for attribute, head in self.heads.items()
            },
        }


def expected_calibration_error(
    truth: Sequence[object],
    prediction: Sequence[object],
    confidence: Sequence[float],
    bins: int = 10,
) -> float:
    truth_values = np.asarray(truth).astype(str)
    predicted_values = np.asarray(prediction).astype(str)
    confidence_values = np.asarray(confidence, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence_values >= start) & (
            confidence_values <= stop if stop == 1.0 else confidence_values < stop
        )
        if not mask.any():
            continue
        accuracy = np.mean(truth_values[mask] == predicted_values[mask])
        error += mask.mean() * abs(accuracy - confidence_values[mask].mean())
    return float(error)


def confidence_operating_points(
    truth: Sequence[object],
    prediction: Sequence[object],
    confidence: Sequence[float],
    thresholds: Sequence[float],
):
    truth_values = np.asarray(truth).astype(str)
    predicted_values = np.asarray(prediction).astype(str)
    confidence_values = np.asarray(confidence, dtype=np.float64)
    rows = []
    for threshold in thresholds:
        accepted = confidence_values >= float(threshold)
        rows.append({
            "confidence_threshold": float(threshold),
            "coverage": float(accepted.mean()),
            "accepted_accuracy": (
                float(np.mean(truth_values[accepted] == predicted_values[accepted]))
                if accepted.any() else 0.0
            ),
            "accepted_samples": int(accepted.sum()),
        })
    return rows


def select_confidence_threshold(
    truth: Sequence[object],
    prediction: Sequence[object],
    confidence: Sequence[float],
    thresholds: Sequence[float],
    target_accuracy: float = 0.9,
) -> Tuple[float, list]:
    """Select a deployment threshold using calibration data only."""
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in [0, 1]")
    rows = confidence_operating_points(truth, prediction, confidence, thresholds)
    nonempty = [row for row in rows if row["accepted_samples"] > 0]
    if not nonempty:
        raise ValueError("No confidence threshold accepts any calibration sample")
    eligible = [row for row in nonempty if row["accepted_accuracy"] >= target_accuracy]
    if eligible:
        winner = max(
            eligible,
            key=lambda row: (row["coverage"], row["accepted_accuracy"], row["confidence_threshold"]),
        )
    else:
        winner = max(
            nonempty,
            key=lambda row: (row["accepted_accuracy"], row["coverage"], row["confidence_threshold"]),
        )
    selected = float(winner["confidence_threshold"])
    for row in rows:
        row["selected"] = bool(row["confidence_threshold"] == selected)
        row["target_accuracy"] = float(target_accuracy)
        row["target_met"] = bool(
            row["accepted_samples"] > 0
            and row["accepted_accuracy"] >= target_accuracy
        )
    return selected, rows
