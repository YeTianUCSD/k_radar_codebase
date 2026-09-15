"""Causal smoothing and offline classification metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)


ATTRIBUTES = ("weather", "road", "lighting")


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def causal_smooth(probabilities: np.ndarray, sequences, window: int):
    if window < 1:
        raise ValueError("window must be positive")
    result = np.empty_like(probabilities)
    sequences = np.asarray(sequences).astype(str)
    for sequence in pd.unique(sequences):
        positions = np.flatnonzero(sequences == sequence)
        cumulative = np.vstack([
            np.zeros((1, probabilities.shape[1]), dtype=np.float64),
            np.cumsum(probabilities[positions], axis=0, dtype=np.float64),
        ])
        for local, position in enumerate(positions):
            start = max(0, local + 1 - window)
            result[position] = (cumulative[local + 1] - cumulative[start]) / (local + 1 - start)
    return result


def evaluate_predictions(index, labels, probabilities, label_names, scope, window):
    metrics, per_class, per_sequence, predictions = [], [], [], index.copy()
    joint = np.ones(len(index), dtype=bool)
    for attribute in ATTRIBUTES:
        probability = causal_smooth(probabilities[attribute], index["sequence"], window)
        predicted = probability.argmax(axis=1)
        truth = labels[attribute]
        joint &= predicted == truth
        predictions[f"true_{attribute}"] = [label_names[attribute][value] for value in truth]
        predictions[f"predicted_{attribute}"] = [label_names[attribute][value] for value in predicted]
        predictions[f"{attribute}_confidence"] = probability.max(axis=1)
        metrics.append({
            "scope": scope, "attribute": attribute, "window": window,
            "samples": len(truth), "accuracy": accuracy_score(truth, predicted),
            "balanced_accuracy": balanced_accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
            "mean_confidence": float(probability.max(axis=1).mean()),
        })
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=np.arange(len(label_names[attribute])),
            zero_division=0,
        )
        for class_id, class_name in enumerate(label_names[attribute]):
            per_class.append({
                "scope": scope, "attribute": attribute, "window": window,
                "class": class_name, "precision": precision[class_id],
                "recall": recall[class_id], "f1": f1[class_id],
                "support": int(support[class_id]),
            })
        for sequence in pd.unique(index["sequence"].astype(str)):
            mask = index["sequence"].astype(str).eq(sequence).to_numpy()
            per_sequence.append({
                "scope": scope, "attribute": attribute, "window": window,
                "sequence": sequence, "samples": int(mask.sum()),
                "accuracy": accuracy_score(truth[mask], predicted[mask]),
            })
    predictions["joint_correct"] = joint
    metrics.append({
        "scope": scope, "attribute": "joint", "window": window,
        "samples": len(index), "accuracy": float(joint.mean()),
        "balanced_accuracy": float("nan"), "macro_f1": float("nan"),
        "mean_confidence": float("nan"),
    })
    return (
        pd.DataFrame(metrics), pd.DataFrame(per_class),
        pd.DataFrame(per_sequence), predictions,
    )


def confusion_tables(index, labels, probabilities, label_names, window):
    tables = {}
    for attribute in ATTRIBUTES:
        smoothed = causal_smooth(probabilities[attribute], index["sequence"], window)
        prediction = smoothed.argmax(axis=1)
        matrix = confusion_matrix(
            labels[attribute], prediction, labels=np.arange(len(label_names[attribute])),
            normalize="true",
        )
        tables[attribute] = pd.DataFrame(
            matrix, index=label_names[attribute], columns=label_names[attribute]
        )
    return tables
