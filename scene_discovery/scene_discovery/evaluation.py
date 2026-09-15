from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)


def hungarian_mapping(true_labels: Sequence[object], cluster_labels: Sequence[int]) -> Dict[int, str]:
    truth = np.asarray(true_labels).astype(str)
    clusters = np.asarray(cluster_labels)
    true_values = np.unique(truth)
    cluster_values = np.unique(clusters)
    counts = np.zeros((len(cluster_values), len(true_values)), dtype=np.int64)
    for row, cluster in enumerate(cluster_values):
        for column, label in enumerate(true_values):
            counts[row, column] = np.count_nonzero((clusters == cluster) & (truth == label))
    selected_clusters, selected_truth = linear_sum_assignment(-counts)
    return {int(cluster_values[row]): str(true_values[column]) for row, column in zip(selected_clusters, selected_truth)}


def apply_mapping(cluster_labels: Sequence[int], mapping: Mapping[int, str]) -> np.ndarray:
    return np.asarray([mapping.get(int(value), f"unmapped:{int(value)}") for value in cluster_labels])


def classification_metrics(true_labels: Sequence[object], predicted_labels: Sequence[object]) -> Dict[str, float]:
    truth = np.asarray(true_labels).astype(str)
    predicted = np.asarray(predicted_labels).astype(str)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }


def clustering_metrics(
    true_labels: Sequence[object], cluster_labels: Sequence[int], mapped_labels: Sequence[object]
) -> Dict[str, float]:
    truth = np.asarray(true_labels).astype(str)
    clusters = np.asarray(cluster_labels)
    result = classification_metrics(truth, mapped_labels)
    result.update({
        "ari": float(adjusted_rand_score(truth, clusters)),
        "nmi": float(normalized_mutual_info_score(truth, clusters)),
        "cluster_count": int(len(np.unique(clusters))),
    })
    return result


def per_scene_metrics(true_labels: Sequence[object], predicted_labels: Sequence[object]):
    truth = np.asarray(true_labels).astype(str)
    predicted = np.asarray(predicted_labels).astype(str)
    labels = sorted(np.unique(truth), key=lambda value: int(value) if value.isdigit() else value)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    return [
        {"sequence": label, "precision": float(p), "recall": float(r), "f1": float(score), "support": int(n)}
        for label, p, r, score, n in zip(labels, precision, recall, f1, support)
    ]

