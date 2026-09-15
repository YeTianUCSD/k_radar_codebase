from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def _label_order(labels: Sequence[object]):
    values = np.unique(np.asarray(labels).astype(str))
    return sorted(values, key=lambda value: int(value) if value.isdigit() else value)


def save_embedding_plot(
    train_features: np.ndarray,
    train_labels: Sequence[object],
    test_features: np.ndarray,
    test_labels: Sequence[object],
    path: Path,
    title: str,
    seed: int = 20260812,
) -> None:
    projector = PCA(n_components=2, random_state=seed)
    train_xy = projector.fit_transform(train_features)
    test_xy = projector.transform(test_features)
    labels = _label_order(np.concatenate((np.asarray(train_labels), np.asarray(test_labels))))
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, xy, y, split in (
        (axes[0], train_xy, np.asarray(train_labels).astype(str), "train"),
        (axes[1], test_xy, np.asarray(test_labels).astype(str), "test"),
    ):
        for index, label in enumerate(labels):
            mask = y == label
            axis.scatter(xy[mask, 0], xy[mask, 1], s=10, alpha=0.65, color=colors(index), label=f"Seq{label}")
        axis.set_title(split)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.2)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_confusion_plot(
    true_labels: Sequence[object], predicted_labels: Sequence[object], path: Path, title: str
) -> None:
    labels = _label_order(true_labels)
    matrix = confusion_matrix(np.asarray(true_labels).astype(str), np.asarray(predicted_labels).astype(str), labels=labels)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), [f"Seq{x}" for x in labels], rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), [f"Seq{x}" for x in labels])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_centroid_distance_plot(features: np.ndarray, labels: Sequence[object], path: Path, title: str) -> None:
    order = _label_order(labels)
    labels_array = np.asarray(labels).astype(str)
    centroids = np.stack([features[labels_array == label].mean(axis=0) for label in order])
    distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(distances, cmap="viridis")
    axis.set_xticks(range(len(order)), [f"Seq{x}" for x in order], rotation=45, ha="right")
    axis.set_yticks(range(len(order)), [f"Seq{x}" for x in order])
    axis.set_title(title)
    for row in range(len(order)):
        for column in range(len(order)):
            axis.text(column, row, f"{distances[row, column]:.2f}", ha="center", va="center", fontsize=6,
                      color="white" if distances[row, column] > distances.max() * 0.55 else "black")
    figure.colorbar(image, ax=axis, fraction=0.046, label="Euclidean distance")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)

