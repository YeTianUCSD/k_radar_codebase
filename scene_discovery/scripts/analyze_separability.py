#!/usr/bin/env python3
"""Measure how well encoder descriptors separate scenes on held-out frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.common import atomic_write_json, make_run_dir, seed_everything, utc_now  # noqa: E402
from scene_discovery.evaluation import classification_metrics, per_scene_metrics  # noqa: E402
from scene_discovery.feature_bank import MODALITIES, DescriptorBank  # noqa: E402
from scene_discovery.preprocessing import PerModalityProjector, ProjectionSettings, modality_combinations  # noqa: E402
from scene_discovery.temporal import aggregate_windows  # noqa: E402
from scene_discovery.visualization import (  # noqa: E402
    save_centroid_distance_plot,
    save_confusion_plot,
    save_embedding_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor-bank", type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/descriptors/v1",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/experiments",
    )
    parser.add_argument("--descriptor", choices=("mean", "mean_std", "spatial"), default="mean_std")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 10, 20, 30])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--silhouette-max-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(args.output_root.expanduser().resolve(), "separability")
    train_index = bank.index("train")
    test_index = bank.index("test")
    train_arrays = {m: bank.array("train", m, args.descriptor) for m in args.modalities}
    test_arrays = {m: bank.array("test", m, args.descriptor) for m in args.modalities}
    projector = PerModalityProjector(
        args.modalities,
        ProjectionSettings(pca_components=args.pca_components, seed=args.seed),
    ).fit(train_arrays)
    train_parts = projector.transform_modalities(train_arrays)
    test_parts = projector.transform_modalities(test_arrays)
    joblib.dump(projector, run_dir / "projector.joblib")

    metric_rows = []
    scene_rows = []
    prediction_frames = []
    rng = np.random.RandomState(args.seed)
    combos = modality_combinations(args.modalities)
    for combo in combos:
        combo_name = "+".join(combo)
        train_frame = np.concatenate([train_parts[m] for m in combo], axis=1)
        test_frame = np.concatenate([test_parts[m] for m in combo], axis=1)
        for window in args.windows:
            stride = 1 if window == 1 else args.stride
            x_train, idx_train = aggregate_windows(train_frame, train_index, window, stride)
            x_test, idx_test = aggregate_windows(test_frame, test_index, window, stride)
            y_train = idx_train["sequence"].astype(str).to_numpy()
            y_test = idx_test["sequence"].astype(str).to_numpy()
            if not len(x_train) or not len(x_test):
                continue

            sample = np.arange(len(x_train))
            if len(sample) > args.silhouette_max_samples:
                sample = rng.choice(sample, args.silhouette_max_samples, replace=False)
            silhouette = float(silhouette_score(x_train[sample], y_train[sample]))

            classifiers = {
                "nearest_centroid": NearestCentroid(),
                "knn_1": KNeighborsClassifier(n_neighbors=1),
                "knn_5": KNeighborsClassifier(n_neighbors=5),
            }
            for model_name, model in classifiers.items():
                model.fit(x_train, y_train)
                for split, features, index, labels in (
                    ("train", x_train, idx_train, y_train),
                    ("test", x_test, idx_test, y_test),
                ):
                    predicted = model.predict(features).astype(str)
                    metrics = classification_metrics(labels, predicted)
                    metric_rows.append({
                        "descriptor": args.descriptor, "modalities": combo_name,
                        "window": window, "stride": stride, "model": model_name,
                        "split": split, "samples": len(labels), "silhouette_train": silhouette,
                        **metrics,
                    })
                    for row in per_scene_metrics(labels, predicted):
                        scene_rows.append({
                            "descriptor": args.descriptor, "modalities": combo_name,
                            "window": window, "model": model_name, "split": split, **row,
                        })
                    predictions = index.copy()
                    predictions["true_sequence"] = labels
                    predictions["predicted_sequence"] = predicted
                    predictions["descriptor"] = args.descriptor
                    predictions["modalities"] = combo_name
                    predictions["window"] = window
                    predictions["model"] = model_name
                    predictions["evaluation_split"] = split
                    prediction_frames.append(predictions)

            if combo == tuple(args.modalities) and window == (20 if 20 in args.windows else args.windows[0]):
                title = f"{args.descriptor}, {combo_name}, window={window}"
                save_embedding_plot(x_train, y_train, x_test, y_test, run_dir / "pca_embedding.png", title, args.seed)
                save_centroid_distance_plot(x_train, y_train, run_dir / "centroid_distances.png", title)
                centroid = classifiers["nearest_centroid"]
                save_confusion_plot(y_test, centroid.predict(x_test), run_dir / "confusion_nearest_centroid.png", title)

    metrics_frame = pd.DataFrame(metric_rows).sort_values(["split", "model", "window", "modalities"])
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(run_dir / "per_scene_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(run_dir / "predictions.csv", index=False)
    best = metrics_frame[metrics_frame.split == "test"].sort_values("accuracy", ascending=False).head(20)
    best.to_csv(run_dir / "best_test_results.csv", index=False)
    metadata = {
        "schema_version": "kradar-scene-separability/v1", "status": "complete",
        "created_utc": utc_now(), "descriptor_bank": str(bank.root),
        "arguments": vars(args), "projector": projector.metadata(),
        "train_only_fit": True, "run_dir": str(run_dir),
    }
    atomic_write_json(run_dir / "run_meta.json", metadata)
    print(best.to_string(index=False))
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()

