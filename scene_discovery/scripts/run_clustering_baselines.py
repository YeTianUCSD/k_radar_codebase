#!/usr/bin/env python3
"""Fit scene clusters on train descriptors and evaluate unchanged on test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.clustering import AgglomerativeCentroidPredictor  # noqa: E402
from scene_discovery.common import atomic_write_json, make_run_dir, seed_everything, utc_now  # noqa: E402
from scene_discovery.evaluation import (  # noqa: E402
    apply_mapping, clustering_metrics, hungarian_mapping, per_scene_metrics,
)
from scene_discovery.feature_bank import MODALITIES, DescriptorBank  # noqa: E402
from scene_discovery.preprocessing import PerModalityProjector, ProjectionSettings, modality_combinations  # noqa: E402
from scene_discovery.temporal import aggregate_windows  # noqa: E402
from scene_discovery.visualization import save_confusion_plot  # noqa: E402


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
    parser.add_argument("--algorithms", nargs="+", choices=("kmeans", "gmm", "agglomerative"),
                        default=["kmeans", "gmm", "agglomerative"])
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 10, 20, 30])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def fit_clusters(name: str, train: np.ndarray, test: np.ndarray, clusters: int, seed: int):
    if name == "kmeans":
        model = KMeans(n_clusters=clusters, n_init=20, random_state=seed)
        train_cluster = model.fit_predict(train)
        test_cluster = model.predict(test)
    elif name == "gmm":
        model = GaussianMixture(n_components=clusters, covariance_type="diag", reg_covar=1e-5,
                                n_init=3, max_iter=300, random_state=seed)
        train_cluster = model.fit_predict(train)
        test_cluster = model.predict(test)
    elif name == "agglomerative":
        estimator = AgglomerativeClustering(n_clusters=clusters, linkage="ward")
        train_cluster = estimator.fit_predict(train)
        ids = np.unique(train_cluster)
        centroids = np.stack([train[train_cluster == cluster].mean(axis=0) for cluster in ids])
        model = AgglomerativeCentroidPredictor(
            estimator=estimator,
            cluster_ids=ids,
            centroids=centroids,
        )
        test_cluster = model.predict(test)
    else:
        raise ValueError(name)
    return model, np.asarray(train_cluster), np.asarray(test_cluster)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(args.output_root.expanduser().resolve(), "clustering")
    train_index = bank.index("train")
    test_index = bank.index("test")
    train_arrays = {m: bank.array("train", m, args.descriptor) for m in args.modalities}
    test_arrays = {m: bank.array("test", m, args.descriptor) for m in args.modalities}
    projector = PerModalityProjector(
        args.modalities, ProjectionSettings(pca_components=args.pca_components, seed=args.seed)
    ).fit(train_arrays)
    train_parts = projector.transform_modalities(train_arrays)
    test_parts = projector.transform_modalities(test_arrays)
    joblib.dump(projector, run_dir / "projector.joblib")

    metric_rows = []
    scene_rows = []
    prediction_frames = []
    mappings: Dict[str, Dict[int, str]] = {}
    for combo in modality_combinations(args.modalities):
        combo_name = "+".join(combo)
        train_frame = np.concatenate([train_parts[m] for m in combo], axis=1)
        test_frame = np.concatenate([test_parts[m] for m in combo], axis=1)
        for window in args.windows:
            stride = 1 if window == 1 else args.stride
            x_train, idx_train = aggregate_windows(train_frame, train_index, window, stride)
            x_test, idx_test = aggregate_windows(test_frame, test_index, window, stride)
            y_train = idx_train["sequence"].astype(str).to_numpy()
            y_test = idx_test["sequence"].astype(str).to_numpy()
            for algorithm in args.algorithms:
                model, train_cluster, test_cluster = fit_clusters(
                    algorithm, x_train, x_test, args.clusters, args.seed
                )
                key = f"{combo_name}/w{window}/{algorithm}"
                mapping = hungarian_mapping(y_train, train_cluster)
                mappings[key] = mapping
                joblib.dump(model, run_dir / f"model_{combo_name.replace('+', '-')}_w{window}_{algorithm}.joblib")
                for split, index, labels, clusters in (
                    ("train", idx_train, y_train, train_cluster),
                    ("test", idx_test, y_test, test_cluster),
                ):
                    mapped = apply_mapping(clusters, mapping)
                    metrics = clustering_metrics(labels, clusters, mapped)
                    metric_rows.append({
                        "descriptor": args.descriptor, "modalities": combo_name,
                        "window": window, "stride": stride, "algorithm": algorithm,
                        "split": split, "samples": len(labels), **metrics,
                    })
                    for row in per_scene_metrics(labels, mapped):
                        scene_rows.append({
                            "descriptor": args.descriptor, "modalities": combo_name,
                            "window": window, "algorithm": algorithm, "split": split, **row,
                        })
                    predictions = index.copy()
                    predictions["true_sequence"] = labels
                    predictions["cluster_id"] = clusters
                    predictions["mapped_sequence"] = mapped
                    predictions["modalities"] = combo_name
                    predictions["window"] = window
                    predictions["algorithm"] = algorithm
                    predictions["evaluation_split"] = split
                    prediction_frames.append(predictions)
                if combo == tuple(args.modalities) and window == (20 if 20 in args.windows else args.windows[0]):
                    mapped_test = apply_mapping(test_cluster, mapping)
                    save_confusion_plot(
                        y_test, mapped_test, run_dir / f"confusion_{algorithm}.png",
                        f"{algorithm}: {combo_name}, window={window}",
                    )

    metrics_frame = pd.DataFrame(metric_rows).sort_values(["split", "algorithm", "window", "modalities"])
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(run_dir / "per_scene_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(run_dir / "predictions.csv", index=False)
    best = metrics_frame[metrics_frame.split == "test"].sort_values(["accuracy", "ari"], ascending=False).head(20)
    best.to_csv(run_dir / "best_test_results.csv", index=False)
    atomic_write_json(run_dir / "cluster_mappings.json", mappings)
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-scene-clustering/v1", "status": "complete",
        "created_utc": utc_now(), "descriptor_bank": str(bank.root),
        "arguments": vars(args), "projector": projector.metadata(),
        "fit_policy": "preprocessing, clustering, and Hungarian mapping fit on train only",
        "run_dir": str(run_dir),
    })
    print(best.to_string(index=False))
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()

