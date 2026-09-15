#!/usr/bin/env python3
"""Simulate label-free context creation on a continuous descriptor stream."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.clustering import StreamingContextManager  # noqa: E402
from scene_discovery.common import atomic_write_json, make_run_dir, seed_everything, utc_now  # noqa: E402
from scene_discovery.evaluation import apply_mapping, clustering_metrics, hungarian_mapping  # noqa: E402
from scene_discovery.feature_bank import DEFAULT_SEQUENCES, MODALITIES, DescriptorBank  # noqa: E402
from scene_discovery.preprocessing import PerModalityProjector, ProjectionSettings  # noqa: E402
from scene_discovery.temporal import aggregate_stream_windows, aggregate_windows, concatenate_scene_stream  # noqa: E402


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
    parser.add_argument(
        "--projector-fit-scenes", nargs="+", default=["1"],
        help="Only these initial known scenes may fit preprocessing and initial prototypes.",
    )
    parser.add_argument(
        "--scene-order", nargs="+", default=list(DEFAULT_SEQUENCES),
        help="Base scene blocks. IDs select the evaluation stream but are never passed to the manager.",
    )
    parser.add_argument(
        "--repeat-scenes", nargs="*", default=["1", "35"],
        help="Blocks appended to every order to measure context reuse.",
    )
    parser.add_argument(
        "--num-random-orders", type=int, default=4,
        help="Additional seeded permutations of the unique base scene order.",
    )
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument(
        "--threshold-multipliers", nargs="+", type=float, default=[0.8, 1.0, 1.2],
        help="Label-free multipliers applied to the train-calibrated novelty threshold.",
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        help="Optional fixed thresholds. If supplied, train calibration is still reported but bypassed.",
    )
    parser.add_argument("--create-persistence", type=int, default=5)
    parser.add_argument("--switch-persistence", type=int, default=3)
    parser.add_argument("--prototype-alpha", type=float, default=0.02)
    parser.add_argument("--candidate-threshold-ratio", type=float, default=0.75)
    parser.add_argument("--merge-threshold-ratio", type=float, default=0.65)
    parser.add_argument("--radius-alpha", type=float, default=0.05)
    parser.add_argument("--radius-std-scale", type=float, default=3.0)
    parser.add_argument("--radius-min-scale", type=float, default=0.75)
    parser.add_argument("--radius-max-scale", type=float, default=1.25)
    parser.add_argument("--radius-warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def normalize_scene(value: object) -> str:
    result = str(value).lower()
    if result.startswith("seq"):
        result = result[3:]
    return str(int(result))


def chronological_split_mask(
    index: pd.DataFrame,
    scenes: Sequence[str],
    calibration_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    scene_values = index["sequence"].astype(str).to_numpy()
    fit = np.zeros(len(index), dtype=bool)
    calibration = np.zeros(len(index), dtype=bool)
    for scene in scenes:
        positions = np.flatnonzero(scene_values == scene)
        if len(positions) < 2:
            raise ValueError(f"Scene {scene} needs at least two train samples")
        cut = max(1, min(len(positions) - 1, int(round(len(positions) * (1.0 - calibration_fraction)))))
        fit[positions[:cut]] = True
        calibration[positions[cut:]] = True
    return fit, calibration


def window_subset(
    features: np.ndarray,
    index: pd.DataFrame,
    mask: np.ndarray,
    window: int,
    stride: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    subset_index = index.loc[mask].reset_index(drop=True)
    subset_features = features[mask]
    return aggregate_windows(subset_features, subset_index, window, 1 if window == 1 else stride)


def prepare_initial_contexts(
    features: np.ndarray,
    index: pd.DataFrame,
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
    scenes: Sequence[str],
    window: int,
    stride: int,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    sequence_values = index["sequence"].astype(str).to_numpy()
    contexts: List[Dict[str, object]] = []
    calibration_distances = []
    for scene in scenes:
        scene_fit = fit_mask & (sequence_values == scene)
        scene_calibration = calibration_mask & (sequence_values == scene)
        fit_x, _ = window_subset(features, index, scene_fit, window, stride)
        calibration_x, _ = window_subset(features, index, scene_calibration, window, stride)
        if not len(fit_x) or not len(calibration_x):
            raise ValueError(
                f"Scene {scene} has too few samples for window={window}; "
                "reduce --window or --calibration-fraction"
            )
        prototype = fit_x.mean(axis=0).astype(np.float32)
        fit_distances = np.linalg.norm(fit_x - prototype[None, :], axis=1)
        calibration_values = np.linalg.norm(calibration_x - prototype[None, :], axis=1)
        contexts.append({
            "scene": scene,
            "prototype": prototype,
            "fit_count": len(fit_x),
            "fit_distances": fit_distances,
            "calibration_count": len(calibration_x),
        })
        calibration_distances.append(calibration_values)
    return contexts, np.concatenate(calibration_distances)


def build_orders(
    base_order: Sequence[str],
    repeat_scenes: Sequence[str],
    random_orders: int,
    seed: int,
) -> List[Tuple[str, List[str]]]:
    if random_orders < 0:
        raise ValueError("--num-random-orders must be non-negative")
    unique_order = list(dict.fromkeys(base_order))
    if not unique_order:
        raise ValueError("--scene-order cannot be empty")
    missing = [scene for scene in repeat_scenes if scene not in unique_order]
    if missing:
        raise ValueError(f"Repeated scenes are absent from the base order: {missing}")
    orders = [("canonical", list(base_order) + list(repeat_scenes))]
    rng = np.random.RandomState(seed)
    seen = {tuple(unique_order)}
    attempts = 0
    while len(orders) < random_orders + 1:
        attempts += 1
        candidate = list(rng.permutation(unique_order))
        key = tuple(candidate)
        if key in seen:
            if attempts > 1000:
                raise RuntimeError("Could not generate enough unique scene orders")
            continue
        seen.add(key)
        orders.append((f"random_{len(orders):02d}", candidate + list(repeat_scenes)))
    return orders


def cluster_purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    correct = 0
    for cluster in np.unique(clusters):
        _, counts = np.unique(labels[clusters == cluster], return_counts=True)
        correct += int(counts.max())
    return correct / len(labels)


def stream_diagnostics(
    frame: pd.DataFrame,
    true_labels: np.ndarray,
    clusters: np.ndarray,
    mapped: np.ndarray,
    initialized_contexts: int,
    active_contexts: int,
    online_creations: int,
    merge_count: int,
) -> Dict[str, float]:
    fragment_counts = [
        len(np.unique(clusters[true_labels == sequence])) for sequence in np.unique(true_labels)
    ]
    block_dominant: Dict[int, int] = {}
    first_context_for_scene: Dict[str, int] = {}
    reuse_matches = 0
    reuse_trials = 0
    switch_latencies = []
    blocks = frame["stream_block"].to_numpy(dtype=int)
    for block in frame["stream_block"].drop_duplicates():
        positions = np.flatnonzero(blocks == int(block))
        if not len(positions):
            continue
        block_clusters = clusters[positions]
        values, counts = np.unique(block_clusters, return_counts=True)
        dominant = int(values[counts.argmax()])
        block_dominant[int(block)] = dominant
        scene = str(true_labels[positions[-1]])
        if scene in first_context_for_scene:
            reuse_trials += 1
            reuse_matches += int(dominant == first_context_for_scene[scene])
        else:
            first_context_for_scene[scene] = dominant
        if int(block) > int(frame["stream_block"].min()):
            correct = np.flatnonzero(mapped[positions].astype(str) == scene)
            switch_latencies.append(int(correct[0]) if len(correct) else len(positions))

    true_contexts = len(np.unique(true_labels))
    expected_new = max(0, true_contexts - initialized_contexts)
    return {
        "purity": cluster_purity(true_labels, clusters),
        "mean_contexts_per_true_scene": float(np.mean(fragment_counts)),
        "max_contexts_per_true_scene": int(np.max(fragment_counts)),
        "initialized_contexts": int(initialized_contexts),
        "active_contexts": int(active_contexts),
        "context_count_error": int(active_contexts - true_contexts),
        "online_context_creations": int(online_creations),
        "expected_new_contexts": int(expected_new),
        "excess_creation_events": int(max(0, online_creations - expected_new)),
        "merge_count": int(merge_count),
        "reuse_trials": int(reuse_trials),
        "context_reuse_accuracy": float(reuse_matches / reuse_trials) if reuse_trials else float("nan"),
        "mean_switch_latency_windows": float(np.mean(switch_latencies)) if switch_latencies else 0.0,
        "max_switch_latency_windows": int(np.max(switch_latencies)) if switch_latencies else 0,
    }


def save_timeline(frame: pd.DataFrame, path: Path, title: str) -> None:
    labels = sorted(frame["true_sequence"].astype(str).unique(), key=lambda value: int(value))
    encoding = {label: position for position, label in enumerate(labels)}
    truth = frame["true_sequence"].astype(str).map(encoding).to_numpy()
    mapped = frame["mapped_sequence"].astype(str).map(encoding).fillna(-1).to_numpy()
    x = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(16, 5), constrained_layout=True)
    axis.step(x, truth, where="post", label="True scene", linewidth=2)
    axis.step(x, mapped + 0.08, where="post", label="Discovered context (mapped)", linewidth=1.2)
    created = frame["is_new_context"].to_numpy(dtype=bool)
    axis.scatter(x[created], mapped[created] + 0.08, marker="x", color="red", label="New context", zorder=5)
    axis.set_yticks(range(len(labels)), [f"Seq{label}" for label in labels])
    axis.set_xlabel("Stream window")
    axis.set_ylabel("Scene")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 0.5:
        raise ValueError("--calibration-fraction must be in (0, 0.5)")
    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be in (0, 1)")
    seed_everything(args.seed)
    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(args.output_root.expanduser().resolve(), "stream")

    fit_scenes = [normalize_scene(value) for value in args.projector_fit_scenes]
    base_order = [normalize_scene(value) for value in args.scene_order]
    repeat_scenes = [normalize_scene(value) for value in args.repeat_scenes]
    train_index = bank.index("train")
    test_index = bank.index("test")
    fit_mask, calibration_mask = chronological_split_mask(
        train_index, fit_scenes, args.calibration_fraction
    )
    train_arrays = {m: bank.array("train", m, args.descriptor) for m in args.modalities}
    test_arrays = {m: bank.array("test", m, args.descriptor) for m in args.modalities}
    projector = PerModalityProjector(
        args.modalities, ProjectionSettings(pca_components=args.pca_components, seed=args.seed)
    ).fit({m: np.asarray(values[fit_mask]) for m, values in train_arrays.items()})
    joblib.dump(projector, run_dir / "projector.joblib")

    train_parts = projector.transform_modalities(train_arrays)
    test_parts = projector.transform_modalities(test_arrays)
    train_features = np.concatenate([train_parts[m] for m in args.modalities], axis=1)
    test_features = np.concatenate([test_parts[m] for m in args.modalities], axis=1)
    initial_contexts, calibration_distances = prepare_initial_contexts(
        train_features,
        train_index,
        fit_mask,
        calibration_mask,
        fit_scenes,
        args.window,
        args.stride,
    )
    calibrated_threshold = float(np.quantile(calibration_distances, args.threshold_quantile))
    if args.thresholds:
        thresholds = sorted(set(float(value) for value in args.thresholds))
        threshold_source = "explicit command-line thresholds"
        deployment_threshold = thresholds[0]
    else:
        thresholds = sorted(set(calibrated_threshold * float(value) for value in args.threshold_multipliers))
        threshold_source = "train-only calibrated threshold multiplied by --threshold-multipliers"
        deployment_threshold = min(thresholds, key=lambda value: abs(value - calibrated_threshold))
    if not thresholds or min(thresholds) <= 0:
        raise ValueError("all thresholds must be positive")

    orders = build_orders(base_order, repeat_scenes, args.num_random_orders, args.seed)
    calibration_payload = {
        "fit_scenes": fit_scenes,
        "projector_fit_rows": int(fit_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "window": args.window,
        "stride": args.stride,
        "threshold_quantile": args.threshold_quantile,
        "calibration_distance_count": int(len(calibration_distances)),
        "calibration_distance_mean": float(calibration_distances.mean()),
        "calibration_distance_std": float(calibration_distances.std()),
        "calibrated_threshold": calibrated_threshold,
        "evaluated_thresholds": thresholds,
        "deployment_threshold": deployment_threshold,
        "threshold_source": threshold_source,
    }
    atomic_write_json(run_dir / "calibration.json", calibration_payload)

    metric_rows = []
    prediction_frames = []
    for order_id, scene_order in orders:
        stream_features, stream_index = concatenate_scene_stream(
            test_features, test_index, scene_order
        )
        window_features, window_index = aggregate_stream_windows(
            stream_features, stream_index, args.window, args.stride
        )
        true_labels = window_index["stream_true_sequence"].astype(str).to_numpy()
        for threshold in thresholds:
            manager = StreamingContextManager(
                threshold=threshold,
                create_persistence=args.create_persistence,
                switch_persistence=args.switch_persistence,
                prototype_alpha=args.prototype_alpha,
                candidate_threshold=threshold * args.candidate_threshold_ratio,
                merge_threshold=threshold * args.merge_threshold_ratio,
                radius_alpha=args.radius_alpha,
                radius_std_scale=args.radius_std_scale,
                radius_min_scale=args.radius_min_scale,
                radius_max_scale=args.radius_max_scale,
                radius_warmup=args.radius_warmup,
            )
            for context in initial_contexts:
                manager.initialize_context(
                    context["prototype"],
                    count=int(context["fit_count"]),
                    distances=context["fit_distances"],
                )
            decisions = [manager.observe(value) for value in window_features]
            raw_ids = np.asarray([decision.context_id for decision in decisions], dtype=int)
            cluster_ids = manager.canonicalize(raw_ids)
            mapping = hungarian_mapping(true_labels, cluster_ids)
            mapped = apply_mapping(cluster_ids, mapping)
            metrics = clustering_metrics(true_labels, cluster_ids, mapped)
            diagnostics = stream_diagnostics(
                window_index,
                true_labels,
                cluster_ids,
                mapped,
                initialized_contexts=len(initial_contexts),
                active_contexts=len(manager.prototypes),
                online_creations=int(sum(decision.is_new for decision in decisions)),
                merge_count=len(manager.aliases),
            )
            metric_rows.append({
                "order_id": order_id,
                "threshold": threshold,
                "is_deployment_threshold": bool(np.isclose(threshold, deployment_threshold)),
                "windows": len(true_labels),
                "scene_blocks": len(scene_order),
                **diagnostics,
                **metrics,
            })
            output = window_index.copy()
            output["order_id"] = order_id
            output["true_sequence"] = true_labels
            output["raw_context_id"] = raw_ids
            output["context_id"] = cluster_ids
            output["mapped_sequence"] = mapped
            output["is_new_context"] = [decision.is_new for decision in decisions]
            output["nearest_context"] = manager.canonicalize(
                [decision.nearest_context for decision in decisions]
            )
            output["nearest_distance"] = [decision.nearest_distance for decision in decisions]
            output["acceptance_radius"] = [decision.acceptance_radius for decision in decisions]
            output["candidate_size"] = [decision.candidate_size for decision in decisions]
            output["threshold"] = threshold
            output["is_deployment_threshold"] = bool(np.isclose(threshold, deployment_threshold))
            prediction_frames.append(output)

            threshold_name = f"{threshold:.6f}".replace(".", "p")
            joblib.dump(manager, run_dir / f"manager_{order_id}_t{threshold_name}.joblib")
            atomic_write_json(
                run_dir / f"manager_{order_id}_t{threshold_name}.json",
                manager.state_dict(),
            )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    summary = metrics_frame.groupby(
        ["threshold", "is_deployment_threshold"], as_index=False
    ).mean(numeric_only=True)
    summary.to_csv(run_dir / "summary.csv", index=False)

    selected = predictions[
        predictions["order_id"].eq("canonical")
        & predictions["is_deployment_threshold"]
    ].reset_index(drop=True)
    save_timeline(
        selected,
        run_dir / "deployment_timeline.png",
        f"Train-calibrated streaming discovery, threshold={deployment_threshold:.4f}",
    )
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-scene-stream-discovery/v2",
        "status": "complete",
        "created_utc": utc_now(),
        "descriptor_bank": str(bank.root),
        "arguments": vars(args),
        "algorithm_inputs": "descriptors only; sequence IDs are used only to construct and score the test stream",
        "preprocessing_policy": "projector and initial prototypes fit only on --projector-fit-scenes train data",
        "threshold_policy": threshold_source,
        "selection_policy": "deployment threshold selected from train calibration only; test labels never select it",
        "window_policy": "causal windows may cross true scene boundaries",
        "projector": projector.metadata(),
        "orders": [{"order_id": key, "scene_order": value} for key, value in orders],
        "calibration": calibration_payload,
        "run_dir": str(run_dir),
    })
    print(summary.sort_values("threshold").to_string(index=False))
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
