#!/usr/bin/env python3
"""Evaluate label-free online context creation from a Seq1-only initialization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

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
from scene_discovery.common import (  # noqa: E402
    atomic_write_json, make_run_dir, seed_everything, utc_now,
)
from scene_discovery.evaluation import (  # noqa: E402
    apply_mapping, clustering_metrics, hungarian_mapping,
)
from scene_discovery.feature_bank import (  # noqa: E402
    DEFAULT_SEQUENCES, MODALITIES, DescriptorBank,
)
from scene_discovery.memory_context import MemoryBankContextManager  # noqa: E402
from scene_discovery.online_protocol import (  # noqa: E402
    build_disjoint_orders, concatenate_partition_stream,
    normalize_sequence, summarize_online_blocks,
)
from scene_discovery.preprocessing import (  # noqa: E402
    PerModalityProjector, ProjectionSettings,
)
from scene_discovery.temporal import (  # noqa: E402
    aggregate_stream_windows, aggregate_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor-bank", type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/descriptors/v1",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/OnlineContextDiscovery",
    )
    parser.add_argument(
        "--descriptor", choices=("mean", "mean_std", "spatial"),
        default="mean_std",
    )
    parser.add_argument(
        "--modalities", nargs="+", choices=MODALITIES,
        default=list(MODALITIES),
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("early_concat", "seq1_calibrated_late_distance"),
        default="seq1_calibrated_late_distance",
    )
    parser.add_argument(
        "--modality-weights", nargs="+", type=float, default=None,
        help="Weights in --modalities order; defaults to equal weights.",
    )
    parser.add_argument("--initial-scenes", nargs="+", default=["1"])
    parser.add_argument(
        "--scene-order", nargs="+", default=list(DEFAULT_SEQUENCES),
        help="Initial scenes are excluded; remaining train partitions are first visits.",
    )
    parser.add_argument(
        "--revisit-order", nargs="+", default=None,
        help="Test partitions used for disjoint revisits; defaults to --scene-order.",
    )
    parser.add_argument("--num-random-orders", type=int, default=4)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument(
        "--threshold-multipliers", nargs="+", type=float,
        default=[0.8, 1.0, 1.2],
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        help="Optional fixed thresholds; never selected using stream labels.",
    )
    parser.add_argument(
        "--manager", choices=("centroid", "memory_bank"), default="memory_bank"
    )
    parser.add_argument("--memory-size", type=int, default=64)
    parser.add_argument("--memory-neighbors", type=int, default=7)
    parser.add_argument("--memory-update-threshold-ratio", type=float, default=0.8)
    parser.add_argument("--memory-min-separation-ratio", type=float, default=0.1)
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


def chronological_masks(
    index: pd.DataFrame,
    scenes: Sequence[str],
    calibration_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = index["sequence"].astype(str).to_numpy()
    fit = np.zeros(len(index), dtype=bool)
    calibration = np.zeros(len(index), dtype=bool)
    for scene in scenes:
        positions = np.flatnonzero(sequences == scene)
        if len(positions) < 2:
            raise ValueError(f"Seq{scene} needs at least two initialization rows")
        cut = max(1, min(
            len(positions) - 1,
            int(round(len(positions) * (1.0 - calibration_fraction))),
        ))
        fit[positions[:cut]] = True
        calibration[positions[cut:]] = True
    return fit, calibration


def subset_windows(features, index, mask, window, stride):
    return aggregate_windows(
        np.asarray(features[mask], dtype=np.float32),
        index.loc[mask].reset_index(drop=True),
        window,
        1 if window == 1 else stride,
    )


def prepare_initial_contexts(
    features: np.ndarray,
    index: pd.DataFrame,
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
    scenes: Sequence[str],
    window: int,
    stride: int,
):
    sequences = index["sequence"].astype(str).to_numpy()
    contexts = []
    calibration_distances = []
    for scene in scenes:
        fit_x, _ = subset_windows(
            features, index, fit_mask & (sequences == scene), window, stride
        )
        calibration_x, _ = subset_windows(
            features, index, calibration_mask & (sequences == scene), window, 1
        )
        if not len(fit_x) or not len(calibration_x):
            raise ValueError(
                f"Seq{scene} has too few initialization rows for window={window}"
            )
        prototype = fit_x.mean(axis=0).astype(np.float32)
        fit_distances = np.linalg.norm(fit_x - prototype[None, :], axis=1)
        calibration_values = np.linalg.norm(
            calibration_x - prototype[None, :], axis=1
        )
        contexts.append({
            "scene": scene,
            "prototype": prototype,
            "fit_count": len(fit_x),
            "fit_distances": fit_distances,
            "fit_features": fit_x,
            "calibration_features": calibration_x,
        })
        calibration_distances.append(calibration_values)
    return contexts, np.concatenate(calibration_distances)


def calibrate_modality_scales(
    projected_train, index, fit_mask, calibration_mask, scenes,
    modalities, window, stride, quantile, fusion_mode,
):
    sequence_values = index["sequence"].astype(str).to_numpy()
    scales = {}
    rows = []
    for modality in modalities:
        calibration_values = []
        fit_values = []
        for scene in scenes:
            scene_fit = fit_mask & (sequence_values == scene)
            scene_calibration = calibration_mask & (sequence_values == scene)
            fit_x, _ = subset_windows(
                projected_train[modality], index, scene_fit, window, stride
            )
            calibration_x, _ = subset_windows(
                projected_train[modality], index, scene_calibration, window, 1
            )
            prototype = fit_x.mean(axis=0)
            fit_values.append(np.linalg.norm(fit_x - prototype[None, :], axis=1))
            calibration_values.append(
                np.linalg.norm(calibration_x - prototype[None, :], axis=1)
            )
        fit_distances = np.concatenate(fit_values)
        calibration_distances = np.concatenate(calibration_values)
        calibrated_radius = float(np.quantile(calibration_distances, quantile))
        scale = (
            max(calibrated_radius, 1e-6)
            if fusion_mode == "seq1_calibrated_late_distance" else 1.0
        )
        scales[modality] = scale
        rows.append({
            "modality": modality,
            "fit_distance_mean": float(fit_distances.mean()),
            "calibration_distance_mean": float(calibration_distances.mean()),
            "calibration_distance_std": float(calibration_distances.std()),
            "calibrated_radius": calibrated_radius,
            "applied_scale": scale,
        })
    return scales, pd.DataFrame(rows)


def memory_calibration_distances(contexts, memory_size, neighbors):
    distances = []
    for context in contexts:
        fit_x = context["fit_features"]
        if len(fit_x) > memory_size:
            positions = np.linspace(0, len(fit_x) - 1, memory_size).round().astype(int)
            memory = fit_x[positions]
        else:
            memory = fit_x
        calibration_x = context["calibration_features"]
        pairwise = np.linalg.norm(
            calibration_x[:, None, :] - memory[None, :, :], axis=2
        )
        count = min(neighbors, memory.shape[0])
        distances.append(np.sort(pairwise, axis=1)[:, :count].mean(axis=1))
    return np.concatenate(distances)


def purity(truth: np.ndarray, contexts: np.ndarray) -> float:
    correct = 0
    for context in np.unique(contexts):
        _, counts = np.unique(truth[contexts == context], return_counts=True)
        correct += int(counts.max())
    return float(correct / len(truth))


def summarize_run(
    frame: pd.DataFrame,
    contexts: np.ndarray,
    mapped: np.ndarray,
    decisions,
    block_metrics: pd.DataFrame,
    manager,
    initialized_contexts: int,
) -> Dict[str, float]:
    truth = frame["stream_true_sequence"].astype(str).to_numpy()
    fragments = [
        len(np.unique(contexts[truth == scene])) for scene in np.unique(truth)
    ]
    first = block_metrics[block_metrics["visit"].eq("first")]
    revisits = block_metrics[block_metrics["visit"].eq("revisit")]
    reusable = revisits[revisits["reuse_correct"].notna()]
    true_contexts = len(np.unique(truth))
    creation_events = int(sum(decision.is_new for decision in decisions))
    base = clustering_metrics(truth, contexts, mapped)
    base.update({
        "purity": purity(truth, contexts),
        "initialized_contexts": initialized_contexts,
        "active_contexts": len(manager.prototypes),
        "true_contexts": true_contexts,
        "context_count_error": len(manager.prototypes) - true_contexts,
        "online_creation_events": creation_events,
        "expected_creation_events": len(first),
        "excess_creation_events": max(0, creation_events - len(first)),
        "first_visit_creation_rate": (
            float(first["created_in_block"].mean()) if len(first) else 0.0
        ),
        "missed_new_scene_blocks": (
            int((~first["created_in_block"]).sum()) if len(first) else 0
        ),
        "first_visit_registry_collisions": (
            int(first["registry_collision"].sum()) if len(first) else 0
        ),
        "duplicate_first_visit_creation_events": (
            int(first["duplicate_creation_events"].sum()) if len(first) else 0
        ),
        "mean_creation_latency_windows": (
            float(first["creation_latency_windows"].mean()) if len(first) else 0.0
        ),
        "revisit_context_reuse_accuracy": (
            float(reusable["reuse_correct"].astype(bool).mean())
            if len(reusable) else float("nan")
        ),
        "revisit_false_creation_blocks": (
            int(revisits["created_in_block"].sum()) if len(revisits) else 0
        ),
        "revisit_false_creation_events": (
            int(revisits["creation_events"].sum()) if len(revisits) else 0
        ),
        "mean_contexts_per_true_scene": float(np.mean(fragments)),
        "max_contexts_per_true_scene": int(np.max(fragments)),
        "merge_count": len(manager.aliases),
        "mean_block_mapped_accuracy": float(block_metrics["mapped_accuracy"].mean()),
        "mean_block_switch_latency_windows": float(
            block_metrics["first_correct_latency_windows"].mean()
        ),
    })
    return base


def save_timeline(frame: pd.DataFrame, path: Path, title: str) -> None:
    labels = sorted(
        frame["stream_true_sequence"].astype(str).unique(), key=lambda value: int(value)
    )
    encoding = {label: position for position, label in enumerate(labels)}
    truth = frame["stream_true_sequence"].astype(str).map(encoding).to_numpy()
    mapped = frame["mapped_sequence"].astype(str).map(encoding).fillna(-1).to_numpy()
    x = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(18, 5), constrained_layout=True)
    axis.step(x, truth, where="post", label="True scene", linewidth=2)
    axis.step(x, mapped + 0.08, where="post", label="Discovered context", linewidth=1.2)
    created = frame["is_new_context"].to_numpy(dtype=bool)
    axis.scatter(x[created], mapped[created] + 0.08, marker="x", color="red", label="Created")
    for boundary in frame.groupby("stream_block", sort=True).head(1).index:
        axis.axvline(boundary, color="black", alpha=0.08, linewidth=0.7)
    axis.set_yticks(range(len(labels)), [f"Seq{label}" for label in labels])
    axis.set_xlabel("Causal stream window")
    axis.set_ylabel("Scene")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 0.5:
        raise ValueError("--calibration-fraction must be in (0, 0.5)")
    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be in (0, 1)")
    if args.window < 1 or args.stride < 1:
        raise ValueError("--window and --stride must be positive")
    seed_everything(args.seed)

    initial_scenes = [normalize_sequence(value) for value in args.initial_scenes]
    scene_order = [normalize_sequence(value) for value in args.scene_order]
    revisit_order = (
        scene_order if args.revisit_order is None
        else [normalize_sequence(value) for value in args.revisit_order]
    )
    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(
        args.output_root.expanduser().resolve(), "online_context_v3"
    )
    indices = {split: bank.index(split) for split in ("train", "test")}
    fit_mask, calibration_mask = chronological_masks(
        indices["train"], initial_scenes, args.calibration_fraction
    )
    raw_arrays = {
        split: {
            modality: np.asarray(bank.array(split, modality, args.descriptor))
            for modality in args.modalities
        }
        for split in ("train", "test")
    }
    projector = PerModalityProjector(
        args.modalities,
        ProjectionSettings(pca_components=args.pca_components, seed=args.seed),
    ).fit({
        modality: values[fit_mask]
        for modality, values in raw_arrays["train"].items()
    })
    joblib.dump(projector, run_dir / "seq1_only_projector.joblib")
    projected = {
        split: projector.transform_modalities(raw_arrays[split])
        for split in ("train", "test")
    }
    raw_weights = (
        np.ones(len(args.modalities), dtype=np.float64)
        if args.modality_weights is None
        else np.asarray(args.modality_weights, dtype=np.float64)
    )
    if len(raw_weights) != len(args.modalities):
        raise ValueError("--modality-weights must match --modalities length")
    if np.any(raw_weights < 0.0) or raw_weights.sum() <= 0.0:
        raise ValueError("modality weights must be non-negative with positive sum")
    normalized_weights = raw_weights / raw_weights.sum()
    modality_scales, modality_calibration = calibrate_modality_scales(
        projected["train"], indices["train"], fit_mask, calibration_mask,
        initial_scenes, args.modalities, args.window, args.stride,
        args.threshold_quantile, args.fusion_mode,
    )
    modality_calibration["weight"] = normalized_weights
    modality_calibration.to_csv(run_dir / "seq1_modality_calibration.csv", index=False)
    fused = {}
    for split in ("train", "test"):
        parts = []
        for modality, weight in zip(args.modalities, normalized_weights):
            factor = np.sqrt(weight) / modality_scales[modality]
            parts.append(projected[split][modality] * factor)
        fused[split] = np.concatenate(parts, axis=1).astype(np.float32)
    initial_contexts, calibration_distances = prepare_initial_contexts(
        fused["train"], indices["train"], fit_mask, calibration_mask,
        initial_scenes, args.window, args.stride,
    )
    if args.manager == "memory_bank":
        calibration_distances = memory_calibration_distances(
            initial_contexts, args.memory_size, args.memory_neighbors
        )
    calibrated_threshold = float(np.quantile(
        calibration_distances, args.threshold_quantile
    ))
    if args.thresholds:
        thresholds = sorted(set(float(value) for value in args.thresholds))
        threshold_source = "explicit; not selected from stream labels"
        deployment_threshold = thresholds[0]
    else:
        thresholds = sorted(set(
            calibrated_threshold * float(value)
            for value in args.threshold_multipliers
        ))
        threshold_source = "Seq1-only calibration quantile times multiplier"
        deployment_threshold = min(
            thresholds, key=lambda value: abs(value - calibrated_threshold)
        )
    if not thresholds or min(thresholds) <= 0.0:
        raise ValueError("all evaluated thresholds must be positive")

    orders = build_disjoint_orders(
        scene_order, initial_scenes, revisit_order,
        args.num_random_orders, args.seed,
    )
    calibration_payload = {
        "initial_scenes": initial_scenes,
        "manager": args.manager,
        "fusion_mode": args.fusion_mode,
        "modality_weights": {
            modality: float(weight)
            for modality, weight in zip(args.modalities, normalized_weights)
        },
        "modality_scales": {
            modality: float(scale) for modality, scale in modality_scales.items()
        },
        "fit_rows": int(fit_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "calibration_window_stride": 1,
        "calibration_distance_count": len(calibration_distances),
        "calibration_distance_mean": float(calibration_distances.mean()),
        "calibration_distance_std": float(calibration_distances.std()),
        "threshold_quantile": args.threshold_quantile,
        "calibrated_threshold": calibrated_threshold,
        "evaluated_thresholds": thresholds,
        "deployment_threshold": deployment_threshold,
        "threshold_source": threshold_source,
    }
    atomic_write_json(run_dir / "seq1_calibration.json", calibration_payload)

    metric_rows = []
    prediction_frames = []
    block_frames = []
    order_payload = []
    for order_id, blocks in orders:
        order_payload.append({"order_id": order_id, "blocks": blocks})
        stream_arrays, stream_index = concatenate_partition_stream(
            {
                "train": {"fused": fused["train"]},
                "test": {"fused": fused["test"]},
            },
            indices,
            blocks,
        )
        stream_features, window_index = aggregate_stream_windows(
            stream_arrays["fused"], stream_index, args.window, args.stride
        )
        truth = window_index["stream_true_sequence"].astype(str).to_numpy()
        for threshold in thresholds:
            if args.manager == "memory_bank":
                manager = MemoryBankContextManager(
                    threshold=threshold,
                    create_persistence=args.create_persistence,
                    switch_persistence=args.switch_persistence,
                    candidate_threshold=threshold * args.candidate_threshold_ratio,
                    memory_size=args.memory_size,
                    neighbors=args.memory_neighbors,
                    update_threshold_ratio=args.memory_update_threshold_ratio,
                    memory_min_separation_ratio=args.memory_min_separation_ratio,
                )
            else:
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
            initial_ids = {}
            for context in initial_contexts:
                if args.manager == "memory_bank":
                    context_id = manager.initialize_context(
                        context["fit_features"], count=context["fit_count"]
                    )
                else:
                    context_id = manager.initialize_context(
                        context["prototype"],
                        count=context["fit_count"],
                        distances=context["fit_distances"],
                    )
                initial_ids[context["scene"]] = context_id
            decisions = [manager.observe(value) for value in stream_features]
            raw_ids = np.asarray([decision.context_id for decision in decisions])
            context_ids = manager.canonicalize(raw_ids)
            initial_ids = {
                scene: manager.canonical_id(context)
                for scene, context in initial_ids.items()
            }
            mapping = hungarian_mapping(truth, context_ids)
            mapped = apply_mapping(context_ids, mapping)
            blocks_frame, _ = summarize_online_blocks(
                window_index,
                context_ids,
                [decision.is_new for decision in decisions],
                mapped,
                initial_ids,
            )
            blocks_frame.insert(0, "order_id", order_id)
            blocks_frame["threshold"] = threshold
            blocks_frame["is_deployment_threshold"] = bool(
                np.isclose(threshold, deployment_threshold)
            )
            block_frames.append(blocks_frame)
            metrics = summarize_run(
                window_index, context_ids, mapped, decisions, blocks_frame,
                manager, len(initial_contexts),
            )
            metric_rows.append({
                "order_id": order_id,
                "manager": args.manager,
                "threshold": threshold,
                "is_deployment_threshold": bool(
                    np.isclose(threshold, deployment_threshold)
                ),
                "windows": len(window_index),
                "scene_blocks": len(blocks),
                **metrics,
            })
            output = window_index.copy()
            output.insert(0, "order_id", order_id)
            output["raw_context_id"] = raw_ids
            output["context_id"] = context_ids
            output["mapped_sequence"] = mapped
            output["is_new_context"] = [d.is_new for d in decisions]
            output["nearest_context"] = manager.canonicalize(
                [d.nearest_context for d in decisions]
            )
            output["nearest_distance"] = [d.nearest_distance for d in decisions]
            output["acceptance_radius"] = [d.acceptance_radius for d in decisions]
            output["candidate_size"] = [d.candidate_size for d in decisions]
            output["threshold"] = threshold
            output["is_deployment_threshold"] = bool(
                np.isclose(threshold, deployment_threshold)
            )
            prediction_frames.append(output)
            threshold_name = f"{threshold:.6f}".replace(".", "p")
            joblib.dump(
                manager, run_dir / f"manager_{order_id}_t{threshold_name}.joblib"
            )
            atomic_write_json(
                run_dir / f"manager_{order_id}_t{threshold_name}.json",
                manager.state_dict(),
            )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    block_metrics = pd.concat(block_frames, ignore_index=True)
    summary = metrics_frame.groupby(
        ["manager", "threshold", "is_deployment_threshold"], as_index=False
    ).mean(numeric_only=True)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    block_metrics.to_csv(run_dir / "block_metrics.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    selected = predictions[
        predictions["order_id"].eq("canonical")
        & predictions["is_deployment_threshold"]
    ].reset_index(drop=True)
    save_timeline(
        selected,
        run_dir / "canonical_deployment_timeline.png",
        f"Seq1-only disjoint online replay, threshold={deployment_threshold:.4f}",
    )
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-online-context-discovery/v3",
        "status": "complete",
        "created_utc": utc_now(),
        "descriptor_bank": str(bank.root),
        "arguments": vars(args),
        "algorithm_inputs": (
            "projected encoder descriptors only; sequence/split/visit fields are "
            "withheld from StreamingContextManager and used only for replay/scoring"
        ),
        "manager": args.manager,
        "initialization_policy": "projector, prototype, and threshold use Seq1 train only",
        "stream_policy": (
            "non-initial train partitions are first visits; disjoint test partitions "
            "are revisits; no descriptor sample is replayed twice in one order"
        ),
        "threshold_policy": threshold_source,
        "selection_warning": (
            "stream labels may compare research thresholds but never select the "
            "deployment threshold"
        ),
        "window_policy": "causal windows may cross block boundaries",
        "projector": projector.metadata(),
        "orders": order_payload,
        "calibration": calibration_payload,
        "run_dir": str(run_dir),
    })
    print("Seq1-only calibration:")
    print(pd.DataFrame([calibration_payload]).to_string(index=False))
    print("\nThreshold summary across orders:")
    print(summary.sort_values("threshold").to_string(index=False))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
