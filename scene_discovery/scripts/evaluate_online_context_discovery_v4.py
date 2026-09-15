#!/usr/bin/env python3
"""Evaluate causal boundary detection plus strict context CREATE/REUSE decisions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path

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

from scene_discovery.causal_context import CausalContextManager  # noqa: E402
from scene_discovery.causal_evaluation import evaluate_causal_events  # noqa: E402
from scene_discovery.common import atomic_write_json, make_run_dir, seed_everything, utc_now  # noqa: E402
from scene_discovery.evaluation import apply_mapping, clustering_metrics, hungarian_mapping  # noqa: E402
from scene_discovery.feature_bank import DEFAULT_SEQUENCES, MODALITIES, DescriptorBank  # noqa: E402
from scene_discovery.online_features import (  # noqa: E402
    calibrate_modality_scales, chronological_masks,
    memory_calibration_distances, prepare_initial_contexts,
)
from scene_discovery.online_protocol import (  # noqa: E402
    build_disjoint_orders, concatenate_partition_stream, normalize_sequence,
)
from scene_discovery.preprocessing import PerModalityProjector, ProjectionSettings  # noqa: E402
from scene_discovery.temporal import aggregate_stream_windows  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor-bank", type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/descriptors/v1",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/OnlineContextDiscoveryV4",
    )
    parser.add_argument("--descriptor", choices=("mean", "mean_std", "spatial"), default="mean_std")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument(
        "--fusion-mode", choices=("early_concat", "seq1_calibrated_late_distance"),
        default="seq1_calibrated_late_distance",
    )
    parser.add_argument("--modality-weights", nargs="+", type=float, default=None)
    parser.add_argument("--initial-scene", default="1")
    parser.add_argument("--scene-order", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--revisit-order", nargs="+", default=None)
    parser.add_argument("--num-random-orders", type=int, default=4)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument(
        "--change-threshold-multipliers", nargs="+", type=float,
        default=[0.6, 0.7, 0.8, 1.0],
    )
    parser.add_argument(
        "--match-threshold-ratios", nargs="+", type=float,
        default=[1.0, 1.1, 1.2],
        help="Segment match radius divided by each change radius; must be >= 1.",
    )
    parser.add_argument("--change-persistence", type=int, default=5)
    parser.add_argument("--boundary-tolerance", type=int, default=10)
    parser.add_argument("--memory-size", type=int, default=64)
    parser.add_argument("--memory-neighbors", type=int, default=7)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument("--memory-update-threshold-ratio", type=float, default=0.8)
    parser.add_argument("--memory-min-separation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def purity(truth, contexts):
    correct = 0
    for context in np.unique(contexts):
        _, counts = np.unique(truth[contexts == context], return_counts=True)
        correct += int(counts.max())
    return float(correct / len(truth))


def save_timeline(frame, decisions, true_events, predicted_events, path, title):
    labels = sorted(frame["stream_true_sequence"].astype(str).unique(), key=int)
    encoding = {label: position for position, label in enumerate(labels)}
    truth = frame["stream_true_sequence"].astype(str).map(encoding).to_numpy()
    contexts = np.asarray([decision.context_id for decision in decisions], dtype=int)
    x = np.arange(len(frame))
    figure, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True, constrained_layout=True)
    axes[0].step(x, truth, where="post", linewidth=1.5)
    axes[0].set_yticks(range(len(labels)), [f"Seq{label}" for label in labels])
    axes[0].set_ylabel("True scene\n(scoring only)")
    axes[1].step(x, contexts, where="post", linewidth=1.2, color="tab:orange")
    axes[1].set_ylabel("Online context ID")
    axes[1].set_xlabel("Causal stream window")
    for axis in axes:
        for boundary in true_events["boundary_index"]:
            axis.axvline(boundary, color="black", linestyle="--", alpha=0.18)
        axis.grid(alpha=0.2)
    if len(predicted_events):
        colors = predicted_events["action"].map({"create": "red", "reuse": "green"}).fillna("gray")
        axes[1].scatter(
            predicted_events["detected_index"], predicted_events["context_id"],
            c=colors, marker="x", s=40, label="Predicted boundary",
        )
        axes[1].legend()
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 0.5:
        raise ValueError("--calibration-fraction must be in (0, 0.5)")
    if not 0.0 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be in (0, 1)")
    if args.window < 1 or args.stride < 1 or args.boundary_tolerance < 0:
        raise ValueError("window/stride must be positive and tolerance non-negative")
    seed_everything(args.seed)

    initial_scene = normalize_sequence(args.initial_scene)
    scene_order = [normalize_sequence(value) for value in args.scene_order]
    revisit_order = (
        scene_order if args.revisit_order is None
        else [normalize_sequence(value) for value in args.revisit_order]
    )
    if initial_scene not in scene_order:
        raise ValueError("--initial-scene must occur in --scene-order")
    multipliers = sorted(set(float(value) for value in args.change_threshold_multipliers))
    if not multipliers or min(multipliers) <= 0.0:
        raise ValueError("change-threshold multipliers must be positive")
    match_ratios = sorted(set(float(value) for value in args.match_threshold_ratios))
    if not match_ratios or min(match_ratios) < 1.0:
        raise ValueError("every --match-threshold-ratios value must be >= 1")

    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(args.output_root.expanduser().resolve(), "online_context_v4")
    indices = {split: bank.index(split) for split in ("train", "test")}
    fit_mask, calibration_mask = chronological_masks(
        indices["train"], [initial_scene], args.calibration_fraction
    )
    raw = {
        split: {
            modality: np.asarray(bank.array(split, modality, args.descriptor))
            for modality in args.modalities
        }
        for split in ("train", "test")
    }
    projector = PerModalityProjector(
        args.modalities,
        ProjectionSettings(pca_components=args.pca_components, seed=args.seed),
    ).fit({modality: values[fit_mask] for modality, values in raw["train"].items()})
    joblib.dump(projector, run_dir / "seq1_only_projector.joblib")
    projected = {
        split: projector.transform_modalities(raw[split])
        for split in ("train", "test")
    }
    weights = (
        np.ones(len(args.modalities), dtype=np.float64)
        if args.modality_weights is None
        else np.asarray(args.modality_weights, dtype=np.float64)
    )
    if len(weights) != len(args.modalities) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("modality weights must match modalities and have a positive sum")
    weights /= weights.sum()
    modality_scales, modality_calibration = calibrate_modality_scales(
        projected["train"], indices["train"], fit_mask, calibration_mask,
        [initial_scene], args.modalities, args.window, args.stride,
        args.threshold_quantile, args.fusion_mode,
    )
    modality_calibration["weight"] = weights
    modality_calibration.to_csv(run_dir / "seq1_modality_calibration.csv", index=False)
    fused = {}
    for split in ("train", "test"):
        parts = [
            projected[split][modality] * np.sqrt(weight) / modality_scales[modality]
            for modality, weight in zip(args.modalities, weights)
        ]
        fused[split] = np.concatenate(parts, axis=1).astype(np.float32)
    initial_contexts = prepare_initial_contexts(
        fused["train"], indices["train"], fit_mask, calibration_mask,
        [initial_scene], args.window, args.stride,
    )
    calibration_distances = memory_calibration_distances(
        initial_contexts, args.memory_size, args.memory_neighbors
    )
    calibrated_threshold = float(np.quantile(calibration_distances, args.threshold_quantile))
    change_thresholds = [calibrated_threshold * value for value in multipliers]
    threshold_settings = [
        (multiplier, threshold, ratio, threshold * ratio)
        for multiplier, threshold in zip(multipliers, change_thresholds)
        for ratio in match_ratios
    ]
    deployment_threshold = min(
        change_thresholds, key=lambda value: abs(value - calibrated_threshold)
    )
    deployment_match_ratio = min(match_ratios, key=lambda value: abs(value - 1.0))
    calibration_payload = {
        "initial_scene": initial_scene,
        "fit_rows": int(fit_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "calibration_distance_count": len(calibration_distances),
        "threshold_quantile": args.threshold_quantile,
        "calibrated_threshold": calibrated_threshold,
        "change_thresholds": change_thresholds,
        "match_threshold_ratios": match_ratios,
        "deployment_change_threshold": deployment_threshold,
        "deployment_match_ratio": deployment_match_ratio,
        "selection_policy": "Seq1-only calibration; stream labels never select deployment",
        "modality_scales": {key: float(value) for key, value in modality_scales.items()},
    }
    atomic_write_json(run_dir / "seq1_calibration.json", calibration_payload)

    orders = build_disjoint_orders(
        scene_order, [initial_scene], revisit_order, args.num_random_orders, args.seed
    )
    metric_rows = []
    decision_frames = []
    event_frames = []
    predicted_event_frames = []
    order_payload = []
    canonical_outputs = None
    for order_id, blocks in orders:
        order_payload.append({"order_id": order_id, "blocks": blocks})
        stream_arrays, stream_index = concatenate_partition_stream(
            {"train": {"fused": fused["train"]}, "test": {"fused": fused["test"]}},
            indices, blocks,
        )
        stream_features, window_index = aggregate_stream_windows(
            stream_arrays["fused"], stream_index, args.window, args.stride
        )
        truth = window_index["stream_true_sequence"].astype(str).to_numpy()
        for multiplier, change_threshold, match_ratio, match_threshold in threshold_settings:
            manager = CausalContextManager(
                change_threshold=change_threshold,
                match_threshold=match_threshold,
                change_persistence=args.change_persistence,
                memory_size=args.memory_size,
                neighbors=args.memory_neighbors,
                switch_margin=args.switch_margin,
                update_threshold_ratio=args.memory_update_threshold_ratio,
                memory_min_separation_ratio=args.memory_min_separation_ratio,
            )
            initial_context = manager.initialize_context(
                initial_contexts[0]["fit_features"], count=initial_contexts[0]["fit_count"]
            )
            decisions = [manager.observe(value) for value in stream_features]
            metrics, true_events, predicted_events = evaluate_causal_events(
                window_index, decisions, {initial_scene: initial_context},
                args.boundary_tolerance, len(manager.memories),
            )
            context_ids = np.asarray([decision.context_id for decision in decisions])
            mapping = hungarian_mapping(truth, context_ids)
            mapped = apply_mapping(context_ids, mapping)
            diagnostics = clustering_metrics(truth, context_ids, mapped)
            metrics.update({
                "window_accuracy": diagnostics["accuracy"],
                "window_balanced_accuracy": diagnostics["balanced_accuracy"],
                "window_macro_f1": diagnostics["macro_f1"],
                "ari": diagnostics["ari"],
                "nmi": diagnostics["nmi"],
                "purity": purity(truth, context_ids),
            })
            is_deployment = bool(
                np.isclose(change_threshold, deployment_threshold)
                and np.isclose(match_ratio, deployment_match_ratio)
            )
            metric_rows.append({
                "order_id": order_id,
                "change_multiplier": multiplier,
                "change_threshold": change_threshold,
                "match_threshold_ratio": match_ratio,
                "match_threshold": match_threshold,
                "is_deployment_threshold": is_deployment,
                "windows": len(window_index),
                **metrics,
            })
            decision_frame = window_index.copy()
            for field in asdict(decisions[0]):
                decision_frame[field] = [getattr(decision, field) for decision in decisions]
            decision_frame.insert(0, "order_id", order_id)
            decision_frame["change_multiplier"] = multiplier
            decision_frame["change_threshold"] = change_threshold
            decision_frame["match_threshold_ratio"] = match_ratio
            decision_frame["match_threshold"] = match_threshold
            decision_frame["is_deployment_threshold"] = is_deployment
            decision_frames.append(decision_frame)
            for output, collection in (
                (true_events, event_frames), (predicted_events, predicted_event_frames)
            ):
                output = output.copy()
                output.insert(0, "order_id", order_id)
                output["change_multiplier"] = multiplier
                output["change_threshold"] = change_threshold
                output["match_threshold_ratio"] = match_ratio
                output["match_threshold"] = match_threshold
                output["is_deployment_threshold"] = is_deployment
                collection.append(output)
            threshold_name = f"{change_threshold:.6f}".replace(".", "p")
            ratio_name = f"{match_ratio:.3f}".replace(".", "p")
            stem = f"manager_{order_id}_t{threshold_name}_r{ratio_name}"
            joblib.dump(manager, run_dir / f"{stem}.joblib")
            atomic_write_json(
                run_dir / f"{stem}.json",
                manager.state_dict(),
            )
            if order_id == "canonical" and is_deployment:
                canonical_outputs = (window_index, decisions, true_events, predicted_events)

    metrics_frame = pd.DataFrame(metric_rows)
    summary = metrics_frame.groupby(
        [
            "change_multiplier", "change_threshold", "match_threshold_ratio",
            "match_threshold", "is_deployment_threshold",
        ],
        as_index=False,
    ).mean(numeric_only=True)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    pd.concat(decision_frames, ignore_index=True).to_csv(run_dir / "decisions.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(run_dir / "true_event_results.csv", index=False)
    pd.concat(predicted_event_frames, ignore_index=True).to_csv(
        run_dir / "predicted_events.csv", index=False
    )
    if canonical_outputs is None:
        raise RuntimeError("canonical deployment output was not produced")
    save_timeline(
        *canonical_outputs,
        run_dir / "canonical_deployment_timeline.png",
        (
            "V4 causal boundary/context replay, "
            f"change={deployment_threshold:.4f}, match_ratio={deployment_match_ratio:.2f}"
        ),
    )
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-online-context-discovery/v4",
        "status": "complete",
        "created_utc": utc_now(),
        "descriptor_bank": str(bank.root),
        "arguments": vars(args),
        "manager_inputs": "projected descriptors only; no sequence, block, split, or visit metadata",
        "decision_policy": (
            "sustained departure -> frozen candidate buffer -> segment-level "
            "stay/expand, REUSE, or CREATE"
        ),
        "evaluation_policy": (
            "ground-truth boundaries are consumed only after inference for causal "
            "one-to-one event matching and strict CREATE/REUSE scoring"
        ),
        "calibration": calibration_payload,
        "orders": order_payload,
        "run_dir": str(run_dir),
    })
    print("V4 Seq1-only calibration:")
    print(pd.DataFrame([calibration_payload]).to_string(index=False))
    print("\nV4 causal metrics across orders:")
    display = [
        "change_multiplier", "change_threshold", "match_threshold_ratio",
        "is_deployment_threshold",
        "change_precision", "change_recall", "change_f1",
        "mean_detection_delay_windows", "correct_creation_rate",
        "correct_reuse_rate", "end_to_end_success_rate", "final_contexts",
        "context_count_error", "window_accuracy", "ari", "nmi",
    ]
    print(summary[display].to_string(index=False))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
