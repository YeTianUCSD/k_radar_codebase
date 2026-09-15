#!/usr/bin/env python3
"""Simulate semantic context creation and reuse from factorized predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

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

from scene_discovery.attribute_registry import SemanticContextRegistry  # noqa: E402
from scene_discovery.attributes import ATTRIBUTES, attribute_frame, sequence_attribute_table  # noqa: E402
from scene_discovery.common import atomic_write_json, make_run_dir, seed_everything, utc_now  # noqa: E402
from scene_discovery.evaluation import classification_metrics  # noqa: E402
from scene_discovery.feature_bank import DEFAULT_SEQUENCES, DescriptorBank  # noqa: E402
from scene_discovery.temporal import aggregate_stream_windows, concatenate_scene_stream  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument(
        "--descriptor-bank", type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/descriptors/v1",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/experiments",
    )
    parser.add_argument("--descriptor", choices=("mean", "mean_std", "spatial"), default="mean_std")
    parser.add_argument("--scene-order", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--repeat-scenes", nargs="*", default=["1", "35"])
    parser.add_argument(
        "--initial-scenes", nargs="*", default=[],
        help="Optional ground-truth seed scenes; omit for fully ID-free bootstrap.",
    )
    parser.add_argument("--num-random-orders", type=int, default=4)
    parser.add_argument("--probability-alpha", type=float, default=0.3)
    parser.add_argument("--weather-confidence", type=float, default=None)
    parser.add_argument("--road-confidence", type=float, default=None)
    parser.add_argument("--lighting-confidence", type=float, default=None)
    parser.add_argument("--create-persistence", type=int, default=5)
    parser.add_argument("--switch-persistence", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def normalize_scene(value: object) -> str:
    result = str(value).lower()
    if result.startswith("seq"):
        result = result[3:]
    return str(int(result))


def build_orders(base_order, repeat_scenes, random_orders, seed):
    unique = list(dict.fromkeys(base_order))
    if any(scene not in unique for scene in repeat_scenes):
        raise ValueError("Every repeated scene must occur in --scene-order")
    orders = [("canonical", list(base_order) + list(repeat_scenes))]
    rng = np.random.RandomState(seed)
    seen = {tuple(unique)}
    attempts = 0
    while len(orders) < random_orders + 1:
        attempts += 1
        candidate = list(rng.permutation(unique))
        key = tuple(candidate)
        if key in seen:
            if attempts > 1000:
                raise RuntimeError("Could not generate enough unique scene orders")
            continue
        seen.add(key)
        orders.append((f"random_{len(orders):02d}", candidate + list(repeat_scenes)))
    return orders


def smooth_probabilities(probabilities: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("--probability-alpha must be in (0, 1]")
    output = np.empty_like(probabilities, dtype=np.float32)
    state = probabilities[0].astype(np.float64)
    output[0] = state
    for position in range(1, len(probabilities)):
        state = (1.0 - alpha) * state + alpha * probabilities[position]
        state = state / state.sum()
        output[position] = state
    return output


def save_timeline(frame: pd.DataFrame, path: Path, title: str) -> None:
    labels = sorted(frame["true_semantic_key"].astype(str).unique())
    encoding = {label: position for position, label in enumerate(labels)}
    truth = frame["true_semantic_key"].map(encoding).to_numpy()
    assigned = frame["assigned_semantic_key"].map(encoding).fillna(-1).to_numpy()
    x = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(17, 6), constrained_layout=True)
    axis.step(x, truth, where="post", linewidth=2, label="True semantic context")
    axis.step(x, assigned + 0.08, where="post", linewidth=1.2, label="Registry assignment")
    created = frame["created_context"].to_numpy(dtype=bool)
    axis.scatter(x[created], assigned[created] + 0.08, marker="x", color="red", label="Created")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Causal stream window")
    axis.set_ylabel("Semantic context")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def block_diagnostics(frame: pd.DataFrame) -> Dict[str, float]:
    reuse_trials = 0
    reuse_correct = 0
    seen = set()
    switch_latencies = []
    blocks = list(frame["stream_block"].drop_duplicates())
    for block_position, block in enumerate(blocks):
        current = frame[frame["stream_block"].eq(block)]
        truth = str(current["true_semantic_key"].iloc[-1])
        assigned = current["assigned_semantic_key"].astype(str).to_numpy()
        correct = np.flatnonzero(assigned == truth)
        if block_position > 0:
            switch_latencies.append(int(correct[0]) if len(correct) else len(current))
        if truth in seen:
            reuse_trials += 1
            reuse_correct += int(np.mean(assigned == truth) >= 0.5)
        else:
            seen.add(truth)
    return {
        "reuse_trials": reuse_trials,
        "reuse_block_accuracy": reuse_correct / reuse_trials if reuse_trials else float("nan"),
        "mean_switch_latency_windows": (
            float(np.mean(switch_latencies)) if switch_latencies else 0.0
        ),
        "max_switch_latency_windows": (
            int(np.max(switch_latencies)) if switch_latencies else 0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.num_random_orders < 0:
        raise ValueError("--num-random-orders must be non-negative")
    seed_everything(args.seed)
    recognizer = joblib.load(args.recognizer.expanduser().resolve())
    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(args.output_root.expanduser().resolve(), "attribute_registry")

    base_order = [normalize_scene(value) for value in args.scene_order]
    repeats = [normalize_scene(value) for value in args.repeat_scenes]
    initial_scenes = [normalize_scene(value) for value in args.initial_scenes]
    orders = build_orders(base_order, repeats, args.num_random_orders, args.seed)
    test_index = bank.index("test")
    test_arrays = {
        modality: bank.array("test", modality, args.descriptor)
        for modality in recognizer.projector.modalities
    }
    projected = recognizer.projector.transform_modalities(test_arrays)
    scene_table = sequence_attribute_table(test_index)
    scene_table["sequence"] = scene_table["sequence"].astype(str)
    scene_to_key = dict(zip(scene_table["sequence"], scene_table["semantic_key"]))
    missing_initial = [scene for scene in initial_scenes if scene not in scene_to_key]
    if missing_initial:
        raise ValueError(f"Initial scenes absent from descriptor test split: {missing_initial}")

    stored_thresholds = getattr(recognizer, "confidence_thresholds", {}) or {}
    cli_thresholds = {
        "weather": args.weather_confidence,
        "road": args.road_confidence,
        "lighting": args.lighting_confidence,
    }
    confidence_thresholds = {}
    threshold_sources = {}
    for attribute in ATTRIBUTES:
        if cli_thresholds[attribute] is not None:
            confidence_thresholds[attribute] = float(cli_thresholds[attribute])
            threshold_sources[attribute] = "command_line"
        elif attribute in stored_thresholds:
            confidence_thresholds[attribute] = float(stored_thresholds[attribute])
            threshold_sources[attribute] = "recognizer_calibration"
        else:
            confidence_thresholds[attribute] = 0.7
            threshold_sources[attribute] = "legacy_fallback_0.7"
    metrics_rows = []
    per_scene_rows = []
    prediction_frames = []
    registry_index = []

    for order_id, scene_order in orders:
        stream_key_universe = {scene_to_key[scene] for scene in set(scene_order)}
        stream_parts = {}
        window_index = None
        for modality, values in projected.items():
            stream_values, current_index = concatenate_scene_stream(
                values, test_index, scene_order
            )
            windows, current_window_index = aggregate_stream_windows(
                stream_values,
                current_index,
                recognizer.window,
                recognizer.stride,
            )
            stream_parts[modality] = windows
            if window_index is None:
                window_index = current_window_index.reset_index(drop=True)
            elif not np.array_equal(
                window_index["stream_row"].to_numpy(),
                current_window_index["stream_row"].to_numpy(),
            ):
                raise RuntimeError("Modality streams are misaligned")

        predictions = {}
        confidences = {}
        for attribute in ATTRIBUTES:
            head = recognizer.heads[attribute]
            matrix = head.feature_matrix(stream_parts)
            raw_probabilities = np.asarray(
                head.estimator.predict_proba(matrix), dtype=np.float32
            )
            probabilities = smooth_probabilities(raw_probabilities, args.probability_alpha)
            positions = probabilities.argmax(axis=1)
            classes = np.asarray(head.estimator.classes_).astype(str)
            predictions[attribute] = classes[positions]
            confidences[attribute] = probabilities.max(axis=1)

        predicted_keys = np.asarray([
            f"{weather}|{road}|{lighting}"
            for weather, road, lighting in zip(
                predictions["weather"], predictions["road"], predictions["lighting"]
            )
        ])
        registry = SemanticContextRegistry(
            confidence_thresholds=confidence_thresholds,
            create_persistence=args.create_persistence,
            switch_persistence=args.switch_persistence,
        )
        for scene in initial_scenes:
            registry.register(scene_to_key[scene])

        decisions = []
        for position, key in enumerate(predicted_keys):
            decisions.append(registry.observe(
                key,
                {
                    attribute: float(confidences[attribute][position])
                    for attribute in ATTRIBUTES
                },
            ))
        assigned_keys = np.asarray(
            [decision.semantic_key for decision in decisions], dtype=object
        )
        true_attributes = attribute_frame(window_index)
        true_keys = true_attributes["semantic_key"].to_numpy()
        raw_exact = predicted_keys == true_keys
        registry_exact = assigned_keys == true_keys
        created_keys = {
            decision.semantic_key for decision in decisions if decision.created
        }
        diagnostics = block_diagnostics(pd.DataFrame({
            "stream_block": window_index["stream_block"],
            "true_semantic_key": true_keys,
            "assigned_semantic_key": assigned_keys,
        }))
        metrics_rows.append({
            "order_id": order_id,
            "windows": len(window_index),
            "scene_blocks": len(scene_order),
            "raw_semantic_accuracy": float(raw_exact.mean()),
            "registry_semantic_accuracy": float(registry_exact.mean()),
            "accepted_rate": float(np.mean([decision.accepted for decision in decisions])),
            "active_contexts": len(registry.key_to_context),
            "expected_contexts": len(stream_key_universe),
            "context_count_error": len(registry.key_to_context) - len(stream_key_universe),
            "created_context_events": int(sum(decision.created for decision in decisions)),
            "reused_context_events": int(sum(decision.reused for decision in decisions)),
            "false_created_contexts": len(created_keys - stream_key_universe),
            "missed_true_contexts": len(stream_key_universe - set(registry.key_to_context)),
            **{
                f"{attribute}_accuracy": classification_metrics(
                    true_attributes[attribute], predictions[attribute]
                )["accuracy"]
                for attribute in ATTRIBUTES
            },
            **diagnostics,
        })
        for sequence in sorted(window_index["stream_true_sequence"].astype(str).unique(), key=int):
            mask = window_index["stream_true_sequence"].astype(str).eq(sequence).to_numpy()
            per_scene_rows.append({
                "order_id": order_id,
                "sequence": sequence,
                "samples": int(mask.sum()),
                "raw_semantic_accuracy": float(raw_exact[mask].mean()),
                "registry_semantic_accuracy": float(registry_exact[mask].mean()),
                **{
                    f"{attribute}_accuracy": float(np.mean(
                        predictions[attribute][mask]
                        == true_attributes[attribute].to_numpy()[mask]
                    ))
                    for attribute in ATTRIBUTES
                },
            })

        output = window_index.copy()
        output["order_id"] = order_id
        output["true_sequence"] = window_index["stream_true_sequence"].astype(str)
        for attribute in ATTRIBUTES:
            output[f"true_{attribute}"] = true_attributes[attribute].to_numpy()
            output[f"predicted_{attribute}"] = predictions[attribute]
            output[f"{attribute}_confidence"] = confidences[attribute]
        output["true_semantic_key"] = true_keys
        output["predicted_semantic_key"] = predicted_keys
        output["assigned_semantic_key"] = assigned_keys
        output["context_id"] = [decision.context_id for decision in decisions]
        output["prediction_accepted"] = [decision.accepted for decision in decisions]
        output["created_context"] = [decision.created for decision in decisions]
        output["reused_context"] = [decision.reused for decision in decisions]
        output["pending_key"] = [decision.pending_key for decision in decisions]
        output["pending_count"] = [decision.pending_count for decision in decisions]
        output["joint_confidence"] = [decision.joint_confidence for decision in decisions]
        output["raw_exact_match"] = raw_exact
        output["registry_exact_match"] = registry_exact
        prediction_frames.append(output)

        joblib_name = f"registry_{order_id}.joblib"
        json_name = f"registry_{order_id}.json"
        joblib.dump(registry, run_dir / joblib_name)
        atomic_write_json(run_dir / json_name, registry.state_dict())
        registry_index.append({
            "order_id": order_id,
            "scene_order": scene_order,
            "joblib": joblib_name,
            "json": json_name,
        })
        if order_id == "canonical":
            save_timeline(
                output,
                run_dir / "canonical_timeline.png",
                "Factorized semantic context registry",
            )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(per_scene_rows).to_csv(run_dir / "per_scene_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        run_dir / "predictions.csv", index=False
    )
    metrics.mean(numeric_only=True).to_frame("mean").to_csv(run_dir / "summary.csv")
    atomic_write_json(run_dir / "registries.json", registry_index)
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-attribute-registry/v1",
        "status": "complete",
        "created_utc": utc_now(),
        "recognizer": str(args.recognizer.expanduser().resolve()),
        "recognizer_metadata": recognizer.metadata(),
        "descriptor_bank": str(bank.root),
        "recognizer_training_sequences": list(
            getattr(recognizer, "training_sequences", [])
        ),
        "evaluation_sequences": list(
            test_index["sequence"].astype(str).drop_duplicates()
        ),
        "training_evaluation_overlap": sorted(
            set(getattr(recognizer, "training_sequences", []))
            & set(test_index["sequence"].astype(str))
        ),
        "arguments": vars(args),
        "confidence_thresholds": confidence_thresholds,
        "confidence_threshold_sources": threshold_sources,
        "bootstrap_mode": (
            "ground_truth_seed" if initial_scenes else "prediction_only"
        ),
        "algorithm_inputs": (
            "factorized predictions and confidences; ground-truth labels only score outputs"
            if not initial_scenes
            else "factorized predictions plus explicitly requested ground-truth initial context; remaining labels only score outputs"
        ),
        "context_policy": "lazy semantic-key creation with per-head confidence gates and temporal persistence",
        "orders": registry_index,
        "run_dir": str(run_dir),
    })
    print(metrics.to_string(index=False))
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
