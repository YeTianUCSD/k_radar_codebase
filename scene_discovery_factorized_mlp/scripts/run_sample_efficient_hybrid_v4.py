#!/usr/bin/env python3
"""Evaluate label-, frame-, and latency-efficient hybrid attribute heads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, REPOSITORY_ROOT / "scene_discovery", SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from factorized_mlp.budget_protocol import (  # noqa: E402
    build_budget_manifest, endpoint_positions, unique_window_frame_count,
)
from factorized_mlp.data import FeatureStandardizer, load_split_arrays  # noqa: E402
from factorized_mlp.fixed_training import train_fixed_epochs  # noqa: E402
from factorized_mlp.hybrid_model import (  # noqa: E402
    HybridAttributeMLP, WeatherMultimodalAttributeMLP,
)
from factorized_mlp.illumination import (  # noqa: E402
    load_aligned_illumination_bank,
)
from factorized_mlp.model import FactorizedGatedMLP  # noqa: E402
from factorized_mlp.spatial_lighting_model import (  # noqa: E402
    WeatherMultimodalSpatialLightingMLP,
)
from factorized_mlp.independent_model import (  # noqa: E402
    IndependentGatedSpatialAttributeMLP,
)
from factorized_mlp.training import predict_logits, seed_everything  # noqa: E402
from factorized_mlp.window_protocol import aggregate_causal_endpoints  # noqa: E402
from run_camera_window_experiment_v2 import (  # noqa: E402
    class_maps, labels_for_index, score_predictions,
)
from scene_discovery.attributes import ATTRIBUTES  # noqa: E402
from scene_discovery.common import atomic_write_json  # noqa: E402


def parse_args(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=(default_config or
                 PROJECT_ROOT / "configs/sample_efficient_hybrid_v4.yml"),
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def make_run_dir(root, quick, experiment_name="sample_efficient_hybrid_v4"):
    stamp = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
    prefix = f"{experiment_name}_quick" if quick else experiment_name
    path = Path(root) / f"{prefix}_{stamp}"
    suffix = 1
    while path.exists():
        path = Path(root) / f"{prefix}_{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    for child in ("manifests", "checkpoints"):
        (path / child).mkdir()
    return path


def make_model(architecture, input_dims, class_counts, modalities, training):
    common = {
        "embedding_dim": int(training["embedding_dim"]),
        "head_hidden_dim": int(training["head_hidden_dim"]),
        "dropout": float(training["dropout"]),
    }
    if architecture == "camera":
        return FactorizedGatedMLP(
            input_dims, class_counts, modalities=("camera",),
            fusion="gated", **common,
        ), ("camera",)
    if architecture in {"hybrid_concat", "hybrid_gated"}:
        fusion = architecture.split("_", 1)[1]
        return HybridAttributeMLP(
            input_dims, class_counts, modalities=modalities,
            fusion=fusion, **common,
        ), tuple(modalities)
    if architecture in {
        "weather_multimodal_concat", "weather_multimodal_gated"
    }:
        fusion = architecture.rsplit("_", 1)[1]
        return WeatherMultimodalAttributeMLP(
            input_dims, class_counts, modalities=modalities,
            fusion=fusion, **common,
        ), tuple(modalities)
    if architecture in {
        "weather_multimodal_concat_spatial_lighting",
        "weather_multimodal_gated_spatial_lighting",
    }:
        fusion = "concat" if "_concat_" in architecture else "gated"
        return WeatherMultimodalSpatialLightingMLP(
            input_dims, class_counts, modalities=modalities, fusion=fusion,
            illumination_embedding_dim=int(
                training.get("illumination_embedding_dim", 16)
            ), **common,
        ), tuple(modalities) + ("illumination",)
    if architecture == "independent_gated_spatial_lighting":
        return IndependentGatedSpatialAttributeMLP(
            input_dims, class_counts,
            weather_modalities=modalities,
            road_modalities=tuple(training.get(
                "road_modalities", ("camera", "lidar")
            )),
            illumination_embedding_dim=int(
                training.get("illumination_embedding_dim", 16)
            ),
            **common,
        ), tuple(modalities) + ("illumination",)
    raise ValueError(f"unknown architecture {architecture!r}")


def standardize(train_features, test_features):
    scaler = FeatureStandardizer.fit(
        train_features,
        np.ones(len(next(iter(train_features.values()))), dtype=bool),
    )
    train = {
        key: scaler.transform(key, value) for key, value in train_features.items()
    }
    test = {
        key: scaler.transform(key, value) for key, value in test_features.items()
    }
    return scaler, train, test


def model_predictions(model, features, names, batch_size, device):
    logits = predict_logits(
        model, features, np.arange(len(next(iter(features.values())))),
        batch_size, device,
    )
    predictions, confidences = {}, {}
    for attribute in ATTRIBUTES:
        values = logits[attribute]
        values = values - values.max(axis=1, keepdims=True)
        exponent = np.exp(values)
        probability = exponent / exponent.sum(axis=1, keepdims=True)
        positions = probability.argmax(axis=1)
        predictions[attribute] = np.asarray(names[attribute])[positions]
        confidences[attribute] = probability.max(axis=1)
    return predictions, confidences


def aggregate_modalities(arrays, index, endpoints, window, modalities):
    features, common_index = {}, None
    for modality in modalities:
        values, current_index = aggregate_causal_endpoints(
            arrays[modality], index, endpoints, window
        )
        features[modality] = values
        if common_index is None:
            common_index = current_index.reset_index(drop=True)
        elif not np.array_equal(
            common_index["split_position"], current_index["split_position"]
        ):
            raise RuntimeError("modality endpoints are misaligned")
    return features, common_index


def pareto_mask(frame):
    values = frame.reset_index(drop=True)
    keep = np.ones(len(values), dtype=bool)
    for current, row in values.iterrows():
        dominated = (
            (values["labeled_windows"] <= row["labeled_windows"])
            & (values["unique_train_frames"] <= row["unique_train_frames"])
            & (values["window"] <= row["window"])
            & (values["joint_accuracy"] >= row["joint_accuracy"])
            & (
                (values["labeled_windows"] < row["labeled_windows"])
                | (values["unique_train_frames"] < row["unique_train_frames"])
                | (values["window"] < row["window"])
                | (values["joint_accuracy"] > row["joint_accuracy"])
            )
        )
        dominated.iloc[current] = False
        keep[current] = not dominated.any()
    return keep


def recommendation_row(frame, order, ascending):
    if frame.empty:
        return None
    row = frame.sort_values(order, ascending=ascending).iloc[0]
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.to_dict().items()
    }


def main(default_config=None):
    args = parse_args(default_config)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    descriptor_bank = Path(config["descriptor_bank"]).expanduser().resolve()
    run_dir = make_run_dir(
        config["output_root"], args.quick,
        config.get("experiment_name", "sample_efficient_hybrid_v4"),
    )
    modalities = tuple(config["modalities"])
    illumination_config = config.get("illumination")
    illumination_bank = None
    feature_names = modalities
    if illumination_config:
        illumination_bank = Path(
            illumination_config["bank"]
        ).expanduser().resolve()
        feature_names = modalities + ("illumination",)
    arrays, indices = {}, {}
    for split in ("train", "test"):
        arrays[split], index = load_split_arrays(
            descriptor_bank, split, modalities, config["descriptor"]
        )
        indices[split] = index.reset_index(drop=True)
        if illumination_bank is not None:
            arrays[split]["illumination"] = load_aligned_illumination_bank(
                illumination_bank, split, indices[split]
            )
    windows = ([int(max(config["windows"]))] if args.quick else
               [int(value) for value in config["windows"]])
    budgets = ([int(min(config["support_budgets"]))] if args.quick else
               [int(value) for value in config["support_budgets"]])
    seeds = ([int(config["seeds"][0])] if args.quick else
             [int(value) for value in config["seeds"]])
    names, mappings = class_maps()
    class_counts = {key: len(value) for key, value in names.items()}
    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifests, endpoint_sets = {}, {}
    for budget in budgets:
        manifest = build_budget_manifest(
            indices["train"], indices["test"], budget, windows
        )
        manifest.to_csv(
            run_dir / "manifests" / f"budget_{budget}.csv", index=False
        )
        manifests[budget] = manifest
        endpoint_sets[budget] = {
            "train": endpoint_positions(manifest, "train", "support_train"),
            "test": endpoint_positions(manifest, "test", "official_test"),
        }

    metrics_parts, class_parts, sequence_parts = [], [], []
    gate_rows, history_rows, usage_rows, parameter_rows = [], [], [], []
    for budget in budgets:
        train_endpoints = endpoint_sets[budget]["train"]
        test_endpoints = endpoint_sets[budget]["test"]
        sequence_count = manifests[budget][
            manifests[budget].source_split.eq("train")
        ].sequence.nunique()
        for window in windows:
            train_features, train_index = aggregate_modalities(
                arrays["train"], indices["train"], train_endpoints,
                window, feature_names,
            )
            test_features, test_index = aggregate_modalities(
                arrays["test"], indices["test"], test_endpoints,
                window, feature_names,
            )
            train_numeric, _ = labels_for_index(train_index, mappings)
            _, test_text = labels_for_index(test_index, mappings)
            scaler, train_z, test_z = standardize(train_features, test_features)
            usage_rows.append({
                "support_budget_per_scene": budget,
                "labeled_windows": int(budget * sequence_count),
                "window": window,
                "unique_train_frames": unique_window_frame_count(
                    train_endpoints, window
                ),
                "train_frame_reads_with_repetition": int(len(train_endpoints) * window),
                "scored_test_windows": int(len(test_endpoints)),
                "unique_test_frames": unique_window_frame_count(
                    test_endpoints, window
                ),
            })
            positions = np.arange(len(train_index))
            input_dims = {
                key: int(value.shape[1]) for key, value in train_z.items()
            }
            for architecture in config["architectures"]:
                for seed in seeds:
                    seed_everything(seed)
                    model, used_modalities = make_model(
                        architecture, input_dims, class_counts,
                        modalities, training,
                    )
                    parameter_count = int(sum(
                        parameter.numel() for parameter in model.parameters()
                    ))
                    result = train_fixed_epochs(
                        model,
                        {key: train_z[key] for key in used_modalities},
                        train_numeric, positions,
                        learning_rate=float(training["learning_rate"]),
                        weight_decay=float(training["weight_decay"]),
                        batch_size=int(training["batch_size"]),
                        epochs=3 if args.quick else int(training["epochs"]),
                        seed=seed, device=device,
                        class_weight_exponents=training.get("class_weight_exponents", {}),
                    )
                    predictions, confidences = model_predictions(
                        result["model"],
                        {key: test_z[key] for key in used_modalities},
                        names, int(training["batch_size"]), device,
                    )
                    frames = score_predictions(
                        test_index, test_text, predictions, confidences,
                        method=architecture, budget=budget, window=window,
                        seed=seed, scope="official_test",
                    )
                    metrics_parts.append(frames[0])
                    class_parts.append(frames[1])
                    sequence_parts.append(frames[2])
                    parameter_rows.append({
                        "architecture": architecture,
                        "parameters": parameter_count,
                    })
                    for row in result["history"]:
                        history_rows.append({
                            "architecture": architecture,
                            "support_budget": budget, "window": window,
                            "seed": seed, **row,
                        })
                    if architecture in {
                        "hybrid_gated", "weather_multimodal_gated",
                        "weather_multimodal_gated_spatial_lighting",
                        "independent_gated_spatial_lighting",
                    }:
                        weights = result["model"].gate_weights()
                        gate_modalities = (
                            result["model"].gate_modalities()
                            if hasattr(result["model"], "gate_modalities") else {}
                        )
                        for attribute, attribute_weights in weights.items():
                            for position, modality in enumerate(
                                gate_modalities.get(attribute, modalities)
                            ):
                                gate_rows.append({
                                    "architecture": architecture,
                                    "support_budget": budget,
                                    "window": window, "seed": seed,
                                    "attribute": attribute,
                                    "modality": modality,
                                    "weight": float(
                                        attribute_weights[position].detach().cpu()
                                    ),
                                    "learned": True,
                                })
                        for attribute in set(ATTRIBUTES) - set(weights):
                            gate_rows.append({
                                "architecture": architecture,
                                "support_budget": budget, "window": window,
                                "seed": seed, "attribute": attribute,
                                "modality": "camera", "weight": 1.0,
                                "learned": False,
                            })
                    if (
                        config.get("save_checkpoint_first_seed", True)
                        and seed == seeds[0]
                    ):
                        torch.save({
                            "model_state_dict": result["model"].cpu().state_dict(),
                            "architecture": architecture,
                            "model_config": training,
                            "modalities": used_modalities,
                            "sensor_modalities": modalities,
                            "standardizer": scaler.state_dict(),
                            "label_names": names,
                            "support_budget": budget,
                            "window": window, "seed": seed,
                            "epochs": result["epochs"],
                        }, run_dir / "checkpoints" /
                            f"{architecture}_budget{budget}_window{window}_seed{seed}.pt")

    metrics = pd.concat(metrics_parts, ignore_index=True)
    classes = pd.concat(class_parts, ignore_index=True)
    sequences = pd.concat(sequence_parts, ignore_index=True)
    usage = pd.DataFrame(usage_rows).drop_duplicates()
    parameters = pd.DataFrame(parameter_rows).drop_duplicates()
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    classes.to_csv(run_dir / "per_class_metrics.csv", index=False)
    sequences.to_csv(run_dir / "per_sequence_metrics.csv", index=False)
    usage.to_csv(run_dir / "sample_usage.csv", index=False)
    parameters.to_csv(run_dir / "model_parameters.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(run_dir / "gate_weights.csv", index=False)
    pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)

    summary = metrics.groupby(
        ["method", "support_budget", "window", "attribute"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    accuracy = summary.pivot_table(
        index=["method", "support_budget", "window"],
        columns="attribute", values="accuracy_mean",
    ).reset_index().rename(columns={
        "method": "architecture", "joint": "joint_accuracy",
        "weather": "weather_accuracy", "road": "road_accuracy",
        "lighting": "lighting_accuracy",
    })
    joint_std = summary[summary.attribute.eq("joint")][
        ["method", "support_budget", "window", "accuracy_std"]
    ].rename(columns={"method": "architecture", "accuracy_std": "joint_std"})
    configurations = accuracy.merge(
        joint_std, on=["architecture", "support_budget", "window"]
    ).merge(
        usage, left_on=["support_budget", "window"],
        right_on=["support_budget_per_scene", "window"]
    ).merge(parameters, on="architecture")
    tolerance = float(config["selection"]["absolute_accuracy_tolerance"])
    best = {
        attribute: float(configurations[f"{attribute}_accuracy"].max())
        for attribute in ("joint", "weather", "road", "lighting")
    }
    configurations["within_joint_tolerance"] = (
        configurations.joint_accuracy >= best["joint"] - tolerance
    )
    configurations["all_attributes_within_tolerance"] = np.logical_and.reduce([
        configurations[f"{attribute}_accuracy"] >= best[attribute] - tolerance
        for attribute in ("weather", "road", "lighting")
    ])
    configurations["eligible"] = configurations.within_joint_tolerance
    if config["selection"].get("require_each_attribute_within_tolerance", True):
        configurations["eligible"] &= configurations.all_attributes_within_tolerance
    configurations = configurations.sort_values(
        ["joint_accuracy", "labeled_windows"], ascending=[False, True]
    )
    configurations.to_csv(run_dir / "all_configurations.csv", index=False)
    within = configurations[configurations.within_joint_tolerance]
    within.to_csv(run_dir / "within_3pct.csv", index=False)
    guarded = configurations[configurations.eligible]
    guarded.to_csv(run_dir / "within_3pct_guarded.csv", index=False)
    pareto = configurations[pareto_mask(configurations)]
    pareto.to_csv(run_dir / "pareto_frontier.csv", index=False)
    choice_pool = guarded if not guarded.empty else within
    recommendations = {
        "accuracy_tolerance": tolerance,
        "best_observed": best,
        "selection_is_posthoc_on_official_test": True,
        "minimum_labels": recommendation_row(
            choice_pool,
            ["labeled_windows", "unique_train_frames", "window", "joint_std", "joint_accuracy"],
            [True, True, True, True, False],
        ),
        "minimum_unique_frames": recommendation_row(
            choice_pool,
            ["unique_train_frames", "labeled_windows", "window", "joint_std", "joint_accuracy"],
            [True, True, True, True, False],
        ),
        "minimum_latency": recommendation_row(
            choice_pool,
            ["window", "labeled_windows", "unique_train_frames", "joint_std", "joint_accuracy"],
            [True, True, True, True, False],
        ),
    }
    atomic_write_json(run_dir / "recommendations.json", recommendations)
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": config.get(
            "schema_version", "kradar-sample-efficient-hybrid/v4"
        ),
        "status": "complete", "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "config_path": str(config_path), "config": config,
        "descriptor_bank": str(descriptor_bank), "device": str(device),
        "feature_dimensions": {
            key: int(arrays["train"][key].shape[1]) for key in feature_names
        },
        "illumination_bank": str(illumination_bank) if illumination_bank else None,
        "no_external_validation_labels": True,
        "selection_warning": (
            "within_3pct and recommendations are exploratory posthoc analyses of "
            "official-test results; confirm the chosen setting on new held-out data"
        ),
        "quick": bool(args.quick), "run_dir": str(run_dir),
    })
    print("Best observed accuracies:")
    print(json.dumps(best, indent=2))
    print("\nRecommendations within tolerance:")
    print(json.dumps(recommendations, indent=2))
    print("\nTop configurations:")
    print(configurations.head(20).to_string(index=False))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
