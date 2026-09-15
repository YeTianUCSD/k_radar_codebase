#!/usr/bin/env python3
"""Compare Camera, multimodal concat, and attribute-gated MLPs fairly."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
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

from factorized_mlp.data import FeatureStandardizer, load_split_arrays  # noqa: E402
from factorized_mlp.model import FactorizedGatedMLP  # noqa: E402
from factorized_mlp.training import predict_logits, seed_everything, train_model  # noqa: E402
from factorized_mlp.window_protocol import (  # noqa: E402
    aggregate_causal_endpoints, build_blocked_window_manifest, endpoint_positions,
)
from run_camera_window_experiment_v2 import (  # noqa: E402
    class_maps, labels_for_index, score_predictions,
)
from scene_discovery.attributes import ATTRIBUTES  # noqa: E402
from scene_discovery.common import atomic_write_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/multimodal_window_v3.yml",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def make_run_dir(root, quick):
    stamp = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
    prefix = "multimodal_window_v3_quick" if quick else "multimodal_window_v3"
    path = Path(root) / f"{prefix}_{stamp}"
    suffix = 1
    while path.exists():
        path = Path(root) / f"{prefix}_{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    for child in ("manifests", "histories", "checkpoints", "predictions"):
        (path / child).mkdir()
    return path


def architecture_settings(architecture, all_modalities):
    if architecture == "camera":
        return ("camera",), "gated"
    if architecture == "concat":
        return tuple(all_modalities), "concat"
    if architecture == "gated":
        return tuple(all_modalities), "gated"
    raise ValueError(f"unknown architecture {architecture!r}")


def configuration_grid(config, quick):
    search = config["model_search"]
    rows = [
        {
            "architecture": architecture,
            "embedding_dim": int(embedding),
            "head_hidden_dim": int(hidden),
            "dropout": float(dropout),
            "learning_rate": float(learning_rate),
            "weight_decay": float(search["weight_decay"]),
        }
        for architecture, embedding, hidden, dropout, learning_rate in product(
            config["architectures"], search["embedding_dims"],
            search["head_hidden_dims"], search["dropouts"],
            search["learning_rates"],
        )
    ]
    if not quick:
        return rows
    return [next(row for row in rows if row["architecture"] == architecture)
            for architecture in config["architectures"]]


def make_model(model_config, input_dims, class_counts, all_modalities):
    modalities, fusion = architecture_settings(
        model_config["architecture"], all_modalities
    )
    return FactorizedGatedMLP(
        input_dims, class_counts, modalities=modalities, fusion=fusion,
        embedding_dim=int(model_config["embedding_dim"]),
        head_hidden_dim=int(model_config["head_hidden_dim"]),
        dropout=float(model_config["dropout"]),
    )


def standardize(train_features, *others):
    scaler = FeatureStandardizer.fit(
        train_features, np.ones(len(next(iter(train_features.values()))), dtype=bool)
    )
    transformed = []
    for features in (train_features,) + others:
        transformed.append({
            modality: scaler.transform(modality, values)
            for modality, values in features.items()
        })
    return scaler, tuple(transformed)


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


def subset_modalities(features, modalities):
    return {modality: features[modality] for modality in modalities}


def main():
    args = parse_args()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    descriptor_bank = Path(config["descriptor_bank"]).expanduser().resolve()
    run_dir = make_run_dir(config["output_root"], args.quick)
    all_modalities = tuple(config["modalities"])
    if "camera" not in all_modalities:
        raise ValueError("modalities must contain camera for the camera baseline")
    arrays, indices = {}, {}
    for split in ("train", "test"):
        arrays[split], index = load_split_arrays(
            descriptor_bank, split, all_modalities, config["descriptor"]
        )
        indices[split] = index.reset_index(drop=True)
    windows = ([int(max(config["windows"]))] if args.quick
               else [int(value) for value in config["windows"]])
    budgets = ([int(max(config["support_budgets"]))] if args.quick
               else [int(value) for value in config["support_budgets"]])
    names, mappings = class_maps()
    class_counts = {key: len(value) for key, value in names.items()}

    manifests = {}
    for budget in budgets:
        manifest = build_blocked_window_manifest(
            indices["train"], indices["test"], budget, windows,
            validation_fraction=float(config["validation_fraction"]),
            gap_frames=int(config["gap_frames"]),
        )
        manifest.to_csv(
            run_dir / "manifests" / f"blocked_budget_{budget}.csv", index=False
        )
        manifests[budget] = manifest

    cache = {}
    for budget in budgets:
        manifest = manifests[budget]
        endpoint_sets = {
            "support_train": endpoint_positions(manifest, "train", "support_train"),
            "support_validation": endpoint_positions(
                manifest, "train", "support_validation"
            ),
            "official_test": endpoint_positions(manifest, "test", "official_test"),
        }
        for window in windows:
            cache[(budget, window)] = {}
            for scope, endpoints in endpoint_sets.items():
                split = "test" if scope == "official_test" else "train"
                features, common_index = {}, None
                for modality in all_modalities:
                    values, current_index = aggregate_causal_endpoints(
                        arrays[split][modality], indices[split], endpoints, window
                    )
                    features[modality] = values
                    if common_index is None:
                        common_index = current_index.reset_index(drop=True)
                    elif not np.array_equal(
                        common_index["split_position"], current_index["split_position"]
                    ):
                        raise RuntimeError("modality window endpoints are misaligned")
                numeric, text = labels_for_index(common_index, mappings)
                cache[(budget, window)][scope] = (
                    features, common_index, numeric, text
                )

    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maximum_budget = max(budgets)
    search_rows, search_histories = [], {}
    selected_configs = {}
    for window in windows:
        support, _, support_labels, _ = cache[(maximum_budget, window)]["support_train"]
        validation, _, validation_labels, _ = cache[(maximum_budget, window)]["support_validation"]
        scaler, (support_z, validation_z) = standardize(support, validation)
        combined = {
            modality: np.concatenate([support_z[modality], validation_z[modality]])
            for modality in all_modalities
        }
        combined_labels = {
            key: np.concatenate([support_labels[key], validation_labels[key]])
            for key in ATTRIBUTES
        }
        fit_positions = np.arange(len(next(iter(support_z.values()))))
        validation_positions = np.arange(
            len(fit_positions), len(fit_positions) + len(next(iter(validation_z.values())))
        )
        candidates = configuration_grid(config, args.quick)
        for config_id, model_config in enumerate(candidates):
            modalities, _ = architecture_settings(
                model_config["architecture"], all_modalities
            )
            seed = int(training["search_seed"])
            seed_everything(seed)
            model = make_model(
                model_config,
                {key: value.shape[1] for key, value in support_z.items()},
                class_counts, all_modalities,
            )
            result = train_model(
                model, subset_modalities(combined, modalities), combined_labels,
                fit_positions, validation_positions,
                learning_rate=float(model_config["learning_rate"]),
                weight_decay=float(model_config["weight_decay"]),
                batch_size=int(training["batch_size"]),
                max_epochs=3 if args.quick else int(training["max_epochs"]),
                patience=2 if args.quick else int(training["patience"]),
                seed=seed, device=device,
            )
            search_rows.append({
                "window": window, "config_id": config_id, **model_config,
                "validation_macro_f1_mean": result["best_macro_f1_mean"],
                "validation_joint_accuracy": result["best_joint_accuracy"],
                "best_epoch": result["best_epoch"],
                "elapsed_seconds": result["elapsed_seconds"],
            })
            search_histories[
                f"window{window}_{model_config['architecture']}_config{config_id}"
            ] = result["history"]
    search = pd.DataFrame(search_rows).sort_values(
        ["validation_macro_f1_mean", "validation_joint_accuracy"], ascending=False
    )
    search.to_csv(run_dir / "model_search.csv", index=False)
    for window in windows:
        for architecture in config["architectures"]:
            winner = search[
                search.window.eq(window) & search.architecture.eq(architecture)
            ].iloc[0]
            selected_configs[f"{architecture}|{window}"] = {
                key: winner[key].item() if hasattr(winner[key], "item") else winner[key]
                for key in (
                    "architecture", "embedding_dim", "head_hidden_dim", "dropout",
                    "learning_rate", "weight_decay",
                )
            }
            selected_configs[f"{architecture}|{window}"]["embedding_dim"] = int(
                selected_configs[f"{architecture}|{window}"]["embedding_dim"]
            )
            selected_configs[f"{architecture}|{window}"]["head_hidden_dim"] = int(
                selected_configs[f"{architecture}|{window}"]["head_hidden_dim"]
            )
    atomic_write_json(run_dir / "selected_configs.json", selected_configs)
    atomic_write_json(run_dir / "histories" / "model_search.json", search_histories)

    metric_parts, class_parts, sequence_parts, gate_rows = [], [], [], []
    final_seeds = ([int(training["search_seed"])] if args.quick else
                   [int(value) for value in training["final_seeds"]])
    for budget in budgets:
        for window in windows:
            support, _, support_labels, _ = cache[(budget, window)]["support_train"]
            validation, validation_index, validation_labels, validation_text = cache[
                (budget, window)
            ]["support_validation"]
            test, test_index, _, test_text = cache[(budget, window)]["official_test"]
            scaler, (support_z, validation_z, test_z) = standardize(
                support, validation, test
            )
            combined = {
                modality: np.concatenate([support_z[modality], validation_z[modality]])
                for modality in all_modalities
            }
            combined_labels = {
                key: np.concatenate([support_labels[key], validation_labels[key]])
                for key in ATTRIBUTES
            }
            fit_positions = np.arange(len(next(iter(support_z.values()))))
            validation_positions = np.arange(
                len(fit_positions), len(fit_positions) + len(next(iter(validation_z.values())))
            )
            input_dims = {key: value.shape[1] for key, value in support_z.items()}
            for architecture in config["architectures"]:
                model_config = selected_configs[f"{architecture}|{window}"]
                modalities, _ = architecture_settings(architecture, all_modalities)
                for seed in final_seeds:
                    seed_everything(seed)
                    model = make_model(
                        model_config, input_dims, class_counts, all_modalities
                    )
                    result = train_model(
                        model, subset_modalities(combined, modalities), combined_labels,
                        fit_positions, validation_positions,
                        learning_rate=float(model_config["learning_rate"]),
                        weight_decay=float(model_config["weight_decay"]),
                        batch_size=int(training["batch_size"]),
                        max_epochs=3 if args.quick else int(training["max_epochs"]),
                        patience=2 if args.quick else int(training["patience"]),
                        seed=seed, device=device,
                    )
                    for scope, features, current_index, truth in (
                        ("support_validation", validation_z, validation_index, validation_text),
                        ("official_test", test_z, test_index, test_text),
                    ):
                        predictions, confidences = model_predictions(
                            result["model"], subset_modalities(features, modalities),
                            names, int(training["batch_size"]), device,
                        )
                        frames = score_predictions(
                            current_index, truth, predictions, confidences,
                            method=architecture, budget=budget, window=window,
                            seed=seed, scope=scope,
                        )
                        metric_parts.append(frames[0]); class_parts.append(frames[1]); sequence_parts.append(frames[2])
                        if (
                            scope == "official_test" and budget == maximum_budget
                            and seed == final_seeds[0]
                        ):
                            frames[3].to_csv(
                                run_dir / "predictions" /
                                f"{architecture}_window{window}.csv", index=False
                            )
                    if architecture == "gated":
                        weights = result["model"].gate_weights()
                        for attribute in ATTRIBUTES:
                            for position, modality in enumerate(modalities):
                                gate_rows.append({
                                    "support_budget": budget, "window": window,
                                    "seed": seed, "attribute": attribute,
                                    "modality": modality,
                                    "weight": float(weights[attribute][position].detach().cpu()),
                                })
                    atomic_write_json(
                        run_dir / "histories" /
                        f"{architecture}_budget{budget}_window{window}_seed{seed}.json",
                        result["history"],
                    )
                    torch.save({
                        "model_state_dict": result["model"].cpu().state_dict(),
                        "model_config": model_config,
                        "modalities": modalities, "standardizer": scaler.state_dict(),
                        "label_names": names, "support_budget": budget,
                        "window": window, "seed": seed,
                        "best_epoch": result["best_epoch"],
                    }, run_dir / "checkpoints" /
                        f"{architecture}_budget{budget}_window{window}_seed{seed}.pt")

    metrics = pd.concat(metric_parts, ignore_index=True)
    classes = pd.concat(class_parts, ignore_index=True)
    sequences = pd.concat(sequence_parts, ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    classes.to_csv(run_dir / "per_class_metrics.csv", index=False)
    sequences.to_csv(run_dir / "per_sequence_metrics.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(run_dir / "gate_weights.csv", index=False)
    summary = metrics.groupby(
        ["method", "support_budget", "window", "scope", "attribute"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    validation = summary[
        summary.scope.eq("support_validation")
        & summary.support_budget.eq(maximum_budget)
        & summary.attribute.isin(ATTRIBUTES)
    ].groupby(["method", "window"], as_index=False).agg(
        validation_macro_f1_mean=("macro_f1_mean", "mean")
    )
    joint = summary[
        summary.scope.eq("support_validation")
        & summary.support_budget.eq(maximum_budget)
        & summary.attribute.eq("joint")
    ][["method", "window", "accuracy_mean"]].rename(
        columns={"accuracy_mean": "validation_joint_accuracy"}
    )
    validation = validation.merge(joint, on=["method", "window"]).sort_values(
        ["validation_macro_f1_mean", "validation_joint_accuracy"], ascending=False
    )
    validation.to_csv(run_dir / "validation_ranking.csv", index=False)
    selected = validation.iloc[0].to_dict()
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-multimodal-window/v3", "status": "complete",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "config_path": str(config_path), "config": config,
        "descriptor_bank": str(descriptor_bank), "device": str(device),
        "feature_dimensions": {
            modality: int(arrays["train"][modality].shape[1])
            for modality in all_modalities
        },
        "selected_on_validation": selected, "quick": bool(args.quick),
        "protocol": (
            "identical V2 blocked causal endpoints; feature aggregation before model; "
            "camera versus 3-modality concat versus per-attribute static softmax gates"
        ),
        "run_dir": str(run_dir),
    })
    official = summary[
        summary.scope.eq("official_test")
        & summary.support_budget.eq(maximum_budget)
    ]
    print("Validation ranking:")
    print(validation.to_string(index=False))
    print("\nOfficial-test summary at maximum support budget:")
    print(official.to_string(index=False))
    if gate_rows:
        print("\nMean learned gated-modality weights:")
        print(pd.DataFrame(gate_rows).groupby(
            ["window", "attribute", "modality"], as_index=False
        ).weight.mean().to_string(index=False))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
