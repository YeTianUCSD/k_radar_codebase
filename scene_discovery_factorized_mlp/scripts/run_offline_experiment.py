#!/usr/bin/env python3
"""Search, train, calibrate, and evaluate few-shot factorized MLP heads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
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
for path in (PROJECT_ROOT, REPOSITORY_ROOT / "scene_discovery"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from factorized_mlp.data import FeatureStandardizer, load_split_arrays  # noqa: E402
from factorized_mlp.evaluation import (  # noqa: E402
    confusion_tables, evaluate_predictions, softmax,
)
from factorized_mlp.fewshot_protocol import build_fewshot_manifest  # noqa: E402
from factorized_mlp.model import FactorizedGatedMLP  # noqa: E402
from factorized_mlp.training import (  # noqa: E402
    fit_temperatures, predict_logits, seed_everything, train_model,
)
from scene_discovery.attributes import ATTRIBUTE_CLASSES, attribute_frame  # noqa: E402
from scene_discovery.common import atomic_write_json, file_sha256  # noqa: E402


ATTRIBUTES = ("weather", "road", "lighting")
ALL_MODALITIES = ("camera", "lidar", "radar")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/offline_v1.yml",
    )
    parser.add_argument("--quick", action="store_true", help="One tiny configuration for smoke tests")
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def make_run_dir(root, quick=False):
    stamp = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
    prefix = "offline_factorized_mlp_quick" if quick else "offline_factorized_mlp"
    path = Path(root) / f"{prefix}_{stamp}"
    suffix = 1
    while path.exists():
        path = Path(root) / f"{prefix}_{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    for name in ("manifests", "checkpoints", "predictions", "confusions", "histories"):
        (path / name).mkdir()
    return path


def label_arrays(index):
    attributes = attribute_frame(index.reset_index(drop=True))
    names = {key: tuple(ATTRIBUTE_CLASSES[key]) for key in ATTRIBUTES}
    mappings = {
        key: {name: position for position, name in enumerate(names[key])}
        for key in ATTRIBUTES
    }
    labels = {
        key: np.asarray([mappings[key][value] for value in attributes[key]], dtype=np.int64)
        for key in ATTRIBUTES
    }
    return labels, names


def roles_by_position(manifest, split, length):
    values = np.full(length, "", dtype=object)
    rows = manifest[manifest.source_split.eq(split)]
    values[rows.split_position.astype(int).to_numpy()] = rows.role.astype(str).to_numpy()
    if np.any(values == ""):
        raise ValueError(f"manifest does not cover every {split} row")
    return values


def standardized_arrays(arrays, standardizer, modalities):
    return {
        modality: standardizer.transform(modality, arrays[modality])
        for modality in modalities
    }


def architecture_settings(name):
    if name == "camera":
        return ("camera",), "gated"
    if name == "concat":
        return ALL_MODALITIES, "concat"
    if name == "gated":
        return ALL_MODALITIES, "gated"
    raise ValueError(f"unknown architecture {name!r}")


def build_model(config, input_dims, class_counts):
    modalities, fusion = architecture_settings(config["architecture"])
    return FactorizedGatedMLP(
        input_dims=input_dims, class_counts=class_counts, modalities=modalities,
        embedding_dim=config["embedding_dim"],
        head_hidden_dim=config["head_hidden_dim"], dropout=config["dropout"],
        fusion=fusion,
    )


def probabilities_from_logits(logits, temperatures):
    return {
        attribute: softmax(logits[attribute] / temperatures[attribute])
        for attribute in ATTRIBUTES
    }


def config_grid(config, quick):
    search = config["model_search"]
    rows = []
    for architecture, embedding, hidden, dropout, learning_rate in product(
        search["architectures"], search["embedding_dims"],
        search["head_hidden_dims"], search["dropouts"],
        search["learning_rates"],
    ):
        rows.append({
            "architecture": architecture, "embedding_dim": int(embedding),
            "head_hidden_dim": int(hidden), "dropout": float(dropout),
            "learning_rate": float(learning_rate),
            "weight_decay": float(search["weight_decay"]),
        })
    return rows[:1] if quick else rows


def main():
    args = parse_args()
    started_utc = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    config = load_config(args.config.expanduser().resolve())
    descriptor_bank = Path(config["descriptor_bank"]).expanduser().resolve()
    run_dir = make_run_dir(
        Path(config["output_root"]).expanduser().resolve(), quick=args.quick
    )
    descriptor = config["descriptor"]
    raw, indices = {}, {}
    for split in ("train", "test"):
        raw[split], indices[split] = load_split_arrays(
            descriptor_bank, split, ALL_MODALITIES, descriptor
        )
        indices[split] = indices[split].reset_index(drop=True)
    labels = {}
    label_names = None
    for split in ("train", "test"):
        labels[split], names = label_arrays(indices[split])
        label_names = names if label_names is None else label_names

    support = config["support"]
    budgets = [max(support["budgets"])] if args.quick else list(support["budgets"])
    manifests = {}
    for budget in budgets:
        manifest = build_fewshot_manifest(
            indices["train"], indices["test"], int(budget),
            int(support["support_pool"]), int(support["validation_frames"]),
            int(support["guard_frames"]),
        )
        manifest.to_csv(run_dir / "manifests" / f"fewshot_budget_{budget}.csv", index=False)
        manifests[int(budget)] = manifest

    maximum_budget = max(budgets)
    search_manifest = manifests[maximum_budget]
    search_roles = roles_by_position(search_manifest, "train", len(indices["train"]))
    train_positions = np.flatnonzero(search_roles == "support_train")
    validation_positions = np.flatnonzero(search_roles == "support_validation")
    standardizer = FeatureStandardizer.fit(raw["train"], search_roles == "support_train")
    normalized_train = standardized_arrays(raw["train"], standardizer, ALL_MODALITIES)
    input_dims = {key: value.shape[1] for key, value in normalized_train.items()}
    class_counts = {key: len(value) for key, value in label_names.items()}
    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = list(config["evaluation"]["windows"])
    search_rows, search_histories = [], {}
    for config_id, model_config in enumerate(config_grid(config, args.quick)):
        seed_everything(int(training["search_seed"]))
        model = build_model(model_config, input_dims, class_counts)
        result = train_model(
            model, normalized_train, labels["train"], train_positions,
            validation_positions,
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"],
            batch_size=int(training["batch_size"]),
            max_epochs=3 if args.quick else int(training["max_epochs"]),
            patience=2 if args.quick else int(training["patience"]),
            seed=int(training["search_seed"]), device=device,
        )
        model = result["model"]
        validation_logits = predict_logits(
            model, normalized_train, validation_positions,
            int(training["batch_size"]), device,
        )
        probabilities = {key: softmax(value) for key, value in validation_logits.items()}
        validation_index = indices["train"].iloc[validation_positions].reset_index(drop=True)
        validation_labels = {
            key: value[validation_positions] for key, value in labels["train"].items()
        }
        for window in windows:
            metrics, _, _, _ = evaluate_predictions(
                validation_index, validation_labels, probabilities, label_names,
                "support_validation", int(window),
            )
            attribute_metrics = metrics[metrics.attribute.isin(ATTRIBUTES)]
            joint = float(metrics.loc[metrics.attribute.eq("joint"), "accuracy"].iloc[0])
            search_rows.append({
                "config_id": config_id, **model_config, "window": int(window),
                "best_epoch": result["best_epoch"],
                "early_stop_macro_f1_mean": result["best_macro_f1_mean"],
                "selection_macro_f1_mean": float(attribute_metrics.macro_f1.mean()),
                "selection_joint_accuracy": joint,
                "elapsed_seconds": result["elapsed_seconds"],
            })
        search_histories[str(config_id)] = result["history"]
    leaderboard = pd.DataFrame(search_rows).sort_values(
        ["selection_macro_f1_mean", "selection_joint_accuracy"], ascending=False
    )
    leaderboard.to_csv(run_dir / "search_leaderboard.csv", index=False)
    atomic_write_json(run_dir / "histories" / "search_histories.json", search_histories)
    selected = leaderboard.iloc[0].to_dict()
    selected_config = {
        key: selected[key] for key in (
            "architecture", "embedding_dim", "head_hidden_dim", "dropout",
            "learning_rate", "weight_decay", "window",
        )
    }
    selected_config["embedding_dim"] = int(selected_config["embedding_dim"])
    selected_config["head_hidden_dim"] = int(selected_config["head_hidden_dim"])
    selected_config["window"] = int(selected_config["window"])
    atomic_write_json(run_dir / "selected_config.json", selected_config)

    final_metrics, class_metrics, sequence_metrics = [], [], []
    gate_rows, learning_rows = [], []
    final_seeds = [int(training["search_seed"])] if args.quick else list(training["final_seeds"])
    for budget in budgets:
        manifest = manifests[int(budget)]
        roles = roles_by_position(manifest, "train", len(indices["train"]))
        fit_positions = np.flatnonzero(roles == "support_train")
        validation_positions = np.flatnonzero(roles == "support_validation")
        standardizer = FeatureStandardizer.fit(raw["train"], roles == "support_train")
        normalized = {
            "train": standardized_arrays(raw["train"], standardizer, ALL_MODALITIES),
            "test": standardized_arrays(raw["test"], standardizer, ALL_MODALITIES),
        }
        for seed in final_seeds:
            seed_everything(int(seed))
            model = build_model(selected_config, input_dims, class_counts)
            result = train_model(
                model, normalized["train"], labels["train"], fit_positions,
                validation_positions,
                learning_rate=float(selected_config["learning_rate"]),
                weight_decay=float(selected_config["weight_decay"]),
                batch_size=int(training["batch_size"]),
                max_epochs=3 if args.quick else int(training["max_epochs"]),
                patience=2 if args.quick else int(training["patience"]),
                seed=int(seed), device=device,
            )
            model = result["model"]
            validation_logits = predict_logits(
                model, normalized["train"], validation_positions,
                int(training["batch_size"]), device,
            )
            temperatures = fit_temperatures(
                validation_logits, labels["train"], validation_positions
            )
            all_logits = {
                split: predict_logits(
                    model, normalized[split], np.arange(len(indices[split])),
                    int(training["batch_size"]), device,
                )
                for split in ("train", "test")
            }
            all_probabilities = {
                split: probabilities_from_logits(all_logits[split], temperatures)
                for split in ("train", "test")
            }
            scopes = {
                "support_train": ("train", fit_positions),
                "support_validation": ("train", validation_positions),
                "train_remainder": ("train", np.flatnonzero(roles == "train_remainder")),
                "train_all": ("train", np.arange(len(indices["train"]))),
                "official_test": ("test", np.arange(len(indices["test"]))),
            }
            for scope, (split, positions) in scopes.items():
                scope_index = indices[split].iloc[positions].reset_index(drop=True)
                scope_labels = {key: value[positions] for key, value in labels[split].items()}
                scope_probabilities = {
                    key: value[positions] for key, value in all_probabilities[split].items()
                }
                for window in windows:
                    metrics, classes, sequences, predictions = evaluate_predictions(
                        scope_index, scope_labels, scope_probabilities, label_names,
                        scope, int(window),
                    )
                    for frame in (metrics, classes, sequences):
                        frame.insert(0, "seed", int(seed))
                        frame.insert(0, "support_budget", int(budget))
                        frame.insert(0, "architecture", selected_config["architecture"])
                    final_metrics.append(metrics)
                    class_metrics.append(classes)
                    sequence_metrics.append(sequences)
                    if (
                        int(budget) == maximum_budget
                        and int(seed) == int(final_seeds[0])
                        and int(window) == int(selected_config["window"])
                    ):
                        predictions.to_csv(
                            run_dir / "predictions" / f"{scope}.csv", index=False
                        )
                        if scope == "official_test":
                            for attribute, table in confusion_tables(
                                scope_index, scope_labels, scope_probabilities,
                                label_names, int(window),
                            ).items():
                                table.to_csv(run_dir / "confusions" / f"{attribute}.csv")
            gates = model.gate_weights()
            for attribute in ATTRIBUTES:
                for position, modality in enumerate(model.modalities):
                    gate_rows.append({
                        "support_budget": int(budget), "seed": int(seed),
                        "attribute": attribute, "modality": modality,
                        "weight": float(gates[attribute][position].detach().cpu()),
                    })
            checkpoint = {
                "model_state_dict": model.cpu().state_dict(),
                "model_config": selected_config,
                "input_dims": input_dims, "class_counts": class_counts,
                "label_names": label_names, "temperatures": temperatures,
                "standardizer": standardizer.state_dict(),
                "support_budget": int(budget), "seed": int(seed),
                "best_epoch": result["best_epoch"],
            }
            torch.save(
                checkpoint,
                run_dir / "checkpoints" / f"model_budget{budget}_seed{seed}.pt",
            )
            atomic_write_json(
                run_dir / "histories" / f"budget{budget}_seed{seed}.json",
                result["history"],
            )

    metrics_frame = pd.concat(final_metrics, ignore_index=True)
    class_frame = pd.concat(class_metrics, ignore_index=True)
    sequence_frame = pd.concat(sequence_metrics, ignore_index=True)
    metrics_frame.to_csv(run_dir / "offline_metrics.csv", index=False)
    class_frame.to_csv(run_dir / "per_class_metrics.csv", index=False)
    sequence_frame.to_csv(run_dir / "per_sequence_metrics.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(run_dir / "gate_weights.csv", index=False)
    selected_window = int(selected_config["window"])
    learning = metrics_frame[
        metrics_frame.scope.isin(["train_remainder", "official_test"])
        & metrics_frame.window.eq(selected_window)
    ].groupby(
        ["architecture", "support_budget", "scope", "attribute", "window"],
        as_index=False,
    ).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
    )
    learning.to_csv(run_dir / "learning_curve.csv", index=False)
    metadata = {
        "schema_version": "kradar-factorized-mlp/offline-v1",
        "status": "complete",
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "config_path": str(args.config.expanduser().resolve()), "config": config,
        "descriptor_bank": str(descriptor_bank),
        "descriptor_manifest_sha256": file_sha256(descriptor_bank / "manifest.json")
            if (descriptor_bank / "manifest.json").exists() else None,
        "feature_dimensions": input_dims,
        "feature_dtypes": {
            split: {key: str(value.dtype) for key, value in raw[split].items()}
            for split in ("train", "test")
        },
        "sample_counts": {
            split: {
                "total": int(len(indices[split])),
                "per_sequence": {
                    str(key): int(value) for key, value in
                    indices[split].groupby("sequence").size().items()
                },
            }
            for split in ("train", "test")
        },
        "label_names": label_names,
        "selected_config": selected_config, "device": str(device),
        "selected_model_parameters": int(sum(
            parameter.numel() for parameter in model.parameters()
        )),
        "search_configurations": int(len(config_grid(config, args.quick))),
        "trained_final_models": int(len(budgets) * len(final_seeds)),
        "torch_version": torch.__version__, "quick": bool(args.quick),
        "selection_policy": (
            "hyperparameters and smoothing window selected on support validation only; "
            "official test is evaluation-only"
        ),
        "run_dir": str(run_dir),
    }
    atomic_write_json(run_dir / "run_meta.json", metadata)
    print("Selected configuration:")
    print(json.dumps(selected_config, indent=2))
    print("\nLearning curve summary:")
    print(learning.to_string(index=False))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
