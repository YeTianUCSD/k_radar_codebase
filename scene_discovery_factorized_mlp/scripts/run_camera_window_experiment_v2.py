#!/usr/bin/env python3
"""Fair Camera-only comparison after causal feature-window aggregation."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT / "scene_discovery"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from factorized_mlp.data import FeatureStandardizer, load_split_arrays  # noqa: E402
from factorized_mlp.model import FactorizedGatedMLP  # noqa: E402
from factorized_mlp.training import predict_logits, seed_everything, train_model  # noqa: E402
from factorized_mlp.window_protocol import (  # noqa: E402
    aggregate_causal_endpoints, build_blocked_window_manifest, endpoint_positions,
)
from scene_discovery.attributes import ATTRIBUTES, ATTRIBUTE_CLASSES, attribute_frame  # noqa: E402
from scene_discovery.common import atomic_write_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/camera_window_v2.yml",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def make_run_dir(root, quick):
    stamp = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
    prefix = "camera_window_v2_quick" if quick else "camera_window_v2"
    path = Path(root) / f"{prefix}_{stamp}"
    suffix = 1
    while path.exists():
        path = Path(root) / f"{prefix}_{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    for child in ("manifests", "histories", "checkpoints", "predictions"):
        (path / child).mkdir()
    return path


def class_maps():
    names = {key: tuple(ATTRIBUTE_CLASSES[key]) for key in ATTRIBUTES}
    mappings = {
        key: {name: position for position, name in enumerate(values)}
        for key, values in names.items()
    }
    return names, mappings


def labels_for_index(index, mappings):
    frame = attribute_frame(index.reset_index(drop=True))
    numeric = {
        key: np.asarray([mappings[key][value] for value in frame[key]], dtype=np.int64)
        for key in ATTRIBUTES
    }
    text = {key: frame[key].astype(str).to_numpy() for key in ATTRIBUTES}
    return numeric, text


def mlp_grid(config, quick):
    search = config["mlp_search"]
    rows = [
        {
            "embedding_dim": int(embedding), "head_hidden_dim": int(hidden),
            "dropout": float(dropout), "learning_rate": float(lr),
            "weight_decay": float(search["weight_decay"]),
        }
        for embedding, hidden, dropout, lr in product(
            search["embedding_dims"], search["head_hidden_dims"],
            search["dropouts"], search["learning_rates"],
        )
    ]
    return rows[:1] if quick else rows


def make_mlp(model_config, input_dim, class_counts):
    return FactorizedGatedMLP(
        {"camera": int(input_dim)}, class_counts, modalities=("camera",),
        embedding_dim=int(model_config["embedding_dim"]),
        head_hidden_dim=int(model_config["head_hidden_dim"]),
        dropout=float(model_config["dropout"]), fusion="gated",
    )


def standardize(train_x, *others):
    scaler = FeatureStandardizer.fit(
        {"camera": train_x}, np.ones(len(train_x), dtype=bool)
    )
    return scaler, tuple(scaler.transform("camera", values) for values in (train_x,) + others)


def make_classifier(name, config, seed, train_size):
    if name == "knn":
        neighbors = min(int(config["knn_neighbors"]), int(train_size))
        return KNeighborsClassifier(
            n_neighbors=neighbors, weights="distance", metric="euclidean"
        )
    if name == "logreg":
        return LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            multi_class="auto", random_state=int(seed),
        )
    raise ValueError(f"unsupported classical model {name!r}")


def score_predictions(
    index, truth, predictions, confidences, *, method, budget, window, seed, scope,
):
    metrics, classes, sequences = [], [], []
    joint = np.ones(len(index), dtype=bool)
    prediction_frame = index.reset_index(drop=True).copy()
    for attribute in ATTRIBUTES:
        actual = np.asarray(truth[attribute]).astype(str)
        predicted = np.asarray(predictions[attribute]).astype(str)
        joint &= actual == predicted
        prediction_frame[f"true_{attribute}"] = actual
        prediction_frame[f"predicted_{attribute}"] = predicted
        prediction_frame[f"{attribute}_confidence"] = confidences[attribute]
        base = {
            "method": method, "support_budget": int(budget),
            "window": int(window), "seed": int(seed), "scope": scope,
            "attribute": attribute, "samples": len(actual),
        }
        metrics.append({
            **base, "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(actual, predicted),
            "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
            "mean_confidence": float(np.mean(confidences[attribute])),
        })
        labels = list(ATTRIBUTE_CLASSES[attribute])
        precision, recall, f1, support = precision_recall_fscore_support(
            actual, predicted, labels=labels, zero_division=0,
        )
        for class_id, class_name in enumerate(labels):
            classes.append({
                **base, "class": class_name, "precision": precision[class_id],
                "recall": recall[class_id], "f1": f1[class_id],
                "support": int(support[class_id]),
            })
        sequence_values = index["sequence"].astype(str).to_numpy()
        for sequence in pd.unique(sequence_values):
            mask = sequence_values == sequence
            sequences.append({
                **base, "sequence": sequence, "samples": int(mask.sum()),
                "accuracy": accuracy_score(actual[mask], predicted[mask]),
            })
    prediction_frame["joint_correct"] = joint
    metrics.append({
        "method": method, "support_budget": int(budget), "window": int(window),
        "seed": int(seed), "scope": scope, "attribute": "joint",
        "samples": len(index), "accuracy": float(joint.mean()),
        "balanced_accuracy": float("nan"), "macro_f1": float("nan"),
        "mean_confidence": float("nan"),
    })
    return pd.DataFrame(metrics), pd.DataFrame(classes), pd.DataFrame(sequences), prediction_frame


def classical_predictions(name, config, seed, train_x, train_truth, evaluation_x):
    predictions, confidences = {}, {}
    for attribute in ATTRIBUTES:
        estimator = make_classifier(name, config, seed, len(train_x))
        estimator.fit(train_x, train_truth[attribute])
        probability = np.asarray(estimator.predict_proba(evaluation_x))
        classes = np.asarray(estimator.classes_).astype(str)
        position = probability.argmax(axis=1)
        predictions[attribute] = classes[position]
        confidences[attribute] = probability.max(axis=1)
    return predictions, confidences


def mlp_predictions(model, features, names, batch_size, device):
    logits = predict_logits(
        model, {"camera": features}, np.arange(len(features)), batch_size, device
    )
    predictions, confidences = {}, {}
    for attribute in ATTRIBUTES:
        values = logits[attribute]
        values = values - values.max(axis=1, keepdims=True)
        probability = np.exp(values) / np.exp(values).sum(axis=1, keepdims=True)
        position = probability.argmax(axis=1)
        predictions[attribute] = np.asarray(names[attribute])[position]
        confidences[attribute] = probability.max(axis=1)
    return predictions, confidences


def main():
    args = parse_args()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    with args.config.expanduser().resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    descriptor_bank = Path(config["descriptor_bank"]).expanduser().resolve()
    run_dir = make_run_dir(config["output_root"], args.quick)
    arrays, indices = {}, {}
    for split in ("train", "test"):
        loaded, index = load_split_arrays(
            descriptor_bank, split, ("camera",), config["descriptor"]
        )
        arrays[split] = loaded["camera"]
        indices[split] = index.reset_index(drop=True)
    windows = [int(max(config["windows"]))] if args.quick else [int(x) for x in config["windows"]]
    budgets = [int(max(config["support_budgets"]))] if args.quick else [int(x) for x in config["support_budgets"]]
    names, mappings = class_maps()
    class_counts = {key: len(value) for key, value in names.items()}
    manifests = {}
    for budget in budgets:
        manifest = build_blocked_window_manifest(
            indices["train"], indices["test"], budget, windows,
            validation_fraction=float(config["validation_fraction"]),
            gap_frames=int(config["gap_frames"]),
        )
        manifest.to_csv(run_dir / "manifests" / f"blocked_budget_{budget}.csv", index=False)
        manifests[budget] = manifest

    cache = {}
    for budget in budgets:
        manifest = manifests[budget]
        endpoints = {
            "support_train": endpoint_positions(manifest, "train", "support_train"),
            "support_validation": endpoint_positions(manifest, "train", "support_validation"),
            "official_test": endpoint_positions(manifest, "test", "official_test"),
        }
        for window in windows:
            cache[(budget, window)] = {}
            for scope, positions in endpoints.items():
                split = "test" if scope == "official_test" else "train"
                x, idx = aggregate_causal_endpoints(
                    arrays[split], indices[split], positions, window
                )
                numeric, text = labels_for_index(idx, mappings)
                cache[(budget, window)][scope] = (x, idx, numeric, text)

    training = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maximum_budget = max(budgets)
    search_rows, search_histories, selected_by_window = [], {}, {}
    for window in windows:
        train_x, _, train_numeric, _ = cache[(maximum_budget, window)]["support_train"]
        validation_x, _, validation_numeric, _ = cache[(maximum_budget, window)]["support_validation"]
        scaler, (train_z, validation_z) = standardize(train_x, validation_x)
        joined = np.concatenate([train_z, validation_z])
        joined_labels = {
            key: np.concatenate([train_numeric[key], validation_numeric[key]])
            for key in ATTRIBUTES
        }
        train_positions = np.arange(len(train_z))
        validation_positions = np.arange(len(train_z), len(joined))
        window_rows = []
        for config_id, model_config in enumerate(mlp_grid(config, args.quick)):
            seed = int(training["search_seed"])
            seed_everything(seed)
            model = make_mlp(model_config, train_z.shape[1], class_counts)
            result = train_model(
                model, {"camera": joined}, joined_labels,
                train_positions, validation_positions,
                learning_rate=model_config["learning_rate"],
                weight_decay=model_config["weight_decay"],
                batch_size=int(training["batch_size"]),
                max_epochs=3 if args.quick else int(training["max_epochs"]),
                patience=2 if args.quick else int(training["patience"]),
                seed=seed, device=device,
            )
            row = {
                "window": window, "config_id": config_id, **model_config,
                "validation_macro_f1_mean": result["best_macro_f1_mean"],
                "validation_joint_accuracy": result["best_joint_accuracy"],
                "best_epoch": result["best_epoch"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
            search_rows.append(row)
            window_rows.append(row)
            search_histories[f"window{window}_config{config_id}"] = result["history"]
        winner = sorted(
            window_rows,
            key=lambda row: (
                row["validation_macro_f1_mean"], row["validation_joint_accuracy"]
            ), reverse=True,
        )[0]
        selected_by_window[window] = {
            key: winner[key] for key in (
                "embedding_dim", "head_hidden_dim", "dropout",
                "learning_rate", "weight_decay",
            )
        }
    pd.DataFrame(search_rows).sort_values(
        ["validation_macro_f1_mean", "validation_joint_accuracy"], ascending=False
    ).to_csv(run_dir / "mlp_search.csv", index=False)
    atomic_write_json(run_dir / "histories" / "mlp_search.json", search_histories)
    atomic_write_json(run_dir / "selected_mlp_by_window.json", selected_by_window)

    metrics_parts, class_parts, sequence_parts, validation_candidates = [], [], [], []
    final_seeds = [int(training["search_seed"])] if args.quick else [int(x) for x in training["final_seeds"]]
    for budget in budgets:
        for window in windows:
            data = cache[(budget, window)]
            train_x, train_idx, train_numeric, train_text = data["support_train"]
            validation_x, validation_idx, validation_numeric, validation_text = data["support_validation"]
            test_x, test_idx, test_numeric, test_text = data["official_test"]
            scaler, (train_z, validation_z, test_z) = standardize(
                train_x, validation_x, test_x
            )
            for method in config["classical_models"]:
                seed = int(training["search_seed"])
                for scope, evaluation_x, evaluation_idx, truth in (
                    ("support_validation", validation_z, validation_idx, validation_text),
                    ("official_test", test_z, test_idx, test_text),
                ):
                    predictions, confidences = classical_predictions(
                        method, config, seed, train_z, train_text, evaluation_x
                    )
                    frames = score_predictions(
                        evaluation_idx, truth, predictions, confidences,
                        method=method, budget=budget, window=window,
                        seed=seed, scope=scope,
                    )
                    metrics_parts.append(frames[0]); class_parts.append(frames[1]); sequence_parts.append(frames[2])
                    if scope == "official_test" and budget == maximum_budget:
                        frames[3].to_csv(
                            run_dir / "predictions" / f"{method}_window{window}.csv", index=False
                        )
            model_config = selected_by_window[window]
            for seed in final_seeds:
                joined = np.concatenate([train_z, validation_z])
                joined_labels = {
                    key: np.concatenate([train_numeric[key], validation_numeric[key]])
                    for key in ATTRIBUTES
                }
                train_positions = np.arange(len(train_z))
                validation_positions = np.arange(len(train_z), len(joined))
                seed_everything(seed)
                model = make_mlp(model_config, train_z.shape[1], class_counts)
                result = train_model(
                    model, {"camera": joined}, joined_labels,
                    train_positions, validation_positions,
                    learning_rate=model_config["learning_rate"],
                    weight_decay=model_config["weight_decay"],
                    batch_size=int(training["batch_size"]),
                    max_epochs=3 if args.quick else int(training["max_epochs"]),
                    patience=2 if args.quick else int(training["patience"]),
                    seed=seed, device=device,
                )
                for scope, evaluation_x, evaluation_idx, truth in (
                    ("support_validation", validation_z, validation_idx, validation_text),
                    ("official_test", test_z, test_idx, test_text),
                ):
                    predictions, confidences = mlp_predictions(
                        result["model"], evaluation_x, names,
                        int(training["batch_size"]), device,
                    )
                    frames = score_predictions(
                        evaluation_idx, truth, predictions, confidences,
                        method="mlp", budget=budget, window=window,
                        seed=seed, scope=scope,
                    )
                    metrics_parts.append(frames[0]); class_parts.append(frames[1]); sequence_parts.append(frames[2])
                    if (
                        scope == "official_test" and budget == maximum_budget
                        and seed == final_seeds[0]
                    ):
                        frames[3].to_csv(
                            run_dir / "predictions" / f"mlp_window{window}.csv", index=False
                        )
                atomic_write_json(
                    run_dir / "histories" / f"mlp_budget{budget}_window{window}_seed{seed}.json",
                    result["history"],
                )
                torch.save({
                    "model_state_dict": result["model"].cpu().state_dict(),
                    "model_config": model_config, "standardizer": scaler.state_dict(),
                    "label_names": names, "support_budget": budget,
                    "window": window, "seed": seed, "best_epoch": result["best_epoch"],
                }, run_dir / "checkpoints" / f"mlp_budget{budget}_window{window}_seed{seed}.pt")

    metrics = pd.concat(metrics_parts, ignore_index=True)
    classes = pd.concat(class_parts, ignore_index=True)
    sequences = pd.concat(sequence_parts, ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    classes.to_csv(run_dir / "per_class_metrics.csv", index=False)
    sequences.to_csv(run_dir / "per_sequence_metrics.csv", index=False)
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
    atomic_write_json(run_dir / "selected_method.json", selected)
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-camera-window/v2",
        "status": "complete", "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "config_path": str(args.config.expanduser().resolve()), "config": config,
        "descriptor_bank": str(descriptor_bank), "device": str(device),
        "feature_dimension": int(arrays["train"].shape[1]),
        "train_frames": int(len(indices["train"])),
        "test_frames": int(len(indices["test"])),
        "windows": windows, "support_budgets": budgets,
        "selected_on_validation": selected, "quick": bool(args.quick),
        "protocol": (
            "camera-only; fixed causal endpoints; aggregate features before classification; "
            "support from early block; max-window gap; validation from final train block; "
            "official test evaluation-only"
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
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
