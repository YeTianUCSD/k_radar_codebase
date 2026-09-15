#!/usr/bin/env python3
"""Evaluate weather, road, and lighting independently across unseen sequences."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.attribute_protocol import (
    align_attribute_predictions,
    build_attribute_support_matrix,
    eligible_sequences,
    per_sequence_accuracy,
    sequence_balanced_confusion,
    sequence_balanced_metrics,
    sequence_balanced_per_class_metrics,
)
from scene_discovery.attributes import (
    ATTRIBUTES,
    ATTRIBUTE_CLASSES,
    attribute_frame,
    expected_calibration_error,
    fit_probabilistic_classifier,
    select_confidence_threshold,
    sequence_attribute_table,
)
from scene_discovery.common import atomic_write_json, make_run_dir, seed_everything, utc_now
from scene_discovery.feature_bank import DEFAULT_SEQUENCES, MODALITIES, DescriptorBank
from scene_discovery.preprocessing import (
    PerModalityProjector,
    ProjectionSettings,
    modality_combinations,
)
from scene_discovery.temporal import aggregate_windows
from scene_discovery.visualization import save_confusion_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor-bank",
        type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/descriptors/v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/experiments",
    )
    parser.add_argument("--descriptor", choices=("mean", "mean_std", "spatial"), default="mean_std")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("logreg", "linear_svm", "knn"),
        default=["logreg", "linear_svm", "knn"],
    )
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 10, 20, 30])
    parser.add_argument("--search-stride", type=int, default=5)
    parser.add_argument("--svm-calibration-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-support-scenes", type=int, default=1)
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.9)
    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],
    )
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def normalize_sequence(value: object) -> str:
    value = str(value).strip().lower()
    if value.startswith("seq"):
        value = value[3:]
    return str(int(value))


def window_data(
    projected: Dict[str, np.ndarray],
    index: pd.DataFrame,
    mask: np.ndarray,
    modalities: Tuple[str, ...],
    window: int,
    stride: int,
):
    features = np.concatenate(
        [np.asarray(projected[modality][mask]) for modality in modalities], axis=1
    )
    subset_index = index.loc[mask].reset_index(drop=True)
    return aggregate_windows(features, subset_index, window, stride)


def predict_estimator(estimator, features):
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float32)
    classes = np.asarray(estimator.classes_).astype(str)
    positions = probabilities.argmax(axis=1)
    return classes[positions], probabilities.max(axis=1), probabilities, classes


def inner_supported_sequences(
    scene_table: pd.DataFrame,
    available_sequences,
    attribute: str,
    minimum_support_scenes: int,
):
    subset = scene_table[
        scene_table["sequence"].astype(str).isin(set(available_sequences))
    ].reset_index(drop=True)
    support = build_attribute_support_matrix(subset, minimum_support_scenes)
    return eligible_sequences(support, attribute)


def search_attribute_config(
    attribute: str,
    outer_sequence: str,
    train_index: pd.DataFrame,
    train_arrays: Dict[str, np.ndarray],
    scene_table: pd.DataFrame,
    combinations,
    args: argparse.Namespace,
):
    sequence_values = train_index["sequence"].astype(str).to_numpy()
    outer_train_sequences = [
        sequence
        for sequence in train_index["sequence"].astype(str).drop_duplicates()
        if sequence != outer_sequence
    ]
    validation_sequences = inner_supported_sequences(
        scene_table,
        outer_train_sequences,
        attribute,
        args.minimum_support_scenes,
    )
    if not validation_sequences:
        raise ValueError(
            f"No supported inner sequence folds for {attribute}, outer Seq{outer_sequence}"
        )

    cache = defaultdict(lambda: {"truth": [], "prediction": [], "confidence": [], "sequence": []})
    for inner_number, validation_sequence in enumerate(validation_sequences):
        fit_mask = (
            (sequence_values != outer_sequence)
            & (sequence_values != validation_sequence)
        )
        validation_mask = sequence_values == validation_sequence
        projector = PerModalityProjector(
            args.modalities,
            ProjectionSettings(
                pca_components=args.pca_components,
                seed=args.seed + inner_number,
            ),
        ).fit({
            modality: np.asarray(values[fit_mask])
            for modality, values in train_arrays.items()
        })
        projected = projector.transform_modalities(train_arrays)
        for window in args.windows:
            stride = 1 if window == 1 else args.search_stride
            for combination in combinations:
                fit_x, fit_idx = window_data(
                    projected,
                    train_index,
                    fit_mask,
                    combination,
                    window,
                    stride,
                )
                validation_x, validation_idx = window_data(
                    projected,
                    train_index,
                    validation_mask,
                    combination,
                    window,
                    stride,
                )
                fit_y = attribute_frame(fit_idx)[attribute].to_numpy()
                validation_y = attribute_frame(validation_idx)[attribute].to_numpy()
                for model_name in args.models:
                    estimator = fit_probabilistic_classifier(
                        model_name,
                        args.seed + inner_number,
                        fit_x,
                        fit_y,
                        fit_idx,
                        args.svm_calibration_fraction,
                    )
                    prediction, confidence, _, _ = predict_estimator(
                        estimator, validation_x
                    )
                    key = (combination, window, model_name)
                    cache[key]["truth"].append(validation_y)
                    cache[key]["prediction"].append(prediction)
                    cache[key]["confidence"].append(confidence)
                    cache[key]["sequence"].append(
                        np.repeat(validation_sequence, len(prediction))
                    )

    rows = []
    packed = {}
    for key, values in cache.items():
        combination, window, model_name = key
        current = {
            name: np.concatenate(parts)
            for name, parts in values.items()
        }
        metrics = sequence_balanced_metrics(
            current["truth"], current["prediction"], current["sequence"]
        )
        rows.append({
            "attribute": attribute,
            "outer_sequence": outer_sequence,
            "modalities": "+".join(combination),
            "window": window,
            "model": model_name,
            "inner_folds": len(pd.unique(current["sequence"])),
            "ece": expected_calibration_error(
                current["truth"], current["prediction"], current["confidence"]
            ),
            **metrics,
        })
        packed[key] = current
    leaderboard = pd.DataFrame(rows).sort_values(
        [
            "sequence_macro_f1",
            "sequence_balanced_accuracy",
            "sequence_accuracy",
            "ece",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    winner = leaderboard.iloc[0]
    winner_key = (
        tuple(str(winner["modalities"]).split("+")),
        int(winner["window"]),
        str(winner["model"]),
    )
    selected_data = packed[winner_key]
    threshold, operating_points = select_confidence_threshold(
        selected_data["truth"],
        selected_data["prediction"],
        selected_data["confidence"],
        args.confidence_thresholds,
        args.target_accepted_accuracy,
    )
    return leaderboard, winner_key, threshold, operating_points


def evaluate_outer_fold(
    attribute: str,
    outer_sequence: str,
    winner_key,
    threshold: float,
    train_index: pd.DataFrame,
    test_index: pd.DataFrame,
    train_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    args: argparse.Namespace,
):
    combination, window, model_name = winner_key
    train_sequences = train_index["sequence"].astype(str).to_numpy()
    test_sequences = test_index["sequence"].astype(str).to_numpy()
    fit_mask = train_sequences != outer_sequence
    test_mask = test_sequences == outer_sequence

    projector = PerModalityProjector(
        args.modalities,
        ProjectionSettings(pca_components=args.pca_components, seed=args.seed),
    ).fit({
        modality: np.asarray(values[fit_mask])
        for modality, values in train_arrays.items()
    })
    projected_train = projector.transform_modalities(train_arrays)
    projected_test = projector.transform_modalities(test_arrays)
    train_stride = 1 if window == 1 else args.search_stride
    train_x, train_idx = window_data(
        projected_train,
        train_index,
        fit_mask,
        combination,
        window,
        train_stride,
    )
    test_x, current_index = window_data(
        projected_test,
        test_index,
        test_mask,
        combination,
        window,
        1,
    )
    train_y = attribute_frame(train_idx)[attribute].to_numpy()
    truth = attribute_frame(current_index)[attribute].to_numpy()
    estimator = fit_probabilistic_classifier(
        model_name,
        args.seed,
        train_x,
        train_y,
        train_idx,
        args.svm_calibration_fraction,
    )
    prediction, confidence, probabilities, classes = predict_estimator(
        estimator, test_x
    )
    output = current_index.copy().reset_index(drop=True)
    output["held_out_sequence"] = outer_sequence
    output[f"true_{attribute}"] = truth
    output[f"predicted_{attribute}"] = prediction
    output[f"{attribute}_confidence"] = confidence
    output[f"{attribute}_accepted"] = confidence >= threshold
    class_to_position = {label: position for position, label in enumerate(classes)}
    for label in ATTRIBUTE_CLASSES[attribute]:
        position = class_to_position.get(label)
        output[f"{attribute}_prob_{label}"] = (
            probabilities[:, position] if position is not None else 0.0
        )
    output["selected_modalities"] = "+".join(combination)
    output["selected_model"] = model_name
    output["selected_window"] = window
    output["selected_threshold"] = threshold
    return output


def summarize_attribute(attribute: str, predictions: pd.DataFrame):
    truth = predictions[f"true_{attribute}"].to_numpy()
    predicted = predictions[f"predicted_{attribute}"].to_numpy()
    sequences = predictions["held_out_sequence"].astype(str).to_numpy()
    summary = sequence_balanced_metrics(truth, predicted, sequences)
    summary.update({
        "attribute": attribute,
        "ece": expected_calibration_error(
            truth, predicted, predictions[f"{attribute}_confidence"]
        ),
        "accepted_rate": float(predictions[f"{attribute}_accepted"].mean()),
        "accepted_accuracy": float(
            predictions.loc[
                predictions[f"{attribute}_accepted"],
                f"true_{attribute}",
            ].eq(
                predictions.loc[
                    predictions[f"{attribute}_accepted"],
                    f"predicted_{attribute}",
                ]
            ).mean()
        ) if predictions[f"{attribute}_accepted"].any() else 0.0,
    })
    return summary


def main() -> None:
    args = parse_args()
    if args.minimum_support_scenes < 1:
        raise ValueError("--minimum-support-scenes must be positive")
    if args.search_stride < 1 or any(window < 1 for window in args.windows):
        raise ValueError("windows and search stride must be positive")
    seed_everything(args.seed)
    sequences = [normalize_sequence(value) for value in args.sequences]
    if len(sequences) != len(set(sequences)):
        raise ValueError("--sequences contains duplicates")

    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(
        args.output_root.expanduser().resolve(), "independent_attributes"
    )
    train_index = bank.index("train")
    test_index = bank.index("test")
    train_index["sequence"] = train_index["sequence"].astype(str)
    test_index["sequence"] = test_index["sequence"].astype(str)
    train_index = train_index[train_index["sequence"].isin(sequences)].reset_index(drop=True)
    test_index = test_index[test_index["sequence"].isin(sequences)].reset_index(drop=True)
    if set(sequences) != set(train_index["sequence"]) or set(sequences) != set(test_index["sequence"]):
        raise ValueError("Requested sequences must exist in both descriptor splits")

    train_arrays = {
        modality: np.asarray(bank.array("train", modality, args.descriptor))[
            bank.index("train")["sequence"].astype(str).isin(sequences).to_numpy()
        ]
        for modality in args.modalities
    }
    test_arrays = {
        modality: np.asarray(bank.array("test", modality, args.descriptor))[
            bank.index("test")["sequence"].astype(str).isin(sequences).to_numpy()
        ]
        for modality in args.modalities
    }
    scene_table = sequence_attribute_table(
        pd.concat([train_index, test_index], ignore_index=True)
    )
    scene_table["sequence"] = scene_table["sequence"].astype(str)
    support = build_attribute_support_matrix(
        scene_table, args.minimum_support_scenes
    )
    support.to_csv(run_dir / "support_matrix.csv", index=False)
    combinations = modality_combinations(args.modalities)

    prediction_frames = {}
    leaderboard_frames = []
    selected_rows = []
    operating_rows = []
    fold_metric_rows = []
    per_sequence_rows = []

    for attribute in ATTRIBUTES:
        attribute_predictions = []
        for outer_sequence in eligible_sequences(support, attribute):
            leaderboard, winner_key, threshold, operating_points = search_attribute_config(
                attribute,
                outer_sequence,
                train_index,
                train_arrays,
                scene_table,
                combinations,
                args,
            )
            leaderboard_frames.append(leaderboard)
            combination, window, model_name = winner_key
            selected_rows.append({
                "attribute": attribute,
                "outer_sequence": outer_sequence,
                "modalities": "+".join(combination),
                "window": window,
                "model": model_name,
                "confidence_threshold": threshold,
            })
            for row in operating_points:
                operating_rows.append({
                    "attribute": attribute,
                    "outer_sequence": outer_sequence,
                    **row,
                })
            output = evaluate_outer_fold(
                attribute,
                outer_sequence,
                winner_key,
                threshold,
                train_index,
                test_index,
                train_arrays,
                test_arrays,
                args,
            )
            attribute_predictions.append(output)
            metrics = summarize_attribute(attribute, output)
            metrics["outer_sequence"] = outer_sequence
            fold_metric_rows.append(metrics)
        predictions = pd.concat(attribute_predictions, ignore_index=True)
        prediction_frames[attribute] = predictions
        attribute_dir = run_dir / attribute
        attribute_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(attribute_dir / "predictions.csv", index=False)
        sequence_metrics = per_sequence_accuracy(
            predictions[f"true_{attribute}"],
            predictions[f"predicted_{attribute}"],
            predictions["held_out_sequence"],
        )
        sequence_metrics.to_csv(attribute_dir / "per_sequence_metrics.csv", index=False)
        per_sequence_rows.extend([
            {"attribute": attribute, **row}
            for row in sequence_metrics.to_dict("records")
        ])
        class_metrics = sequence_balanced_per_class_metrics(
            predictions[f"true_{attribute}"],
            predictions[f"predicted_{attribute}"],
            predictions["held_out_sequence"],
        )
        class_metrics.to_csv(attribute_dir / "per_class_metrics.csv", index=False)
        frame_confusion = pd.crosstab(
            predictions[f"true_{attribute}"],
            predictions[f"predicted_{attribute}"],
            normalize="index",
        )
        frame_confusion.to_csv(attribute_dir / "frame_confusion_matrix.csv")
        sequence_balanced_confusion(
            predictions[f"true_{attribute}"],
            predictions[f"predicted_{attribute}"],
            predictions["held_out_sequence"],
        ).to_csv(attribute_dir / "confusion_matrix.csv")
        save_confusion_plot(
            predictions[f"true_{attribute}"],
            predictions[f"predicted_{attribute}"],
            attribute_dir / "confusion_matrix.png",
            f"Independent LOSO: {attribute}",
        )

    joint_frames = []
    joint_supported = support.loc[support["joint_supported"], "held_out_sequence"].astype(str)
    for sequence in joint_supported:
        current = {
            attribute: prediction_frames[attribute][
                prediction_frames[attribute]["held_out_sequence"].astype(str).eq(sequence)
            ]
            for attribute in ATTRIBUTES
        }
        joint_frames.append(align_attribute_predictions(current))
    joint = pd.concat(joint_frames, ignore_index=True)
    joint_dir = run_dir / "joint"
    joint_dir.mkdir(parents=True, exist_ok=True)
    joint.to_csv(joint_dir / "predictions.csv", index=False)
    joint_metrics = sequence_balanced_metrics(
        joint["true_semantic_key"],
        joint["predicted_semantic_key"],
        joint["held_out_sequence"],
    )
    pd.DataFrame([joint_metrics]).to_csv(joint_dir / "metrics.csv", index=False)
    joint.groupby("held_out_sequence", as_index=False).agg(
        samples=("exact_match", "size"),
        exact_match_accuracy=("exact_match", "mean"),
        mean_joint_confidence=("joint_confidence_min", "mean"),
    ).to_csv(joint_dir / "per_sequence_metrics.csv", index=False)
    error_breakdown = joint["error_type"].value_counts().rename_axis(
        "error_type"
    ).reset_index(name="samples")
    error_breakdown["fraction"] = error_breakdown["samples"] / len(joint)
    error_breakdown.to_csv(joint_dir / "error_breakdown.csv", index=False)
    joint.groupby("true_semantic_key", as_index=False).agg(
        sequences=("held_out_sequence", "nunique"),
        samples=("exact_match", "size"),
        accuracy=("exact_match", "mean"),
    ).to_csv(joint_dir / "per_combination_metrics.csv", index=False)

    attribute_summary = pd.DataFrame([
        summarize_attribute(attribute, prediction_frames[attribute])
        for attribute in ATTRIBUTES
    ])
    attribute_summary.to_csv(run_dir / "attribute_summary.csv", index=False)
    pd.DataFrame(fold_metric_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(per_sequence_rows).to_csv(run_dir / "per_sequence_metrics.csv", index=False)
    pd.concat(leaderboard_frames, ignore_index=True).to_csv(
        run_dir / "inner_search_leaderboard.csv", index=False
    )
    selected_frame = pd.DataFrame(selected_rows)
    selected_frame.to_csv(run_dir / "selected_configs.csv", index=False)
    atomic_write_json(
        run_dir / "selected_configs.json",
        {"outer_fold_selections": selected_frame.to_dict("records")},
    )
    pd.DataFrame(operating_rows).to_csv(
        run_dir / "confidence_operating_points.csv", index=False
    )
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-independent-attributes/v2",
        "status": "complete",
        "created_utc": utc_now(),
        "descriptor_bank": str(bank.root),
        "arguments": vars(args),
        "protocol": (
            "nested leave-one-sequence-out; sequence IDs define groups only and "
            "are never model inputs or prediction targets"
        ),
        "selection_metric": "sequence_macro_f1, then sequence balanced accuracy",
        "outer_test": "descriptor test split of a sequence wholly excluded from training",
        "joint_policy": "intersection of independently supported folds, aligned by causal window_end_row",
        "run_dir": str(run_dir),
    })
    print("Support matrix:")
    print(support.to_string(index=False))
    print("\nIndependent attribute summary:")
    print(attribute_summary.to_string(index=False))
    print("\nJoint summary:")
    print(pd.DataFrame([joint_metrics]).to_string(index=False))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
