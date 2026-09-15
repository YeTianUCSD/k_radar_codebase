#!/usr/bin/env python3
"""Train and evaluate factorized weather, road, and lighting recognizers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.attributes import (  # noqa: E402
    ATTRIBUTES, ATTRIBUTE_CLASSES, AttributeHead, FactorizedAttributeRecognizer,
    attribute_frame, chronological_fit_validation_masks, confidence_operating_points,
    expected_calibration_error, fit_probabilistic_classifier, select_confidence_threshold,
    sequence_attribute_table,
)
from scene_discovery.common import atomic_write_json, make_run_dir, seed_everything, utc_now  # noqa: E402
from scene_discovery.evaluation import classification_metrics  # noqa: E402
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
    parser.add_argument(
        "--models", nargs="+", choices=("logreg", "linear_svm", "knn"),
        default=["logreg", "linear_svm", "knn"],
    )
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 10, 20, 30])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--svm-calibration-fraction", type=float, default=0.2)
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.9)
    parser.add_argument(
        "--confidence-thresholds", nargs="+", type=float,
        default=[0.0, 0.5, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def window_data(
    parts: Dict[str, np.ndarray],
    index: pd.DataFrame,
    mask: np.ndarray,
    modalities: Tuple[str, ...],
    window: int,
    stride: int,
):
    features = np.concatenate([parts[modality][mask] for modality in modalities], axis=1)
    subset_index = index.loc[mask].reset_index(drop=True)
    return aggregate_windows(features, subset_index, window, 1 if window == 1 else stride)


def prediction_metrics(truth, prediction, confidence):
    result = classification_metrics(truth, prediction)
    result["ece"] = expected_calibration_error(truth, prediction, confidence)
    result["mean_confidence"] = float(np.mean(confidence))
    return result


def fit_and_predict(
    model_name, seed, train_x, train_y, train_index, evaluation_x,
    svm_calibration_fraction=0.2,
):
    estimator = fit_probabilistic_classifier(
        model_name, seed, train_x, train_y, train_index, svm_calibration_fraction
    )
    probabilities = np.asarray(estimator.predict_proba(evaluation_x), dtype=np.float32)
    classes = np.asarray(estimator.classes_).astype(str)
    positions = probabilities.argmax(axis=1)
    return estimator, classes[positions], probabilities.max(axis=1), probabilities, classes


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(args.output_root.expanduser().resolve(), "attributes")

    train_index = bank.index("train")
    test_index = bank.index("test")
    scene_table = sequence_attribute_table(pd.concat([train_index, test_index], ignore_index=True))
    scene_table.to_csv(run_dir / "scene_attributes.csv", index=False)
    fit_mask, validation_mask = chronological_fit_validation_masks(
        train_index, args.validation_fraction
    )
    train_arrays = {
        modality: bank.array("train", modality, args.descriptor)
        for modality in args.modalities
    }
    test_arrays = {
        modality: bank.array("test", modality, args.descriptor)
        for modality in args.modalities
    }

    selection_projector = PerModalityProjector(
        args.modalities,
        ProjectionSettings(pca_components=args.pca_components, seed=args.seed),
    ).fit({
        modality: np.asarray(values[fit_mask])
        for modality, values in train_arrays.items()
    })
    selection_parts = selection_projector.transform_modalities(train_arrays)
    combinations = modality_combinations(args.modalities)
    prepared = {}
    for window in args.windows:
        for combo in combinations:
            fit_x, fit_idx = window_data(
                selection_parts, train_index, fit_mask, combo, window, args.stride
            )
            validation_x, validation_idx = window_data(
                selection_parts, train_index, validation_mask, combo, window, args.stride
            )
            prepared[(combo, window)] = (fit_x, fit_idx, validation_x, validation_idx)

    leaderboard_rows = []
    cached_predictions = {}
    for window in args.windows:
        for combo in combinations:
            combo_name = "+".join(combo)
            fit_x, fit_idx, validation_x, validation_idx = prepared[(combo, window)]
            fit_labels = attribute_frame(fit_idx)
            validation_labels = attribute_frame(validation_idx)
            for attribute in ATTRIBUTES:
                train_y = fit_labels[attribute].to_numpy()
                validation_y = validation_labels[attribute].to_numpy()
                for model_name in args.models:
                    estimator, prediction, confidence, _, _ = fit_and_predict(
                        model_name, args.seed, fit_x, train_y, fit_idx, validation_x,
                        args.svm_calibration_fraction
                    )
                    metrics = prediction_metrics(validation_y, prediction, confidence)
                    key = (attribute, combo, window, model_name)
                    cached_predictions[key] = {
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                    leaderboard_rows.append({
                        "attribute": attribute,
                        "modalities": combo_name,
                        "window": window,
                        "stride": 1 if window == 1 else args.stride,
                        "model": model_name,
                        "fit_samples": len(train_y),
                        "validation_samples": len(validation_y),
                        **metrics,
                    })

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard.to_csv(run_dir / "attribute_leaderboard.csv", index=False)
    selected_by_window = {}
    joint_validation_rows = []
    validation_operating_rows = []
    for window in args.windows:
        selected_by_window[window] = {}
        validation_truth = None
        predictions = {}
        confidences = {}
        for attribute in ATTRIBUTES:
            candidates = leaderboard[
                leaderboard["attribute"].eq(attribute)
                & leaderboard["window"].eq(window)
            ].sort_values(
                ["macro_f1", "balanced_accuracy", "accuracy", "ece"],
                ascending=[False, False, False, True],
            )
            winner = candidates.iloc[0]
            combo = tuple(str(winner["modalities"]).split("+"))
            key = (attribute, combo, window, str(winner["model"]))
            selected_by_window[window][attribute] = {
                "modalities": combo,
                "model": str(winner["model"]),
                "validation_metrics": {
                    key_name: float(winner[key_name])
                    for key_name in ("accuracy", "balanced_accuracy", "macro_f1", "ece")
                },
            }
            predictions[attribute] = cached_predictions[key]["prediction"]
            confidences[attribute] = cached_predictions[key]["confidence"]
            current_validation_truth = attribute_frame(prepared[(combo, window)][3])
            selected_threshold, threshold_rows = select_confidence_threshold(
                current_validation_truth[attribute].to_numpy(),
                predictions[attribute],
                confidences[attribute],
                args.confidence_thresholds,
                args.target_accepted_accuracy,
            )
            selected_by_window[window][attribute]["confidence_threshold"] = selected_threshold
            for row in threshold_rows:
                validation_operating_rows.append({
                    "window": window, "attribute": attribute, **row
                })
            if validation_truth is None:
                validation_truth = attribute_frame(prepared[(combo, window)][3])
        predicted_key = np.asarray([
            f"{weather}|{road}|{lighting}"
            for weather, road, lighting in zip(
                predictions["weather"], predictions["road"], predictions["lighting"]
            )
        ])
        exact = predicted_key == validation_truth["semantic_key"].to_numpy()
        joint_validation_rows.append({
            "window": window,
            "validation_samples": len(exact),
            "exact_match_accuracy": float(exact.mean()),
            "mean_attribute_accuracy": float(np.mean([
                np.mean(predictions[attribute] == validation_truth[attribute].to_numpy())
                for attribute in ATTRIBUTES
            ])),
            "joint_confidence_min_mean": float(np.stack(list(confidences.values())).min(axis=0).mean()),
            "mean_selected_macro_f1": float(np.mean([
                selected_by_window[window][attribute]["validation_metrics"]["macro_f1"]
                for attribute in ATTRIBUTES
            ])),
        })

    joint_validation = pd.DataFrame(joint_validation_rows).sort_values(
        ["exact_match_accuracy", "mean_selected_macro_f1", "mean_attribute_accuracy"],
        ascending=False,
    )
    joint_validation.to_csv(run_dir / "joint_validation.csv", index=False)
    selected_window = int(joint_validation.iloc[0]["window"])
    selected_configs = selected_by_window[selected_window]
    confidence_thresholds = {
        attribute: float(selected_configs[attribute]["confidence_threshold"])
        for attribute in ATTRIBUTES
    }
    selected_validation_operating = [
        row for row in validation_operating_rows if row["window"] == selected_window
    ]
    pd.DataFrame(selected_validation_operating).to_csv(
        run_dir / "confidence_operating_points.csv", index=False
    )
    pd.DataFrame(validation_operating_rows).to_csv(
        run_dir / "all_validation_confidence_operating_points.csv", index=False
    )

    final_projector = PerModalityProjector(
        args.modalities,
        ProjectionSettings(pca_components=args.pca_components, seed=args.seed),
    ).fit(train_arrays)
    joblib.dump(final_projector, run_dir / "projector.joblib")
    final_train_parts = final_projector.transform_modalities(train_arrays)
    final_test_parts = final_projector.transform_modalities(test_arrays)
    heads = {}
    output_index = None
    output_truth = None
    prediction_columns = {}
    confidence_columns = {}
    probability_columns = {}
    test_metric_rows = []
    per_class_rows = []
    operating_rows = []

    for attribute in ATTRIBUTES:
        config = selected_configs[attribute]
        combo = tuple(config["modalities"])
        train_x, train_idx = window_data(
            final_train_parts,
            train_index,
            np.ones(len(train_index), dtype=bool),
            combo,
            selected_window,
            args.stride,
        )
        test_x, current_test_idx = window_data(
            final_test_parts,
            test_index,
            np.ones(len(test_index), dtype=bool),
            combo,
            selected_window,
            args.stride,
        )
        train_labels = attribute_frame(train_idx)
        test_labels = attribute_frame(current_test_idx)
        estimator, prediction, confidence, probabilities, classes = fit_and_predict(
            config["model"], args.seed, train_x,
            train_labels[attribute].to_numpy(), train_idx, test_x,
            args.svm_calibration_fraction,
        )
        heads[attribute] = AttributeHead(
            attribute=attribute,
            modalities=combo,
            model_name=config["model"],
            estimator=estimator,
        )
        prediction_columns[attribute] = prediction
        confidence_columns[attribute] = confidence
        probability_columns[attribute] = (probabilities, classes)
        if output_index is None:
            output_index = current_test_idx.reset_index(drop=True)
            output_truth = test_labels.reset_index(drop=True)
        elif not np.array_equal(
            output_index["sample_id"].astype(str).to_numpy(),
            current_test_idx["sample_id"].astype(str).to_numpy(),
        ):
            raise RuntimeError("Selected attribute heads produced misaligned test windows")

        metrics = prediction_metrics(
            test_labels[attribute].to_numpy(), prediction, confidence
        )
        test_metric_rows.append({
            "scope": "attribute",
            "attribute": attribute,
            "modalities": "+".join(combo),
            "window": selected_window,
            "model": config["model"],
            "samples": len(prediction),
            **metrics,
        })
        labels = list(ATTRIBUTE_CLASSES[attribute])
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels[attribute].to_numpy(), prediction,
            labels=labels, zero_division=0,
        )
        for label, p_value, r_value, f_value, count in zip(
            labels, precision, recall, f1, support
        ):
            per_class_rows.append({
                "attribute": attribute,
                "class": label,
                "precision": float(p_value),
                "recall": float(r_value),
                "f1": float(f_value),
                "support": int(count),
            })
        for row in confidence_operating_points(
            test_labels[attribute].to_numpy(),
            prediction,
            confidence,
            args.confidence_thresholds,
        ):
            operating_rows.append({"attribute": attribute, **row})
        save_confusion_plot(
            test_labels[attribute].to_numpy(),
            prediction,
            run_dir / f"confusion_{attribute}.png",
            f"{attribute}: {config['model']}, {'+'.join(combo)}, window={selected_window}",
        )

    recognizer = FactorizedAttributeRecognizer(
        projector=final_projector,
        heads=heads,
        window=selected_window,
        stride=1 if selected_window == 1 else args.stride,
        taxonomy={key: list(value) for key, value in ATTRIBUTE_CLASSES.items()},
        confidence_thresholds=confidence_thresholds,
        training_sequences=list(
            train_index["sequence"].astype(str).drop_duplicates()
        ),
    )
    joblib.dump(recognizer, run_dir / "recognizer.joblib")

    predictions_frame = output_index.copy()
    for attribute in ATTRIBUTES:
        predictions_frame[f"true_{attribute}"] = output_truth[attribute].to_numpy()
        predictions_frame[f"predicted_{attribute}"] = prediction_columns[attribute]
        predictions_frame[f"{attribute}_confidence"] = confidence_columns[attribute]
        predictions_frame[f"{attribute}_accepted"] = (
            confidence_columns[attribute] >= confidence_thresholds[attribute]
        )
        probabilities, classes = probability_columns[attribute]
        class_to_position = {label: position for position, label in enumerate(classes)}
        for label in ATTRIBUTE_CLASSES[attribute]:
            position = class_to_position.get(label)
            predictions_frame[f"{attribute}_prob_{label}"] = (
                probabilities[:, position] if position is not None else 0.0
            )
    predictions_frame["true_semantic_key"] = output_truth["semantic_key"].to_numpy()
    predictions_frame["predicted_semantic_key"] = (
        predictions_frame["predicted_weather"]
        + "|"
        + predictions_frame["predicted_road"]
        + "|"
        + predictions_frame["predicted_lighting"]
    )
    confidence_stack = np.stack([confidence_columns[key] for key in ATTRIBUTES])
    predictions_frame["joint_confidence_min"] = confidence_stack.min(axis=0)
    predictions_frame["joint_confidence_product"] = confidence_stack.prod(axis=0)
    predictions_frame["exact_match"] = (
        predictions_frame["true_semantic_key"]
        == predictions_frame["predicted_semantic_key"]
    )
    predictions_frame.to_csv(run_dir / "predictions.csv", index=False)

    joint_metrics = classification_metrics(
        predictions_frame["true_semantic_key"],
        predictions_frame["predicted_semantic_key"],
    )
    test_metric_rows.append({
        "scope": "joint",
        "attribute": "semantic_key",
        "modalities": "factorized",
        "window": selected_window,
        "model": "selected_per_head",
        "samples": len(predictions_frame),
        **joint_metrics,
        "ece": expected_calibration_error(
            predictions_frame["true_semantic_key"],
            predictions_frame["predicted_semantic_key"],
            predictions_frame["joint_confidence_min"],
        ),
        "mean_confidence": float(predictions_frame["joint_confidence_min"].mean()),
    })
    pd.DataFrame(test_metric_rows).to_csv(run_dir / "test_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(run_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(
        run_dir / "test_confidence_operating_points.csv", index=False
    )
    predictions_frame.groupby("sequence", as_index=False).agg(
        samples=("exact_match", "size"),
        exact_match_accuracy=("exact_match", "mean"),
        mean_joint_confidence=("joint_confidence_min", "mean"),
    ).to_csv(run_dir / "per_scene_joint_metrics.csv", index=False)

    selected_payload = {
        "selection_split": "chronological final fraction of descriptor train split",
        "selected_window": selected_window,
        "heads": selected_configs,
        "joint_validation": joint_validation.iloc[0].to_dict(),
    }
    atomic_write_json(run_dir / "selected_configs.json", selected_payload)
    atomic_write_json(run_dir / "taxonomy.json", {
        "classes": ATTRIBUTE_CLASSES,
        "source_columns": {
            "weather": "climate",
            "road": "road_type",
            "lighting": "capture_time",
        },
        "notes": {
            "road_countryside": "Source metadata uses countryside; no silent remap to suburban.",
        },
    })
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-scene-attributes/v2",
        "status": "complete",
        "created_utc": utc_now(),
        "descriptor_bank": str(bank.root),
        "arguments": vars(args),
        "selection_policy": "model, modality, and common window selected on chronological train validation only; test evaluated once after refit on full train",
        "recognizer": recognizer.metadata(),
        "run_dir": str(run_dir),
    })
    print(pd.DataFrame(test_metric_rows).to_string(index=False))
    print(f"Selected window: {selected_window}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
