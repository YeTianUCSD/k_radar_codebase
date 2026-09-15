#!/usr/bin/env python3
"""Evaluate factorized attribute recognition on unseen sequence combinations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.attributes import (  # noqa: E402
    ATTRIBUTES, ATTRIBUTE_CLASSES, AttributeHead, FactorizedAttributeRecognizer,
    attribute_frame, chronological_fit_validation_masks, expected_calibration_error,
    fit_probabilistic_classifier, sequence_attribute_table,
)
from scene_discovery.common import atomic_write_json, make_run_dir, seed_everything, utc_now  # noqa: E402
from scene_discovery.evaluation import classification_metrics  # noqa: E402
from scene_discovery.feature_bank import DEFAULT_SEQUENCES, MODALITIES, DescriptorBank  # noqa: E402
from scene_discovery.preprocessing import PerModalityProjector, ProjectionSettings, modality_combinations  # noqa: E402
from scene_discovery.temporal import aggregate_windows  # noqa: E402


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
    parser.add_argument("--held-out-scenes", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 10, 20, 30])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--svm-calibration-fraction", type=float, default=0.2)
    parser.add_argument(
        "--min-support-scenes", type=int, default=1,
        help="Minimum remaining sequences required for every held-out attribute value.",
    )
    parser.add_argument(
        "--include-unsupported", action="store_true",
        help="Also score folds whose held-out scene contains an attribute class absent from training.",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def normalize_scene(value: object) -> str:
    result = str(value).lower()
    if result.startswith("seq"):
        result = result[3:]
    return str(int(result))


def window_data(parts, index, mask, modalities: Tuple[str, ...], window, stride):
    features = np.concatenate([parts[modality][mask] for modality in modalities], axis=1)
    subset_index = index.loc[mask].reset_index(drop=True)
    return aggregate_windows(features, subset_index, window, 1 if window == 1 else stride)


def fit_predict(
    model_name, seed, train_x, train_y, train_index, evaluation_x,
    svm_calibration_fraction=0.2,
):
    estimator = fit_probabilistic_classifier(
        model_name, seed, train_x, train_y, train_index, svm_calibration_fraction
    )
    probabilities = np.asarray(estimator.predict_proba(evaluation_x), dtype=np.float32)
    classes = np.asarray(estimator.classes_).astype(str)
    positions = probabilities.argmax(axis=1)
    return estimator, classes[positions], probabilities.max(axis=1)


def concatenate_keys(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([
        f"{weather}|{road}|{lighting}"
        for weather, road, lighting in zip(
            predictions["weather"], predictions["road"], predictions["lighting"]
        )
    ])


def main() -> None:
    args = parse_args()
    if args.min_support_scenes < 1:
        raise ValueError("--min-support-scenes must be positive")
    seed_everything(args.seed)
    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(args.output_root.expanduser().resolve(), "compositional")

    train_index = bank.index("train")
    test_index = bank.index("test")
    train_arrays = {
        modality: bank.array("train", modality, args.descriptor)
        for modality in args.modalities
    }
    test_arrays = {
        modality: bank.array("test", modality, args.descriptor)
        for modality in args.modalities
    }
    combined_scene_table = sequence_attribute_table(
        pd.concat([train_index, test_index], ignore_index=True)
    )
    combined_scene_table["sequence"] = combined_scene_table["sequence"].astype(str)
    combined_scene_table.to_csv(run_dir / "scene_attributes.csv", index=False)

    support_rows = []
    folds = []
    requested = [normalize_scene(value) for value in args.held_out_scenes]
    for held_out in requested:
        held_rows = combined_scene_table[
            combined_scene_table["sequence"].eq(held_out)
        ]
        if len(held_rows) != 1:
            raise ValueError(f"Expected exactly one attribute row for Seq{held_out}")
        held_attributes = held_rows.iloc[0]
        remaining = combined_scene_table[
            ~combined_scene_table["sequence"].eq(held_out)
        ]
        support_counts = {
            attribute: int(remaining[attribute].eq(held_attributes[attribute]).sum())
            for attribute in ATTRIBUTES
        }
        support = {
            attribute: count >= args.min_support_scenes
            for attribute, count in support_counts.items()
        }
        fully_supported = all(support.values())
        support_rows.append({
            "held_out_sequence": held_out,
            **{f"true_{attribute}": held_attributes[attribute] for attribute in ATTRIBUTES},
            "true_semantic_key": held_attributes["semantic_key"],
            **{f"{attribute}_support_scenes": support_counts[attribute] for attribute in ATTRIBUTES},
            **{f"{attribute}_supported": value for attribute, value in support.items()},
            "fully_supported": fully_supported,
            "evaluated": fully_supported or args.include_unsupported,
        })
        if fully_supported or args.include_unsupported:
            folds.append(held_out)
    pd.DataFrame(support_rows).to_csv(run_dir / "support_matrix.csv", index=False)
    if not folds:
        raise ValueError("No eligible held-out scenes; use --include-unsupported to force evaluation")

    combinations = modality_combinations(args.modalities)
    all_search_rows = []
    all_metric_rows = []
    all_predictions = []
    selected_payload = {}

    for fold_number, held_out in enumerate(folds):
        sequence_values = train_index["sequence"].astype(str).to_numpy()
        outer_train_mask = sequence_values != held_out
        outer_index = train_index.loc[outer_train_mask].reset_index(drop=True)
        inner_fit_local, inner_validation_local = chronological_fit_validation_masks(
            outer_index, args.validation_fraction
        )
        outer_positions = np.flatnonzero(outer_train_mask)
        inner_fit_mask = np.zeros(len(train_index), dtype=bool)
        inner_validation_mask = np.zeros(len(train_index), dtype=bool)
        inner_fit_mask[outer_positions[inner_fit_local]] = True
        inner_validation_mask[outer_positions[inner_validation_local]] = True

        selection_projector = PerModalityProjector(
            args.modalities,
            ProjectionSettings(
                pca_components=args.pca_components,
                seed=args.seed + fold_number,
            ),
        ).fit({
            modality: np.asarray(values[inner_fit_mask])
            for modality, values in train_arrays.items()
        })
        selection_parts = selection_projector.transform_modalities(train_arrays)
        candidates = []
        prediction_cache = {}
        prepared = {}
        for window in args.windows:
            for combo in combinations:
                fit_x, fit_idx = window_data(
                    selection_parts, train_index, inner_fit_mask,
                    combo, window, args.stride,
                )
                validation_x, validation_idx = window_data(
                    selection_parts, train_index, inner_validation_mask,
                    combo, window, args.stride,
                )
                prepared[(combo, window)] = (fit_x, fit_idx, validation_x, validation_idx)
                fit_labels = attribute_frame(fit_idx)
                validation_labels = attribute_frame(validation_idx)
                for attribute in ATTRIBUTES:
                    for model_name in args.models:
                        estimator, prediction, confidence = fit_predict(
                            model_name,
                            args.seed + fold_number,
                            fit_x,
                            fit_labels[attribute].to_numpy(),
                            fit_idx,
                            validation_x,
                            args.svm_calibration_fraction,
                        )
                        metrics = classification_metrics(
                            validation_labels[attribute].to_numpy(), prediction
                        )
                        metrics["ece"] = expected_calibration_error(
                            validation_labels[attribute].to_numpy(),
                            prediction,
                            confidence,
                        )
                        record = {
                            "held_out_sequence": held_out,
                            "attribute": attribute,
                            "modalities": "+".join(combo),
                            "window": window,
                            "model": model_name,
                            "fit_samples": len(fit_x),
                            "validation_samples": len(validation_x),
                            **metrics,
                        }
                        candidates.append(record)
                        prediction_cache[(attribute, combo, window, model_name)] = prediction
        fold_search = pd.DataFrame(candidates)
        all_search_rows.extend(candidates)

        selected_by_window = {}
        joint_rows = []
        for window in args.windows:
            selected_by_window[window] = {}
            predictions = {}
            validation_truth = None
            for attribute in ATTRIBUTES:
                winners = fold_search[
                    fold_search["attribute"].eq(attribute)
                    & fold_search["window"].eq(window)
                ].sort_values(
                    ["macro_f1", "balanced_accuracy", "accuracy", "ece"],
                    ascending=[False, False, False, True],
                )
                winner = winners.iloc[0]
                combo = tuple(str(winner["modalities"]).split("+"))
                model_name = str(winner["model"])
                selected_by_window[window][attribute] = {
                    "modalities": combo,
                    "model": model_name,
                    "validation_macro_f1": float(winner["macro_f1"]),
                }
                predictions[attribute] = prediction_cache[
                    (attribute, combo, window, model_name)
                ]
                if validation_truth is None:
                    validation_truth = attribute_frame(prepared[(combo, window)][3])
            predicted_keys = concatenate_keys(predictions)
            joint_rows.append({
                "window": window,
                "exact_match_accuracy": float(np.mean(
                    predicted_keys == validation_truth["semantic_key"].to_numpy()
                )),
                "mean_selected_macro_f1": float(np.mean([
                    selected_by_window[window][attribute]["validation_macro_f1"]
                    for attribute in ATTRIBUTES
                ])),
            })
        joint_frame = pd.DataFrame(joint_rows).sort_values(
            ["exact_match_accuracy", "mean_selected_macro_f1"], ascending=False
        )
        selected_window = int(joint_frame.iloc[0]["window"])
        selected = selected_by_window[selected_window]

        final_projector = PerModalityProjector(
            args.modalities,
            ProjectionSettings(
                pca_components=args.pca_components,
                seed=args.seed + fold_number,
            ),
        ).fit({
            modality: np.asarray(values[outer_train_mask])
            for modality, values in train_arrays.items()
        })
        train_parts = final_projector.transform_modalities(train_arrays)
        test_parts = final_projector.transform_modalities(test_arrays)
        heldout_test_mask = test_index["sequence"].astype(str).to_numpy() == held_out
        heads = {}
        predictions = {}
        confidences = {}
        output_index = None
        output_truth = None
        for attribute in ATTRIBUTES:
            combo = tuple(selected[attribute]["modalities"])
            train_x, train_idx = window_data(
                train_parts, train_index, outer_train_mask,
                combo, selected_window, args.stride,
            )
            test_x, current_index = window_data(
                test_parts, test_index, heldout_test_mask,
                combo, selected_window, args.stride,
            )
            train_labels = attribute_frame(train_idx)
            test_labels = attribute_frame(current_index)
            estimator, prediction, confidence = fit_predict(
                selected[attribute]["model"],
                args.seed + fold_number,
                train_x,
                train_labels[attribute].to_numpy(),
                train_idx,
                test_x,
                args.svm_calibration_fraction,
            )
            heads[attribute] = AttributeHead(
                attribute=attribute,
                modalities=combo,
                model_name=selected[attribute]["model"],
                estimator=estimator,
            )
            predictions[attribute] = prediction
            confidences[attribute] = confidence
            metrics = classification_metrics(
                test_labels[attribute].to_numpy(), prediction
            )
            all_metric_rows.append({
                "held_out_sequence": held_out,
                "scope": "attribute",
                "attribute": attribute,
                "modalities": "+".join(combo),
                "window": selected_window,
                "model": selected[attribute]["model"],
                "samples": len(prediction),
                "supported": bool(pd.DataFrame(support_rows).set_index("held_out_sequence").loc[held_out, f"{attribute}_supported"]),
                "ece": expected_calibration_error(
                    test_labels[attribute].to_numpy(), prediction, confidence
                ),
                **metrics,
            })
            if output_index is None:
                output_index = current_index.reset_index(drop=True)
                output_truth = test_labels.reset_index(drop=True)

        recognizer = FactorizedAttributeRecognizer(
            projector=final_projector,
            heads=heads,
            window=selected_window,
            stride=1 if selected_window == 1 else args.stride,
            taxonomy={key: list(value) for key, value in ATTRIBUTE_CLASSES.items()},
            training_sequences=list(
                train_idx["sequence"].astype(str).drop_duplicates()
            ),
        )
        joblib.dump(recognizer, run_dir / f"recognizer_heldout_seq{held_out}.joblib")
        predicted_keys = concatenate_keys(predictions)
        true_keys = output_truth["semantic_key"].to_numpy()
        exact = predicted_keys == true_keys
        joint_metrics = classification_metrics(true_keys, predicted_keys)
        all_metric_rows.append({
            "held_out_sequence": held_out,
            "scope": "joint",
            "attribute": "semantic_key",
            "modalities": "factorized",
            "window": selected_window,
            "model": "selected_per_head",
            "samples": len(exact),
            "supported": bool(pd.DataFrame(support_rows).set_index("held_out_sequence").loc[held_out, "fully_supported"]),
            "ece": expected_calibration_error(
                true_keys, predicted_keys, np.stack(list(confidences.values())).min(axis=0)
            ),
            **joint_metrics,
        })

        output = output_index.copy()
        output["held_out_sequence"] = held_out
        for attribute in ATTRIBUTES:
            output[f"true_{attribute}"] = output_truth[attribute].to_numpy()
            output[f"predicted_{attribute}"] = predictions[attribute]
            output[f"{attribute}_confidence"] = confidences[attribute]
        output["true_semantic_key"] = true_keys
        output["predicted_semantic_key"] = predicted_keys
        output["joint_confidence_min"] = np.stack(list(confidences.values())).min(axis=0)
        output["exact_match"] = exact
        all_predictions.append(output)
        selected_payload[held_out] = {
            "selected_window": selected_window,
            "heads": selected,
            "joint_validation": joint_frame.iloc[0].to_dict(),
            "recognizer_file": f"recognizer_heldout_seq{held_out}.joblib",
        }

    pd.DataFrame(all_search_rows).to_csv(run_dir / "inner_validation_leaderboard.csv", index=False)
    metrics_frame = pd.DataFrame(all_metric_rows)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    predictions_frame.to_csv(run_dir / "predictions.csv", index=False)
    summary = metrics_frame.groupby(
        ["scope", "attribute"], as_index=False
    )[["accuracy", "balanced_accuracy", "macro_f1", "ece"]].mean()
    summary.to_csv(run_dir / "summary.csv", index=False)
    atomic_write_json(run_dir / "selected_configs.json", selected_payload)
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-scene-compositional/v1",
        "status": "complete",
        "created_utc": utc_now(),
        "descriptor_bank": str(bank.root),
        "arguments": vars(args),
        "evaluated_folds": folds,
        "protocol": "outer held-out sequence excluded from projector, model fitting, and inner configuration selection",
        "unsupported_policy": "skipped unless --include-unsupported",
        "support_definition": "each held-out attribute value must appear in at least --min-support-scenes remaining sequences",
        "run_dir": str(run_dir),
    })
    print(pd.DataFrame(support_rows).to_string(index=False))
    print(summary.to_string(index=False))
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
