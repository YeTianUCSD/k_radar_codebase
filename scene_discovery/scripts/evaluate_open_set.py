#!/usr/bin/env python3
"""Evaluate held-out-scene detection without exposing the held-out Seq ID."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.common import atomic_write_json, make_run_dir, seed_everything, utc_now  # noqa: E402
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
    parser.add_argument("--held-out-scenes", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 20])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument(
        "--threshold-quantiles", nargs="+", type=float,
        default=[0.9, 0.95, 0.975, 0.99, 0.995],
        help="Quantiles computed exclusively from the chronological known-scene calibration split.",
    )
    parser.add_argument(
        "--threshold-quantile", type=float,
        help="Deprecated single-quantile alias retained for command compatibility.",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def normalize_scene(value: object) -> str:
    result = str(value).lower()
    if result.startswith("seq"):
        result = result[3:]
    return str(int(result))


def chronological_masks(index: pd.DataFrame, excluded: str, calibration_fraction: float):
    fit = np.zeros(len(index), dtype=bool)
    calibration = np.zeros(len(index), dtype=bool)
    sequence_values = index["sequence"].astype(str).to_numpy()
    for sequence in index["sequence"].astype(str).drop_duplicates():
        if sequence == excluded:
            continue
        positions = np.flatnonzero(sequence_values == sequence)
        cut = max(1, min(len(positions) - 1, int(round(len(positions) * (1.0 - calibration_fraction)))))
        fit[positions[:cut]] = True
        calibration[positions[cut:]] = True
    return fit, calibration


def distances_to_centroids(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)


def false_positive_rate_at_95_tpr(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    positions = np.flatnonzero(tpr >= 0.95)
    return float(fpr[positions[0]]) if len(positions) else 1.0


def window_subset(features, index, mask, window, stride):
    subset_index = index.loc[mask].reset_index(drop=True)
    subset_features = features[mask]
    return aggregate_windows(subset_features, subset_index, window, 1 if window == 1 else stride)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 0.5:
        raise ValueError("--calibration-fraction must be in (0, 0.5)")
    quantiles = [args.threshold_quantile] if args.threshold_quantile is not None else args.threshold_quantiles
    quantiles = sorted(set(float(value) for value in quantiles))
    if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("threshold quantiles must be in (0, 1)")

    seed_everything(args.seed)
    bank = DescriptorBank(args.descriptor_bank)
    run_dir = make_run_dir(args.output_root.expanduser().resolve(), "open_set")
    train_index = bank.index("train")
    test_index = bank.index("test")
    train_arrays = {m: bank.array("train", m, args.descriptor) for m in args.modalities}
    test_arrays = {m: bank.array("test", m, args.descriptor) for m in args.modalities}
    metric_rows = []
    prediction_frames = []
    model_records = []

    for held_out in [normalize_scene(value) for value in args.held_out_scenes]:
        fit_mask, calibration_mask = chronological_masks(
            train_index, held_out, args.calibration_fraction
        )
        projector = PerModalityProjector(
            args.modalities,
            ProjectionSettings(pca_components=args.pca_components, seed=args.seed),
        ).fit({m: np.asarray(values[fit_mask]) for m, values in train_arrays.items()})
        train_parts = projector.transform_modalities(train_arrays)
        test_parts = projector.transform_modalities(test_arrays)
        projector_name = f"projector_heldout_seq{held_out}.joblib"
        joblib.dump(projector, run_dir / projector_name)

        known_test_mask = test_index["sequence"].astype(str).to_numpy() != held_out
        unknown_test_mask = ~known_test_mask
        for combo in modality_combinations(args.modalities):
            combo_name = "+".join(combo)
            train_features = np.concatenate([train_parts[m] for m in combo], axis=1)
            test_features = np.concatenate([test_parts[m] for m in combo], axis=1)
            for window in args.windows:
                fit_x, fit_idx = window_subset(
                    train_features, train_index, fit_mask, window, args.stride
                )
                calibration_x, _ = window_subset(
                    train_features, train_index, calibration_mask, window, args.stride
                )
                known_x, known_idx = window_subset(
                    test_features, test_index, known_test_mask, window, args.stride
                )
                unknown_x, unknown_idx = window_subset(
                    test_features, test_index, unknown_test_mask, window, args.stride
                )
                known_labels = sorted(
                    fit_idx["sequence"].astype(str).unique(),
                    key=lambda value: int(value),
                )
                fit_labels = fit_idx["sequence"].astype(str).to_numpy()
                centroids = np.stack(
                    [fit_x[fit_labels == label].mean(axis=0) for label in known_labels]
                )
                calibration_distance = distances_to_centroids(calibration_x, centroids).min(axis=1)
                known_distances = distances_to_centroids(known_x, centroids)
                unknown_distances = distances_to_centroids(unknown_x, centroids)
                known_score = known_distances.min(axis=1)
                unknown_score = unknown_distances.min(axis=1)
                known_prediction = np.asarray(known_labels)[known_distances.argmin(axis=1)]
                known_truth = known_idx["sequence"].astype(str).to_numpy()
                detection_labels = np.concatenate(
                    [np.zeros(len(known_score), dtype=int), np.ones(len(unknown_score), dtype=int)]
                )
                detection_scores = np.concatenate([known_score, unknown_score])
                combo_slug = combo_name.replace("+", "-")
                model_name = f"model_heldout_seq{held_out}_{combo_slug}_w{window}.joblib"
                thresholds = {
                    str(quantile): float(np.quantile(calibration_distance, quantile))
                    for quantile in quantiles
                }
                joblib.dump({
                    "projector_file": projector_name,
                    "held_out_sequence": held_out,
                    "modalities": combo,
                    "window": window,
                    "stride": 1 if window == 1 else args.stride,
                    "known_labels": np.asarray(known_labels),
                    "centroids": centroids,
                    "thresholds": thresholds,
                }, run_dir / model_name)
                model_records.append({
                    "held_out_sequence": held_out,
                    "modalities": combo_name,
                    "window": window,
                    "model_file": model_name,
                    "projector_file": projector_name,
                })

                for quantile in quantiles:
                    threshold = thresholds[str(quantile)]
                    accepted = known_score <= threshold
                    accepted_accuracy = (
                        float(np.mean(known_prediction[accepted] == known_truth[accepted]))
                        if accepted.any() else 0.0
                    )
                    metric_rows.append({
                        "held_out_sequence": held_out,
                        "descriptor": args.descriptor,
                        "modalities": combo_name,
                        "window": window,
                        "threshold_quantile": quantile,
                        "threshold": threshold,
                        "known_samples": len(known_score),
                        "unknown_samples": len(unknown_score),
                        "known_classification_accuracy": float(np.mean(known_prediction == known_truth)),
                        "known_acceptance_rate": float(np.mean(accepted)),
                        "accepted_known_accuracy": accepted_accuracy,
                        "unknown_recall": float(np.mean(unknown_score > threshold)),
                        "known_false_unknown_rate": float(np.mean(known_score > threshold)),
                        "unknown_auroc": float(roc_auc_score(detection_labels, detection_scores)),
                        "unknown_auprc": float(average_precision_score(detection_labels, detection_scores)),
                        "fpr_at_95_tpr": false_positive_rate_at_95_tpr(detection_labels, detection_scores),
                    })
                    known_output = known_idx.copy()
                    known_output["held_out_sequence"] = held_out
                    known_output["is_unknown"] = 0
                    known_output["novelty_score"] = known_score
                    known_output["threshold_quantile"] = quantile
                    known_output["threshold"] = threshold
                    known_output["predicted_sequence"] = known_prediction
                    known_output["predicted_unknown"] = known_score > threshold
                    unknown_output = unknown_idx.copy()
                    unknown_output["held_out_sequence"] = held_out
                    unknown_output["is_unknown"] = 1
                    unknown_output["novelty_score"] = unknown_score
                    unknown_output["threshold_quantile"] = quantile
                    unknown_output["threshold"] = threshold
                    unknown_output["predicted_sequence"] = np.asarray(known_labels)[unknown_distances.argmin(axis=1)]
                    unknown_output["predicted_unknown"] = unknown_score > threshold
                    for output in (known_output, unknown_output):
                        output["modalities"] = combo_name
                        output["window"] = window
                        prediction_frames.append(output)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    summary = metrics.groupby(
        ["modalities", "window", "threshold_quantile"], as_index=False
    ).mean(numeric_only=True)
    summary.to_csv(run_dir / "summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(run_dir / "predictions.csv", index=False)
    atomic_write_json(run_dir / "models.json", model_records)
    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-scene-open-set/v2",
        "status": "complete",
        "created_utc": utc_now(),
        "descriptor_bank": str(bank.root),
        "arguments": vars(args),
        "evaluated_threshold_quantiles": quantiles,
        "protocol": "leave-one-scene-out; chronological fit/calibration split; all thresholds use calibration only; test used only for evaluation",
        "run_dir": str(run_dir),
    })
    print(summary.sort_values(
        ["unknown_auroc", "unknown_recall"], ascending=False
    ).head(30).to_string(index=False))
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
