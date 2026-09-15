#!/usr/bin/env python3
"""Decode independent attribute probabilities into known or novel contexts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.attribute_protocol import (  # noqa: E402
    align_attribute_predictions,
    sequence_balanced_metrics,
)
from scene_discovery.attributes import ATTRIBUTES  # noqa: E402
from scene_discovery.common import (  # noqa: E402
    atomic_write_json,
    make_run_dir,
    utc_now,
)
from scene_discovery.semantic_decoder import (  # noqa: E402
    candidate_score_frame,
    decode_semantic_contexts,
    prepare_semantic_registry,
    score_known_combinations,
)
from scene_discovery.visualization import save_confusion_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        required=True,
        help="Completed evaluate_independent_attributes.py result directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/SemanticDecoder",
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=None,
        help="Optional scene-attribute table; defaults to support_matrix.csv.",
    )
    parser.add_argument(
        "--attribute-weights",
        nargs=3,
        type=float,
        metavar=("WEATHER", "ROAD", "LIGHTING"),
        default=[1.0, 1.0, 1.0],
    )
    parser.add_argument("--min-known-score", type=float, default=0.0)
    parser.add_argument("--min-known-margin", type=float, default=0.0)
    parser.add_argument(
        "--ignore-attribute-acceptance",
        action="store_true",
        help="Do not use validation-calibrated per-attribute acceptance flags.",
    )
    parser.add_argument(
        "--focus-sequences", nargs="+", default=["38", "46", "58"]
    )
    parser.add_argument("--epsilon", type=float, default=1e-12)
    return parser.parse_args()


def registry_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename = {}
    if "sequence" not in frame and "held_out_sequence" in frame:
        rename["held_out_sequence"] = "sequence"
    for attribute in ATTRIBUTES:
        if attribute not in frame and f"true_{attribute}" in frame:
            rename[f"true_{attribute}"] = attribute
    return frame.rename(columns=rename)


def load_and_align(predictions_root: Path) -> pd.DataFrame:
    combined_path = predictions_root / "predictions.csv"
    if combined_path.is_file():
        combined = pd.read_csv(combined_path)
        combined["sequence"] = combined["sequence"].astype(str)
        combined["held_out_sequence"] = combined["sequence"]
        return combined

    frames = {}
    for attribute in ATTRIBUTES:
        path = predictions_root / attribute / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {attribute} predictions: {path}")
        frames[attribute] = pd.read_csv(path)
        frames[attribute]["held_out_sequence"] = frames[attribute][
            "held_out_sequence"
        ].astype(str)
        frames[attribute]["sequence"] = frames[attribute]["sequence"].astype(str)
    return align_attribute_predictions(frames)


def method_metrics(
    predictions: pd.DataFrame,
    prediction_column: str,
    method: str,
) -> dict:
    truth = predictions["true_semantic_key"].astype(str).to_numpy()
    predicted = predictions[prediction_column].astype(str).to_numpy()
    sequences = predictions["held_out_sequence"].astype(str).to_numpy()
    accepted = predicted != "unknown"
    correct = predicted == truth
    metrics = sequence_balanced_metrics(truth, predicted, sequences)
    metrics.update({
        "method": method,
        "coverage": float(accepted.mean()),
        "accepted_accuracy": (
            float(correct[accepted].mean()) if accepted.any() else 0.0
        ),
        "unknown_rate": float((~accepted).mean()),
    })
    return metrics


def per_sequence_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sequence, frame in predictions.groupby("held_out_sequence", sort=False):
        hard_correct = frame["hard_semantic_key"].eq(frame["true_semantic_key"])
        constrained_correct = frame["constrained_semantic_key"].eq(
            frame["true_semantic_key"]
        )
        accepted = frame["known_accepted"].astype(bool)
        open_correct = frame["open_set_semantic_key"].eq(
            frame["true_semantic_key"]
        )
        rows.append({
            "sequence": str(sequence),
            "true_semantic_key": frame["true_semantic_key"].iloc[0],
            "samples": len(frame),
            "hard_accuracy": float(hard_correct.mean()),
            "constrained_accuracy": float(constrained_correct.mean()),
            "rescued_fraction": float((~hard_correct & constrained_correct).mean()),
            "damaged_fraction": float((hard_correct & ~constrained_correct).mean()),
            "open_set_accuracy": float(open_correct.mean()),
            "known_coverage": float(accepted.mean()),
            "accepted_accuracy": (
                float(open_correct[accepted].mean()) if accepted.any() else 0.0
            ),
            "mean_known_score": float(frame["known_score"].mean()),
            "mean_known_margin": float(frame["known_margin"].mean()),
        })
    return pd.DataFrame(rows)


def save_confusions(predictions: pd.DataFrame, run_dir: Path) -> None:
    methods = {
        "hard": "hard_semantic_key",
        "constrained": "constrained_semantic_key",
        "open_set": "open_set_semantic_key",
    }
    for method, column in methods.items():
        matrix = pd.crosstab(
            predictions["true_semantic_key"],
            predictions[column],
            margins=True,
        )
        matrix.to_csv(run_dir / f"confusion_{method}_counts.csv")
        normalized = pd.crosstab(
            predictions["true_semantic_key"],
            predictions[column],
            normalize="index",
        )
        normalized.to_csv(run_dir / f"confusion_{method}_normalized.csv")
        save_confusion_plot(
            predictions["true_semantic_key"],
            predictions[column],
            run_dir / f"confusion_{method}.png",
            f"Semantic decoder: {method}",
        )


def novelty_diagnostic(
    predictions: pd.DataFrame,
    registry: pd.DataFrame,
    weights,
    epsilon: float,
) -> pd.DataFrame:
    """Score pseudo-novel samples after removing their true key from registry."""
    pieces = []
    for true_key, frame in predictions.groupby("true_semantic_key", sort=False):
        reduced = registry[~registry["semantic_key"].eq(true_key)].reset_index(drop=True)
        if reduced.empty:
            continue
        decoded, _ = score_known_combinations(frame, reduced, weights, epsilon)
        decoded.index = frame.index
        pieces.append(decoded[["known_score", "known_margin"]].rename(columns={
            "known_score": "pseudo_novel_best_score",
            "known_margin": "pseudo_novel_best_margin",
        }))
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces).sort_index()
    result.insert(0, "true_semantic_key", predictions.loc[result.index, "true_semantic_key"])
    result.insert(0, "held_out_sequence", predictions.loc[result.index, "held_out_sequence"])
    return result.reset_index(drop=True)


def diagnostic_sweep(
    predictions: pd.DataFrame,
    pseudo_novel: pd.DataFrame,
) -> pd.DataFrame:
    if pseudo_novel.empty:
        return pd.DataFrame()
    score_thresholds = np.linspace(0.0, 1.0, 21)
    margin_thresholds = np.linspace(0.0, 0.5, 11)
    rows = []
    known_correct = predictions["constrained_semantic_key"].eq(
        predictions["true_semantic_key"]
    ).to_numpy()
    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:
            known_accept = (
                predictions["known_score"].ge(score_threshold)
                & predictions["known_margin"].ge(margin_threshold)
            ).to_numpy()
            novel_accept = (
                pseudo_novel["pseudo_novel_best_score"].ge(score_threshold)
                & pseudo_novel["pseudo_novel_best_margin"].ge(margin_threshold)
            ).to_numpy()
            rows.append({
                "minimum_score": float(score_threshold),
                "minimum_margin": float(margin_threshold),
                "known_accept_rate": float(known_accept.mean()),
                "known_correct_and_accepted_rate": float(
                    (known_accept & known_correct).mean()
                ),
                "pseudo_novel_reject_rate": float((~novel_accept).mean()),
                "balanced_detection_accuracy": float(
                    0.5 * (known_accept.mean() + (~novel_accept).mean())
                ),
            })
    return pd.DataFrame(rows).sort_values(
        ["balanced_detection_accuracy", "known_correct_and_accepted_rate"],
        ascending=False,
    )


def main() -> None:
    args = parse_args()
    predictions_root = args.predictions_root.expanduser().resolve()
    registry_path = (
        args.registry_csv.expanduser().resolve()
        if args.registry_csv is not None
        else (
            predictions_root / "support_matrix.csv"
            if (predictions_root / "support_matrix.csv").is_file()
            else predictions_root / "scene_attributes.csv"
        )
    )
    if not registry_path.is_file():
        raise FileNotFoundError(f"Semantic registry source does not exist: {registry_path}")

    weights = dict(zip(ATTRIBUTES, args.attribute_weights))
    registry = prepare_semantic_registry(registry_source(registry_path))
    aligned = load_and_align(predictions_root)
    aligned["hard_semantic_key"] = aligned["predicted_semantic_key"]
    aligned["hard_registry_known"] = aligned["hard_semantic_key"].isin(
        set(registry["semantic_key"])
    )
    decoded, scores = decode_semantic_contexts(
        aligned,
        registry,
        weights,
        minimum_score=args.min_known_score,
        minimum_margin=args.min_known_margin,
        require_attribute_acceptance=not args.ignore_attribute_acceptance,
        epsilon=args.epsilon,
    )
    output = pd.concat(
        [aligned.reset_index(drop=True), decoded.reset_index(drop=True)], axis=1
    )
    output["hard_correct"] = output["hard_semantic_key"].eq(
        output["true_semantic_key"]
    )
    output["constrained_correct"] = output["constrained_semantic_key"].eq(
        output["true_semantic_key"]
    )
    output["open_set_correct"] = output["open_set_semantic_key"].eq(
        output["true_semantic_key"]
    )
    output["decoder_effect"] = "unchanged_wrong"
    output.loc[output["hard_correct"] & output["constrained_correct"], "decoder_effect"] = "unchanged_correct"
    output.loc[~output["hard_correct"] & output["constrained_correct"], "decoder_effect"] = "rescued"
    output.loc[output["hard_correct"] & ~output["constrained_correct"], "decoder_effect"] = "damaged"

    run_dir = make_run_dir(args.output_root.expanduser().resolve(), "semantic_decoder")
    registry.to_csv(run_dir / "semantic_registry.csv", index=False)
    output.to_csv(run_dir / "predictions.csv", index=False)
    candidate_score_frame(aligned, registry, scores).to_csv(
        run_dir / "candidate_scores.csv", index=False
    )

    metric_rows = [
        method_metrics(output, "hard_semantic_key", "hard_top1_tuple"),
        method_metrics(output, "constrained_semantic_key", "constrained_map"),
        method_metrics(output, "open_set_semantic_key", "open_set_constrained"),
    ]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    scene_metrics = per_sequence_metrics(output)
    scene_metrics.to_csv(run_dir / "per_sequence_metrics.csv", index=False)
    focus = {
        (str(value).strip()[3:] if str(value).strip().lower().startswith("seq") else str(value).strip())
        for value in args.focus_sequences
    }
    scene_metrics[
        scene_metrics["sequence"].astype(str).isin(focus)
    ].to_csv(run_dir / "focus_sequence_metrics.csv", index=False)
    output["decoder_effect"].value_counts().rename_axis(
        "decoder_effect"
    ).reset_index(name="samples").to_csv(run_dir / "decoder_effects.csv", index=False)
    save_confusions(output, run_dir)

    pseudo_novel = novelty_diagnostic(output, registry, weights, args.epsilon)
    pseudo_novel.to_csv(run_dir / "pseudo_novel_predictions.csv", index=False)
    sweep = diagnostic_sweep(output, pseudo_novel)
    sweep.to_csv(run_dir / "open_set_diagnostic_sweep.csv", index=False)

    atomic_write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-semantic-decoder/v1",
        "status": "complete",
        "created_utc": utc_now(),
        "predictions_root": str(predictions_root),
        "registry_source": str(registry_path),
        "attribute_weights": weights,
        "minimum_known_score": args.min_known_score,
        "minimum_known_margin": args.min_known_margin,
        "require_attribute_acceptance": not args.ignore_attribute_acceptance,
        "score": "weighted geometric mean of per-attribute class probabilities",
        "candidate_policy": "maximum score over registered semantic tuples",
        "unknown_policy": "score, margin, and optional calibrated attribute acceptance",
        "pseudo_novel_warning": (
            "Diagnostic only: the true tuple is removed from the registry on the same "
            "evaluation data; sweep thresholds are not deployment-calibrated."
        ),
        "sequence_id_policy": (
            "sequence is used only for reporting; decoding keys are weather|road|lighting"
        ),
        "run_dir": str(run_dir),
    })

    print("Semantic decoder summary:")
    print(metrics.to_string(index=False))
    print("\nPer-sequence summary:")
    print(scene_metrics.to_string(index=False))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
