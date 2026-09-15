#!/usr/bin/env python3
"""Evaluate boundary-blind semantic context creation and reuse in three passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (
    PROJECT_ROOT,
    REPOSITORY_ROOT / "scene_discovery",
    REPOSITORY_ROOT / "scene_discovery_factorized_mlp",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from factorized_mlp.data import load_split_arrays  # noqa: E402
from factorized_mlp.illumination import (  # noqa: E402
    load_aligned_illumination_bank,
)
from online_context.controller import OnlineContextController  # noqa: E402
from online_context.evaluation import attach_decisions, summarize_run  # noqa: E402
from online_context.predictor import AttributePredictor  # noqa: E402
from online_context.stream import (  # noqa: E402
    build_three_pass_stream, materialize_stream_features,
)
from scene_discovery.attributes import attribute_frame  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/online_context_v1.yml",
    )
    return parser.parse_args()


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(part) for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(part) for part in item]
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return None if not np.isfinite(item) else float(item)
        return item
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(clean(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(target)


def make_run_dir(root):
    stamp = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")
    path = Path(root) / f"online_context_v1_{stamp}"
    suffix = 1
    while path.exists():
        path = Path(root) / f"online_context_v1_{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    (path / "development").mkdir()
    return path


def probabilities_at(probabilities, position):
    return {key: values[position] for key, values in probabilities.items()}


def run_positions(controller, probabilities, positions, decisions):
    for position in positions:
        decisions.append(controller.observe(
            probabilities_at(probabilities, int(position)), int(position)
        ))


def development_rank(row, expected_context_count):
    """Fixed lexicographic selection that never reads Test results."""
    def finite(value, fallback):
        return float(value) if np.isfinite(value) else fallback
    return (
        abs(int(row["final_context_count"]) - int(expected_context_count)),
        int(row["first_visit_duplicate_creations"]),
        int(row["second_visit_false_creations"]),
        -finite(row["first_visit_creation_recall"], -1.0),
        -finite(row["second_visit_scene_reuse_accuracy"], -1.0),
        -finite(row["change_f1"], -1.0),
        finite(row["mean_detection_delay"], float("inf")),
        int(row["decision_window"]),
    )


def main():
    args = parse_args()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream_file:
        config = yaml.safe_load(stream_file)
    run_dir = make_run_dir(Path(config["output_root"]).expanduser().resolve())
    predictor = AttributePredictor(config["checkpoint"], config.get("device", "auto"))
    descriptor_bank = Path(config["descriptor_bank"]).expanduser().resolve()
    arrays_by_split, indices = {}, {}
    for split in ("train", "test"):
        arrays_by_split[split], indices[split] = load_split_arrays(
            descriptor_bank, split, predictor.sensor_modalities, config["descriptor"]
        )
        indices[split] = indices[split].reset_index(drop=True)
        if "illumination" in predictor.modalities:
            illumination_bank = config.get("illumination_bank")
            if not illumination_bank:
                raise ValueError(
                    "spatial-lighting checkpoint requires illumination_bank"
                )
            arrays_by_split[split]["illumination"] = (
                load_aligned_illumination_bank(
                    Path(illumination_bank).expanduser().resolve(),
                    split, indices[split]
                )
            )
    support_manifest = pd.read_csv(
        Path(config["support_manifest"]).expanduser().resolve()
    )
    stream, orders, support_ids = build_three_pass_stream(
        indices["train"], indices["test"], support_manifest,
        config["sequences"], config["base_sequence"], config["stream_seed"],
    )
    stream.to_csv(run_dir / "stream_manifest.csv", index=False)
    write_json(run_dir / "stream_orders.json", orders)

    features = materialize_stream_features(stream, arrays_by_split, predictor.modalities)
    probabilities = predictor.predict_probabilities(
        features, batch_size=int(config["inference_batch_size"])
    )
    np.savez_compressed(run_dir / "attribute_probabilities.npz", **probabilities)
    base_rows = indices["train"][
        indices["train"]["sequence"].astype(str).eq(str(config["base_sequence"]))
    ].head(1)
    base_labels = attribute_frame(base_rows).iloc[0]
    base_key = (
        base_labels["weather"], base_labels["road"], base_labels["lighting"]
    )
    online, evaluation = config["online"], config["evaluation"]
    dev_positions = np.flatnonzero(~stream["phase"].eq("third_visit_test").to_numpy())
    test_positions = np.flatnonzero(stream["phase"].eq("third_visit_test").to_numpy())
    controllers, decision_sets, development_rows = {}, {}, []

    for decision_window in online["decision_windows"]:
        controller = OnlineContextController(
            predictor.label_names, base_key,
            decision_window=int(decision_window),
            confirmation_frames=int(online["confirmation_frames"]),
            majority_ratio=float(online["majority_ratio"]),
            new_context_confirmation_frames=int(online.get(
                "new_context_confirmation_frames",
                online["confirmation_frames"],
            )),
            new_context_majority_ratio=float(online.get(
                "new_context_majority_ratio", online["majority_ratio"]
            )),
            confidence_thresholds=online.get("confidence", {}),
            confidence_policy=str(online.get("confidence_policy", "observe_only")),
            pause_update_when_pending=bool(
                online.get("pause_update_when_pending", True)
            ),
        )
        decisions = []
        run_positions(controller, probabilities, dev_positions, decisions)
        dev_stream = stream.iloc[dev_positions].reset_index(drop=True)
        dev_frame = attach_decisions(dev_stream, decisions)
        summary, blocks, phases, matches = summarize_run(
            dev_frame, len(controller.registry),
            int(evaluation["stable_grace_frames"]),
            int(evaluation["change_tolerance_frames"]),
        )
        summary["decision_window"] = int(decision_window)
        development_rows.append(summary)
        directory = run_dir / "development" / f"window_{decision_window}"
        directory.mkdir()
        dev_frame.to_csv(directory / "frame_predictions.csv", index=False)
        blocks.to_csv(directory / "block_metrics.csv", index=False)
        phases.to_csv(directory / "phase_metrics.csv", index=False)
        matches.to_csv(directory / "change_matches.csv", index=False)
        write_json(directory / "summary.json", summary)
        write_json(directory / "context_registry.json", controller.registry.to_records())
        controllers[int(decision_window)] = controller
        decision_sets[int(decision_window)] = decisions

    comparison = pd.DataFrame(development_rows)
    comparison.to_csv(run_dir / "window_comparison_train.csv", index=False)
    expected_count = int(evaluation["expected_context_count"])
    selected_row = min(
        development_rows, key=lambda row: development_rank(row, expected_count)
    )
    selected_window = int(selected_row["decision_window"])
    controller = controllers[selected_window]
    decisions = decision_sets[selected_window]
    run_positions(controller, probabilities, test_positions, decisions)
    full_frame = attach_decisions(stream, decisions)
    summary, blocks, phases, matches = summarize_run(
        full_frame, len(controller.registry),
        int(evaluation["stable_grace_frames"]),
        int(evaluation["change_tolerance_frames"]),
    )
    summary.update({
        "selected_decision_window": selected_window,
        "selection_used_test": False,
        "confidence_policy": str(online["confidence_policy"]),
        "support_samples_excluded": len(support_ids),
    })
    full_frame.to_csv(run_dir / "frame_predictions.csv", index=False)
    blocks.to_csv(run_dir / "block_metrics.csv", index=False)
    phases.to_csv(run_dir / "phase_metrics.csv", index=False)
    matches.to_csv(run_dir / "change_matches.csv", index=False)
    full_frame[full_frame["event"].isin(["create", "switch"])].to_csv(
        run_dir / "context_events.csv", index=False
    )
    write_json(run_dir / "context_registry.json", controller.registry.to_records())
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "run_meta.json", {
        "schema_version": "kradar-semantic-online-context/v1",
        "status": "complete", "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "config_path": str(config_path), "config": config,
        "checkpoint": str(predictor.checkpoint_path),
        "checkpoint_metadata": predictor.checkpoint_metadata,
        "selected_decision_window": selected_window,
        "selection_used_test": False,
        "controller_input_excludes_truth_and_boundaries": True,
        "run_dir": str(run_dir),
    })
    print("Train-only decision-window comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSelected decision window: {selected_window}")
    print("\nFull three-pass summary:")
    print(json.dumps(summary, indent=2))
    print("\nPhase metrics:")
    print(phases.to_string(index=False))
    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
