"""Summarize context routing across arbitrary time-window segments."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize base-versus-target context routing accuracy."
    )
    parser.add_argument("--routing_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_label", default="normal")
    parser.add_argument("--target_label", default="seq58")
    parser.add_argument("--base_context_id", type=int, default=0)
    parser.add_argument("--ignore_transition_frames", type=int, default=0)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("The routing CSV contains no frames.")
    required = {
        "segment_index",
        "segment_name_for_metrics_only",
        "segment_local_step",
        "status",
        "active_context_id",
        "created_context_id",
    }
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Routing CSV is missing columns: {sorted(missing)}")
    for row in rows:
        row["segment_index"] = int(row["segment_index"])
        row["segment_local_step"] = int(row["segment_local_step"])
        row["active_context_id"] = int(row["active_context_id"])
    return rows


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    args = parse_args()
    if args.ignore_transition_frames < 0:
        raise ValueError("ignore_transition_frames must be nonnegative.")
    rows = load_rows(Path(args.routing_csv))
    valid_labels = {args.base_label, args.target_label}
    observed_labels = {
        str(row["segment_name_for_metrics_only"]) for row in rows
    }
    unknown_labels = observed_labels - valid_labels
    if unknown_labels:
        raise ValueError(f"Unexpected metric-only labels: {sorted(unknown_labels)}")

    context_label_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        context_label_counts[row["active_context_id"]][
            str(row["segment_name_for_metrics_only"])
        ] += 1

    target_candidates = [
        context_id
        for context_id in context_label_counts
        if context_id != args.base_context_id
    ]
    target_context_id = None
    if target_candidates:
        target_context_id = max(
            target_candidates,
            key=lambda context_id: (
                context_label_counts[context_id][args.target_label],
                -context_id,
            ),
        )

    strict_mapping: dict[int, str] = {
        args.base_context_id: args.base_label,
    }
    if target_context_id is not None:
        strict_mapping[target_context_id] = args.target_label
    majority_mapping = {
        context_id: counts.most_common(1)[0][0]
        for context_id, counts in context_label_counts.items()
    }

    strict_correct = 0
    majority_correct = 0
    stable_correct = 0
    stable_total = 0
    pending_frames = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    segments: dict[int, list[dict[str, Any]]] = defaultdict(list)
    creation_events = []
    for row in rows:
        truth = str(row["segment_name_for_metrics_only"])
        context_id = row["active_context_id"]
        strict_prediction = strict_mapping.get(context_id, "unmapped")
        majority_prediction = majority_mapping[context_id]
        is_strict_correct = strict_prediction == truth
        strict_correct += int(is_strict_correct)
        majority_correct += int(majority_prediction == truth)
        confusion[truth][strict_prediction] += 1
        if row["segment_local_step"] > args.ignore_transition_frames:
            stable_total += 1
            stable_correct += int(is_strict_correct)
        if row["status"] == "pending":
            pending_frames += 1
        segments[row["segment_index"]].append(row)
        if row["created_context_id"] not in ("", None):
            creation_events.append(
                {
                    "segment_index": row["segment_index"],
                    "label": truth,
                    "local_step": row["segment_local_step"],
                    "context_id": int(row["created_context_id"]),
                }
            )

    switch_delays = []
    for segment_index in sorted(segments):
        segment_rows = segments[segment_index]
        truth = str(segment_rows[0]["segment_name_for_metrics_only"])
        first_correct = next(
            (
                row["segment_local_step"]
                for row in segment_rows
                if strict_mapping.get(row["active_context_id"], "unmapped") == truth
            ),
            None,
        )
        switch_delays.append(
            {
                "segment_index": segment_index,
                "label": truth,
                "frames": len(segment_rows),
                "first_correct_local_step": first_correct,
                "switch_delay_frames": (
                    None if first_correct is None else first_correct - 1
                ),
            }
        )

    first_segment_for_label: dict[str, int] = {}
    returning_segment_ids: set[int] = set()
    for segment_index in sorted(segments):
        label = str(segments[segment_index][0]["segment_name_for_metrics_only"])
        if label in first_segment_for_label:
            returning_segment_ids.add(segment_index)
        else:
            first_segment_for_label[label] = segment_index
    false_return_creations = sum(
        event["segment_index"] in returning_segment_ids
        for event in creation_events
    )
    successful_delays = [
        item["switch_delay_frames"]
        for item in switch_delays
        if item["switch_delay_frames"] is not None
    ]
    context_ids = sorted(context_label_counts)
    expected_contexts = len(observed_labels)
    payload = {
        "routing_csv": str(Path(args.routing_csv).resolve()),
        "total_frames": len(rows),
        "strict_frame_accuracy": safe_ratio(strict_correct, len(rows)),
        "stable_frame_accuracy": safe_ratio(stable_correct, stable_total),
        "stable_frames": stable_total,
        "ignored_transition_frames_per_segment": args.ignore_transition_frames,
        "majority_mapping_accuracy": safe_ratio(majority_correct, len(rows)),
        "pending_rate": safe_ratio(pending_frames, len(rows)),
        "num_contexts": len(context_ids),
        "expected_contexts": expected_contexts,
        "extra_contexts": max(0, len(context_ids) - expected_contexts),
        "num_segments": len(segments),
        "num_scene_transitions": max(0, len(segments) - 1),
        "successfully_routed_segments": len(successful_delays),
        "segment_routing_success_rate": safe_ratio(
            len(successful_delays), len(segments)
        ),
        "mean_switch_delay_frames": (
            None
            if not successful_delays
            else float(sum(successful_delays) / len(successful_delays))
        ),
        "base_context_id": args.base_context_id,
        "target_context_id": target_context_id,
        "strict_mapping": strict_mapping,
        "majority_mapping": majority_mapping,
        "context_label_counts": {
            context_id: dict(context_label_counts[context_id])
            for context_id in context_ids
        },
        "confusion_matrix": {
            truth: dict(counts) for truth, counts in confusion.items()
        },
        "creation_events": creation_events,
        "returning_segment_ids": sorted(returning_segment_ids),
        "false_context_creations_on_return": int(false_return_creations),
        "switch_delays": switch_delays,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as destination:
        yaml.safe_dump(payload, destination, sort_keys=False)

    print(f"Strict frame accuracy: {payload['strict_frame_accuracy']:.6f}")
    print(f"Stable frame accuracy: {payload['stable_frame_accuracy']:.6f}")
    print(f"Majority mapping accuracy: {payload['majority_mapping_accuracy']:.6f}")
    print(
        f"Contexts: {payload['num_contexts']} "
        f"(expected {payload['expected_contexts']})"
    )
    print(
        f"Segment routing success: "
        f"{payload['successfully_routed_segments']}/{payload['num_segments']}"
    )
    print(f"Pending rate: {payload['pending_rate']:.6f}")
    print(f"False creations on return: {payload['false_context_creations_on_return']}")
    print(f"Metrics: {output}")


if __name__ == "__main__":
    main()
