"""Summarize discovery and return routing for multiple anonymous contexts."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


REQUIRED_COLUMNS = {
    "segment_index",
    "segment_name_for_metrics_only",
    "segment_local_step",
    "status",
    "active_context_id",
    "created_context_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multi-scene context discovery and frozen-mapping return routing."
        )
    )
    parser.add_argument("--discovery_csv", required=True)
    parser.add_argument("--return_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_labels", nargs="+", required=True)
    parser.add_argument("--base_label", default="seq5")
    parser.add_argument("--base_context_id", type=int, default=0)
    parser.add_argument("--ignore_transition_frames", type=int, default=0)
    parser.add_argument("--minimum_mapping_purity", type=float, default=0.0)
    parser.add_argument("--minimum_context_purity", type=float, default=0.95)
    parser.add_argument("--minimum_context_support", type=int, default=8)
    return parser.parse_args()


def _optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def _optional_flag(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, "")
    return False if value in ("", None) else bool(int(value))


def load_routing_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Routing CSV contains no frames: {path}")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise KeyError(f"Routing CSV is missing columns: {sorted(missing)}")
    output = []
    for row in rows:
        resolved = dict(row)
        resolved["segment_index"] = int(row["segment_index"])
        resolved["segment_local_step"] = int(row["segment_local_step"])
        resolved["active_context_id"] = int(row["active_context_id"])
        decision_id = _optional_int(row.get("decision_context_id"))
        if "decision_context_id" not in row:
            decision_id = (
                None if str(row["status"]) == "pending"
                else int(row["active_context_id"])
            )
        resolved["decision_context_id"] = decision_id
        resolved["decision_accepted"] = (
            _optional_flag(row, "decision_accepted")
            if "decision_accepted" in row
            else decision_id is not None
        )
        resolved["memory_updated"] = _optional_flag(row, "memory_updated")
        resolved["model_updated"] = _optional_flag(row, "model_updated")
        resolved["created_context_id"] = _optional_int(row["created_context_id"])
        resolved["segment_name_for_metrics_only"] = str(
            row["segment_name_for_metrics_only"]
        )
        resolved["status"] = str(row["status"])
        output.append(resolved)
    return output


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def validate_labels(
    discovery_rows: Sequence[Mapping[str, Any]],
    return_rows: Sequence[Mapping[str, Any]],
    expected_labels: Sequence[str],
    base_label: str,
) -> tuple[str, ...]:
    labels = tuple(str(label) for label in expected_labels)
    if not labels or any(not label for label in labels):
        raise ValueError("expected_labels must contain non-empty labels.")
    if len(set(labels)) != len(labels):
        raise ValueError("expected_labels must be unique.")
    if base_label not in labels:
        raise ValueError("base_label must be included in expected_labels.")
    observed = {
        str(row["segment_name_for_metrics_only"])
        for row in [*discovery_rows, *return_rows]
    }
    unknown = observed - set(labels)
    missing = set(labels) - observed
    if unknown:
        raise ValueError(f"Unexpected metric-only labels: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Expected labels absent from routing logs: {sorted(missing)}")
    return labels


def collect_creation_events(
    rows: Sequence[Mapping[str, Any]],
    phase: str,
) -> list[dict[str, Any]]:
    events = []
    for row in rows:
        context_id = row["created_context_id"]
        if context_id is None:
            continue
        events.append(
            {
                "phase": phase,
                "segment_index": int(row["segment_index"]),
                "label": str(row["segment_name_for_metrics_only"]),
                "local_step": int(row["segment_local_step"]),
                "context_id": int(context_id),
            }
        )
    return events


def build_creation_mapping(
    events: Sequence[Mapping[str, Any]],
    *,
    base_label: str,
    base_context_id: int,
) -> tuple[dict[int, str], dict[str, list[int]], list[dict[str, Any]]]:
    mapping = {int(base_context_id): str(base_label)}
    contexts_by_label: dict[str, list[int]] = defaultdict(list)
    contexts_by_label[str(base_label)].append(int(base_context_id))
    conflicts = []
    for event in events:
        context_id = int(event["context_id"])
        label = str(event["label"])
        existing = mapping.get(context_id)
        if existing is not None and existing != label:
            conflicts.append(
                {
                    "context_id": context_id,
                    "existing_label": existing,
                    "new_label": label,
                }
            )
            continue
        mapping[context_id] = label
        if context_id not in contexts_by_label[label]:
            contexts_by_label[label].append(context_id)
    return (
        mapping,
        {label: sorted(ids) for label, ids in contexts_by_label.items()},
        conflicts,
    )


def stable_rows(
    rows: Sequence[Mapping[str, Any]],
    ignore_transition_frames: int,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if int(row["segment_local_step"]) > ignore_transition_frames
    ]


def decision_context_id(row: Mapping[str, Any]) -> int | None:
    if "decision_context_id" in row:
        return _optional_int(row.get("decision_context_id"))
    if str(row.get("status", "")) == "pending":
        return None
    return int(row["active_context_id"])


def decision_accepted(row: Mapping[str, Any]) -> bool:
    if "decision_accepted" in row:
        return bool(row["decision_accepted"])
    return decision_context_id(row) is not None


def context_label_counts(
    rows: Iterable[Mapping[str, Any]],
    *,
    accepted_only: bool = True,
) -> dict[int, Counter[str]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        context_id = decision_context_id(row)
        if context_id is None and accepted_only:
            continue
        if context_id is None:
            context_id = int(row["active_context_id"])
        counts[int(context_id)][
            str(row["segment_name_for_metrics_only"])
        ] += 1
    return counts


def build_dominant_mapping(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_purity: float = 0.0,
) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    if not 0.0 <= minimum_purity <= 1.0:
        raise ValueError("minimum_purity must be in [0, 1].")
    counts = context_label_counts(rows)
    mapping: dict[int, str] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    for context_id, values in sorted(counts.items()):
        label, dominant_count = values.most_common(1)[0]
        total = sum(values.values())
        purity = safe_ratio(dominant_count, total)
        accepted = purity >= minimum_purity
        if accepted:
            mapping[context_id] = label
        diagnostics[context_id] = {
            "dominant_label": label,
            "dominant_frames": int(dominant_count),
            "frames": int(total),
            "purity": purity,
            "mapping_accepted": accepted,
            "label_counts": dict(sorted(values.items())),
        }
    return mapping, diagnostics


def build_hungarian_mapping(
    rows: Sequence[Mapping[str, Any]],
    expected_labels: Sequence[str],
) -> dict[int, str]:
    context_ids = sorted({int(row["active_context_id"]) for row in rows})
    if not context_ids:
        return {}
    matrix = np.zeros((len(expected_labels), len(context_ids)), dtype=np.int64)
    label_index = {label: index for index, label in enumerate(expected_labels)}
    context_index = {
        context_id: index for index, context_id in enumerate(context_ids)
    }
    for row in rows:
        matrix[
            label_index[str(row["segment_name_for_metrics_only"])],
            context_index[int(row["active_context_id"])],
        ] += 1
    candidates = sorted(
        (
            (int(matrix[label_index, context_index]), label_index, context_index)
            for label_index in range(len(expected_labels))
            for context_index in range(len(context_ids))
        ),
        reverse=True,
    )
    used_labels: set[int] = set()
    used_contexts: set[int] = set()
    mapping: dict[int, str] = {}
    for count, label_index, context_index in candidates:
        if count <= 0:
            break
        if label_index in used_labels or context_index in used_contexts:
            continue
        used_labels.add(label_index)
        used_contexts.add(context_index)
        mapping[context_ids[context_index]] = str(expected_labels[label_index])
    return mapping


def evaluate_phase(
    rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[int, str],
    ignore_transition_frames: int,
) -> dict[str, Any]:
    kept = stable_rows(rows, ignore_transition_frames)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    pending = 0
    for row in kept:
        truth = str(row["segment_name_for_metrics_only"])
        accepted = decision_accepted(row)
        context_id = decision_context_id(row)
        prediction = (
            mapping.get(context_id, "unmapped")
            if accepted and context_id is not None
            else "pending"
        )
        is_correct = prediction == truth
        correct += int(is_correct)
        pending += int(not accepted)
        confusion[truth][prediction] += 1
        per_label[truth]["frames"] += 1
        per_label[truth]["correct"] += int(is_correct)
        per_label[truth]["pending"] += int(not accepted)
        per_label[truth]["accepted"] += int(accepted)

    segments: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        segments[int(row["segment_index"])].append(row)
    switch_delays = []
    successful_segments = 0
    for segment_index in sorted(segments):
        segment = segments[segment_index]
        truth = str(segment[0]["segment_name_for_metrics_only"])
        first_correct = next(
            (
                int(row["segment_local_step"])
                for row in segment
                if decision_accepted(row)
                and mapping.get(decision_context_id(row), "unmapped") == truth
            ),
            None,
        )
        successful_segments += int(first_correct is not None)
        switch_delays.append(
            {
                "segment_index": segment_index,
                "label": truth,
                "frames": len(segment),
                "first_correct_local_step": first_correct,
                "switch_delay_frames": (
                    None if first_correct is None else first_correct - 1
                ),
            }
        )

    counts = context_label_counts(kept)
    accepted_frames = sum(sum(values.values()) for values in counts.values())
    purity_numerator = sum(max(values.values()) for values in counts.values() if values)
    _, purity_by_context = build_dominant_mapping(kept)
    per_label_accuracies = [
        safe_ratio(values["correct"], values["frames"])
        for values in per_label.values()
    ]
    memory_updates = [row for row in kept if bool(row.get("memory_updated", False))]
    model_updates = [row for row in kept if bool(row.get("model_updated", False))]

    def contamination(update_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        incorrect = sum(
            mapping.get(decision_context_id(row), "unmapped")
            != str(row["segment_name_for_metrics_only"])
            for row in update_rows
        )
        return {
            "updates": len(update_rows),
            "incorrect_updates": int(incorrect),
            "contamination_rate": safe_ratio(incorrect, len(update_rows)),
        }

    return {
        "total_frames": len(rows),
        "stable_frames": len(kept),
        "ignored_transition_frames_per_segment": ignore_transition_frames,
        "stable_frame_accuracy": safe_ratio(correct, len(kept)),
        "macro_return_compatible_accuracy": (
            float(np.mean(per_label_accuracies)) if per_label_accuracies else 0.0
        ),
        "minimum_label_accuracy": (
            float(min(per_label_accuracies)) if per_label_accuracies else 0.0
        ),
        "selective_accuracy": safe_ratio(correct, accepted_frames),
        "decision_coverage": safe_ratio(accepted_frames, len(kept)),
        "stable_pending_rate": safe_ratio(pending, len(kept)),
        "num_segments": len(segments),
        "successfully_routed_segments": successful_segments,
        "segment_routing_success_rate": safe_ratio(
            successful_segments, len(segments)
        ),
        "context_purity": safe_ratio(purity_numerator, accepted_frames),
        "merge_leakage": 1.0 - safe_ratio(purity_numerator, accepted_frames),
        "per_context_purity": purity_by_context,
        "memory_update_safety": contamination(memory_updates),
        "model_update_safety": contamination(model_updates),
        "per_label": {
            label: {
                "frames": int(values["frames"]),
                "correct": int(values["correct"]),
                "accuracy": safe_ratio(values["correct"], values["frames"]),
                "pending_rate": safe_ratio(values["pending"], values["frames"]),
                "decision_coverage": safe_ratio(
                    values["accepted"], values["frames"]
                ),
            }
            for label, values in sorted(per_label.items())
        },
        "confusion_matrix": {
            truth: dict(sorted(predictions.items()))
            for truth, predictions in sorted(confusion.items())
        },
        "context_label_counts": {
            context_id: dict(sorted(values.items()))
            for context_id, values in sorted(counts.items())
        },
        "switch_delays": switch_delays,
    }


def dominant_context_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    expected_labels: Sequence[str],
    ignore_transition_frames: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts_by_label: dict[str, Counter[int]] = defaultdict(Counter)
    for row in stable_rows(rows, ignore_transition_frames):
        counts_by_label[str(row["segment_name_for_metrics_only"])][
            int(row["active_context_id"])
        ] += 1
    dominant = {
        label: counts_by_label[label].most_common(1)[0][0]
        for label in expected_labels
        if counts_by_label[label]
    }
    labels_by_context: dict[int, list[str]] = defaultdict(list)
    for label, context_id in dominant.items():
        labels_by_context[context_id].append(label)
    merges = [
        {"context_id": context_id, "labels": sorted(labels)}
        for context_id, labels in sorted(labels_by_context.items())
        if len(labels) > 1
    ]
    return dominant, merges


def summarize(
    discovery_rows: Sequence[Mapping[str, Any]],
    return_rows: Sequence[Mapping[str, Any]],
    *,
    expected_labels: Sequence[str],
    base_label: str = "seq5",
    base_context_id: int = 0,
    ignore_transition_frames: int = 0,
    minimum_mapping_purity: float = 0.0,
    minimum_context_purity: float = 0.95,
    minimum_context_support: int = 8,
) -> dict[str, Any]:
    if ignore_transition_frames < 0:
        raise ValueError("ignore_transition_frames must be nonnegative.")
    if not 0.0 <= minimum_context_purity <= 1.0:
        raise ValueError("minimum_context_purity must be in [0, 1].")
    if minimum_context_support <= 0:
        raise ValueError("minimum_context_support must be positive.")
    labels = validate_labels(
        discovery_rows,
        return_rows,
        expected_labels,
        base_label,
    )
    discovery_events = collect_creation_events(discovery_rows, "discovery")
    return_events = collect_creation_events(return_rows, "return")
    creation_mapping, creation_contexts_by_label, mapping_conflicts = build_creation_mapping(
        discovery_events,
        base_label=base_label,
        base_context_id=base_context_id,
    )
    discovery_stable = stable_rows(discovery_rows, ignore_transition_frames)
    mapping, mapping_diagnostics = build_dominant_mapping(
        discovery_stable, minimum_purity=minimum_mapping_purity
    )
    contexts_by_label: dict[str, list[int]] = defaultdict(list)
    for context_id, label in mapping.items():
        contexts_by_label[label].append(context_id)
    discovered_labels = set(mapping.values())
    expected_new_labels = set(labels) - {base_label}
    discovered_new_labels = discovered_labels & expected_new_labels
    missing_new_labels = sorted(expected_new_labels - discovered_new_labels)
    hungarian_mapping = build_hungarian_mapping(discovery_stable, labels)
    discovery_metrics = evaluate_phase(
        discovery_rows,
        mapping,
        ignore_transition_frames,
    )
    return_metrics = evaluate_phase(
        return_rows,
        mapping,
        ignore_transition_frames,
    )
    hungarian_discovery = evaluate_phase(
        discovery_rows,
        hungarian_mapping,
        ignore_transition_frames,
    )
    hungarian_return = evaluate_phase(
        return_rows,
        hungarian_mapping,
        ignore_transition_frames,
    )

    all_events = [*discovery_events, *return_events]
    all_contexts_by_label: dict[str, set[int]] = defaultdict(set)
    all_contexts_by_label[base_label].add(int(base_context_id))
    for event in all_events:
        all_contexts_by_label[str(event["label"])].add(int(event["context_id"]))
    fragmentations = [
        {"label": label, "context_ids": sorted(context_ids)}
        for label, context_ids in sorted(all_contexts_by_label.items())
        if len(context_ids) > 1
    ]
    dominant, merges = dominant_context_diagnostics(
        discovery_rows,
        labels,
        ignore_transition_frames,
    )
    context_purity_violations = [
        {
            "context_id": int(context_id),
            "frames": int(values["frames"]),
            "purity": float(values["purity"]),
            "label_counts": dict(values["label_counts"]),
        }
        for context_id, values in mapping_diagnostics.items()
        if values["frames"] >= minimum_context_support
        and values["purity"] < minimum_context_purity
    ]

    discovery_context_ids = sorted(
        {int(row["active_context_id"]) for row in discovery_rows}
        | {int(event["context_id"]) for event in discovery_events}
    )
    return_context_ids = sorted(
        {int(row["active_context_id"]) for row in return_rows}
        | {int(event["context_id"]) for event in return_events}
    )
    primary_criteria = {
        "all_scenes_have_pure_discovery_context": not missing_new_labels
        and base_label in discovered_labels,
        "discovery_context_purity_at_least_0_95": (
            discovery_metrics["context_purity"] >= 0.95
        ),
        "no_dominant_context_merging": not merges,
        "all_supported_contexts_meet_purity": not context_purity_violations,
        "return_macro_accuracy_at_least_0_90": (
            return_metrics["macro_return_compatible_accuracy"] >= 0.90
        ),
        "minimum_scene_return_accuracy_at_least_0_80": (
            return_metrics["minimum_label_accuracy"] >= 0.80
        ),
        "memory_update_contamination_at_most_0_01": (
            discovery_metrics["memory_update_safety"]["contamination_rate"] <= 0.01
            and return_metrics["memory_update_safety"]["contamination_rate"] <= 0.01
        ),
        "model_update_contamination_at_most_0_01": (
            discovery_metrics["model_update_safety"]["contamination_rate"] <= 0.01
            and return_metrics["model_update_safety"]["contamination_rate"] <= 0.01
        ),
    }
    diagnostic_criteria = {
        "exactly_expected_contexts": len(discovery_context_ids) == len(labels),
        "no_false_creations_on_return": not return_events,
        "no_fragmentation": not fragmentations,
        "no_mapping_conflicts": not mapping_conflicts,
        "no_merging_diagnostic": not merges,
    }
    criteria = {
        **primary_criteria,
        "overall_pass": all(primary_criteria.values()),
        "diagnostics_not_used_for_overall_pass": diagnostic_criteria,
    }

    return {
        "expected_labels": list(labels),
        "base_label": base_label,
        "base_context_id": int(base_context_id),
        "expected_contexts": len(labels),
        "discovery_mapping_for_metrics_only": dict(sorted(mapping.items())),
        "creation_mapping_for_diagnostics_only": dict(
            sorted(creation_mapping.items())
        ),
        "creation_mapping_for_metrics_only": dict(sorted(mapping.items())),
        "mapping_minimum_purity": float(minimum_mapping_purity),
        "minimum_context_purity": float(minimum_context_purity),
        "minimum_context_support": int(minimum_context_support),
        "context_purity_violations": context_purity_violations,
        "mapping_diagnostics": mapping_diagnostics,
        "hungarian_mapping_for_diagnostics_only": dict(
            sorted(hungarian_mapping.items())
        ),
        "mapping_conflicts": mapping_conflicts,
        "contexts_by_label_after_discovery": {
            label: ids for label, ids in sorted(contexts_by_label.items())
        },
        "discovery": {
            **discovery_metrics,
            "num_contexts": len(discovery_context_ids),
            "context_ids": discovery_context_ids,
            "created_contexts": len(discovery_events),
            "expected_new_contexts": len(expected_new_labels),
            "new_scene_discovery_recall": safe_ratio(
                len(discovered_new_labels), len(expected_new_labels)
            ),
            "missing_new_labels": missing_new_labels,
            "extra_contexts": max(0, len(discovery_context_ids) - len(labels)),
            "hungarian_stable_frame_accuracy": hungarian_discovery[
                "stable_frame_accuracy"
            ],
        },
        "return": {
            **return_metrics,
            "num_contexts_seen": len(return_context_ids),
            "context_ids_seen": return_context_ids,
            "false_context_creations": len(return_events),
            "hungarian_stable_frame_accuracy": hungarian_return[
                "stable_frame_accuracy"
            ],
        },
        "fragmentations": fragmentations,
        "dominant_context_by_label_diagnostics_only": dict(sorted(dominant.items())),
        "merges": merges,
        "creation_events": all_events,
        "pass_criteria": criteria,
    }


def write_long_confusion(
    path: Path,
    matrix: Mapping[str, Mapping[str, int]],
) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["truth", "prediction", "count"])
        for truth, predictions in sorted(matrix.items()):
            for prediction, count in sorted(predictions.items()):
                writer.writerow([truth, prediction, int(count)])


def write_events(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    fields = ["phase", "segment_index", "label", "local_step", "context_id"]
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow({field: event.get(field, "") for field in fields})


def write_switch_delays(
    path: Path,
    discovery: Sequence[Mapping[str, Any]],
    returned: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "phase",
        "segment_index",
        "label",
        "frames",
        "first_correct_local_step",
        "switch_delay_frames",
    ]
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for phase, rows in (("discovery", discovery), ("return", returned)):
            for row in rows:
                writer.writerow({"phase": phase, **row})


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "routing_metrics.yml").open("w") as destination:
        yaml.safe_dump(dict(payload), destination, sort_keys=False)
    mapping_payload = {
        "for_metrics_only": True,
        "discovery_dominant_mapping": payload["discovery_mapping_for_metrics_only"],
        "creation_mapping_for_diagnostics_only": payload[
            "creation_mapping_for_diagnostics_only"
        ],
        "minimum_mapping_purity": payload["mapping_minimum_purity"],
        "hungarian_mapping_for_diagnostics_only": payload[
            "hungarian_mapping_for_diagnostics_only"
        ],
    }
    with (output_dir / "context_label_mapping.yml").open("w") as destination:
        yaml.safe_dump(mapping_payload, destination, sort_keys=False)
    write_long_confusion(
        output_dir / "discovery_confusion.csv",
        payload["discovery"]["confusion_matrix"],
    )
    write_long_confusion(
        output_dir / "return_confusion.csv",
        payload["return"]["confusion_matrix"],
    )
    write_events(output_dir / "creation_events.csv", payload["creation_events"])
    write_switch_delays(
        output_dir / "switch_delays.csv",
        payload["discovery"]["switch_delays"],
        payload["return"]["switch_delays"],
    )


def main() -> None:
    args = parse_args()
    discovery_path = Path(args.discovery_csv).expanduser().resolve()
    return_path = Path(args.return_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = summarize(
        load_routing_rows(discovery_path),
        load_routing_rows(return_path),
        expected_labels=args.expected_labels,
        base_label=str(args.base_label),
        base_context_id=int(args.base_context_id),
        ignore_transition_frames=int(args.ignore_transition_frames),
        minimum_mapping_purity=float(args.minimum_mapping_purity),
        minimum_context_purity=float(args.minimum_context_purity),
        minimum_context_support=int(args.minimum_context_support),
    )
    payload["discovery_csv"] = str(discovery_path)
    payload["return_csv"] = str(return_path)
    write_outputs(payload, output_dir)

    print(
        "Discovery contexts: "
        f"{payload['discovery']['num_contexts']}/{payload['expected_contexts']}"
    )
    print(
        "Discovery stable accuracy: "
        f"{payload['discovery']['stable_frame_accuracy']:.6f}"
    )
    print(
        "Return stable accuracy: "
        f"{payload['return']['stable_frame_accuracy']:.6f}"
    )
    print(
        "False creations on return: "
        f"{payload['return']['false_context_creations']}"
    )
    print(f"Metrics directory: {output_dir}")


if __name__ == "__main__":
    main()
