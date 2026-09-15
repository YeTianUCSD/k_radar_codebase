#!/usr/bin/env python3
"""Replay automatic context routing on cached, unprojected descriptors."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_adaptation.bandwidth import resolve_bandwidth
from context_adaptation.runtime import build_context_memory, build_projector, build_router


DESCRIPTOR_COLUMN = "unprojected_descriptor_json"
OUTPUT_FIELDS = [
    "step_idx", "phase", "segment_index", "segment_name_for_metrics_only",
    "segment_local_step", "status", "reason", "decision_context_id",
    "decision_accepted", "active_context_id", "active_context_lifecycle",
    "best_context_id", "best_score", "second_score", "best_confidence",
    "second_confidence", "historical_evidence", "novelty_strength",
    "novelty_evidence", "candidate_evidence", DESCRIPTOR_COLUMN,
    "raw_projected_feature_json", "smoothed_projected_feature_json",
    "scores_json", "density_thresholds_json", "routing_thresholds_json",
    "near_thresholds_json", "update_thresholds_json", "created_context_id",
    "created_context_name", "retired_context_id", "context_promoted", "model_updated",
    "memory_updated",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context_config", type=Path, required=True)
    parser.add_argument("--discovery_csv", type=Path, required=True)
    parser.add_argument("--return_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_label", default="seq5")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--projection_dim", type=int)
    parser.add_argument("--projection_seed", type=int)
    parser.add_argument("--bandwidth_mode", choices=("fixed", "median"))
    parser.add_argument("--bandwidth_scale", type=float)
    parser.add_argument(
        "--disable_return_creation", action="store_true",
        help="Reject novel Return candidates instead of creating contexts.",
    )
    return parser.parse_args()


def load_descriptor_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Descriptor CSV contains no rows: {path}")
    required = {
        DESCRIPTOR_COLUMN, "segment_index", "segment_name_for_metrics_only",
        "segment_local_step",
    }
    missing = required - set(rows[0])
    if missing:
        raise KeyError(
            f"Descriptor CSV {path} is missing columns: {sorted(missing)}. "
            "Generate it with the descriptor-cache-enabled online router."
        )
    result = []
    descriptor_dim = None
    for index, row in enumerate(rows, 1):
        raw = row.get(DESCRIPTOR_COLUMN, "").strip()
        if not raw:
            raise ValueError(f"Missing descriptor at row {index} in {path}")
        descriptor = np.asarray(json.loads(raw), dtype=np.float32)
        if descriptor.ndim != 1 or descriptor.size == 0:
            raise ValueError(f"Invalid descriptor shape at row {index}: {descriptor.shape}")
        if not np.all(np.isfinite(descriptor)):
            raise ValueError(f"Non-finite descriptor at row {index} in {path}")
        descriptor_dim = descriptor.size if descriptor_dim is None else descriptor_dim
        if descriptor.size != descriptor_dim:
            raise ValueError("Descriptor dimension changed within a CSV.")
        result.append({
            "segment_index": int(row["segment_index"]),
            "segment_name_for_metrics_only": str(row["segment_name_for_metrics_only"]),
            "segment_local_step": int(row["segment_local_step"]),
            "descriptor": descriptor,
        })
    return result


def bootstrap_runtime(
    config: Mapping[str, Any],
    discovery_rows: Sequence[Mapping[str, Any]],
    base_label: str,
):
    normal = np.stack([
        row["descriptor"] for row in discovery_rows
        if row["segment_name_for_metrics_only"] == base_label
    ])
    if normal.shape[0] < 2:
        raise ValueError(f"Need at least two {base_label} descriptors for bootstrap.")
    projector = build_projector(config)
    projector.update_normalization(normal)
    projector.finalize()
    projected = projector.transform_descriptor(normal)
    bandwidth, bandwidth_info = resolve_bandwidth(config["KDE"], projected)
    memory = build_context_memory(
        config, input_dim=projector.projection_dim, bandwidth_override=bandwidth
    )
    base = config.get("BASE_CONTEXT", {})
    entry = memory.create_context(
        projected,
        context_name=str(base.get("CONTEXT_NAME", "normal")),
        model_context_name=str(base.get("MODEL_CONTEXT_NAME", base_label)),
        model_update_enabled=False,
        lifecycle="stable",
    )
    router = build_router(
        config, input_dim=projector.projection_dim,
        candidate_bandwidth_override=bandwidth,
    )
    router.set_active_context(entry.context_id)
    return projector, memory, router, normal.shape[0], bandwidth_info


def route_phase(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    projector: Any,
    memory: Any,
    router: Any,
    config: Mapping[str, Any],
    start_step: int,
    allow_context_creation: bool,
    max_steps: int,
) -> tuple[list[dict[str, Any]], int]:
    update_memory = bool(config.get("ONLINE", {}).get("UPDATE_MEMORY_IF_CONFIDENT", True))
    update_scale = float(config.get("ROUTER", {}).get("UPDATE_THRESHOLD_SCALE", 1.0))
    output = []
    step = start_step
    for source in rows:
        if max_steps > 0 and len(output) >= max_steps:
            break
        step += 1
        descriptor = np.asarray(source["descriptor"], dtype=projector.dtype)
        raw_x = projector.transform_descriptor(descriptor)
        x = router.prepare_feature(raw_x)
        scores = memory.score_all(x)
        density = memory.density_thresholds()
        routing = memory.routing_thresholds(router.routing_threshold_scale)
        near = memory.near_thresholds(router.near_threshold_scale)
        update_thresholds = memory.update_thresholds(update_scale)
        decision = router.step(
            x, scores, density, routing_thresholds=routing,
            near_thresholds=near, context_lifecycles=memory.context_lifecycles(),
            allow_context_creation=allow_context_creation,
        )
        created_id = None
        created_name = None
        promoted = False
        memory_updated = False
        retired_id = None
        if decision.status == "create":
            if decision.candidate_features is None:
                raise RuntimeError("Create decision has no candidate features.")
            entry = memory.create_context(
                decision.candidate_features,
                created_step=step,
                activate=True,
                model_update_enabled=True,
                lifecycle="provisional",
            )
            router.confirm_created_context(entry.context_id)
            context_id = entry.context_id
            created_id = entry.context_id
            created_name = entry.context_name
        else:
            if decision.context_id is None:
                raise RuntimeError(f"{decision.status} decision has no context ID.")
            context_id = int(decision.context_id)
        entry = memory.activate(context_id, step=step)
        if router.active_context_id != context_id:
            raise RuntimeError("Router and Context Memory active IDs diverged.")
        decision = router.apply_update_gate(
            decision,
            context_id=context_id,
            lifecycle=entry.lifecycle,
            update_threshold=update_thresholds.get(context_id),
        )

        should_update = decision.should_update_memory and update_memory and context_id in scores
        if should_update:
            if entry.is_provisional:
                promoted = memory.update_provisional(
                    context_id, x, step=step, pre_update_score=scores[context_id]
                )
                memory_updated = True
            else:
                memory_updated = memory.update_stable_if_confident(
                    context_id, x, threshold_scale=update_scale,
                    step=step, pre_update_score=scores[context_id],
                )
        elif entry.is_provisional and decision.reason == "provisional_context_mismatch":
            memory.mark_provisional_mismatch(context_id)
            if memory.should_retire(context_id):
                retired_id = context_id
                fallback_ids = [
                    candidate_id for candidate_id in scores
                    if candidate_id != context_id
                ]
                fallback_id = max(
                    fallback_ids,
                    key=lambda candidate_id: scores[candidate_id]
                    / max(near[candidate_id], 1e-12),
                )
                memory.retire_context(
                    context_id, fallback_context_id=fallback_id
                )
                router.retire_context(
                    context_id, fallback_context_id=fallback_id
                )
                context_id = fallback_id
                entry = memory.get(context_id)

        accepted = decision.status != "pending"
        output.append({
            "step_idx": step,
            "phase": phase,
            "segment_index": source["segment_index"],
            "segment_name_for_metrics_only": source["segment_name_for_metrics_only"],
            "segment_local_step": source["segment_local_step"],
            "status": decision.status,
            "reason": decision.reason,
            "decision_context_id": context_id if accepted else None,
            "decision_accepted": int(accepted),
            "active_context_id": context_id,
            "active_context_lifecycle": entry.lifecycle,
            "best_context_id": decision.best_context_id,
            "best_score": decision.best_score,
            "second_score": decision.second_score,
            "best_confidence": decision.best_confidence,
            "second_confidence": decision.second_confidence,
            "historical_evidence": decision.historical_evidence,
            "novelty_strength": decision.novelty_strength,
            "novelty_evidence": decision.novelty_evidence,
            "candidate_evidence": decision.candidate_evidence,
            DESCRIPTOR_COLUMN: json.dumps(descriptor.tolist()),
            "raw_projected_feature_json": json.dumps(np.asarray(raw_x).tolist()),
            "smoothed_projected_feature_json": json.dumps(np.asarray(x).tolist()),
            "scores_json": json.dumps(scores, sort_keys=True),
            "density_thresholds_json": json.dumps(density, sort_keys=True),
            "routing_thresholds_json": json.dumps(routing, sort_keys=True),
            "near_thresholds_json": json.dumps(near, sort_keys=True),
            "update_thresholds_json": json.dumps(update_thresholds, sort_keys=True),
            "created_context_id": created_id,
            "created_context_name": created_name,
            "retired_context_id": retired_id,
            "context_promoted": int(promoted),
            "model_updated": 0,
            "memory_updated": int(memory_updated),
        })
    return output, step


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def main() -> None:
    args = parse_args()
    config_path = args.context_config.expanduser().resolve()
    with config_path.open() as source:
        config = copy.deepcopy(yaml.safe_load(source))
    if args.projection_dim is not None:
        config["PROJECTION"]["DIM"] = int(args.projection_dim)
    if args.projection_seed is not None:
        config["PROJECTION"]["SEED"] = int(args.projection_seed)
    if args.bandwidth_mode is not None:
        config["KDE"]["BANDWIDTH_MODE"] = str(args.bandwidth_mode)
    if args.bandwidth_scale is not None:
        config["KDE"]["BANDWIDTH_SCALE"] = float(args.bandwidth_scale)
    discovery_source = load_descriptor_rows(args.discovery_csv.expanduser().resolve())
    return_source = load_descriptor_rows(args.return_csv.expanduser().resolve())
    projector, memory, router, bootstrap_samples, bandwidth_info = bootstrap_runtime(
        config, discovery_source, str(args.base_label)
    )
    discovery, step = route_phase(
        discovery_source, phase="discovery", projector=projector, memory=memory,
        router=router, config=config, start_step=0, allow_context_creation=True,
        max_steps=args.max_steps,
    )
    returned, step = route_phase(
        return_source, phase="return", projector=projector, memory=memory,
        router=router, config=config, start_step=step,
        allow_context_creation=not args.disable_return_creation,
        max_steps=args.max_steps,
    )
    output_dir = args.output_dir.expanduser().resolve()
    write_rows(output_dir / "discovery_routing_log.csv", discovery)
    write_rows(output_dir / "return_routing_log.csv", returned)
    metadata = {
        "context_config": str(config_path),
        "source_discovery_csv": str(args.discovery_csv.expanduser().resolve()),
        "source_return_csv": str(args.return_csv.expanduser().resolve()),
        "base_label": str(args.base_label),
        "bootstrap_samples": int(bootstrap_samples),
        "descriptor_dim": int(projector.input_dim),
        "projection_dim": int(projector.projection_dim),
        "projection_seed": int(projector.random_state),
        "bandwidth": bandwidth_info,
        "discovery_frames": len(discovery),
        "return_frames": len(returned),
        "final_context_ids": sorted(memory.contexts),
        "return_creation_enabled": not args.disable_return_creation,
    }
    with (output_dir / "replay_metadata.yml").open("w") as destination:
        yaml.safe_dump(metadata, destination, sort_keys=False)
    print(f"CPU replay complete: {output_dir}")
    print(f"Discovery frames={len(discovery)}, Return frames={len(returned)}")
    print(f"Final contexts={sorted(memory.contexts)}")


if __name__ == "__main__":
    main()
