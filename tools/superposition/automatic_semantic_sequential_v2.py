#!/usr/bin/env python3
"""Leakage-free, boundary-blind semantic routing plus PSP online adaptation.

The sequence boundary is visible only to the benchmark assembler and metrics.
Neither the semantic controller nor the model receives a sequence identifier.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
for path in (ROOT, ROOT / "scene_discovery_online", ROOT / "scene_discovery_factorized_mlp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from context_adaptation.dynamic_psp import (  # noqa: E402
    activate_context_residuals,
    add_context_optimizer_group,
    restore_inherited_contexts,
)
from context_adaptation.model_adapter import (  # noqa: E402
    ContextAwareModelAdapter,
    configure_residual_only_training,
    freeze_encoder_batch_stats,
)
from context_adaptation.semantic_superposition.context_manager import SemanticContextManager  # noqa: E402
from context_adaptation.semantic_superposition.checkpoint_protocol import (  # noqa: E402
    build_protocol_signature,
    validate_protocol_signature,
)
from context_adaptation.semantic_superposition.controller import CheckpointableSemanticController  # noqa: E402
from context_adaptation.semantic_superposition.parent_selector import HistoricalContextSelector  # noqa: E402
from context_adaptation.semantic_superposition.protocol import (  # noqa: E402
    build_visits,
    exclude_dataset_samples,
    load_excluded_support_ids,
    routed_context_hook,
    sample_id_from_batch,
    sample_id_from_meta,
)
from context_adaptation.semantic_superposition.replay_buffer import (  # noqa: E402
    PendingReplayBuffer,
    compact_encoded_batch,
)
from context_adaptation.semantic_superposition.runtime_predictor import EncodedAttributePredictor  # noqa: E402
from tools.superposition.automatic_semantic_sequential import (  # noqa: E402
    optimizer_snapshot,
    restore_optimizer_snapshot,
)
from tools.superposition.oracle_sequential import (  # noqa: E402
    append_csv,
    atomic_torch_save,
    audit_data_protocol,
    build_optimizer,
    build_pipeline,
    clear_batch,
    close_writers,
    evaluate_scene,
    load_experiment,
    load_raw_or_oracle_checkpoint,
    make_dataset,
    materialize_effective_base_config,
    parameter_snapshot,
    require_mapping,
    resolve_path,
    restore_parameter_snapshot,
    validate_base_config,
)

VERSION = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--init-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-target-scenes", type=int, default=-1)
    parser.add_argument("--max-steps-per-scene", type=int, default=-1)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def semantic_key(value):
    key = tuple(str(item).strip().lower() for item in value)
    if len(key) != 3:
        raise ValueError("semantic key needs weather, road and lighting")
    return key


def network_device(network):
    return next(network.parameters()).device


def train_encoded(pipeline, optimizer, manager, context_id, encoded, grad_clip):
    name = manager.name(context_id)
    _, parameters = manager.activate(context_id)
    pipeline.network.train()
    freeze_encoder_batch_stats(pipeline.network)
    optimizer.zero_grad(set_to_none=True)
    output = ContextAwareModelAdapter(pipeline.network).forward_with_context(encoded, name)
    loss = ContextAwareModelAdapter(pipeline.network).loss(output)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite online loss for {name}")
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
    optimizer.step()
    return float(loss.detach().cpu())


def save_state(path, *, pipeline, optimizer, controller, manager, replay,
               predictor, progress, counters, metadata,
               checkpoint_version=VERSION, protocol_signature=None):
    semantic_ids = {entry.context_id for entry in controller.registry.entries()}
    if set(manager.model_names) != semantic_ids:
        raise RuntimeError("semantic and model Context registries diverged")
    atomic_torch_save({
        "semantic_superposition_checkpoint_version": int(checkpoint_version),
        "protocol_signature": protocol_signature,
        "model_state_dict": pipeline.network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_context_names": [g.get("context_name") for g in optimizer.param_groups],
        "controller_state": controller.state_dict(),
        "context_manager_state": manager.state_dict(),
        "context_manifest": list(manager.model_names.values()),
        "replay_state": replay.state_dict(),
        "attribute_checkpoint": str(predictor.predictor.checkpoint_path),
        "progress": dict(progress),
        "counters": counters,
        "metadata": dict(metadata),
    }, path)


def load_resume(path, *, checkpoint_version=VERSION, protocol_signature=None):
    payload = torch.load(resolve_path(path), map_location="cpu")
    actual_version = int(payload.get("semantic_superposition_checkpoint_version", -1))
    if actual_version != int(checkpoint_version):
        raise ValueError(
            "resume checkpoint version mismatch: "
            f"expected semantic-superposition V{checkpoint_version}, "
            f"found V{actual_version}"
        )
    if int(checkpoint_version) >= 3:
        if protocol_signature is None:
            raise ValueError("V3 runner requires an expected protocol signature")
        validate_protocol_signature(
            payload.get("protocol_signature"), protocol_signature
        )
    if payload.get("metadata", {}).get("safe_resume_point") != "segment_boundary":
        raise ValueError("resume is allowed only from a segment-boundary checkpoint")
    return payload


def main(*, checkpoint_version=VERSION,
         pipeline_id="automatic_semantic_sequential_v2"):
    args = parse_args()
    exp_path = resolve_path(args.experiment_config)
    config, base_scene, scenes, source_config, _ = load_experiment(exp_path)
    if args.max_target_scenes > 0:
        scenes = scenes[:args.max_target_scenes]
    protocol_signature = None
    if int(checkpoint_version) >= 3:
        protocol_signature = build_protocol_signature(
            pipeline_id=pipeline_id,
            checkpoint_version=checkpoint_version,
            experiment_config=config,
            selected_scenes=[scene.name for scene in scenes],
            init_model=args.init_model,
            execution_controls={
                "max_target_scenes": int(args.max_target_scenes),
                "max_steps_per_scene": int(args.max_steps_per_scene),
                "skip_baseline": bool(args.skip_baseline),
            },
        )
    run_dir = resolve_path(args.output_dir)
    resume = (
        load_resume(
            args.resume_checkpoint,
            checkpoint_version=checkpoint_version,
            protocol_signature=protocol_signature,
        )
        if args.resume_checkpoint else None
    )
    if resume is None and (run_dir / "checkpoints/latest.checkpoint").exists():
        raise FileExistsError(f"run already exists; use --resume-checkpoint: {run_dir}")
    for name in ("logs", "metrics", "checkpoints", "evaluations", "config_snapshots"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(exp_path, run_dir / "config_snapshots/experiment.yml")

    data = require_mapping(config["DATA"], "DATA")
    online = require_mapping(config["ONLINE"], "ONLINE")
    evaluation = require_mapping(config["EVALUATION"], "EVALUATION")
    router_cfg = require_mapping(config["SEMANTIC_ROUTER"], "SEMANTIC_ROUTER")
    replay_cfg = require_mapping(config["PENDING_REPLAY"], "PENDING_REPLAY")
    stream_cfg = require_mapping(config.get("STREAM", {}), "STREAM")
    audit_cfg = require_mapping(config.get("AUDIT", {}), "AUDIT")
    if int(data.get("ONLINE_BATCH_SIZE", 1)) != 1:
        raise ValueError("ONLINE_BATCH_SIZE must be 1")
    runtime_config = materialize_effective_base_config(
        source_config, base_scene=base_scene, data_config=data,
        output_path=run_dir / "config_snapshots/effective_base_model_config.yml")
    validate_base_config(runtime_config, base_scene)
    audit_data_protocol(runtime_config, scenes=(base_scene, *scenes), run_dir=run_dir)
    workers = int(args.num_workers if args.num_workers is not None else data.get("NUM_WORKERS", 0))
    pipeline, _, _ = build_pipeline(
        base_config=runtime_config, base_scene=base_scene, run_dir=run_dir,
        num_workers=workers, best_metric=evaluation)
    scope = str(online.get("TRAINABLE_SCOPE", "fuser_head"))
    load_raw_or_oracle_checkpoint(
        pipeline.network, resolve_path(args.init_model), base_scene=base_scene,
        trainable_scope=scope,
        checkpoint_load_policy=str(config["BASE"].get("CHECKPOINT_LOAD_POLICY", "compatible_base_context")))

    restored_results = []
    if resume is not None:
        manager_state = resume["context_manager_state"]
        restored_results = restore_inherited_contexts(
            pipeline.network, resume["context_manifest"], manager_state["parents"],
            trainable_scope=scope, atol=float(audit_cfg.get("INHERITANCE_ATOL", 1e-6)))
        pipeline.network.load_state_dict(resume["model_state_dict"], strict=True)
    configure_residual_only_training(pipeline.network, trainable_scope=scope)
    _, base_parameters = activate_context_residuals(
        pipeline.network, base_scene.name, trainable_scope=scope)
    optimizer = build_optimizer(base_parameters, online)
    optimizer.param_groups[0]["context_name"] = base_scene.name
    for result, _ in restored_results:
        add_context_optimizer_group(optimizer, result, learning_rate=float(online["LR"]))
    if resume is not None:
        expected_groups = list(resume["optimizer_context_names"])
        actual_groups = [g.get("context_name") for g in optimizer.param_groups]
        if expected_groups != actual_groups:
            raise ValueError(f"optimizer Context groups differ: {actual_groups} != {expected_groups}")
        optimizer.load_state_dict(resume["optimizer_state_dict"])

    predictor = EncodedAttributePredictor(
        str(resolve_path(router_cfg["ATTRIBUTE_CHECKPOINT"])),
        device=str(router_cfg.get("DEVICE", "auto")),
        feature_keys=router_cfg.get("FEATURE_KEYS", {}))
    if resume is not None and str(resolve_path(resume["attribute_checkpoint"])) != str(resolve_path(router_cfg["ATTRIBUTE_CHECKPOINT"])):
        raise ValueError("attribute checkpoint differs from resume checkpoint")
    controller = CheckpointableSemanticController(
        predictor.label_names, semantic_key(router_cfg["BASE_KEY"]),
        decision_window=int(router_cfg.get("DECISION_WINDOW", 5)),
        confirmation_frames=int(router_cfg.get("CONFIRMATION_FRAMES", 3)),
        majority_ratio=float(router_cfg.get("MAJORITY_RATIO", 0.8)),
        new_context_confirmation_frames=int(router_cfg.get("NEW_CONTEXT_CONFIRMATION_FRAMES", 5)),
        new_context_majority_ratio=float(router_cfg.get("NEW_CONTEXT_MAJORITY_RATIO", 1.0)),
        confidence_thresholds=router_cfg.get("CONFIDENCE", {}),
        confidence_policy="enforce", pause_update_when_pending=True)
    manager = SemanticContextManager(
        pipeline.network, optimizer, base_context_name=base_scene.name,
        trainable_scope=scope, learning_rate=float(online["LR"]),
        inheritance_atol=float(audit_cfg.get("INHERITANCE_ATOL", 1e-6)),
        restored_state=None if resume is None else resume["context_manager_state"])
    replay = PendingReplayBuffer(
        max_frames=int(replay_cfg.get("MAX_FRAMES", 16)),
        storage_dtype=str(replay_cfg.get("STORAGE_DTYPE", "float16")))
    if resume is not None:
        controller.load_state_dict(resume["controller_state"])
        replay.load_state_dict(resume["replay_state"])
    selector = HistoricalContextSelector(pipeline.network)

    support_manifest = resolve_path(stream_cfg["SUPPORT_MANIFEST"])
    support_roles = tuple(stream_cfg.get("SUPPORT_ROLES", ["support_train"]))
    exclude_roles = tuple(stream_cfg.get("EXCLUDE_ROLES", ["support_train"]))
    support_ids = load_excluded_support_ids(
        support_manifest, roles=support_roles)
    excluded = load_excluded_support_ids(
        support_manifest, roles=exclude_roles)
    expected_excluded_per_scene = int(stream_cfg.get(
        "EXPECTED_EXCLUDED_PER_SCENE",
        stream_cfg.get("EXPECTED_SUPPORT_PER_SCENE", 30 if exclude_roles else 0),
    ))
    visits = build_visits(
        [scene.name for scene in scenes], seed=int(stream_cfg.get("ORDER_SEED", 20260913)),
        specifications=stream_cfg.get("VISITS"))
    scene_by_name = {scene.name: scene for scene in scenes}
    truth_keys = {str(k): semantic_key(v) for k, v in router_cfg["SCENE_KEYS"].items()}
    if set(scene_by_name) - set(truth_keys):
        raise ValueError("SCENE_KEYS does not cover all target scenes")

    counters = resume.get("counters", {}) if resume else {}
    step_idx = int(counters.get("step_idx", 0))
    update_idx = int(counters.get("update_idx", 0))
    eval_index = int(counters.get("evaluation_index", 1))
    baseline = dict(counters.get("baseline_scores", {}))
    start_segment = int(resume["progress"]["next_segment"]) if resume else 0
    route_maps = defaultdict(dict)
    counts = defaultdict(Counter)
    frames = Counter()
    correct = Counter()
    events = defaultdict(Counter)
    updates = Counter()
    best_records = []
    timing_records = list(counters.get("timing_records", ()))
    started = time.time()
    stage_selection = str(online.get("STAGE_SELECTION", "last")).lower()
    if stage_selection not in {"last", "oracle_best"}:
        raise ValueError("ONLINE.STAGE_SELECTION must be last or oracle_best")
    if resume is not None:
        if stage_selection != resume["metadata"].get("stage_selection"):
            raise ValueError("stage-selection policy differs from resume checkpoint")
        for name, raw in counters.get("counts", {}).items(): counts[name].update({int(k): int(v) for k, v in raw.items()})
        frames.update(counters.get("frames", {})); correct.update(counters.get("correct", {}))
        updates.update(counters.get("updates", {}))
        for name, raw in counters.get("events", {}).items(): events[name].update(raw)
        for name, raw in counters.get("route_maps", {}).items(): route_maps[name].update(raw)
        best_records.extend(counters.get("best_records", ()))

    def serializable_counters():
        return {
            "step_idx": step_idx, "update_idx": update_idx,
            "evaluation_index": eval_index, "baseline_scores": baseline,
            "counts": {k: dict(v) for k, v in counts.items()},
            "frames": dict(frames), "correct": dict(correct),
            "events": {k: dict(v) for k, v in events.items()},
            "updates": dict(updates),
            "route_maps": {k: dict(v) for k, v in route_maps.items()},
            "best_records": list(best_records),
            "timing_records": list(timing_records),
        }

    try:
        if resume is None and not args.skip_baseline:
            for scene in (base_scene, *scenes):
                score, _ = evaluate_scene(
                    pipeline, base_config=runtime_config, scene=scene,
                    context_name=base_scene.name, phase="baseline", run_dir=run_dir,
                    metric=evaluation, evaluation_index=eval_index)
                eval_index += 1; baseline[scene.name] = score

        segments = [(visit, scene_by_name[name]) for visit in visits for name in visit.order]
        protocol_audit = {"support_manifest": str(support_manifest),
                          "support_roles": list(support_roles),
                          "exclude_roles": list(exclude_roles),
                          "support_stream_policy": (
                              "excluded_before_loading" if exclude_roles
                              else "included_at_original_dataset_positions"
                          ),
                          "support_manifest_excluded_total": len(excluded),
                          "expected_excluded_per_scene": expected_excluded_per_scene,
                          "visits": [{"name": v.name, "split": v.split, "update": v.update_enabled, "order": list(v.order)} for v in visits]}
        with (run_dir / "metrics/stream_protocol.json").open("w") as out:
            json.dump(protocol_audit, out, indent=2)

        for segment_index, (visit, scene) in enumerate(segments):
            if segment_index < start_segment:
                continue
            segment_started = time.time()
            segment_started_utc = datetime.now(timezone.utc).isoformat()
            oracle_evaluation_sec = 0.0
            optimizer_update_sec = 0.0
            parent_selection_sec = 0.0
            dataset = make_dataset(runtime_config, scene, visit.split)
            original_sample_ids = [
                sample_id_from_meta(item["meta"], visit.split)
                for item in dataset.list_dict_item
            ]
            source_support_positions = [
                position for position, identifier in enumerate(original_sample_ids)
                if identifier in support_ids
            ]
            exclusion_audit = exclude_dataset_samples(dataset, excluded, visit.split)
            support_positions = [
                position for position, item in enumerate(dataset.list_dict_item)
                if sample_id_from_meta(item["meta"], visit.split) in support_ids
            ]
            expected_excluded = (
                expected_excluded_per_scene if visit.split == "train" else 0
            )
            if exclusion_audit["excluded"] != expected_excluded:
                raise RuntimeError(
                    f"support exclusion mismatch for {visit.name}/{scene.name}: "
                    f"expected={expected_excluded}, actual={exclusion_audit['excluded']}"
                )
            audit_row = {
                "visit": visit.name,
                "scene": scene.name,
                "support_frames_in_source_dataset": len(source_support_positions),
                "support_frames_at_original_positions": len(support_positions),
                "first_support_position": support_positions[0] if support_positions else "",
                "last_support_position": support_positions[-1] if support_positions else "",
                "loader_shuffle": 0,
                **{k: v for k, v in exclusion_audit.items() if k != "excluded_ids"},
            }
            append_csv(run_dir / "metrics/support_exclusion.csv", tuple(audit_row), audit_row)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                                collate_fn=dataset.collate_fn, drop_last=False)
            progress = tqdm(loader, desc=f"* {visit.name} ({scene.name} hidden)")
            local_updates = Counter()
            touched = set()
            trackers = {}

            def evaluate_candidate(cid, event):
                nonlocal eval_index, oracle_evaluation_sec
                name = manager.name(cid)
                names, parameters = manager.activate(cid)
                evaluation_started = time.time()
                score, _ = evaluate_scene(
                    pipeline, base_config=runtime_config, scene=scene,
                    context_name=name,
                    phase=f"oracle_stage/{visit.name}/{scene.name}/{name}/{event}_{local_updates[cid]:06d}",
                    run_dir=run_dir, metric=evaluation, evaluation_index=eval_index)
                oracle_evaluation_sec += time.time() - evaluation_started
                eval_index += 1
                state = trackers.setdefault(cid, {"score": None, "update": None, "params": None, "optimizer": None, "last": None})
                is_best = score is not None and (state["score"] is None or score > state["score"])
                if is_best:
                    state.update(score=float(score), update=int(local_updates[cid]),
                                 params=parameter_snapshot(pipeline.network, names),
                                 optimizer=optimizer_snapshot(optimizer, parameters))
                state["last"] = int(local_updates[cid])
                row = {"visit": visit.name, "scene": scene.name, "event": event,
                       "context_id": cid, "context_name": name,
                       "local_update": local_updates[cid], "global_update": update_idx,
                       "score": "" if score is None else score, "is_new_best": int(is_best)}
                append_csv(run_dir / "metrics/online_curve.csv", tuple(row), row)

            for local_step, batch in enumerate(progress, 1):
                if args.max_steps_per_scene > 0 and local_step > args.max_steps_per_scene:
                    clear_batch(batch); break
                step_idx += 1
                pipeline.network.train(); freeze_encoder_batch_stats(pipeline.network)
                illumination = predictor.illumination_from_batch(batch)
                encoded = ContextAwareModelAdapter(pipeline.network).encode(batch)
                probabilities = predictor.predict(encoded, illumination)
                before = controller.state_dict()
                previous_context_id = controller.active_entry.context_id
                decision = controller.observe(probabilities, step_idx)
                identifier = sample_id_from_batch(batch, visit.split)
                accepted = all(float(getattr(decision, f"{key}_confidence")) >= float(router_cfg["CONFIDENCE"][key])
                               for key in ("weather", "road", "lighting"))
                if visit.update_enabled and accepted and decision.event in {"pending", "create", "switch"}:
                    replay.append(step=step_idx, semantic_key=decision.provisional_key,
                                  encoded=compact_encoded_batch(encoded, pipeline.network.fuser.key_feats),
                                  sample_id=identifier)
                parent_id = None; replayed = 0; loss = None
                if decision.event == "create":
                    confirmed = replay.matching(decision.active_key) if visit.update_enabled else ()
                    if confirmed:
                        selection_started = time.time()
                        selected, scores = selector.select(confirmed, manager.model_names)
                        parent_selection_sec += time.time() - selection_started
                        parent_id = selected.context_id
                        for score in scores:
                            row = {"visit": visit.name, "step": step_idx,
                                   "new_context_id": decision.context_id,
                                   "candidate_context_id": score.context_id,
                                   "candidate_context_name": score.context_name,
                                   "mean_detection_loss": score.mean_loss, "frames": score.frame_count,
                                   "rank": score.rank, "selected": int(score.rank == 1),
                                   "selection_mode": "confirmed_window_detection_loss"}
                            append_csv(run_dir / "metrics/parent_selection.csv", tuple(row), row)
                    else:
                        parent_id = previous_context_id
                        row = {"visit": visit.name, "step": step_idx,
                               "new_context_id": decision.context_id,
                               "candidate_context_id": parent_id,
                               "candidate_context_name": manager.name(parent_id),
                               "mean_detection_loss": "", "frames": 0, "rank": 1,
                               "selected": 1,
                               "selection_mode": "previous_context_no_labels_read_only"}
                        append_csv(run_dir / "metrics/parent_selection.csv", tuple(row), row)
                    try:
                        manager.create_from_parent(decision.context_id, parent_id)
                    except Exception:
                        controller.load_state_dict(before); raise
                elif decision.event == "switch":
                    manager.activate(decision.context_id)

                update_frames = ()
                if visit.update_enabled and decision.event in {"create", "switch"}:
                    update_frames = replay.matching(decision.active_key); replay.clear()
                elif visit.update_enabled and decision.event == "stable" and decision.update_enabled:
                    if len(replay):
                        replay.append(step=step_idx, semantic_key=decision.provisional_key,
                                      encoded=compact_encoded_batch(encoded, pipeline.network.fuser.key_feats),
                                      sample_id=identifier)
                        update_frames = replay.matching(decision.active_key); replay.clear()
                    else:
                        update_frames = (encoded,)
                elif decision.event == "abstain" or not visit.update_enabled:
                    replay.clear()

                cid = decision.context_id
                if update_frames and stage_selection == "oracle_best" and cid not in touched:
                    evaluate_candidate(cid, "pre_update")
                for item in update_frames:
                    materialized = item.materialize(network_device(pipeline.network)) if hasattr(item, "materialize") else item
                    update_started = time.time()
                    loss = train_encoded(pipeline, optimizer, manager, cid, materialized,
                                         float(online.get("GRAD_CLIP", 0.0)))
                    optimizer_update_sec += time.time() - update_started
                    update_idx += 1; local_updates[cid] += 1; updates[scene.name] += 1; replayed += 1
                if update_frames:
                    touched.add(cid)
                interval = int(online.get("EVAL_EVERY_UPDATES", 50))
                if stage_selection == "oracle_best" and interval > 0 and cid in touched and local_updates[cid] % interval == 0 and trackers[cid]["last"] != local_updates[cid]:
                    evaluate_candidate(cid, "periodic")

                metric_key = f"{visit.name}/{scene.name}"
                counts[metric_key][cid] += 1; frames[metric_key] += 1
                correct[metric_key] += int(tuple(decision.active_key) == truth_keys[scene.name])
                events[metric_key][decision.event] += 1
                if visit == visits[-1]: route_maps[scene.name][identifier] = manager.name(cid)
                row = {"step": step_idx, "update": update_idx, "visit": visit.name,
                       "split": visit.split, "updates_enabled": int(visit.update_enabled),
                       "scene_for_evaluation_only": scene.name, "local_step": local_step,
                       "sample_id": identifier, "event": decision.event, "context_id": cid,
                       "model_context_name": manager.name(cid), "active_key": "|".join(decision.active_key),
                       "provisional_key": "|".join(decision.provisional_key),
                       "route_correct": int(tuple(decision.active_key) == truth_keys[scene.name]),
                       "parent_context_id": "" if parent_id is None else parent_id,
                       "replayed_updates": replayed, "loss": "" if loss is None else loss,
                       "elapsed_sec": time.time() - started}
                append_csv(run_dir / "metrics/routing_log.csv", tuple(row), row)
                clear_batch(batch)

            if stage_selection == "oracle_best":
                for cid in sorted(touched):
                    if trackers[cid]["last"] != local_updates[cid]:
                        evaluate_candidate(cid, "post_update")
                    state = trackers[cid]
                    if state["params"] is None:
                        raise RuntimeError(f"no valid best candidate for Context {cid}")
                    restore_parameter_snapshot(pipeline.network, state["params"])
                    restore_optimizer_snapshot(optimizer, state["optimizer"])
                    best_records.append({"visit": visit.name, "scene": scene.name,
                                         "context_id": cid, "best_score": state["score"],
                                         "best_update": state["update"], "updates": local_updates[cid]})
            replay.clear()
            del loader, dataset
            segment_wall_sec = time.time() - segment_started
            timing_row = {
                "segment_index": segment_index,
                "visit": visit.name,
                "scene": scene.name,
                "split": visit.split,
                "updates_enabled": int(visit.update_enabled),
                "frames": int(frames[f"{visit.name}/{scene.name}"]),
                "optimizer_updates": int(sum(local_updates.values())),
                "optimizer_update_sec": optimizer_update_sec,
                "parent_selection_sec": parent_selection_sec,
                "oracle_evaluation_sec": oracle_evaluation_sec,
                "online_pipeline_sec": segment_wall_sec - oracle_evaluation_sec,
                "segment_wall_sec": segment_wall_sec,
                "sec_per_optimizer_update": (
                    optimizer_update_sec / sum(local_updates.values())
                    if sum(local_updates.values()) else ""
                ),
                "started_utc": segment_started_utc,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
            timing_records.append(timing_row)
            append_csv(run_dir / "metrics/segment_timing.csv", tuple(timing_row), timing_row)
            state_counters = serializable_counters()
            save_state(run_dir / "checkpoints/latest.checkpoint", pipeline=pipeline,
                       optimizer=optimizer, controller=controller, manager=manager,
                       replay=replay, predictor=predictor,
                       progress={"next_segment": segment_index + 1, "segments": len(segments)},
                       counters=state_counters,
                       metadata={"safe_resume_point": "segment_boundary",
                                 "stage_selection": stage_selection,
                                 "last_visit": visit.name, "last_scene": scene.name,
                                 "elapsed_segment_sec": segment_wall_sec,
                                 "optimizer_update_sec": optimizer_update_sec,
                                 "oracle_evaluation_sec": oracle_evaluation_sec},
                       checkpoint_version=checkpoint_version,
                       protocol_signature=protocol_signature)

        summary_path = run_dir / "metrics/routing_scene_summary.csv"
        if summary_path.exists():
            summary_path.unlink()
        for key in sorted(frames):
            visit_name, scene_name = key.split("/", 1)
            row = {"visit": visit_name, "scene": scene_name, "frames": frames[key],
                   "route_accuracy": correct[key] / frames[key],
                   "dominant_context_id": counts[key].most_common(1)[0][0],
                   "create_events": events[key]["create"], "switch_events": events[key]["switch"],
                   "pending_frames": events[key]["pending"], "abstain_frames": events[key]["abstain"]}
            append_csv(summary_path, tuple(row), row)

        final_path = run_dir / "metrics/final_routed_scene_summary.csv"
        if final_path.exists():
            final_path.unlink()
        final_rows = []
        base_score, _ = evaluate_scene(
            pipeline, base_config=runtime_config, scene=base_scene,
            context_name=base_scene.name, phase="final_base_fixed_context",
            run_dir=run_dir, metric=evaluation, evaluation_index=eval_index)
        eval_index += 1
        base_row = {"scene": base_scene.name, "evaluation": "fixed_base_context",
                    "dominant_context": base_scene.name,
                    "initial_seq1_score": baseline.get(base_scene.name, ""),
                    "final_routed_score": "" if base_score is None else base_score,
                    "change": "" if base_score is None or not isinstance(baseline.get(base_scene.name), (int, float)) else base_score - baseline[base_scene.name],
                    "updates": 0}
        final_rows.append(base_row)
        append_csv(final_path, tuple(base_row), base_row)
        for scene in scenes:
            route_map = route_maps[scene.name]
            if not route_map:
                raise RuntimeError(f"no read-only test routes recorded for {scene.name}")
            with routed_context_hook(pipeline.network, route_map, "test"):
                score, _ = evaluate_scene(
                    pipeline, base_config=runtime_config, scene=scene,
                    context_name=base_scene.name, phase="final_actual_per_frame_routing",
                    run_dir=run_dir, metric=evaluation, evaluation_index=eval_index)
            eval_index += 1
            dominant = Counter(route_map.values()).most_common(1)[0][0]
            row = {"scene": scene.name, "evaluation": "actual_per_frame_routing",
                   "dominant_context": dominant, "initial_seq1_score": baseline.get(scene.name, ""),
                   "final_routed_score": "" if score is None else score,
                   "change": "" if score is None or not isinstance(baseline.get(scene.name), (int, float)) else score - baseline[scene.name],
                   "updates": updates[scene.name]}
            final_rows.append(row)
            append_csv(final_path, tuple(row), row)
        scored = [r for r in final_rows if isinstance(r["final_routed_score"], (int, float))]
        if scored:
            row = {"scene": "average", "evaluation": "actual_per_frame_routing", "dominant_context": "",
                   "initial_seq1_score": "", "final_routed_score": sum(r["final_routed_score"] for r in scored) / len(scored),
                   "change": "", "updates": sum(r["updates"] for r in final_rows)}
            append_csv(final_path, tuple(row), row)

        timing_summary_path = run_dir / "metrics/update_timing_scene_summary.csv"
        if timing_summary_path.exists():
            timing_summary_path.unlink()
        train_timings = [row for row in timing_records if row["updates_enabled"]]
        timing_summary_rows = []
        for scene in scenes:
            rows = [row for row in train_timings if row["scene"] == scene.name]
            count = sum(int(row["optimizer_updates"]) for row in rows)
            item = {
                "scene": scene.name,
                "train_visits": len(rows),
                "frames_processed": sum(int(row["frames"]) for row in rows),
                "optimizer_updates": count,
                "optimizer_update_sec": sum(float(row["optimizer_update_sec"]) for row in rows),
                "parent_selection_sec": sum(float(row["parent_selection_sec"]) for row in rows),
                "online_pipeline_sec": sum(float(row["online_pipeline_sec"]) for row in rows),
                "wall_sec_including_oracle_eval": sum(float(row["segment_wall_sec"]) for row in rows),
            }
            item["sec_per_optimizer_update"] = (
                item["optimizer_update_sec"] / count if count else ""
            )
            timing_summary_rows.append(item)
            append_csv(timing_summary_path, tuple(item), item)
        if timing_summary_rows:
            total_updates = sum(row["optimizer_updates"] for row in timing_summary_rows)
            total_row = {
                "scene": "total",
                "train_visits": sum(row["train_visits"] for row in timing_summary_rows),
                "frames_processed": sum(row["frames_processed"] for row in timing_summary_rows),
                "optimizer_updates": total_updates,
                "optimizer_update_sec": sum(row["optimizer_update_sec"] for row in timing_summary_rows),
                "parent_selection_sec": sum(row["parent_selection_sec"] for row in timing_summary_rows),
                "online_pipeline_sec": sum(row["online_pipeline_sec"] for row in timing_summary_rows),
                "wall_sec_including_oracle_eval": sum(row["wall_sec_including_oracle_eval"] for row in timing_summary_rows),
                "sec_per_optimizer_update": "",
            }
            total_row["sec_per_optimizer_update"] = (
                total_row["optimizer_update_sec"] / total_updates
                if total_updates else ""
            )
            append_csv(timing_summary_path, tuple(total_row), total_row)
            average_row = dict(total_row)
            average_row["scene"] = "average_per_scene"
            for key in ("train_visits", "frames_processed", "optimizer_updates",
                        "optimizer_update_sec", "parent_selection_sec",
                        "online_pipeline_sec", "wall_sec_including_oracle_eval"):
                average_row[key] = total_row[key] / len(timing_summary_rows)
            append_csv(timing_summary_path, tuple(average_row), average_row)

        save_state(run_dir / "checkpoints/final.semantic_superposition.checkpoint",
                   pipeline=pipeline, optimizer=optimizer, controller=controller,
                   manager=manager, replay=replay, predictor=predictor,
                   progress={"next_segment": len(segments), "segments": len(segments)},
                   counters=serializable_counters(),
                   metadata={"safe_resume_point": "segment_boundary", "status": "complete",
                             "stage_selection": stage_selection, "best_records": best_records,
                             "final_rows": final_rows},
                   checkpoint_version=checkpoint_version,
                   protocol_signature=protocol_signature)
        with (run_dir / "context_registry.json").open("w") as out:
            json.dump({"semantic": controller.registry.to_records(),
                       "model_names": manager.model_names, "parents": manager.parents,
                       "best_records_oracle_only": best_records}, out, indent=2, sort_keys=True)
        print(f"* Final Contexts: {manager.model_names}")
        print(f"* Actual routed AP: {run_dir / 'metrics/final_routed_scene_summary.csv'}")
    finally:
        close_writers(pipeline)


if __name__ == "__main__":
    main()
