#!/usr/bin/env python3
"""Boundary-blind semantic routing with best-parent PSP online adaptation.

Scene/sequence names are used only to assemble the benchmark stream and report
metrics. Routing receives encoder tensors and frozen V7 attribute probabilities.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import torch
import yaml
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
for path in (
    ROOT,
    ROOT / "scene_discovery_online",
    ROOT / "scene_discovery_factorized_mlp",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from context_adaptation.dynamic_psp import activate_context_residuals  # noqa: E402
from context_adaptation.model_adapter import (  # noqa: E402
    ContextAwareModelAdapter,
    configure_residual_only_training,
    freeze_encoder_batch_stats,
)
from context_adaptation.semantic_superposition.context_manager import (  # noqa: E402
    SemanticContextManager,
)
from context_adaptation.semantic_superposition.controller import (  # noqa: E402
    CheckpointableSemanticController,
)
from context_adaptation.semantic_superposition.parent_selector import (  # noqa: E402
    HistoricalContextSelector,
)
from context_adaptation.semantic_superposition.replay_buffer import (  # noqa: E402
    PendingReplayBuffer,
    compact_encoded_batch,
)
from context_adaptation.semantic_superposition.runtime_predictor import (  # noqa: E402
    EncodedAttributePredictor,
)
from context_adaptation.semantic_superposition.protocol import (  # noqa: E402
    build_visits,
    exclude_dataset_samples,
    load_excluded_support_ids,
    routed_context_hook,
    sample_id_from_batch,
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
    materialize_effective_base_config,
    parameter_snapshot,
    require_mapping,
    resolve_path,
    restore_parameter_snapshot,
    validate_base_config,
)


CHECKPOINT_VERSION = 2


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def optimizer_snapshot(optimizer, parameters):
    """Snapshot only one Context's optimizer moments, keyed by parameter object."""
    return {
        parameter: _clone_tree(optimizer.state.get(parameter, {}))
        for parameter in parameters
    }


def restore_optimizer_snapshot(optimizer, snapshot):
    for parameter, state in snapshot.items():
        optimizer.state[parameter] = {
            key: (
                value.to(device=parameter.device)
                if torch.is_tensor(value)
                else _clone_tree(value)
            )
            for key, value in state.items()
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--init-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-target-scenes", type=int, default=-1)
    parser.add_argument("--max-steps-per-scene", type=int, default=-1)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def semantic_key(value):
    result = tuple(str(item).strip().lower() for item in value)
    if len(result) != 3:
        raise ValueError("semantic key must have weather, road, and lighting")
    return result


def sample_id(batch, split=None):
    return sample_id_from_batch(batch, split)


def network_device(network):
    return next(network.parameters()).device


def train_encoded(
    pipeline,
    optimizer,
    manager,
    context_id,
    encoded,
    *,
    grad_clip,
):
    context_name = manager.name(context_id)
    _, parameters = manager.activate(context_id)
    pipeline.network.train()
    freeze_encoder_batch_stats(pipeline.network)
    optimizer.zero_grad(set_to_none=True)
    output = ContextAwareModelAdapter(pipeline.network).forward_with_context(
        encoded, context_name
    )
    loss = ContextAwareModelAdapter(pipeline.network).loss(output)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite online loss for {context_name}")
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
    optimizer.step()
    return float(loss.detach().cpu().item())


def save_checkpoint(
    path,
    *,
    pipeline,
    optimizer,
    controller,
    manager,
    step_idx,
    update_idx,
    attribute_checkpoint,
    scene_assignments,
    metadata=None,
):
    if manager.model_names.keys() != {
        entry.context_id for entry in controller.registry.entries()
    }:
        raise RuntimeError("semantic and model Context IDs differ")
    atomic_torch_save({
        "semantic_superposition_checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": pipeline.network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_context_names": [
            group.get("context_name") for group in optimizer.param_groups
        ],
        "controller_state": controller.state_dict(),
        "context_manager_state": manager.state_dict(),
        "context_manifest": list(manager.model_names.values()),
        "attribute_checkpoint": str(attribute_checkpoint),
        "step_idx": int(step_idx),
        "update_idx": int(update_idx),
        "scene_assignments_for_evaluation_only": dict(scene_assignments),
        "metadata": dict(metadata or {}),
    }, path)


def main():
    args = parse_args()
    experiment_path = resolve_path(args.experiment_config)
    config, base_scene, scenes, source_base_config, _ = load_experiment(experiment_path)
    run_dir = resolve_path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "metrics", "checkpoints", "evaluations", "config_snapshots"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(experiment_path, run_dir / "config_snapshots/experiment.yml")

    data = require_mapping(config.get("DATA"), "DATA")
    online = require_mapping(config.get("ONLINE"), "ONLINE")
    evaluation = require_mapping(config.get("EVALUATION"), "EVALUATION")
    router_cfg = require_mapping(config.get("SEMANTIC_ROUTER"), "SEMANTIC_ROUTER")
    selector_cfg = require_mapping(config.get("PARENT_SELECTION", {}), "PARENT_SELECTION")
    replay_cfg = require_mapping(config.get("PENDING_REPLAY", {}), "PENDING_REPLAY")
    audit_cfg = require_mapping(config.get("AUDIT", {}), "AUDIT")
    if str(selector_cfg.get("METRIC", "mean_detection_loss")) != "mean_detection_loss":
        raise ValueError("PARENT_SELECTION.METRIC must be mean_detection_loss")
    if str(selector_cfg.get("CANDIDATES", "all_registered_contexts")) != "all_registered_contexts":
        raise ValueError(
            "PARENT_SELECTION.CANDIDATES must be all_registered_contexts"
        )
    if not bool(replay_cfg.get("ENABLED", True)):
        raise ValueError("PENDING_REPLAY must be enabled for boundary decisions")
    runtime_config = materialize_effective_base_config(
        source_base_config,
        base_scene=base_scene,
        data_config=data,
        output_path=run_dir / "config_snapshots/effective_base_model_config.yml",
    )
    validate_base_config(runtime_config, base_scene)
    selected_scenes = list(scenes)
    if args.max_target_scenes > 0:
        selected_scenes = selected_scenes[:args.max_target_scenes]
    audit_data_protocol(
        runtime_config, scenes=(base_scene, *selected_scenes), run_dir=run_dir
    )

    num_workers = int(
        args.num_workers if args.num_workers is not None else data.get("NUM_WORKERS", 0)
    )
    if int(data.get("ONLINE_BATCH_SIZE", 1)) != 1:
        raise ValueError("semantic routing requires ONLINE_BATCH_SIZE=1")
    pipeline, _, _ = build_pipeline(
        base_config=runtime_config,
        base_scene=base_scene,
        run_dir=run_dir,
        num_workers=num_workers,
        best_metric=evaluation,
    )
    initial_path = resolve_path(args.init_model)
    load_raw_or_oracle_checkpoint(
        pipeline.network,
        initial_path,
        base_scene=base_scene,
        trainable_scope=str(online.get("TRAINABLE_SCOPE", "fuser_head")),
        checkpoint_load_policy=str(
            require_mapping(config.get("BASE"), "BASE").get(
                "CHECKPOINT_LOAD_POLICY", "compatible_base_context"
            )
        ),
    )
    scope = str(online.get("TRAINABLE_SCOPE", "fuser_head"))
    configure_residual_only_training(pipeline.network, trainable_scope=scope)
    _, base_parameters = activate_context_residuals(
        pipeline.network, base_scene.name, trainable_scope=scope
    )
    optimizer = build_optimizer(base_parameters, online)
    optimizer.param_groups[0]["context_name"] = base_scene.name

    predictor = EncodedAttributePredictor(
        str(resolve_path(router_cfg["ATTRIBUTE_CHECKPOINT"])),
        device=str(router_cfg.get("DEVICE", "auto")),
        feature_keys=router_cfg.get("FEATURE_KEYS", {}),
    )
    base_key = semantic_key(router_cfg["BASE_KEY"])
    controller = CheckpointableSemanticController(
        predictor.label_names,
        base_key,
        decision_window=int(router_cfg.get("DECISION_WINDOW", 5)),
        confirmation_frames=int(router_cfg.get("CONFIRMATION_FRAMES", 3)),
        majority_ratio=float(router_cfg.get("MAJORITY_RATIO", 0.8)),
        new_context_confirmation_frames=int(
            router_cfg.get("NEW_CONTEXT_CONFIRMATION_FRAMES", 5)
        ),
        new_context_majority_ratio=float(
            router_cfg.get("NEW_CONTEXT_MAJORITY_RATIO", 1.0)
        ),
        confidence_thresholds=router_cfg.get("CONFIDENCE", {}),
        confidence_policy="enforce",
        pause_update_when_pending=True,
    )
    manager = SemanticContextManager(
        pipeline.network,
        optimizer,
        base_context_name=base_scene.name,
        trainable_scope=scope,
        learning_rate=float(online.get("LR", 1e-4)),
        inheritance_atol=float(audit_cfg.get("INHERITANCE_ATOL", 1e-6)),
    )
    selector = HistoricalContextSelector(pipeline.network)
    replay = PendingReplayBuffer(
        max_frames=int(replay_cfg.get("MAX_FRAMES", 16)),
        storage_dtype=str(replay_cfg.get("STORAGE_DTYPE", "float16")),
    )
    replay_feature_keys = tuple(pipeline.network.fuser.key_feats)
    thresholds = {
        key: float(router_cfg.get("CONFIDENCE", {}).get(key, 0.0))
        for key in ("weather", "road", "lighting")
    }
    grad_clip = float(online.get("GRAD_CLIP", 0.0))
    metric = evaluation
    baseline_scores = {}
    evaluation_index = 1
    if not args.skip_baseline:
        for scene in (base_scene, *selected_scenes):
            score, _ = evaluate_scene(
                pipeline,
                base_config=runtime_config,
                scene=scene,
                context_name=base_scene.name,
                phase="baseline",
                run_dir=run_dir,
                metric=metric,
                evaluation_index=evaluation_index,
            )
            evaluation_index += 1
            baseline_scores[scene.name] = score

    truth_keys = {
        str(name): semantic_key(value)
        for name, value in require_mapping(
            router_cfg.get("SCENE_KEYS", {}), "SEMANTIC_ROUTER.SCENE_KEYS"
        ).items()
    }
    missing_truth = [scene.name for scene in selected_scenes if scene.name not in truth_keys]
    if missing_truth:
        raise ValueError(f"missing evaluation-only SCENE_KEYS: {missing_truth}")

    routing_path = run_dir / "metrics/routing_log.csv"
    parent_path = run_dir / "metrics/parent_selection.csv"
    scene_counts = {scene.name: Counter() for scene in selected_scenes}
    scene_updates = Counter()
    scene_route_correct = Counter()
    scene_frames = Counter()
    scene_events = {scene.name: Counter() for scene in selected_scenes}
    best_records = {}
    step_idx = 0
    update_idx = 0
    started = time.time()
    stage_selection = str(online.get("STAGE_SELECTION", "last")).lower()
    if stage_selection not in {"last", "oracle_best"}:
        raise ValueError("ONLINE.STAGE_SELECTION must be last or oracle_best")
    eval_every_updates = int(online.get("EVAL_EVERY_UPDATES", 50))

    for scene in selected_scenes:
        from torch.utils.data import DataLoader
        from tools.superposition.oracle_sequential import make_dataset

        dataset = make_dataset(runtime_config, scene, "train")
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=num_workers,
            collate_fn=dataset.collate_fn, drop_last=False,
        )
        progress = tqdm(loader, desc=f"* Auto semantic stream ({scene.name} hidden)")
        scene_start = time.time()
        local_updates = 0
        tracked_context_id = None
        tracked_parent_id = None
        best_score = None
        best_update = None
        best_parameter_state = None
        best_optimizer_state = None
        last_eval_update = None

        def evaluate_stage_candidate(context_id, local_update, event):
            """Benchmark-only evaluator; never feeds labels/scores to the router."""
            nonlocal evaluation_index, best_score, best_update
            nonlocal best_parameter_state, best_optimizer_state, last_eval_update
            context_name = manager.name(context_id)
            active_names, active_parameters = manager.activate(context_id)
            evaluation_index += 1
            score, _ = evaluate_scene(
                pipeline,
                base_config=runtime_config,
                scene=scene,
                context_name=context_name,
                phase=f"stage_selection/{scene.name}/{event}_{local_update:06d}",
                run_dir=run_dir,
                metric=metric,
                evaluation_index=evaluation_index,
            )
            last_eval_update = int(local_update)
            is_best = score is not None and (
                best_score is None or float(score) > float(best_score)
            )
            if is_best:
                best_score = float(score)
                best_update = int(local_update)
                best_parameter_state = parameter_snapshot(
                    pipeline.network, active_names
                )
                best_optimizer_state = optimizer_snapshot(
                    optimizer, active_parameters
                )
            row = {
                "scene": scene.name,
                "event": event,
                "context_id": context_id,
                "context_name": context_name,
                "local_update": local_update,
                "global_update": update_idx,
                "score": "" if score is None else score,
                "is_new_best": int(is_best),
                "best_score": "" if best_score is None else best_score,
                "best_update": "" if best_update is None else best_update,
                "elapsed_sec": time.time() - scene_start,
            }
            append_csv(run_dir / "metrics/online_curve.csv", tuple(row), row)
            return score

        for local_step, batch in enumerate(progress, start=1):
            if args.max_steps_per_scene > 0 and local_step > args.max_steps_per_scene:
                clear_batch(batch)
                break
            step_idx += 1
            illumination = predictor.illumination_from_batch(batch)
            pipeline.network.train()
            freeze_encoder_batch_stats(pipeline.network)
            encoded = ContextAwareModelAdapter(pipeline.network).encode(batch)
            probabilities = predictor.predict(encoded, illumination)
            before_decision = controller.state_dict()
            decision = controller.observe(probabilities, step_idx)
            accepted = all(
                float(getattr(decision, f"{name}_confidence")) >= thresholds[name]
                for name in thresholds
            )
            if accepted and decision.event in {"pending", "create", "switch"}:
                replay.append(
                    step=step_idx,
                    semantic_key=decision.provisional_key,
                    encoded=compact_encoded_batch(encoded, replay_feature_keys),
                    sample_id=sample_id(batch),
                )

            parent_id = None
            replayed = 0
            loss_value = None
            if decision.event == "create":
                confirmed = replay.matching(decision.active_key)
                best_parent, scores = selector.select(confirmed, manager.model_names)
                parent_id = best_parent.context_id
                try:
                    manager.create_from_parent(decision.context_id, parent_id)
                except Exception:
                    controller.load_state_dict(before_decision)
                    raise
                for score in scores:
                    row = {
                        "step": step_idx,
                        "new_context_id": decision.context_id,
                        "semantic_key": "|".join(decision.active_key),
                        "candidate_context_id": score.context_id,
                        "candidate_context_name": score.context_name,
                        "mean_detection_loss": score.mean_loss,
                        "frames": score.frame_count,
                        "rank": score.rank,
                        "selected": int(score.rank == 1),
                    }
                    append_csv(parent_path, tuple(row), row)
                if tracked_context_id is None:
                    tracked_context_id = decision.context_id
                    tracked_parent_id = parent_id
                    if stage_selection == "oracle_best":
                        evaluate_stage_candidate(
                            tracked_context_id, local_updates, "pre_update"
                        )
                for frame in confirmed:
                    loss_value = train_encoded(
                        pipeline, optimizer, manager, decision.context_id,
                        frame.materialize(network_device(pipeline.network)),
                        grad_clip=grad_clip,
                    )
                    update_idx += 1
                    scene_updates[scene.name] += 1
                    local_updates += 1
                    replayed += 1
                replay.clear()
                save_checkpoint(
                    run_dir / f"checkpoints/context_created_{decision.context_id:04d}.checkpoint",
                    pipeline=pipeline, optimizer=optimizer, controller=controller,
                    manager=manager, step_idx=step_idx, update_idx=update_idx,
                    attribute_checkpoint=predictor.predictor.checkpoint_path,
                    scene_assignments={}, metadata={"event": "context_created"},
                )
            elif decision.event == "switch":
                manager.activate(decision.context_id)
                if tracked_context_id is None:
                    tracked_context_id = decision.context_id
                    if stage_selection == "oracle_best":
                        evaluate_stage_candidate(
                            tracked_context_id, local_updates, "pre_update"
                        )
                confirmed = replay.matching(decision.active_key)
                for frame in confirmed:
                    loss_value = train_encoded(
                        pipeline, optimizer, manager, decision.context_id,
                        frame.materialize(network_device(pipeline.network)),
                        grad_clip=grad_clip,
                    )
                    update_idx += 1
                    scene_updates[scene.name] += 1
                    local_updates += 1
                    replayed += 1
                replay.clear()
            elif decision.event == "stable" and decision.update_enabled:
                if tracked_context_id is None:
                    tracked_context_id = decision.context_id
                    if stage_selection == "oracle_best":
                        evaluate_stage_candidate(
                            tracked_context_id, local_updates, "pre_update"
                        )
                if len(replay):
                    replay.append(
                        step=step_idx,
                        semantic_key=decision.provisional_key,
                        encoded=compact_encoded_batch(encoded, replay_feature_keys),
                        sample_id=sample_id(batch),
                    )
                    frames = replay.matching(decision.active_key)
                    for frame in frames:
                        loss_value = train_encoded(
                            pipeline, optimizer, manager, decision.context_id,
                            frame.materialize(network_device(pipeline.network)),
                            grad_clip=grad_clip,
                        )
                        update_idx += 1
                        scene_updates[scene.name] += 1
                        local_updates += 1
                        replayed += 1
                    replay.clear()
                else:
                    loss_value = train_encoded(
                        pipeline, optimizer, manager, decision.context_id, encoded,
                        grad_clip=grad_clip,
                    )
                    update_idx += 1
                    scene_updates[scene.name] += 1
                    local_updates += 1
                    replayed = 1
            elif decision.event == "abstain":
                replay.clear()

            if (
                stage_selection == "oracle_best"
                and tracked_context_id is not None
                and eval_every_updates > 0
                and local_updates > 0
                and local_updates % eval_every_updates == 0
                and last_eval_update != local_updates
            ):
                evaluate_stage_candidate(
                    tracked_context_id, local_updates, "periodic"
                )

            scene_counts[scene.name][decision.context_id] += 1
            true_key = truth_keys[scene.name]
            route_correct = tuple(decision.active_key) == true_key
            scene_frames[scene.name] += 1
            scene_route_correct[scene.name] += int(route_correct)
            scene_events[scene.name][decision.event] += 1
            row = {
                "step": step_idx,
                "update": update_idx,
                "scene_for_evaluation_only": scene.name,
                "local_step": local_step,
                "event": decision.event,
                "context_id": decision.context_id,
                "model_context_name": manager.name(decision.context_id),
                "active_key": "|".join(decision.active_key),
                "provisional_key": "|".join(decision.provisional_key),
                "route_correct": int(route_correct),
                "parent_context_id": "" if parent_id is None else parent_id,
                "pending_count": decision.pending_count,
                "replayed_updates": replayed,
                "model_updated": int(replayed > 0),
                "loss": "" if loss_value is None else loss_value,
                "elapsed_sec": time.time() - started,
            }
            append_csv(routing_path, tuple(row), row)
            progress.set_postfix(
                context=decision.context_id, event=decision.event,
                updates=update_idx,
            )
            clear_batch(batch)
        del loader, dataset

        if tracked_context_id is not None and stage_selection == "oracle_best":
            if last_eval_update != local_updates:
                evaluate_stage_candidate(
                    tracked_context_id, local_updates, "post_update"
                )
            if best_parameter_state is None or best_optimizer_state is None:
                raise RuntimeError(
                    f"No valid best candidate for {scene.name} context "
                    f"{tracked_context_id}"
                )
            restore_parameter_snapshot(pipeline.network, best_parameter_state)
            restore_optimizer_snapshot(optimizer, best_optimizer_state)
            manager.activate(tracked_context_id)
            best_records[scene.name] = {
                "context_id": tracked_context_id,
                "parent_context_id": tracked_parent_id,
                "best_score": best_score,
                "best_update": best_update,
                "updates": local_updates,
                "selection_policy": stage_selection,
                "elapsed_sec": time.time() - scene_start,
            }
            save_checkpoint(
                run_dir / (
                    f"checkpoints/context_{tracked_context_id:04d}_"
                    f"{scene.name}_best.checkpoint"
                ),
                pipeline=pipeline, optimizer=optimizer,
                controller=controller, manager=manager,
                step_idx=step_idx, update_idx=update_idx,
                attribute_checkpoint=predictor.predictor.checkpoint_path,
                scene_assignments={},
                metadata={"event": "stage_best_restored", **best_records[scene.name]},
            )
            print(
                f"* Historical best restored: scene={scene.name}, "
                f"context={tracked_context_id}, score={best_score}, "
                f"update={best_update}"
            )
        elif tracked_context_id is not None:
            best_records[scene.name] = {
                "context_id": tracked_context_id,
                "parent_context_id": tracked_parent_id,
                "best_score": None,
                "best_update": local_updates,
                "updates": local_updates,
                "selection_policy": stage_selection,
                "elapsed_sec": time.time() - scene_start,
            }

    scene_assignments = {
        scene: counts.most_common(1)[0][0]
        for scene, counts in scene_counts.items() if counts
    }
    entries_by_id = {
        entry.context_id: entry for entry in controller.registry.entries()
    }
    for scene in selected_scenes:
        context_id = scene_assignments[scene.name]
        entry = entries_by_id[context_id]
        row = {
            "scene": scene.name,
            "frames": int(scene_frames[scene.name]),
            "route_accuracy": (
                float(scene_route_correct[scene.name]) / scene_frames[scene.name]
            ),
            "assigned_context_id": context_id,
            "assigned_semantic_key": "|".join(entry.key),
            "true_semantic_key": "|".join(truth_keys[scene.name]),
            "create_events": int(scene_events[scene.name]["create"]),
            "switch_events": int(scene_events[scene.name]["switch"]),
            "abstain_frames": int(scene_events[scene.name]["abstain"]),
            "pending_frames": int(scene_events[scene.name]["pending"]),
        }
        append_csv(run_dir / "metrics/routing_scene_summary.csv", tuple(row), row)
    final_rows = []
    for scene in (base_scene, *selected_scenes):
        context_id = 0 if scene.name == base_scene.name else scene_assignments[scene.name]
        context_name = manager.name(context_id)
        score, _ = evaluate_scene(
            pipeline,
            base_config=runtime_config,
            scene=scene,
            context_name=context_name,
            phase="final_static_context",
            run_dir=run_dir,
            metric=metric,
            evaluation_index=evaluation_index,
        )
        evaluation_index += 1
        initial = baseline_scores.get(scene.name)
        row = {
            "scene": scene.name,
            "assigned_context_id": context_id,
            "assigned_model_context": context_name,
            "initial_seq1_score": "" if initial is None else initial,
            "final_context_score": "" if score is None else score,
            "change": "" if initial is None or score is None else score - initial,
            "updates": int(scene_updates.get(scene.name, 0)),
            "best_update": (
                "" if scene.name == base_scene.name
                else best_records.get(scene.name, {}).get("best_update", "")
            ),
            "parent_context_id": (
                "" if scene.name == base_scene.name
                else best_records.get(scene.name, {}).get("parent_context_id", "")
            ),
        }
        final_rows.append(row)
        append_csv(run_dir / "metrics/final_scene_summary.csv", tuple(row), row)

    scored_rows = [
        row for row in final_rows
        if isinstance(row["final_context_score"], (int, float))
    ]
    if scored_rows:
        initial_rows = [
            row for row in scored_rows
            if isinstance(row["initial_seq1_score"], (int, float))
        ]
        average_row = {
            "scene": "average",
            "assigned_context_id": "",
            "assigned_model_context": "",
            "initial_seq1_score": (
                "" if not initial_rows else
                sum(row["initial_seq1_score"] for row in initial_rows)
                / len(initial_rows)
            ),
            "final_context_score": (
                sum(row["final_context_score"] for row in scored_rows)
                / len(scored_rows)
            ),
            "change": "",
            "updates": sum(row["updates"] for row in final_rows),
            "best_update": "",
            "parent_context_id": "",
        }
        if isinstance(average_row["initial_seq1_score"], (int, float)):
            average_row["change"] = (
                average_row["final_context_score"]
                - average_row["initial_seq1_score"]
            )
        final_rows.append(average_row)
        append_csv(
            run_dir / "metrics/final_scene_summary.csv",
            tuple(average_row), average_row,
        )

    save_checkpoint(
        run_dir / "checkpoints/final.semantic_superposition.checkpoint",
        pipeline=pipeline, optimizer=optimizer, controller=controller,
        manager=manager, step_idx=step_idx, update_idx=update_idx,
        attribute_checkpoint=predictor.predictor.checkpoint_path,
        scene_assignments=scene_assignments,
        metadata={
            "status": "complete",
            "final_scene_rows": final_rows,
            "stage_selection": best_records,
        },
    )
    with (run_dir / "context_registry.json").open("w") as output:
        json.dump({
            "semantic": controller.registry.to_records(),
            "model_names": manager.model_names,
            "parents": manager.parents,
            "stage_selection": best_records,
            "scene_assignments_for_evaluation_only": scene_assignments,
        }, output, indent=2, sort_keys=True)
    print(f"* Final semantic contexts: {manager.model_names}")
    print(f"* Final scene summary: {run_dir / 'metrics/final_scene_summary.csv'}")
    print(f"* Results: {run_dir}")
    close_writers(pipeline)


if __name__ == "__main__":
    main()
