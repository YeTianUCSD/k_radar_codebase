"""Evaluate dynamic-context checkpoints with oracle or automatic routing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Optional

import torch
import yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_adaptation.checkpoint import (
    load_context_checkpoint,
    restore_model_and_context_state,
)
from context_adaptation.context_router import ContextRouter
from context_adaptation.model_adapter import ContextAwareModelAdapter
from context_adaptation.runtime import write_single_context_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a dynamic automatic-context PSP checkpoint."
    )
    parser.add_argument("--config", required=True, help="Evaluation dataset config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_root", default="./results/ContextAdaptation")
    parser.add_argument("--run_name", default="eval_auto_context")
    parser.add_argument("--mode", choices=["auto", "oracle"], default="auto")
    parser.add_argument(
        "--context_id",
        type=int,
        default=None,
        help="Required for oracle mode; ignored in auto mode.",
    )
    parser.add_argument(
        "--initial_context_id",
        type=int,
        default=None,
        help="Initial active context in auto mode. Defaults to checkpoint state.",
    )
    parser.add_argument("--metric_scene_name", default="unknown")
    parser.add_argument("--conf_thr", type=float, default=0.3)
    parser.add_argument("--best_metric_cls", default="auto")
    parser.add_argument("--best_metric_kind", choices=["bev", "3d"], default="3d")
    parser.add_argument("--best_metric_ious", type=float, nargs="+", default=[0.3, 0.5])
    return parser.parse_args()


class EvaluationRouter:
    """Run checkpoint-configured routing without mutating committed contexts."""

    def __init__(self, memory: Any, router: Any, initial_context_id: Optional[int]) -> None:
        self.memory = memory
        router_config = dict(router.state_dict()["config"])
        self.router = ContextRouter(**router_config)
        active_context_id = (
            int(initial_context_id)
            if initial_context_id is not None
            else int(memory.active_context_id)
        )
        self.memory.get(active_context_id)
        self.router.set_active_context(active_context_id)

    def route(self, raw_x: Any) -> dict[str, Any]:
        x = self.router.prepare_feature(raw_x)
        scores = self.memory.score_all(x)
        density_thresholds = self.memory.density_thresholds()
        routing_thresholds = self.memory.routing_thresholds(
            self.router.routing_threshold_scale
        )
        near_thresholds = self.memory.near_thresholds(
            self.router.near_threshold_scale
        )
        decision = self.router.step(
            x,
            scores,
            density_thresholds,
            routing_thresholds=routing_thresholds,
            near_thresholds=near_thresholds,
            context_lifecycles=self.memory.context_lifecycles(),
            allow_context_creation=False,
        )
        if decision.context_id is None:
            raise RuntimeError("Read-only evaluation routing lost its active context.")
        return {
            "context_id": int(decision.context_id),
            "status": decision.status,
            "reason": decision.reason,
            "best_context_id": decision.best_context_id,
            "best_score": decision.best_score,
            "second_score": decision.second_score,
            "best_confidence": decision.best_confidence,
            "second_confidence": decision.second_confidence,
            "historical_evidence": decision.historical_evidence,
            "novelty_strength": decision.novelty_strength,
            "novelty_evidence": decision.novelty_evidence,
            "candidate_evidence": decision.candidate_evidence,
            "raw_feature": raw_x,
            "smoothed_feature": x,
            "scores": scores,
            "density_thresholds": density_thresholds,
            "routing_thresholds": routing_thresholds,
            "near_thresholds": near_thresholds,
        }


def append_route(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "frame_idx",
        "metric_scene_name",
        "status",
        "reason",
        "active_context_id",
        "active_context_name",
        "active_context_lifecycle",
        "best_context_id",
        "best_score",
        "second_score",
        "best_confidence",
        "second_confidence",
        "historical_evidence",
        "novelty_strength",
        "novelty_evidence",
        "candidate_evidence",
        "raw_projected_feature_json",
        "smoothed_projected_feature_json",
        "scores_json",
        "thresholds_json",
        "density_thresholds_json",
        "routing_thresholds_json",
        "near_thresholds_json",
    ]
    is_new = not path.exists()
    with path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def close_writers(pipeline: Any) -> None:
    for writer_name in ("log_train_iter", "log_train_epoch", "log_test"):
        writer = getattr(pipeline, writer_name, None)
        if writer is not None:
            writer.close()


def main() -> None:
    args = parse_args()
    if args.mode == "oracle" and args.context_id is None:
        raise ValueError("--context_id is required in oracle mode.")
    payload = load_context_checkpoint(args.checkpoint, map_location="cpu")
    manifest = tuple(str(name) for name in payload["context_manifest"])
    if not manifest:
        raise RuntimeError("Checkpoint context manifest is empty.")
    runtime_path, runtime_config = write_single_context_runtime_config(
        args.config,
        model_context_name=manifest[0],
        output_root=args.output_root,
        run_name=args.run_name,
        batch_size=1,
        num_workers=0,
        enable_logging=True,
        enable_validation=True,
    )
    runtime_config["GENERAL"]["LOGGING"]["BEST_METRIC"] = {
        "CLS": str(args.best_metric_cls),
        "KIND": str(args.best_metric_kind),
        "IOUS": [float(value) for value in args.best_metric_ious],
        "CONF_THR": float(args.conf_thr),
        "REDUCE": "mean",
        "ONLY_CLASSES_WITH_GT": True,
    }
    runtime_config["VAL"]["IS_CONSIDER_VAL_SUBSET"] = False
    runtime_config["VAL"]["LIST_VAL_CONF_THR"] = [float(args.conf_thr)]
    with open(runtime_path, "w") as output:
        yaml.safe_dump(runtime_config, output, sort_keys=False)

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pipeline = None
    try:
        pipeline = PipelineDetection_v1_0(path_cfg=runtime_path, mode="test")
        memory, checkpoint_router, projector = restore_model_and_context_state(
            pipeline.network,
            payload,
        )
        pipeline.network.eval()
        run_path = Path(pipeline.path_log)
        routing_path = run_path / "evaluation_routing.csv"
        original_forward = pipeline.network.forward

        if args.mode == "oracle":
            entry = memory.get(args.context_id)
            pipeline.network.default_scene_context = entry.model_context_name
            print(
                f"* Oracle context: id={entry.context_id}, "
                f"name={entry.model_context_name}"
            )
        else:
            evaluator = EvaluationRouter(
                memory,
                checkpoint_router,
                args.initial_context_id,
            )
            adapter = ContextAwareModelAdapter(pipeline.network)
            frame_counter = {"value": 0}

            def automatic_forward(network_self: torch.nn.Module, batch: Any) -> Any:
                encoded = adapter.encode(batch)
                projected = projector.transform_batch(encoded)
                if projected.shape != (1, projector.projection_dim):
                    raise RuntimeError(
                        f"Automatic evaluation requires batch size 1; got {projected.shape}."
                    )
                route = evaluator.route(projected[0])
                entry = memory.get(route["context_id"])
                frame_counter["value"] += 1
                append_route(
                    routing_path,
                    {
                        "frame_idx": frame_counter["value"],
                        "metric_scene_name": args.metric_scene_name,
                        "status": route["status"],
                        "reason": route["reason"],
                        "active_context_id": entry.context_id,
                        "active_context_name": entry.context_name,
                        "active_context_lifecycle": entry.lifecycle,
                        "best_context_id": route["best_context_id"],
                        "best_score": route["best_score"],
                        "second_score": route["second_score"],
                        "best_confidence": route["best_confidence"],
                        "second_confidence": route["second_confidence"],
                        "historical_evidence": route["historical_evidence"],
                        "novelty_strength": route["novelty_strength"],
                        "novelty_evidence": route["novelty_evidence"],
                        "candidate_evidence": route["candidate_evidence"],
                        "raw_projected_feature_json": json.dumps(
                            route["raw_feature"].tolist()
                        ),
                        "smoothed_projected_feature_json": json.dumps(
                            route["smoothed_feature"].tolist()
                        ),
                        "scores_json": json.dumps(route["scores"], sort_keys=True),
                        "thresholds_json": json.dumps(
                            route["density_thresholds"],
                            sort_keys=True,
                        ),
                        "density_thresholds_json": json.dumps(
                            route["density_thresholds"],
                            sort_keys=True,
                        ),
                        "routing_thresholds_json": json.dumps(
                            route["routing_thresholds"],
                            sort_keys=True,
                        ),
                        "near_thresholds_json": json.dumps(
                            route["near_thresholds"],
                            sort_keys=True,
                        ),
                    },
                )
                return adapter.forward_with_context(encoded, entry.model_context_name)

            pipeline.network.forward = types.MethodType(
                automatic_forward,
                pipeline.network,
            )

        rows = pipeline.validate_kitti(
            epoch=0,
            list_conf_thr=[float(args.conf_thr)],
            is_subset=False,
        )
        score = pipeline.pick_best_metric_score(rows)
        pipeline.network.forward = original_forward
        summary = {
            "mode": args.mode,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config": str(Path(args.config).resolve()),
            "metric_scene_name": args.metric_scene_name,
            "context_id": args.context_id,
            "initial_context_id": args.initial_context_id,
            "score": None if score is None else float(score),
            "routing_csv": str(routing_path) if args.mode == "auto" else None,
        }
        with (run_path / "evaluation_summary.yml").open("w") as output:
            yaml.safe_dump(summary, output, sort_keys=False)
        print(f"* Evaluation score = {score}")
        print(f"* Evaluation output = {run_path}")
    finally:
        if pipeline is not None:
            close_writers(pipeline)
        if os.path.exists(runtime_path):
            os.unlink(runtime_path)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
