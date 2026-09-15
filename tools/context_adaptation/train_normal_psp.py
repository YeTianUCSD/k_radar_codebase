"""Train the initial PSP model with exactly one normal context."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_adaptation.runtime import load_yaml, write_single_context_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PSP ASF model with one normal scene context."
    )
    parser.add_argument("--config", required=True, help="Base Seq1 PSP config.")
    parser.add_argument(
        "--context_config",
        default="./configs/context_adaptation/auto_context_rp2.yml",
    )
    parser.add_argument("--output_root", default="./results/ContextAdaptation")
    parser.add_argument("--run_name", default="train_normal_psp")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--full_eval_every", type=int, default=1)
    parser.add_argument("--best_metric_cls", default="auto")
    parser.add_argument("--best_metric_kind", choices=["bev", "3d"], default="3d")
    parser.add_argument("--best_metric_ious", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--best_metric_conf", type=float, default=0.3)
    parser.add_argument("--skip_final_eval", action="store_true")
    return parser.parse_args()


def close_writers(pipeline: object) -> None:
    for writer_name in ("log_train_iter", "log_train_epoch", "log_test"):
        writer = getattr(pipeline, writer_name, None)
        if writer is not None:
            writer.close()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")
    if args.full_eval_every <= 0:
        raise ValueError("full_eval_every must be positive.")

    context_config = load_yaml(args.context_config)
    model_context_name = str(
        context_config["BASE_CONTEXT"]["MODEL_CONTEXT_NAME"]
    )
    runtime_path, runtime_config = write_single_context_runtime_config(
        args.config,
        model_context_name=model_context_name,
        output_root=args.output_root,
        run_name=args.run_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        enable_logging=True,
        enable_validation=True,
    )
    runtime_config["OPTIMIZER"]["MAX_EPOCH"] = int(args.epochs)
    runtime_config["VAL"]["IS_CONSIDER_VAL_SUBSET"] = False
    runtime_config["VAL"]["VAL_PER_EPOCH_FULL"] = int(args.full_eval_every)
    runtime_config["GENERAL"]["LOGGING"]["IS_SAVE_MODEL"] = True
    runtime_config["GENERAL"]["LOGGING"]["INTERVAL_EPOCH_MODEL"] = 1
    runtime_config["GENERAL"]["LOGGING"]["INTERVAL_EPOCH_UTIL"] = 1
    runtime_config["GENERAL"]["LOGGING"]["BEST_METRIC"] = {
        "CLS": str(args.best_metric_cls),
        "KIND": str(args.best_metric_kind),
        "IOUS": [float(value) for value in args.best_metric_ious],
        "CONF_THR": float(args.best_metric_conf),
        "REDUCE": "mean",
        "ONLY_CLASSES_WITH_GT": True,
    }
    with open(runtime_path, "w") as output:
        yaml.safe_dump(runtime_config, output, sort_keys=False)

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pipeline = None
    try:
        pipeline = PipelineDetection_v1_0(path_cfg=runtime_path, mode="train")
        shutil.copy2(os.path.realpath(__file__), Path(pipeline.path_log) / "executed_code.txt")
        print(f"* Normal model context = {model_context_name}")
        print(
            "* Scene list = "
            f"{runtime_config['MODEL']['SUPERPOSITION']['SCENE_LIST']}"
        )
        pipeline.train_network()
        if not args.skip_final_eval:
            pipeline.validate_kitti_conditional(
                list_conf_thr=[float(args.best_metric_conf)],
                is_subset=False,
                is_print_memory=False,
            )
        print(f"* Normal PSP training output = {pipeline.path_log}")
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
