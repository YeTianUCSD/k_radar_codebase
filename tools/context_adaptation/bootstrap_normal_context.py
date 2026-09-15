"""Bootstrap the normal Context Memory from a trained single-context PSP model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from context_adaptation.bandwidth import resolve_bandwidth
from context_adaptation.checkpoint import save_context_checkpoint
from context_adaptation.model_adapter import ContextAwareModelAdapter
from context_adaptation.runtime import (
    build_context_memory,
    build_projector,
    build_router,
    load_model_checkpoint,
    load_yaml,
    write_single_context_runtime_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a normal-scene Context Memory from frozen ASF encoders."
    )
    parser.add_argument("--config", required=True, help="Base single-sequence PSP config.")
    parser.add_argument("--init_model", required=True, help="Trained normal PSP checkpoint.")
    parser.add_argument(
        "--context_config",
        default="./configs/context_adaptation/auto_context_rp2.yml",
        help="Context feature, projection, KDE, and router config.",
    )
    parser.add_argument("--output", required=True, help="Output context checkpoint.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum normal frames used in each pass. -1 uses the full split.",
    )
    parser.add_argument(
        "--base_context_name",
        default=None,
        help="Logical Context Memory name. Defaults to BASE_CONTEXT.CONTEXT_NAME.",
    )
    parser.add_argument(
        "--base_model_context_name",
        default=None,
        help="PSP scene name. Defaults to BASE_CONTEXT.MODEL_CONTEXT_NAME.",
    )
    parser.add_argument(
        "--allow_incompatible_checkpoint",
        action="store_true",
        help="Allow missing or unexpected model keys during initialization.",
    )
    return parser.parse_args()


def build_loader(dataset: object, args: argparse.Namespace) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )


def trim_to_remaining(values: np.ndarray, seen: int, maximum: int) -> np.ndarray:
    if maximum <= 0:
        return values
    remaining = maximum - seen
    return values[: max(0, remaining)]


def close_writers(pipeline: object) -> None:
    for writer_name in ("log_train_iter", "log_train_epoch", "log_test"):
        writer = getattr(pipeline, writer_name, None)
        if writer is not None:
            writer.close()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("max_samples must be -1 or a positive integer.")

    context_config = load_yaml(args.context_config)
    base_config = context_config["BASE_CONTEXT"]
    context_name = args.base_context_name or str(base_config["CONTEXT_NAME"])
    model_context_name = args.base_model_context_name or str(
        base_config["MODEL_CONTEXT_NAME"]
    )
    runtime_path, _ = write_single_context_runtime_config(
        args.config,
        model_context_name=model_context_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        enable_logging=False,
        enable_validation=False,
    )

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pipeline = None
    try:
        mode = "train" if args.split == "train" else "test"
        pipeline = PipelineDetection_v1_0(path_cfg=runtime_path, mode=mode)
        missing, unexpected = load_model_checkpoint(
            pipeline.network,
            args.init_model,
            strict=False,
        )
        if (missing or unexpected) and not args.allow_incompatible_checkpoint:
            raise RuntimeError(
                "Initial model is incompatible with the single-context runtime model. "
                f"Missing keys: {missing[:12]}; unexpected keys: {unexpected[:12]}."
            )
        print(
            f"* Loaded model with {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys."
        )

        dataset = pipeline.dataset_train if args.split == "train" else pipeline.dataset_test
        adapter = ContextAwareModelAdapter(pipeline.network)
        projector = build_projector(context_config)
        pipeline.network.eval()

        fitted_samples = 0
        first_loader = build_loader(dataset, args)
        for batch in tqdm(first_loader, desc="* Fit normal descriptor statistics"):
            encoded = adapter.encode(batch)
            descriptors = projector.extract_descriptor(encoded)
            descriptors = trim_to_remaining(descriptors, fitted_samples, args.max_samples)
            if descriptors.shape[0] == 0:
                break
            projector.update_normalization(descriptors)
            fitted_samples += int(descriptors.shape[0])
            del encoded, batch
            if args.max_samples > 0 and fitted_samples >= args.max_samples:
                break
        projector.finalize()
        print(f"* Finalized projector: {projector.info}")

        projected_chunks = []
        projected_samples = 0
        second_loader = build_loader(dataset, args)
        for batch in tqdm(second_loader, desc="* Build normal KDE"):
            encoded = adapter.encode(batch)
            descriptors = projector.extract_descriptor(encoded)
            descriptors = trim_to_remaining(
                descriptors,
                projected_samples,
                args.max_samples,
            )
            if descriptors.shape[0] == 0:
                break
            projected = projector.transform_descriptor(descriptors)
            projected_chunks.append(np.asarray(projected, dtype=projector.dtype))
            projected_samples += int(descriptors.shape[0])
            del encoded, batch
            if args.max_samples > 0 and projected_samples >= args.max_samples:
                break
        if not projected_chunks:
            raise RuntimeError("No normal projected features were extracted.")
        normal_features = np.concatenate(projected_chunks, axis=0)
        if normal_features.ndim != 2:
            raise RuntimeError(f"Unexpected projected feature shape: {normal_features.shape}")

        bandwidth, bandwidth_info = resolve_bandwidth(
            context_config["KDE"], normal_features
        )
        memory = build_context_memory(
            context_config,
            input_dim=projector.projection_dim,
            bandwidth_override=bandwidth,
        )
        entry = memory.create_context(
            normal_features,
            context_name=context_name,
            model_context_name=model_context_name,
            created_step=0,
            activate=True,
        )
        router = build_router(
            context_config,
            input_dim=projector.projection_dim,
            candidate_bandwidth_override=bandwidth,
        )
        router.set_active_context(entry.context_id)

        output = save_context_checkpoint(
            args.output,
            network=pipeline.network,
            context_memory=memory,
            router=router,
            projector=projector,
            optimizer=None,
            step_idx=0,
            update_idx=0,
            metadata={
                "stage": "bootstrap_normal_context",
                "source_config": str(Path(args.config).resolve()),
                "source_model": str(Path(args.init_model).resolve()),
                "context_config": str(Path(args.context_config).resolve()),
                "split": args.split,
                "normal_samples": int(normal_features.shape[0]),
                "bandwidth": bandwidth_info,
            },
        )
        print(f"* Normal context ID = {entry.context_id}")
        print(f"* Normal model context = {entry.model_context_name}")
        print(f"* KDE bandwidth = {bandwidth:.6f} ({bandwidth_info['mode']})")
        print(f"* Normal KDE threshold = {entry.threshold:.6f}")
        print(f"* Saved context checkpoint = {output}")
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
