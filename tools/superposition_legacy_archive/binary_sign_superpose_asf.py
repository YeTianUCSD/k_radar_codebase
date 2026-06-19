import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime

import torch
import yaml


FILE = os.path.abspath(__file__)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"* {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and evaluate a binary-sign superposition bundle for ASF fuser/head checkpoints."
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base seq1 checkpoint, e.g. model_16.pt")
    parser.add_argument("--target_model", type=str, required=True,
                        help="Target seq58 checkpoint, e.g. best.checkpoint from online fuser_head")
    parser.add_argument("--base_context", type=str, default="seq1",
                        help="Name for the base context")
    parser.add_argument("--target_context", type=str, default="seq58",
                        help="Name for the bound residual context")
    parser.add_argument("--modules", type=str, nargs="+", default=["fuser", "head"],
                        help="Top-level state_dict prefixes to superpose")
    parser.add_argument("--seed", type=int, default=20260612,
                        help="Seed used to generate deterministic binary sign contexts")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output root for bundle, materialized checkpoints, and eval logs")
    parser.add_argument("--bundle_name", type=str, default="asf_superposed_bundle",
                        help="Name prefix for generated artifacts")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only build bundle and materialized checkpoints")

    parser.add_argument("--seq1_config", type=str, default="./configs/ASF_v2_0_seq1.yml",
                        help="Config used to evaluate the recovered base context")
    parser.add_argument("--seq58_config", type=str, default="./configs/ASF_v2_0_seq58_eval.yml",
                        help="Config used to evaluate the recovered target context")
    parser.add_argument("--conf_thr", type=float, nargs="+", default=[0.3],
                        help="Confidence thresholds passed to validate_kitti")
    parser.add_argument("--best_metric_cls", type=str, default="auto")
    parser.add_argument("--best_metric_kind", choices=["bev", "3d"], default="3d")
    parser.add_argument("--best_metric_ious", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--best_metric_conf", type=float, default=0.3)
    return parser.parse_args()


def make_runtime_config(path_config, output_root, run_name, best_metric):
    with open(path_config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["GENERAL"]["NAME"] = run_name
    cfg["GENERAL"]["LOGGING"]["PATH_LOGGING"] = output_root
    cfg["GENERAL"]["LOGGING"]["IS_SAVE_MODEL"] = False
    cfg["VAL"]["IS_VALIDATE"] = True
    cfg["GENERAL"]["LOGGING"]["BEST_METRIC"] = best_metric

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yml",
        prefix="kradar_superpose_eval_",
        delete=False,
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return tmp.name


def load_state_dict(path):
    log(f"Loading checkpoint: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def is_selected_key(key, modules):
    return any(key.startswith(f"{prefix}.") for prefix in modules)


def stable_seed(name, seed):
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def make_sign_context_like(tensor, key, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(key, seed))
    bits = torch.randint(0, 2, tensor.shape, generator=generator, dtype=torch.int64)
    signs = bits.to(torch.float32).mul_(2.0).sub_(1.0)
    return signs.to(dtype=tensor.dtype)


def build_superposition_bundle(base_state, target_state, modules, target_context, seed):
    log(
        "Building binary-sign bundle for modules: "
        + ", ".join(modules)
        + f" with target context '{target_context}'"
    )
    selected_keys = [key for key in sorted(base_state.keys()) if is_selected_key(key, modules)]
    if selected_keys != [key for key in sorted(target_state.keys()) if is_selected_key(key, modules)]:
        raise RuntimeError("Base and target selected keys do not match")

    bound_deltas = {}
    contexts = {}
    stats_rows = []
    for key in selected_keys:
        base_tensor = base_state[key]
        target_tensor = target_state[key]

        if base_tensor.shape != target_tensor.shape:
            raise RuntimeError(f"Shape mismatch for key {key}: {tuple(base_tensor.shape)} vs {tuple(target_tensor.shape)}")

        if not torch.is_floating_point(base_tensor):
            continue

        delta = (target_tensor - base_tensor).detach().cpu()
        context = make_sign_context_like(delta, f"{target_context}:{key}", seed)
        bound_delta = delta * context

        bound_deltas[key] = bound_delta
        contexts[key] = context
        stats_rows.append({
            "key": key,
            "numel": delta.numel(),
            "mean_abs_delta": float(delta.abs().mean()),
            "max_abs_delta": float(delta.abs().max()),
            "l2_delta": float(torch.linalg.vector_norm(delta.reshape(-1), ord=2)),
        })

    log(f"Collected {len(bound_deltas)} floating-point tensors for bound residual storage")
    return {
        "format_version": 1,
        "method": "binary_sign_binding",
        "modules": list(modules),
        "contexts": {
            "base": "identity",
            target_context: "binary_sign",
        },
        "bound_deltas": bound_deltas,
        "context_tensors": contexts,
        "stats_rows": stats_rows,
    }


def materialize_state_dict(base_state, bundle, context_name):
    log(f"Materializing state_dict for context: {context_name}")
    recovered = {}
    for key, value in base_state.items():
        recovered[key] = value.detach().cpu().clone()

    if context_name == "base":
        return recovered

    if context_name not in bundle["contexts"]:
        raise KeyError(f"Unknown context {context_name}")

    for key, bound_delta in bundle["bound_deltas"].items():
        context = bundle["context_tensors"][key]
        recovered[key] = recovered[key] + (bound_delta * context).to(dtype=recovered[key].dtype)
    return recovered


def tensor_diff_stats(lhs, rhs, modules):
    rows = []
    for key in sorted(lhs.keys()):
        if not is_selected_key(key, modules):
            continue
        if not torch.is_floating_point(lhs[key]):
            continue
        diff = (lhs[key] - rhs[key]).detach().cpu().float()
        rows.append({
            "key": key,
            "numel": diff.numel(),
            "mean_abs_diff": float(diff.abs().mean()),
            "max_abs_diff": float(diff.abs().max()),
            "l2_diff": float(torch.linalg.vector_norm(diff.reshape(-1), ord=2)),
        })
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def save_checkpoint(path, state_dict, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    torch.save(meta, f"{path}.state")


def summarize_metrics(pline, eval_rows, eval_name, config_path, checkpoint_path, timing):
    score = pline.pick_best_metric_score(eval_rows)
    metric_cfg = pline.best_metric_cfg
    selected = pline.select_best_metric_rows(eval_rows)
    metric_values = {}
    for row in eval_rows:
        cls_name = str(row["cls"]).lower()
        iou_key = str(row["iou"]).replace(".", "_")
        prefix = f"{cls_name}_iou_{iou_key}"
        metric_values[f"{prefix}_has_gt"] = int(bool(row.get("has_gt", True)))
        metric_values[f"{prefix}_bev"] = row["bev"]
        metric_values[f"{prefix}_3d"] = row["3d"]

    summary = {
        "eval_name": eval_name,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "checkpoint_name": os.path.basename(checkpoint_path),
        "score": "" if score is None else score,
        "score_kind": metric_cfg["kind"],
        "selected_metric_count": len(selected),
        "selected_metrics": ";".join(
            f"{row['cls']}/{row['iou']}/{row['3d']:.6f}" for row in selected
        ),
        "selected_score_values": ";".join(
            f"{float(row[metric_cfg['kind']]):.6f}" for row in selected
        ),
        "log_dir": pline.path_log,
        "timestamp": timestamp_now(),
    }
    summary.update(timing)
    summary.update(metric_values)
    return summary


def evaluate_checkpoint(config_path, checkpoint_path, output_root, run_name, args):
    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    best_metric = {
        "CLS": args.best_metric_cls,
        "KIND": args.best_metric_kind,
        "IOUS": [float(x) for x in args.best_metric_ious],
        "CONF_THR": float(args.best_metric_conf),
        "REDUCE": "mean",
        "ONLY_CLASSES_WITH_GT": True,
    }

    total_time_start = time.time()
    log(f"Preparing evaluation run '{run_name}'")
    runtime_config = make_runtime_config(config_path, output_root, run_name, best_metric)

    setup_time_start = time.time()
    log(f"Initializing pipeline with config: {config_path}")
    pline = PipelineDetection_v1_0(runtime_config, mode="test")
    if not hasattr(pline, "val_keyword"):
        pline.set_validate()
    setup_time_sec = time.time() - setup_time_start

    load_time_start = time.time()
    log(f"Loading materialized checkpoint for eval: {checkpoint_path}")
    pline.load_dict_model(checkpoint_path)
    load_model_time_sec = time.time() - load_time_start
    pline.network.eval()
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, "executed_code.txt"))

    eval_time_start = time.time()
    log(f"Running full evaluation for '{run_name}'")
    eval_rows = pline.validate_kitti(
        epoch=None,
        list_conf_thr=args.conf_thr,
        is_subset=False,
    )
    eval_time_sec = time.time() - eval_time_start
    total_time_sec = time.time() - total_time_start

    summary = summarize_metrics(
        pline,
        eval_rows,
        run_name,
        config_path,
        checkpoint_path,
        timing={
            "setup_time_sec": setup_time_sec,
            "load_model_time_sec": load_model_time_sec,
            "eval_time_sec": eval_time_sec,
            "total_time_sec": total_time_sec,
        },
    )
    log(
        f"Finished '{run_name}': "
        f"score={summary['score']}, total_time_sec={total_time_sec:.2f}"
    )

    for writer_name in ("log_train_iter", "log_train_epoch", "log_test"):
        writer = getattr(pline, writer_name, None)
        if writer is not None:
            writer.close()

    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    log("Starting ASF binary-sign superposition pipeline")
    log(f"Output root: {args.output_root}")
    log(f"Base context: {args.base_context}")
    log(f"Target context: {args.target_context}")

    base_state = load_state_dict(args.base_model)
    target_state = load_state_dict(args.target_model)

    bundle = build_superposition_bundle(
        base_state=base_state,
        target_state=target_state,
        modules=args.modules,
        target_context=args.target_context,
        seed=args.seed,
    )

    bundle_path = os.path.join(args.output_root, f"{args.bundle_name}.pt")
    bundle_meta_path = os.path.join(args.output_root, f"{args.bundle_name}.yml")
    diff_csv_path = os.path.join(args.output_root, f"{args.bundle_name}_delta_stats.csv")

    torch.save({
        "base_model_path": os.path.abspath(args.base_model),
        "target_model_path": os.path.abspath(args.target_model),
        "base_state_dict": base_state,
        "bundle": {
            "format_version": bundle["format_version"],
            "method": bundle["method"],
            "modules": bundle["modules"],
            "contexts": bundle["contexts"],
            "bound_deltas": bundle["bound_deltas"],
            "context_tensors": bundle["context_tensors"],
        },
    }, bundle_path)
    log(f"Saved superposed bundle: {bundle_path}")
    write_yaml(bundle_meta_path, {
        "timestamp": timestamp_now(),
        "base_model_path": os.path.abspath(args.base_model),
        "target_model_path": os.path.abspath(args.target_model),
        "base_context": args.base_context,
        "target_context": args.target_context,
        "modules": args.modules,
        "method": bundle["method"],
        "seed": args.seed,
        "bundle_path": bundle_path,
    })
    log(f"Saved bundle metadata: {bundle_meta_path}")
    write_csv(diff_csv_path, bundle["stats_rows"])
    log(f"Saved delta statistics: {diff_csv_path}")

    materialized_dir = os.path.join(args.output_root, "materialized")
    seq1_state = materialize_state_dict(base_state, bundle, "base")
    seq58_state = materialize_state_dict(base_state, bundle, args.target_context)

    seq1_ckpt = os.path.join(materialized_dir, f"{args.bundle_name}_{args.base_context}.checkpoint")
    seq58_ckpt = os.path.join(materialized_dir, f"{args.bundle_name}_{args.target_context}.checkpoint")
    save_checkpoint(seq1_ckpt, seq1_state, {
        "context": args.base_context,
        "source": "base",
        "bundle_path": bundle_path,
    })
    log(f"Saved materialized seq1 checkpoint: {seq1_ckpt}")
    save_checkpoint(seq58_ckpt, seq58_state, {
        "context": args.target_context,
        "source": "base_plus_unbound_delta",
        "bundle_path": bundle_path,
    })
    log(f"Saved materialized seq58 checkpoint: {seq58_ckpt}")

    reconstruction_dir = os.path.join(args.output_root, "reconstruction_checks")
    log("Running reconstruction consistency checks")
    seq58_vs_target = tensor_diff_stats(seq58_state, target_state, args.modules)
    seq1_vs_base = tensor_diff_stats(seq1_state, base_state, args.modules)
    write_csv(os.path.join(reconstruction_dir, "seq58_vs_target.csv"), seq58_vs_target)
    write_csv(os.path.join(reconstruction_dir, "seq1_vs_base.csv"), seq1_vs_base)

    summary_path = os.path.join(reconstruction_dir, "summary.txt")
    with open(summary_path, "w") as f:
        max_seq58 = max((row["max_abs_diff"] for row in seq58_vs_target), default=0.0)
        max_seq1 = max((row["max_abs_diff"] for row in seq1_vs_base), default=0.0)
        f.write(f"seq58_recovered_vs_target_max_abs_diff: {max_seq58:.10f}\n")
        f.write(f"seq1_recovered_vs_base_max_abs_diff: {max_seq1:.10f}\n")
        f.write(f"bundle_path: {bundle_path}\n")
        f.write(f"seq1_checkpoint: {seq1_ckpt}\n")
        f.write(f"seq58_checkpoint: {seq58_ckpt}\n")
    log(f"Saved reconstruction summary: {summary_path}")

    if args.skip_eval:
        log("Skipping evaluation as requested")
        return

    eval_root = os.path.join(args.output_root, "eval")
    log("Starting evaluation phase")
    summary_rows = []
    summary_rows.append(
        evaluate_checkpoint(
            config_path=args.seq1_config,
            checkpoint_path=seq1_ckpt,
            output_root=eval_root,
            run_name=f"{args.bundle_name}_{args.base_context}_on_seq1",
            args=args,
        )
    )
    summary_rows.append(
        evaluate_checkpoint(
            config_path=args.seq58_config,
            checkpoint_path=seq58_ckpt,
            output_root=eval_root,
            run_name=f"{args.bundle_name}_{args.target_context}_on_seq58",
            args=args,
        )
    )
    summary_csv = os.path.join(args.output_root, "eval_summary.csv")
    write_csv(summary_csv, summary_rows)
    log(f"Saved evaluation summary: {summary_csv}")
    log("ASF binary-sign superposition pipeline finished")


if __name__ == "__main__":
    main()
