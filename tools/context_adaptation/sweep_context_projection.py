#!/usr/bin/env python3
"""Sweep projection dimensions and automatic-bandwidth scales on cached descriptors."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.context_adaptation.replay_context_routing import (
    bootstrap_runtime,
    load_descriptor_rows,
    route_phase,
    write_rows,
)
from tools.context_adaptation.summarize_multiscene_routing import summarize, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context_config", type=Path, required=True)
    parser.add_argument("--discovery_csv", type=Path, required=True)
    parser.add_argument("--return_csv", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--expected_labels", nargs="+", required=True)
    parser.add_argument("--base_label", default="seq5")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--projection_seeds", nargs="+", type=int, default=[20260812])
    parser.add_argument("--bandwidth_scales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--ignore_transition_frames", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--disable_return_creation", action="store_true")
    return parser.parse_args()


def validate_grid(args: argparse.Namespace) -> None:
    if not args.dimensions or any(value <= 0 for value in args.dimensions):
        raise ValueError("dimensions must be positive")
    if not args.projection_seeds:
        raise ValueError("projection_seeds must not be empty")
    if not args.bandwidth_scales or any(value <= 0 for value in args.bandwidth_scales):
        raise ValueError("bandwidth_scales must be positive")


def run_name(dimension: int, seed: int, scale: float) -> str:
    scale_name = f"{scale:g}".replace(".", "p")
    return f"rp{dimension:02d}_seed{seed}_bw{scale_name}"


def main() -> None:
    args = parse_args()
    validate_grid(args)
    with args.context_config.expanduser().resolve().open() as source:
        base_config = yaml.safe_load(source)
    discovery_source = load_descriptor_rows(args.discovery_csv.expanduser().resolve())
    return_source = load_descriptor_rows(args.return_csv.expanduser().resolve())
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for dimension in args.dimensions:
        for seed in args.projection_seeds:
            for scale in args.bandwidth_scales:
                name = run_name(dimension, seed, scale)
                print(f"[SWEEP] {name}", flush=True)
                config = copy.deepcopy(base_config)
                config["PROJECTION"]["DIM"] = int(dimension)
                config["PROJECTION"]["SEED"] = int(seed)
                config["KDE"]["BANDWIDTH_MODE"] = "median"
                config["KDE"]["BANDWIDTH_SCALE"] = float(scale)
                projector, memory, router, bootstrap_samples, bandwidth = bootstrap_runtime(
                    config, discovery_source, str(args.base_label)
                )
                discovery, step = route_phase(
                    discovery_source, phase="discovery", projector=projector,
                    memory=memory, router=router, config=config, start_step=0,
                    allow_context_creation=True, max_steps=args.max_steps,
                )
                returned, _ = route_phase(
                    return_source, phase="return", projector=projector,
                    memory=memory, router=router, config=config, start_step=step,
                    allow_context_creation=not args.disable_return_creation,
                    max_steps=args.max_steps,
                )
                run_dir = output_root / name
                write_rows(run_dir / "discovery_routing_log.csv", discovery)
                write_rows(run_dir / "return_routing_log.csv", returned)
                metrics = summarize(
                    discovery, returned, expected_labels=args.expected_labels,
                    base_label=str(args.base_label),
                    ignore_transition_frames=int(args.ignore_transition_frames),
                )
                metrics.update({
                    "sweep_name": name,
                    "projection_dim": int(dimension),
                    "projection_seed": int(seed),
                    "bandwidth": bandwidth,
                    "bootstrap_samples": int(bootstrap_samples),
                })
                write_outputs(metrics, run_dir / "metrics")
                summary_rows.append({
                    "run_name": name,
                    "projection_dim": dimension,
                    "projection_seed": seed,
                    "bandwidth_scale": scale,
                    "resolved_bandwidth": bandwidth["resolved_bandwidth"],
                    "num_contexts": metrics["discovery"]["num_contexts"],
                    "context_purity": metrics["discovery"]["context_purity"],
                    "merge_leakage": metrics["discovery"]["merge_leakage"],
                    "return_macro_accuracy": metrics["return"]["macro_return_compatible_accuracy"],
                    "return_minimum_scene_accuracy": metrics["return"]["minimum_label_accuracy"],
                    "return_decision_coverage": metrics["return"]["decision_coverage"],
                    "memory_update_contamination": max(
                        metrics["discovery"]["memory_update_safety"]["contamination_rate"],
                        metrics["return"]["memory_update_safety"]["contamination_rate"],
                    ),
                    "overall_pass": metrics["pass_criteria"]["overall_pass"],
                })

    summary_path = output_root / "sweep_summary.csv"
    with summary_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    ranking = sorted(
        summary_rows,
        key=lambda row: (
            int(
                float(row["context_purity"]) < 0.95
                or float(row["memory_update_contamination"]) > 0.01
            ),
            -float(row["return_macro_accuracy"]),
            -float(row["return_minimum_scene_accuracy"]),
            -float(row["context_purity"]),
            int(row["projection_dim"]),
        ),
    )
    with (output_root / "ranking.yml").open("w") as destination:
        yaml.safe_dump(ranking, destination, sort_keys=False)
    print(f"[DONE] {len(summary_rows)} replay runs; summary={summary_path}")


if __name__ == "__main__":
    main()
