#!/usr/bin/env python3
"""Build deterministic few-shot manifests without copying descriptor arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPOSITORY_ROOT / "scene_discovery"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from factorized_mlp.data import load_split_arrays  # noqa: E402
from factorized_mlp.fewshot_protocol import build_fewshot_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-budget", type=int, default=30)
    parser.add_argument("--support-pool", type=int, default=30)
    parser.add_argument("--validation-frames", type=int, default=20)
    parser.add_argument("--guard-frames", type=int, default=10)
    args = parser.parse_args()
    _, train_index = load_split_arrays(args.descriptor_bank, "train", ["camera"], "mean_std")
    _, test_index = load_split_arrays(args.descriptor_bank, "test", ["camera"], "mean_std")
    manifest = build_fewshot_manifest(
        train_index, test_index, args.support_budget, args.support_pool,
        args.validation_frames, args.guard_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(manifest.groupby(["source_split", "role", "sequence"]).size().to_string())
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
