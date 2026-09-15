#!/usr/bin/env python3
"""Validate the structure and row alignment of an encoder FeatureBank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_discovery.common import atomic_write_json, utc_now  # noqa: E402
from scene_discovery.feature_bank import MODALITIES, FeatureBank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-bank",
        type=Path,
        default=REPOSITORY_ROOT / "results/EncoderFeatureBank/10scenes_fp16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results/SceneDiscovery/feature_bank_validation.json",
    )
    parser.add_argument("--finite-mode", choices=("none", "sample", "full"), default="sample")
    parser.add_argument("--finite-sample-rows", type=int, default=8)
    return parser.parse_args()


def finite_rows(array: np.ndarray, mode: str, count: int) -> np.ndarray:
    if mode == "full":
        return np.arange(array.shape[0])
    if mode == "sample" and array.shape[0]:
        return np.unique(np.linspace(0, array.shape[0] - 1, min(count, array.shape[0]), dtype=int))
    return np.empty((0,), dtype=int)


def main() -> None:
    args = parse_args()
    bank = FeatureBank(args.feature_bank)
    errors: List[str] = []
    warnings: List[str] = []
    partitions: List[Dict[str, Any]] = []
    all_indices: List[pd.DataFrame] = []

    if bank.metadata.get("status") != "complete":
        warnings.append(f"Global status is {bank.metadata.get('status')!r}, not 'complete'")

    for partition in bank.partitions():
        prefix = f"seq{partition.sequence}/{partition.split}"
        record: Dict[str, Any] = {
            "sequence": partition.sequence,
            "split": partition.split,
            "expected_samples": partition.sample_count,
            "checks": {},
            "features": {},
        }
        complete = partition.path / "COMPLETE.json"
        record["checks"]["complete_marker"] = complete.is_file()
        if not complete.is_file():
            errors.append(f"{prefix}: missing COMPLETE.json")

        try:
            valid = partition.load_valid()
            valid_count = int(np.count_nonzero(valid))
            record["valid_count"] = valid_count
            record["checks"]["valid_shape"] = tuple(valid.shape) == (partition.sample_count,)
            record["checks"]["all_valid"] = valid_count == partition.sample_count
            if not record["checks"]["valid_shape"] or not record["checks"]["all_valid"]:
                errors.append(f"{prefix}: invalid validity bitmap")
        except Exception as exc:
            errors.append(f"{prefix}: cannot read valid.npy: {exc}")

        for modality in MODALITIES:
            try:
                array = partition.load_array(modality)
                feature_record = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "size_bytes": partition.array_path(modality).stat().st_size,
                }
                if array.ndim != 4 or array.shape[0] != partition.sample_count:
                    errors.append(f"{prefix}/{modality}: unexpected shape {array.shape}")
                if array.dtype != np.float16:
                    errors.append(f"{prefix}/{modality}: expected float16, got {array.dtype}")
                indices = finite_rows(array, args.finite_mode, args.finite_sample_rows)
                feature_record["finite_rows_checked"] = int(len(indices))
                feature_record["finite"] = bool(np.isfinite(array[indices]).all()) if len(indices) else None
                if feature_record["finite"] is False:
                    errors.append(f"{prefix}/{modality}: NaN or Inf found")
                record["features"][modality] = feature_record
            except Exception as exc:
                errors.append(f"{prefix}/{modality}: cannot read feature: {exc}")

        try:
            manifest = partition.load_manifest()
            record["manifest_rows"] = len(manifest)
            record["checks"]["manifest_rows"] = len(manifest) == partition.sample_count
            record["checks"]["row_index"] = np.array_equal(
                manifest["row_index"].to_numpy(), np.arange(len(manifest))
            )
            record["checks"]["sequence"] = set(manifest["sequence"]) == {partition.sequence}
            record["checks"]["split"] = set(manifest["split"]) == {partition.split}
            if not all(record["checks"].get(key, False) for key in ("manifest_rows", "row_index", "sequence", "split")):
                errors.append(f"{prefix}: manifest alignment check failed")
            subset = manifest[["sample_id", "sequence", "split"]].copy()
            subset["partition"] = prefix
            all_indices.append(subset)
        except Exception as exc:
            errors.append(f"{prefix}: cannot validate manifest: {exc}")

        metadata_path = partition.path / "sample_metadata.jsonl"
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata_rows = sum(1 for line in stream if line.strip())
            record["metadata_rows"] = metadata_rows
            record["checks"]["metadata_rows"] = metadata_rows == partition.sample_count
            if metadata_rows != partition.sample_count:
                errors.append(f"{prefix}: sample_metadata.jsonl has {metadata_rows} rows")
        except Exception as exc:
            errors.append(f"{prefix}: cannot validate sample metadata: {exc}")
        partitions.append(record)

    combined = pd.concat(all_indices, ignore_index=True) if all_indices else pd.DataFrame()
    duplicate_ids = [] if combined.empty else sorted(combined.loc[combined.sample_id.duplicated(False), "sample_id"].unique())
    if duplicate_ids:
        errors.append(f"Duplicate sample_id values found: {len(duplicate_ids)}")

    payload = {
        "schema_version": "scene-discovery-feature-validation/v1",
        "created_utc": utc_now(),
        "feature_bank": str(bank.root),
        "feature_bank_status": bank.metadata.get("status"),
        "finite_check": {"mode": args.finite_mode, "sample_rows": args.finite_sample_rows},
        "status": "pass" if not errors else "fail",
        "partition_count": len(partitions),
        "sample_count": int(sum(item["expected_samples"] for item in partitions)),
        "duplicate_sample_ids": duplicate_ids,
        "errors": errors,
        "warnings": warnings,
        "partitions": partitions,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(f"Validation {payload['status']}: {payload['partition_count']} partitions, {payload['sample_count']} samples")
    print(f"Report: {args.output.resolve()}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

