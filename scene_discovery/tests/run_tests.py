#!/usr/bin/env python3
"""Run the lightweight scene-discovery tests without external test frameworks."""

from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent
for path in (TEST_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    modules = [
        "test_descriptors",
        "test_preprocessing_temporal",
        "test_clustering",
        "test_attributes",
        "test_attribute_registry",
        "test_attribute_protocol",
        "test_semantic_decoder",
        "test_online_protocol",
        "test_memory_context",
        "test_causal_context",
        "test_weather_protocol",
        "test_adaptive_weather_context",
        "test_feature_bank",
    ]
    failures = []
    count = 0
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            count += 1
            try:
                function()
                print(f"PASS {module_name}.{name}")
            except Exception:
                failures.append(f"{module_name}.{name}")
                traceback.print_exc()
    print(f"{count - len(failures)}/{count} tests passed")
    if failures:
        print("Failures: " + ", ".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
