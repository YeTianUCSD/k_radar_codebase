#!/usr/bin/env python3
"""Small dependency-free test runner for the factorized MLP experiment."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
for path in (ROOT, REPOSITORY_ROOT / "scene_discovery"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main():
    tests = []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        module = importlib.import_module(path.stem)
        tests.extend(
            (path.stem, name, value)
            for name, value in vars(module).items()
            if name.startswith("test_") and callable(value)
        )
    for module, name, function in tests:
        function()
        print(f"PASS {module}.{name}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
