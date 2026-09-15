#!/usr/bin/env python3
"""Train and evaluate the V6 independent spatial-lighting attribute model."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_sample_efficient_hybrid_v4 import main  # noqa: E402


if __name__ == "__main__":
    main(PROJECT_ROOT / "configs/spatial_lighting_v6.yml")
