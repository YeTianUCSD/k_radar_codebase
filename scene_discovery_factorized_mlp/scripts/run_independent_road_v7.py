#!/usr/bin/env python3
"""Train and evaluate independent Weather/Road/Lighting attribute paths."""

from pathlib import Path

from run_sample_efficient_hybrid_v4 import main


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    main(PROJECT_ROOT / "configs/independent_road_v7.yml")
