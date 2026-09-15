#!/usr/bin/env python3
"""V3 entry point with strict checkpoint protocol-signature validation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.superposition.automatic_semantic_sequential_v2 import main  # noqa: E402


if __name__ == "__main__":
    main(
        checkpoint_version=3,
        pipeline_id="automatic_semantic_sequential_v3",
    )
