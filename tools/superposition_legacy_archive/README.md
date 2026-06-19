# Legacy Superposition Tools

These scripts implement the older delta/bundle/materialization workflow that we used before moving the main ASF pipeline to scene-conditioned shared-parameter superposition.

Archived files:
- `asf_online_superposition.py`: older manager for bound-delta storage/materialization.
- `binary_sign_superpose_asf.py`: older offline bundling/evaluation helper for exact or approximate recovery experiments.

Current main-path tools:
- `tools/online_adapt_asf.py`: online adaptation with scene context active during forward/training.
- `tools/eval_scene_context_asf.py`: evaluate one shared checkpoint under different scene contexts.
