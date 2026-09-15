# Boundary-blind semantic online context V1

This module evaluates a deployable V5 `weather_multimodal_gated` checkpoint as
a continuous per-frame stream. The controller receives only the three attribute
probability vectors; sequence IDs, phase labels, and true boundaries remain in
the evaluator output and never enter `OnlineContextController.observe`.

The stream has four continuous phases:

1. Seq1 non-support Train frames bootstrap Context 0.
2. The other nine non-support Train sequences arrive once in random block order.
3. All ten non-support Train sequences are replayed in a different order.
4. All ten Test sequences arrive in a third order.

Decision windows 5 and 10 are evaluated on phases 1--3 (bootstrap plus both
Train visits). A fixed lexicographic objective selects one using Train only.
Only that controller retains its registry and enters Test. Confidence is logged
but not enforced, matching the offline V5 argmax protocol.

Run:

```bash
python scene_discovery_online/scripts/evaluate_online_context_v1.py \
  --config scene_discovery_online/configs/online_context_v1.yml
```

Important outputs include `window_comparison_train.csv`,
`frame_predictions.csv`, `context_events.csv`, `block_metrics.csv`,
`phase_metrics.csv`, `context_registry.json`, and `summary.json`.

The compressed `attribute_probabilities.npz` stores all 14 per-frame class
probabilities so later temporal and threshold sweeps do not rerun the neural head.
