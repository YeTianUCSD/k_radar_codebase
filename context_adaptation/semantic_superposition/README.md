# Automatic semantic superposition

This package connects the frozen V7 weather/road/lighting classifier to the
dynamic PSP model. The model-facing Context ID is allocated at runtime and is
never copied from a K-Radar sequence label.

## Runtime path

1. The frozen camera, LiDAR, and radar encoders process one incoming frame.
2. `EncodedAttributePredictor` computes the same mean/std descriptors and raw
   camera illumination statistics used to train V7.
3. The asymmetric temporal controller emits one of `stable`, `pending`,
   `switch`, `create`, or `abstain`.
4. Pending frames are withheld from optimization and retained in a bounded
   CPU replay buffer. Only the three FP16 BEV features, full-precision
   `gt_boxes`, and `batch_size` are retained.
5. `switch` activates the registered PSP residual bank. `create` evaluates all
   registered Contexts on exactly the same confirmed frames, chooses the
   lowest mean pre-update detection loss, and inherits that Context.
6. Confirmed pending frames are replayed once into the selected/new Context;
   normal stable frames are updated once on arrival.

The cumulative network already contains every Context residual. Therefore
parent selection does not load N full checkpoints: after a stage restores its
best residual, all registered best states coexist in one model.

## Evaluation policy

The V2 protocol removes the classifier's 30 `support_train` frames from every
scene before online adaptation. It runs three continuous, boundary-blind
visits: a randomized first train visit, a differently ordered train revisit,
and a read-only test revisit. Controller history is never reset at a hidden
sequence boundary.

`ONLINE.STAGE_SELECTION: oracle_best` matches the previous oracle experiment:
the evaluator uses the benchmark's hidden scene boundary and held-out split to
evaluate update 0, every configured interval, and the final update, then
restores that Context's best parameters and optimizer moments. Those boundary
and metric values are never passed to the semantic router. This mode is for a
fair score comparison with the old pipeline, not a strictly causal deployment.

Set `ONLINE.STAGE_SELECTION: last` for a causal run. It never uses evaluation
scores to alter online state, but consequently cannot claim a test-selected
"best update".

## Outputs

- `metrics/routing_log.csv`: frame-level routing, pending, replay, and updates.
- `metrics/routing_scene_summary.csv`: evaluation-only routing accuracy and
  events by true scene.
- `metrics/parent_selection.csv`: every candidate loss and selected parent.
- `metrics/online_curve.csv`: update-0/periodic/final metric and best history.
- `metrics/support_exclusion.csv`: per-segment proof that support frames were
  excluded before tensor loading.
- `metrics/segment_timing.csv`: each visit/scene's optimizer, parent-selection,
  online-pipeline, Oracle-evaluation, and total wall-clock time.
- `metrics/update_timing_scene_summary.csv`: two train visits aggregated per
  scene, plus total and average-per-scene update time.
- `metrics/final_routed_scene_summary.csv`: detection AP computed with the
  actual per-frame Context decisions, including Seq1 and the ten-scene mean.
- `checkpoints/latest.checkpoint`: rolling, segment-boundary resume point.
- `checkpoints/final.semantic_superposition.checkpoint`: full model, optimizer,
  semantic registry, pending replay, Context mapping, routes, and run metadata.
- `context_registry.json`: semantic keys, model Context names, inheritance
  graph, benchmark assignments, and stage-selection records.

The corrected runner is
`tools/superposition/automatic_semantic_sequential_v2.py`. Use the explicit
`best_parent_causal_v2.yml` configuration for deployment-style results and
`best_parent_oracle_v2.yml` only for ceiling comparison.
