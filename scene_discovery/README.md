# Scene discovery from K-Radar encoder features

This module tests whether the frozen camera, LiDAR, and radar encoder features
can identify scenes without passing sequence IDs to the inference algorithms.
It is isolated from the online adaptation pipeline.

## Data policy

- Source features are read from `results/EncoderFeatureBank/10scenes_fp16`.
- Source arrays are always opened with NumPy memory mapping.
- FeatureBank files are never modified.
- Scaling, PCA, classifiers, clusters, and cluster-to-scene mappings are fit on
  the train split only. The factorized known-scene experiment trains on all ten
  train sequences and records this overlap explicitly as an upper-bound protocol.
- Sequence IDs and semantic descriptions are used only to score and interpret
  predictions, except when an optional ground-truth seed is explicitly requested.
  The factorized registry defaults to prediction-only bootstrap with no seed ID.
- Temporal windows never cross a sequence during static train/test evaluation.
  The streaming simulator intentionally allows causal windows to cross a true
  boundary because a deployed system does not know the boundary in advance.

## Scripts

1. `validate_feature_bank.py`: checks partition completeness, alignment,
   duplicate IDs, FP16 shapes, validity masks, and optional finite values.
2. `build_descriptors.py`: builds FP32 channel mean, mean/std, and spatial
   pyramid descriptors in chunks. Every `seq/split` has an independent
   completion marker, so the same command resumes only missing work.
3. `analyze_separability.py`: reports held-out nearest-centroid/KNN metrics,
   silhouette scores, PCA plots, centroid distances, and confusion matrices.
4. `run_clustering_baselines.py`: fits K-Means, diagonal GMM, and
   agglomerative baselines without scene labels; Hungarian alignment is learned
   on train only and then frozen for test.
5. `evaluate_open_set.py`: performs leave-one-scene-out unknown detection with
   a chronological train calibration segment, scans calibration-only threshold
   quantiles, and saves reusable projector/centroid model packages.
6. `simulate_stream_discovery.py`: runs automatic context creation on causal
   streams with consistent candidate buffers, adaptive per-context radii, nearby
   context merging, repeated-scene reuse metrics, and multiple scene orders.
   Its deployment threshold is selected from train calibration only; labels are
   consulted only after inference for evaluation.

7. `analyze_attributes.py`: searches modality/model/window choices on a
   chronological validation part of the train split, selects per-head confidence
   thresholds from that validation split, refits the locked factorized recognizer,
   and evaluates test once. Linear SVM probability calibration uses a separate
   chronological tail rather than frame-level cross-validation.
8. `evaluate_compositional_generalization.py`: excludes an entire sequence
   from preprocessing, selection, and fitting. The support matrix reports how
   many remaining sequences contain each component class; unsupported folds are
   skipped by default.
9. `simulate_attribute_registry.py`: smooths the three predicted probability
   streams and lazily creates or reuses semantic context keys after confidence
   and persistence checks. It reads validation-selected thresholds from the
   recognizer and starts without any sequence ID by default.

10. `evaluate_independent_attributes.py`: evaluates weather, road, and lighting
    with separate eligibility rules and nested leave-one-sequence-out selection.
    It reports sequence-balanced per-class metrics and combines heads only on
    common supported folds and causal frame anchors.

All experiment folders contain `run_meta.json`, metrics, and per-sample
predictions.

## Quick start

Use the same environment as encoder extraction:

```bash
cd /home/code/hyperradar/k_radar_codebase
export PYTHON_BIN=/home/miniconda/envs/kradar_asf/bin/python
```

Validate without scanning every feature value:

```bash
$PYTHON_BIN scene_discovery/scripts/validate_feature_bank.py \
  --finite-mode sample
```

Build descriptors once, or resume an interrupted build with the same command:

```bash
$PYTHON_BIN scene_discovery/scripts/build_descriptors.py \
  --feature-bank results/EncoderFeatureBank/10scenes_fp16 \
  --output-root results/SceneDiscovery/descriptors/v1 \
  --descriptors mean mean_std spatial \
  --chunk-size 2
```

Per-partition results remain under `_partitions/`; final train/test arrays are
consolidated only after all partitions finish. Completed outputs are reused only
when the configuration, source metadata SHA-256, source file signatures, row
counts, shapes, and dtypes still match. Use `--overwrite` only for an explicit
full rebuild. Keeping resumable and consolidated arrays uses about twice the
compact descriptor size.

Run the first two analyses:

```bash
$PYTHON_BIN scene_discovery/scripts/analyze_separability.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --windows 1 10 20 30 --stride 5

$PYTHON_BIN scene_discovery/scripts/run_clustering_baselines.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --windows 1 10 20 30 --stride 5

$PYTHON_BIN scene_discovery/scripts/evaluate_open_set.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --windows 1 20 --stride 5 \
  --threshold-quantiles 0.9 0.95 0.975 0.99 0.995

$PYTHON_BIN scene_discovery/scripts/simulate_stream_discovery.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --modalities camera lidar radar \
  --projector-fit-scenes 1 --window 20 --stride 5 \
  --threshold-multipliers 0.8 1.0 1.2 --num-random-orders 4
```

Run factorized semantic-attribute experiments:

```bash
$PYTHON_BIN scene_discovery/scripts/analyze_attributes.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --models logreg linear_svm knn \
  --windows 1 10 20 30 --stride 5

$PYTHON_BIN scene_discovery/scripts/evaluate_compositional_generalization.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --models logreg linear_svm knn \
  --held-out-scenes 1 5 9 46 58 --windows 1 10 20 30 --stride 5 \
  --min-support-scenes 1

$PYTHON_BIN scene_discovery/scripts/simulate_attribute_registry.py \
  --recognizer /path/to/attributes_RUN/recognizer.joblib \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --repeat-scenes 1 35 --num-random-orders 4

$PYTHON_BIN scene_discovery/scripts/evaluate_independent_attributes.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --modalities camera lidar radar \
  --models logreg linear_svm knn --windows 1 10 20 30 \
  --sequences 1 35 46 19 58 5 22 34 9 38 \
  --minimum-support-scenes 1

# Decode either an attributes_RUN (10-scene closed set) or an
# independent_attributes_RUN (strict LOSO). Both require v2 probability columns.
$PYTHON_BIN scene_discovery/scripts/evaluate_semantic_decoder.py \
  --predictions-root /path/to/attributes_OR_independent_attributes_RUN \
  --output-root results/SceneDiscovery/SemanticDecoder \
  --attribute-weights 1 1 1 \
  --min-known-score 0.0 --min-known-margin 0.0
```

Run the strict Seq1-only online replay. Future train partitions are first visits,
and disjoint test partitions are revisits; labels are used only for scoring:

```bash
$PYTHON_BIN scene_discovery/scripts/evaluate_online_context_discovery.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --modalities camera lidar radar \
  --fusion-mode seq1_calibrated_late_distance --modality-weights 1 1 1 \
  --manager memory_bank --memory-size 64 --memory-neighbors 7 \
  --initial-scenes 1 --scene-order 1 35 46 19 58 5 22 34 9 38 \
  --revisit-order 1 35 46 19 58 5 22 34 9 38 \
  --window 20 --stride 5 --threshold-quantile 0.99 \
  --threshold-multipliers 0.8 1.0 1.2 --num-random-orders 4
```

The full sequence is available as:

```bash
bash scene_discovery/run_baseline.sh
```

## Descriptor definitions

- `mean`: per-channel spatial mean.
- `mean_std`: per-channel spatial mean and population standard deviation.
- `spatial`: global mean/std followed by means from a row-major 2x2 grid.

Each modality is standardized and reduced independently (64 PCA dimensions by
default), L2-normalized independently, and concatenated only afterward.

## Tests

```bash
$PYTHON_BIN scene_discovery/tests/run_tests.py
```

## V4 causal online boundary evaluation

Unlike the V3 block-dominant diagnostic, V4 emits boundary events online,
freezes memory while a change is suspected, and makes an immediate segment-level
STAY/EXPAND, REUSE, or CREATE decision. Ground-truth boundaries are consumed only
after inference for event scoring.

```bash
$PYTHON_BIN scene_discovery/scripts/evaluate_online_context_discovery_v4.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --modalities camera lidar radar \
  --fusion-mode seq1_calibrated_late_distance --modality-weights 1 1 1 \
  --initial-scene 1 --scene-order 1 35 46 19 58 5 22 34 9 38 \
  --revisit-order 1 35 46 19 58 5 22 34 9 38 \
  --window 20 --stride 5 --threshold-quantile 0.99 \
  --change-threshold-multipliers 0.6 0.7 0.8 1.0 \
  --match-threshold-ratios 1.0 1.1 1.2 \
  --change-persistence 5 --boundary-tolerance 10 \
  --memory-size 64 --memory-neighbors 7 --num-random-orders 4
```

The primary metrics are change precision/recall/F1, causal detection delay,
strict correct-creation rate, collision-aware correct-reuse rate, and strict
end-to-end event success. A CREATE succeeds only when exactly one context is
created in the true segment. A REUSE succeeds only when it returns to the
one-to-one registered context without creating another context. Block labels
remain evaluation-only and are never supplied to `CausalContextManager`.


## V5 Seq1-only open-world weather discovery

V5 preserves the causal V4 manager but changes the discovery target from sequence
identity to weather identity. The encoder/projector and all thresholds are fitted
only with Seq1 (`normal`). The remaining scene train partitions arrive as
unlabelled, contiguous first visits in canonical or randomized order, followed by
disjoint test-partition revisits. Neither scene boundaries nor weather labels enter
the manager; labels are attached only after inference for scoring.

Under the ten-scene protocol, the ideal registry contains six anonymous contexts:
normal, overcast, rain, sleet, fog, and heavy snow. In particular, Seq1/5/9/19
should reuse one normal context, and Seq46/58 should reuse one heavy-snow context.

```bash
 scene_discovery/scripts/evaluate_online_weather_discovery_v5.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --modalities camera lidar radar \
  --fusion-mode seq1_calibrated_late_distance --modality-weights 1 1 1 \
  --initial-scene 1 --scene-order 1 35 46 19 58 5 22 34 9 38 \
  --window 20 --stride 5 --threshold-quantile 0.99 \
  --change-threshold-multipliers 0.6 0.7 0.8 1.0 \
  --match-threshold-ratios 1.0 1.1 1.2 \
  --change-persistence 5 --boundary-tolerance 10 \
  --memory-size 64 --memory-neighbors 7 --num-random-orders 4
```

The primary outputs are `summary.csv`, `metrics.csv`,
`true_weather_event_results.csv`, `predicted_weather_events.csv`,
`weather_decisions.csv`, and `per_weather_metrics.csv`. Manager snapshots remain
anonymous; the post-hoc `mapped_weather` column exists only to explain evaluation.


## V5.1 adaptive weather discovery

V5.1 keeps the V5 protocol unchanged and replaces only the online manager. It
separates dual-timescale departure detection from context assignment, collects a
stable candidate segment, requires coverage-based matching for REUSE, delays a
new context in a provisional state, learns a bounded radius per context, and
applies a cooldown after every committed transition. It still receives feature
vectors only and does not receive scene boundaries or weather labels.

```bash
 scene_discovery/scripts/evaluate_online_weather_discovery_v51.py \
  --descriptor-bank results/SceneDiscovery/descriptors/v1 \
  --descriptor mean_std --modalities camera lidar radar \
  --fusion-mode seq1_calibrated_late_distance --modality-weights 1 1 1 \
  --initial-scene 1 --scene-order 1 35 46 19 58 5 22 34 9 38 \
  --window 20 --stride 5 --threshold-quantile 0.99 \
  --change-threshold-multipliers 0.6 0.7 0.8 1.0 \
  --match-threshold-ratios 1.0 1.1 1.2 \
  --change-persistence 4 --candidate-min-windows 8 \
  --candidate-max-windows 24 --provisional-windows 4 \
  --cooldown-windows 12 --reference-windows 32 --recent-windows 6 \
  --distribution-threshold-ratio 0.6 --strong-departure-ratio 1.5 \
  --coverage-threshold 0.7 --context-quantile 0.95 \
  --min-radius-ratio 0.8 --max-radius-ratio 2.0 \
  --boundary-tolerance 15 --memory-size 64 --memory-neighbors 7 \
  --num-random-orders 4
```

In addition to the V5 outputs, `summary.csv` reports false CREATE events per
1000 windows, provisional entries/cancellations, and the resulting context count.
