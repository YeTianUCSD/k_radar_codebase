# Factorized MLP scene attributes (offline phase)

This directory is independent of the earlier `scene_discovery` experiments. Its
first phase asks whether a small labeled support set can train one shared MLP
with three independent heads to classify weather, road type, and lighting.
Online new/known context discovery is intentionally deferred to phase two.

## Leakage-safe protocol

For each sequence, the chronological training split is partitioned as follows:

1. First 30 frames: nested support pool. Experiments use 10, 20, or 30 labels.
2. Next 20 frames: validation for early stopping, hyperparameters, temperature
   calibration, and causal window selection.
3. Next 10 frames: unused guard interval.
4. Remaining training frames: unbiased train-remainder evaluation.
5. Official test split: final evaluation only; it never selects a model.

Support subsets are deterministic and nested (`10 subset 20 subset 30`) so the
learning curve changes only the amount of labeled data. Class-weighted losses
reduce the effect of the four Normal sequences versus rare weather classes.

## Model search

The search compares camera-only, plain three-modality concatenation, and learned
three-modality fusion. In the learned fusion model, weather, road, and lighting
have separate softmax gate parameters. Therefore each task can learn its own
Camera/LiDAR/Radar weighting. The projectors are shared, while classification
heads are separate. The current gates are global task-specific weights, not
sample-dependent attention; dynamic gating can be tested later if justified.

Search runs only at the largest support budget and only on validation data. The
selected architecture/hyperparameters are then trained for every support budget
and three random seeds. Predictions are evaluated per frame and after causal
windows of 1, 5, and 10 frames; smoothing is reset at sequence boundaries.

## Entry points

- `scripts/build_fewshot_manifest.py`: inspect or export a split manifest only.
- `scripts/run_offline_experiment.py`: search, train, calibrate, evaluate, save.
- `tests/run_tests.py`: run fast protocol/model/causality tests.

The full experiment writes a new timestamped directory under
`results/SceneDiscoveryFactorizedMLP`. Key outputs are:

- `search_leaderboard.csv` and `selected_config.json`
- `offline_metrics.csv`, `learning_curve.csv`
- `per_class_metrics.csv`, `per_sequence_metrics.csv`
- `gate_weights.csv`
- `predictions/*.csv`, `confusions/*.csv`
- `checkpoints/*.pt`, training histories, manifests, and `run_meta.json`

`--quick` is a three-epoch flow test. Its accuracy is not an experimental result.

## Camera causal-window V2

`scripts/run_camera_window_experiment_v2.py` is a controlled diagnostic that
keeps the modality and evaluated samples fixed. It compares Camera-only KNN,
class-balanced logistic regression, and the three-head MLP after aggregating
features over causal windows of 1, 5, 10, or 20 frames.

V2 uses common window endpoints for every value of X. Support endpoints are
spread over the early 70% of each train sequence, the final 20% is a contiguous
validation block, and a 20-frame gap ensures that maximum-size support and
validation windows share no raw frames. The official test split is untouched.

The maximum window is currently 20 because Seq34 has only 96 train frames. A
30-frame window, 30 support endpoints, a disjoint gap, and a useful validation
block cannot all fit in that sequence. The support budget counts labeled window
endpoints; preceding frames inside a causal window are context, not extra labels.

## Multimodal causal-window V3

`scripts/run_multimodal_window_experiment_v3.py` reuses the exact V2 manifests,
causal endpoints, windows, budgets, seeds, losses, and evaluation scopes. It
changes only the representation architecture:

- `camera`: Camera-only control, identical to the V2 MLP branch.
- `concat`: separately project Camera/LiDAR/Radar and concatenate them.
- `gated`: separately project the modalities, then learn a static softmax gate
  for each of weather, road, and lighting before its classification head.

Class-weighted cross entropy compensates label imbalance; it does not learn
modality importance. Only the explicit `gated` architecture produces learned,
interpretable modality weights, saved in `gate_weights.csv` for every budget,
window, seed, attribute, and modality.

## Sample-efficient Hybrid V4

`scripts/run_sample_efficient_hybrid_v4.py` treats 3, 5, 10, 20, or 30 labelled
window endpoints per sequence as the complete supervision budget. It uses no
additional labelled validation block: hyperparameters and 80 training epochs are
fixed before evaluation, and every support label is used for fitting.

V4 tests X in 1, 3, 5, 10, and 20 with five seeds and three architectures:

- `camera`: the Camera-only control.
- `hybrid_concat`: Weather/Road concatenate three modalities; Lighting uses a
  dedicated Camera projector.
- `hybrid_gated`: Weather/Road learn independent three-modality softmax gates;
  Lighting uses the same isolated Camera-only path.

The run records both labelled-window counts and unique raw frames contributing
to their windows. `within_3pct.csv`, `within_3pct_guarded.csv`, and
`pareto_frontier.csv` expose the accuracy/resource trade-off. Recommendations
are explicitly posthoc analyses of official-test results and must be confirmed
on new held-out data before reporting a final unbiased score.

## Commands

```bash
cd /home/code/hyperradar/k_radar_codebase

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

/home/miniconda/envs/kradar_asf/bin/python \
  scene_discovery_factorized_mlp/tests/run_tests.py

CUDA_VISIBLE_DEVICES=0 /home/miniconda/envs/kradar_asf/bin/python -u \
  scene_discovery_factorized_mlp/scripts/run_offline_experiment.py \
  --config scene_discovery_factorized_mlp/configs/offline_v1.yml

CUDA_VISIBLE_DEVICES=0 /home/miniconda/envs/kradar_asf/bin/python -u \
  scene_discovery_factorized_mlp/scripts/run_camera_window_experiment_v2.py \
  --config scene_discovery_factorized_mlp/configs/camera_window_v2.yml

CUDA_VISIBLE_DEVICES=0 /home/miniconda/envs/kradar_asf/bin/python -u \
  scene_discovery_factorized_mlp/scripts/run_multimodal_window_experiment_v3.py \
  --config scene_discovery_factorized_mlp/configs/multimodal_window_v3.yml

CUDA_VISIBLE_DEVICES=0 /home/miniconda/envs/kradar_asf/bin/python -u \
  scene_discovery_factorized_mlp/scripts/run_sample_efficient_hybrid_v4.py \
  --config scene_discovery_factorized_mlp/configs/sample_efficient_hybrid_v4.yml
```


### V5: weather-only multimodal, low-latency windows

`configs/weather_multimodal_low_latency_v5.yml` compares Camera-only against
concat and gated Camera/LiDAR/Radar fusion for **weather only**. Road and
lighting share an isolated Camera-only representation. The scan is restricted
to causal windows 1, 3, and 5 while retaining the V4 budgets and five seeds.
Run it with:

```bash
python scene_discovery_factorized_mlp/scripts/run_sample_efficient_hybrid_v4.py \
  --config scene_discovery_factorized_mlp/configs/weather_multimodal_low_latency_v5.yml
```

### V6: independent Road/Lighting paths with spatial illumination

`configs/spatial_lighting_v6.yml` keeps the selected V5 Weather path but gives
Road and Lighting independent Camera projectors. Lighting also receives a
48-dimensional raw-image summary containing global and top/middle/bottom luma
statistics plus global RGB/HSV statistics. Arrays are aligned to the descriptor
bank by `sample_id` and are standardized from support-training rows only.

Build the small auxiliary feature bank once:

```bash
python -u \
  scene_discovery_factorized_mlp/scripts/build_illumination_statistics.py \
  --config scene_discovery_factorized_mlp/configs/spatial_lighting_v6.yml
```

Then train the 30-labelled-frame, feature-window-1 setting with five seeds:

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  scene_discovery_factorized_mlp/scripts/run_spatial_lighting_v6.py \
  --config scene_discovery_factorized_mlp/configs/spatial_lighting_v6.yml
```

The original Camera/Encoder feature bank is read-only. V6 writes illumination
