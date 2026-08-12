# Online Adaptation for Multimodal 3D Object Detection

This repository studies online adaptation for multimodal 3D object detection in autonomous driving. 

## Dataset

[K-Radar](https://github.com/kaist-avelab/K-Radar) is a large-scale autonomous-driving dataset containing synchronized 4D radar tensors, LiDAR point clouds, camera images, RTK-GPS measurements, and 3D bounding-box annotations under diverse road and weather conditions.

Please prepare the dataset by following the instructions in the official K-Radar repository. The dataset, pretrained sensor encoders, and generated results are not included here; update the paths in `configs/` for your local setup.


## Environment

This work is developed on top of the original K-Radar environment. For details about the original dataset, benchmark, and baseline implementation, please refer to the official [K-Radar repository](https://github.com/kaist-avelab/K-Radar) and paper.

The experiments use Python 3.8 and a conda environment named `kradar_asf`:

```bash
conda create -n kradar_asf python=3.8 -y
conda activate kradar_asf
pip install -r requirements.txt
```

PyTorch, CUDA, spconv, and the compiled extensions must be compatible. Depending on the local CUDA/PyTorch versions, the operators under `ops/` and `utils/Rotated_IoU/` may need to be rebuilt.

### Repository Structure

```text
configs/        K-Radar, ASF, sequence-split, adaptation, and evaluation configs.
datasets/       Dataset loading, calibration, preprocessing, and batch collation.
models/         Sensor encoders, ASF fusion modules, and 3D detection heads.
ops/            CUDA/C++ operators and compiled extensions.
pipelines/      Shared training and evaluation pipeline.
tools/          Baseline, checkpoint evaluation, and online adaptation scripts.
utils/          Optimization, geometry, post-processing, and KITTI-style evaluation.
pretrained/     Locally prepared pretrained sensor-encoder weights.
results/        Generated checkpoints, logs, metrics, and evaluation outputs.
```

## Multimodal Fusion Baseline

The baseline uses pretrained Camera, LiDAR, and 4D Radar encoders to produce BEV features. ASF combines the available sensor features, and a shared anchor-based detection head predicts 3D objects. For the Sequence 1 baseline, the sensor encoders are frozen while the fusion module and detection head are trained and evaluated on Sequence 1.

Training and checkpoint evaluation are packaged in `tools/run_seq1_asf_baseline.sh`.

Run training followed by evaluation of all saved checkpoints:

```bash
cd /home/code/hyperradar/k_radar_codebase
FULL_EVAL_EVERY=20 bash tools/run_seq1_asf_baseline.sh all
```

Using `FULL_EVAL_EVERY=20` avoids running a full validation pass after every training epoch. The `all` mode evaluates every `model_*.pt` checkpoint after training, so validating every epoch during training would duplicate most of the evaluation work.

The two stages can also run separately:

```bash
# Train ASF on Sequence 1 for 20 epochs.
FULL_EVAL_EVERY=20 bash tools/run_seq1_asf_baseline.sh train

# Evaluate checkpoints with the Sequence 1 config.
bash tools/run_seq1_asf_baseline.sh eval /path/to/checkpoint/models
```

Settings can be overridden with environment variables:

```bash
CUDA_VISIBLE_DEVICES=1 EPOCHS=20 FULL_EVAL_EVERY=20 \
  bash tools/run_seq1_asf_baseline.sh all
```

The evaluation stage writes a consolidated `summary.csv` containing Sequence 1 BEV and 3D metrics for each `model_*.pt` checkpoint. `best.checkpoint` and `latest.checkpoint` are not included unless they are passed explicitly to `tools/eval_checkpoints.py`.

## Online Adaptation

The supervised online adaptation pipeline starts from an ASF checkpoint trained on Sequence 1 and processes the Sequence 58 training split in temporal order with `batch_size=1` and no shuffling. Each incoming labeled batch produces one model update; the model is periodically saved and evaluated on the Sequence 58 test split.

The command is packaged in `tools/run_seq58_online_adaptation.sh`. Pass the source checkpoint as the first argument and the trainable scope as the second:

```bash
cd /home/code/hyperradar/k_radar_codebase

bash tools/run_seq58_online_adaptation.sh \
  /path/to/sequence1/models/model_16.pt \
  fuser_head
```

For a long-running experiment:

```bash
mkdir -p results/run_logs

nohup bash tools/run_seq58_online_adaptation.sh \
  /path/to/sequence1/models/model_16.pt \
  fuser_head \
  > results/run_logs/online_seq58_fuser_head.log 2>&1 &
```

The `--trainable` setting controls which parameters receive gradient updates:

| Mode | Updated parameters |
|---|---|
| `cls` | Classification layer of the detection head only |
| `head` | Complete detection head |
| `fuser_head` | ASF fusion module and detection head |
| `full` | Sensor encoders, fusion module, and detection head |

The table describes parameters that receive gradient updates. The current adaptation config sets `FREEZE_BN: False`, and the network remains in training mode during each online update. BatchNorm running statistics in otherwise frozen modules may therefore still change for `cls`, `head`, and `fuser_head`. Set `FREEZE_BN: True` or explicitly keep the relevant BatchNorm layers in evaluation mode when an experiment requires all frozen module state to remain unchanged.

The default online settings use AdamW with a learning rate of `1e-4`, save every 50 updates, and evaluate every 50 updates. These values can be overridden through environment variables defined at the top of the bash script.

## Superposition-Based Knowledge-Preserving Online Update

We design a superposition-based online update model to mitigate catastrophic forgetting during cross-sequence adaptation. Knowledge from different driving sequences is represented within one shared model through scene contexts and scene-specific residual parameters. 

The pipeline has three stages:

1. Train the PSP-enabled ASF model on Sequence 1 with the `seq1` context and select the best checkpoint.
2. Adapt the checkpoint to Sequence 58 with the `seq58` context. Only the Sequence 58 residual parameters in the fusion module and detection head are updated; the shared parameters and Sequence 1 residuals remain unchanged.
3. Use the same adapted checkpoint with the `seq58` context for Sequence 58 and the `seq1` context for Sequence 1. Switching the scene context activates the corresponding knowledge path without maintaining a separate complete model for each sequence.

All stages are packaged in:

```text
tools/run_superposition_pipeline.sh
```

Run the complete pipeline with:

```bash
cd /home/code/hyperradar/k_radar_codebase
bash tools/run_superposition_pipeline.sh all
```

For a long-running experiment:

```bash
mkdir -p results/Superposition/v3

nohup bash tools/run_superposition_pipeline.sh all \
  > results/Superposition/v3/pipeline.log 2>&1 &
```

The stages can also be executed separately. When an explicit checkpoint is not provided, the script automatically selects the latest matching run under `results/Superposition/v3`.

```bash
# Train the PSP-enabled ASF model on Sequence 1.
bash tools/run_superposition_pipeline.sh train

# Adapt the latest Sequence 1 best checkpoint to Sequence 58.
bash tools/run_superposition_pipeline.sh adapt

# Evaluate the adapted checkpoint on both Sequence 58 and Sequence 1.
bash tools/run_superposition_pipeline.sh eval

# Run only one of the two evaluations.
bash tools/run_superposition_pipeline.sh eval_seq58
bash tools/run_superposition_pipeline.sh eval_seq1
```

To resume from explicit checkpoints:

```bash
SEQ1_CHECKPOINT=/path/to/seq1/models/best.checkpoint \
  bash tools/run_superposition_pipeline.sh adapt

ADAPTED_CHECKPOINT=/path/to/seq58/models/best.checkpoint \
  bash tools/run_superposition_pipeline.sh eval
```

By default, outputs are written to `results/Superposition/v3`. The training and adaptation stages create separate log files, and each scene evaluation writes its own `run.log` and `summary.csv` inside the adapted Sequence 58 run directory.

## Notes

- The online adaptation pipeline described here is supervised: Sequence 58 ground-truth boxes are used for online updates.
- Adjust absolute dataset and pretrained-weight paths in the corresponding YAML configs.
- Outputs include baseline, periodic, best, and last checkpoints together with CSV metric summaries.
