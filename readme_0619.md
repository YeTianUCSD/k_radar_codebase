# Online Adaptation on K-radar dataset

This repository is built on top of the official [K-Radar](https://github.com/kaist-avelab/K-Radar) codebase. It is mainly used for experiments on multi-modal 3D object detection with K-Radar, especially the Availability-aware Sensor Fusion (ASF) pipeline and online adaptation experiments across different K-Radar sequences.

The original K-Radar project provides a large-scale 4D radar dataset and benchmark for autonomous driving, including 4D radar tensors, LiDAR, camera, RTK-GPS, and 3D bounding-box annotations. This repository keeps the core K-Radar/ASF pipeline and adds experiment scripts/configurations for:

- Training ASF-style sensor fusion models.
- Evaluating trained checkpoints on selected K-Radar sequences.
- Running online adaptation from a source sequence to a target sequence.
- Comparing different trainable scopes during online learning.

## Acknowledgement

This repository is based on:

```text
https://github.com/kaist-avelab/K-Radar
```

Please refer to the official K-Radar repository and paper for the original dataset, benchmark, and baseline implementation.

## Repository Structure

```text
configs/        Configuration files for K-Radar, ASF, sequence splits, and evaluation.
datasets/       Dataset loading and preprocessing utilities.
models/         Detection, fusion, and model components.
ops/            CUDA/C++ operators and compiled extensions.
pipelines/      Training and evaluation pipelines.
tools/          Utility scripts, checkpoint evaluation, online adaptation, visualization helpers.
utils/          Evaluation and geometry utilities.
uis/            GUI-related utilities from the original K-Radar codebase.
```

Large local folders are intentionally not included in this repository:

```text
pretrained/     Pretrained model weights.
results/        Training, evaluation, and online adaptation outputs.
docs/           Original documentation/images.
resources/      Original resources and large assets.
```

## Environment

The experiments were run with a conda environment named `kradar_asf`.

A minimal dependency list is provided in:

```bash
requirements.txt
```

Example setup:

```bash
conda create -n kradar_asf python=3.8 -y
conda activate kradar_asf

pip install -r requirements.txt
```

Depending on your CUDA/PyTorch version, you may need to install PyTorch, spconv, and CUDA extensions manually. The `ops/` and `utils/Rotated_IoU/` folders contain compiled/operator-related code used by the detection pipeline.

## Dataset and Weights

This repository does not include the K-Radar dataset, pretrained model weights, or generated experiment results.

Please prepare the K-Radar dataset following the official K-Radar instructions:

```text
https://github.com/kaist-avelab/K-Radar
```

Expected local folders on our server:

```text
/home/code/hyperradar/dataset/k_radar
/home/code/hyperradar/k_radar_codebase/pretrained
/home/code/hyperradar/k_radar_codebase/results
```

You may need to modify dataset paths inside the config files before running experiments.

## Quick Training Example

The following command runs a quick ASF training test on Sequence 1 with a small subset. This is mainly for checking whether the environment, dataset path, and training pipeline are working.

```bash
cd /home/code/hyperradar/k_radar_codebase

python main_train_0_args.py \
  --config ./configs/ASF_v2_0_seq1.yml \
  --output_dir /home/code/hyperradar/k_radar_codebase/results \
  --exp_name seq1_asf_quick_subset \
  --max_epoch 1 \
  --batch_size 2 \
  --num_workers 0 \
  --use_val_subset \
  --num_subset 20 \
  --val_per_epoch_subset 1 \
  --interval_epoch_model 1 \
  --interval_epoch_util 1 \
  --skip_final_eval
```

## Checkpoint Evaluation

After training a model on Sequence 1, we can evaluate saved checkpoints using `tools/eval_checkpoints.py`.

Example: evaluate checkpoints trained on Sequence 1 using the Sequence 1 config.

```bash
cd /home/code/hyperradar/k_radar_codebase

RUN_NAME=eval_train_seq1_20ckpt_on_seq1
RUN_DIR="/home/code/hyperradar/k_radar_codebase/results/${RUN_NAME}"
/bin/mkdir -p "$RUN_DIR"
LOG_FILE="${RUN_DIR}/run_$(/bin/date +%y%m%d_%H%M%S).log"

/usr/bin/nohup /bin/bash -lc "
set -euo pipefail

export PATH=/home/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\$PATH
source /home/miniconda/etc/profile.d/conda.sh
conda activate kradar_asf

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/code/hyperradar/k_radar_codebase

/home/miniconda/envs/kradar_asf/bin/python tools/eval_checkpoints.py  \
  --checkpoint_dir /home/code/hyperradar/k_radar_codebase/results/train_seq1_20epoch_test_seq1/exp_260514_101619_train_seq1_20epoch_test_seq1/models \
  --eval seq1=./configs/ASF_v2_0_seq1.yml \
  --output_root ${RUN_DIR} \
  --summary_csv ${RUN_DIR}/summary.csv \
  --conf_thr 0.3 \
  --best_metric_cls auto \
  --best_metric_kind 3d \
  --best_metric_ious 0.3 0.5 \
  --best_metric_conf 0.3
" > "$LOG_FILE" 2>&1 &
```

The evaluation script writes logs and a summary CSV to:

```text
/home/code/hyperradar/k_radar_codebase/results/${RUN_NAME}
```

## Online Adaptation

This repository includes an online adaptation script:

```text
tools/online_adapt_asf.py
```

The typical setting is:

1. Train or load a source model from Sequence 1.
2. Use Sequence 58 as the target/adaptation sequence.
3. Update selected parts of the model online.
4. Periodically evaluate and save checkpoints.

Example command:

```bash
cd /home/code/hyperradar/k_radar_codebase

RUN_NAME=online_seq58_fuser_head_from_seq1_model16_original_baseline
OUTPUT_ROOT=/home/code/hyperradar/k_radar_codebase/results/Superposition
LOG_FILE="${OUTPUT_ROOT}/${RUN_NAME}_$(/bin/date +%y%m%d_%H%M%S).log"

/usr/bin/nohup /bin/bash -lc "
set -euo pipefail

export PATH=/home/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\$PATH
source /home/miniconda/etc/profile.d/conda.sh
conda activate kradar_asf

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/code/hyperradar/k_radar_codebase
mkdir -p ${OUTPUT_ROOT}

echo '[START] ${RUN_NAME} at' \$(date)
echo '[PWD]' \$(pwd)
echo '[CUDA_VISIBLE_DEVICES]' \$CUDA_VISIBLE_DEVICES
echo '[CONFIG] ./configs/ASF_v2_0_seq58_adapt.yml'
echo '[INIT_MODEL] /home/code/hyperradar/k_radar_codebase/results/offline_baseline/train_seq1_20epoch_test_seq1/exp_260514_101619_train_seq1_20epoch_test_seq1/models/model_16.pt'
echo '[OUTPUT_ROOT] ${OUTPUT_ROOT}'

/home/miniconda/envs/kradar_asf/bin/python -u tools/online_adapt_asf.py \
  --config ./configs/ASF_v2_0_seq58_adapt.yml \
  --init_model /home/code/hyperradar/k_radar_codebase/results/offline_baseline/train_seq1_20epoch_test_seq1/exp_260514_101619_train_seq1_20epoch_test_seq1/models/model_16.pt \
  --output_root ${OUTPUT_ROOT} \
  --run_name ${RUN_NAME} \
  --trainable fuser_head \
  --batch_size 1 \
  --num_workers 0 \
  --max_steps -1 \
  --optimizer adamw \
  --lr 0.0001 \
  --weight_decay 0.0 \
  --grad_clip 0 \
  --eval_every_updates 50 \
  --save_every_updates 50 \
  --conf_thr 0.3 \
  --best_metric_cls auto \
  --best_metric_kind 3d \
  --best_metric_ious 0.3 0.5 \
  --best_metric_conf 0.3

echo '[DONE] ${RUN_NAME} finished at' \$(date)
" > "$LOG_FILE" 2>&1 &

echo "LOG_FILE=$LOG_FILE"
```

## Online Adaptation Modes

The `--trainable` argument controls which parts of the model are updated during online adaptation:

```text
head        = update detection head only
fuser_head  = update fusion module + detection head
full        = update encoder + fusion module + detection head
```

This allows us to compare lightweight online adaptation against more aggressive model updates.

## Superposition Model Pipeline

This repository also includes a scene-conditioned superposition pipeline for carrying Sequence 1 and Sequence 58 inside one shared PSP-enabled model.

The workflow is:

1. Train a PSP model from scratch on Sequence 1 using the `seq1` scene context.
2. Load the best Sequence 1 PSP checkpoint and run supervised online adaptation on Sequence 58 using the `seq58` scene context.
3. Evaluate the adapted shared model twice: once on Sequence 58 with the `seq58` context key, and once on Sequence 1 with the `seq1` context key.

The relevant configs and scripts are:

```text
configs/ASF_v2_0_seq1_psp.yml
configs/ASF_v2_0_seq58_adapt_psp.yml
configs/ASF_v2_0_seq1_eval_psp.yml
configs/ASF_v2_0_seq58_eval_psp.yml
tools/online_adapt_asf_psp.py
tools/eval_scene_context_asf.py
tools/run_superposition_pipeline.sh
```

If you want one place to review the entire runnable pipeline, use:

```bash
cd /home/code/hyperradar/k_radar_codebase

bash tools/run_superposition_pipeline.sh
```

That script runs the following stages in order:

```text
1. train_seq1_20epoch_test_seq1_psp
2. online_seq58_fuser_head_from_seq1_psp_best
3. eval_seq58_shared_psp_best_on_seq58
4. eval_seq1_shared_psp_best_on_seq1
```

It writes training and online adaptation logs under:

```text
/home/code/hyperradar/k_radar_codebase/results/Superposition
```

and it stores the two final evaluation summaries inside the online adaptation run directory.

If you prefer to launch each stage separately with `nohup`, the equivalent commands are shown below.

### 1. Train Sequence 1 PSP From Scratch

```bash
cd /home/code/hyperradar/k_radar_codebase

RUN_NAME=train_seq1_20epoch_test_seq1_psp
RESULT_ROOT=/home/code/hyperradar/k_radar_codebase/results/Superposition
LOG_FILE="${RESULT_ROOT}/${RUN_NAME}_$(/bin/date +%y%m%d_%H%M%S).log"

mkdir -p "$RESULT_ROOT"

/usr/bin/nohup /bin/bash -lc "
set -euo pipefail

export PATH=/home/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\$PATH
source /home/miniconda/etc/profile.d/conda.sh
conda activate kradar_asf

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/code/hyperradar/k_radar_codebase

echo '[START] ${RUN_NAME} at' \$(date)
echo '[PWD]' \$(pwd)
echo '[CUDA_VISIBLE_DEVICES]' \$CUDA_VISIBLE_DEVICES

/home/miniconda/envs/kradar_asf/bin/python -u main_train_0_args.py   --config ./configs/ASF_v2_0_seq1_psp.yml   --output_root ${RESULT_ROOT}   --run_name ${RUN_NAME}   --epochs 20   --batch_size 2   --num_workers 0   --full_eval_every 1   --best_metric_cls auto   --best_metric_kind 3d   --best_metric_ious 0.3 0.5   --best_metric_conf 0.3   --skip_final_eval

echo '[DONE] ${RUN_NAME} finished at' \$(date)
" > "$LOG_FILE" 2>&1 &
```

### 2. Online Update On Sequence 58

Replace `INIT_MODEL` with the `best.checkpoint` produced by step 1.

```bash
cd /home/code/hyperradar/k_radar_codebase

RUN_NAME=online_seq58_fuser_head_from_seq1_psp_best
RESULT_ROOT=/home/code/hyperradar/k_radar_codebase/results/Superposition
INIT_MODEL=/home/code/hyperradar/k_radar_codebase/results/Superposition/train_seq1_20epoch_test_seq1_psp_exp_YYMMDD_HHMMSS/models/best.checkpoint
LOG_FILE="${RESULT_ROOT}/${RUN_NAME}_$(/bin/date +%y%m%d_%H%M%S).log"

mkdir -p "$RESULT_ROOT"

/usr/bin/nohup /bin/bash -lc "
set -euo pipefail

export PATH=/home/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\$PATH
source /home/miniconda/etc/profile.d/conda.sh
conda activate kradar_asf

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/code/hyperradar/k_radar_codebase

echo '[START] ${RUN_NAME} at' \$(date)
echo '[PWD]' \$(pwd)
echo '[CUDA_VISIBLE_DEVICES]' \$CUDA_VISIBLE_DEVICES
echo '[INIT_MODEL]' ${INIT_MODEL}

/home/miniconda/envs/kradar_asf/bin/python -u tools/online_adapt_asf_psp.py   --config ./configs/ASF_v2_0_seq58_adapt_psp.yml   --init_model ${INIT_MODEL}   --output_root ${RESULT_ROOT}   --run_name ${RUN_NAME}   --trainable fuser_head   --batch_size 1   --num_workers 0   --max_steps -1   --optimizer adamw   --lr 0.0001   --weight_decay 0.0   --grad_clip 0   --eval_every_updates 50   --save_every_updates 50   --conf_thr 0.3   --best_metric_cls auto   --best_metric_kind 3d   --best_metric_ious 0.3 0.5   --best_metric_conf 0.3   --scene_context_train seq58   --scene_context_eval seq58

echo '[DONE] ${RUN_NAME} finished at' \$(date)
" > "$LOG_FILE" 2>&1 &
```

### 3. Evaluate The Shared PSP Model

After online adaptation, take the new Sequence 58 run's `models/best.checkpoint` and evaluate it on both sequences with their matching scene context keys.

Sequence 58 evaluation:

```bash
cd /home/code/hyperradar/k_radar_codebase

RUN_NAME=eval_seq58_shared_psp_best_on_seq58
RUN_DIR=/home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/${RUN_NAME}
LOG_FILE="${RUN_DIR}/run.log"

mkdir -p "$RUN_DIR"

nohup bash -lc '
set -euo pipefail

export PATH=/home/miniconda/bin:$PATH
source /home/miniconda/etc/profile.d/conda.sh
conda activate kradar_asf

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/code/hyperradar/k_radar_codebase

echo "[START] eval_seq58_shared_psp_best_on_seq58 at $(date)"
echo "[MODEL] /home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/models/best.checkpoint"

/home/miniconda/envs/kradar_asf/bin/python -u tools/eval_scene_context_asf.py   --checkpoints /home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/models/best.checkpoint   --eval seq58=./configs/ASF_v2_0_seq58_eval_psp.yml   --scene seq58=seq58   --output_root /home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/eval_seq58_shared_psp_best_on_seq58/eval_outputs   --summary_csv /home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/eval_seq58_shared_psp_best_on_seq58/summary.csv   --conf_thr 0.3   --best_metric_cls auto   --best_metric_kind 3d   --best_metric_ious 0.3 0.5   --best_metric_conf 0.3

echo "[DONE] eval_seq58_shared_psp_best_on_seq58 at $(date)"
' > "$LOG_FILE" 2>&1 &
```

Sequence 1 evaluation:

```bash
cd /home/code/hyperradar/k_radar_codebase

RUN_NAME=eval_seq1_shared_psp_best_on_seq1
RUN_DIR=/home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/${RUN_NAME}
LOG_FILE="${RUN_DIR}/run.log"

mkdir -p "$RUN_DIR"

nohup bash -lc '
set -euo pipefail

export PATH=/home/miniconda/bin:$PATH
source /home/miniconda/etc/profile.d/conda.sh
conda activate kradar_asf

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/code/hyperradar/k_radar_codebase

echo "[START] eval_seq1_shared_psp_best_on_seq1 at $(date)"
echo "[MODEL] /home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/models/best.checkpoint"

/home/miniconda/envs/kradar_asf/bin/python -u tools/eval_scene_context_asf.py   --checkpoints /home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/models/best.checkpoint   --eval seq1=./configs/ASF_v2_0_seq1_eval_psp.yml   --scene seq1=seq1   --output_root /home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/eval_seq1_shared_psp_best_on_seq1/eval_outputs   --summary_csv /home/code/hyperradar/k_radar_codebase/results/Superposition/online_seq58_fuser_head_from_seq1_psp_best_exp_YYMMDD_HHMMSS/eval_seq1_shared_psp_best_on_seq1/summary.csv   --conf_thr 0.3   --best_metric_cls auto   --best_metric_kind 3d   --best_metric_ious 0.3 0.5   --best_metric_conf 0.3

echo "[DONE] eval_seq1_shared_psp_best_on_seq1 at $(date)"
' > "$LOG_FILE" 2>&1 &
```

## Current Experimental Setup

The current experiments mainly use:

```text
Source sequence: Sequence 1
Target/adaptation sequence: Sequence 58
Model: ASF-style camera + LiDAR + 4D radar fusion model
```

A common workflow is:

```text
1. Train ASF on Sequence 1.
2. Evaluate Sequence 1 checkpoints.
3. Select a checkpoint, e.g., model_16.pt.
4. Run online adaptation on Sequence 58.
5. Compare different trainable scopes: head, fuser_head, full.
```

## Notes

- The repository does not include pretrained weights or generated results.
- Please update absolute paths in the commands if your local directory structure is different.
- Some CUDA/C++ extensions may need to match your local CUDA, PyTorch, and compiler versions.
