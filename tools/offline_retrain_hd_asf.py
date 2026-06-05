#!/usr/bin/env python3
"""
Offline HD retraining for ASF/K-Radar on the source sequence.

This script retrains ASF weights against an HD-only classification head.
At every epoch it rebuilds HD memory from the current source-train features,
trains the selected CNN/ASF parameters with that memory fixed, then rebuilds
memory again before evaluation/checkpointing.

The original CNN training and online HD adaptation scripts are untouched.
"""

import argparse
import csv
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hd_utils_asf import (
    clear_batch,
    close_writers,
    copy_checkpoint,
    extract_hd_features_by_labels,
    load_model_checkpoint,
    make_runtime_config,
    require_hd_head,
    run_eval,
    save_checkpoint,
    timestamp_now,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Offline HyperLidar-style HD memory retrain for ASF/K-Radar.')
    parser.add_argument('--config', required=True, help='HD retrain config. Train split is source train; test split is eval.')
    parser.add_argument('--init_model', required=True, help='CNN/ASF checkpoint used for fixed feature extraction.')
    parser.add_argument('--hd_memory', required=True, help='Initial HD memory, used for embedder/projection initialization.')
    parser.add_argument('--output_root', default='./results')
    parser.add_argument('--run_name', default='offline_retrain_hd_asf')
    parser.add_argument('--run_stamp', default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5,
                        help='Total HD epochs. Epoch 1 is full memory build; later epochs are buffer retrain.')
    parser.add_argument('--max_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--trainable', choices=['none', 'memory_only', 'hd_adapter', 'detection_head', 'head', 'fuser_head', 'full'], default='none',
                        help='none/memory_only updates only HD memory; hd_adapter optimizes only the HD adapter; detection_head/head optimize detector head weights; fuser_head optimizes fusion and detector head; full optimizes all non-HD parameters.')
    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adamw')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--include_negative', action='store_true', default=True,
                        help='Include background anchors with max_neg_* caps. Enabled by default.')
    parser.add_argument('--no_include_negative', dest='include_negative', action='store_false')
    parser.add_argument('--max_pos_per_class', type=int, default=0,
                        help='0 means use all positives per batch.')
    parser.add_argument('--max_total_pos', type=int, default=0)
    parser.add_argument('--max_neg_per_batch', type=int, default=8192)
    parser.add_argument('--max_neg_ratio', type=float, default=3.0)

    parser.add_argument('--eval_every_epochs', type=int, default=1)
    parser.add_argument('--save_every_epochs', type=int, default=1)
    parser.add_argument('--conf_thr', type=float, default=0.3)
    parser.add_argument('--hd_train_quantize', choices=['config', 'false', 'true', 'ste'], default='config',
                        help='Config override for HD encoding during backprop.')
    parser.add_argument('--hd_logit_scale', type=float, default=None,
                        help='Optional HD logit scale override.')
    parser.add_argument('--hd_cls_weight', type=float, default=None,
                        help='Optional override for HD classification loss weight.')
    parser.add_argument('--enable_scl_loss', action='store_true')
    parser.add_argument('--skip_baseline_eval', action='store_true')
    parser.add_argument('--skip_final_eval', action='store_true')
    parser.add_argument('--best_metric_cls', default='auto')
    parser.add_argument('--best_metric_kind', choices=['bev', '3d'], default='3d')
    parser.add_argument('--best_metric_ious', type=float, nargs='+', default=[0.3, 0.5])
    parser.add_argument('--best_metric_conf', type=float, default=0.3)
    return parser.parse_args()


def freeze_network(network):
    for param in network.parameters():
        param.requires_grad = False
    return {'trainable_params': 0, 'enabled_modules': {'hd_memory_only': 0}}


def set_trainable_scope(network, scope):
    for param in network.parameters():
        param.requires_grad = False

    if scope in ['none', 'memory_only']:
        return freeze_network(network)

    enabled_modules = {}

    if scope == 'hd_adapter':
        head = getattr(network, 'head', None)
        adapter = getattr(head, 'hd_adapter', None) if head is not None else None
        if adapter is None:
            raise RuntimeError('hd_adapter scope requires network.head.hd_adapter')
        enabled_adapter = 0
        for param in adapter.parameters():
            if param.is_floating_point() or param.is_complex():
                param.requires_grad = True
                enabled_adapter += param.numel()
        enabled_modules['hd_adapter'] = int(enabled_adapter)
        if enabled_adapter <= 0:
            raise RuntimeError('No trainable parameters found for scope=hd_adapter. Enable MODEL.HEAD.HD.ADAPTER.')
        return {'trainable_params': int(enabled_adapter), 'enabled_modules': enabled_modules}

    if scope == 'fuser_head':
        fuser = getattr(network, 'fuser', None)
        if fuser is None:
            raise RuntimeError('fuser_head scope requires network.fuser')
        enabled_fuser = 0
        for param in fuser.parameters():
            if param.is_floating_point() or param.is_complex():
                param.requires_grad = True
                enabled_fuser += param.numel()
        enabled_modules['fuser'] = int(enabled_fuser)
    elif scope == 'full':
        enabled_full = 0
        for name, param in network.named_parameters():
            if name.startswith('head.hd_core.'):
                param.requires_grad = False
                continue
            if param.is_floating_point() or param.is_complex():
                param.requires_grad = True
                enabled_full += param.numel()
        enabled_modules['full_without_hd_core'] = int(enabled_full)
        if enabled_full <= 0:
            raise RuntimeError('No trainable parameters found for scope=full.')
        return {'trainable_params': int(enabled_full), 'enabled_modules': enabled_modules}
    elif scope not in ['detection_head', 'head']:
        raise RuntimeError(f'Unsupported offline HD trainable scope: {scope}')

    head = getattr(network, 'head', None)
    if head is None:
        raise RuntimeError(f'{scope} scope requires network.head')
    enabled_head = 0
    for name, param in head.named_parameters():
        if name.startswith('hd_core.'):
            param.requires_grad = False
            continue
        if param.is_floating_point() or param.is_complex():
            param.requires_grad = True
            enabled_head += param.numel()
    enabled_modules['head_without_hd_core'] = int(enabled_head)

    enabled = sum(enabled_modules.values())
    if enabled <= 0:
        raise RuntimeError(f'No trainable parameters found for scope={scope}.')
    return {'trainable_params': int(enabled), 'enabled_modules': enabled_modules}


def build_optimizer(network, args):
    params = [p for p in network.parameters() if p.requires_grad]
    if not params:
        return None
    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise RuntimeError(f'Unsupported optimizer: {args.optimizer}')


def grad_norm(parameters):
    total_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = param.grad.detach().data.norm(2).item()
        total_sq += norm * norm
    return total_sq ** 0.5


def append_row(path_csv, row):
    fieldnames = [
        'event', 'timestamp', 'epoch', 'step_idx', 'update_idx', 'loss', 'grad_norm',
        'num_total', 'num_bg', 'num_pos', 'num_correct', 'num_wrong', 'num_random',
        'score', 'best_score', 'checkpoint', 'best_checkpoint', 'best_source_checkpoint',
        'train_time_sec', 'eval_time_sec', 'save_time_sec', 'elapsed_sec',
    ]
    is_new = not os.path.exists(path_csv)
    with open(path_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in fieldnames})


def write_best_summary(path_log, epoch, update_idx, score, best_path, source_path, args):
    txt = os.path.join(path_log, 'best_summary.txt')
    with open(txt, 'w') as f:
        f.write(f'best_epoch: {epoch}\n')
        f.write(f'best_update: {update_idx}\n')
        f.write(f'score: {score:.6f}\n')
        f.write(f'metric_cls: {args.best_metric_cls}\n')
        f.write(f'metric_kind: {args.best_metric_kind}\n')
        f.write(f'metric_ious: {[float(x) for x in args.best_metric_ious]}\n')
        f.write(f'metric_conf_thr: {float(args.best_metric_conf)}\n')
        f.write(f'best_checkpoint: {best_path}\n')
        f.write(f'source_checkpoint: {source_path}\n')


def apply_runtime_overrides(path_config, args):
    with open(path_config, 'r') as f:
        cfg = yaml.safe_load(f)

    head_cfg = cfg.setdefault('MODEL', {}).setdefault('HEAD', {})
    hd_cfg = head_cfg.setdefault('HD', {})
    if args.hd_train_quantize != 'config':
        if args.hd_train_quantize == 'ste':
            hd_cfg['TRAIN_QUANTIZE'] = 'ste'
        else:
            hd_cfg['TRAIN_QUANTIZE'] = args.hd_train_quantize == 'true'
    if args.hd_logit_scale is not None:
        hd_cfg['LOGIT_SCALE'] = float(args.hd_logit_scale)
    if args.hd_cls_weight is not None:
        loss_weights = head_cfg.setdefault('LOSS_CONFIG', {}).setdefault('LOSS_WEIGHTS', {})
        loss_weights['cls_weight'] = float(args.hd_cls_weight)

    with open(path_config, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def set_hd_update_mode(network, train_network=False):
    if train_network:
        network.train()
        network.training = True
    else:
        network.eval()
    head = require_hd_head(network)
    head.train()
    return head


@torch.no_grad()
def select_candidates(head, feat_map, labels, args):
    return head.get_hd_features_by_labels(
        feat_map.detach(),
        labels.detach(),
        max_pos_per_class=args.max_pos_per_class,
        max_total_pos=args.max_total_pos,
        include_negative=bool(args.include_negative and head.hd_core.cfg.use_background),
        max_neg_per_batch=args.max_neg_per_batch,
        max_neg_ratio=args.max_neg_ratio,
    )


def labels_to_memory_indices(hd_core, labels_1based):
    return hd_core._labels_to_memory_indices(labels_1based)


@torch.no_grad()
def rebuild_hd_memory_inplace(pline, loader, args, desc='* Rebuild HD memory'):
    head = require_hd_head(pline.network)
    head.hd_core.memory.reset()
    stats_sum = {'num_total': 0, 'num_bg': 0, 'num_pos': 0, 'num_correct': 0, 'num_wrong': 0, 'num_random': 0}

    for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
        if args.max_steps_per_epoch > 0 and batch_idx >= args.max_steps_per_epoch:
            clear_batch(batch)
            break
        feat_sel, labels_sel = extract_hd_features_by_labels(
            pline.network,
            batch,
            max_pos_per_class=args.max_pos_per_class,
            max_total_pos=args.max_total_pos,
            include_negative=bool(args.include_negative and head.hd_core.cfg.use_background),
            max_neg_per_batch=args.max_neg_per_batch,
            max_neg_ratio=args.max_neg_ratio,
        )
        if labels_sel.numel() > 0:
            head.hd_core.build_update(feat_sel, labels_sel, alpha=1.0)
            mem_labels = labels_to_memory_indices(head.hd_core, labels_sel)
            stats_sum['num_total'] += int(labels_sel.numel())
            if bool(head.hd_core.cfg.use_background):
                stats_sum['num_bg'] += int((mem_labels == 0).sum().item())
                stats_sum['num_pos'] += int((mem_labels > 0).sum().item())
            else:
                stats_sum['num_pos'] += int(labels_sel.numel())
        clear_batch(batch)

    head.hd_core.memory.normalize_()
    return stats_sum


def run_hd_memory_epoch(pline, loader, args, epoch, optimizer=None, trainable_parameters=None):
    is_trainable = optimizer is not None
    set_hd_update_mode(pline.network, train_network=is_trainable)

    loss_sum = 0.0
    grad_norm_sum = 0.0
    pbar = tqdm(loader, desc=f'* Offline HD retrain epoch {epoch}/{args.epochs}')
    steps = 0
    for step_idx, batch in enumerate(pbar, start=1):
        if args.max_steps_per_epoch > 0 and step_idx > args.max_steps_per_epoch:
            clear_batch(batch)
            break
        steps += 1

        if is_trainable:
            optimizer.zero_grad(set_to_none=True)
            dict_net = pline.network(batch)
            if pline.get_loss_from == 'head':
                loss = pline.network.head.loss(dict_net)
            elif pline.get_loss_from == 'detector':
                loss = pline.network.loss(dict_net)
            else:
                raise RuntimeError(f'Unsupported get_loss_from={pline.get_loss_from}')

            loss_value = float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float(loss)
            loss_sum += loss_value
            grad_value = 0.0
            did_backward = False
            if torch.is_tensor(loss) and torch.isfinite(loss):
                if loss.requires_grad:
                    loss.backward()
                    did_backward = True
            elif loss_value != 0.0:
                raise RuntimeError(f'Non-finite offline HD loss at epoch={epoch}, step={step_idx}')

            if did_backward:
                grad_value = grad_norm(trainable_parameters)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=float(args.grad_clip))
                optimizer.step()
            grad_norm_sum += grad_value

        pbar.set_postfix(
            loss=f'{loss_sum / max(1, steps):.4f}' if is_trainable else '0.0000',
        )
        clear_batch(batch)

    avg_loss = loss_sum / max(1, steps) if is_trainable else 0.0
    avg_grad_norm = grad_norm_sum / max(1, steps) if is_trainable else 0.0
    return steps, avg_loss, avg_grad_norm


@torch.no_grad()
def rebuild_hd_memory(pline, args, checkpoint_path, runtime_config):
    missing, unexpected = load_model_checkpoint(pline.network, checkpoint_path)
    head = require_hd_head(pline.network)
    head.hd_core.memory.reset()

    loader = torch.utils.data.DataLoader(
        pline.dataset_train,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=pline.dataset_train.collate_fn,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    total_pos = 0
    total_bg = 0
    class_counts = torch.zeros(head.num_class, dtype=torch.long)
    t0 = time.time()
    for batch_idx, batch in enumerate(tqdm(loader, desc='* Rebuild HD memory from best checkpoint')):
        if args.max_steps_per_epoch > 0 and batch_idx >= args.max_steps_per_epoch:
            clear_batch(batch)
            break
        feat_sel, labels_sel = extract_hd_features_by_labels(
            pline.network,
            batch,
            max_pos_per_class=args.max_pos_per_class,
            max_total_pos=args.max_total_pos,
            include_negative=bool(args.include_negative and head.hd_core.cfg.use_background),
            max_neg_per_batch=args.max_neg_per_batch,
            max_neg_ratio=args.max_neg_ratio,
        )
        if labels_sel.numel() > 0:
            head.hd_core.build_update(feat_sel, labels_sel)
            pos_mask = labels_sel > 0
            if pos_mask.any():
                shifted = labels_sel[pos_mask].detach().cpu().long() - 1
                class_counts.index_add_(0, shifted, torch.ones_like(shifted))
                total_pos += int(pos_mask.sum().item())
            total_bg += int((labels_sel == 0).sum().item())
        clear_batch(batch)

    head.hd_core.memory.normalize_()
    build_time = time.time() - t0

    hd_dir = os.path.join(pline.path_log, 'hd_memory')
    os.makedirs(hd_dir, exist_ok=True)
    mem_path = os.path.join(hd_dir, 'hd_memory.pth')
    meta = {
        'source_checkpoint': checkpoint_path,
        'config': args.config,
        'runtime_config': runtime_config,
        'trainable': args.trainable,
        'use_background': bool(head.hd_core.cfg.use_background),
        'total_positive_anchors': int(total_pos),
        'total_background_anchors': int(total_bg),
        'class_counts': [int(x) for x in class_counts.tolist()],
        'max_neg_per_batch': int(args.max_neg_per_batch),
        'max_neg_ratio': float(args.max_neg_ratio),
        'class_names': list(head.class_names),
        'build_time_sec': float(build_time),
        'timestamp': timestamp_now(),
        'missing_keys': list(missing),
        'unexpected_keys': list(unexpected),
    }
    head.hd_core.save_memory(mem_path, meta=meta)

    summary_path = os.path.join(pline.path_log, 'rebuild_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(meta.keys()) + ['memory_path'])
        writer.writeheader()
        row = dict(meta)
        row['memory_path'] = mem_path
        writer.writerow(row)

    return mem_path, meta


def main():
    args = parse_args()
    best_metric = {
        'CLS': args.best_metric_cls,
        'KIND': args.best_metric_kind,
        'IOUS': [float(x) for x in args.best_metric_ious],
        'CONF_THR': float(args.best_metric_conf),
        'REDUCE': 'mean',
        'ONLY_CLASSES_WITH_GT': True,
    }
    runtime_config, _runtime_cfg = make_runtime_config(
        args.config,
        args.output_root,
        args.run_name,
        args.batch_size,
        args.num_workers,
        conf_thr=args.conf_thr,
        run_stamp=args.run_stamp,
        save_model=True,
        best_metric=best_metric,
    )
    _runtime_cfg = apply_runtime_overrides(runtime_config, args)

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pline = PipelineDetection_v1_0(path_cfg=runtime_config, mode='train')
    missing, unexpected = load_model_checkpoint(pline.network, args.init_model)
    head = require_hd_head(pline.network)
    hd_meta = head.hd_core.load_memory(args.hd_memory, map_location='cpu')

    if (not args.enable_scl_loss) and hasattr(pline.network, 'is_scl'):
        pline.network.is_scl = False
        if hasattr(pline.network, 'fuser') and hasattr(pline.network.fuser, 'is_scl'):
            pline.network.fuser.is_scl = False

    trainable_info = set_trainable_scope(pline.network, args.trainable)
    optimizer = build_optimizer(pline.network, args)
    trainable_parameters = [p for p in pline.network.parameters() if p.requires_grad]

    with open(os.path.join(pline.path_log, 'run_meta.yml'), 'w') as f:
        yaml.safe_dump({
            'pipeline': 'hd_offline_retrain_epoch_rebuild',
            'config': args.config,
            'runtime_config': runtime_config,
            'init_model': args.init_model,
            'hd_memory': args.hd_memory,
            'hd_memory_meta': hd_meta,
            'trainable': args.trainable,
            'trainable_info': trainable_info,
            'optimizer': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'grad_clip': args.grad_clip,
            'include_negative': bool(args.include_negative),
            'epochs': args.epochs,
            'max_steps_per_epoch': args.max_steps_per_epoch,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'max_pos_per_class': args.max_pos_per_class,
            'max_total_pos': args.max_total_pos,
            'max_neg_per_batch': args.max_neg_per_batch,
            'max_neg_ratio': args.max_neg_ratio,
            'use_background': bool(head.hd_core.cfg.use_background),
            'hd_train_quantize': head.hd_core.cfg.train_quantize,
            'hd_quantize': bool(head.hd_core.cfg.quantize),
            'hd_logit_scale': float(head.hd_core.logit_scale.item()),
            'detach_cls_in_train': bool(head.model_cfg.HD.get('DETACH_CLS_IN_TRAIN', True)),
            'best_metric': best_metric,
            'missing_keys': list(missing),
            'unexpected_keys': list(unexpected),
        }, f, sort_keys=False)

    loader = torch.utils.data.DataLoader(
        pline.dataset_train,
        batch_size=int(args.batch_size),
        shuffle=bool(args.shuffle),
        collate_fn=pline.dataset_train.collate_fn,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    model_dir = os.path.join(pline.path_log, 'models')
    os.makedirs(model_dir, exist_ok=True)
    metrics_csv = os.path.join(pline.path_log, 'offline_retrain_hd_metrics.csv')
    best_path = os.path.join(model_dir, 'best.checkpoint')
    best_source_path = ''
    best_score = None
    t_start = time.time()
    update_idx = 0

    baseline_path = os.path.join(model_dir, 'baseline.checkpoint')
    baseline_meta = {'pipeline': 'hd_epoch_rebuild', 'trainable': args.trainable}
    save_checkpoint(baseline_path, pline.network, 0, update_idx, best_score, 'baseline', baseline_meta)
    if not args.skip_baseline_eval:
        eval_t0 = time.time()
        score, _rows = run_eval(pline, epoch=0, conf_thr=args.conf_thr)
        eval_time = time.time() - eval_t0
        best_score = score
        save_checkpoint(baseline_path, pline.network, 0, update_idx, best_score, 'baseline', baseline_meta)
        copy_checkpoint(baseline_path, best_path)
        best_source_path = baseline_path
        if best_score is not None:
            write_best_summary(pline.path_log, 0, update_idx, best_score, best_path, best_source_path, args)
        append_row(metrics_csv, {
            'event': 'baseline_eval',
            'timestamp': timestamp_now(),
            'epoch': 0,
            'step_idx': 0,
            'update_idx': update_idx,
            'score': '' if score is None else score,
            'best_score': '' if best_score is None else best_score,
            'checkpoint': baseline_path,
            'best_checkpoint': best_path,
            'best_source_checkpoint': best_source_path,
            'eval_time_sec': eval_time,
            'elapsed_sec': time.time() - t_start,
        })
    else:
        copy_checkpoint(baseline_path, best_path)
        best_source_path = baseline_path

    for epoch in range(1, int(args.epochs) + 1):
        rebuild_hd_memory_inplace(
            pline, loader, args, desc=f'* Rebuild HD memory before epoch {epoch}/{args.epochs}'
        )

        train_t0 = time.time()
        steps, avg_loss, avg_grad_norm = run_hd_memory_epoch(
            pline, loader, args, epoch, optimizer=optimizer, trainable_parameters=trainable_parameters
        )
        train_time = time.time() - train_t0
        update_idx += steps

        stats_sum = rebuild_hd_memory_inplace(
            pline, loader, args, desc=f'* Rebuild HD memory after epoch {epoch}/{args.epochs}'
        )

        checkpoint_path = ''
        save_time = 0.0
        if args.save_every_epochs > 0 and (epoch % args.save_every_epochs) == 0:
            checkpoint_path = os.path.join(model_dir, f'model_{epoch}.pt')
            save_t0 = time.time()
            save_checkpoint(checkpoint_path, pline.network, epoch, update_idx, best_score, f'epoch_{epoch}', {'pipeline': 'hd_epoch_rebuild', 'trainable': args.trainable})
            save_time = time.time() - save_t0

        score = None
        eval_time = 0.0
        if args.eval_every_epochs > 0 and (epoch % args.eval_every_epochs) == 0:
            if not checkpoint_path:
                checkpoint_path = os.path.join(model_dir, f'model_{epoch}.pt')
                save_t0 = time.time()
                save_checkpoint(checkpoint_path, pline.network, epoch, update_idx, best_score, f'epoch_{epoch}', {'pipeline': 'hd_epoch_rebuild', 'trainable': args.trainable})
                save_time += time.time() - save_t0
            eval_t0 = time.time()
            score, _rows = run_eval(pline, epoch=epoch, conf_thr=args.conf_thr)
            eval_time = time.time() - eval_t0
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                copy_checkpoint(checkpoint_path, best_path)
                best_source_path = checkpoint_path
                write_best_summary(pline.path_log, epoch, update_idx, best_score, best_path, best_source_path, args)

        append_row(metrics_csv, {
            'event': 'epoch',
            'timestamp': timestamp_now(),
            'epoch': epoch,
            'step_idx': steps,
            'update_idx': update_idx,
            'loss': avg_loss,
            'grad_norm': avg_grad_norm,
            'num_total': stats_sum['num_total'],
            'num_bg': stats_sum['num_bg'],
            'num_pos': stats_sum['num_pos'],
            'num_correct': stats_sum['num_correct'],
            'num_wrong': stats_sum['num_wrong'],
            'num_random': stats_sum['num_random'],
            'score': '' if score is None else score,
            'best_score': '' if best_score is None else best_score,
            'checkpoint': checkpoint_path,
            'best_checkpoint': best_path,
            'best_source_checkpoint': best_source_path,
            'train_time_sec': train_time,
            'eval_time_sec': eval_time,
            'save_time_sec': save_time,
            'elapsed_sec': time.time() - t_start,
        })
        print(f'* Epoch {epoch}: trainable={args.trainable} score={score}, best={best_score}, loss={avg_loss:.6f}, stats={stats_sum}', flush=True)

    if not args.skip_final_eval and int(args.epochs) > 0 and (args.eval_every_epochs <= 0 or (int(args.epochs) % args.eval_every_epochs) != 0):
        final_path = os.path.join(model_dir, f'model_{int(args.epochs)}.pt')
        save_checkpoint(final_path, pline.network, int(args.epochs), update_idx, best_score, f'epoch_{int(args.epochs)}', {'pipeline': 'hd_epoch_rebuild', 'trainable': args.trainable})
        eval_t0 = time.time()
        score, _rows = run_eval(pline, epoch=int(args.epochs), conf_thr=args.conf_thr)
        eval_time = time.time() - eval_t0
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            copy_checkpoint(final_path, best_path)
            best_source_path = final_path
            write_best_summary(pline.path_log, int(args.epochs), update_idx, best_score, best_path, best_source_path, args)
        append_row(metrics_csv, {
            'event': 'final_eval',
            'timestamp': timestamp_now(),
            'epoch': int(args.epochs),
            'update_idx': update_idx,
            'score': '' if score is None else score,
            'best_score': '' if best_score is None else best_score,
            'checkpoint': final_path,
            'best_checkpoint': best_path,
            'best_source_checkpoint': best_source_path,
            'eval_time_sec': eval_time,
            'elapsed_sec': time.time() - t_start,
        })

    last_path = os.path.join(model_dir, 'last.checkpoint')
    save_checkpoint(last_path, pline.network, int(args.epochs), update_idx, best_score, 'last', {'pipeline': 'hd_epoch_rebuild', 'trainable': args.trainable})
    append_row(metrics_csv, {
        'event': 'last_saved',
        'timestamp': timestamp_now(),
        'epoch': int(args.epochs),
        'update_idx': update_idx,
        'best_score': '' if best_score is None else best_score,
        'checkpoint': last_path,
        'best_checkpoint': best_path,
        'best_source_checkpoint': best_source_path,
        'elapsed_sec': time.time() - t_start,
    })

    rebuild_t0 = time.time()
    rebuilt_memory_path, rebuilt_memory_meta = rebuild_hd_memory(pline, args, best_path, runtime_config)
    rebuild_time = time.time() - rebuild_t0
    append_row(metrics_csv, {
        'event': 'rebuild_best_memory',
        'timestamp': timestamp_now(),
        'epoch': int(args.epochs),
        'update_idx': update_idx,
        'best_score': '' if best_score is None else best_score,
        'checkpoint': rebuilt_memory_path,
        'best_checkpoint': best_path,
        'best_source_checkpoint': best_source_path,
        'save_time_sec': rebuild_time,
        'elapsed_sec': time.time() - t_start,
    })

    print(f'* Offline HD memory retrain finished. Best score={best_score}, best={best_path}', flush=True)
    print(f'* Rebuilt best HD memory: {rebuilt_memory_path}', flush=True)
    print(f'* Rebuilt memory positives={rebuilt_memory_meta["total_positive_anchors"]}, bg={rebuilt_memory_meta["total_background_anchors"]}', flush=True)
    close_writers(pline)
    os._exit(0)


if __name__ == '__main__':
    main()
