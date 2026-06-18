#!/usr/bin/env python3
"""
Offline HyperLidar-style HD retraining for ASF/K-Radar on the source sequence.

This script keeps the loaded source HD memory as the starting point, uses the
first epoch to initialize hard-sample state from full candidates, and then
applies sampled perceptron-style HD memory updates in later epochs.

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
                        help='Total HD epochs. Epoch 1 initializes hard-sample state; later epochs use sampled buffer retrain.')
    parser.add_argument('--max_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--trainable', choices=['none', 'memory_only', 'hd_adapter', 'detection_head', 'head', 'fuser_head', 'full'], default='none',
                        help='none/memory_only updates only HD memory; hd_adapter optimizes only the HD adapter; detection_head/head optimize detector head weights; fuser_head optimizes fusion and detector head; full optimizes all non-HD parameters.')
    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adamw')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--memory_update', choices=['none', 'perceptron'], default='none',
                        help='HD memory update rule used inside each training epoch.')
    parser.add_argument('--memory_alpha', type=float, default=1.0,
                        help='Step size for sampled HD memory updates during retraining.')

    parser.add_argument('--buffer_mode', choices=['none', 'sampled'], default='none',
                        help='How to choose HD-memory update samples after epoch 1.')
    parser.add_argument('--buffer_percent', type=float, default=0.0,
                        help='Fraction of HD candidates retained for buffered retraining.')
    parser.add_argument('--wrong_buffer_percent', type=float, default=0.0,
                        help='Fraction of HD candidates selected from historical hard wrong samples.')
    parser.add_argument('--random_buffer_percent', type=float, default=0.0,
                        help='Fraction of HD candidates randomly selected for buffered retraining.')

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
def collect_hd_candidates_from_forward(head, args):
    feat_map = head.forward_ret_dict.get('hd_features', None)
    labels_map = head.forward_ret_dict.get('box_cls_labels', None)
    if feat_map is None or labels_map is None:
        raise RuntimeError('HD retrain expects head.forward_ret_dict to contain hd_features and box_cls_labels.')

    feat_anchor = head.hd_core.make_anchor_features(feat_map.detach())
    labels = labels_map.reshape(-1).long().detach()

    selected_parts = []
    max_pos_per_class = int(args.max_pos_per_class)
    for cls_id in range(1, head.num_class + 1):
        cls_idx = torch.nonzero(labels == cls_id, as_tuple=False).view(-1)
        if cls_idx.numel() == 0:
            continue
        if max_pos_per_class > 0:
            cls_idx = cls_idx[:max_pos_per_class]
        selected_parts.append(cls_idx)

    include_negative = bool(args.include_negative and head.hd_core.cfg.use_background)
    num_pos = sum(int(x.numel()) for x in selected_parts)
    if include_negative:
        neg_idx = torch.nonzero(labels == 0, as_tuple=False).view(-1)
        max_neg = int(args.max_neg_per_batch)
        max_neg_ratio = float(args.max_neg_ratio)
        if max_neg_ratio > 0.0 and num_pos > 0:
            ratio_cap = int(round(max_neg_ratio * num_pos))
            max_neg = ratio_cap if max_neg <= 0 else min(max_neg, ratio_cap)
        if max_neg > 0:
            neg_idx = neg_idx[:max_neg]
        if neg_idx.numel() > 0:
            selected_parts.append(neg_idx)

    if not selected_parts:
        empty_feat = feat_anchor.new_empty((0, feat_anchor.shape[1]))
        empty_labels = labels.new_empty((0,), dtype=torch.long)
        empty_indices = labels.new_empty((0,), dtype=torch.long)
        return empty_feat, empty_labels, empty_indices

    selected = torch.cat(selected_parts, dim=0)
    max_total_pos = int(args.max_total_pos)
    if max_total_pos > 0 and num_pos > max_total_pos:
        pos_mask = labels[selected] > 0
        pos_selected = selected[pos_mask][:max_total_pos]
        other_selected = selected[~pos_mask]
        selected = torch.cat([pos_selected, other_selected], dim=0)

    return feat_anchor[selected], labels[selected], selected


@torch.no_grad()
def compute_hd_buffer_predictions(head, feat_sel, labels_sel):
    labels_mem = labels_to_memory_indices(head.hd_core, labels_sel)
    hv_sel = head.hd_core.embedder.forward_chunked(feat_sel, chunk=int(head.hd_core.cfg.encode_chunk))
    hv_sel = torch.nn.functional.normalize(hv_sel.float(), p=2, dim=1)
    logits_sel = head.hd_core.memory.logits(hv_sel, temperature=float(head.hd_core.cfg.temperature))
    pred_sel = logits_sel.argmax(dim=1).long()

    losses = logits_sel.new_zeros((labels_mem.numel(),), dtype=logits_sel.dtype)
    wrong_mask = pred_sel != labels_mem
    if wrong_mask.any():
        row_idx = torch.arange(labels_mem.shape[0], device=labels_mem.device)
        true_scores = logits_sel[row_idx, labels_mem]
        wrong_scores = logits_sel[row_idx, pred_sel]
        losses[wrong_mask] = (wrong_scores - true_scores)[wrong_mask]

    return {
        'labels_mem': labels_mem,
        'hv_sel': hv_sel,
        'pred_sel': pred_sel,
        'wrong_mask': wrong_mask,
        'losses': losses,
    }


@torch.no_grad()
def select_buffer_indices(num_items, wrong_state, args):
    if num_items <= 0:
        empty = torch.zeros((0,), device=wrong_state.device, dtype=torch.long)
        return empty, 0

    total_target = int(num_items * float(args.buffer_percent))
    if total_target <= 0:
        empty = torch.zeros((0,), device=wrong_state.device, dtype=torch.long)
        return empty, 0

    wrong_target = int(num_items * float(args.wrong_buffer_percent))
    random_target = int(num_items * float(args.random_buffer_percent))
    if wrong_target + random_target < total_target:
        random_target += total_target - (wrong_target + random_target)
    elif wrong_target + random_target > total_target:
        overflow = wrong_target + random_target - total_target
        random_target = max(0, random_target - overflow)

    wrong_target = min(wrong_target, num_items)
    sorted_indices = torch.argsort(wrong_state, descending=True)
    wrong_indices = sorted_indices[:wrong_target]

    selected_mask = torch.zeros((num_items,), device=wrong_state.device, dtype=torch.bool)
    if wrong_indices.numel() > 0:
        selected_mask[wrong_indices] = True

    remaining_indices = torch.nonzero(~selected_mask, as_tuple=False).view(-1)
    random_target = min(random_target, int(remaining_indices.numel()))
    if random_target > 0:
        rand_perm = torch.randperm(remaining_indices.numel(), device=wrong_state.device)[:random_target]
        random_indices = remaining_indices[rand_perm]
    else:
        random_indices = torch.zeros((0,), device=wrong_state.device, dtype=torch.long)

    selected = torch.cat([wrong_indices, random_indices], dim=0)
    if selected.numel() < total_target:
        if selected.numel() > 0:
            selected_mask[selected] = True
        remaining_indices = torch.nonzero(~selected_mask, as_tuple=False).view(-1)
        extra_needed = min(total_target - selected.numel(), int(remaining_indices.numel()))
        if extra_needed > 0:
            extra_perm = torch.randperm(remaining_indices.numel(), device=wrong_state.device)[:extra_needed]
            extra_indices = remaining_indices[extra_perm]
            selected = torch.cat([selected, extra_indices], dim=0)
            random_indices = torch.cat([random_indices, extra_indices], dim=0)

    return selected, int(random_indices.numel())


@torch.no_grad()
def update_wrong_buffer_state(previous_state, selected_indices, wrong_mask_selected, losses_selected):
    next_state = previous_state.clone()
    if selected_indices.numel() > 0:
        next_state[selected_indices] = 0.0
        if wrong_mask_selected.any():
            wrong_selected_indices = selected_indices[wrong_mask_selected]
            next_state[wrong_selected_indices] = losses_selected[wrong_mask_selected].to(next_state.dtype)
    return next_state


def run_hd_memory_epoch(pline, loader, args, epoch, optimizer=None, trainable_parameters=None, wrong_buffer_state=None):
    is_trainable = optimizer is not None
    set_hd_update_mode(pline.network, train_network=is_trainable)
    head = require_hd_head(pline.network)
    if wrong_buffer_state is None:
        wrong_buffer_state = {}

    loss_sum = 0.0
    grad_norm_sum = 0.0
    stats_sum = {'num_total': 0, 'num_bg': 0, 'num_pos': 0, 'num_correct': 0, 'num_wrong': 0, 'num_random': 0}
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
        else:
            _dict_net = pline.network(batch)

        feat_sel, labels_sel, _candidate_indices = collect_hd_candidates_from_forward(head, args)
        if labels_sel.numel() > 0:
            pred_info = compute_hd_buffer_predictions(head, feat_sel, labels_sel)
            current_state = wrong_buffer_state.get(step_idx, None)
            if current_state is None or current_state.shape[0] != labels_sel.shape[0]:
                current_state = pred_info['losses'].new_zeros((labels_sel.shape[0],), dtype=torch.float32)
            else:
                current_state = current_state.to(pred_info['losses'].device, dtype=torch.float32)

            if epoch <= 1 or args.buffer_mode == 'none' or args.memory_update == 'none':
                next_state = pred_info['losses'].new_zeros((labels_sel.shape[0],), dtype=torch.float32)
                if pred_info['wrong_mask'].any():
                    next_state[pred_info['wrong_mask']] = pred_info['losses'][pred_info['wrong_mask']].to(next_state.dtype)
                wrong_buffer_state[step_idx] = next_state.detach().cpu()

                stats_sum['num_total'] += int(labels_sel.numel())
                mem_labels = pred_info['labels_mem']
                if bool(head.hd_core.cfg.use_background):
                    stats_sum['num_bg'] += int((mem_labels == 0).sum().item())
                    stats_sum['num_pos'] += int((mem_labels > 0).sum().item())
                else:
                    stats_sum['num_pos'] += int(mem_labels.numel())
            else:
                stats_sum['num_correct'] += int((~pred_info['wrong_mask']).sum().item())
                stats_sum['num_wrong'] += int(pred_info['wrong_mask'].sum().item())

                selected_indices, num_random = select_buffer_indices(labels_sel.shape[0], current_state, args)
                wrong_selected = pred_info['wrong_mask'][selected_indices] if selected_indices.numel() > 0 else pred_info['wrong_mask'].new_zeros((0,), dtype=torch.bool)
                losses_selected = pred_info['losses'][selected_indices] if selected_indices.numel() > 0 else pred_info['losses'].new_zeros((0,))
                next_state = update_wrong_buffer_state(current_state, selected_indices, wrong_selected, losses_selected)
                wrong_buffer_state[step_idx] = next_state.detach().cpu()

                if selected_indices.numel() > 0:
                    labels_mem_selected = pred_info['labels_mem'][selected_indices]
                    hv_selected = pred_info['hv_sel'][selected_indices]
                    pred_selected = pred_info['pred_sel'][selected_indices]

                    if wrong_selected.any() and args.memory_update == 'perceptron':
                        hv_wrong = hv_selected[wrong_selected]
                        labels_true_wrong = labels_mem_selected[wrong_selected]
                        labels_pred_wrong = pred_selected[wrong_selected]
                        alpha = float(args.memory_alpha)
                        head.hd_core.memory.add_(labels_true_wrong, hv_wrong, alpha=alpha)
                        head.hd_core.memory.add_(labels_true_wrong, hv_wrong, alpha=alpha)
                        head.hd_core.memory.add_(labels_pred_wrong, -hv_wrong, alpha=alpha)
                        head.hd_core.memory.add_(labels_pred_wrong, -hv_wrong, alpha=alpha)
                        head.hd_core.memory.normalize_()

                    stats_sum['num_total'] += int(selected_indices.numel())
                    stats_sum['num_random'] += int(num_random)
                    if bool(head.hd_core.cfg.use_background):
                        stats_sum['num_bg'] += int((labels_mem_selected == 0).sum().item())
                        stats_sum['num_pos'] += int((labels_mem_selected > 0).sum().item())
                    else:
                        stats_sum['num_pos'] += int(labels_mem_selected.numel())

        pbar.set_postfix(
            loss=f'{loss_sum / max(1, steps):.4f}' if is_trainable else '0.0000',
            total=stats_sum['num_total'],
            bg=stats_sum['num_bg'],
            pos=stats_sum['num_pos'],
            wrong=stats_sum['num_wrong'],
            rand=stats_sum['num_random'],
        )
        clear_batch(batch)

    avg_loss = loss_sum / max(1, steps) if is_trainable else 0.0
    avg_grad_norm = grad_norm_sum / max(1, steps) if is_trainable else 0.0
    return steps, avg_loss, avg_grad_norm, stats_sum, wrong_buffer_state


def main():
    args = parse_args()
    print('* Parsed args for offline HD retrain', flush=True)
    best_metric = {
        'CLS': args.best_metric_cls,
        'KIND': args.best_metric_kind,
        'IOUS': [float(x) for x in args.best_metric_ious],
        'CONF_THR': float(args.best_metric_conf),
        'REDUCE': 'mean',
        'ONLY_CLASSES_WITH_GT': True,
    }
    print('* Building runtime config', flush=True)
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
    print(f'* Runtime config ready: {runtime_config}', flush=True)

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    print('* Creating PipelineDetection_v1_0', flush=True)
    pline = PipelineDetection_v1_0(path_cfg=runtime_config, mode='train')
    print('* PipelineDetection_v1_0 created', flush=True)
    print(f'* Loading init model: {args.init_model}', flush=True)
    missing, unexpected = load_model_checkpoint(pline.network, args.init_model)
    print('* Init model loaded', flush=True)
    head = require_hd_head(pline.network)
    print(f'* Loading HD memory: {args.hd_memory}', flush=True)
    hd_meta = head.hd_core.load_memory(args.hd_memory, map_location='cpu')
    print('* HD memory loaded', flush=True)

    if (not args.enable_scl_loss) and hasattr(pline.network, 'is_scl'):
        pline.network.is_scl = False
        if hasattr(pline.network, 'fuser') and hasattr(pline.network.fuser, 'is_scl'):
            pline.network.fuser.is_scl = False

    trainable_info = set_trainable_scope(pline.network, args.trainable)
    optimizer = build_optimizer(pline.network, args)
    trainable_parameters = [p for p in pline.network.parameters() if p.requires_grad]

    print('* Writing run metadata', flush=True)
    with open(os.path.join(pline.path_log, 'run_meta.yml'), 'w') as f:
        yaml.safe_dump({
            'pipeline': 'hd_offline_retrain' if args.buffer_mode == 'none' else 'hyperlidar_style_hd_offline_retrain',
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
            'memory_update': args.memory_update,
            'memory_alpha': args.memory_alpha,

            'buffer_mode': args.buffer_mode,
            'buffer_percent': args.buffer_percent,
            'wrong_buffer_percent': args.wrong_buffer_percent,
            'random_buffer_percent': args.random_buffer_percent,
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
    wrong_buffer_state = {}

    baseline_path = os.path.join(model_dir, 'baseline.checkpoint')
    baseline_meta = {'pipeline': 'hd_offline_retrain', 'trainable': args.trainable}
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

    print('* Entering training loop', flush=True)
    for epoch in range(1, int(args.epochs) + 1):
        train_t0 = time.time()
        steps, avg_loss, avg_grad_norm, train_stats, wrong_buffer_state = run_hd_memory_epoch(
            pline, loader, args, epoch, optimizer=optimizer, trainable_parameters=trainable_parameters, wrong_buffer_state=wrong_buffer_state
        )
        train_time = time.time() - train_t0
        update_idx += steps

        checkpoint_path = ''
        save_time = 0.0
        if args.save_every_epochs > 0 and (epoch % args.save_every_epochs) == 0:
            checkpoint_path = os.path.join(model_dir, f'model_{epoch}.pt')
            save_t0 = time.time()
            save_checkpoint(checkpoint_path, pline.network, epoch, update_idx, best_score, f'epoch_{epoch}', {'pipeline': 'hd_offline_retrain', 'trainable': args.trainable})
            save_time = time.time() - save_t0

        score = None
        eval_time = 0.0
        if args.eval_every_epochs > 0 and (epoch % args.eval_every_epochs) == 0:
            if not checkpoint_path:
                checkpoint_path = os.path.join(model_dir, f'model_{epoch}.pt')
                save_t0 = time.time()
                save_checkpoint(checkpoint_path, pline.network, epoch, update_idx, best_score, f'epoch_{epoch}', {'pipeline': 'hd_offline_retrain', 'trainable': args.trainable})
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
            'num_total': train_stats['num_total'],
            'num_bg': train_stats['num_bg'],
            'num_pos': train_stats['num_pos'],
            'num_correct': train_stats['num_correct'],
            'num_wrong': train_stats['num_wrong'],
            'num_random': train_stats['num_random'],
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
        print(
            f'* Epoch {epoch}: trainable={args.trainable} score={score}, best={best_score}, ' 
            f'loss={avg_loss:.6f}, train_stats={train_stats}',
            flush=True,
        )

    if not args.skip_final_eval and int(args.epochs) > 0 and (args.eval_every_epochs <= 0 or (int(args.epochs) % args.eval_every_epochs) != 0):
        final_path = os.path.join(model_dir, f'model_{int(args.epochs)}.pt')
        save_checkpoint(final_path, pline.network, int(args.epochs), update_idx, best_score, f'epoch_{int(args.epochs)}', {'pipeline': 'hd_offline_retrain', 'trainable': args.trainable})
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
    save_checkpoint(last_path, pline.network, int(args.epochs), update_idx, best_score, 'last', {'pipeline': 'hd_offline_retrain', 'trainable': args.trainable})
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

    print(f'* Offline HD memory retrain finished. Best score={best_score}, best={best_path}', flush=True)
    close_writers(pline)
    os._exit(0)


if __name__ == '__main__':
    main()
