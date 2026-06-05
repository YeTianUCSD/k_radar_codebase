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
    parser = argparse.ArgumentParser(description='ASF HD-only supervised online memory adaptation.')
    parser.add_argument('--config', required=True, help='HD config. Train split is stream; test split is eval.')
    parser.add_argument('--init_model', required=True, help='Existing CNN/ASF checkpoint.')
    parser.add_argument('--hd_memory', required=True, help='Source HD memory built from seq1 train.')
    parser.add_argument('--output_root', default='./results')
    parser.add_argument('--run_name', default='online_asf_hd')
    parser.add_argument('--run_stamp', default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--trainable', choices=['cls_only', 'detection_head', 'fuser_head'], default='cls_only',
                        help='cls_only updates only HD memory; detection_head also optimizes detector head weights; fuser_head optimizes fusion and detector head weights.')
    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adamw')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--max_pos_per_class', type=int, default=4096)
    parser.add_argument('--max_total_pos', type=int, default=0)
    parser.add_argument('--max_neg_per_batch', type=int, default=8192)
    parser.add_argument('--max_neg_ratio', type=float, default=3.0)
    parser.add_argument('--eval_every_updates', type=int, default=50)
    parser.add_argument('--save_every_updates', type=int, default=50)
    parser.add_argument('--conf_thr', type=float, default=0.3)
    parser.add_argument('--skip_baseline_eval', action='store_true')
    parser.add_argument('--skip_final_eval', action='store_true')
    parser.add_argument('--best_metric_cls', default='auto')
    parser.add_argument('--best_metric_kind', choices=['bev', '3d'], default='3d')
    parser.add_argument('--best_metric_ious', type=float, nargs='+', default=[0.3, 0.5])
    parser.add_argument('--best_metric_conf', type=float, default=0.3)
    return parser.parse_args()


def append_row(path_csv, row):
    fieldnames = [
        'event', 'timestamp', 'step_idx', 'update_idx', 'loss', 'grad_norm',
        'num_total', 'num_bg', 'num_pos', 'num_correct', 'num_wrong',
        'score', 'best_score', 'checkpoint', 'best_checkpoint', 'best_source_checkpoint',
        'update_time_sec', 'eval_time_sec', 'save_time_sec', 'elapsed_sec',
    ]
    is_new = not os.path.exists(path_csv)
    with open(path_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in fieldnames})


def write_best_summary(path_log, update_idx, score, best_path, source_path, args):
    txt = os.path.join(path_log, 'best_summary.txt')
    with open(txt, 'w') as f:
        f.write(f'best_update: {update_idx}\n')
        f.write(f'score: {score:.6f}\n')
        f.write(f'metric_cls: {args.best_metric_cls}\n')
        f.write(f'metric_kind: {args.best_metric_kind}\n')
        f.write(f'metric_ious: {[float(x) for x in args.best_metric_ious]}\n')
        f.write(f'metric_conf_thr: {float(args.best_metric_conf)}\n')
        f.write(f'best_checkpoint: {best_path}\n')
        f.write(f'source_checkpoint: {source_path}\n')



def set_trainable_scope(network, scope):
    for param in network.parameters():
        param.requires_grad = False
    if scope == 'cls_only':
        return {'trainable_params': 0, 'enabled_modules': {}}

    enabled_modules = {}

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
    elif scope != 'detection_head':
        raise RuntimeError(f'Unsupported HD trainable scope: {scope}')

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
    runtime_config, runtime_cfg = make_runtime_config(
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

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pline = PipelineDetection_v1_0(path_cfg=runtime_config, mode='train')
    missing, unexpected = load_model_checkpoint(pline.network, args.init_model)
    head = require_hd_head(pline.network)
    hd_meta = head.hd_core.load_memory(args.hd_memory, map_location='cpu')

    trainable_info = set_trainable_scope(pline.network, args.trainable)
    optimizer = build_optimizer(pline.network, args)
    trainable_parameters = [p for p in pline.network.parameters() if p.requires_grad]

    with open(os.path.join(pline.path_log, 'run_meta.yml'), 'w') as f:
        yaml.safe_dump({
            'config': args.config,
            'runtime_config': runtime_config,
            'init_model': args.init_model,
            'hd_memory': args.hd_memory,
            'hd_memory_meta': hd_meta,
            'alpha': args.alpha,
            'trainable': args.trainable,
            'optimizer': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'grad_clip': args.grad_clip,
            'trainable_info': trainable_info,
            'use_background': bool(head.hd_core.cfg.use_background),
            'max_neg_per_batch': args.max_neg_per_batch,
            'max_neg_ratio': args.max_neg_ratio,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'eval_every_updates': args.eval_every_updates,
            'save_every_updates': args.save_every_updates,
            'best_metric': best_metric,
            'missing_keys': list(missing),
            'unexpected_keys': list(unexpected),
        }, f, sort_keys=False)

    stream_loader = torch.utils.data.DataLoader(
        pline.dataset_train,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=pline.dataset_train.collate_fn,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    model_dir = os.path.join(pline.path_log, 'models')
    os.makedirs(model_dir, exist_ok=True)
    metrics_csv = os.path.join(pline.path_log, 'online_metrics.csv')
    best_path = os.path.join(model_dir, 'best.checkpoint')
    best_source_path = ''
    best_score = None
    t_start = time.time()

    baseline_path = os.path.join(model_dir, 'baseline.checkpoint')
    save_t0 = time.time()
    save_checkpoint(baseline_path, pline.network, 0, 0, best_score, 'baseline', {'alpha': args.alpha, 'trainable': args.trainable})
    baseline_save_time = time.time() - save_t0

    if not args.skip_baseline_eval:
        eval_t0 = time.time()
        score, _rows = run_eval(pline, epoch=0, conf_thr=args.conf_thr)
        eval_time = time.time() - eval_t0
        best_score = score
        save_checkpoint(baseline_path, pline.network, 0, 0, best_score, 'baseline', {'alpha': args.alpha, 'trainable': args.trainable})
        copy_checkpoint(baseline_path, best_path)
        best_source_path = baseline_path
        if best_score is not None:
            write_best_summary(pline.path_log, 0, best_score, best_path, best_source_path, args)
        append_row(metrics_csv, {
            'event': 'baseline_eval',
            'timestamp': timestamp_now(),
            'step_idx': 0,
            'update_idx': 0,
            'score': '' if score is None else score,
            'best_score': '' if best_score is None else best_score,
            'checkpoint': baseline_path,
            'best_checkpoint': best_path,
            'best_source_checkpoint': best_source_path,
            'eval_time_sec': eval_time,
            'save_time_sec': baseline_save_time,
            'elapsed_sec': time.time() - t_start,
        })
    else:
        copy_checkpoint(baseline_path, best_path)
        best_source_path = baseline_path

    step_idx = 0
    update_idx = 0
    last_eval_update = 0 if not args.skip_baseline_eval else -1
    pbar = tqdm(stream_loader, desc='* HD online stream')
    for batch in pbar:
        if args.max_steps > 0 and step_idx >= args.max_steps:
            clear_batch(batch)
            break
        step_idx += 1
        update_idx += 1

        update_t0 = time.time()
        loss_value = ''
        norm_before_clip = ''
        if args.trainable in ['detection_head', 'fuser_head']:
            pline.network.train()
            pline.network.training = True
            optimizer.zero_grad(set_to_none=True)
            dict_net = pline.network(batch)
            if pline.get_loss_from == 'head':
                loss = pline.network.head.loss(dict_net)
            elif pline.get_loss_from == 'detector':
                loss = pline.network.loss(dict_net)
            else:
                raise RuntimeError(f'Unsupported get_loss_from={pline.get_loss_from}')
            loss_value = float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float(loss)
            did_backward = False
            if torch.is_tensor(loss) and torch.isfinite(loss):
                if loss.requires_grad:
                    loss.backward()
                    did_backward = True
            elif loss_value != 0.0:
                raise RuntimeError(f'Non-finite online loss at step={step_idx}')
            if did_backward:
                norm_before_clip = grad_norm(trainable_parameters)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=float(args.grad_clip))
                optimizer.step()
            else:
                norm_before_clip = 0.0
            feat_pos, labels_pos = extract_hd_features_by_labels(
                pline.network,
                batch,
                max_pos_per_class=args.max_pos_per_class,
                max_total_pos=args.max_total_pos,
                include_negative=bool(head.hd_core.cfg.use_background),
                max_neg_per_batch=args.max_neg_per_batch,
                max_neg_ratio=args.max_neg_ratio,
            )
        else:
            feat_pos, labels_pos = extract_hd_features_by_labels(
                pline.network,
                batch,
                max_pos_per_class=args.max_pos_per_class,
                max_total_pos=args.max_total_pos,
                include_negative=bool(head.hd_core.cfg.use_background),
                max_neg_per_batch=args.max_neg_per_batch,
                max_neg_ratio=args.max_neg_ratio,
            )
        stats = head.hd_core.adaptive_update(feat_pos, labels_pos, alpha=float(args.alpha))
        update_time = time.time() - update_t0

        checkpoint_path = ''
        save_time = 0.0
        if args.save_every_updates > 0 and update_idx % args.save_every_updates == 0:
            checkpoint_path = os.path.join(model_dir, f'update_{update_idx:04d}.checkpoint')
            save_t0 = time.time()
            save_checkpoint(checkpoint_path, pline.network, step_idx, update_idx, best_score, f'update_{update_idx:04d}', {'alpha': args.alpha, 'trainable': args.trainable})
            save_time = time.time() - save_t0

        score = None
        eval_time = 0.0
        if args.eval_every_updates > 0 and update_idx % args.eval_every_updates == 0:
            if not checkpoint_path:
                checkpoint_path = os.path.join(model_dir, f'update_{update_idx:04d}.checkpoint')
                save_t0 = time.time()
                save_checkpoint(checkpoint_path, pline.network, step_idx, update_idx, best_score, f'update_{update_idx:04d}', {'alpha': args.alpha, 'trainable': args.trainable})
                save_time += time.time() - save_t0
            eval_t0 = time.time()
            score, _rows = run_eval(pline, epoch=update_idx, conf_thr=args.conf_thr)
            eval_time = time.time() - eval_t0
            last_eval_update = update_idx
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                copy_checkpoint(checkpoint_path, best_path)
                best_source_path = checkpoint_path
                write_best_summary(pline.path_log, update_idx, best_score, best_path, best_source_path, args)

        append_row(metrics_csv, {
            'event': 'update',
            'timestamp': timestamp_now(),
            'step_idx': step_idx,
            'update_idx': update_idx,
            'loss': loss_value,
            'grad_norm': norm_before_clip,
            'num_total': stats.get('num_total', stats.get('num_pos', 0)),
            'num_bg': stats.get('num_bg', 0),
            'num_pos': stats['num_pos'],
            'num_correct': stats['num_correct'],
            'num_wrong': stats['num_wrong'],
            'score': '' if score is None else score,
            'best_score': '' if best_score is None else best_score,
            'checkpoint': checkpoint_path,
            'best_checkpoint': best_path,
            'best_source_checkpoint': best_source_path,
            'update_time_sec': update_time,
            'eval_time_sec': eval_time,
            'save_time_sec': save_time,
            'elapsed_sec': time.time() - t_start,
        })
        pbar.set_postfix(bg=stats.get('num_bg', 0), pos=stats['num_pos'], wrong=stats['num_wrong'], best='nan' if best_score is None else f'{best_score:.4f}')
        clear_batch(batch)

    if not args.skip_final_eval and update_idx > 0 and last_eval_update != update_idx:
        final_path = os.path.join(model_dir, f'update_{update_idx:04d}.checkpoint')
        save_checkpoint(final_path, pline.network, step_idx, update_idx, best_score, f'update_{update_idx:04d}', {'alpha': args.alpha, 'trainable': args.trainable})
        eval_t0 = time.time()
        score, _rows = run_eval(pline, epoch=update_idx, conf_thr=args.conf_thr)
        eval_time = time.time() - eval_t0
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            copy_checkpoint(final_path, best_path)
            best_source_path = final_path
            write_best_summary(pline.path_log, update_idx, best_score, best_path, best_source_path, args)
        append_row(metrics_csv, {
            'event': 'final_eval',
            'timestamp': timestamp_now(),
            'step_idx': step_idx,
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
    save_checkpoint(last_path, pline.network, step_idx, update_idx, best_score, 'last', {'alpha': args.alpha, 'trainable': args.trainable})
    append_row(metrics_csv, {
        'event': 'last_saved',
        'timestamp': timestamp_now(),
        'step_idx': step_idx,
        'update_idx': update_idx,
        'best_score': '' if best_score is None else best_score,
        'checkpoint': last_path,
        'best_checkpoint': best_path,
        'best_source_checkpoint': best_source_path,
        'elapsed_sec': time.time() - t_start,
    })
    print(f'* HD online finished. Best score={best_score}, best={best_path}')
    close_writers(pline)
    os._exit(0)


if __name__ == '__main__':
    main()
