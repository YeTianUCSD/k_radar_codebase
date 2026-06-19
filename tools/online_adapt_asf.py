'''
Supervised online adaptation runner for K-Radar ASF.

The stream split is consumed once, in dataset order, and every incoming batch
performs one fine-tuning update. Periodic full evaluation runs on the fixed
test split from the same config.
'''

import argparse
import csv
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='ASF supervised online adaptation')
    parser.add_argument('--config', type=str, required=True,
                        help='ASF config. Train split is used as stream, test split as fixed eval set.')
    parser.add_argument('--init_model', type=str, required=True,
                        help='Initial seq1 checkpoint. Supports raw model state_dict or util dict.')
    parser.add_argument('--output_root', type=str, default='./results',
                        help='Root output directory.')
    parser.add_argument('--run_name', type=str, default='online_asf',
                        help='Semantic run name. Final dir gets an exp timestamp suffix.')
    parser.add_argument('--run_stamp', type=str, default=None,
                        help='Optional YYMMDD_HHMMSS stamp used in the final run directory name.')

    parser.add_argument('--trainable', choices=['cls', 'head', 'fuser_head', 'full'], default='head',
                        help='Parameter scope to update online. cls updates only head.conv_cls; head updates the full detector head.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Stream batch size.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Stream and eval dataloader workers.')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Maximum stream batches to consume. -1 means all.')

    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adamw')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=float, default=0.0)

    parser.add_argument('--eval_every_updates', type=int, default=10,
                        help='Run full seq test evaluation every N updates. 0 disables periodic eval.')
    parser.add_argument('--save_every_updates', type=int, default=10,
                        help='Save update checkpoint every N updates. 0 disables periodic checkpoint saving.')
    parser.add_argument('--conf_thr', type=float, default=0.3,
                        help='Confidence threshold used for full evaluation output.')
    parser.add_argument('--skip_baseline_eval', action='store_true',
                        help='Skip eval before the first online update.')
    parser.add_argument('--skip_final_eval', action='store_true',
                        help='Skip final eval if last update is not already evaluated.')

    parser.add_argument('--best_metric_cls', type=str, default='auto',
                        help='Class keyword used for best checkpoint. auto averages classes with GT.')
    parser.add_argument('--best_metric_kind', choices=['bev', '3d'], default='3d')
    parser.add_argument('--best_metric_ious', type=float, nargs='+', default=[0.3, 0.5])
    parser.add_argument('--best_metric_conf', type=float, default=0.3)

    parser.add_argument('--superpose_enable', action='store_true',
                        help='Enable online superposition export during adaptation.')
    parser.add_argument('--superpose_base_scene', type=str, default='seq1',
                        help='Name of the frozen base scene/model.')
    parser.add_argument('--superpose_scene', type=str, default=None,
                        help='Name of the current online adaptation scene, e.g. seq58.')
    parser.add_argument('--superpose_modules', type=str, nargs='+', default=['fuser', 'head'],
                        help='Top-level state_dict prefixes managed by superposition.')
    parser.add_argument('--superpose_seed', type=int, default=20260612,
                        help='Seed for deterministic binary sign contexts.')
    parser.add_argument('--superpose_bundle', type=str, default=None,
                        help='Optional existing superposition bundle used to continue accumulating scenes.')
    parser.add_argument('--superpose_materialize_all_scenes', action='store_true',
                        help='Materialize checkpoints for every known scene at each export.')
    parser.add_argument('--superpose_store_deltas', action='store_true',
                        help='Also cache exact per-scene deltas inside the bundle for later exact-recovery analysis.')
    return parser.parse_args()


def make_runtime_config(path_config, args):
    with open(path_config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['GENERAL']['NAME'] = args.run_name
    cfg['GENERAL']['LOGGING']['PATH_LOGGING'] = args.output_root
    cfg['GENERAL']['LOGGING']['IS_SAVE_MODEL'] = True
    cfg['GENERAL']['LOGGING']['ALLOW_EXISTING_DIR'] = True
    if args.run_stamp is not None:
        cfg['GENERAL']['LOGGING']['RUN_STAMP'] = args.run_stamp
    cfg['OPTIMIZER']['BATCH_SIZE'] = int(args.batch_size)
    cfg['OPTIMIZER']['NUM_WORKERS'] = int(args.num_workers)
    cfg['VAL']['IS_VALIDATE'] = True
    cfg['VAL']['IS_CONSIDER_VAL_SUBSET'] = False
    cfg['VAL']['LIST_VAL_CONF_THR'] = [float(args.conf_thr)]
    cfg['GENERAL']['LOGGING']['BEST_METRIC'] = {
        'CLS': args.best_metric_cls,
        'KIND': args.best_metric_kind,
        'IOUS': [float(x) for x in args.best_metric_ious],
        'CONF_THR': float(args.best_metric_conf),
        'REDUCE': 'mean',
        'ONLY_CLASSES_WITH_GT': True,
    }

    tmp = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yml',
        prefix='kradar_online_runtime_',
        delete=False,
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return tmp.name, cfg


def load_model_checkpoint(network, path_ckpt):
    payload = torch.load(path_ckpt, map_location='cpu')
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        state_dict = payload['model_state_dict']
    elif isinstance(payload, dict) and 'state_dict' in payload:
        state_dict = payload['state_dict']
    else:
        state_dict = payload
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def set_module_trainable(module, flag):
    n_params = 0
    if module is None:
        return n_params
    for param in module.parameters():
        n_params += param.numel()
        param.requires_grad = bool(flag) and (param.is_floating_point() or param.is_complex())
    return n_params


def set_trainable_scope(network, scope):
    total = 0
    for param in network.parameters():
        param.requires_grad = False
        total += param.numel()

    enabled = {}
    if scope == 'cls':
        head = getattr(network, 'head', None)
        conv_cls = getattr(head, 'conv_cls', None) if head is not None else None
        if conv_cls is None:
            raise RuntimeError('Trainable scope cls requires network.head.conv_cls')
        enabled['head.conv_cls'] = set_module_trainable(conv_cls, True)
    elif scope == 'head':
        enabled['head'] = set_module_trainable(getattr(network, 'head', None), True)
    elif scope == 'fuser_head':
        enabled['fuser'] = set_module_trainable(getattr(network, 'fuser', None), True)
        enabled['head'] = set_module_trainable(getattr(network, 'head', None), True)
    elif scope == 'full':
        for param in network.parameters():
            param.requires_grad = param.is_floating_point() or param.is_complex()
        enabled['full'] = total
    else:
        raise RuntimeError(f'Unsupported trainable scope: {scope}')

    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    if trainable <= 0:
        raise RuntimeError(f'No trainable parameters found for scope={scope}')
    return total, trainable, enabled


def build_optimizer(network, args):
    params = [p for p in network.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'sgd':
        return torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise RuntimeError(f'Unsupported optimizer: {args.optimizer}')


def grad_norm(parameters):
    total_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = param.grad.detach().data.norm(2).item()
        total_sq += norm * norm
    return total_sq ** 0.5


def save_checkpoint(path, network, optimizer, step_idx, update_idx, best_score, args, tag):
    torch.save(network.state_dict(), path)
    meta = {
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'step_idx': int(step_idx),
        'update_idx': int(update_idx),
        'best_score': None if best_score is None else float(best_score),
        'trainable': args.trainable,
        'tag': tag,
    }
    torch.save(meta, str(path) + '.state')


def copy_checkpoint(src_path, dst_path):
    shutil.copy2(src_path, dst_path)
    src_state = str(src_path) + '.state'
    dst_state = str(dst_path) + '.state'
    if os.path.exists(src_state):
        shutil.copy2(src_state, dst_state)


def append_metric_row(path_csv, row):
    fieldnames = [
        'event', 'timestamp', 'step_idx', 'update_idx', 'loss', 'grad_norm',
        'score', 'best_score', 'checkpoint', 'best_checkpoint',
        'best_source_checkpoint', 'superpose_bundle', 'superpose_materialized_dir',
        'superpose_base_checkpoint', 'superpose_active_checkpoint', 'update_time_sec', 'eval_time_sec',
        'save_time_sec', 'elapsed_sec',
    ]
    is_new = not os.path.exists(path_csv)
    with open(path_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in fieldnames})


def write_run_meta(path_log, args, runtime_cfg, trainable_info, missing, unexpected):
    meta = {
        'config': args.config,
        'init_model': args.init_model,
        'run_name': args.run_name,
        'run_stamp': args.run_stamp,
        'trainable': args.trainable,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'max_steps': args.max_steps,
        'optimizer': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'grad_clip': args.grad_clip,
        'eval_every_updates': args.eval_every_updates,
        'save_every_updates': args.save_every_updates,
        'conf_thr': args.conf_thr,
        'best_metric': runtime_cfg['GENERAL']['LOGGING']['BEST_METRIC'],
        'superposition': {
            'enabled': bool(args.superpose_enable),
            'base_scene': args.superpose_base_scene,
            'active_scene': args.superpose_scene,
            'modules': list(args.superpose_modules),
            'seed': int(args.superpose_seed),
            'bundle': args.superpose_bundle,
            'materialize_all_scenes': bool(args.superpose_materialize_all_scenes),
            'store_deltas': bool(args.superpose_store_deltas),
            'materialize_mode': 'approximate',
        },
        'trainable_info': trainable_info,
        'missing_keys': list(missing),
        'unexpected_keys': list(unexpected),
    }
    with open(os.path.join(path_log, 'run_meta.yml'), 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def write_best_summary(path_log, update_idx, score, best_path, source_path, args):
    metric = {
        'cls': args.best_metric_cls,
        'kind': args.best_metric_kind,
        'ious': [float(x) for x in args.best_metric_ious],
        'conf_thr': float(args.best_metric_conf),
        'only_classes_with_gt': True,
    }
    path_txt = os.path.join(path_log, 'best_summary.txt')
    with open(path_txt, 'w') as f:
        f.write(f'best_update: {update_idx}\n')
        f.write(f'score: {score:.6f}\n')
        f.write(f"metric_cls: {metric['cls']}\n")
        f.write(f"metric_kind: {metric['kind']}\n")
        f.write(f"metric_ious: {metric['ious']}\n")
        f.write(f"metric_conf_thr: {metric['conf_thr']}\n")
        f.write(f"only_classes_with_gt: {metric['only_classes_with_gt']}\n")
        f.write(f'best_checkpoint: {best_path}\n')
        f.write(f'source_checkpoint: {source_path}\n')

    path_csv = os.path.join(path_log, 'best_summary.csv')
    with open(path_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'best_update', 'score', 'metric_cls', 'metric_kind',
            'metric_ious', 'metric_conf_thr', 'best_checkpoint',
            'source_checkpoint',
        ])
        writer.writerow([
            update_idx,
            score,
            metric['cls'],
            metric['kind'],
            ';'.join(str(x) for x in metric['ious']),
            metric['conf_thr'],
            best_path,
            source_path,
        ])


def build_superposition_manager(args, path_log):
    if not args.superpose_enable:
        return None
    if not args.superpose_scene:
        raise RuntimeError('--superpose_scene is required when --superpose_enable is set')

    from tools.superposition.asf_online_superposition import ASFOnlineSuperpositionManager, load_state_dict

    output_dir = os.path.join(path_log, 'superposition')
    if args.superpose_bundle:
        manager = ASFOnlineSuperpositionManager.from_bundle(
            args.superpose_bundle,
            active_scene=args.superpose_scene,
            output_dir=output_dir,
        )
        manager.store_scene_deltas = bool(manager.store_scene_deltas or args.superpose_store_deltas)
        if manager.store_scene_deltas:
            manager.ensure_scene(args.superpose_scene)
        if list(args.superpose_modules) != list(manager.modules):
            raise RuntimeError(
                f'Superposition modules mismatch: cli={args.superpose_modules}, bundle={manager.modules}'
            )
        if str(args.superpose_base_scene) != str(manager.base_scene):
            raise RuntimeError(
                f'Superposition base scene mismatch: cli={args.superpose_base_scene}, bundle={manager.base_scene}'
            )
    else:
        manager = ASFOnlineSuperpositionManager(
            base_state=load_state_dict(args.init_model),
            base_model_path=args.init_model,
            modules=args.superpose_modules,
            base_scene=args.superpose_base_scene,
            active_scene=args.superpose_scene,
            seed=args.superpose_seed,
            output_dir=output_dir,
            store_scene_deltas=args.superpose_store_deltas,
        )
    return manager


def export_superposition_snapshot(manager, network, args, tag, step_idx, update_idx, score=None, best_score=None, source_checkpoint=None):
    if manager is None:
        return {}
    manager.capture_scene_delta_from_state_dict(network.state_dict(), scene_name=args.superpose_scene)
    if args.superpose_materialize_all_scenes:
        scenes = [manager.base_scene] + list(manager.scene_order)
    else:
        scenes = [manager.base_scene, args.superpose_scene]
    return manager.export(
        tag=tag,
        step_idx=step_idx,
        update_idx=update_idx,
        score=score,
        best_score=best_score,
        source_checkpoint=source_checkpoint,
        materialize_scene_names=scenes,
        materialize_exact=False,
    )


def superpose_metric_fields(args, export_info):
    if not args.superpose_enable or not export_info:
        return {
            'superpose_bundle': '',
            'superpose_materialized_dir': '',
            'superpose_base_checkpoint': '',
            'superpose_active_checkpoint': '',
        }
    return {
        'superpose_bundle': export_info.get('bundle_path', ''),
        'superpose_materialized_dir': export_info.get('materialized_dir', ''),
        'superpose_base_checkpoint': export_info.get('scene_paths', {}).get(args.superpose_base_scene, ''),
        'superpose_active_checkpoint': export_info.get('scene_paths', {}).get(args.superpose_scene, ''),
    }


def validate_superposition_init_state(manager, network, args):
    if manager is None:
        return
    expected_state = manager.expected_init_state(args.superpose_scene)
    current_state = network.state_dict()
    max_abs_diff = 0.0
    max_key = ''
    for key in manager.selected_keys:
        diff = (current_state[key].detach().cpu() - expected_state[key].detach().cpu()).abs()
        if diff.numel() == 0:
            continue
        cur_max = float(diff.max())
        if cur_max > max_abs_diff:
            max_abs_diff = cur_max
            max_key = key
    if max_abs_diff > 1e-6:
        if manager.has_scene(args.superpose_scene):
            expected_desc = f"the recovered state for existing scene '{args.superpose_scene}' from --superpose_bundle"
        else:
            expected_desc = f"the base scene '{manager.base_scene}' state before starting new scene '{args.superpose_scene}'"
        raise RuntimeError(
            'Superposition init mismatch: --init_model does not match ' + expected_desc + '. '
            + f'Max abs diff = {max_abs_diff:.6g} at key={max_key}. '
            + 'Use the matching recovered checkpoint or the base model before continuing.'
        )


def clear_batch(batch):
    if not isinstance(batch, dict):
        return
    if 'pointer' in batch:
        for item in batch['pointer']:
            for key in list(item.keys()):
                if key != 'meta':
                    item[key] = None
    for key in list(batch.keys()):
        batch[key] = None


def run_eval(pline, epoch, args):
    eval_rows = pline.validate_kitti(epoch=epoch, list_conf_thr=[float(args.conf_thr)], is_subset=False)
    return pline.pick_best_metric_score(eval_rows), eval_rows


def timestamp_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def close_writers(pline):
    for writer_name in ('log_train_iter', 'log_train_epoch', 'log_test'):
        writer = getattr(pline, writer_name, None)
        if writer is not None:
            writer.close()


def main():
    args = parse_args()

    path_runtime_config, runtime_cfg = make_runtime_config(args.config, args)
    print(f'* Runtime config generated: {path_runtime_config}', flush=True)
    print(f"* Output root = {runtime_cfg['GENERAL']['LOGGING']['PATH_LOGGING']}", flush=True)
    print(f"* Run name = {runtime_cfg['GENERAL']['NAME']}", flush=True)
    print(f"* Trainable scope = {args.trainable}", flush=True)
    print(f"* Best metric = {runtime_cfg['GENERAL']['LOGGING']['BEST_METRIC']}", flush=True)

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pline = PipelineDetection_v1_0(path_cfg=path_runtime_config, mode='train')
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))

    missing, unexpected = load_model_checkpoint(pline.network, args.init_model)
    print(f'* Loaded init model: {args.init_model}')
    print(f'* Missing keys: {len(missing)} / Unexpected keys: {len(unexpected)}')

    total_params, trainable_params, enabled = set_trainable_scope(pline.network, args.trainable)
    trainable_info = {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'trainable_percent': 100.0 * float(trainable_params) / max(1, float(total_params)),
        'enabled_modules': {k: int(v) for k, v in enabled.items()},
    }
    print(
        '* Trainable params = '
        f"{trainable_info['trainable_params']}/{trainable_info['total_params']} "
        f"({trainable_info['trainable_percent']:.4f}%)"
    )
    print(f"* Enabled modules = {trainable_info['enabled_modules']}")

    optimizer = build_optimizer(pline.network, args)
    write_run_meta(pline.path_log, args, runtime_cfg, trainable_info, missing, unexpected)
    superpose_manager = build_superposition_manager(args, pline.path_log)
    if superpose_manager is not None:
        validate_superposition_init_state(superpose_manager, pline.network, args)
        superpose_manager.initialize_active_scene_from_state_dict(pline.network.state_dict(), args.superpose_scene)
        print(f'* Superposition enabled: base={superpose_manager.base_scene}, active={superpose_manager.active_scene}')
        print(f'* Superposition modules = {superpose_manager.modules}')
        print(f'* Superposition materialize mode = approximate (pure unbind)')
        print(f'* Superposition store exact deltas = {superpose_manager.store_scene_deltas}')
        if args.superpose_bundle:
            print(f'* Continuing superposition bundle: {args.superpose_bundle}')

    stream_loader = torch.utils.data.DataLoader(
        pline.dataset_train,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=pline.dataset_train.collate_fn,
        num_workers=int(args.num_workers),
        drop_last=False,
    )
    print(f'* Stream dataset size = {len(pline.dataset_train)}')
    print(f'* Eval dataset size = {len(pline.dataset_test)}')
    print(f'* Stream loader batches = {len(stream_loader)}')

    model_dir = os.path.join(pline.path_log, 'models')
    os.makedirs(model_dir, exist_ok=True)
    metrics_csv = os.path.join(pline.path_log, 'online_metrics.csv')
    best_score = None
    best_path = os.path.join(model_dir, 'best.checkpoint')
    best_source_path = ''
    t_start = time.time()

    baseline_path = os.path.join(model_dir, 'baseline.checkpoint')
    save_t0 = time.time()
    save_checkpoint(baseline_path, pline.network, optimizer, 0, 0, best_score, args, 'baseline')
    baseline_superpose = export_superposition_snapshot(
        superpose_manager,
        pline.network,
        args,
        tag='baseline',
        step_idx=0,
        update_idx=0,
        score=None,
        best_score=best_score,
        source_checkpoint=baseline_path,
    )
    baseline_save_time = time.time() - save_t0

    if not args.skip_baseline_eval:
        eval_t0 = time.time()
        score, _rows = run_eval(pline, epoch=0, args=args)
        baseline_eval_time = time.time() - eval_t0
        best_score = score
        save_t0 = time.time()
        save_checkpoint(baseline_path, pline.network, optimizer, 0, 0, best_score, args, 'baseline')
        copy_checkpoint(baseline_path, best_path)
        baseline_save_time += time.time() - save_t0
        best_source_path = baseline_path
        if best_score is not None:
            write_best_summary(pline.path_log, 0, best_score, best_path, best_source_path, args)
        baseline_superpose = export_superposition_snapshot(
            superpose_manager,
            pline.network,
            args,
            tag='baseline',
            step_idx=0,
            update_idx=0,
            score=score,
            best_score=best_score,
            source_checkpoint=baseline_path,
        )
        append_metric_row(metrics_csv, {
            'event': 'baseline_eval',
            'timestamp': timestamp_now(),
            'step_idx': 0,
            'update_idx': 0,
            'score': '' if score is None else score,
            'best_score': '' if best_score is None else best_score,
            'checkpoint': baseline_path,
            'best_checkpoint': best_path,
            'best_source_checkpoint': best_source_path,
            'eval_time_sec': baseline_eval_time,
            'save_time_sec': baseline_save_time,
            'elapsed_sec': time.time() - t_start,
            **superpose_metric_fields(args, baseline_superpose),
        })
        print(f'* Baseline eval score = {score}')
    else:
        copy_checkpoint(baseline_path, best_path)
        best_source_path = baseline_path
        append_metric_row(metrics_csv, {
            'event': 'baseline_saved',
            'timestamp': timestamp_now(),
            'step_idx': 0,
            'update_idx': 0,
            'checkpoint': baseline_path,
            'best_checkpoint': best_path,
            'best_source_checkpoint': best_source_path,
            'save_time_sec': baseline_save_time,
            'elapsed_sec': time.time() - t_start,
            **superpose_metric_fields(args, baseline_superpose),
        })

    step_idx = 0
    update_idx = 0
    last_eval_update = 0 if not args.skip_baseline_eval else -1
    idx_log_iter = 0
    trainable_parameters = [p for p in pline.network.parameters() if p.requires_grad]

    pbar = tqdm(stream_loader, desc='* Online stream')
    for batch in pbar:
        if args.max_steps > 0 and step_idx >= args.max_steps:
            clear_batch(batch)
            break

        step_idx += 1
        update_idx += 1
        update_t0 = time.time()
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
        if torch.is_tensor(loss) and torch.isfinite(loss):
            loss.backward()
        elif loss_value == 0.0:
            pass
        else:
            meta = batch.get('meta', None) if isinstance(batch, dict) else None
            raise RuntimeError(f'Non-finite online loss at step={step_idx}, meta={meta}')

        norm_before_clip = grad_norm(trainable_parameters)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=float(args.grad_clip))
        optimizer.step()
        update_time = time.time() - update_t0

        if pline.is_logging and 'logging' in dict_net:
            idx_log_iter += 1
            for key, val in dict_net['logging'].items():
                pline.log_train_iter.add_scalar(f'online/{key}', val, idx_log_iter)
            pline.log_train_iter.add_scalar('online/loss', loss_value, idx_log_iter)
            pline.log_train_iter.add_scalar('online/learning_rate', float(args.lr), idx_log_iter)

        checkpoint_path = ''
        save_time = 0.0
        checkpoint_superpose = {}
        if args.save_every_updates > 0 and (update_idx % args.save_every_updates) == 0:
            checkpoint_path = os.path.join(model_dir, f'update_{update_idx:04d}.checkpoint')
            save_t0 = time.time()
            save_checkpoint(
                checkpoint_path,
                pline.network,
                optimizer,
                step_idx,
                update_idx,
                best_score,
                args,
                f'update_{update_idx:04d}',
            )
            checkpoint_superpose = export_superposition_snapshot(
                superpose_manager,
                pline.network,
                args,
                tag=f'update_{update_idx:04d}',
                step_idx=step_idx,
                update_idx=update_idx,
                score=None,
                best_score=best_score,
                source_checkpoint=checkpoint_path,
            )
            save_time += time.time() - save_t0

        score = None
        eval_time = 0.0
        if args.eval_every_updates > 0 and (update_idx % args.eval_every_updates) == 0:
            if not checkpoint_path:
                checkpoint_path = os.path.join(model_dir, f'update_{update_idx:04d}.checkpoint')
                save_t0 = time.time()
                save_checkpoint(
                    checkpoint_path,
                    pline.network,
                    optimizer,
                    step_idx,
                    update_idx,
                    best_score,
                    args,
                    f'update_{update_idx:04d}',
                )
                checkpoint_superpose = export_superposition_snapshot(
                    superpose_manager,
                    pline.network,
                    args,
                    tag=f'update_{update_idx:04d}',
                    step_idx=step_idx,
                    update_idx=update_idx,
                    score=None,
                    best_score=best_score,
                    source_checkpoint=checkpoint_path,
                )
                save_time += time.time() - save_t0
            eval_t0 = time.time()
            score, _rows = run_eval(pline, epoch=update_idx, args=args)
            eval_time = time.time() - eval_t0
            last_eval_update = update_idx
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                save_t0 = time.time()
                copy_checkpoint(checkpoint_path, best_path)
                save_time += time.time() - save_t0
                best_source_path = checkpoint_path
                write_best_summary(pline.path_log, update_idx, best_score, best_path, best_source_path, args)
                best_superpose = export_superposition_snapshot(
                    superpose_manager,
                    pline.network,
                    args,
                    tag='best',
                    step_idx=step_idx,
                    update_idx=update_idx,
                    score=score,
                    best_score=best_score,
                    source_checkpoint=checkpoint_path,
                )
                append_metric_row(metrics_csv, {
                    'event': 'best_updated',
                    'timestamp': timestamp_now(),
                    'step_idx': step_idx,
                    'update_idx': update_idx,
                    'score': score,
                    'best_score': best_score,
                    'checkpoint': checkpoint_path,
                    'best_checkpoint': best_path,
                    'best_source_checkpoint': best_source_path,
                    'update_time_sec': update_time,
                    'eval_time_sec': eval_time,
                    'save_time_sec': save_time,
                    'elapsed_sec': time.time() - t_start,
                    **superpose_metric_fields(args, best_superpose),
                })
                print(f'* Best checkpoint updated: update={update_idx}, score={best_score:.6f}')

        append_metric_row(metrics_csv, {
            'event': 'update',
            'timestamp': timestamp_now(),
            'step_idx': step_idx,
            'update_idx': update_idx,
            'loss': loss_value,
            'grad_norm': norm_before_clip,
            'score': '' if score is None else score,
            'best_score': '' if best_score is None else best_score,
            'checkpoint': checkpoint_path,
            'best_checkpoint': best_path if best_source_path else '',
            'best_source_checkpoint': best_source_path,
            'update_time_sec': update_time,
            'eval_time_sec': eval_time,
            'save_time_sec': save_time,
            'elapsed_sec': time.time() - t_start,
            **superpose_metric_fields(args, checkpoint_superpose),
        })
        pbar.set_postfix(loss=f'{loss_value:.4f}', best='nan' if best_score is None else f'{best_score:.4f}')
        clear_batch(batch)

    if not args.skip_final_eval and update_idx > 0 and last_eval_update != update_idx:
        final_path = os.path.join(model_dir, f'update_{update_idx:04d}.checkpoint')
        save_t0 = time.time()
        save_checkpoint(
            final_path,
            pline.network,
            optimizer,
            step_idx,
            update_idx,
            best_score,
            args,
            f'update_{update_idx:04d}',
        )
        final_superpose = export_superposition_snapshot(
            superpose_manager,
            pline.network,
            args,
            tag=f'update_{update_idx:04d}',
            step_idx=step_idx,
            update_idx=update_idx,
            score=None,
            best_score=best_score,
            source_checkpoint=final_path,
        )
        final_save_time = time.time() - save_t0
        eval_t0 = time.time()
        score, _rows = run_eval(pline, epoch=update_idx, args=args)
        final_eval_time = time.time() - eval_t0
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            save_t0 = time.time()
            copy_checkpoint(final_path, best_path)
            final_save_time += time.time() - save_t0
            best_source_path = final_path
            write_best_summary(pline.path_log, update_idx, best_score, best_path, best_source_path, args)
            best_superpose = export_superposition_snapshot(
                superpose_manager,
                pline.network,
                args,
                tag='best',
                step_idx=step_idx,
                update_idx=update_idx,
                score=score,
                best_score=best_score,
                source_checkpoint=final_path,
            )
            append_metric_row(metrics_csv, {
                'event': 'best_updated',
                'timestamp': timestamp_now(),
                'step_idx': step_idx,
                'update_idx': update_idx,
                'score': score,
                'best_score': best_score,
                'checkpoint': final_path,
                'best_checkpoint': best_path,
                'best_source_checkpoint': best_source_path,
                'eval_time_sec': final_eval_time,
                'save_time_sec': final_save_time,
                'elapsed_sec': time.time() - t_start,
                **superpose_metric_fields(args, best_superpose),
            })
            print(f'* Best checkpoint updated at final eval: update={update_idx}, score={best_score:.6f}')
        final_superpose = export_superposition_snapshot(
            superpose_manager,
            pline.network,
            args,
            tag='final',
            step_idx=step_idx,
            update_idx=update_idx,
            score=score,
            best_score=best_score,
            source_checkpoint=final_path,
        )
        append_metric_row(metrics_csv, {
            'event': 'final_eval',
            'timestamp': timestamp_now(),
            'step_idx': step_idx,
            'update_idx': update_idx,
            'score': '' if score is None else score,
            'best_score': '' if best_score is None else best_score,
            'checkpoint': final_path,
            'best_checkpoint': best_path if best_source_path else '',
            'best_source_checkpoint': best_source_path,
            'eval_time_sec': final_eval_time,
            'save_time_sec': final_save_time,
            'elapsed_sec': time.time() - t_start,
            **superpose_metric_fields(args, final_superpose),
        })

    last_path = os.path.join(model_dir, 'last.checkpoint')
    save_t0 = time.time()
    save_checkpoint(last_path, pline.network, optimizer, step_idx, update_idx, best_score, args, 'last')
    last_superpose = export_superposition_snapshot(
        superpose_manager,
        pline.network,
        args,
        tag='last',
        step_idx=step_idx,
        update_idx=update_idx,
        score=None,
        best_score=best_score,
        source_checkpoint=last_path,
    )
    last_save_time = time.time() - save_t0
    append_metric_row(metrics_csv, {
        'event': 'last_saved',
        'timestamp': timestamp_now(),
        'step_idx': step_idx,
        'update_idx': update_idx,
        'best_score': '' if best_score is None else best_score,
        'checkpoint': last_path,
        'best_checkpoint': best_path if best_source_path else '',
        'best_source_checkpoint': best_source_path,
        'save_time_sec': last_save_time,
        'elapsed_sec': time.time() - t_start,
        **superpose_metric_fields(args, last_superpose),
    })
    print(f'* Online adaptation finished. steps={step_idx}, updates={update_idx}, best_score={best_score}')
    print(f'* Output path = {pline.path_log}')

    close_writers(pline)
    os._exit(0)


if __name__ == '__main__':
    main()
