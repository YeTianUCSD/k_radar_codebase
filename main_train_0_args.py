'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import argparse
import shutil
import tempfile
import yaml


def make_runtime_config(path_config, args):
    with open(path_config, 'r') as f:
        cfg = yaml.safe_load(f)

    is_overridden = False

    output_root = args.output_root or args.output_dir
    run_name = args.run_name or args.exp_name
    max_epoch = args.epochs if args.epochs is not None else args.max_epoch
    full_eval_every = args.full_eval_every if args.full_eval_every is not None else args.val_per_epoch_full

    if output_root is not None:
        cfg['GENERAL']['LOGGING']['PATH_LOGGING'] = output_root
        is_overridden = True
    if run_name is not None:
        cfg['GENERAL']['NAME'] = run_name
        is_overridden = True
    if max_epoch is not None:
        cfg['OPTIMIZER']['MAX_EPOCH'] = max_epoch
        is_overridden = True
    if args.batch_size is not None:
        cfg['OPTIMIZER']['BATCH_SIZE'] = args.batch_size
        is_overridden = True
    if args.num_workers is not None:
        cfg['OPTIMIZER']['NUM_WORKERS'] = args.num_workers
        is_overridden = True
    if args.num_subset is not None:
        cfg['VAL']['NUM_SUBSET'] = args.num_subset
        is_overridden = True
    if args.use_val_subset:
        cfg['VAL']['IS_VALIDATE'] = True
        cfg['VAL']['IS_CONSIDER_VAL_SUBSET'] = True
        is_overridden = True
    if full_eval_every is not None:
        cfg['VAL']['IS_VALIDATE'] = True
        cfg['VAL']['VAL_PER_EPOCH_FULL'] = full_eval_every
        is_overridden = True
    if args.val_per_epoch_subset is not None:
        cfg['VAL']['IS_VALIDATE'] = True
        cfg['VAL']['VAL_PER_EPOCH_SUBSET'] = args.val_per_epoch_subset
        is_overridden = True
    save_every_epoch = not args.no_save_every_epoch
    if save_every_epoch:
        cfg['GENERAL']['LOGGING']['IS_SAVE_MODEL'] = True
        cfg['GENERAL']['LOGGING']['INTERVAL_EPOCH_MODEL'] = 1
        cfg['GENERAL']['LOGGING']['INTERVAL_EPOCH_UTIL'] = 1
        is_overridden = True
    if args.interval_epoch_model is not None:
        cfg['GENERAL']['LOGGING']['IS_SAVE_MODEL'] = True
        cfg['GENERAL']['LOGGING']['INTERVAL_EPOCH_MODEL'] = args.interval_epoch_model
        is_overridden = True
    if args.interval_epoch_util is not None:
        cfg['GENERAL']['LOGGING']['IS_SAVE_MODEL'] = True
        cfg['GENERAL']['LOGGING']['INTERVAL_EPOCH_UTIL'] = args.interval_epoch_util
        is_overridden = True
    if any(v is not None for v in (
        args.best_metric_cls,
        args.best_metric_kind,
        args.best_metric_iou,
        args.best_metric_ious,
        args.best_metric_conf,
    )):
        metric_ious = args.best_metric_ious
        if metric_ious is None:
            metric_ious = [0.3, 0.5] if args.best_metric_iou is None else [args.best_metric_iou]
        cfg['GENERAL']['LOGGING']['BEST_METRIC'] = {
            'CLS': args.best_metric_cls or 'auto',
            'KIND': args.best_metric_kind or '3d',
            'IOUS': metric_ious,
            'CONF_THR': 0.3 if args.best_metric_conf is None else args.best_metric_conf,
            'REDUCE': 'mean',
            'ONLY_CLASSES_WITH_GT': True,
        }
        is_overridden = True

    if not is_overridden:
        return path_config, None

    tmp = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yml',
        prefix='kradar_runtime_',
        delete=False,
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return tmp.name, cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection Pipeline Training')
    parser.add_argument('--config', type=str, default='./configs/ASF_v2_0_final.yml',
                        help='Path to config file')
    parser.add_argument('--output_root', type=str, default=None,
                        help='Root directory for all runs. Final dir is <run_name>_exp_YYMMDD_HHMMSS under this root')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Semantic run name. Final dir is <run_name>_exp_YYMMDD_HHMMSS')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs. Alias of --max_epoch')
    parser.add_argument('--full_eval_every', type=int, default=None,
                        help='Run full validation every N epochs. Alias of --val_per_epoch_full')
    parser.add_argument('--no_save_every_epoch', action='store_true',
                        help='Do not force model/util checkpoints to be saved every epoch')
    parser.add_argument('--best_metric_cls', type=str, default=None,
                        help='Class keyword used to select best checkpoint. Use auto to average care classes with GT, default: auto')
    parser.add_argument('--best_metric_kind', type=str, choices=['bev', '3d'], default=None,
                        help='Metric kind used to select best checkpoint, default: 3d')
    parser.add_argument('--best_metric_iou', type=float, default=None,
                        help='Single IoU threshold used to select best checkpoint')
    parser.add_argument('--best_metric_ious', type=float, nargs='+', default=None,
                        help='IoU thresholds averaged to select best checkpoint, default: 0.3 0.5')
    parser.add_argument('--best_metric_conf', type=float, default=None,
                        help='Confidence threshold used to select best checkpoint, default: 0.3')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Deprecated alias of --output_root')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Deprecated alias of --run_name')
    parser.add_argument('--max_epoch', type=int, default=None,
                        help='Deprecated alias of --epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override OPTIMIZER.BATCH_SIZE')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override OPTIMIZER.NUM_WORKERS')
    parser.add_argument('--val_per_epoch_full', type=int, default=None,
                        help='Deprecated alias of --full_eval_every')
    parser.add_argument('--val_per_epoch_subset', type=int, default=None,
                        help='Override VAL.VAL_PER_EPOCH_SUBSET and enable validation')
    parser.add_argument('--num_subset', type=int, default=None,
                        help='Override VAL.NUM_SUBSET')
    parser.add_argument('--use_val_subset', action='store_true',
                        help='Enable subset validation during training')
    parser.add_argument('--interval_epoch_model', type=int, default=None,
                        help='Override LOGGING.INTERVAL_EPOCH_MODEL and enable model saving')
    parser.add_argument('--interval_epoch_util', type=int, default=None,
                        help='Override LOGGING.INTERVAL_EPOCH_UTIL and enable util checkpoint saving')
    parser.add_argument('--skip_final_eval', action='store_true',
                        help='Skip conditional evaluation after the last training epoch')
    args = parser.parse_args()

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    path_runtime_config, runtime_cfg = make_runtime_config(args.config, args)
    if runtime_cfg is not None:
        print(f'* Runtime config generated: {path_runtime_config}')
        print(f"* Override PATH_LOGGING = {runtime_cfg['GENERAL']['LOGGING']['PATH_LOGGING']}")
        print(f"* Override NAME = {runtime_cfg['GENERAL']['NAME']}")
        print(f"* Override MAX_EPOCH = {runtime_cfg['OPTIMIZER']['MAX_EPOCH']}")
        print(f"* Override BATCH_SIZE = {runtime_cfg['OPTIMIZER']['BATCH_SIZE']}")
        print(f"* Override NUM_WORKERS = {runtime_cfg['OPTIMIZER']['NUM_WORKERS']}")
        print(f"* Override IS_CONSIDER_VAL_SUBSET = {runtime_cfg['VAL']['IS_CONSIDER_VAL_SUBSET']}")
        print(f"* Override NUM_SUBSET = {runtime_cfg['VAL']['NUM_SUBSET']}")
        print(f"* Override VAL_PER_EPOCH_FULL = {runtime_cfg['VAL']['VAL_PER_EPOCH_FULL']}")
        print(f"* Override INTERVAL_EPOCH_MODEL = {runtime_cfg['GENERAL']['LOGGING']['INTERVAL_EPOCH_MODEL']}")
        print(f"* Override INTERVAL_EPOCH_UTIL = {runtime_cfg['GENERAL']['LOGGING']['INTERVAL_EPOCH_UTIL']}")
        if 'BEST_METRIC' in runtime_cfg['GENERAL']['LOGGING']:
            print(f"* Override BEST_METRIC = {runtime_cfg['GENERAL']['LOGGING']['BEST_METRIC']}")

    pline = PipelineDetection_v1_0(path_cfg=path_runtime_config, mode='train')

    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))

    pline.train_network()

    if not args.skip_final_eval:
        pline.validate_kitti_conditional(list_conf_thr=[0.3], is_subset=False, is_print_memory=False)

    for writer_name in ('log_train_iter', 'log_train_epoch', 'log_test'):
        writer = getattr(pline, writer_name, None)
        if writer is not None:
            writer.close()

    os._exit(0)
