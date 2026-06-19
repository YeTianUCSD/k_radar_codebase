import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_name_path_item(item):
    if '=' not in item:
        path = item
        name = os.path.splitext(os.path.basename(path))[0]
        return name, path
    name, path = item.split('=', 1)
    return name, path


def parse_name_value_item(item):
    if '=' not in item:
        raise ValueError(f'Expected name=value format, got: {item}')
    name, value = item.split('=', 1)
    return name, value


def collect_checkpoints(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    paths = [os.path.abspath(path) for path in paths if os.path.isfile(path)]
    return sorted(set(paths))


def make_runtime_config(path_config, output_root, run_name, best_metric, scene_context=None):
    with open(path_config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['GENERAL']['NAME'] = run_name
    cfg['GENERAL']['LOGGING']['PATH_LOGGING'] = output_root
    cfg['GENERAL']['LOGGING']['IS_SAVE_MODEL'] = False
    val_cfg = cfg.setdefault('VAL', {})
    val_cfg['IS_VALIDATE'] = True
    cfg['GENERAL']['LOGGING']['BEST_METRIC'] = best_metric

    model_cfg = cfg.setdefault('MODEL', {})
    superposition_cfg = model_cfg.setdefault('SUPERPOSITION', {})
    if scene_context is not None:
        superposition_cfg['ENABLED'] = True
        superposition_cfg['ACTIVE_SCENE'] = str(scene_context)
        superposition_cfg.setdefault('BASE_SCENE', str(scene_context))

    tmp = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yml',
        prefix='kradar_scene_eval_runtime_',
        delete=False,
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return tmp.name


def timestamp_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def summarize_metrics(pline, eval_rows, eval_name, config_path, checkpoint_path, scene_context, timing=None):
    score = pline.pick_best_metric_score(eval_rows)
    metric_cfg = pline.best_metric_cfg
    selected = pline.select_best_metric_rows(eval_rows)
    summary = {
        'eval_name': eval_name,
        'scene_context': scene_context,
        'config': config_path,
        'checkpoint': checkpoint_path,
        'checkpoint_name': os.path.basename(checkpoint_path),
        'score': '' if score is None else score,
        'score_kind': metric_cfg['kind'],
        'selected_metric_count': len(selected),
        'selected_metrics': ';'.join(
            f"{row['cls']}/{row['iou']}/{row['3d']:.6f}" for row in selected
        ),
        'selected_score_values': ';'.join(
            f"{float(row[metric_cfg['kind']]):.6f}" for row in selected
        ),
        'log_dir': pline.path_log,
    }
    if timing:
        summary.update(timing)
    return summary



def close_writers(pline):
    for writer_name in ('log_train_iter', 'log_train_epoch', 'log_test'):
        writer = getattr(pline, writer_name, None)
        if writer is not None:
            writer.close()

def write_csv(path_csv, rows):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    fieldnames = [
        'eval_name',
        'scene_context',
        'config',
        'checkpoint',
        'checkpoint_name',
        'score',
        'score_kind',
        'selected_metric_count',
        'selected_metrics',
        'selected_score_values',
        'timestamp',
        'setup_time_sec',
        'load_model_time_sec',
        'eval_time_sec',
        'total_time_sec',
        'log_dir',
    ]
    with open(path_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate shared scene-conditioned ASF checkpoints on one or more scene contexts.'
    )
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Checkpoint paths or glob patterns')
    parser.add_argument('--eval', type=str, nargs='+', required=True,
                        help='Eval configs, format name=path')
    parser.add_argument('--scene', type=str, nargs='+', required=True,
                        help='Scene contexts, format eval_name=scene_name')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory for per-checkpoint eval logs')
    parser.add_argument('--summary_csv', type=str, required=True,
                        help='Path for consolidated CSV summary')
    parser.add_argument('--conf_thr', type=float, nargs='+', default=[0.3],
                        help='Confidence thresholds for full eval')
    parser.add_argument('--best_metric_kind', choices=['bev', '3d'], default='3d')
    parser.add_argument('--best_metric_ious', type=float, nargs='+', default=[0.3, 0.5])
    parser.add_argument('--best_metric_conf', type=float, default=0.3)
    parser.add_argument('--best_metric_cls', type=str, default='auto')
    args = parser.parse_args()

    checkpoints = collect_checkpoints(args.checkpoints)
    if len(checkpoints) == 0:
        raise FileNotFoundError('No checkpoints found')

    eval_items = dict(parse_name_path_item(item) for item in args.eval)
    scene_items = dict(parse_name_value_item(item) for item in args.scene)
    missing_scenes = sorted(set(eval_items.keys()) - set(scene_items.keys()))
    if missing_scenes:
        raise ValueError(f'Missing --scene entries for eval names: {missing_scenes}')

    os.makedirs(args.output_root, exist_ok=True)
    best_metric = {
        'CLS': args.best_metric_cls,
        'KIND': args.best_metric_kind,
        'IOUS': args.best_metric_ious,
        'CONF_THR': args.best_metric_conf,
        'REDUCE': 'mean',
        'ONLY_CLASSES_WITH_GT': True,
    }

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    summary_rows = []
    for eval_name, config_path in eval_items.items():
        scene_context = scene_items[eval_name]
        for checkpoint_path in checkpoints:
            total_time_start = time.time()
            ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            run_name = f'eval_{eval_name}_{scene_context}_{ckpt_name}'
            runtime_config = make_runtime_config(
                config_path,
                args.output_root,
                run_name,
                best_metric,
                scene_context=scene_context,
            )

            print(f'* Eval {eval_name} [{scene_context}]: {checkpoint_path}')
            setup_time_start = time.time()
            pline = PipelineDetection_v1_0(runtime_config, mode='test')
            if not hasattr(pline, 'val_keyword'):
                pline.set_validate()
            setup_time_sec = time.time() - setup_time_start

            load_time_start = time.time()
            pline.load_dict_model(checkpoint_path)
            pline.network.default_scene_context = str(scene_context)
            load_model_time_sec = time.time() - load_time_start
            pline.network.eval()
            shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))

            eval_time_start = time.time()
            eval_rows = pline.validate_kitti(
                epoch=None,
                list_conf_thr=args.conf_thr,
                is_subset=False,
            )
            eval_time_sec = time.time() - eval_time_start
            total_time_sec = time.time() - total_time_start

            summary_rows.append(
                summarize_metrics(
                    pline,
                    eval_rows,
                    eval_name,
                    config_path,
                    checkpoint_path,
                    scene_context,
                    timing={
                        'timestamp': timestamp_now(),
                        'setup_time_sec': setup_time_sec,
                        'load_model_time_sec': load_model_time_sec,
                        'eval_time_sec': eval_time_sec,
                        'total_time_sec': total_time_sec,
                    },
                )
            )
            write_csv(args.summary_csv, summary_rows)
            close_writers(pline)

    os._exit(0)


if __name__ == '__main__':
    main()
