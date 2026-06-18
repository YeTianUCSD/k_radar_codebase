import argparse
import csv
import glob
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_eval_item(item):
    if '=' not in item:
        path = item
        name = os.path.splitext(os.path.basename(path))[0]
        return name, path
    name, path = item.split('=', 1)
    return name, path


def checkpoint_sort_key(path):
    name = os.path.basename(path)
    match = re.search(r'model_(\d+)\.pt$', name)
    if match:
        return (0, int(match.group(1)), name)
    if name == 'best.checkpoint':
        return (1, 0, name)
    if name == 'latest.checkpoint':
        return (2, 0, name)
    return (3, 0, name)


def checkpoint_epoch(path):
    name = os.path.basename(path)
    match = re.search(r'model_(\d+)\.pt$', name)
    if match:
        return int(match.group(1))
    return None


def collect_checkpoints(args):
    if args.checkpoints:
        paths = []
        for pattern in args.checkpoints:
            matches = glob.glob(pattern)
            paths.extend(matches if matches else [pattern])
    else:
        patterns = [os.path.join(args.checkpoint_dir, 'model_*.pt')]
        if args.include_best_latest:
            patterns.extend([
                os.path.join(args.checkpoint_dir, 'best.checkpoint'),
                os.path.join(args.checkpoint_dir, 'latest.checkpoint'),
            ])
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(pattern))

    paths = [os.path.abspath(path) for path in paths if os.path.isfile(path)]
    return sorted(set(paths), key=checkpoint_sort_key)


def make_runtime_config(path_config, output_root, run_name, best_metric):
    with open(path_config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['GENERAL']['NAME'] = run_name
    cfg['GENERAL']['LOGGING']['PATH_LOGGING'] = output_root
    cfg['GENERAL']['LOGGING']['IS_SAVE_MODEL'] = False
    cfg['VAL']['IS_VALIDATE'] = True
    cfg['GENERAL']['LOGGING']['BEST_METRIC'] = best_metric

    tmp = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yml',
        prefix='kradar_eval_runtime_',
        delete=False,
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    return tmp.name


def timestamp_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def summarize_metrics(
    pline,
    eval_rows,
    eval_name,
    config_path,
    checkpoint_path,
    timing=None,
):
    score = pline.pick_best_metric_score(eval_rows)
    metric_cfg = pline.best_metric_cfg
    selected = pline.select_best_metric_rows(eval_rows)
    metric_values = {}
    for row in eval_rows:
        cls_name = str(row['cls']).lower()
        iou_key = str(row['iou']).replace('.', '_')
        prefix = f"{cls_name}_iou_{iou_key}"
        metric_values[f'{prefix}_has_gt'] = int(bool(row.get('has_gt', True)))
        metric_values[f'{prefix}_bev'] = row['bev']
        metric_values[f'{prefix}_3d'] = row['3d']

    summary = {
        'eval_name': eval_name,
        'config': config_path,
        'checkpoint': checkpoint_path,
        'checkpoint_name': os.path.basename(checkpoint_path),
        'checkpoint_epoch': checkpoint_epoch(checkpoint_path),
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
    summary.update(metric_values)
    return summary


def write_csv(path_csv, rows):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    base_fieldnames = [
        'eval_name',
        'config',
        'checkpoint',
        'checkpoint_name',
        'checkpoint_epoch',
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
    metric_fieldnames = sorted({
        key for row in rows for key in row.keys()
        if key not in base_fieldnames
    })
    fieldnames = base_fieldnames + metric_fieldnames
    with open(path_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(path_csv, rows):
    by_ckpt = {}
    for row in rows:
        by_ckpt.setdefault(row['checkpoint_name'], {})[row['eval_name']] = row

    eval_names = sorted({row['eval_name'] for row in rows})
    if len(eval_names) < 2:
        return

    path_compare = os.path.splitext(path_csv)[0] + '_compare.csv'
    fieldnames = ['checkpoint_name', 'checkpoint_epoch']
    fieldnames.extend([f'{name}_score' for name in eval_names])
    fieldnames.extend([f'{name}_selected_metrics' for name in eval_names])
    if len(eval_names) == 2:
        fieldnames.append(f'{eval_names[1]}_minus_{eval_names[0]}')

    compare_rows = []
    for ckpt_name, eval_dict in sorted(by_ckpt.items(), key=lambda x: checkpoint_sort_key(x[0])):
        row = {
            'checkpoint_name': ckpt_name,
            'checkpoint_epoch': next(iter(eval_dict.values()))['checkpoint_epoch'],
        }
        scores = []
        for name in eval_names:
            eval_row = eval_dict.get(name, {})
            score = eval_row.get('score', '')
            row[f'{name}_score'] = score
            row[f'{name}_selected_metrics'] = eval_row.get('selected_metrics', '')
            scores.append(score)
        if len(eval_names) == 2 and scores[0] != '' and scores[1] != '':
            row[f'{eval_names[1]}_minus_{eval_names[0]}'] = float(scores[1]) - float(scores[0])
        compare_rows.append(row)

    with open(path_compare, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compare_rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate many checkpoints on one or more K-Radar configs.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing model_*.pt checkpoints')
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None,
                        help='Explicit checkpoint paths or glob patterns')
    parser.add_argument('--eval', type=str, nargs='+', required=True,
                        help='Eval configs, format name=path. Example: seq1=./configs/ASF_v2_0_seq1.yml')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory for per-checkpoint eval logs')
    parser.add_argument('--summary_csv', type=str, required=True,
                        help='Path for consolidated CSV summary')
    parser.add_argument('--conf_thr', type=float, nargs='+', default=[0.3],
                        help='Confidence thresholds for full eval')
    parser.add_argument('--best_metric_kind', choices=['bev', '3d'], default='3d')
    parser.add_argument('--best_metric_ious', type=float, nargs='+', default=[0.3, 0.5])
    parser.add_argument('--best_metric_conf', type=float, default=0.3)
    parser.add_argument('--best_metric_cls', type=str, default='auto',
                        help='auto means average care classes that have GT')
    parser.add_argument('--include_best_latest', action='store_true',
                        help='Also evaluate best.checkpoint/latest.checkpoint from checkpoint_dir')
    parser.add_argument('--print_memory', action='store_true')
    parser.add_argument('--hd_memory', type=str, default=None,
                        help='Optional HD memory path loaded after each model checkpoint. Use with AnchorHeadSingleHD configs.')
    args = parser.parse_args()

    if args.checkpoints is None and args.checkpoint_dir is None:
        raise ValueError('Provide --checkpoint_dir or --checkpoints')

    checkpoints = collect_checkpoints(args)
    if len(checkpoints) == 0:
        raise FileNotFoundError('No checkpoints found')

    eval_items = [parse_eval_item(item) for item in args.eval]
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
    for eval_name, config_path in eval_items:
        for checkpoint_path in checkpoints:
            total_time_start = time.time()
            ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            run_name = f'eval_{eval_name}_{ckpt_name}'
            runtime_config = make_runtime_config(
                config_path,
                args.output_root,
                run_name,
                best_metric,
            )

            print(f'* Eval {eval_name}: {checkpoint_path}')
            setup_time_start = time.time()
            pline = PipelineDetection_v1_0(runtime_config, mode='test')
            if not hasattr(pline, 'val_keyword'):
                pline.set_validate()
            setup_time_sec = time.time() - setup_time_start

            load_time_start = time.time()
            pline.load_dict_model(checkpoint_path)
            if args.hd_memory is not None:
                head = getattr(pline.network, 'head', None)
                if head is None or not hasattr(head, 'hd_core'):
                    raise RuntimeError('--hd_memory requires a network head with hd_core. Use an HD config.')
                head.hd_core.load_memory(args.hd_memory, map_location='cpu')
            load_model_time_sec = time.time() - load_time_start
            pline.network.eval()
            shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))

            eval_time_start = time.time()
            eval_rows = pline.validate_kitti(
                epoch=checkpoint_epoch(checkpoint_path),
                list_conf_thr=args.conf_thr,
                is_subset=False,
            )
            eval_time_sec = time.time() - eval_time_start
            total_time_sec = time.time() - total_time_start
            print(
                f'* Timing {eval_name}/{ckpt_name}: '
                f'setup={setup_time_sec:.2f}s, load={load_model_time_sec:.2f}s, '
                f'eval={eval_time_sec:.2f}s, total={total_time_sec:.2f}s'
            )
            summary = summarize_metrics(
                pline,
                eval_rows,
                eval_name,
                config_path,
                checkpoint_path,
                timing={
                    'timestamp': timestamp_now(),
                    'setup_time_sec': setup_time_sec,
                    'load_model_time_sec': load_model_time_sec,
                    'eval_time_sec': eval_time_sec,
                    'total_time_sec': total_time_sec,
                },
            )
            summary_rows.append(summary)
            print(
                f"* Score {eval_name}/{ckpt_name}: {summary['score']} "
                f"(selected {summary['selected_metric_count']} metrics)"
            )
            print(f"* Selected metrics {eval_name}/{ckpt_name}: {summary['selected_metrics']}")
            print(f"* Selected score values {eval_name}/{ckpt_name}: {summary['selected_score_values']}")
            write_csv(args.summary_csv, summary_rows)

            for writer_name in ('log_train_iter', 'log_train_epoch', 'log_test'):
                writer = getattr(pline, writer_name, None)
                if writer is not None:
                    writer.close()

    write_csv(args.summary_csv, summary_rows)
    write_comparison(args.summary_csv, summary_rows)
    print(f'* Summary saved: {args.summary_csv}')
    os._exit(0)


if __name__ == '__main__':
    main()
