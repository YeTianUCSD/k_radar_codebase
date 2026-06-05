import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def alpha_tag(alpha):
    text = f'{float(alpha):g}'
    return text.replace('-', 'm').replace('.', 'p')


def now_stamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse_best_summary(path_run):
    path = path_run / 'best_summary.txt'
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        out[key.strip()] = value.strip()
    return out


def write_sweep_row(path_csv, row):
    fields = [
        'alpha', 'status', 'returncode', 'run_dir', 'log_file', 'start_time',
        'end_time', 'elapsed_sec', 'best_update', 'best_score',
        'best_checkpoint', 'source_checkpoint',
    ]
    is_new = not path_csv.exists()
    with path_csv.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in fields})


def main():
    parser = argparse.ArgumentParser(description='Run ASF HD online alpha sweep sequentially.')
    parser.add_argument('--config', default='./configs/ASF_v2_0_seq58_adapt_hd.yml')
    parser.add_argument('--init_model', default='/home/code/hyperradar/k_radar_codebase/results/offline_baseline/train_seq1_20epoch_test_seq1/exp_260514_101619_train_seq1_20epoch_test_seq1/models/model_16.pt')
    parser.add_argument('--hd_memory', default='/home/code/hyperradar/k_radar_codebase/results/hd_experiments/hdmem_trainseq1_from_cnn_seq1_model16_hdonly_source_exp_260515_200922/hd_memory/hd_memory.pth')
    parser.add_argument('--output_root', default='/home/code/hyperradar/k_radar_codebase/results/hd_experiments')
    parser.add_argument('--run_prefix', default='online_seq58_hdonly_adaptive')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval_every_updates', type=int, default=50)
    parser.add_argument('--save_every_updates', type=int, default=50)
    parser.add_argument('--conf_thr', type=float, default=0.3)
    parser.add_argument('--cuda_visible_devices', default='0')
    parser.add_argument('--keep_going', action='store_true')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_stamp = now_stamp()
    sweep_csv = output_root / f'hd_alpha_sweep_{sweep_stamp}.csv'

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.cuda_visible_devices)
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['NUMEXPR_NUM_THREADS'] = '1'
    env.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
    env.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')

    for alpha in args.alphas:
        tag = alpha_tag(alpha)
        run_name = f'{args.run_prefix}_alpha_{tag}_from_hdmem_trainseq1_cnn_seq1_model16'
        run_stamp = now_stamp()
        run_dir = output_root / f'{run_name}_exp_{run_stamp}'
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / 'nohup.log'

        cmd = [
            sys.executable, '-u', str(ROOT / 'tools' / 'online_adapt_asf_hd.py'),
            '--config', args.config,
            '--init_model', args.init_model,
            '--hd_memory', args.hd_memory,
            '--output_root', str(output_root),
            '--run_name', run_name,
            '--run_stamp', run_stamp,
            '--batch_size', str(args.batch_size),
            '--num_workers', str(args.num_workers),
            '--alpha', str(alpha),
            '--eval_every_updates', str(args.eval_every_updates),
            '--save_every_updates', str(args.save_every_updates),
            '--conf_thr', str(args.conf_thr),
            '--best_metric_cls', 'auto',
            '--best_metric_kind', '3d',
            '--best_metric_ious', '0.3', '0.5',
            '--best_metric_conf', '0.3',
        ]

        start = datetime.now()
        t0 = time.time()
        status = 'ok'
        returncode = 0
        with log_file.open('w') as log:
            log.write(f'[START] alpha={alpha} at {start:%Y-%m-%d %H:%M:%S}\n')
            log.write(f'[RUN_DIR] {run_dir}\n')
            log.write('[CMD] ' + ' '.join(cmd) + '\n\n')
            log.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            returncode = int(proc.returncode)
            if returncode != 0:
                status = 'failed'
            end = datetime.now()
            log.write(f'\n[DONE] alpha={alpha} status={status} returncode={returncode} at {end:%Y-%m-%d %H:%M:%S}\n')

        elapsed = time.time() - t0
        best = parse_best_summary(run_dir)
        write_sweep_row(sweep_csv, {
            'alpha': alpha,
            'status': status,
            'returncode': returncode,
            'run_dir': str(run_dir),
            'log_file': str(log_file),
            'start_time': start.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_sec': f'{elapsed:.3f}',
            'best_update': best.get('best_update', ''),
            'best_score': best.get('score', ''),
            'best_checkpoint': best.get('best_checkpoint', ''),
            'source_checkpoint': best.get('source_checkpoint', ''),
        })

        print(f'* alpha={alpha} status={status} best={best.get("score", "NA")} run={run_dir}', flush=True)
        if returncode != 0 and not args.keep_going:
            print(f'* Stopping sweep after alpha={alpha}; use --keep_going to continue after failures.', flush=True)
            return returncode

    print(f'* Sweep summary saved: {sweep_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
