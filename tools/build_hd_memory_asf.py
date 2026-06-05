import argparse
import csv
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hd_utils_asf import (
    clear_batch,
    close_writers,
    extract_hd_features_by_labels,
    load_model_checkpoint,
    make_runtime_config,
    require_hd_head,
    timestamp_now,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Build ASF HD memory from a labeled train split.')
    parser.add_argument('--config', required=True, help='HD config. Train split is used to build memory.')
    parser.add_argument('--checkpoint', required=True, help='Existing CNN/ASF checkpoint used for features and box head.')
    parser.add_argument('--output_root', default='./results')
    parser.add_argument('--run_name', default='hdmem_asf')
    parser.add_argument('--run_stamp', default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_batches', type=int, default=-1)
    parser.add_argument('--max_pos_per_class', type=int, default=4096)
    parser.add_argument('--max_total_pos', type=int, default=0)
    parser.add_argument('--max_neg_per_batch', type=int, default=8192)
    parser.add_argument('--max_neg_ratio', type=float, default=3.0)
    parser.add_argument('--conf_thr', type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_config, _runtime_cfg = make_runtime_config(
        args.config,
        args.output_root,
        args.run_name,
        args.batch_size,
        args.num_workers,
        conf_thr=args.conf_thr,
        run_stamp=args.run_stamp,
        save_model=False,
    )

    from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

    pline = PipelineDetection_v1_0(path_cfg=runtime_config, mode='train')
    missing, unexpected = load_model_checkpoint(pline.network, args.checkpoint)
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
    for batch_idx, batch in enumerate(tqdm(loader, desc='* Build HD memory')):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            clear_batch(batch)
            break
        feat_pos, labels_pos = extract_hd_features_by_labels(
            pline.network,
            batch,
            max_pos_per_class=args.max_pos_per_class,
            max_total_pos=args.max_total_pos,
            include_negative=bool(head.hd_core.cfg.use_background),
            max_neg_per_batch=args.max_neg_per_batch,
            max_neg_ratio=args.max_neg_ratio,
        )
        if labels_pos.numel() > 0:
            head.hd_core.build_update(feat_pos, labels_pos)
            pos_mask = labels_pos > 0
            if pos_mask.any():
                shifted = labels_pos[pos_mask].detach().cpu().long() - 1
                class_counts.index_add_(0, shifted, torch.ones_like(shifted))
                total_pos += int(pos_mask.sum().item())
            total_bg += int((labels_pos == 0).sum().item())
        clear_batch(batch)

    head.hd_core.memory.normalize_()
    build_time = time.time() - t0

    hd_dir = os.path.join(pline.path_log, 'hd_memory')
    os.makedirs(hd_dir, exist_ok=True)
    mem_path = os.path.join(hd_dir, 'hd_memory.pth')
    meta = {
        'source_checkpoint': args.checkpoint,
        'config': args.config,
        'runtime_config': runtime_config,
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

    summary_path = os.path.join(pline.path_log, 'build_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(meta.keys()) + ['memory_path'])
        writer.writeheader()
        row = dict(meta)
        row['memory_path'] = mem_path
        writer.writerow(row)

    print(f'* HD memory saved: {mem_path}')
    print(f'* Total positive anchors: {total_pos}')
    print(f'* Total background anchors: {total_bg}')
    print(f'* Class counts: {meta["class_counts"]}')
    print(f'* Build time: {build_time:.2f}s')
    close_writers(pline)
    os._exit(0)


if __name__ == '__main__':
    main()
