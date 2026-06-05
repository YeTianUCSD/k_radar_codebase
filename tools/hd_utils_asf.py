import os
import shutil
import tempfile
from datetime import datetime

import torch
import yaml


def timestamp_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def make_runtime_config(path_config, output_root, run_name, batch_size, num_workers, conf_thr=0.3,
                        run_stamp=None, save_model=False, best_metric=None):
    with open(path_config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['GENERAL']['NAME'] = run_name
    cfg['GENERAL']['LOGGING']['PATH_LOGGING'] = output_root
    cfg['GENERAL']['LOGGING']['IS_SAVE_MODEL'] = bool(save_model)
    cfg['GENERAL']['LOGGING']['ALLOW_EXISTING_DIR'] = True
    if run_stamp is not None:
        cfg['GENERAL']['LOGGING']['RUN_STAMP'] = run_stamp
    cfg['OPTIMIZER']['BATCH_SIZE'] = int(batch_size)
    cfg['OPTIMIZER']['NUM_WORKERS'] = int(num_workers)
    cfg['VAL']['IS_VALIDATE'] = True
    cfg['VAL']['IS_CONSIDER_VAL_SUBSET'] = False
    cfg['VAL']['LIST_VAL_CONF_THR'] = [float(conf_thr)]
    if best_metric is not None:
        cfg['GENERAL']['LOGGING']['BEST_METRIC'] = best_metric

    tmp = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yml',
        prefix='kradar_hd_runtime_',
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
    return network.load_state_dict(state_dict, strict=False)


def require_hd_head(network):
    head = getattr(network, 'head', None)
    if head is None or not hasattr(head, 'hd_core'):
        raise RuntimeError('The network head does not expose hd_core. Use an AnchorHeadSingleHD config.')
    return head


def save_checkpoint(path, network, step_idx, update_idx, best_score=None, tag='checkpoint', extra=None):
    torch.save(network.state_dict(), path)
    meta = {
        'step_idx': int(step_idx),
        'update_idx': int(update_idx),
        'best_score': None if best_score is None else float(best_score),
        'tag': tag,
    }
    if extra:
        meta.update(extra)
    torch.save(meta, str(path) + '.state')


def copy_checkpoint(src_path, dst_path):
    shutil.copy2(src_path, dst_path)
    src_state = str(src_path) + '.state'
    dst_state = str(dst_path) + '.state'
    if os.path.exists(src_state):
        shutil.copy2(src_state, dst_state)


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


def set_forward_for_hd_update(network):
    network.eval()
    head = require_hd_head(network)
    head.train()
    return head


@torch.no_grad()
def extract_positive_hd_features(network, batch, max_pos_per_class=0, max_total_pos=0):
    head = set_forward_for_hd_update(network)
    _ = network(batch)
    feat_map = head.forward_ret_dict.get('hd_features', None)
    labels = head.forward_ret_dict.get('box_cls_labels', None)
    if feat_map is None or labels is None:
        raise RuntimeError('HD update requires head.forward_ret_dict hd_features and box_cls_labels.')
    return head.get_positive_hd_features(
        feat_map,
        labels,
        max_pos_per_class=max_pos_per_class,
        max_total_pos=max_total_pos,
    )



@torch.no_grad()
def extract_hd_features_by_labels(network, batch, max_pos_per_class=0, max_total_pos=0,
                                  include_negative=False, max_neg_per_batch=0, max_neg_ratio=0.0):
    head = set_forward_for_hd_update(network)
    _ = network(batch)
    feat_map = head.forward_ret_dict.get('hd_features', None)
    labels = head.forward_ret_dict.get('box_cls_labels', None)
    if feat_map is None or labels is None:
        raise RuntimeError('HD update requires head.forward_ret_dict hd_features and box_cls_labels.')
    return head.get_hd_features_by_labels(
        feat_map,
        labels,
        max_pos_per_class=max_pos_per_class,
        max_total_pos=max_total_pos,
        include_negative=include_negative,
        max_neg_per_batch=max_neg_per_batch,
        max_neg_ratio=max_neg_ratio,
    )

def run_eval(pline, epoch, conf_thr):
    pline.network.eval()
    rows = pline.validate_kitti(epoch=epoch, list_conf_thr=[float(conf_thr)], is_subset=False)
    return pline.pick_best_metric_score(rows), rows


def close_writers(pline):
    for writer_name in ('log_train_iter', 'log_train_epoch', 'log_test'):
        writer = getattr(pline, writer_name, None)
        if writer is not None:
            writer.close()
