'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr
'''

import torch
import numpy as np
import open3d as o3d
import os
os.environ.setdefault('NUMBA_CUDA_USE_NVIDIA_BINDING', '1')
import csv
import time
from tqdm import tqdm
import shutil
from torch.utils.data import Subset

# Ingnore numba warning
from numba.core.errors import NumbaWarning
import warnings
import logging
warnings.simplefilter('ignore', category=NumbaWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)

from torch.utils.tensorboard import SummaryWriter

from utils.util_pipeline import *
from utils.util_point_cloud import *
from utils.util_config import cfg, cfg_from_yaml_file

from utils.util_point_cloud import Object3D
import utils.kitti_eval.kitti_common as kitti
from utils.kitti_eval.eval import get_official_eval_result
from utils.kitti_eval.eval_revised import get_official_eval_result_revised

from utils.util_optim import clip_grad_norm_

class PipelineDetection_v1_0():
    def __init__(self, path_cfg=None, mode='train'):
        '''
        * mode in ['train', 'test', 'vis']
        *   'train' denotes both train & test
        *   'test'  denotes mode for inference
        '''
        self.cfg = cfg_from_yaml_file(path_cfg, cfg)
        self.mode = mode
        self.update_cfg_regarding_mode()

        if self.cfg.GENERAL.SEED is not None:
            try:
                set_random_seed(cfg.GENERAL.SEED, cfg.GENERAL.IS_CUDA_SEED, cfg.GENERAL.IS_DETERMINISTIC)
            except:
                print('* Exception error: check cfg.GENERAL for seed')
                set_random_seed(cfg.GENERAL.SEED)
        
        print('* K-Radar dataset is being loaded.')
        self.dataset_train = build_dataset(self, split='train') if self.mode == 'train' else None
        self.dataset_test = build_dataset(self, split='test')
        print('* The dataset is loaded.')
        if mode == 'train': # for setting scheduler
            self.cfg.DATASET.NUM = len(self.dataset_train)
        elif mode in ['test', 'vis']:
            self.cfg.DATASET.NUM = len(self.dataset_test)
        # print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) # check if it is updated

        self.network = build_network(self).cuda()
        self.optimizer = build_optimizer(self, self.network)
        self.scheduler = build_scheduler(self, self.optimizer)
        self.epoch_start = 0

        # Logging
        if self.cfg.GENERAL.LOGGING.IS_LOGGING:
            self.set_logging(path_cfg)

        # Validation
        if self.cfg.VAL.IS_VALIDATE:
            self.set_validate()
        else:
            self.is_validate = False
        
        if self.cfg.GENERAL.RESUME.IS_RESUME:
            self.resume_network()

        self.cfg_dataset_ver2 = self.cfg.get('cfg_dataset_ver2', False)
        self.get_loss_from = self.cfg.get('get_loss_from', 'head')
        self.optim_fastai = True \
            if self.cfg.OPTIMIZER.NAME in ['adam_onecycle', 'adam_cosineanneal'] else False
        self.grad_norm_clip = self.cfg.OPTIMIZER.get('GRAD_NORM_CLIP', -1)

        # Vis
        self.set_vis()
        
        # self.show_pline_description()
        
        # Thanks to Felix Fent (in TUM) and Miao Zhang (in Bosch Research)
        # Fixed mixed interpolation (issue #28) and z_center (issue #36) in evaluation
        self.is_validation_updated = self.cfg.get('is_validation_updated', False)

        ### Distil ###
        cfg_distil = self.cfg.get('DISTIL', None)
        if cfg_distil is not None:
            self.distil = True
            self.infer_head_of_distil_model = cfg_distil.get('INFER_HEAD', False)
            import yaml
            from easydict import EasyDict
            with open(cfg_distil.CFG, 'r') as f:
                new_config = yaml.safe_load(f)
            from models.skeletons import build_skeleton
            distil_model = build_skeleton(EasyDict(new_config))
            
            if not self.infer_head_of_distil_model:
                if hasattr(distil_model, 'head'):
                    import torch.nn as nn
                    distil_model.head = nn.Identity()
            distil_model.load_state_dict(torch.load(cfg_distil.PTH), strict=False)
            self.distil_model = distil_model.cuda().eval()
            # inactivate self.head in distilation model
            print('* The model for distilation is loaded.')
        else:
            self.distil = False
        ### Distil ###

    def update_cfg_regarding_mode(self):
        '''
        * You don't have to update values in cfg changed in dataset
        * They are related in pointer
        * e.g., check print(self.cfg.DATASET.CLASS_INFO.NUM_CLS) after dataset initialization
        '''
        if self.mode == 'train':
            pass
        elif self.mode == 'test':
            self.cfg.OPTIMIZER.NUM_WORKERS = 0
        elif self.mode == 'vis':
            self.cfg.OPTIMIZER.NUM_WORKERS = 0
            self.cfg.GET_ITEM = {
                'rdr_sparse_cube'   : True,
                'rdr_tesseract'     : False,
                'rdr_cube'          : True,
                'rdr_cube_doppler'  : False,
                'ldr_pc_64'         : True,
                'cam_front_img'     : True,
            }
        else:
            print('* Exception error (Pipeline): check modify_cfg')
        return

    def set_validate(self):
        self.is_validate = True
        self.is_consider_subset = self.cfg.VAL.IS_CONSIDER_VAL_SUBSET
        self.val_per_epoch_subset = self.cfg.VAL.VAL_PER_EPOCH_SUBSET
        self.val_num_subset = self.cfg.VAL.NUM_SUBSET
        self.val_per_epoch_full = self.cfg.VAL.VAL_PER_EPOCH_FULL

        self.val_keyword = self.cfg.VAL.CLASS_VAL_KEYWORD # for kitti_eval
        list_val_keyword_keys = list(self.val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
        self.list_val_care_idx = []

        # index matching with kitti_eval
        for cls_name in self.cfg.VAL.LIST_CARE_VAL:
            idx_val_cls = list_val_keyword_keys.index(cls_name)
            self.list_val_care_idx.append(idx_val_cls)
        # print(self.list_val_care_idx)

        ### Consider output of network and dataset ###
        if self.cfg.VAL.REGARDING == 'anchor':
            self.val_regarding = 0 # anchor
            self.list_val_conf_thr = self.cfg.VAL.LIST_VAL_CONF_THR
        else:
            print('* Exception error: check VAL.REGARDING')
        ### Consider output of network and dataset ###

    def set_vis(self):
        if self.cfg_dataset_ver2:
            pass # TODO
        else:
            self.dict_cls_name_to_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
            self.dict_cls_id_to_name = dict()
            for k, v in self.dict_cls_name_to_id.items():
                if v != -1:
                    self.dict_cls_id_to_name[v] = k
            self.dict_cls_name_to_bgr = self.cfg.VIS.CLASS_BGR
            self.dict_cls_name_to_rgb = self.cfg.VIS.CLASS_RGB
    
    def show_pline_description(self):
        print('* newtork (description start) -------')
        print(self.network)
        print('* newtork (description end) ---------')
        print('* optimizer (description start) -----')
        print(self.optimizer)
        print('* optimizer (description end) -------')
        print(f'* mode = {self.mode}')
        len_data = self.cfg.DATASET.NUM
        print(f'* dataset length = {len_data}')
    
    def set_logging(self, path_cfg, is_print_where=True):
        self.is_logging = True
        str_local_time = self.cfg.GENERAL.LOGGING.get('RUN_STAMP', get_local_time_str())
        str_exp = self.cfg.GENERAL.NAME + '_exp_' + str_local_time
        self.path_log = os.path.join(self.cfg.GENERAL.LOGGING.PATH_LOGGING, str_exp)
        allow_existing_dir = self.cfg.GENERAL.LOGGING.get('ALLOW_EXISTING_DIR', False)
        if is_print_where:
            print(f'* Start logging in {str_exp}')
        if not (os.path.exists(self.path_log)):
            os.makedirs(self.path_log)
        elif allow_existing_dir:
            print(f'* Logging folder already exists and will be reused: {self.path_log}')
        else:
            print('* Exception error (Pipeline): same folder exists, try again')
            exit()

        self.log_train_iter = SummaryWriter(os.path.join(self.path_log, 'train_iter'), comment='iteration')
        self.log_train_epoch = SummaryWriter(os.path.join(self.path_log, 'train_epoch'), comment='epoch')
        self.log_test = SummaryWriter(os.path.join(self.path_log, 'test'), comment='test')
        self.log_iter_start = None

        self.is_save_model = self.cfg.GENERAL.LOGGING.IS_SAVE_MODEL
        try:
            self.interval_epoch_model = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_MODEL
            self.interval_epoch_util = self.cfg.GENERAL.LOGGING.INTERVAL_EPOCH_UTIL
        except:
            self.interval_epoch_model = 1
            self.interval_epoch_util = 5
            print('* Exception error (Pipeline): check LOGGING.INTERVAL_EPOCH_MODEL/UTIL')
        if self.is_save_model:
            os.makedirs(os.path.join(self.path_log, 'models'), exist_ok=allow_existing_dir)
            os.makedirs(os.path.join(self.path_log, 'utils'), exist_ok=allow_existing_dir)
        self.best_score = None
        self.best_epoch = None
        self.best_metric_cfg = self.get_best_metric_cfg()

        # cfg backup (same files, just for identification)
        name_file_origin = path_cfg.split('/')[-1] # original cfg file name
        name_file_cfg = 'config.yml'
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_origin))
        shutil.copy2(path_cfg, os.path.join(self.path_log, name_file_cfg))

        # code backup (TBD)

    def resume_network(self):
        path_exp = self.cfg.GENERAL.RESUME.PATH_EXP
        path_state_dict = os.path.join(path_exp, 'utils')
        epoch = self.cfg.GENERAL.RESUME.START_EP
        list_util_files = [
            x for x in os.listdir(path_state_dict)
            if x.startswith('util_') and x.endswith('.pt')
        ]
        list_epochs = sorted(list(map(lambda x: int(x.split('.')[0].split('_')[1]), list_util_files)))
        epoch = list_epochs[-1] if epoch is None else epoch

        path_state_dict = os.path.join(path_state_dict, f'util_{epoch}.pt')
        print('* Start resume, path_state_dict =  ', path_state_dict)
        state_dict = torch.load(path_state_dict)

        try:
            self.epoch_start = epoch
            self.network.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.log_iter_start = state_dict['idx_log_iter']
            print(f'* Network & Optimizer are loaded / Resume epoch is {epoch} / Start from {self.epoch_start + 1} ...')
        except:
            print('* Exception error (Pipeline): check resume network')
            exit()

        if ('scheduler_state_dict' in state_dict.keys()) and (not (self.scheduler is None)):
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            print('* Scheduler is loaded')
        else:
            print('* Scheduler is started from vanilla')

        ### Copy logging folder ###
        list_copy_dirs = ['train_epoch', 'train_iter', 'test', 'test_kitti']
        if (self.cfg.GENERAL.RESUME.IS_COPY_LOGS) and (self.is_logging):
            for copy_dir in list_copy_dirs:
                shutil.copytree(os.path.join(path_exp, copy_dir), \
                    os.path.join(self.path_log, copy_dir), dirs_exist_ok=True)
        ### Copy logging folder ###

        return

    def get_best_metric_cfg(self):
        cfg_metric = self.cfg.GENERAL.LOGGING.get('BEST_METRIC', dict())
        if 'IOUS' in cfg_metric:
            ious = cfg_metric.get('IOUS')
        else:
            ious = [cfg_metric.get('IOU', 0.3)]
        if not isinstance(ious, list):
            ious = [ious]
        return {
            'cls': str(cfg_metric.get('CLS', 'auto')).lower(),
            'kind': str(cfg_metric.get('KIND', '3d')).lower(),
            'ious': [float(iou) for iou in ious],
            'conf_thr': float(cfg_metric.get('CONF_THR', 0.3)),
            'only_classes_with_gt': bool(cfg_metric.get('ONLY_CLASSES_WITH_GT', True)),
        }

    def train_network(self, is_shuffle=True):
        self.network.train()
        data_loader_train = torch.utils.data.DataLoader(self.dataset_train, \
            batch_size = self.cfg.OPTIMIZER.BATCH_SIZE, shuffle = is_shuffle, \
            collate_fn = self.dataset_train.collate_fn,
            num_workers = self.cfg.OPTIMIZER.NUM_WORKERS, drop_last = True)

        epoch_start = self.epoch_start
        epoch_end = self.cfg.OPTIMIZER.MAX_EPOCH

        if self.is_logging:
            idx_log_iter = 0 if self.log_iter_start is None else self.log_iter_start

        if self.optim_fastai:
            accumulated_iter = 0
            cfg_optim = self.cfg.OPTIMIZER
            use_amp = cfg_optim.get('USE_AMP', False)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=cfg_optim.get('LOSS_SCALE_FP16', 2.0**16))

        last_model_path = None
        last_util_path = None
        for epoch in range(epoch_start, epoch_end):
            epoch_time_start = time.time()
            epoch_display = epoch + 1
            print(f'* Training epoch = {epoch_display}/{epoch_end}')
            if self.is_logging:
                print(f'* Logging path = {self.path_log}')
            
            self.network.train()
            self.network.training = True
            avg_loss = []
            train_time_start = time.time()
            for idx_iter, dict_datum in enumerate(tqdm(data_loader_train)):
                if self.optim_fastai:
                    self.scheduler.step(accumulated_iter, epoch)
                
                if self.distil:
                    with torch.no_grad():
                        dict_datum = self.distil_model(dict_datum)
                        dict_datum['ldr_bev_feat'] = dict_datum['spatial_features_2d']
                
                # try:
                dict_net = self.network(dict_datum)
                # except:
                #     print('* error: ', dict_datum['meta'])
                
                if self.get_loss_from == 'head':
                    loss = self.network.head.loss(dict_net)
                elif self.get_loss_from == 'detector':
                    loss = self.network.loss(dict_net)
                
                try:
                    log_avg_loss = loss.cpu().detach().item()
                except:
                    log_avg_loss = loss
                avg_loss.append(log_avg_loss)

                if self.optim_fastai:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.network.parameters(), cfg_optim.GRAD_NORM_CLIP)
                    scaler.step(self.optimizer)
                    scaler.update()
                    accumulated_iter += 1
                else:
                    if loss == 0.:
                        pass
                        # print('loss is 0.') # No objs for all samples
                    elif torch.isfinite(loss):
                        loss.backward()
                    else:
                        print('* Exception error (pipeline): nan or inf loss happend')
                        print('* Meta: ', dict_datum['meta'])

                    self.optimizer.step()
                    if not (self.scheduler is None):
                        self.scheduler.step()
                
                self.optimizer.zero_grad()

                if self.is_logging:
                    dict_logging = dict_net['logging']
                    idx_log_iter +=1
                    for k, v in dict_logging.items():
                        self.log_train_iter.add_scalar(f'train/{k}', v, idx_log_iter)
                    if not (self.scheduler is None):
                        if self.optim_fastai:
                            lr = float(self.optimizer.lr)
                            self.log_train_iter.add_scalar(f'train/learning_rate', lr, idx_log_iter)
                        else:
                            lr = self.scheduler.get_last_lr()
                            self.log_train_iter.add_scalar(f'train/learning_rate', lr[0], idx_log_iter)

                # free memory (Killed error, checked with htop)
                if 'pointer' in dict_datum.keys():
                    for dict_item in dict_datum['pointer']:
                        for k in dict_item.keys():
                            if k != 'meta':
                                dict_item[k] = None
                for temp_key in dict_datum.keys():
                    dict_datum[temp_key] = None
            train_time_sec = time.time() - train_time_start

            path_dict_model = None
            path_dict_util = None
            save_time_sec = 0.0
            if self.is_save_model:
                path_dict_model = os.path.join(self.path_log, 'models', f'model_{epoch_display}.pt')
                path_dict_util = os.path.join(self.path_log, 'utils', f'util_{epoch_display}.pt')

                if epoch_display % self.interval_epoch_model == 0:
                    save_time_start = time.time()
                    torch.save(self.network.state_dict(), path_dict_model)
                    save_time_sec += time.time() - save_time_start
                    last_model_path = path_dict_model
                if epoch_display % self.interval_epoch_util == 0:
                    dict_util = {
                        'epoch': epoch_display,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'idx_log_iter': idx_log_iter, 
                    }
                    if self.optim_fastai:
                        dict_util.update({'it': accumulated_iter})
                    else:
                        if not (self.scheduler is None):
                            dict_util.update({'scheduler_state_dict': self.scheduler.state_dict()})
                    save_time_start = time.time()
                    torch.save(dict_util, path_dict_util)
                    save_time_sec += time.time() - save_time_start
                    last_util_path = path_dict_util

            avg_epoch_loss = float(np.mean(avg_loss)) if len(avg_loss) > 0 else float('nan')
            print(f'* Epoch {epoch_display} avg_loss = {avg_epoch_loss:.6f}')

            if self.is_logging:
                self.log_train_epoch.add_scalar(f'train/avg_loss', avg_epoch_loss, epoch_display)

            eval_subset_time_sec = 0.0
            eval_full_time_sec = 0.0
            best_update_time_sec = 0.0
            if self.is_validate:
                if self.is_consider_subset:
                    if (epoch_display % self.val_per_epoch_subset) == 0:
                        eval_time_start = time.time()
                        self.validate_kitti(epoch_display, list_conf_thr=self.list_val_conf_thr, is_subset=True)
                        eval_subset_time_sec += time.time() - eval_time_start
                if (epoch_display % self.val_per_epoch_full) == 0:
                    eval_time_start = time.time()
                    eval_metrics = self.validate_kitti(epoch_display, list_conf_thr=self.list_val_conf_thr)
                    eval_full_time_sec += time.time() - eval_time_start
                    best_time_start = time.time()
                    self.update_best_checkpoint(
                        epoch_display,
                        eval_metrics,
                        path_dict_model,
                        path_dict_util,
                    )
                    best_update_time_sec += time.time() - best_time_start

            epoch_time_sec = time.time() - epoch_time_start
            print(
                f'* Epoch {epoch_display} timing: '
                f'train={train_time_sec:.2f}s, save={save_time_sec:.2f}s, '
                f'eval_subset={eval_subset_time_sec:.2f}s, eval_full={eval_full_time_sec:.2f}s, '
                f'best={best_update_time_sec:.2f}s, total={epoch_time_sec:.2f}s'
            )
            self.write_epoch_summary(
                epoch=epoch_display,
                avg_loss=avg_epoch_loss,
                path_model=path_dict_model if path_dict_model and os.path.exists(path_dict_model) else None,
                path_util=path_dict_util if path_dict_util and os.path.exists(path_dict_util) else None,
                num_iter=len(avg_loss),
                train_time_sec=train_time_sec,
                save_time_sec=save_time_sec,
                eval_subset_time_sec=eval_subset_time_sec,
                eval_full_time_sec=eval_full_time_sec,
                best_update_time_sec=best_update_time_sec,
                epoch_time_sec=epoch_time_sec,
            )

        if self.is_save_model:
            self.copy_latest_checkpoint(last_model_path, last_util_path)
            if self.best_score is None:
                self.update_best_checkpoint(
                    epoch_end,
                    None,
                    last_model_path,
                    last_util_path,
                    force=True,
                )

    def copy_latest_checkpoint(self, path_model=None, path_util=None):
        if not hasattr(self, 'path_log'):
            return

        path_latest_model = os.path.join(self.path_log, 'models', 'latest.checkpoint')
        path_latest_util = os.path.join(self.path_log, 'utils', 'latest.checkpoint')

        if path_model is not None and os.path.exists(path_model):
            shutil.copy2(path_model, path_latest_model)
            print(f'* Latest checkpoint copied: {path_latest_model}')
        if path_util is not None and os.path.exists(path_util):
            shutil.copy2(path_util, path_latest_util)

    def update_best_checkpoint(self, epoch, eval_metrics, path_model=None, path_util=None, force=False):
        if not hasattr(self, 'path_log') or not self.is_save_model:
            return
        if path_model is None or not os.path.exists(path_model):
            return

        score = None if eval_metrics is None else self.pick_best_metric_score(eval_metrics)
        if score is None and not force:
            print(f'* Best checkpoint skipped: metric not found for epoch {epoch}')
            return
        if score is None:
            score = float('-inf')

        if force or self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            path_best_model = os.path.join(self.path_log, 'models', 'best.checkpoint')
            path_best_util = os.path.join(self.path_log, 'utils', 'best.checkpoint')
            shutil.copy2(path_model, path_best_model)
            if path_util is not None and os.path.exists(path_util):
                shutil.copy2(path_util, path_best_util)
            self.write_best_summary(epoch, score, path_model, path_util)
            print(f'* Best checkpoint updated: epoch {epoch}, score={score:.6f}')

    def pick_best_metric_score(self, eval_metrics):
        cfg_metric = self.best_metric_cfg
        scores = []
        for item in eval_metrics:
            if abs(float(item['conf_thr']) - cfg_metric['conf_thr']) > 1e-6:
                continue
            if cfg_metric['cls'] != 'auto' and str(item['cls']).lower() != cfg_metric['cls']:
                continue
            if not any(abs(float(item['iou']) - iou) <= 1e-6 for iou in cfg_metric['ious']):
                continue
            if cfg_metric['only_classes_with_gt'] and not item.get('has_gt', True):
                continue
            scores.append(float(item[cfg_metric['kind']]))
        if len(scores) == 0:
            return None
        return float(np.mean(scores))

    def write_best_summary(self, epoch, score, path_model=None, path_util=None):
        path_summary = os.path.join(self.path_log, 'best_summary.txt')
        metric = self.best_metric_cfg
        with open(path_summary, 'w') as f:
            f.write(f'best_epoch: {epoch}\n')
            f.write(f"metric_cls: {metric['cls']}\n")
            f.write(f"metric_kind: {metric['kind']}\n")
            f.write(f"metric_ious: {metric['ious']}\n")
            f.write(f"metric_conf_thr: {metric['conf_thr']}\n")
            f.write(f"only_classes_with_gt: {metric['only_classes_with_gt']}\n")
            f.write(f'score: {score:.6f}\n')
            if path_model is not None:
                f.write(f'source_model: {path_model}\n')
            if path_util is not None:
                f.write(f'source_util: {path_util}\n')

    def has_gt_for_cls(self, gt_annos, cls_name):
        for anno in gt_annos:
            if 'name' not in anno:
                continue
            names = anno['name']
            if np.any(names == cls_name):
                return True
        return False

    def write_epoch_summary(
        self,
        epoch,
        avg_loss,
        path_model=None,
        path_util=None,
        num_iter=None,
        train_time_sec=None,
        save_time_sec=None,
        eval_subset_time_sec=None,
        eval_full_time_sec=None,
        best_update_time_sec=None,
        epoch_time_sec=None,
    ):
        if not hasattr(self, 'path_log'):
            return

        path_summary = os.path.join(self.path_log, 'summary.txt')
        with open(path_summary, 'a') as f:
            f.write(f'[epoch {epoch}]\n')
            f.write(f'avg_loss: {avg_loss:.6f}\n')
            if num_iter is not None:
                f.write(f'num_iter: {num_iter}\n')
            if train_time_sec is not None:
                f.write(f'train_time_sec: {train_time_sec:.6f}\n')
            if save_time_sec is not None:
                f.write(f'save_time_sec: {save_time_sec:.6f}\n')
            if eval_subset_time_sec is not None:
                f.write(f'eval_subset_time_sec: {eval_subset_time_sec:.6f}\n')
            if eval_full_time_sec is not None:
                f.write(f'eval_full_time_sec: {eval_full_time_sec:.6f}\n')
            if best_update_time_sec is not None:
                f.write(f'best_update_time_sec: {best_update_time_sec:.6f}\n')
            if epoch_time_sec is not None:
                f.write(f'epoch_time_sec: {epoch_time_sec:.6f}\n')
            if path_model is not None:
                f.write(f'model: {path_model}\n')
            if path_util is not None:
                f.write(f'util: {path_util}\n')
            f.write('\n')

        path_csv = os.path.join(self.path_log, 'train_summary.csv')
        is_new = not os.path.exists(path_csv)
        with open(path_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    'epoch', 'avg_loss', 'num_iter',
                    'train_time_sec', 'save_time_sec',
                    'eval_subset_time_sec', 'eval_full_time_sec',
                    'best_update_time_sec', 'epoch_time_sec',
                    'model', 'util',
                ])
            writer.writerow([
                epoch,
                avg_loss,
                '' if num_iter is None else num_iter,
                '' if train_time_sec is None else train_time_sec,
                '' if save_time_sec is None else save_time_sec,
                '' if eval_subset_time_sec is None else eval_subset_time_sec,
                '' if eval_full_time_sec is None else eval_full_time_sec,
                '' if best_update_time_sec is None else best_update_time_sec,
                '' if epoch_time_sec is None else epoch_time_sec,
                path_model or '',
                path_util or '',
            ])

    def write_eval_summary(self, epoch, path_dir, conf_thr, condition, dict_metrics):
        if not hasattr(self, 'path_log'):
            return

        path_csv = os.path.join(self.path_log, 'eval_summary.csv')
        is_new = not os.path.exists(path_csv)
        with open(path_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['epoch', 'path_dir', 'conf_thr', 'condition', 'cls', 'iou', 'bev', '3d'])
            for iou, bev, det3d in zip(dict_metrics['iou'], dict_metrics['bev'], dict_metrics['3d']):
                writer.writerow([
                    'none' if epoch is None else epoch,
                    path_dir,
                    conf_thr,
                    condition,
                    dict_metrics['cls'],
                    iou,
                    bev,
                    det3d,
                ])

    def load_dict_model(self, path_dict_model, is_strict=False):
        pt_dict_model = torch.load(path_dict_model)
        self.network.load_state_dict(pt_dict_model, strict=is_strict)

    # V2
    def vis_infer(self, sample_indices, conf_thr=0.7, is_nms=True, vis_mode=['lpc', 'spcube', 'cube'], is_train=False):
        '''
        * sample_indices: e.g. [0, 1, 2, 3, 4]
        * assume batch_size = 1 for convenience
        * vis_mode (TBD)
        '''
        self.network.eval()
        
        if is_train:
            dataset_loaded = self.dataset_train
        else:
            dataset_loaded = self.dataset_test
        subset = Subset(dataset_loaded, sample_indices)
        data_loader = torch.utils.data.DataLoader(subset,
                batch_size = 1, shuffle = False,
                collate_fn = self.dataset_test.collate_fn,
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        for dict_datum in data_loader:
            dict_out = self.network(dict_datum)
            dict_out = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms)

            ### Vis data ###
            pc_lidar = dict_datum['ldr64']
            # rdr_spcube = dict_datum['rdr_sparse_cube']
            # rdr_cube = dict_datum['rdr_cube']
            ### Vis data ###

            ### Labels ###
            labels = dict_out['label'][0]
            list_obj_label = []
            for label_obj in labels:
                cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
                obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                list_obj_label.append(obj)
            ### Labels ###

            ### Preds: post processing bbox ###
            list_obj_pred = []
            list_cls_pred = []
            if dict_datum['pp_num_bbox'] == 0:
                pass
            else:
                pp_cls = dict_datum['pp_cls']
                for idx_pred, pred_obj in enumerate(dict_datum['pp_bbox']):
                    conf_score, xc, yc, zc, xl, yl, zl, rot = pred_obj
                    obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
                    list_obj_pred.append(obj)
                    list_cls_pred.append('Sedan')
                    # list_cls_pred.append(self.dict_cls_id_to_name[pp_cls[idx_pred]])
            ### Preds: post processing bbox ###

            ### Vis for open3d ###
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [6, 7], #[5, 6],[4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    [0, 2], [1, 3], [4, 6], [5, 7]]
            colors_label = [[0, 0, 0] for _ in range(len(lines))]
            list_line_set_label = []
            list_line_set_pred = []
            for label_obj in list_obj_label:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors_label)
                list_line_set_label.append(line_set)
            
            for idx_pred, pred_obj in enumerate(list_obj_pred):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                # colors_pred = [self.dict_cls_name_to_rgb[list_cls_pred[idx_pred]] for _ in range(len(lines))]
                colors_pred = [[1.,0.,0.] for _ in range(len(lines))]
                line_set.colors = o3d.utility.Vector3dVector(colors_pred)
                list_line_set_pred.append(line_set)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
            o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred)
            ### Vis for open3d ###

        return list_obj_label, list_obj_pred

    # V2
    def validate_kitti(self, epoch=None, list_conf_thr=None, is_subset=False):
        self.network.training=False
        self.network.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background
        

        ### Check is_validate with small dataset ###
        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='* Test (Subset): ')
            log_header = 'val_sub'
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(self.dataset_test), desc='* Test (Total): ')
            log_header = 'val_tot'

        data_loader = torch.utils.data.DataLoader(self.dataset_test, \
                batch_size=1, shuffle=is_shuffle, collate_fn=self.dataset_test.collate_fn, \
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        # initialize via VAL.LIST_VAL_CONF_THR
        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        # print(path_dir)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
                f.write('')
            f.close()

        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break
            
            try:
                dict_out = self.network(dict_datum) # inference
                is_feature_inferenced = True
            except:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False

            idx_name = str(idx_datum).zfill(6)

            ### for every conf in list_conf_thr ###
            for conf_thr in list_conf_thr:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'val.txt'
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]
                            
                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)                
                if dict_out is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out = dict_datum_to_kitti(self, dict_out)

                if len(dict_out['kitti_gt']) == 0: # no eval for emptry obj label
                    pass
                else:
                    ### Gt ###
                    for idx_label, label in enumerate(dict_out['kitti_gt']):
                        open_mode = 'w' if idx_label == 0 else 'a'
                        with open(labels_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(label+'\n')
                    ### Gt ###

                    ### Process description ###
                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out['kitti_desc'])
                    ### Process description ###

                    ### Pred: do not care len 0 with if else: already care as dummy ###
                    for idx_pred, pred in enumerate(dict_out['kitti_pred']):
                        open_mode = 'w' if idx_pred == 0 else 'a'
                        with open(preds_dir + '/' + idx_name + '.txt', open_mode) as f:
                            f.write(pred+'\n')
                    ### Pred: do not care len 0 with if else: already care as dummy ###

                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)
            
            # free memory (Killed error, checked with htop)
            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        eval_metric_rows = []
        ### Validate per conf ###
        for conf_thr in list_conf_thr:
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
            split_path = path_dir + f'/{conf_thr}/' + 'val.txt'

            dt_annos = kitti.get_label_annos(preds_dir)
            val_ids = read_imageset_file(split_path)
            gt_annos = kitti.get_label_annos(labels_dir, val_ids)

            list_metrics = []
            for idx_cls_val in self.list_val_care_idx:
                if self.is_validation_updated:
                    # Thanks to Felix Fent (in TUM) and Miao Zhang (in Bosch Research)
                    # Fixed mixed interpolation (issue #28) and z_center (issue #36) in evaluation
                    dict_metrics, result_log = get_official_eval_result_revised(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                else:
                    dict_metrics, result_log = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                print(f'-----conf{conf_thr}-----')
                print(result_log)
                list_metrics.append(dict_metrics)

            for dict_metrics in list_metrics:
                cls_name = dict_metrics['cls']
                has_gt = self.has_gt_for_cls(gt_annos, cls_name)
                ious = dict_metrics['iou']
                bevs = dict_metrics['bev']
                ap3ds = dict_metrics['3d']
                self.log_test.add_scalars(f'{log_header}/BEV_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': bevs[0],
                    f'iou_{ious[1]}_{cls_name}': bevs[1],
                    f'iou_{ious[2]}_{cls_name}': bevs[2],
                }, epoch)
                self.log_test.add_scalars(f'{log_header}/3D_conf_thr_{conf_thr}', {
                    f'iou_{ious[0]}_{cls_name}': ap3ds[0],
                    f'iou_{ious[1]}_{cls_name}': ap3ds[1],
                    f'iou_{ious[2]}_{cls_name}': ap3ds[2],
                }, epoch)
                self.write_eval_summary(epoch, path_dir, conf_thr, 'total', dict_metrics)
                for iou, bev, ap3d in zip(ious, bevs, ap3ds):
                    eval_metric_rows.append({
                        'epoch': epoch,
                        'path_dir': path_dir,
                        'conf_thr': conf_thr,
                        'condition': 'total',
                        'cls': cls_name,
                        'iou': iou,
                        'bev': bev,
                        '3d': ap3d,
                        'has_gt': has_gt,
                    })
        ### Validate per conf ###
        return eval_metric_rows

    def validate_kitti_conditional(self, epoch=None, list_conf_thr=None, is_subset=False, is_print_memory=False):
        self.network.eval()

        eval_ver2 = self.cfg.get('cfg_eval_ver2', False)
        if eval_ver2:
            class_names = []
            dict_label = self.dataset_test.label.copy()
            list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
            for temp_key in list_for_pop:
                dict_label.pop(temp_key)
            for k, v in dict_label.items():
                _, logit_idx, _, _ = v
                if logit_idx > 0:
                    class_names.append(k)
            self.dict_cls_id_to_name = dict()
            for idx_cls, cls_name in enumerate(class_names):
                self.dict_cls_id_to_name[(idx_cls+1)] = cls_name # 1 for Background
        
        road_cond_list = ['urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university']
        time_cond_list = ['day', 'night']
        weather_cond_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']

        # Check is_validate with small dataset
        if is_subset:
            is_shuffle = True
            tqdm_bar = tqdm(total=self.val_num_subset, desc='Test (Subset): ')
        else:
            is_shuffle = False
            tqdm_bar = tqdm(total=len(self.dataset_test), desc='Test (Total): ')

        data_loader = torch.utils.data.DataLoader(self.dataset_test, \
                batch_size = 1, shuffle = is_shuffle, collate_fn = self.dataset_test.collate_fn, \
                num_workers = self.cfg.OPTIMIZER.NUM_WORKERS)
        
        if epoch is None:
            dir_epoch = 'none'
        else:
            dir_epoch = f'epoch_{epoch}_subset' if is_subset else f'epoch_{epoch}_total'

        # initialize via VAL.LIST_VAL_CONF_THR
        path_dir = os.path.join(self.path_log, 'test_kitti', dir_epoch)
        for conf_thr in list_conf_thr:
            os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)

            os.makedirs(os.path.join(path_dir, f'{conf_thr}', 'all'), exist_ok=True)
            with open(path_dir + f'/{conf_thr}/' + 'all/val.txt', 'w') as f:
                f.write('')

            for road_cond in road_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', road_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + road_cond + '/val.txt', 'w') as f:
                    f.write('')

            for time_cond in time_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', time_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + time_cond + '/val.txt', 'w') as f:
                    f.write('')

            for weather_cond in weather_cond_list:
                os.makedirs(os.path.join(path_dir, f'{conf_thr}', weather_cond), exist_ok=True)
                with open(path_dir + f'/{conf_thr}/' + weather_cond + '/val.txt', 'w') as f:
                    f.write('')

            pred_dir_list = []
            label_dir_list = []
            desc_dir_list = []
            split_path_list = []

            ### For All Conditions ###
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
            list_dir = [preds_dir, labels_dir, desc_dir]
            split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

            for temp_dir in list_dir:
                os.makedirs(temp_dir, exist_ok=True)

            pred_dir_list.append(preds_dir)
            label_dir_list.append(labels_dir)
            desc_dir_list.append(desc_dir)
            split_path_list.append(split_path)
                            
            ### For Specific Conditions ###
            for road_cond in road_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', road_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + road_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)
                
                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
            
            for time_cond in time_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', time_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + time_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)
            
            for weather_cond in weather_cond_list:
                preds_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', weather_cond, 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + weather_cond +'/val.txt'
                
                for temp_dir in list_dir:
                    os.makedirs(temp_dir, exist_ok=True)

                pred_dir_list.append(preds_dir)
                label_dir_list.append(labels_dir)
                desc_dir_list.append(desc_dir)
                split_path_list.append(split_path)

        # Creating gts and preds txt files for evaluation
        for idx_datum, dict_datum in enumerate(data_loader):
            if is_subset & (idx_datum >= self.val_num_subset):
                break

            try:
                dict_out = self.network(dict_datum) # inference
                is_feature_inferenced = True
            except:
                print('* Exception error (Pipeline): error during inferencing a sample -> empty prediction')
                print('* Meta info: ', dict_out['meta'])
                is_feature_inferenced = False

            if is_print_memory:
                print('max_memory: ', torch.cuda.max_memory_allocated(device='cuda'))
                
            idx_name = str(idx_datum).zfill(6)
            
            road_cond_tag, time_cond_tag, weather_cond_tag = \
                dict_out['meta'][0]['desc']['road_type'], dict_out['meta'][0]['desc']['capture_time'], dict_out['meta'][0]['desc']['climate']
            # print(dict_out['desc'][0])

            ### for every conf in list_conf_thr ###
            for conf_thr in list_conf_thr:
                ### For All Conditions ###
                preds_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'preds')
                labels_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'gts')
                desc_dir = os.path.join(path_dir, f'{conf_thr}', 'all', 'desc')
                list_dir = [preds_dir, labels_dir, desc_dir]
                split_path = path_dir + f'/{conf_thr}/' + 'all/val.txt'

                preds_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'preds')
                labels_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'gts')
                desc_dir_road = os.path.join(path_dir, f'{conf_thr}', road_cond_tag, 'desc')
                split_path_road =path_dir + f'/{conf_thr}/' + road_cond_tag + '/val.txt'

                preds_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'preds')
                labels_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'gts')
                desc_dir_time = os.path.join(path_dir, f'{conf_thr}', time_cond_tag, 'desc')
                split_path_time = path_dir + f'/{conf_thr}/' + time_cond_tag + '/val.txt'

                preds_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'preds')
                labels_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'gts')
                desc_dir_weather = os.path.join(path_dir, f'{conf_thr}', weather_cond_tag, 'desc')
                split_path_weather =path_dir + f'/{conf_thr}/' + weather_cond_tag + '/val.txt'

                os.makedirs(labels_dir_road, exist_ok=True)
                os.makedirs(labels_dir_time, exist_ok=True)
                os.makedirs(labels_dir_weather, exist_ok=True)
                os.makedirs(desc_dir_road, exist_ok=True)
                os.makedirs(desc_dir_time, exist_ok=True)
                os.makedirs(desc_dir_weather, exist_ok=True)
                os.makedirs(preds_dir_road, exist_ok=True)
                os.makedirs(preds_dir_time, exist_ok=True)
                os.makedirs(preds_dir_weather, exist_ok=True)
                
                if is_feature_inferenced:
                    if eval_ver2:
                        pred_dicts = dict_out['pred_dicts'][0]
                        pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
                        pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
                        pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
                        list_pp_bbox = []
                        list_pp_cls = []

                        for idx_pred in range(len(pred_labels)):
                            x, y, z, l, w, h, th = pred_boxes[idx_pred]
                            score = pred_scores[idx_pred]
                            
                            if score > conf_thr:
                                cls_idx = int(np.round(pred_labels[idx_pred]))
                                cls_name = class_names[cls_idx-1]
                                list_pp_bbox.append([score, x, y, z, l, w, h, th])
                                list_pp_cls.append(cls_idx)
                            else:
                                continue
                        pp_num_bbox = len(list_pp_cls)
                        dict_out_current = dict_out
                        dict_out_current.update({
                            'pp_bbox': list_pp_bbox,
                            'pp_cls': list_pp_cls,
                            'pp_num_bbox': pp_num_bbox,
                            'pp_desc': dict_out['meta'][0]['desc']
                        })
                    else:
                        dict_out_current = self.network.list_modules[-1].get_nms_pred_boxes_for_single_sample(dict_out, conf_thr, is_nms=True)
                else:
                    dict_out_current = update_dict_feat_not_inferenced(dict_out) # mostly sleet for lpc (e.g. no measurement)

                if dict_out_current is None:
                    print('* Exception error (Pipeline): dict_item is None in validation')
                    continue

                dict_out_current = dict_datum_to_kitti(self, dict_out_current)

                if len(dict_out_current['kitti_gt']) == 0: # not eval emptry label
                    pass
                else:
                    ### Gt ###
                    for idx_label, label in enumerate(dict_out_current['kitti_gt']):
                        if idx_label == 0:
                            mode = 'w'
                        else:
                            mode = 'a'

                        with open(labels_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')
                        with open(labels_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write(label+'\n')

                    ### Process description ###
                    with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_road + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_time + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])
                    with open(desc_dir_weather + '/' + idx_name + '.txt', 'w') as f:
                        f.write(dict_out_current['kitti_desc'])

                    ### Process description ###
                    if len(dict_out_current['kitti_pred']) == 0:
                        with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                        with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                            f.write('\n')
                    else:
                        for idx_pred, pred in enumerate(dict_out_current['kitti_pred']):
                            if idx_pred == 0:
                                mode = 'w'
                            else:
                                mode = 'a'

                            with open(preds_dir + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_road + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_time + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                            with open(preds_dir_weather + '/' + idx_name + '.txt', mode) as f:
                                f.write(pred+'\n')
                    
                    str_log = idx_name + '\n'
                    with open(split_path, 'a') as f:
                        f.write(str_log)
                    with open(split_path_road, 'a') as f:
                        f.write(str_log)
                    with open(split_path_time, 'a') as f:
                        f.write(str_log)
                    with open(split_path_weather, 'a') as f:
                        f.write(str_log)
                        
            # free memory (Killed error, checked with htop)
            if 'pointer' in dict_datum.keys():
                for dict_item in dict_datum['pointer']:
                    for k in dict_item.keys():
                        if k != 'meta':
                            dict_item[k] = None
            for temp_key in dict_datum.keys():
                dict_datum[temp_key] = None
            tqdm_bar.update(1)
        tqdm_bar.close()

        ### Validate per conf ###
        all_condition_list = ['all'] + road_cond_list + time_cond_list + weather_cond_list
        for conf_thr in list_conf_thr:
            for condition in all_condition_list:
                try:
                    preds_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'preds')
                    labels_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'gts')
                    desc_dir = os.path.join(path_dir, f'{conf_thr}', condition, 'desc')
                    split_path = path_dir + f'/{conf_thr}/' + condition + '/val.txt'

                    dt_annos = kitti.get_label_annos(preds_dir)
                    val_ids = read_imageset_file(split_path)
                    gt_annos = kitti.get_label_annos(labels_dir, val_ids)
                    list_metrics = []
                    list_results = []
                    for idx_cls_val in self.list_val_care_idx:
                        if self.is_validation_updated:
                            # Thanks to Felix Fent (in TUM) and Miao Zhang (in Bosch Research)
                            # Fixed mixed interpolation (issue #28) and z_center (issue #36) in evaluation
                            dict_metrics, result = get_official_eval_result_revised(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                        else:
                            dict_metrics, result = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
                        list_metrics.append(dict_metrics)
                        list_results.append(result)
                    print('Conf thr: ', str(conf_thr), ', Condition: ', condition)
                    with open(os.path.join(path_dir, f'{conf_thr}', 'complete_results.txt'), 'a') as f:
                        for dic_metric in list_metrics:
                            print('='*50)
                            print('Cls: ', dic_metric['cls'])
                            print('IoU:', dic_metric['iou'])
                            print('BEV: ', dic_metric['bev'])
                            print('3D: ', dic_metric['3d'])
                            print('-'*50)
                            self.write_eval_summary(epoch, path_dir, conf_thr, condition, dic_metric)
                            
                            f.write('Conf thr: ' + str(conf_thr) +  ', Condition: ' + condition + '\n')
                            f.write('cls: ' + dic_metric['cls'] + '\n')
                            f.write('iou: ')
                            for iou in dic_metric['iou']:
                                f.write(str(iou) + ' ')
                            f.write('\n')
                            f.write('bev: ')
                            for bev in dic_metric['bev']:
                                f.write(str(bev) + ' ')
                            f.write('\n')
                            f.write('3d  :')
                            for det3d in dic_metric['3d']:
                                f.write(str(det3d) + ' ')
                            f.write('\n\n')
                    print('\n')
                except:
                    print('* Exception error (Pipeline): Samples for the codition are not found')

        path_check = os.path.join(path_dir, 'Conf_thr', 'complete_results.txt')
        print(f'* Check {path_check}')
        ### Validate per conf ###
