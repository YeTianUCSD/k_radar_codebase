import torch
import torch.nn as nn
import yaml
import numpy as np
from easydict import EasyDict

from models import skeletons, fuser, head


class FusionBaseIntegratedPSP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.init_configs(cfg)
        point_cloud_range = np.array(self.dataset_cfg.roi.xyz)
        voxel_size = self.dataset_cfg.roi.voxel_size
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)

        cam_cfg = self.model_cfg.get('CAMERA', None)
        ldr_cfg = self.model_cfg.get('LIDAR', None)
        rdr_cfg = self.model_cfg.get('RADAR', None)

        cam_encoder = self.load_each_encoder(cam_cfg, type='cam') if cam_cfg is not None else nn.Identity()
        ldr_encoder = self.load_each_encoder(ldr_cfg, type='ldr') if ldr_cfg is not None else nn.Identity()
        rdr_encoder = self.load_each_encoder(rdr_cfg, type='rdr') if rdr_cfg is not None else nn.Identity()

        self.add_module('cam', cam_encoder)
        self.add_module('ldr', ldr_encoder)
        self.add_module('rdr', rdr_encoder)

        is_freeze = self.model_cfg.FREEZE
        is_freeze_bn = self.model_cfg.FREEZE_BN
        self.freeze_encoders(is_freeze, is_freeze_bn)

        self.is_scl = self.model_cfg.get('SCL', False)
        fusion_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            self.model_cfg.FUSER,
            grid_size,
            scl=self.is_scl,
            superposition_cfg=self.superposition_cfg,
        )
        self.add_module('fuser', fusion_module)

        head_module = head.__all__[self.model_cfg.HEAD.NAME](cfg=cfg)
        self.add_module('head', head_module)

        self.loss_indiv_weight = self.model_cfg.LOSS.get('INDIV_WEIGHT', 1.0)
        self.is_logging = cfg.GENERAL.LOGGING.IS_LOGGING

        path_loaded = self.model_cfg.get('LOADED', None)
        if path_loaded is not None:
            self.load_state_dict(torch.load(path_loaded), strict=False)

        freeze_detection_head = self.model_cfg.get('FREEZE_DETECTION_HEAD', False)
        if freeze_detection_head:
            for param in self.head.parameters():
                param.requires_grad = False

    def _resolve_scene_context(self, batch_dict):
        scene_name = batch_dict.get('scene_context', None)
        if scene_name is not None:
            return str(scene_name)
        if self.scene_superposition_enabled and self.default_scene_context is None:
            raise RuntimeError('scene_context is required when MODEL.SUPERPOSITION.ENABLED is True')
        return self.default_scene_context

    def init_configs(self, cfg):
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.dataset_cfg = cfg.DATASET
        self.superposition_cfg = self.model_cfg.get('SUPERPOSITION', EasyDict())
        self.scene_superposition_enabled = bool(self.superposition_cfg.get('ENABLED', False))
        self.default_scene_context = self.superposition_cfg.get('ACTIVE_SCENE', self.superposition_cfg.get('BASE_SCENE', None))

        self.num_class = 0
        self.class_names = []
        dict_label = self.cfg.DATASET.label.copy()
        list_for_pop = ['calib', 'onlyR', 'Label', 'consider_cls', 'consider_roi', 'remove_0_obj']
        for temp_key in list_for_pop:
            dict_label.pop(temp_key)
        self.dict_cls_name_to_id = dict()
        for k, v in dict_label.items():
            _, logit_idx, _, _ = v
            self.dict_cls_name_to_id[k] = logit_idx
            self.dict_cls_name_to_id['Background'] = 0
            if logit_idx > 0:
                self.num_class += 1
                self.class_names.append(k)

    def load_each_encoder(self, encoder_cfg, type='cam'):
        with open(encoder_cfg.CFG, 'r') as f:
            new_config = yaml.safe_load(f)
        new_config = EasyDict(new_config)
        encoder = skeletons.__all__[new_config.MODEL.SKELETON](new_config)

        if encoder_cfg.PRETRAINED is not None:
            encoder.load_state_dict(torch.load(encoder_cfg.PRETRAINED))

        if type=='cam':
            encoder.head = nn.Identity()
        elif type=='ldr':
            encoder.head = nn.Identity()
        elif type=='rdr':
            encoder.head = nn.Identity()
            encoder.list_modules = encoder.list_modules[:-1]
        else:
            raise NotImplementedError('* check the type of encoder')

        setattr(self, type+'_key', encoder_cfg.KEY)
        return encoder.cuda()

    def _freeze_bn(self):
        for m in self.cam.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                m.eval()
        for m in self.ldr.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                m.eval()
        for m in self.rdr.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                m.eval()

    def freeze_encoders(self, freeze=True, freeze_bn=True):
        if freeze:
            for param in self.cam.parameters():
                param.requires_grad = False
            for param in self.ldr.parameters():
                param.requires_grad = False
            for param in self.rdr.parameters():
                param.requires_grad = False

        self.freeze_bn_enabled = freeze_bn
        if freeze_bn:
            self._freeze_bn()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn_enabled:
            self._freeze_bn()

    def forward(self, batch_dict):
        batch_dict['scene_context'] = self._resolve_scene_context(batch_dict)
        batch_dict = self.cam(batch_dict)
        batch_dict = self.ldr(batch_dict)
        batch_dict = self.rdr(batch_dict)
        batch_dict = self.fuser(batch_dict)
        batch_dict = self.head(batch_dict)
        return batch_dict

    def loss(self, batch_dict):
        loss = 0.
        rpn_loss = self.head.loss(batch_dict)
        loss += rpn_loss

        if self.is_scl:
            list_individual_feat = batch_dict['list_individual_feat']
            list_key_feats = []
            list_key_feats.extend(self.fuser.key_feats)
            temp_arr = range(len(self.fuser.key_feats))
            temp_n = len(self.fuser.key_feats)
            for temp_i in range(temp_n):
                for temp_j in range(temp_i + 1, temp_n):
                    idx_pair_0 = temp_arr[temp_i]
                    idx_pair_1 = temp_arr[temp_j]
                    temp_key_log = self.fuser.key_feats[idx_pair_0] + '_plus_' + self.fuser.key_feats[idx_pair_1]
                    list_key_feats.append(temp_key_log)

            assert len(list_key_feats) == len(list_individual_feat), '* Check # of individual feats'

            for key_feat, individual_feat in zip(list_key_feats, list_individual_feat):
                batch_dict = self.head(batch_dict, individual_feat)
                individual_rpn_loss = self.loss_indiv_weight*self.head.loss(batch_dict, key_feat)
                loss += individual_rpn_loss

        return loss
