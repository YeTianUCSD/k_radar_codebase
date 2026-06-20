import torch

from einops import repeat

from .a2_fusion import A2Fusion
from models.superposition import (
    SceneResidualBank,
    resolve_superposition_scene_names,
    convert_module_to_psp,
    set_psp_scene_context,
    PSPMultiheadAttention,
)


class A2FusionPSP(A2Fusion):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__(model_cfg, grid_size, **kwargs)
        superposition_cfg = kwargs.get('superposition_cfg', None)
        seed = int(getattr(superposition_cfg, 'SEED', 20260619)) if superposition_cfg is not None else 20260619
        enabled = bool(getattr(superposition_cfg, 'ENABLED', False)) if superposition_cfg is not None else False
        scene_names = resolve_superposition_scene_names(superposition_cfg)

        self.aware_query_scene_bank = SceneResidualBank(self.aware_query.shape, scene_names) if enabled and scene_names else None

        self.modulate_to_embed = False
        self.modulate_query = False
        self.modulate_post_mha = False

        for temp_key in self.key_feats:
            setattr(
                self,
                f'to_embed_{temp_key}',
                convert_module_to_psp(
                    getattr(self, f'to_embed_{temp_key}'),
                    prefix=f'fuser.to_embed.{temp_key}',
                    seed=seed,
                    enabled=enabled,
                    scene_names=scene_names,
                ),
            )
            setattr(
                self,
                f'to_patch_embed_{temp_key}',
                convert_module_to_psp(
                    getattr(self, f'to_patch_embed_{temp_key}'),
                    prefix=f'fuser.to_patch_embed.{temp_key}',
                    seed=seed,
                    enabled=enabled,
                    scene_names=scene_names,
                ),
            )

        self.pft = convert_module_to_psp(self.pft, prefix='fuser.pft', seed=seed, enabled=enabled, scene_names=scene_names)
        self.fuser = PSPMultiheadAttention.from_multihead_attention(
            self.fuser,
            key_name='fuser.mha',
            seed=seed,
            enabled=enabled,
            scene_names=scene_names,
        )

    def _get_scene_aware_query(self, scene_name):
        aware_query = self.aware_query
        if self.aware_query_scene_bank is None:
            return aware_query
        if scene_name is None:
            raise RuntimeError('scene_context is required for scene-specific aware_query')
        return aware_query + self.aware_query_scene_bank.get(scene_name, device=aware_query.device, dtype=aware_query.dtype)

    def forward(self, batch_dict):
        scene_name = batch_dict.get('scene_context', None)
        set_psp_scene_context(self, scene_name)

        list_feats = []

        is_get_feats_to_vis = False
        if not self.training and 'get_feats_to_vis' in batch_dict.keys():
            is_get_feats_to_vis = batch_dict['get_feats_to_vis']
            batch_dict['feat_b4_fusion'] = []

        for idx_key, temp_key in enumerate(self.key_feats):
            if not self.training and 'avail_feats' in batch_dict.keys() and temp_key not in batch_dict['avail_feats']:
                continue

            temp_feat = getattr(self, f'to_embed_{temp_key}')(batch_dict[temp_key])
            if is_get_feats_to_vis:
                batch_dict['feat_b4_fusion'].append(temp_feat)

            temp_feat = torch.unsqueeze(getattr(self, f'to_patch_{temp_key}')(temp_feat), dim=1)
            temp_feat = getattr(self, f'to_patch_embed_{temp_key}')(temp_feat)
            list_feats.append(temp_feat)

        kv_feats = torch.cat(list_feats, dim=1)
        b_patch, n_sensor, _ = kv_feats.shape
        scene_aware_query = self._get_scene_aware_query(scene_name)
        q_feat = repeat(scene_aware_query, 'b n c -> (b b_repeat) n c', b_repeat=b_patch)

        if self.training and self.is_scl:
            list_individual_feat = []
            for temp_kv_feat in list_feats:
                temp_fused_feat = self.fuser(q_feat, temp_kv_feat, temp_kv_feat)[0]
                list_individual_feat.append(self.to_fused_feat(self.pft(temp_fused_feat)))
            temp_arr = range(len(list_feats))
            temp_n = len(list_feats)
            for temp_i in range(temp_n):
                for temp_j in range(temp_i + 1, temp_n):
                    idx_pair_0 = temp_arr[temp_i]
                    idx_pair_1 = temp_arr[temp_j]
                    temp_kv_feat = torch.cat([list_feats[idx_pair_0], list_feats[idx_pair_1]], dim=1)
                    temp_fused_feat = self.fuser(q_feat, temp_kv_feat, temp_kv_feat)[0]
                    list_individual_feat.append(self.to_fused_feat(self.pft(temp_fused_feat)))
            batch_dict['list_individual_feat'] = list_individual_feat
        elif (not self.training) and 'feat_indiv' in batch_dict.keys():
            list_individual_feat = []
            for temp_kv_feat in list_feats:
                temp_fused_feat = self.fuser(q_feat, temp_kv_feat, temp_kv_feat)[0]
                list_individual_feat.append(self.to_fused_feat(self.pft(temp_fused_feat)))
            batch_dict['feat_indiv'] = list_individual_feat

        if 'get_att_maps' in batch_dict.keys():
            fused_feat, att_maps = self.fuser(q_feat, kv_feats, kv_feats)
            batch_dict['get_att_maps'] = self.get_feat_w_channel(att_maps)
        else:
            fused_feat = self.fuser(q_feat, kv_feats, kv_feats)[0]

        if is_get_feats_to_vis:
            batch_dict['pre_fused_feat'] = self.to_fused_feat(fused_feat)

        fused_feat = self.pft(fused_feat)
        fused_feat = self.to_fused_feat(fused_feat)

        batch_dict['fused_feat'] = fused_feat
        return batch_dict
