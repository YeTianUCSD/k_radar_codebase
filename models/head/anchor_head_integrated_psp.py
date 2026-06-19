from .anchor_head_integrated import AnchorHeadSingleIntegrated
from models.superposition import PSPConv2d, set_psp_scene_context


class AnchorHeadSingleIntegratedPSP(AnchorHeadSingleIntegrated):
    def __init__(self, cfg):
        super().__init__(cfg)
        superposition_cfg = cfg.MODEL.get('SUPERPOSITION', None)
        seed = int(getattr(superposition_cfg, 'SEED', 20260619)) if superposition_cfg is not None else 20260619
        enabled = bool(getattr(superposition_cfg, 'ENABLED', False)) if superposition_cfg is not None else False

        self.modulate_head_input = False
        self.modulate_head_branches = False

        self.conv_cls = PSPConv2d.from_conv2d(self.conv_cls, key_name='head.conv_cls', seed=seed, enabled=enabled)
        self.conv_box = PSPConv2d.from_conv2d(self.conv_box, key_name='head.conv_box', seed=seed, enabled=enabled)
        if self.conv_dir_cls is not None:
            self.conv_dir_cls = PSPConv2d.from_conv2d(self.conv_dir_cls, key_name='head.conv_dir_cls', seed=seed, enabled=enabled)

    def forward(self, data_dict, key_features=None):
        set_psp_scene_context(self, data_dict.get('scene_context', None))
        return super().forward(data_dict, key_features=key_features)
