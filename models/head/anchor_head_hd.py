import torch

from .anchor_head_integrated import AnchorHeadSingleIntegrated
from models.hd import HDCore


class AnchorHeadSingleHD(AnchorHeadSingleIntegrated):
    """
    Anchor head with HD-only classification.

    Classification logits come only from HD memory. Box and direction heads are
    the original CNN heads, so online adaptation can update only HD memory while
    keeping the ASF detector weights frozen.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        input_channels = int(self.model_cfg.INPUT_CHANNELS)
        hd_cfg = self.model_cfg.get('HD', None)
        if hd_cfg is None:
            raise RuntimeError('AnchorHeadSingleHD requires MODEL.HEAD.HD config.')
        self.hd_core = HDCore.from_head_cfg(
            hd_cfg=hd_cfg,
            feat_dim=input_channels,
            num_classes=self.num_class,
            num_anchors=self.num_anchors_per_location,
        )

    def forward(self, data_dict, spatial_features_2d=None):
        data_dict['gt_boxes'] = data_dict['gt_boxes'].cuda()
        if spatial_features_2d is None:
            spatial_features_2d = data_dict[self.key_features]

        cls_preds = self.hd_core.logits_from_feature_map(spatial_features_2d)
        detach_cls_in_train = bool(self.model_cfg.HD.get('DETACH_CLS_IN_TRAIN', True))
        if self.training and detach_cls_in_train:
            cls_preds = cls_preds.detach()
        box_preds = self.conv_box(spatial_features_2d)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['hd_features'] = spatial_features_2d

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(gt_boxes=data_dict['gt_boxes'])
            self.forward_ret_dict.update(targets_dict)
        else:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds,
                box_preds=box_preds,
                dir_cls_preds=dir_cls_preds,
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
            data_dict = self.post_processing(data_dict)

        return data_dict

    @torch.no_grad()
    def get_hd_features_by_labels(
        self,
        feat_map,
        box_cls_labels,
        max_pos_per_class=0,
        max_total_pos=0,
        include_negative=False,
        max_neg_per_batch=0,
        max_neg_ratio=0.0,
    ):
        feat_anchor = self.hd_core.make_anchor_features(feat_map)
        labels = box_cls_labels.reshape(-1).long()

        selected_parts = []
        max_pos_per_class = int(max_pos_per_class)
        for cls_id in range(1, self.num_class + 1):
            cls_idx = torch.nonzero(labels == cls_id, as_tuple=False).view(-1)
            if cls_idx.numel() == 0:
                continue
            if max_pos_per_class > 0 and cls_idx.numel() > max_pos_per_class:
                keep = torch.randperm(cls_idx.numel(), device=cls_idx.device)[:max_pos_per_class]
                cls_idx = cls_idx[keep]
            selected_parts.append(cls_idx)

        num_pos = sum(int(x.numel()) for x in selected_parts)
        if include_negative:
            neg_idx = torch.nonzero(labels == 0, as_tuple=False).view(-1)
            max_neg = int(max_neg_per_batch)
            max_neg_ratio = float(max_neg_ratio)
            if max_neg_ratio > 0.0 and num_pos > 0:
                ratio_cap = int(round(max_neg_ratio * num_pos))
                max_neg = ratio_cap if max_neg <= 0 else min(max_neg, ratio_cap)
            if max_neg > 0 and neg_idx.numel() > max_neg:
                keep = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:max_neg]
                neg_idx = neg_idx[keep]
            if neg_idx.numel() > 0:
                selected_parts.append(neg_idx)

        if not selected_parts:
            return feat_anchor.new_empty((0, feat_anchor.shape[1])), labels.new_empty((0,), dtype=torch.long)

        selected = torch.cat(selected_parts, dim=0)
        max_total_pos = int(max_total_pos)
        if max_total_pos > 0 and num_pos > max_total_pos:
            pos_mask = labels[selected] > 0
            pos_selected = selected[pos_mask]
            other_selected = selected[~pos_mask]
            keep = torch.randperm(pos_selected.numel(), device=pos_selected.device)[:max_total_pos]
            selected = torch.cat([pos_selected[keep], other_selected], dim=0)

        return feat_anchor[selected], labels[selected]

    @torch.no_grad()
    def get_positive_hd_features(self, feat_map, box_cls_labels, max_pos_per_class=0, max_total_pos=0):
        return self.get_hd_features_by_labels(
            feat_map,
            box_cls_labels,
            max_pos_per_class=max_pos_per_class,
            max_total_pos=max_total_pos,
            include_negative=False,
        )
