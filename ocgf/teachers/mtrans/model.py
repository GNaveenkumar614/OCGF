# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# From MTrans (Liu et al., ECCV 2022; github.com/Cliu2/MTrans) via the team fork github.com/GNaveenkumar614/Transmodel,
# commit ebf8a90 (2026-10-06), file models/MTrans.py.
# Modified for OCGF:
#   - multi-class: one size anchor per class (a buffer, saved with the checkpoint) instead of the single car anchor,
#     and a class embedding as an extra input of the box and the confidence/direction heads;
#   - predict() returns boxes in the LiDAR frame with the raw confidence. It replaces format_kitti_labels, which
#     divided the score by the batch maximum of the confidence, making every score depend on the rest of the batch;
#   - get_loss works on batches without labelled objects and returns per-object IoU for per-class metrics;
#   - head outputs and the loss in float32, so the forward pass can run under mixed precision;
#   - orientation helpers vectorised (same results); easydict replaced by AttrDict.
"""MTrans: a 2D box and the LiDAR points in its frustum -> 3D box, foreground segmentation, completed points and a
confidence regressed to the 3D IoU of the box (a quality estimate, not a calibrated probability)."""
import math

import numpy as np
import torch
import torchvision
from torch import nn

from .attention import AttentionPointEncoder
from .fpn import FPN
from .iou import cal_diou_3d
from .ops import build_image_location_map_single, img2pc


class MTrans(nn.Module):
    def __init__(self, cfgs, anchors):
        """cfgs: the model config; anchors: [n_classes, 3] mean (l, w, h) per class, from the labelled frames only."""
        super().__init__()
        self.cfgs = cfgs
        self.register_buffer("anchors", torch.as_tensor(np.asarray(anchors, dtype=np.float32)).view(-1, 3))
        n_classes = self.anchors.size(0)
        class_dim = cfgs.get("class_embedding_dim", 32)
        self.class_embedding = nn.Embedding(n_classes, class_dim)

        # CNN image encoder
        cimg, cpts = cfgs.POINT_ATTENTION.input_img_channel, cfgs.POINT_ATTENTION.input_pts_channel
        self.cnn = FPN(4, 128, cimg)

        # MAttn Transformer
        self.attention_layers = AttentionPointEncoder(cfgs.POINT_ATTENTION)
        self.xyzd_embedding = nn.Sequential(
            nn.Linear(3, cpts),
            nn.LayerNorm(cpts),
            nn.ReLU(inplace=True),
            nn.Linear(cpts, cpts)
        )
        self.unknown_f3d = nn.Parameter(torch.zeros(cpts))
        self.unknown_f3d = nn.init.normal_(self.unknown_f3d)

        hidden_size = cfgs.POINT_ATTENTION.hidden_size
        # heads: 3D box regression
        self.foreground_head = nn.Sequential(
            nn.Linear(hidden_size + cimg + 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3)
        )

        if not cfgs.POINT_ATTENTION.use_cls_token:
            raise RuntimeError("MTrans needs use_cls_token: true")
        box_input = hidden_size + cimg + hidden_size + class_dim
        self.box_head = nn.Sequential(
            nn.Linear(box_input, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfgs.box_drop),
            nn.Linear(512, 7)
        )
        self.conf_dir_head = nn.Sequential(
            nn.Linear(box_input, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 3)
        )

        # Image transformations, apply data augmentation dynamically rather than in dataloader
        self.pred_transforms = torch.nn.Sequential(
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        self.train_transforms = torch.nn.Sequential(
            torchvision.transforms.RandomAutocontrast(p=0.5),
            torchvision.transforms.RandomAdjustSharpness(np.random.rand() * 2, p=0.5),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.3)], p=0.5),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def forward(self, data_dict):
        image = data_dict.images                                        # (B, C, H, W)
        overlap_masks = data_dict.overlap_masks                         # (B, 1, H, W)
        sub_cloud = data_dict.sub_clouds[:, :, :3]                      # (B, N, 3) for x, y, z point cloud
        sub_cloud2d = data_dict.sub_clouds2d                            # (B, N, 2) for projected point cloud
        real_point_mask = data_dict.real_point_mask                     # (B, N), 1-realpoint; 0-padding; 2-mask; 3-jitter
        device = sub_cloud.device
        pred_dict = {
            'batch_size': image.size(0)
        }

        impact_points_mask = real_point_mask == 1  # (B, N), points with known 3D coords, unmasked, unjittered
        unmaksed_known_points = (impact_points_mask) + (real_point_mask == 3)  # (B, N), points has 3D coords, no masked, no padding
        nonpadding_points = (unmaksed_known_points) + (real_point_mask == 2)  # (B, N), points known, no padding

        # normalize point cloud
        sub_cloud_center = (sub_cloud * impact_points_mask.unsqueeze(-1)).sum(dim=1) / (impact_points_mask.sum(dim=1, keepdim=True) + 1e-6)  # (B, 3)
        data_dict.locations = data_dict.locations - sub_cloud_center
        sub_cloud = sub_cloud - sub_cloud_center.unsqueeze(1)
        data_dict.sub_clouds[:, :, :3] = data_dict.sub_clouds[:, :, :3] - sub_cloud_center.unsqueeze(1)
        sub_cloud = sub_cloud * (nonpadding_points).unsqueeze(-1)
        pred_dict['subcloud_center'] = sub_cloud_center
        # random augment point cloud with 50% probability
        if self.training:
            random_shift = (torch.rand(sub_cloud.size(0), 1, 3, device=sub_cloud.device) * torch.tensor([[[2, 2, 0.5]]], device=sub_cloud.device)
                            - torch.tensor([[[1, 1, 0.25]]], device=sub_cloud.device))
            random_shift = random_shift * torch.randint(0, 2, (sub_cloud.size(0), 1, 1), device=random_shift.device).bool()
            sub_cloud = sub_cloud + random_shift
            data_dict.locations = data_dict.locations + random_shift.squeeze(1)
        # random scale point cloud
        if self.training:
            random_scale = (torch.rand(sub_cloud.size(0), 1, 3, device=sub_cloud.device) * 0.2 + 0.9)
            random_idx = torch.randint(0, 2, (sub_cloud.size(0), 1, 1), device=random_scale.device).bool()
            random_scale = random_scale * random_idx + 1 * (~random_idx)

            sub_cloud = sub_cloud * random_scale
            data_dict.locations = data_dict.locations * random_scale.squeeze(1)
            data_dict.dimensions = data_dict.dimensions * random_scale.squeeze(1)
        # random flip
        if self.training:
            # flip y coords
            random_idx = torch.randint(0, 2, (sub_cloud.size(0), 1), device=sub_cloud.device) * 2 - 1  # [-1, 1]
            sub_cloud[:, :, 1] = sub_cloud[:, :, 1] * random_idx
            data_dict.locations[:, 1] = data_dict.locations[:, 1] * random_idx.squeeze(1)
            data_dict.yaws = data_dict.yaws * random_idx
            data_dict.sub_clouds = sub_cloud

            flipped_image = torch.flip(image, dims=(-1,))
            image = image * (random_idx != -1).view(-1, 1, 1, 1) + flipped_image * (random_idx == -1).view(-1, 1, 1, 1)
            flipped_x = image.size(-1) - 1 - sub_cloud2d[:, :, 0]
            sub_cloud2d[:, :, 0] = sub_cloud2d[:, :, 0] * (random_idx != -1) + flipped_x * (random_idx == -1)
            data_dict.images = image

        pred_dict['gt_coords_3d'] = sub_cloud

        # random augment input image
        if self.training:
            image = torch.cat([self.train_transforms(image), overlap_masks], dim=1)
        else:
            # normalize image
            image = torch.cat([self.pred_transforms(image), overlap_masks], dim=1)

        # 1. extract information of images
        B, _, H, W = image.size()
        image_features, _ = self.cnn(image)

        # 2. build new cloud, which contains blank slots to be interpolated
        B, N, _ = sub_cloud.size()
        scale = self.cfgs.sparse_query_rate
        qH, qW = H // scale, W // scale
        # hide and jitter point cloud
        jittered_cloud = sub_cloud.clone()
        jittered_cloud[(real_point_mask == 3).unsqueeze(-1).repeat(1, 1, 3)] += torch.rand((real_point_mask == 3).sum() * 3, device=sub_cloud.device) * 0.1 - 0.05
        jittered_cloud = jittered_cloud * (unmaksed_known_points.unsqueeze(-1))

        key_c2d = sub_cloud2d                                                                   # (B, N, 2)
        key_f2d = img2pc(image_features, key_c2d).transpose(-1, -2)                             # (B, N, Ci)
        key_f3d = self.xyzd_embedding(jittered_cloud) * (unmaksed_known_points.unsqueeze(-1)) + \
            self.unknown_f3d.view(1, 1, -1).repeat(B, N, 1) * (~unmaksed_known_points.unsqueeze(-1))
        query_c2d = (build_image_location_map_single(qH, qW, device) * scale).view(1, -1, 2).repeat(B, 1, 1)     # (B, H*W, 2)
        query_f2d = img2pc(image_features, query_c2d).transpose(-1, -2)                         # (B, H*W, Ci)
        query_f3d = self.unknown_f3d.view(1, 1, -1).repeat(B, query_f2d.size(1), 1)

        # 3. Self-attention to decode missing 3D features
        # only unmasked known foreground will be attended
        attn_mask = unmaksed_known_points
        query_f3d, key_f3d, cls_f3d = self.attention_layers(query_c2d, query_f2d, query_f3d, key_c2d, key_f2d, key_f3d, attn_mask)
        pred_key_coords_3d = self.xyz_head(key_f3d)                                                # (B, N, 3)
        pred_dict['pred_coords_3d'] = pred_key_coords_3d

        diff_xyz = (pred_key_coords_3d.detach() - sub_cloud) * nonpadding_points.unsqueeze(-1)
        pred_key_foreground = self.foreground_head(torch.cat([key_f2d, key_f3d, diff_xyz], dim=-1))   # (B, N, 2)
        pred_dict['pred_foreground_logits'] = pred_key_foreground

        pred_query_coords_3d = self.xyz_head(query_f3d)
        pred_dict['enriched_points'] = pred_query_coords_3d
        pred_query_foreground = self.foreground_head(torch.cat([query_f2d, query_f3d, torch.zeros_like(pred_query_coords_3d)], dim=-1))
        pred_dict['enriched_foreground_logits'] = pred_query_foreground

        # 4. Predict 3D box
        # norm center
        all_points = torch.cat([sub_cloud], dim=1)
        all_forground_mask = torch.cat([pred_key_foreground], dim=1).argmax(dim=-1, keepdim=True) * unmaksed_known_points.unsqueeze(-1)
        seg_center = (all_points * all_forground_mask).sum(dim=1) / ((all_forground_mask).sum(dim=1) + 1e-6)
        data_dict.locations = data_dict.locations - seg_center
        pred_dict['second_offset'] = seg_center

        # make the logits more discriminative
        query_fore_logits = (pred_query_foreground * 5).softmax(dim=-1)
        key_fore_logits = (pred_key_foreground * 5).softmax(dim=-1)
        global_f3d = ((query_f3d * query_fore_logits[:, :, 1:2]).sum(dim=1) + (key_f3d * key_fore_logits[:, :, 1:2]).sum(dim=1)) / \
            (query_fore_logits[:, :, 1:2].sum(dim=1) + key_fore_logits[:, :, 1:2].sum(dim=1) + 1e-10)         # (B, Cp)
        global_f2d = torch.nn.AdaptiveAvgPool2d((1, 1))(image_features).squeeze(-1).squeeze(-1)  # (B, Ci)

        box_feature = torch.cat([cls_f3d.squeeze(1), global_f3d, global_f2d, self.class_embedding(data_dict.classes)], dim=-1)
        # float32 from here on: under mixed precision, exp / tanh / arcsin of half-precision outputs overflow or saturate
        box = self.box_head(box_feature).float()
        conf_dir_pred = self.conf_dir_head(box_feature).float()

        location, dimension, yaw = box[:, 0:3], box[:, 3:6], box[:, 6:7]
        direction, confidence = conf_dir_pred[:, 0:2], conf_dir_pred[:, 2:3]
        confidence = torch.nn.Sigmoid()(confidence)

        dim_anchor = self.anchors[data_dict.classes]                    # (B, 3): each object's class anchor
        da = torch.norm(dim_anchor[:, :2], dim=-1, keepdim=True)
        ha = dim_anchor[:, 2:3]
        pred_loc = location * torch.cat([da, da, ha], dim=-1)
        pred_dim = torch.exp(dimension) * dim_anchor
        pred_yaw = torch.tanh(yaw)
        pred_yaw = torch.arcsin(pred_yaw)

        pred_dict['location'] = pred_loc
        pred_dict['dimension'] = pred_dim
        pred_dict['yaw'] = pred_yaw
        pred_dict['direction'] = direction
        pred_dict['conf'] = confidence

        return pred_dict

    @torch.no_grad()
    def predict(self, data_dict):
        """Boxes [B, 7] (x, y, z, l, w, h, yaw) in the LiDAR frame and raw confidences [B]. Eval mode only: then an
        object's result does not depend on the other objects in the batch."""
        assert not self.training, "call model.eval() first"
        pred = self.forward(data_dict)
        center = pred['location'] + pred['subcloud_center'] + pred['second_offset']
        yaw = self.adjust_direction(pred['yaw'], pred['direction'].argmax(dim=-1))
        return torch.cat([center, pred['dimension'], yaw], dim=-1), pred['conf'].squeeze(-1)

    def get_loss(self, pred_dict, data_dict):
        """Returns (total loss, {name: (value, count)} running stats, IoU3D per labelled object, confidence per
        labelled object). Segmentation and box terms use labelled objects only; the point reconstruction uses all."""
        stats = {}
        # the loss always runs in float32, also after a mixed-precision forward pass
        pred_dict = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in pred_dict.items()}
        B = pred_dict['batch_size']
        has_label = data_dict.use_3d_label                      # (B)
        real_point_mask = data_dict.real_point_mask            # (B, N)
        n_labelled = int(has_label.sum())
        zero = pred_dict['conf'].sum() * 0

        # 1. foreground loss
        loss_segment = zero
        if n_labelled:
            segment_logits = pred_dict['pred_foreground_logits'].transpose(-1, -2)[has_label]   # (B', 2, N)
            segment_gt = data_dict.foreground_label[has_label]                              # (B', N)
            loss_per_point = nn.CrossEntropyLoss(reduction='none', ignore_index=2)(segment_logits, segment_gt)
            # balance fore and background, take mean across batch samples
            lseg = zero
            if (segment_gt == 1).sum() > 0:
                lseg = lseg + (loss_per_point * (segment_gt == 1)).sum(dim=-1) / ((segment_gt == 1).sum(dim=-1) + 1e-6)
            if (segment_gt == 0).sum() > 0:
                lseg = lseg + (loss_per_point * (segment_gt == 0)).sum(dim=-1) / ((segment_gt == 0).sum(dim=-1) + 1e-6)
            # dice loss
            segment_prob = segment_logits.softmax(dim=1)[:, 1, :]
            inter = 2 * (segment_prob * (segment_gt == 1)).sum(dim=-1) + 1e-6
            uni = (segment_prob * (segment_gt != 2)).sum(dim=-1) + (segment_gt == 1).sum(dim=-1) + 1e-6
            loss_segment = lseg.mean() + (1 - inter / uni).mean()

            # metric: Segment IoU
            segment_pred = segment_logits.argmax(dim=1) * (segment_gt != 2)
            intersection = (segment_pred * (segment_gt == 1)).sum(dim=1)
            union = ((segment_pred + (segment_gt == 1)).bool()).sum(dim=1) + 1e-10
            stats['segment_iou'] = ((intersection / union).mean().item(), n_labelled)
            stats['loss_segment'] = (loss_segment.item(), n_labelled)

        # 2. depth loss (point reconstruction; needs no labels)
        gt_coords = pred_dict['gt_coords_3d']                     # (B, N, 3)
        pred_coords = pred_dict['pred_coords_3d']                 # (B, N, 3)
        loss_mask = pred_dict['pred_foreground_logits'].argmax(dim=-1).float()  # ignore background
        loss_mask = (loss_mask) * (real_point_mask != 0)                        # ignore padding

        loss_depth = nn.SmoothL1Loss(reduction='none')(pred_coords, gt_coords).sum(dim=-1)
        loss_depth = loss_depth * loss_mask

        # balance mask/jitter/impact points
        l = (loss_depth * (real_point_mask == 1)).sum(dim=1) / (((real_point_mask == 1) * loss_mask).sum(dim=1) + 1e-6) * 0.1
        l = l + (loss_depth * (real_point_mask == 2)).sum(dim=1) / (((real_point_mask == 2) * loss_mask).sum(dim=1) + 1e-6)
        assert (real_point_mask != 3).all()
        loss_depth = l.mean()

        # metric: mean xyz err
        err_dist = torch.norm(pred_coords - gt_coords, dim=-1)
        err_dist = (err_dist * (loss_mask == 1) * (real_point_mask == 2)).sum(dim=-1) / (((loss_mask == 1) * (real_point_mask == 2)).sum(dim=-1) + 1e-6)
        stats['loss_depth'] = (loss_depth.item(), B)
        stats['err_dist'] = (err_dist.mean().item(), B)

        # 3. box loss
        loss_box = loss_dir = loss_conf = zero
        iou3d = confidence = torch.zeros(0, device=gt_coords.device)
        if n_labelled:
            pred_loc, pred_dim, pred_yaw = pred_dict['location'][has_label], pred_dict['dimension'][has_label], pred_dict['yaw'][has_label]
            gt_loc, gt_dim, gt_yaw = data_dict.locations[has_label], data_dict.dimensions[has_label], data_dict.yaws[has_label]
            gt_boxes = torch.cat([gt_loc, gt_dim, gt_yaw], dim=-1)
            pred_boxes = torch.cat([pred_loc, pred_dim, pred_yaw], dim=-1)
            l_iou, iou3d, iou2d = cal_diou_3d(pred_boxes.unsqueeze(1), gt_boxes.unsqueeze(1))
            loss_box = l_iou.mean()

            # loss for direction
            gt_dir = self.clamp_orientation_range(gt_yaw)
            gt_dir = ((gt_dir >= -np.pi / 2) * (gt_dir < np.pi / 2)).long().squeeze(-1)
            pred_dir = pred_dict['direction'][has_label]
            loss_dir = torch.nn.CrossEntropyLoss()(pred_dir, gt_dir)

            # loss for confidence
            confidence = pred_dict['conf'][has_label]
            loss_conf = torch.nn.SmoothL1Loss()(confidence, iou3d)
            assert not iou3d.isnan().any()
            iou3d, confidence = iou3d.detach().flatten(), confidence.detach().flatten()

            stats['loss_box'] = (loss_box.item(), n_labelled)
            stats['iou3d'] = (iou3d.mean().item(), n_labelled)
            stats['recall_7'] = ((iou3d > 0.7).float().mean().item(), n_labelled)
            stats['acc_dir'] = ((pred_dir.argmax(dim=-1) == gt_dir).float().mean().item(), n_labelled)
            stats['err_conf'] = ((confidence - iou3d).abs().mean().item(), n_labelled)

        loss = loss_segment + loss_depth + loss_box * 5 + loss_conf + loss_dir
        stats['loss'] = (loss.item(), B)
        return loss, stats, iou3d, confidence

    @staticmethod
    def clamp_orientation_range(angles):
        """Angles wrapped into (-pi, pi]."""
        return angles - 2 * math.pi * torch.ceil((angles - math.pi) / (2 * math.pi))

    def adjust_direction(self, yaw, direction):
        """yaw: (B, 1) in (-pi/2, pi/2); direction: (B,) 1 = facing forward (yaw in [-pi/2, pi/2)), 0 = backward."""
        yaw = self.clamp_orientation_range(yaw)
        front = (yaw >= -math.pi / 2) & (yaw < math.pi / 2)
        flip = (direction.view_as(front) == 1) != front
        return self.clamp_orientation_range(yaw + math.pi * flip.to(yaw.dtype))
