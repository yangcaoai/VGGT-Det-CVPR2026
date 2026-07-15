# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import List, Tuple

import torch
from mmcv.cnn import Scale
from mmcv.ops import nms3d, nms3d_normal
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
# from mmengine.config import ConfigDict
from mmengine.model import BaseModule, bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.bbox_3d.utils import rotation_3d_in_axis
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from functools import partial
from projects.VGGTDet.detr3_models.helpers import GenericMLP
from projects.VGGTDet.detr3_models.utils.box_util import get_3d_box_batch_depth_tensor, generalized_box3d_iou
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

import time
from projects.VGGTDet.detr3_models.utils.votenet_pc_util import write_oriented_bbox, write_ply, write_ply_rgb, write_bbox
from mmdet3d.structures.ops.iou3d_calculator import axis_aligned_bbox_overlaps_3d
from mmcv.ops import diff_iou_rotated_3d
from Rotated_IoU.oriented_iou_loss import cal_giou_3d


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    # origin: point-cloud center.
    points = torch.stack(
        torch.meshgrid([
            torch.arange(n_voxels[0]),  # 40 W width, x
            torch.arange(n_voxels[1]),  # 40 D depth, y
            torch.arange(n_voxels[2])  # 16 H Height, z
        ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points

def swap_length_and_width(size_pred, pose_matrix, align_matrix):
    """
    支持 batch 大小 B > 1

    参数:
        size_pred: (B, 3, N)，表示 [长, 宽, 高]
        pose_matrix: (B, 4, 4)
        align_matrix: (B, 4, 4)

    返回:
        size_pred_new: (B, 3, N)，已调换或未调换的尺寸
    """

    # 计算复合变换矩阵 (B, 4, 4)
    align_matrix = torch.stack(align_matrix, dim=0).float()  # (B, 4, 4）
    pose_matrix = torch.stack(pose_matrix, dim=0).float()    # (B, 4, 4)
    transform_matrix = torch.matmul(align_matrix.float(), pose_matrix.float())  # (B,4,4)
    rotation_matrix = transform_matrix[:, :3, :3]  # (B,3,3)

    # 获取旋转矩阵每行最大值的索引
    abs_rot = torch.abs(rotation_matrix)  # (B,3,3)
    max_indices = torch.argmax(abs_rot, dim=2)  # (B,3) 每行最大值的列索引, 取出 目标交换索引

    size_pred_swapped = size_pred.clone()  # (B, 3, N)

    # 交换条件
    swap_cond = (max_indices[:, 0] > max_indices[:, 1])
    swap_mask = swap_cond
    
    # 获取需要交换的batch索引
    swap_indices = swap_mask.nonzero().squeeze(1)  # 1D索引张量
    
    # 执行交换 (只交换长和宽，保持高度不变)
    if swap_indices.numel() > 0:
        swap_batches = size_pred_swapped[swap_indices]  # (num_swap, 3, N)
        # 交换长和宽: 
        swapped_sizes = torch.stack([
            swap_batches[:, 1, :],  
            swap_batches[:, 0, :],  
            swap_batches[:, 2, :]   # 高不变
        ], dim=1)
        
        # 更新需要交换的batch
        size_pred_swapped[swap_indices] = swapped_sizes

    return size_pred_swapped



@MODELS.register_module()
class VGGTDetHead(BaseModule):
    r"""`ImVoxelNet<https://arxiv.org/abs/2106.01178>`_ head for indoor
    datasets.

    Args:
        n_classes (int): Number of classes.
        n_levels (int): Number of feature levels.
        n_channels (int): Number of channels in input tensors.
        n_reg_outs (int): Number of regression layer channels.
        pts_assign_threshold (int): Min number of location per box to
            be assigned with.
        pts_center_threshold (int): Max number of locations per box to
            be assigned with.
        center_loss (dict, optional): Config of centerness loss.
            Default: dict(type='CrossEntropyLoss', use_sigmoid=True).
        bbox_loss (dict, optional): Config of bbox loss.
            Default: dict(type='RotatedIoU3DLoss').
        cls_loss (dict, optional): Config of classification loss.
            Default: dict(type='FocalLoss').
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 n_classes: int,
                 n_levels: int,
                 n_channels: int,
                 n_reg_outs: int,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 prior_generator: ConfigType,
                #  center_loss: ConfigType = dict(
                #      type='mmdet.CrossEntropyLoss', use_sigmoid=True),
                #  bbox_loss: ConfigType = dict(type='RotatedIoU3DLoss'),
                 cls_loss: ConfigType = dict(type='mmdet.FocalLoss', use_sigmoid=True),
                 objness_loss: ConfigType = dict(type='mmdet.FocalLoss', use_sigmoid=True),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 mlp_dropout=0.3,
                 matcher_cost_weights={'cls':1.0, 'center':0.0, 'obj_ness':0.0, 'giou':2.0},
                 loss_weights={'center_loss': 5.0, 'size_loss': 1.0,
                    'cls_loss': 1.0,
                    'objness_loss': 1.0,
                    'iou_loss': 1.0,
                    'not_objness_loss': 0.25},
                learn_center_diff=False,
                visualize_3d_bbox=False,
                visualize_2d_bbox=False,
                visualize_path=None,
                if_v2_head=False,
                if_project_frist_frame_back=False,
                matcher='one2one',
                matcher_iou_thres=0.25,
                matcher_max_dynamic_samples=10,
                visual_cfg: OptConfigType = None,
                if_swap_length_and_width=False,
                    ):
        super(VGGTDetHead, self).__init__(init_cfg)
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.n_reg_outs = n_reg_outs
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.prior_generator = TASK_UTILS.build(prior_generator)
        # self.center_loss = MODELS.build(center_loss)
        # self.bbox_loss = MODELS.build(bbox_loss)
        class_weights = torch.ones((self.n_classes+1), device='cuda') * 1.0
        class_weights[-1] = loss_weights['not_objness_loss']
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights) #MODELS.build(cls_loss)
        self.objness_loss = MODELS.build(objness_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if if_v2_head:
            self.mlp_func = partial(
                GenericMLP,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                hidden_dims=[n_channels, n_channels//2, n_channels//4, n_channels//8],
                dropout=mlp_dropout,
                input_dim=n_channels,
            )
        else:
            self.mlp_func = partial(
                GenericMLP,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                hidden_dims=[n_channels, n_channels],
                dropout=mlp_dropout,
                input_dim=n_channels,
            )
        self._init_layers(n_channels, n_reg_outs, n_classes, n_levels)
        assert matcher in ['one2one', 'one2more']
        if matcher == 'one2one':
            self.matcher = UnifiedMatcher(cost_weights=matcher_cost_weights)
        elif matcher == 'one2more':
            self.matcher = UnifiedMatcherMoreThanOne(cost_weights=matcher_cost_weights, matcher_iou_thres=matcher_iou_thres, matcher_max_dynamic_samples=matcher_max_dynamic_samples)
        self.loss_weights = loss_weights
        self.learn_center_diff = learn_center_diff
        self.visualize_3d_bbox = visualize_3d_bbox
        self.visualize_2d_bbox = visualize_2d_bbox
        self.visualize_path = visualize_path
        self.if_project_frist_frame_back = if_project_frist_frame_back
        if self.visualize_3d_bbox:
            self.test_cfg['score_thr'] = visual_cfg['score_thr']
            self.test_cfg['iou_thr'] = visual_cfg['iou_thr']
        self.if_swap_length_and_width = if_swap_length_and_width

    def _init_layers(self, n_channels, n_reg_outs, n_classes, n_levels):
        """Initialize neural network layers of the head."""
        # self.conv_center = nn.Conv3d(n_channels, 1, 3, padding=1, bias=False)

        self.center_head = self.mlp_func(output_dim=3)
        # self.conv_reg = nn.Conv3d(
        #     n_channels, n_reg_outs, 3, padding=1, bias=False)
        self.size_head = self.mlp_func(output_dim=3)

        self.semcls_head = self.mlp_func(output_dim=n_classes+1) # foreground categories
        # self.objness_head = self.mlp_func(output_dim=1) # objectness
        # self.conv_cls = nn.Conv3d(n_channels, n_classes, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(n_levels)])

    # def init_weights(self):
    #     """Initialize all layer weights."""
    #     normal_init(self.conv_center, std=.01)
    #     normal_init(self.conv_reg, std=.01)
    #     normal_init(self.conv_cls, std=.01, bias=bias_init_with_prob(.01))
    def project_the_first_frame_back(self, x: Tensor, pose_matrix, axis_align_matrix):
        batch_size, _, num_boxes = x.shape
        # 将 pose_matrix 堆叠成 Tensor，方便批量计算
        pose_matrix = torch.stack(pose_matrix, dim=0).to(x.device, dtype=x.dtype)  # [16, 4, 4]
        axis_align_matrix = torch.stack(axis_align_matrix, dim=0).to(x.device, dtype=x.dtype)
        # Step 1: 将 x 转换为齐次坐标，增加一维
        ones = torch.ones(batch_size, 1, num_boxes, device=x.device)  # [16, 1, 256]
        x_homogeneous = torch.cat([x, ones], dim=1)  # [16, 4, 256]

        # Step 2: 使用 pose_matrix 对齐次坐标进行变换
        # 矩阵乘法：pose_matrix [16, 4, 4] 和 x_homogeneous [16, 4, 256]
        x_global_homogeneous = torch.bmm(pose_matrix, x_homogeneous)  # [16, 4, 256]
        x_global_homogeneous = torch.bmm(axis_align_matrix, x_global_homogeneous)
        # Step 3: 转换回 3D 坐标（去掉齐次坐标的最后一维）
        # x', y', z' = x', y', z' / w
        w = torch.clamp(x_global_homogeneous[:, 3:4, :], min=1e-8)  # 避免 w 为 0
        x_global = x_global_homogeneous[:, :3, :] / w
        # x_global = x_global_homogeneous[:, :3, :] / x_global_homogeneous[:, 3:4, :]  # [16, 3, 256]
        return x_global

    def _forward_single(self, x: Tensor, scale: Scale, query_xyz, pose_matrix, axis_align_matrix, avg_distance):
        """Forward pass per level.

        Args:
            x (Tensor): Per level 3d neck output tensor.
            scale (mmcv.cnn.Scale): Per level multiplication weight.

        Returns:
            tuple[Tensor]: Centerness, bbox and classification predictions.
        """
        if self.learn_center_diff:
            query_xyz = query_xyz.permute(0, 2, 1)
            if self.if_project_frist_frame_back:
                center_pred = self.project_the_first_frame_back(self.center_head(x)+query_xyz, pose_matrix, axis_align_matrix)
            else:
                center_pred = self.center_head(x)+query_xyz

        else:
            if self.if_project_frist_frame_back:
                center_pred = self.project_the_first_frame_back(self.center_head(x), pose_matrix, axis_align_matrix)
            else:
                center_pred = self.center_head(x)

            # avg_distance_tensor = torch.stack(avg_distance).unsqueeze(-1)
        size_pred = torch.exp(scale(self.size_head(x)))

        if self.if_swap_length_and_width:
            size_pred = swap_length_and_width(size_pred, pose_matrix, axis_align_matrix)

        return (center_pred, size_pred, #/ avg_distance_tensor,
                self.semcls_head(x)) # , self.objness_head(x)

    def forward(self, x, batch_inputs_dict, batch_data_samples):
        if 'query_xyz' in batch_inputs_dict.keys():
            return multi_apply(self._forward_single, x, self.scales, [batch_inputs_dict['query_xyz'] for _ in range(self.n_levels)], [batch_inputs_dict['pose_matrix'] for _ in range(self.n_levels)], [batch_inputs_dict['axis_align_matrix'] for _ in range(self.n_levels)], [batch_inputs_dict['avg_distance'] for _ in range(self.n_levels)]) 
        else:
            return multi_apply(self._forward_single, x, self.scales, [None for _ in range(self.n_levels)], [batch_inputs_dict['pose_matrix'] for _ in range(self.n_levels)], [batch_inputs_dict['axis_align_matrix'] for _ in range(self.n_levels)], [batch_inputs_dict['avg_distance'] for _ in range(self.n_levels)]) 

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList, batch_inputs_dict: dict,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # valid_pred = x[-1]
        outs = self(x, batch_inputs_dict, batch_data_samples) # x len: 8, every tensor shape: [bs, feat_dim, num_queries]

        if 'points' in batch_inputs_dict.keys():
            batch_input_points = batch_inputs_dict['points']
        else:
            # batch_input_points = [None for i in range(len(batch_input_metas))]
            batch_input_points = [None for i in range(len(batch_data_samples))]

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (batch_gt_instances_3d,
                              batch_input_metas, batch_input_points, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self,
                     center_preds: List[List[Tensor]],
                     size_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                    #  objness_preds: List[List[Tensor]],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_input_points,
                     batch_gt_instances_ignore: OptInstanceList = None,
                     **kwargs) -> dict:
        """Per scene loss function.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
                The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            valid_pred (Tensor): Valid mask prediction for all scenes.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_input_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        bboxes_3d.gravity_center: 物体的中心点
        bboxes_3d.tensor[:, 3:6]: 物体的长宽高

        Returns:
            dict: Centerness, bbox, and classification loss values.
        """
        # valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        center_losses, size_losses, cls_losses, objness_losses, giou_losses = [], [], [], [], []
        for i in range(len(batch_input_metas)):
            center_loss, size_loss, cls_loss, giou_loss = self._loss_by_feat_single(
                center_preds=[x[i] for x in center_preds],
                size_preds=[x[i] for x in size_preds],
                cls_preds=[x[i] for x in cls_preds],
                # objness_preds=[x[i] for x in objness_preds],
                # valid_preds=[x[i] for x in valid_preds],
                input_meta=batch_input_metas[i],
                gt_bboxes=batch_gt_instances_3d[i].bboxes_3d,
                gt_labels=batch_gt_instances_3d[i].labels_3d,
                input_points=batch_input_points[i])
            center_losses.append(center_loss)
            size_losses.append(size_loss)
            cls_losses.append(cls_loss)
            # objness_losses.append(objness_loss)
            giou_losses.append(giou_loss)

        return dict(
            center_loss=torch.mean(torch.stack(center_losses)),
            size_loss=torch.mean(torch.stack(size_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses)),
            # objness_loss=torch.mean(torch.stack(objness_losses)),
            giou_loss=torch.mean(torch.stack(giou_losses))
            )

    def _loss_by_feat_single(self, center_preds, size_preds, cls_preds, #objness_preds,
                              input_meta, gt_bboxes, gt_labels, input_points):
        """
        输入参数：
            center_preds: List[Tensor(3, N)] 各阶段中心预测
            size_preds: List[Tensor(3, N)] 各阶段尺寸预测
            cls_preds: List[Tensor(C, N)] 各阶段分类预测
            objness_preds: List[Tensor(1, N)] 各阶段存在性预测
            gt_bboxes: DepthInstance3DBoxes实例
            gt_labels: Tensor(M,)
        返回：
            center_loss, size_loss, giou_loss, cls_loss
        """
        # 拼接多阶段预测
        all_centers = torch.cat([c.t() for c in center_preds], dim=0)  # (Total_Pred, 3)
        all_sizes = torch.cat([s.t() for s in size_preds], dim=0)      # (Total_Pred, 3) 
        all_cls = torch.cat([c.t() for c in cls_preds], dim=0)         # (Total_Pred, C)
        # all_objness = torch.cat([c.t() for c in objness_preds], dim=0)
        


        # 提取GT数据
        gt_centers = gt_bboxes.gravity_center
        gt_sizes = gt_bboxes.tensor[:, 3:6]

        all_pred_indices = []
        all_gt_indices = []
        offset = 0  # 跨阶段的预测索引偏移量

        # time1 = time.time()
        # a = gt_sizes.cpu()
        for stage_idx in range(len(center_preds)):
            centers, sizes, cls_scores = center_preds[stage_idx].t(), size_preds[stage_idx].t(), cls_preds[stage_idx].t() #, objness_preds[stage_idx].t()
            cls_scores_softmax = F.softmax(cls_scores, dim=1)
            obj_scores = 1.0 - cls_scores_softmax[:, -1]
            n_predictions = centers.size(0)

            # 当前阶段匹配
            pred_indices, gt_indices = self.matcher._get_targets(
                centers, sizes, cls_scores, obj_scores,
                gt_centers, gt_sizes, gt_labels
            )
            
            # 调整预测索引的全局偏移
            all_pred_indices.append(pred_indices + offset)
            all_gt_indices.append(gt_indices)
            
            # 更新偏移量为下一阶段准备
            offset += n_predictions

        # time2 = time.time()
        # a = gt_sizes.cpu()

        pred_indices, gt_indices = torch.cat(all_pred_indices), torch.cat(all_gt_indices)
        # 执行全局匹配
        # pred_indices, gt_indices = self.matcher._get_targets(
        #     all_centers, all_sizes, all_cls, all_objness,
        #     gt_centers, gt_sizes, gt_labels
        # )

        # 提取匹配的预测和GT
        matched_centers = all_centers[pred_indices]
        matched_sizes = all_sizes[pred_indices]
        matched_cls = all_cls[pred_indices]
        # matched_objness = all_objness[pred_indices]
        matched_gt_centers = gt_centers[gt_indices]
        matched_gt_sizes = gt_sizes[gt_indices]
        matched_gt_labels = gt_labels[gt_indices]





        # time3 = time.time()
        # a = gt_sizes.cpu()

        # 计算各损失项
        # 中心L1损失
        center_loss = F.l1_loss(matched_centers, matched_gt_centers) * self.loss_weights['center_loss']
        
        # 尺寸L1损失
        size_loss = F.l1_loss(matched_sizes, matched_gt_sizes) * self.loss_weights['size_loss']
        
        # 分类交叉熵损失
        # cls_loss = self.cls_loss(matched_cls, matched_gt_labels, avg_facter=matched_cls.shape[0]) #??? 这里是shape[0]么？

        # time4 = time.time()
        # a = gt_sizes.cpu()
        # 分类Focal Loss，会自动算上sigmoid
        cls_target = torch.ones((all_centers.shape[0]), device=all_centers.device) * self.n_classes # 整数
        cls_target = cls_target.long()
        cls_target[pred_indices] = matched_gt_labels


        cls_loss = self.cls_loss(all_cls, cls_target) * self.loss_weights['cls_loss']


        # cls_targets_onehot = F.one_hot(
        #     matched_gt_labels, 
        #     num_classes=self.n_classes
        # ).float()

        # cls_loss = self.cls_loss(
        #     matched_cls, 
        #     cls_targets_onehot, avg_factor = (matched_cls.shape[0]+ 1e-16)
        # )  * self.loss_weights['cls_loss']
        
        # 存在性Focal Loss（二分类）
        # objness_target = torch.zeros_like(all_objness)
        # objness_target[pred_indices] = 1.0
        # objness_loss_weight = torch.ones_like(all_objness) * self.loss_weights['not_objness_loss']
        # objness_loss_weight[pred_indices] = self.loss_weights['objness_loss']

        # objness_loss_all = self.objness_loss(
        #     all_objness,  # (N,1)
        #     objness_target,  # (N,1)
        #     # avg_factor=matched_cls.shape[0],
        #     reduction_override='none'
        # )   * objness_loss_weight
        # objness_loss = torch.sum(objness_loss_all) / (torch.sum(objness_loss_weight) + 1e-16)

        # time5 = time.time()
        # a = gt_sizes.cpu()
        # GIoU损失


        pred_tp_bbox = self._center_size_pred_to_bbox(matched_centers, matched_sizes)
        gt_tp_bbox = self._center_size_pred_to_bbox(matched_gt_centers, matched_gt_sizes)

        giou = axis_aligned_bbox_overlaps_3d(pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0), mode='giou', is_aligned=True)

        # pred_corners = get_3d_box_batch_depth_tensor(
        #     matched_sizes.unsqueeze(0),
        #     torch.zeros(1, len(pred_indices), device=all_centers.device),
        #     matched_centers.unsqueeze(0)
        # )
        # gt_corners = get_3d_box_batch_depth_tensor(
        #     matched_gt_sizes.unsqueeze(0),
        #     torch.zeros(1, len(gt_indices), device=all_centers.device),
        #     matched_gt_centers.unsqueeze(0)
        # )

        # giou = generalized_box3d_iou(pred_corners, gt_corners, torch.tensor([len(gt_indices)]), rotated_boxes=(torch.sum(torch.abs(gt_bboxes.tensor[:, -1])) > 1e-16), needs_grad=(self.loss_weights['iou_loss'] > 0))
        giou_loss = (1.0 - giou).mean() * self.loss_weights['iou_loss']

        # time7 = time.time()
        # a = gt_sizes.cpu()

        # print('---------time------------')
        # print(time2-time1)
        # print(time3-time2)
        # print(time4-time3)
        # print(time5-time4)
        # print(time6-time5)
        # print(time7-time6)

        return center_loss, size_loss, cls_loss, giou_loss

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList, batch_inputs_dict, 
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 6.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        # valid_pred = x[-1]
        outs = self(x, batch_inputs_dict, batch_data_samples) 
        predictions = self.predict_by_feat(
            *outs,
            batch_input_metas=batch_input_metas,
            rescale=rescale, batch_inputs_dict=batch_inputs_dict, batch_data_samples=batch_data_samples)
        return predictions

    def predict_by_feat(self, center_preds: List[List[Tensor]],
                     size_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                        batch_input_metas: List[dict], batch_inputs_dict: dict, batch_data_samples,
                        **kwargs) -> List[InstanceData]:
        """Generate boxes for all scenes.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            valid_pred (Tensor): Valid mask prediction for all scenes.
            batch_input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[tuple[Tensor]]: Predicted bboxes, scores, and labels for
                all scenes.
        """
        # valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        results = []
        if 'points' in batch_inputs_dict.keys():
            batch_input_points = batch_inputs_dict['points']
        else:
            batch_input_points = [None for i in range(len(batch_input_metas))]
        if 'points_first_axis' in batch_inputs_dict.keys():
            batch_points_first_axis = batch_inputs_dict['points_first_axis']
        else:
            batch_points_first_axis = [None for i in range(len(batch_input_metas))]
        for i in range(len(batch_input_metas)):
            results.append(
                self._predict_by_feat_single(
                center_preds=[x[i] for x in center_preds],
                size_preds=[x[i] for x in size_preds],
                cls_preds=[x[i] for x in cls_preds],
                # objness_preds=[x[i] for x in objness_preds],
                # valid_preds=[x[i] for x in valid_preds],
                input_meta=batch_input_metas[i],
                input_points=batch_input_points[i],
                data_samples=batch_data_samples[i],
                pose_matrix=batch_inputs_dict['pose_matrix'],
                points_first_axis=batch_points_first_axis))
        return results

    def _predict_by_feat_single(self, center_preds, size_preds, cls_preds,
                                input_meta: dict, input_points, data_samples, pose_matrix, points_first_axis) -> InstanceData:
        """Generate boxes for single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            valid_preds (tuple[Tensor]): Upsampled valid masks for all feature
                levels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor]: Predicted bounding boxes, scores and labels.
        """
        # all_centers = torch.cat([c.t() for c in center_preds], dim=0)  # (Total_Pred, 3)
        # all_sizes = torch.cat([s.t() for s in size_preds], dim=0)      # (Total_Pred, 3) 
        # all_cls = torch.cat([c.t() for c in cls_preds], dim=0)         # (Total_Pred, C)
        # all_objness = torch.cat([c.t() for c in objness_preds], dim=0)
    
        # featmap_sizes = [featmap.size()[-3:] for featmap in center_preds]
        # points = self._get_points(
        #     featmap_sizes=featmap_sizes,
        #     origin=input_meta['lidar2img']['origin'],
        #     device=center_preds[0].device)
        
        # for center_pred, size_pred, cls_pred, objness_pred in zip(
        #         center_preds, size_preds, cls_preds, objness_preds):
        #     center_pred = center_pred.permute(1, 2, 3, 0).reshape(-1, 1)
        #     bbox_pred = bbox_pred.permute(1, 2, 3,
        #                                   0).reshape(-1, bbox_pred.shape[0])
        #     cls_pred = cls_pred.permute(1, 2, 3,
        #                                 0).reshape(-1, cls_pred.shape[0])
        #     valid_pred = valid_pred.permute(1, 2, 3, 0).reshape(-1, 1)

        mlvl_bboxes, mlvl_scores = [], []
        for stage_idx in range(len(center_preds)):
            centers, sizes, cls_scores = center_preds[stage_idx].t(), size_preds[stage_idx].t(), cls_preds[stage_idx].t()
            cls_scores = F.softmax(cls_scores, dim=1)
            objectness = 1 - cls_scores[:, -1]
            scores = cls_scores[:, :-1] * objectness.unsqueeze(-1)

            # scores = cls_pred.sigmoid() * center_pred.sigmoid() * valid_pred
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                # bbox_pred = bbox_pred[ids]
                centers = centers[ids]
                sizes = sizes[ids]
                scores = scores[ids]
                # point = point[ids]

            bboxes = self._center_size_pred_to_bbox(centers, sizes) # 输出是tp box
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes_after_nms, scores, labels = self._nms(bboxes, scores, input_meta) # bboxes(n_box, 6) (x_center, y_center, z_center, w, h, z)

        bboxes = input_meta['box_type_3d'](
            bboxes_after_nms, box_dim=6, with_yaw=False, origin=(.5, .5, .5))
        
       # gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes
        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        # results = InstanceData()
        # results.bboxes_3d = data_samples.gt_instances_3d.bboxes_3d
        # results.scores_3d = torch.ones_like(data_samples.gt_instances_3d.labels_3d)
        # results.labels_3d = data_samples.gt_instances_3d.labels_3d

        if self.visualize_3d_bbox:
            keep_indices = torch.where(scores > -1)[0]  # 返回满足条件的索引
            selected_boxes = bboxes_after_nms[keep_indices]
            
            gt_boxes = data_samples.gt_instances_3d.bboxes_3d
            gt_boxes = torch.cat(
                (gt_boxes.gravity_center, gt_boxes.tensor[:, 3:6]), dim=1) # gt_bboxes.gravity_center: 中心点，gt_bboxes.tensor[:, 3:6]： 长宽高
            scene_path = input_meta['img_path'][0].split('/')[-2]

# -           max_giou, max_gt_box_idx, max_pred_box_idx = self.find_max_iou_from_center_size_boxes(bboxes_after_nms, gt_boxes)
# -           if isinstance(max_pred_box_idx, torch.Tensor):
# -               max_pred_box_idx = max_pred_box_idx.item()
            write_bbox(gt_boxes.cpu().numpy(), self.visualize_path+'/'+'%s_gt_boxes.ply' % scene_path)

# -           max_gt_box_idx_tmp = max_gt_box_idx[:, max_pred_box_idx]
# -           if isinstance(max_gt_box_idx_tmp, torch.Tensor):
# -               max_gt_box_idx_tmp = max_gt_box_idx_tmp.item()

            # write_bbox(gt_boxes[max_gt_box_idx_tmp].unsqueeze(0).cpu().numpy(), self.visualize_path+'/'+'%s_max_iou_gt_boxes.ply' % scene_path)

            write_bbox(bboxes_after_nms.cpu().numpy(), self.visualize_path+'/'+'%s_pred_boxes.ply' % scene_path)
            # write_bbox(bboxes_after_nms[max_pred_box_idx].unsqueeze(0).cpu().numpy(), self.visualize_path+'/'+'%s_max_iou_%f_boxes.ply' % (scene_path, max_giou))
            if input_points is not None:
                write_ply_rgb(input_points.cpu().numpy(), self.visualize_path+'/'+f'{scene_path}_gt_points.ply')
                write_ply_rgb(points_first_axis[0].cpu().numpy(), self.visualize_path+'/'+f'{scene_path}_gt_points_first_axis.ply')

        return results

    def find_max_iou_from_center_size_boxes(self, boxes1, boxes2):
        # pred_corners = get_3d_box_batch_depth_tensor(
        #     boxes1[:, :3].unsqueeze(0),
        #     torch.zeros(1, boxes1.shape[0], device=boxes1.device),
        #     boxes1[:, 3:6].unsqueeze(0)
        # )
        # gt_corners = get_3d_box_batch_depth_tensor(
        #     boxes2[:, :3].unsqueeze(0),
        #     torch.zeros(1, boxes2.shape[0], device=boxes1.device),
        #     boxes2[:, 3:6].unsqueeze(0)
        # )

        # time6 = time.time()
        # a = gt_sizes.cpu()
        # giou = generalized_box3d_iou(pred_corners, gt_corners, torch.tensor([boxes1.shape[0]]), rotated_boxes=False, needs_grad=False)
        # giou_max_gt, max_gt_box_idx = torch.max(giou, axis=2)
        # max_giou, max_pred_box_idx = torch.max(giou_max_gt, axis=1)

        boxes1_tp = self._center_size_pred_to_bbox(boxes1[:, :3], boxes1[:, 3:6])
        boxes2_tp = self._center_size_pred_to_bbox(boxes2[:, :3], boxes2[:, 3:6])
        giou_2 = axis_aligned_bbox_overlaps_3d(boxes1_tp.unsqueeze(0), boxes2_tp.unsqueeze(0), mode='giou') # giou
        giou_max_gt, max_gt_box_idx = torch.max(giou_2, axis=2)
        max_giou, max_pred_box_idx = torch.max(giou_max_gt, axis=1)
        assert max_giou <= 1 and  max_giou >= -1
        return max_giou, max_gt_box_idx, max_pred_box_idx



    @staticmethod
    def _upsample_valid_preds(valid_pred, features):
        """Upsample valid mask predictions.

        Args:
            valid_pred (Tensor): Valid mask prediction.
            features (Tensor): Feature tensor.

        Returns:
            tuple[Tensor]: Upsampled valid masks for all feature levels.
        """
        return [
            nn.Upsample(size=x.shape[-3:],
                        mode='trilinear')(valid_pred).round().bool()
            for x in features
        ]

    @torch.no_grad()
    def _get_points(self, featmap_sizes, origin, device):
        mlvl_points = []
        tmp_voxel_size = [.16, .16, .2]
        for i, featmap_size in enumerate(featmap_sizes):
            mlvl_points.append(
                get_points(
                    n_voxels=torch.tensor(featmap_size),
                    voxel_size=torch.tensor(tmp_voxel_size) * (2**i),
                    origin=torch.tensor(origin)).reshape(3, -1).transpose(
                        0, 1).to(device))
        return mlvl_points

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        return torch.stack([
            points[:, 0] - bbox_pred[:, 0], points[:, 1] - bbox_pred[:, 2], # boxes[..., 0] - boxes[..., 3] / 2, boxes[..., 1] - boxes[..., 4] / 2
            points[:, 2] - bbox_pred[:, 4], points[:, 0] + bbox_pred[:, 1], # boxes[..., 2] - boxes[..., 5] / 2, boxes[..., 0] + boxes[..., 3] / 2
            points[:, 1] + bbox_pred[:, 3], points[:, 2] + bbox_pred[:, 5]  # boxes[..., 1] + boxes[..., 4] / 2, boxes[..., 2] + boxes[..., 5] / 2
        ], -1)

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0]/2.0, centers[:, 1] - sizes[:, 1]/2.0,
            centers[:, 2] - sizes[:, 2]/2.0, centers[:, 0] + sizes[:, 0]/2.0,
            centers[:, 1] + sizes[:, 1]/2.0, centers[:, 2] + sizes[:, 2]/2.0
        ], -1)

    def _bbox_pred_to_loss(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    # The function is directly copied from FCAF3DHead.
    @staticmethod
    def _get_face_distances(points, boxes):
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        dx_min = points[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - points[..., 0]
        dy_min = points[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - points[..., 1]
        dz_min = points[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - points[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)

    @staticmethod
    def _get_centerness(face_distances):
        """Compute point centerness w.r.t containing box.

        Args:
            face_distances (Tensor): Face distances of shape (B, N, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).

        Returns:
            Tensor: Centerness of shape (B, N).
        """
        x_dims = face_distances[..., [0, 1]]
        y_dims = face_distances[..., [2, 3]]
        z_dims = face_distances[..., [4, 5]]
        centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
            y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
            z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
        return torch.sqrt(centerness_targets)

    # @torch.no_grad()
    # def _get_targets(self, center_preds, size_preds, cls_preds, objness_preds, gt_bboxes, gt_labels):
    #     """Compute targets for final locations for a single scene.

    #     Args:
    #         points (list[Tensor]): Final locations for all levels.
    #         gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
    #         gt_labels (Tensor): Ground truth labels.

    #     Returns:
    #         tuple[Tensor]: Centerness, bbox and classification
    #             targets for all locations.
    #     """
    #     float_max = 1e8
    #     expanded_scales = [
    #         points[i].new_tensor(i).expand(len(points[i])).to(gt_labels.device)
    #         for i in range(len(points))
    #     ]
    #     points = torch.cat(points, dim=0).to(gt_labels.device) # (N1+N2+N3, 3)
    #     scales = torch.cat(expanded_scales, dim=0)

    #     # below is based on FCOSHead._get_target_single
    #     n_points = len(points)
    #     n_boxes = len(gt_bboxes)
    #     volumes = gt_bboxes.volume.to(points.device)
    #     volumes = volumes.expand(n_points, n_boxes).contiguous()
    #     gt_bboxes = torch.cat(
    #         (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:6]), dim=1) # gt_bboxes.gravity_center: 中心点，gt_bboxes.tensor[:, 3:6]： 长宽高
    #     gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 6)
    #     expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
    #     bbox_targets = self._get_face_distances(expanded_points, gt_bboxes) # (N1+N2+N3, n_bbox, 6) each point to bbox's 6 faces distance.

    #     # condition1: inside a gt bbox
    #     inside_gt_bbox_mask = bbox_targets[..., :6].min(
    #         -1)[0] > 0  # skip angle

    #     # condition2: positive points per scale >= limit
    #     # calculate positive points per scale
    #     n_pos_points_per_scale = []
    #     for i in range(self.n_levels):
    #         n_pos_points_per_scale.append(
    #             torch.sum(inside_gt_bbox_mask[scales == i], dim=0))
    #     # find best scale
    #     n_pos_points_per_scale = torch.stack(n_pos_points_per_scale, dim=0) # (3, n_bbox). 3scales. each scale, how many points fit in each bbox
    #     lower_limit_mask = n_pos_points_per_scale < self.pts_assign_threshold
    #     # fix nondeterministic argmax for torch<1.7
    #     extra = torch.arange(self.n_levels, 0, -1).unsqueeze(1).expand(
    #         self.n_levels, n_boxes).to(lower_limit_mask.device)
    #     lower_index = torch.argmax(lower_limit_mask.int() * extra, dim=0) - 1
    #     lower_index = torch.where(lower_index < 0,
    #                               torch.zeros_like(lower_index), lower_index)
    #     all_upper_limit_mask = torch.all(
    #         torch.logical_not(lower_limit_mask), dim=0)
    #     best_scale = torch.where(
    #         all_upper_limit_mask,
    #         torch.ones_like(all_upper_limit_mask) * self.n_levels - 1,
    #         lower_index)
    #     # keep only points with best scale
    #     best_scale = torch.unsqueeze(best_scale, 0).expand(n_points, n_boxes)
    #     scales = torch.unsqueeze(scales, 1).expand(n_points, n_boxes)
    #     inside_best_scale_mask = best_scale == scales

    #     # condition3: limit topk locations per box by centerness
    #     centerness = self._get_centerness(bbox_targets) # (N1+N2+N3, n_bbox)
    #     centerness = torch.where(inside_gt_bbox_mask, centerness,
    #                              torch.ones_like(centerness) * -1)
    #     centerness = torch.where(inside_best_scale_mask, centerness,
    #                              torch.ones_like(centerness) * -1)
    #     top_centerness = torch.topk(
    #         centerness, self.pts_center_threshold + 1, dim=0).values[-1]
    #     inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

    #     # if there are still more than one objects for a location,
    #     # we choose the one with minimal area
    #     volumes = torch.where(inside_gt_bbox_mask, volumes,
    #                           torch.ones_like(volumes) * float_max)
    #     volumes = torch.where(inside_best_scale_mask, volumes,
    #                           torch.ones_like(volumes) * float_max)
    #     volumes = torch.where(inside_top_centerness_mask, volumes,
    #                           torch.ones_like(volumes) * float_max)
    #     min_area, min_area_inds = volumes.min(dim=1)

    #     labels = gt_labels[min_area_inds]
    #     labels = torch.where(min_area == float_max,
    #                          torch.ones_like(labels) * -1, labels)
    #     bbox_targets = bbox_targets[range(n_points), min_area_inds]
    #     centerness_targets = self._get_centerness(bbox_targets)

    #     return centerness_targets, self._bbox_pred_to_bbox(
    #         points, bbox_targets), labels

    def _nms(self, bboxes, scores, img_meta): # bbox is 6-dim. (x_min, y_min, z_min, x_max, y_max, z_max)
        scores, labels = scores.max(dim=1)
        ids = scores > self.test_cfg.score_thr
        bboxes = bboxes[ids]
        scores = scores[ids]
        labels = labels[ids]
        ids = self.aligned_3d_nms(bboxes, scores, labels,
                                  self.test_cfg.iou_thr)
        bboxes = bboxes[ids]
        bboxes = torch.stack(
            ((bboxes[:, 0] + bboxes[:, 3]) / 2.,
             (bboxes[:, 1] + bboxes[:, 4]) / 2.,
             (bboxes[:, 2] + bboxes[:, 5]) / 2., bboxes[:, 3] - bboxes[:, 0],
             bboxes[:, 4] - bboxes[:, 1], bboxes[:, 5] - bboxes[:, 2]),
            dim=1) # (convert to (x_center, y_center, z_center, w, h, z))
        return bboxes, scores[ids], labels[ids]

    @staticmethod
    def aligned_3d_nms(boxes, scores, classes, thresh):
        """3d nms for aligned boxes.

        Args:
            boxes (torch.Tensor): Aligned box with shape [n, 6].
            scores (torch.Tensor): Scores of each box.
            classes (torch.Tensor): Class of each box.
            thresh (float): Iou threshold for nms.

        Returns:
            torch.Tensor: Indices of selected boxes.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        z1 = boxes[:, 2]
        x2 = boxes[:, 3]
        y2 = boxes[:, 4]
        z2 = boxes[:, 5]
        area = (x2 - x1) * (y2 - y1) * (z2 - z1)
        zero = boxes.new_zeros(1, )

        score_sorted = torch.argsort(scores)
        pick = []
        while (score_sorted.shape[0] != 0):
            last = score_sorted.shape[0]
            i = score_sorted[-1]
            pick.append(i)

            xx1 = torch.max(x1[i], x1[score_sorted[:last - 1]])
            yy1 = torch.max(y1[i], y1[score_sorted[:last - 1]])
            zz1 = torch.max(z1[i], z1[score_sorted[:last - 1]])
            xx2 = torch.min(x2[i], x2[score_sorted[:last - 1]])
            yy2 = torch.min(y2[i], y2[score_sorted[:last - 1]])
            zz2 = torch.min(z2[i], z2[score_sorted[:last - 1]])
            classes1 = classes[i]
            classes2 = classes[score_sorted[:last - 1]]
            inter_l = torch.max(zero, xx2 - xx1)
            inter_w = torch.max(zero, yy2 - yy1)
            inter_h = torch.max(zero, zz2 - zz1)

            inter = inter_l * inter_w * inter_h
            iou = inter / (area[i] + area[score_sorted[:last - 1]] - inter)
            iou = iou * (classes1 == classes2).float()
            score_sorted = score_sorted[torch.nonzero(
                iou <= thresh, as_tuple=False).flatten()]

        indices = boxes.new_tensor(pick, dtype=torch.long)
        return indices



class UnifiedMatcher(nn.Module):
    def __init__(self, cost_weights={'cls':1.0, 'center':0.0, 'obj_ness':0.0, 'giou':2.0}):
        super().__init__()
        self.cost_weights = cost_weights
        
    @torch.no_grad()
    def _get_targets(self, all_centers, all_sizes, all_cls, all_objness, gt_centers, gt_sizes, gt_labels):
        """
        批量匹配所有预测框（多阶段联合处理）
        输入：
            all_centers: (Total_Pred, 3) 所有阶段的中心预测拼接结果
            all_sizes: (Total_Pred, 3) 所有阶段的尺寸预测拼接结果
            all_cls: (Total_Pred, C) 所有阶段的分类预测拼接结果
            all_objness: (Total_Pred, 1) 所有阶段的objectness预测拼接结果
            gt_centers: (M, 3)
            gt_sizes: (M, 3)
            gt_labels: (M,)
        返回：
            pred_indices: 匹配的预测索引
            gt_indices: 匹配的真实索引
        """
        # # 生成预测框角点
        # pred_corners = get_3d_box_batch_depth_tensor(
        #     all_sizes.unsqueeze(0), 
        #     torch.zeros(1, all_centers.size(0), device=all_centers.device),
        #     all_centers.unsqueeze(0)
        # )  # (1, Total_Pred, 8, 3)

        # # 生成真实框角点
        # gt_corners = get_3d_box_batch_depth_tensor(
        #     gt_sizes.unsqueeze(0),
        #     torch.zeros(1, gt_centers.size(0), device=gt_centers.device),
        #     gt_centers.unsqueeze(0)
        # )  # (1, M, 8, 3)

        if all_objness.dim() == 1:
            all_objness = all_objness.unsqueeze(-1) 

        pred_tp_bbox = self._center_size_pred_to_bbox(all_centers, all_sizes)
        gt_tp_bbox = self._center_size_pred_to_bbox(gt_centers, gt_sizes)

        with torch.no_grad():
            giou = axis_aligned_bbox_overlaps_3d(pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0), mode='giou')
            assert giou.shape[0] == 1
            giou = giou.squeeze(0)

        # 计算成本矩阵
        cost_class = -all_cls.sigmoid()[:, gt_labels]  # (Total_Pred, M)
        cost_center = torch.cdist(all_centers, gt_centers, p=1)  # (Total_Pred, M)
        cost_objness = -all_objness.sigmoid()       # (Total_Pred, M)
        # giou = generalized_box3d_iou(pred_corners, gt_corners, torch.tensor([gt_centers.size(0)]))[0]  # (Total_Pred, M)
        cost_giou =  -giou

        # 加权总成本
        total_cost = (
            self.cost_weights['cls'] * cost_class +
            self.cost_weights['center'] * cost_center +
            self.cost_weights['obj_ness'] * cost_objness +
            self.cost_weights['giou'] * cost_giou
        )

        # 全局最优匹配
        pred_indices, gt_indices = linear_sum_assignment(total_cost.cpu().numpy())
        return torch.from_numpy(pred_indices).long().to(all_centers.device), torch.from_numpy(gt_indices).long().to(all_centers.device)

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0]/2.0, centers[:, 1] - sizes[:, 1]/2.0,
            centers[:, 2] - sizes[:, 2]/2.0, centers[:, 0] + sizes[:, 0]/2.0,
            centers[:, 1] + sizes[:, 1]/2.0, centers[:, 2] + sizes[:, 2]/2.0
        ], -1)


class UnifiedMatcherMoreThanOne(nn.Module):
    def __init__(self, cost_weights={'cls': 1.0, 'center': 0.0, 'obj_ness': 0.0, 'giou': 2.0}, matcher_iou_thres=0.25, matcher_max_dynamic_samples=10):
        super().__init__()
        self.cost_weights = cost_weights
        self.iou_threshold = matcher_iou_thres,
        self.matcher_max_dynamic_samples = matcher_max_dynamic_samples

    @torch.no_grad()
    def _get_targets(self, all_centers, all_sizes, all_cls, all_objness, gt_centers, gt_sizes, gt_labels):
        if all_objness.dim() == 1:
            all_objness = all_objness.unsqueeze(-1) 
        
        # 边界框转换
        pred_tp_bbox = self._center_size_pred_to_bbox(all_centers, all_sizes)
        gt_tp_bbox = self._center_size_pred_to_bbox(gt_centers, gt_sizes)

        # GIoU计算
        with torch.no_grad():
            giou = axis_aligned_bbox_overlaps_3d(pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0), mode='giou')
            assert giou.shape[0] == 1
            giou = giou.squeeze(0)  # (Total_Pred, M)

        # 成本矩阵计算
        cost_class = -all_cls.sigmoid()[:, gt_labels]
        cost_center = torch.cdist(all_centers, gt_centers, p=1)
        cost_objness = -all_objness.sigmoid()
        cost_giou = -giou

        # 加权总成本
        total_cost = (
            self.cost_weights['cls'] * cost_class +
            self.cost_weights['center'] * cost_center +
            self.cost_weights['obj_ness'] * cost_objness +
            self.cost_weights['giou'] * cost_giou
        )

        # 匈牙利算法匹配
        pred_indices, gt_indices = linear_sum_assignment(total_cost.cpu().numpy())
        pred_indices = torch.from_numpy(pred_indices).long().to(all_centers.device)
        gt_indices = torch.from_numpy(gt_indices).long().to(all_centers.device)

        # ================= 动态正样本分配（关键修改）=================
        # 初始化已使用的预测框掩码（包含匈牙利匹配的预测框）
        used_pred_mask = torch.zeros(giou.size(0), dtype=torch.bool, device=giou.device)
        used_pred_mask[pred_indices] = True  # 标记匈牙利匹配的预测框

        # 找到所有满足 IoU 阈值的预测框-真实框对
        iou_mask = giou > self.iou_threshold[0]

        # 初始化动态匹配结果
        dynamic_preds = []
        dynamic_gts = []

        # 根据每个 GT box 的最大 IoU 值排序
        max_iou_per_gt = giou.max(dim=0).values  # 每个 GT box 的最大 IoU
        sorted_gt_indices = torch.argsort(max_iou_per_gt)  # 按 IoU 从小到大排序

        for gt_idx in sorted_gt_indices:
            # 获取当前真实框的候选预测框（排除已用预测框）
            candidate_mask = iou_mask[:, gt_idx] & ~used_pred_mask
            candidate_preds = torch.nonzero(candidate_mask, as_tuple=True)[0]
            
            if candidate_preds.numel() == 0:
                continue

            # 按 IoU 降序排序
            giou_values = giou[candidate_preds, gt_idx]
            
            if self.matcher_max_dynamic_samples < len(giou_values):
                _, topk_indices = torch.topk(giou_values, k=self.matcher_max_dynamic_samples)
                selected_preds = candidate_preds[topk_indices]
            else:
                selected_preds = candidate_preds

            # 记录选中的预测框
            dynamic_preds.append(selected_preds)
            dynamic_gts.append(torch.full_like(selected_preds, gt_idx))

            # 更新已用预测框掩码
            used_pred_mask[selected_preds] = True

        # 合并动态匹配结果
        if dynamic_preds:
            dynamic_preds = torch.cat(dynamic_preds)
            dynamic_gts = torch.cat(dynamic_gts)
        else:
            dynamic_preds = torch.empty(0, dtype=torch.long, device=giou.device)
            dynamic_gts = torch.empty(0, dtype=torch.long, device=giou.device)

        # 合并匈牙利匹配和动态匹配结果
        combined_preds = torch.cat([pred_indices, dynamic_preds])
        combined_gts = torch.cat([gt_indices, dynamic_gts])

        return combined_preds, combined_gts

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0]/2.0, centers[:, 1] - sizes[:, 1]/2.0,
            centers[:, 2] - sizes[:, 2]/2.0, centers[:, 0] + sizes[:, 0]/2.0,
            centers[:, 1] + sizes[:, 1]/2.0, centers[:, 2] + sizes[:, 2]/2.0
        ], -1)
    

class UnifiedMatcherMoreThanOneWithAngle(nn.Module):
    def __init__(self, cost_weights={'cls': 1.0, 'center': 0.0, 'obj_ness': 0.0, 'giou': 2.0}, matcher_iou_thres=0.25, matcher_max_dynamic_samples=10):
        super().__init__()
        self.cost_weights = cost_weights
        self.iou_threshold = matcher_iou_thres,
        self.matcher_max_dynamic_samples = matcher_max_dynamic_samples

    @torch.no_grad()
    def batched_diff_iou_rotated_3d(self, pred_bboxes, gt_bboxes):
        """
        计算批量全对全旋转 3D IoU。

        参数:
        - pred_bboxes: (B, N1, 7) 预测边界框，B 是批量大小，N1 是每批预测框的数量，7 是旋转 3D 框的维度。
        - gt_bboxes: (B, N2, 7) 真值边界框，B 是批量大小，N2 是每批真值框的数量，7 是旋转 3D 框的维度。

        返回:
        - ious: (B, N1, N2) 每批次预测框和真值框的 IoU 矩阵。
        """
        B, N1, _ = pred_bboxes.shape  # 批量大小和预测框数量
        _, N2, _ = gt_bboxes.shape   # 真值框数量

        # 扩展维度以构造全对全组合
        pred_bboxes_expanded = pred_bboxes.unsqueeze(2).expand(-1, -1, N2, -1)  # (B, N1, N2, 7)
        gt_bboxes_expanded = gt_bboxes.unsqueeze(1).expand(-1, N1, -1, -1)      # (B, N1, N2, 7)

        # 将 (B, N1, N2, 7) 调整为 (B * N1 * N2, 7)
        combined_pred_bboxes = pred_bboxes_expanded.reshape(B, -1, 7)  # (B, N1 * N2, 7)
        combined_gt_bboxes = gt_bboxes_expanded.reshape(B, -1, 7)      # (B, N1 * N2, 7)

        # 调用 diff_iou_rotated_3d，输入形状必须为 (B, N, 7) 和 (B, N, 7)
        # ious = diff_iou_rotated_3d(combined_pred_bboxes, combined_gt_bboxes)  # (B, N1 * N2)
        ious = cal_giou_3d(combined_pred_bboxes, combined_gt_bboxes)  # (B, N1 * N2)
        # 将结果恢复为 (B, N1, N2)
        ious = ious.reshape(B, N1, N2)

        return ious

    @torch.no_grad()
    def _get_targets(self, all_centers, all_sizes, all_cls, all_objness, all_angles, gt_centers, gt_sizes, gt_labels, gt_angles):
        if all_objness.dim() == 1:
            all_objness = all_objness.unsqueeze(-1) 

        if gt_angles.dim() == 1:
            gt_angles = gt_angles.unsqueeze(-1)

        pred_bboxes = torch.cat([all_centers, all_sizes, all_angles], dim=1)

        gt_bboxes = torch.cat([gt_centers, gt_sizes, gt_angles], dim=1)

        with torch.no_grad():
            if pred_bboxes.dim() == 2:
                pred_bboxes = pred_bboxes.unsqueeze(0)
            if gt_bboxes.dim() == 2:
                gt_bboxes = gt_bboxes.unsqueeze(0)
            giou = self.batched_diff_iou_rotated_3d(pred_bboxes, gt_bboxes)
            if giou.shape[0] == 1:
                giou = giou.squeeze(0)
        # print(1)
        # 生成预测框角点
        # assert all_angles.shape[1] == 1
        # pred_corners = get_3d_box_batch_depth_tensor(
        #     all_sizes.unsqueeze(0),  # [1, 256, 3]
        #     all_angles[:, 0].unsqueeze(0), # [1, 256]
        #     all_centers.unsqueeze(0) # [1, 256, 3]
        # )  # (1, Total_Pred, 8, 3)

        # # 生成真实框角点
        # gt_corners = get_3d_box_batch_depth_tensor(
        #     gt_sizes.unsqueeze(0), # [1, 20, 3]
        #     gt_angles.unsqueeze(0), # [1, 20]
        #     gt_centers.unsqueeze(0) # [1, 20, 3]
        # )  # (1, M, 8, 3)

        # 边界框转换
        # pred_tp_bbox = self._center_size_pred_to_bbox(all_centers, all_sizes)
        # gt_tp_bbox = self._center_size_pred_to_bbox(gt_centers, gt_sizes)

        # GIoU计算
        # with torch.no_grad():
        #     # giou = axis_aligned_bbox_overlaps_3d(pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0), mode='giou')
        #     # assert giou.shape[0] == 1
        #     # giou = giou.squeeze(0)  # (Total_Pred, M)
        #     giou = generalized_box3d_iou(pred_corners, gt_corners, torch.tensor([gt_centers.size(0)]), rotated_boxes=True, needs_grad=False)[0] 

        # 成本矩阵计算
        cost_class = -all_cls.sigmoid()[:, gt_labels]
        cost_center = torch.cdist(all_centers, gt_centers, p=1)
        cost_objness = -all_objness.sigmoid()
        cost_giou = -giou

        # 加权总成本
        total_cost = (
            self.cost_weights['cls'] * cost_class +
            self.cost_weights['center'] * cost_center +
            self.cost_weights['obj_ness'] * cost_objness +
            self.cost_weights['giou'] * cost_giou
        )

        # 匈牙利算法匹配
        pred_indices, gt_indices = linear_sum_assignment(total_cost.cpu().numpy())
        pred_indices = torch.from_numpy(pred_indices).long().to(all_centers.device)
        gt_indices = torch.from_numpy(gt_indices).long().to(all_centers.device)

        # ================= 动态正样本分配（关键修改）=================
        # 初始化已使用的预测框掩码（包含匈牙利匹配的预测框）
        used_pred_mask = torch.zeros(giou.size(0), dtype=torch.bool, device=giou.device)
        used_pred_mask[pred_indices] = True  # 标记匈牙利匹配的预测框

        # 找到所有满足 IoU 阈值的预测框-真实框对
        iou_mask = giou > self.iou_threshold[0]

        # 初始化动态匹配结果
        dynamic_preds = []
        dynamic_gts = []

        # 根据每个 GT box 的最大 IoU 值排序
        max_iou_per_gt = giou.max(dim=0).values  # 每个 GT box 的最大 IoU
        sorted_gt_indices = torch.argsort(max_iou_per_gt)  # 按 IoU 从小到大排序

        for gt_idx in sorted_gt_indices:
            # 获取当前真实框的候选预测框（排除已用预测框）
            candidate_mask = iou_mask[:, gt_idx] & ~used_pred_mask
            candidate_preds = torch.nonzero(candidate_mask, as_tuple=True)[0]
            
            if candidate_preds.numel() == 0:
                continue

            # 按 IoU 降序排序
            giou_values = giou[candidate_preds, gt_idx]
            
            if self.matcher_max_dynamic_samples < len(giou_values):
                _, topk_indices = torch.topk(giou_values, k=self.matcher_max_dynamic_samples)
                selected_preds = candidate_preds[topk_indices]
            else:
                selected_preds = candidate_preds

            # 记录选中的预测框
            dynamic_preds.append(selected_preds)
            dynamic_gts.append(torch.full_like(selected_preds, gt_idx))

            # 更新已用预测框掩码
            used_pred_mask[selected_preds] = True

        # 合并动态匹配结果
        if dynamic_preds:
            dynamic_preds = torch.cat(dynamic_preds)
            dynamic_gts = torch.cat(dynamic_gts)
        else:
            dynamic_preds = torch.empty(0, dtype=torch.long, device=giou.device)
            dynamic_gts = torch.empty(0, dtype=torch.long, device=giou.device)

        # 合并匈牙利匹配和动态匹配结果
        combined_preds = torch.cat([pred_indices, dynamic_preds])
        combined_gts = torch.cat([gt_indices, dynamic_gts])

        return combined_preds, combined_gts

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0]/2.0, centers[:, 1] - sizes[:, 1]/2.0,
            centers[:, 2] - sizes[:, 2]/2.0, centers[:, 0] + sizes[:, 0]/2.0,
            centers[:, 1] + sizes[:, 1]/2.0, centers[:, 2] + sizes[:, 2]/2.0
        ], -1)



def euler_angles_to_rotation_matrix(angles):
    """
    从欧拉角 (roll, pitch, yaw) 转换为旋转矩阵
    :param angles: [batch_size, 3, box_num], 欧拉角 (roll, pitch, yaw)
    :return: 旋转矩阵 [batch_size, box_num, 3, 3]
    """
    roll, pitch, yaw = angles[:, 0, :], angles[:, 1, :], angles[:, 2, :]  # [batch_size, box_num]

    # 计算正弦和余弦
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # 绕 x 轴的旋转矩阵
    R_x = torch.stack([
        torch.stack([torch.ones_like(cos_r), torch.zeros_like(sin_r), torch.zeros_like(sin_r)], dim=-1),  # [batch_size, box_num, 3]
        torch.stack([torch.zeros_like(sin_r), cos_r, -sin_r], dim=-1),
        torch.stack([torch.zeros_like(sin_r), sin_r, cos_r], dim=-1)
    ], dim=-2)  # [batch_size, box_num, 3, 3]

    # 绕 y 轴的旋转矩阵
    R_y = torch.stack([
        torch.stack([cos_p, torch.zeros_like(sin_p), sin_p], dim=-1),
        torch.stack([torch.zeros_like(sin_p), torch.ones_like(cos_p), torch.zeros_like(sin_p)], dim=-1),
        torch.stack([-sin_p, torch.zeros_like(sin_p), cos_p], dim=-1)
    ], dim=-2)

    # 绕 z 轴的旋转矩阵
    R_z = torch.stack([
        torch.stack([cos_y, -sin_y, torch.zeros_like(cos_y)], dim=-1),
        torch.stack([sin_y, cos_y, torch.zeros_like(cos_y)], dim=-1),
        torch.stack([torch.zeros_like(cos_y), torch.zeros_like(sin_y), torch.ones_like(cos_y)], dim=-1)
    ], dim=-2)

    # 合成旋转矩阵：R = Rz * Ry * Rx
    R = torch.matmul(torch.matmul(R_z, R_y), R_x)  # [batch_size, box_num, 3, 3]
    return R


def rotation_to_homogeneous_matrix(R, t=None):
    """
    将旋转矩阵扩展为齐次变换矩阵
    :param R: [batch_size, box_num, 3, 3], 旋转矩阵
    :param t: [batch_size, box_num, 3], 平移向量（可选）
    :return: 齐次变换矩阵 [batch_size, box_num, 4, 4]
    """
    batch_size, box_num = R.shape[0], R.shape[1]
    T = torch.eye(4, device=R.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, box_num, 1, 1)  # [batch_size, box_num, 4, 4]
    T[:, :, :3, :3] = R  # 填入旋转矩阵
    if t is not None:
        T[:, :, :3, 3] = t  # 填入平移向量
    return T


def rotation_matrix_to_euler_angles(R):
    """
    从旋转矩阵转换为欧拉角 (roll, pitch, yaw)
    :param R: [batch_size, box_num, 3, 3], 旋转矩阵
    :return: [batch_size, 3, box_num], 欧拉角
    """
    sy = torch.sqrt(R[:, :, 0, 0] ** 2 + R[:, :, 1, 0] ** 2)  # [batch_size, box_num]
    singular = sy < 1e-6

    # 非奇异情况
    roll = torch.atan2(R[:, :, 2, 1], R[:, :, 2, 2])  # [batch_size, box_num]
    pitch = torch.atan2(-R[:, :, 2, 0], sy)           # [batch_size, box_num]
    yaw = torch.atan2(R[:, :, 1, 0], R[:, :, 0, 0])   # [batch_size, box_num]

    # 奇异情况
    roll_s = torch.atan2(-R[:, :, 1, 2], R[:, :, 1, 1])  # [batch_size, box_num]
    pitch_s = torch.atan2(-R[:, :, 2, 0], sy)            # [batch_size, box_num]
    yaw_s = torch.zeros_like(yaw)                        # [batch_size, box_num]

    # 选择奇异值和非奇异值
    roll = torch.where(singular, roll_s, roll)
    pitch = torch.where(singular, pitch_s, pitch)
    yaw = torch.where(singular, yaw_s, yaw)

    return torch.stack([roll, pitch, yaw], dim=1)  # [batch_size, 3, box_num]


def transform_angles_with_matrices(predicted_angles, T1_list, T2_list):
    """
    使用两个齐次变换矩阵（T1 和 T2）对预测角度进行变换
    :param predicted_angles: [batch_size, 3], 预测的欧拉角 (roll, pitch, yaw)
    :param T1: [batch_size, 4, 4], 第一个变换矩阵
    :param T2: [batch_size, 4, 4], 第二个变换矩阵
    :return: [batch_size, 3], 变换后的欧拉角
    """
    T1 = torch.stack(T1_list, dim=0).to(predicted_angles.device, dtype=predicted_angles.dtype)
    T2 = torch.stack(T2_list, dim=0).to(predicted_angles.device, dtype=predicted_angles.dtype)
    box_num = predicted_angles.shape[2]
    T1_expanded = T1.unsqueeze(1).expand(-1, box_num, -1, -1)  # [batch_size, box_num, 4, 4]
    T2_expanded = T2.unsqueeze(1).expand(-1, box_num, -1, -1)  # [batch_size, box_num, 4, 4]

    R_box = euler_angles_to_rotation_matrix(predicted_angles)  # [batch_size, box_num, 3, 3]
    T_box = rotation_to_homogeneous_matrix(R_box)             # [batch_size, box_num, 4, 4]
    T_transformed = torch.matmul(torch.matmul(T1_expanded, T_box), T2_expanded)  # T' = T1 * T_box * T2
    R_transformed = T_transformed[:, :, :3, :3]               # [batch_size, box_num, 3, 3]
    transformed_angles = rotation_matrix_to_euler_angles(R_transformed)  # [batch_size, 3, box_num]
    return transformed_angles





def rotate_yaw_angles_no_loop(predicted_angles, T1_list, T2_list):
    """
    对 predicted_angles 先进行 T1 旋转，再进行 T2 旋转，返回最终旋转后的角度。

    Args:
        predicted_angles (torch.Tensor): shape (batch_size, 1, box_num)，每个 box 的 yaw 角度。
        T1_list (list): list length: batch_size, 每个item的shape: (3, 3)，每个 batch 的旋转矩阵 T1。
        T2_list (list): list length: batch_size, 每个item的shape: (3, 3)，每个 batch 的旋转矩阵 T2。

    Returns:
        torch.Tensor: shape (batch_size, 1, box_num)，经过 T1 和 T2 旋转后的角度。
    """
    batch_size, _, box_num = predicted_angles.shape

    # 将 predicted_angles 从 shape (batch_size, 1, box_num) 转为 (batch_size, box_num)
    predicted_angles = predicted_angles.squeeze(1)  # shape: (batch_size, box_num)

    # 将 yaw 转为 3D 向量形式 (cos(yaw), sin(yaw), 0)
    yaw_vectors = torch.stack([
        torch.cos(predicted_angles),  # x 分量
        torch.sin(predicted_angles),  # y 分量
        torch.zeros_like(predicted_angles)  # z 分量为 0
    ], dim=2)  # shape: (batch_size, box_num, 3)

    # 转换 yaw_vectors 的维度以适应批量矩阵乘法
    yaw_vectors = yaw_vectors.permute(0, 2, 1)  # shape: (batch_size, 3, box_num)

    # 将旋转矩阵从 list 转为 tensor，确保 shape 为 (batch_size, 3, 3)
    T1 = torch.stack(T1_list, dim=0)[:, :3, :3]  # shape: (batch_size, 3, 3)
    T2 = torch.stack(T2_list, dim=0)[:, :3, :3].to(torch.float32)  # shape: (batch_size, 3, 3)

    # 先对 yaw_vectors 应用 T1 旋转
    rotated_vectors_T1 = torch.bmm(T1, yaw_vectors)  # shape: (batch_size, 3, box_num)

    # 再对结果应用 T2 旋转
    rotated_vectors_T2 = torch.bmm(T2, rotated_vectors_T1)  # shape: (batch_size, 3, box_num)

    # 提取旋转后的 x 和 y 分量
    x_T2, y_T2 = rotated_vectors_T2[:, 0, :], rotated_vectors_T2[:, 1, :]  # shape: (batch_size, box_num)

    # 计算最终旋转后的角度 (arctan2(y, x))
    rotated_angles = torch.atan2(y_T2, x_T2)  # shape: (batch_size, box_num)

    # 恢复到原始 shape (batch_size, 1, box_num)
    rotated_angles = rotated_angles.unsqueeze(1)  # shape: (batch_size, 1, box_num)

    return rotated_angles



@MODELS.register_module()
class VGGTDetHeadOneAngle_AngleIOU(BaseModule):
    r"""`ImVoxelNet<https://arxiv.org/abs/2106.01178>`_ head for indoor
    datasets.

    Args:
        n_classes (int): Number of classes.
        n_levels (int): Number of feature levels.
        n_channels (int): Number of channels in input tensors.
        n_reg_outs (int): Number of regression layer channels.
        pts_assign_threshold (int): Min number of location per box to
            be assigned with.
        pts_center_threshold (int): Max number of locations per box to
            be assigned with.
        center_loss (dict, optional): Config of centerness loss.
            Default: dict(type='CrossEntropyLoss', use_sigmoid=True).
        bbox_loss (dict, optional): Config of bbox loss.
            Default: dict(type='RotatedIoU3DLoss').
        cls_loss (dict, optional): Config of classification loss.
            Default: dict(type='FocalLoss').
        train_cfg (dict, optional): Config for train stage. Defaults to None.
        test_cfg (dict, optional): Config for test stage. Defaults to None.
        init_cfg (dict, optional): Config for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 n_classes: int,
                 n_levels: int,
                 n_channels: int,
                 n_reg_outs: int,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 prior_generator: ConfigType,
                #  center_loss: ConfigType = dict(
                #      type='mmdet.CrossEntropyLoss', use_sigmoid=True),
                #  bbox_loss: ConfigType = dict(type='RotatedIoU3DLoss'),
                 cls_loss: ConfigType = dict(type='mmdet.FocalLoss', use_sigmoid=True),
                 objness_loss: ConfigType = dict(type='mmdet.FocalLoss', use_sigmoid=True),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 mlp_dropout=0.3,
                 matcher_cost_weights={'cls':1.0, 'center':0.0, 'obj_ness':0.0, 'giou':2.0},
                 loss_weights={'center_loss': 5.0, 'size_loss': 1.0,
                    'cls_loss': 1.0,
                    'objness_loss': 1.0,
                    'iou_loss': 1.0,
                    'not_objness_loss': 0.25, 'angle_loss': 0.5},
                learn_center_diff=False,
                visualize_3d_bbox=False,
                visualize_2d_bbox=False,
                visualize_path=None,
                if_v2_head=False,
                if_project_frist_frame_back=False,
                if_swap_length_and_width=False,
                matcher='one2one',
                matcher_iou_thres=0.25,
                matcher_max_dynamic_samples=10,
                angle_bin=12,
                visual_cfg: OptConfigType = None,
                    ):
        super(VGGTDetHeadOneAngle_AngleIOU, self).__init__(init_cfg)
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.n_reg_outs = n_reg_outs
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.prior_generator = TASK_UTILS.build(prior_generator)
        # self.center_loss = MODELS.build(center_loss)
        # self.bbox_loss = MODELS.build(bbox_loss)
        class_weights = torch.ones((self.n_classes+1), device='cuda') * 1.0
        class_weights[-1] = loss_weights['not_objness_loss']
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights) #MODELS.build(cls_loss)
        self.objness_loss = MODELS.build(objness_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if if_v2_head:
            self.mlp_func = partial(
                GenericMLP,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                hidden_dims=[n_channels, n_channels//2, n_channels//4, n_channels//8],
                dropout=mlp_dropout,
                input_dim=n_channels,
            )
        else:
            self.mlp_func = partial(
                GenericMLP,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                hidden_dims=[n_channels, n_channels],
                dropout=mlp_dropout,
                input_dim=n_channels,
            )
        assert matcher in ['one2one', 'one2more', 'one2more_angle']
        if matcher == 'one2one':
            self.matcher = UnifiedMatcher(cost_weights=matcher_cost_weights)
        elif matcher == 'one2more':
            self.matcher = UnifiedMatcherMoreThanOne(cost_weights=matcher_cost_weights, matcher_iou_thres=matcher_iou_thres, matcher_max_dynamic_samples=matcher_max_dynamic_samples)
        elif matcher == 'one2more_angle':
            self.matcher = UnifiedMatcherMoreThanOneWithAngle(cost_weights=matcher_cost_weights, matcher_iou_thres=matcher_iou_thres, matcher_max_dynamic_samples=matcher_max_dynamic_samples)
        self.loss_weights = loss_weights
        self.learn_center_diff = learn_center_diff
        self.visualize_3d_bbox = visualize_3d_bbox
        self.visualize_2d_bbox = visualize_2d_bbox
        self.visualize_path = visualize_path
        self.if_project_frist_frame_back = if_project_frist_frame_back
        self.if_swap_length_and_width = if_swap_length_and_width
        self.angle_bin = angle_bin
        self.bin_width = 2 * torch.pi / angle_bin
        # if self.visualize_3d_bbox:
        #     self.test_cfg['score_thr'] = visual_cfg['score_thr']
        #     self.test_cfg['iou_thr'] = visual_cfg['iou_thr']
        self._init_layers(n_channels, n_reg_outs, n_classes, n_levels)

    def _init_layers(self, n_channels, n_reg_outs, n_classes, n_levels):
        """Initialize neural network layers of the head."""
        # self.conv_center = nn.Conv3d(n_channels, 1, 3, padding=1, bias=False)

        self.center_head = self.mlp_func(output_dim=3)
        # self.conv_reg = nn.Conv3d(
        #     n_channels, n_reg_outs, 3, padding=1, bias=False)
        self.size_head = self.mlp_func(output_dim=3)

        self.semcls_head = self.mlp_func(output_dim=n_classes+1) # foreground categories
        # self.objness_head = self.mlp_func(output_dim=1) # objectness
        # self.conv_cls = nn.Conv3d(n_channels, n_classes, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(n_levels)])

        self.angle_head = self.mlp_func(output_dim=1)
        # self.yaw_angle_regress_head = self.mlp_func(output_dim=1)

    # def init_weights(self):
    #     """Initialize all layer weights."""
    #     normal_init(self.conv_center, std=.01)
    #     normal_init(self.conv_reg, std=.01)
    #     normal_init(self.conv_cls, std=.01, bias=bias_init_with_prob(.01))
    def project_the_first_frame_back(self, x: Tensor, pose_matrix, axis_align_matrix):
        batch_size, _, num_boxes = x.shape
        # 将 pose_matrix 堆叠成 Tensor，方便批量计算
        pose_matrix = torch.stack(pose_matrix, dim=0).to(x.device, dtype=x.dtype)  # [16, 4, 4]
        axis_align_matrix = torch.stack(axis_align_matrix, dim=0).to(x.device, dtype=x.dtype)
        # Step 1: 将 x 转换为齐次坐标，增加一维
        ones = torch.ones(batch_size, 1, num_boxes, device=x.device)  # [16, 1, 256]
        x_homogeneous = torch.cat([x, ones], dim=1)  # [16, 4, 256]

        # Step 2: 使用 pose_matrix 对齐次坐标进行变换
        # 矩阵乘法：pose_matrix [16, 4, 4] 和 x_homogeneous [16, 4, 256]
        x_global_homogeneous = torch.bmm(pose_matrix, x_homogeneous)  # [16, 4, 256]
        x_global_homogeneous = torch.bmm(axis_align_matrix, x_global_homogeneous)
        # Step 3: 转换回 3D 坐标（去掉齐次坐标的最后一维）
        # x', y', z' = x', y', z' / w
        w = torch.clamp(x_global_homogeneous[:, 3:4, :], min=1e-8)  # 避免 w 为 0
        x_global = x_global_homogeneous[:, :3, :] / w
        # x_global = x_global_homogeneous[:, :3, :] / x_global_homogeneous[:, 3:4, :]  # [16, 3, 256]
        return x_global


    def pred_angle(self, x):
        """
        使用 softmax 加权平均计算每个 box 的预测角度，使其可导。
        :param x: 输入特征，形状为 (batch_size, 512, 256)，其中 256 是 box 的数量
        :return: predicted_angles，形状为 (batch_size, 3, 256)
        """
        batch_size, feature_dim, box_num = x.shape

        # # Roll 的分类和回归
        # roll_angle_logits = self.roll_angle_class_head(x)  # 分类 logits, (batch_size, num_bins, box_num)
        # roll_angle_offsets = self.roll_angle_regress_head(x)  # 偏移量, (batch_size, 1, box_num)
        # roll_probs = torch.softmax(roll_angle_logits, dim=1)  # 分类 logits 的 softmax 概率, (batch_size, num_bins, box_num)
        # roll_bin_centers = (
        #     torch.arange(roll_angle_logits.size(1), device=x.device).float() * self.bin_width - torch.pi + self.bin_width / 2
        # ).view(1, -1, 1)  # bin 中心值, (1, num_bins, 1)
        # roll_predicted_angle = torch.sum(roll_probs * roll_bin_centers, dim=1, keepdim=True) + roll_angle_offsets  # (batch_size, 1, box_num)

        # # Pitch 的分类和回归
        # pitch_angle_logits = self.pitch_angle_class_head(x)  # 分类 logits, (batch_size, num_bins, box_num)
        # pitch_angle_offsets = self.pitch_angle_regress_head(x)  # 偏移量, (batch_size, 1, box_num)
        # pitch_probs = torch.softmax(pitch_angle_logits, dim=1)  # 分类 logits 的 softmax 概率, (batch_size, num_bins, box_num)
        # pitch_bin_centers = (
        #     torch.arange(pitch_angle_logits.size(1), device=x.device).float() * self.bin_width - torch.pi + self.bin_width / 2
        # ).view(1, -1, 1)  # bin 中心值, (1, num_bins, 1)
        # pitch_predicted_angle = torch.sum(pitch_probs * pitch_bin_centers, dim=1, keepdim=True) + pitch_angle_offsets  # (batch_size, 1, box_num)

        # Yaw 的分类和回归
        # yaw_angle_logits = self.yaw_angle_class_head(x)  # 分类 logits, (batch_size, num_bins, box_num)
        # yaw_angle_offsets = self.yaw_angle_regress_head(x)  # 偏移量, (batch_size, 1, box_num)
        # yaw_probs = torch.softmax(yaw_angle_logits, dim=1)  # 分类 logits 的 softmax 概率, (batch_size, num_bins, box_num)
        # yaw_bin_centers = (
        #     torch.arange(yaw_angle_logits.size(1), device=x.device).float() * self.bin_width - torch.pi + self.bin_width / 2
        # ).view(1, -1, 1)  # bin 中心值, (1, num_bins, 1)
        # yaw_predicted_angle = torch.sum(yaw_probs * yaw_bin_centers, dim=1, keepdim=True) + yaw_angle_offsets  # (batch_size, 1, box_num)

        predicted_angles = self.angle_head(x) # shape: (batch_size, 1, box_num)
        # 合并三个角度
        # predicted_angles = torch.cat(
        #     [roll_predicted_angle, pitch_predicted_angle, yaw_predicted_angle], dim=1
        # )  # (batch_size, 3, box_num)

        return predicted_angles

    def _forward_single(self, x: Tensor, scale: Scale, query_xyz, pose_matrix, axis_align_matrix, avg_distance):
        """Forward pass per level.

        Args:
            x (Tensor): Per level 3d neck output tensor.
            scale (mmcv.cnn.Scale): Per level multiplication weight.

        Returns:
            tuple[Tensor]: Centerness, bbox and classification predictions.
        """
        if self.learn_center_diff:
            query_xyz = query_xyz.permute(0, 2, 1)
            if self.if_project_frist_frame_back:
                center_pred = self.project_the_first_frame_back(self.center_head(x)+query_xyz, pose_matrix, axis_align_matrix)
            else:
                center_pred = self.center_head(x)+query_xyz

        else:
            if self.if_project_frist_frame_back:
                center_pred = self.project_the_first_frame_back(self.center_head(x), pose_matrix, axis_align_matrix)
            else:
                center_pred = self.center_head(x)

        predicted_angles = self.pred_angle(x)
        # transformed_angles = rotate_yaw_angles_no_loop(predicted_angles, pose_matrix, axis_align_matrix)
            # avg_distance_tensor = torch.stack(avg_distance).unsqueeze(-1)

        size_pred = torch.exp(scale(self.size_head(x)))
   
        return (center_pred, size_pred, #/ avg_distance_tensor,
                self.semcls_head(x), predicted_angles) # , self.objness_head(x)

    def forward(self, x, batch_inputs_dict, batch_data_samples):
        if 'query_xyz' in batch_inputs_dict.keys():
            return multi_apply(self._forward_single, x, self.scales, [batch_inputs_dict['query_xyz'] for _ in range(self.n_levels)], [batch_inputs_dict['pose_matrix'] for _ in range(self.n_levels)], [batch_inputs_dict['axis_align_matrix'] for _ in range(self.n_levels)], [batch_inputs_dict['avg_distance'] for _ in range(self.n_levels)]) 
        else:
            return multi_apply(self._forward_single, x, self.scales, [None for _ in range(self.n_levels)], [batch_inputs_dict['pose_matrix'] for _ in range(self.n_levels)], [batch_inputs_dict['axis_align_matrix'] for _ in range(self.n_levels)], [batch_inputs_dict['avg_distance'] for _ in range(self.n_levels)]) 

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList, batch_inputs_dict: dict,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # valid_pred = x[-1]
        outs = self(x, batch_inputs_dict, batch_data_samples) # x len: 8, every tensor shape: [bs, feat_dim, num_queries]

        if 'points' in batch_inputs_dict.keys():
            batch_input_points = batch_inputs_dict['points']
        else:
            # batch_input_points = [None for i in range(len(batch_input_metas))]
            batch_input_points = [None for i in range(len(batch_data_samples))]

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (batch_gt_instances_3d,
                              batch_input_metas, batch_input_points, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self,
                     center_preds: List[List[Tensor]],
                     size_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     angle_preds: List[List[Tensor]],
                    #  objness_preds: List[List[Tensor]],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_input_points,
                     batch_gt_instances_ignore: OptInstanceList = None,
                     **kwargs) -> dict:
        """Per scene loss function.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
                The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes. The first list contains predictions from different
                levels. The second list contains predictions in a mini-batch.
            valid_pred (Tensor): Valid mask prediction for all scenes.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
            batch_input_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        bboxes_3d.gravity_center: 物体的中心点
        bboxes_3d.tensor[:, 3:6]: 物体的长宽高

        Returns:
            dict: Centerness, bbox, and classification loss values.
        """
        # valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        center_losses, size_losses, cls_losses, objness_losses, giou_losses, angle_losses = [], [], [], [], [], []
        for i in range(len(batch_input_metas)):
            center_loss, size_loss, cls_loss, giou_loss, angle_loss = self._loss_by_feat_single(
                center_preds=[x[i] for x in center_preds],
                size_preds=[x[i] for x in size_preds],
                cls_preds=[x[i] for x in cls_preds],
                angle_preds=[x[i] for x in angle_preds],
                # objness_preds=[x[i] for x in objness_preds],
                # valid_preds=[x[i] for x in valid_preds],
                input_meta=batch_input_metas[i],
                gt_bboxes=batch_gt_instances_3d[i].bboxes_3d,
                gt_labels=batch_gt_instances_3d[i].labels_3d,
                input_points=batch_input_points[i])
            center_losses.append(center_loss)
            size_losses.append(size_loss)
            cls_losses.append(cls_loss)
            # objness_losses.append(objness_loss)
            giou_losses.append(giou_loss)
            angle_losses.append(angle_loss)

        return dict(
            center_loss=torch.mean(torch.stack(center_losses)),
            size_loss=torch.mean(torch.stack(size_losses)),
            cls_loss=torch.mean(torch.stack(cls_losses)),
            # objness_loss=torch.mean(torch.stack(objness_losses)),
            giou_loss=torch.mean(torch.stack(giou_losses)),
            angle_loss=torch.mean(torch.stack(angle_losses))
            )

    def _loss_by_feat_single(self, center_preds, size_preds, cls_preds, angle_preds, #objness_preds,
                              input_meta, gt_bboxes, gt_labels, input_points):
        """
        输入参数：
            center_preds: List[Tensor(3, N)] 各阶段中心预测
            size_preds: List[Tensor(3, N)] 各阶段尺寸预测
            cls_preds: List[Tensor(C, N)] 各阶段分类预测
            objness_preds: List[Tensor(1, N)] 各阶段存在性预测
            gt_bboxes: DepthInstance3DBoxes实例
            gt_labels: Tensor(M,)
        返回：
            center_loss, size_loss, giou_loss, cls_loss
        """
        # 拼接多阶段预测
        all_centers = torch.cat([c.t() for c in center_preds], dim=0)  # (Total_Pred, 3)
        all_sizes = torch.cat([s.t() for s in size_preds], dim=0)      # (Total_Pred, 3) 
        all_cls = torch.cat([c.t() for c in cls_preds], dim=0)         # (Total_Pred, C)
        all_angles = torch.cat([c.t() for c in angle_preds], dim=0) 
        # all_objness = torch.cat([c.t() for c in objness_preds], dim=0)
        


        # 提取GT数据
        gt_centers = gt_bboxes.gravity_center
        gt_sizes = gt_bboxes.tensor[:, 3:6]

        all_pred_indices = []
        all_gt_indices = []
        offset = 0  # 跨阶段的预测索引偏移量

        # time1 = time.time()
        # a = gt_sizes.cpu()
        for stage_idx in range(len(center_preds)):
            centers, sizes, cls_scores, angles = center_preds[stage_idx].t(), size_preds[stage_idx].t(), cls_preds[stage_idx].t(), angle_preds[stage_idx].t() #, objness_preds[stage_idx].t()
            cls_scores_softmax = F.softmax(cls_scores, dim=1)
            obj_scores = 1.0 - cls_scores_softmax[:, -1]
            n_predictions = centers.size(0)

            # 当前阶段匹配
            pred_indices, gt_indices = self.matcher._get_targets(
                centers, sizes, cls_scores, obj_scores, angles,
                gt_centers, gt_sizes, gt_labels, gt_bboxes.tensor[:, 6]
            )
            
            # 调整预测索引的全局偏移
            all_pred_indices.append(pred_indices + offset)
            all_gt_indices.append(gt_indices)
            
            # 更新偏移量为下一阶段准备
            offset += n_predictions

        # time2 = time.time()
        # a = gt_sizes.cpu()

        pred_indices, gt_indices = torch.cat(all_pred_indices), torch.cat(all_gt_indices)
        # 执行全局匹配
        # pred_indices, gt_indices = self.matcher._get_targets(
        #     all_centers, all_sizes, all_cls, all_objness,
        #     gt_centers, gt_sizes, gt_labels
        # )

        # 提取匹配的预测和GT
        matched_centers = all_centers[pred_indices]
        matched_sizes = all_sizes[pred_indices]
        matched_cls = all_cls[pred_indices]
        matched_angles = all_angles[pred_indices].squeeze(-1)
        # matched_objness = all_objness[pred_indices]
        matched_gt_centers = gt_centers[gt_indices]
        matched_gt_sizes = gt_sizes[gt_indices]
        matched_gt_labels = gt_labels[gt_indices]
        matched_gt_angles = gt_bboxes.tensor[:, 6][gt_indices]


        angles_loss = F.mse_loss(matched_angles, matched_gt_angles) * self.loss_weights['angle_loss']



        # 计算各损失项
        # 中心L1损失
        center_loss = F.l1_loss(matched_centers, matched_gt_centers) * self.loss_weights['center_loss']
        
        # 尺寸L1损失
        size_loss = F.l1_loss(matched_sizes, matched_gt_sizes) * self.loss_weights['size_loss']
        
        # 分类交叉熵损失
        # cls_loss = self.cls_loss(matched_cls, matched_gt_labels, avg_facter=matched_cls.shape[0]) #??? 这里是shape[0]么？

        # 分类Focal Loss，会自动算上sigmoid
        cls_target = torch.ones((all_centers.shape[0]), device=all_centers.device) * self.n_classes # 整数
        cls_target = cls_target.long()
        cls_target[pred_indices] = matched_gt_labels


        cls_loss = self.cls_loss(all_cls, cls_target) * self.loss_weights['cls_loss']


        # cls_targets_onehot = F.one_hot(
        #     matched_gt_labels, 
        #     num_classes=self.n_classes
        # ).float()

        # cls_loss = self.cls_loss(
        #     matched_cls, 
        #     cls_targets_onehot, avg_factor = (matched_cls.shape[0]+ 1e-16)
        # )  * self.loss_weights['cls_loss']
        
        # 存在性Focal Loss（二分类）
        # objness_target = torch.zeros_like(all_objness)
        # objness_target[pred_indices] = 1.0
        # objness_loss_weight = torch.ones_like(all_objness) * self.loss_weights['not_objness_loss']
        # objness_loss_weight[pred_indices] = self.loss_weights['objness_loss']

        # objness_loss_all = self.objness_loss(
        #     all_objness,  # (N,1)
        #     objness_target,  # (N,1)
        #     # avg_factor=matched_cls.shape[0],
        #     reduction_override='none'
        # )   * objness_loss_weight
        # objness_loss = torch.sum(objness_loss_all) / (torch.sum(objness_loss_weight) + 1e-16)

        # GIoU损失


        # pred_tp_bbox = self._center_size_pred_to_bbox(matched_centers, matched_sizes)
        # gt_tp_bbox = self._center_size_pred_to_bbox(matched_gt_centers, matched_gt_sizes)

        # giou = axis_aligned_bbox_overlaps_3d(pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0), mode='giou', is_aligned=True)


        matched_pred_bboxes = torch.cat([matched_centers.unsqueeze(0), matched_sizes.unsqueeze(0), matched_angles.unsqueeze(0).unsqueeze(-1)], dim=-1)

        matched_gt_bboxes = torch.cat([matched_gt_centers.unsqueeze(0), matched_gt_sizes.unsqueeze(0), matched_gt_angles.unsqueeze(0).unsqueeze(-1)], dim=-1)
        # giou = diff_iou_rotated_3d(matched_pred_bboxes, matched_gt_bboxes)
        giou = cal_giou_3d(matched_pred_bboxes, matched_gt_bboxes)
        # pred_corners = get_3d_box_batch_depth_tensor(
        #     matched_sizes.unsqueeze(0), # [1, 40, 3]
        #     matched_angles.unsqueeze(0), # [1, 40]
        #     matched_centers.unsqueeze(0) # [1, 40, 3]
        # )
        # gt_corners = get_3d_box_batch_depth_tensor(
        #     matched_gt_sizes.unsqueeze(0), # [1, 40, 3]
        #     matched_gt_angles.unsqueeze(0), # [1, 40]
        #     matched_gt_centers.unsqueeze(0) # [1, 40, 3]
        # )

        # giou = generalized_box3d_iou(pred_corners, gt_corners, torch.tensor([len(gt_indices)]), rotated_boxes=(torch.sum(torch.abs(gt_bboxes.tensor[:, -1])) > 1e-16), needs_grad=(self.loss_weights['iou_loss'] > 0))
        giou_loss = (1.0 - giou).mean() * self.loss_weights['iou_loss']





        return center_loss, size_loss, cls_loss, giou_loss, angles_loss

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList, batch_inputs_dict, 
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 6.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        # valid_pred = x[-1]
        outs = self(x, batch_inputs_dict, batch_data_samples) 
        predictions = self.predict_by_feat(
            *outs,
            batch_input_metas=batch_input_metas,
            rescale=rescale, batch_inputs_dict=batch_inputs_dict, batch_data_samples=batch_data_samples)
        return predictions

    def predict_by_feat(self, center_preds: List[List[Tensor]],
                     size_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     angle_preds: List[List[Tensor]],
                        batch_input_metas: List[dict], batch_inputs_dict: dict, batch_data_samples,
                        **kwargs) -> List[InstanceData]:
        """Generate boxes for all scenes.

        Args:
            center_preds (list[list[Tensor]]): Centerness predictions for
                all scenes.
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            valid_pred (Tensor): Valid mask prediction for all scenes.
            batch_input_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[tuple[Tensor]]: Predicted bboxes, scores, and labels for
                all scenes.
        """
        # valid_preds = self._upsample_valid_preds(valid_pred, center_preds)
        results = []
        if 'points' in batch_inputs_dict.keys():
            batch_input_points = batch_inputs_dict['points']
        else:
            batch_input_points = [None for i in range(len(batch_input_metas))]
        if 'points_first_axis' in batch_inputs_dict.keys():
            batch_points_first_axis = batch_inputs_dict['points_first_axis']
        else:
            batch_points_first_axis = [None for i in range(len(batch_input_metas))]
        for i in range(len(batch_input_metas)):
            results.append(
                self._predict_by_feat_single(
                center_preds=[x[i] for x in center_preds],
                size_preds=[x[i] for x in size_preds],
                cls_preds=[x[i] for x in cls_preds],
                angle_preds=[x[i] for x in angle_preds],
                # objness_preds=[x[i] for x in objness_preds],
                # valid_preds=[x[i] for x in valid_preds],
                input_meta=batch_input_metas[i],
                input_points=batch_input_points[i],
                data_samples=batch_data_samples[i],
                # pose_matrix=batch_inputs_dict['pose_matrix'],
                points_first_axis=batch_points_first_axis))
        return results

    def _predict_by_feat_single(self, center_preds, size_preds, cls_preds, angle_preds,
                                input_meta: dict, input_points, data_samples,  points_first_axis) -> InstanceData:
        """Generate boxes for single sample.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            valid_preds (tuple[Tensor]): Upsampled valid masks for all feature
                levels.
            input_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor]: Predicted bounding boxes, scores and labels.
        """
        # all_centers = torch.cat([c.t() for c in center_preds], dim=0)  # (Total_Pred, 3)
        # all_sizes = torch.cat([s.t() for s in size_preds], dim=0)      # (Total_Pred, 3) 
        # all_cls = torch.cat([c.t() for c in cls_preds], dim=0)         # (Total_Pred, C)
        # all_objness = torch.cat([c.t() for c in objness_preds], dim=0)
    
        # featmap_sizes = [featmap.size()[-3:] for featmap in center_preds]
        # points = self._get_points(
        #     featmap_sizes=featmap_sizes,
        #     origin=input_meta['lidar2img']['origin'],
        #     device=center_preds[0].device)
        
        # for center_pred, size_pred, cls_pred, objness_pred in zip(
        #         center_preds, size_preds, cls_preds, objness_preds):
        #     center_pred = center_pred.permute(1, 2, 3, 0).reshape(-1, 1)
        #     bbox_pred = bbox_pred.permute(1, 2, 3,
        #                                   0).reshape(-1, bbox_pred.shape[0])
        #     cls_pred = cls_pred.permute(1, 2, 3,
        #                                 0).reshape(-1, cls_pred.shape[0])
        #     valid_pred = valid_pred.permute(1, 2, 3, 0).reshape(-1, 1)

        mlvl_bboxes, mlvl_scores = [], []
        for stage_idx in range(len(center_preds)):
            centers, sizes, cls_scores, angles = center_preds[stage_idx].t(), size_preds[stage_idx].t(), cls_preds[stage_idx].t(), angle_preds[stage_idx].t()
            cls_scores = F.softmax(cls_scores, dim=1)
            objectness = 1 - cls_scores[:, -1]
            scores = cls_scores[:, :-1] * objectness.unsqueeze(-1)

            # scores = cls_pred.sigmoid() * center_pred.sigmoid() * valid_pred
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                # bbox_pred = bbox_pred[ids]
                centers = centers[ids]
                sizes = sizes[ids]
                scores = scores[ids]
                angles =  angles[ids]

                # point = point[ids]

            # bboxes = self._center_size_pred_to_bbox_with_angle(centers, sizes,  angles) # 输出是tp box
            bboxes = torch.cat([centers, sizes, angles], dim=1)
            # bboxes = torch.stack([
            #     centers[:, 0] - sizes[:, 0]/2.0, centers[:, 1] - sizes[:, 1]/2.0,
            #     centers[:, 2] - sizes[:, 2]/2.0, centers[:, 0] + sizes[:, 0]/2.0,
            #     centers[:, 1] + sizes[:, 1]/2.0, centers[:, 2] + sizes[:, 2]/2.0, angles.squeeze(-1)
            # ], -1)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        # bboxes_after_nms, scores, labels = self._nms(bboxes, scores, input_meta) # bboxes(n_box, 6) (x_center, y_center, z_center, w, h, z)
        bboxes_after_nms, scores, labels = self._single_scene_multiclass_nms(
                    bboxes, scores, input_meta)

        bboxes = input_meta['box_type_3d'](
            bboxes_after_nms, box_dim=7, with_yaw=True, origin=(.5, .5, .5))
        
       # gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes
        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        # results = InstanceData()
        # results.bboxes_3d = data_samples.gt_instances_3d.bboxes_3d
        # results.scores_3d = torch.ones_like(data_samples.gt_instances_3d.labels_3d)
        # results.labels_3d = data_samples.gt_instances_3d.labels_3d

        if self.visualize_3d_bbox:
            # print(1) # 是不是gravity_center和gt的不一致？
            gt_boxes = data_samples.gt_instances_3d.bboxes_3d
            gt_boxes = torch.cat(
                (gt_boxes.gravity_center, gt_boxes.tensor[:, 3:6], gt_boxes.tensor[:, 6].unsqueeze(-1)), dim=1) # gt_bboxes.gravity_center: 中心点，gt_bboxes.tensor[:, 3:6]： 长宽高
            scene_path = input_meta['img_path'][0].split('/')[-2]

            # max_giou, max_gt_box_idx, max_pred_box_idx = self.find_max_iou_from_center_size_boxes(bboxes_after_nms, gt_boxes)
            # if isinstance(max_pred_box_idx, torch.Tensor):
            #     max_pred_box_idx = max_pred_box_idx.item()
            write_oriented_bbox(gt_boxes.cpu().numpy(), self.visualize_path+'/'+'%s_gt_boxes.ply' % scene_path)

            # max_gt_box_idx_tmp = max_gt_box_idx[:, max_pred_box_idx]
            # if isinstance(max_gt_box_idx_tmp, torch.Tensor):
            #     max_gt_box_idx_tmp = max_gt_box_idx_tmp.item()

            # write_oriented_bbox(gt_boxes[max_gt_box_idx_tmp].unsqueeze(0).cpu().numpy(), self.visualize_path+'/'+'%s_max_iou_gt_boxes.ply' % scene_path)

            write_oriented_bbox(bboxes_after_nms.cpu().numpy(), self.visualize_path+'/'+'%s_pred_boxes.ply' % scene_path)
            # write_oriented_bbox(bboxes_after_nms[max_pred_box_idx].unsqueeze(0).cpu().numpy(), self.visualize_path+'/'+'%s_max_iou_%f_boxes.ply' % (scene_path, max_giou))
            if input_points is not None:
                write_ply_rgb(input_points.cpu().numpy(), self.visualize_path+'/'+'%s_gt_points.ply' % scene_path)
                # write_ply_rgb(points_first_axis[0].cpu().numpy(), self.visualize_path+'/'+f'{scene_path}_gt_points_first_axis.ply')
        return results

    def find_max_iou_from_center_size_boxes(self, boxes1, boxes2):
        pred_corners = get_3d_box_batch_depth_tensor(
            boxes1[:, :3].unsqueeze(0),
            boxes1[:, 6].unsqueeze(0),
            boxes1[:, 3:6].unsqueeze(0)
        )
        gt_corners = get_3d_box_batch_depth_tensor(
            boxes2[:, :3].unsqueeze(0),
            boxes2[:, 6].unsqueeze(0),
            boxes2[:, 3:6].unsqueeze(0)
        )

        # time6 = time.time()
        # a = gt_sizes.cpu()
        giou = generalized_box3d_iou(pred_corners, gt_corners, torch.tensor([boxes1.shape[0]]), rotated_boxes=True, needs_grad=False) # 一定记得改pred_corners，别让相同的
        giou_max_gt, max_gt_box_idx = torch.max(giou, axis=2)
        max_giou, max_pred_box_idx = torch.max(giou_max_gt, axis=1)

        # boxes1_tp = self._center_size_pred_to_bbox(boxes1[:, :3], boxes1[:, 3:6])
        # boxes2_tp = self._center_size_pred_to_bbox(boxes2[:, :3], boxes2[:, 3:6])
        # giou_2 = axis_aligned_bbox_overlaps_3d(boxes1_tp.unsqueeze(0), boxes2_tp.unsqueeze(0), mode='giou') # giou
        # giou_max_gt, max_gt_box_idx = torch.max(giou_2, axis=2)
        # max_giou, max_pred_box_idx = torch.max(giou_max_gt, axis=1)
        assert max_giou <= 1 and  max_giou >= -1
        return max_giou, max_gt_box_idx, max_pred_box_idx



    @staticmethod
    def _upsample_valid_preds(valid_pred, features):
        """Upsample valid mask predictions.

        Args:
            valid_pred (Tensor): Valid mask prediction.
            features (Tensor): Feature tensor.

        Returns:
            tuple[Tensor]: Upsampled valid masks for all feature levels.
        """
        return [
            nn.Upsample(size=x.shape[-3:],
                        mode='trilinear')(valid_pred).round().bool()
            for x in features
        ]

    @torch.no_grad()
    def _get_points(self, featmap_sizes, origin, device):
        mlvl_points = []
        tmp_voxel_size = [.16, .16, .2]
        for i, featmap_size in enumerate(featmap_sizes):
            mlvl_points.append(
                get_points(
                    n_voxels=torch.tensor(featmap_size),
                    voxel_size=torch.tensor(tmp_voxel_size) * (2**i),
                    origin=torch.tensor(origin)).reshape(3, -1).transpose(
                        0, 1).to(device))
        return mlvl_points

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        return torch.stack([
            points[:, 0] - bbox_pred[:, 0], points[:, 1] - bbox_pred[:, 2], # boxes[..., 0] - boxes[..., 3] / 2, boxes[..., 1] - boxes[..., 4] / 2
            points[:, 2] - bbox_pred[:, 4], points[:, 0] + bbox_pred[:, 1], # boxes[..., 2] - boxes[..., 5] / 2, boxes[..., 0] + boxes[..., 3] / 2
            points[:, 1] + bbox_pred[:, 3], points[:, 2] + bbox_pred[:, 5]  # boxes[..., 1] + boxes[..., 4] / 2, boxes[..., 2] + boxes[..., 5] / 2
        ], -1)

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0]/2.0, centers[:, 1] - sizes[:, 1]/2.0,
            centers[:, 2] - sizes[:, 2]/2.0, centers[:, 0] + sizes[:, 0]/2.0,
            centers[:, 1] + sizes[:, 1]/2.0, centers[:, 2] + sizes[:, 2]/2.0
        ], -1)


    def _center_size_pred_to_bbox_with_angle(self, centers, sizes, angles):
        return torch.stack([
            centers[:, 0] - sizes[:, 0]/2.0, centers[:, 1] - sizes[:, 1]/2.0,
            centers[:, 2] - sizes[:, 2]/2.0, centers[:, 0] + sizes[:, 0]/2.0,
            centers[:, 1] + sizes[:, 1]/2.0, centers[:, 2] + sizes[:, 2]/2.0, angles.squeeze(-1)
        ], -1)

    def _bbox_pred_to_loss(self, points, bbox_preds):
        return self._bbox_pred_to_bbox(points, bbox_preds)

    # The function is directly copied from FCAF3DHead.
    @staticmethod
    def _get_face_distances(points, boxes):
        """Calculate distances from point to box faces.

        Args:
            points (Tensor): Final locations of shape (N_points, N_boxes, 3).
            boxes (Tensor): 3D boxes of shape (N_points, N_boxes, 7)

        Returns:
            Tensor: Face distances of shape (N_points, N_boxes, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).
        """
        dx_min = points[..., 0] - boxes[..., 0] + boxes[..., 3] / 2
        dx_max = boxes[..., 0] + boxes[..., 3] / 2 - points[..., 0]
        dy_min = points[..., 1] - boxes[..., 1] + boxes[..., 4] / 2
        dy_max = boxes[..., 1] + boxes[..., 4] / 2 - points[..., 1]
        dz_min = points[..., 2] - boxes[..., 2] + boxes[..., 5] / 2
        dz_max = boxes[..., 2] + boxes[..., 5] / 2 - points[..., 2]
        return torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max),
                           dim=-1)

    @staticmethod
    def _get_centerness(face_distances):
        """Compute point centerness w.r.t containing box.

        Args:
            face_distances (Tensor): Face distances of shape (B, N, 6),
                (dx_min, dx_max, dy_min, dy_max, dz_min, dz_max).

        Returns:
            Tensor: Centerness of shape (B, N).
        """
        x_dims = face_distances[..., [0, 1]]
        y_dims = face_distances[..., [2, 3]]
        z_dims = face_distances[..., [4, 5]]
        centerness_targets = x_dims.min(dim=-1)[0] / x_dims.max(dim=-1)[0] * \
            y_dims.min(dim=-1)[0] / y_dims.max(dim=-1)[0] * \
            z_dims.min(dim=-1)[0] / z_dims.max(dim=-1)[0]
        return torch.sqrt(centerness_targets)

    # @torch.no_grad()
    # def _get_targets(self, center_preds, size_preds, cls_preds, objness_preds, gt_bboxes, gt_labels):
    #     """Compute targets for final locations for a single scene.

    #     Args:
    #         points (list[Tensor]): Final locations for all levels.
    #         gt_bboxes (BaseInstance3DBoxes): Ground truth boxes.
    #         gt_labels (Tensor): Ground truth labels.

    #     Returns:
    #         tuple[Tensor]: Centerness, bbox and classification
    #             targets for all locations.
    #     """
    #     float_max = 1e8
    #     expanded_scales = [
    #         points[i].new_tensor(i).expand(len(points[i])).to(gt_labels.device)
    #         for i in range(len(points))
    #     ]
    #     points = torch.cat(points, dim=0).to(gt_labels.device) # (N1+N2+N3, 3)
    #     scales = torch.cat(expanded_scales, dim=0)

    #     # below is based on FCOSHead._get_target_single
    #     n_points = len(points)
    #     n_boxes = len(gt_bboxes)
    #     volumes = gt_bboxes.volume.to(points.device)
    #     volumes = volumes.expand(n_points, n_boxes).contiguous()
    #     gt_bboxes = torch.cat(
    #         (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:6]), dim=1) # gt_bboxes.gravity_center: 中心点，gt_bboxes.tensor[:, 3:6]： 长宽高
    #     gt_bboxes = gt_bboxes.to(points.device).expand(n_points, n_boxes, 6)
    #     expanded_points = points.unsqueeze(1).expand(n_points, n_boxes, 3)
    #     bbox_targets = self._get_face_distances(expanded_points, gt_bboxes) # (N1+N2+N3, n_bbox, 6) each point to bbox's 6 faces distance.

    #     # condition1: inside a gt bbox
    #     inside_gt_bbox_mask = bbox_targets[..., :6].min(
    #         -1)[0] > 0  # skip angle

    #     # condition2: positive points per scale >= limit
    #     # calculate positive points per scale
    #     n_pos_points_per_scale = []
    #     for i in range(self.n_levels):
    #         n_pos_points_per_scale.append(
    #             torch.sum(inside_gt_bbox_mask[scales == i], dim=0))
    #     # find best scale
    #     n_pos_points_per_scale = torch.stack(n_pos_points_per_scale, dim=0) # (3, n_bbox). 3scales. each scale, how many points fit in each bbox
    #     lower_limit_mask = n_pos_points_per_scale < self.pts_assign_threshold
    #     # fix nondeterministic argmax for torch<1.7
    #     extra = torch.arange(self.n_levels, 0, -1).unsqueeze(1).expand(
    #         self.n_levels, n_boxes).to(lower_limit_mask.device)
    #     lower_index = torch.argmax(lower_limit_mask.int() * extra, dim=0) - 1
    #     lower_index = torch.where(lower_index < 0,
    #                               torch.zeros_like(lower_index), lower_index)
    #     all_upper_limit_mask = torch.all(
    #         torch.logical_not(lower_limit_mask), dim=0)
    #     best_scale = torch.where(
    #         all_upper_limit_mask,
    #         torch.ones_like(all_upper_limit_mask) * self.n_levels - 1,
    #         lower_index)
    #     # keep only points with best scale
    #     best_scale = torch.unsqueeze(best_scale, 0).expand(n_points, n_boxes)
    #     scales = torch.unsqueeze(scales, 1).expand(n_points, n_boxes)
    #     inside_best_scale_mask = best_scale == scales

    #     # condition3: limit topk locations per box by centerness
    #     centerness = self._get_centerness(bbox_targets) # (N1+N2+N3, n_bbox)
    #     centerness = torch.where(inside_gt_bbox_mask, centerness,
    #                              torch.ones_like(centerness) * -1)
    #     centerness = torch.where(inside_best_scale_mask, centerness,
    #                              torch.ones_like(centerness) * -1)
    #     top_centerness = torch.topk(
    #         centerness, self.pts_center_threshold + 1, dim=0).values[-1]
    #     inside_top_centerness_mask = centerness > top_centerness.unsqueeze(0)

    #     # if there are still more than one objects for a location,
    #     # we choose the one with minimal area
    #     volumes = torch.where(inside_gt_bbox_mask, volumes,
    #                           torch.ones_like(volumes) * float_max)
    #     volumes = torch.where(inside_best_scale_mask, volumes,
    #                           torch.ones_like(volumes) * float_max)
    #     volumes = torch.where(inside_top_centerness_mask, volumes,
    #                           torch.ones_like(volumes) * float_max)
    #     min_area, min_area_inds = volumes.min(dim=1)

    #     labels = gt_labels[min_area_inds]
    #     labels = torch.where(min_area == float_max,
    #                          torch.ones_like(labels) * -1, labels)
    #     bbox_targets = bbox_targets[range(n_points), min_area_inds]
    #     centerness_targets = self._get_centerness(bbox_targets)

    #     return centerness_targets, self._bbox_pred_to_bbox(
    #         points, bbox_targets), labels

    # def _nms(self, bboxes, scores, img_meta): # bbox is 6-dim. (x_min, y_min, z_min, x_max, y_max, z_max)
    #     scores, labels = scores.max(dim=1)
    #     ids = scores > self.test_cfg.score_thr
    #     bboxes = bboxes[ids]
    #     scores = scores[ids]
    #     labels = labels[ids]
    #     ids = self.aligned_3d_nms(bboxes, scores, labels,
    #                               self.test_cfg.iou_thr)
    #     bboxes = bboxes[ids]
    #     bboxes = torch.stack(
    #         ((bboxes[:, 0] + bboxes[:, 3]) / 2.,
    #          (bboxes[:, 1] + bboxes[:, 4]) / 2.,
    #          (bboxes[:, 2] + bboxes[:, 5]) / 2., bboxes[:, 3] - bboxes[:, 0],
    #          bboxes[:, 4] - bboxes[:, 1], bboxes[:, 5] - bboxes[:, 2], bboxes[:, 6]),
    #         dim=1) # (convert to (x_center, y_center, z_center, w, h, z))
    #     return bboxes, scores[ids], labels[ids]

    # @staticmethod
    # def aligned_3d_nms(boxes, scores, classes, thresh):
    #     """3d nms for aligned boxes.

    #     Args:
    #         boxes (torch.Tensor): Aligned box with shape [n, 6].
    #         scores (torch.Tensor): Scores of each box.
    #         classes (torch.Tensor): Class of each box.
    #         thresh (float): Iou threshold for nms.

    #     Returns:
    #         torch.Tensor: Indices of selected boxes.
    #     """
    #     x1 = boxes[:, 0]
    #     y1 = boxes[:, 1]
    #     z1 = boxes[:, 2]
    #     x2 = boxes[:, 3]
    #     y2 = boxes[:, 4]
    #     z2 = boxes[:, 5]
    #     area = (x2 - x1) * (y2 - y1) * (z2 - z1)
    #     zero = boxes.new_zeros(1, )

    #     score_sorted = torch.argsort(scores)
    #     pick = []
    #     while (score_sorted.shape[0] != 0):
    #         last = score_sorted.shape[0]
    #         i = score_sorted[-1]
    #         pick.append(i)

    #         xx1 = torch.max(x1[i], x1[score_sorted[:last - 1]])
    #         yy1 = torch.max(y1[i], y1[score_sorted[:last - 1]])
    #         zz1 = torch.max(z1[i], z1[score_sorted[:last - 1]])
    #         xx2 = torch.min(x2[i], x2[score_sorted[:last - 1]])
    #         yy2 = torch.min(y2[i], y2[score_sorted[:last - 1]])
    #         zz2 = torch.min(z2[i], z2[score_sorted[:last - 1]])
    #         classes1 = classes[i]
    #         classes2 = classes[score_sorted[:last - 1]]
    #         inter_l = torch.max(zero, xx2 - xx1)
    #         inter_w = torch.max(zero, yy2 - yy1)
    #         inter_h = torch.max(zero, zz2 - zz1)

    #         inter = inter_l * inter_w * inter_h
    #         iou = inter / (area[i] + area[score_sorted[:last - 1]] - inter)
    #         iou = iou * (classes1 == classes2).float()
    #         score_sorted = score_sorted[torch.nonzero(
    #             iou <= thresh, as_tuple=False).flatten()]

    #     indices = boxes.new_tensor(pick, dtype=torch.long)
    #     return indices




    def _single_scene_multiclass_nms(self, bboxes, scores, input_meta):
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            input_meta (dict): Scene meta data.

        Returns:
            tuple[Tensor]: Predicted bboxes, scores and labels.
        """
        n_classes = scores.shape[1]
        with_yaw = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if with_yaw:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if with_yaw:
            box_dim = 7
        else:
            box_dim = 6
            nms_bboxes = nms_bboxes[:, :box_dim]

        return nms_bboxes, nms_scores, nms_labels