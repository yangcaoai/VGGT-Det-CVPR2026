# Copyright (c) OpenMMLab. All rights reserved.
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
                matcher_max_dynamic_samples=10
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
        pose_matrix = torch.stack(pose_matrix, dim=0).to(x.device, dtype=x.dtype)  # [16, 4, 4]
        axis_align_matrix = torch.stack(axis_align_matrix, dim=0).to(x.device, dtype=x.dtype)
        ones = torch.ones(batch_size, 1, num_boxes, device=x.device)  # [16, 1, 256]
        x_homogeneous = torch.cat([x, ones], dim=1)  # [16, 4, 256]

        x_global_homogeneous = torch.bmm(pose_matrix, x_homogeneous)  # [16, 4, 256]
        x_global_homogeneous = torch.bmm(axis_align_matrix, x_global_homogeneous)
        # x', y', z' = x', y', z' / w
        w = torch.clamp(x_global_homogeneous[:, 3:4, :], min=1e-8)
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
        return (center_pred, torch.exp(scale(self.size_head(x))), #/ avg_distance_tensor,
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
        all_centers = torch.cat([c.t() for c in center_preds], dim=0)  # (Total_Pred, 3)
        all_sizes = torch.cat([s.t() for s in size_preds], dim=0)      # (Total_Pred, 3) 
        all_cls = torch.cat([c.t() for c in cls_preds], dim=0)         # (Total_Pred, C)
        # all_objness = torch.cat([c.t() for c in objness_preds], dim=0)
        


        gt_centers = gt_bboxes.gravity_center
        gt_sizes = gt_bboxes.tensor[:, 3:6]

        all_pred_indices = []
        all_gt_indices = []
        offset = 0

        # time1 = time.time()
        # a = gt_sizes.cpu()
        for stage_idx in range(len(center_preds)):
            centers, sizes, cls_scores = center_preds[stage_idx].t(), size_preds[stage_idx].t(), cls_preds[stage_idx].t() #, objness_preds[stage_idx].t()
            cls_scores_softmax = F.softmax(cls_scores, dim=1)
            obj_scores = 1.0 - cls_scores_softmax[:, -1]
            n_predictions = centers.size(0)

            pred_indices, gt_indices = self.matcher._get_targets(
                centers, sizes, cls_scores, obj_scores,
                gt_centers, gt_sizes, gt_labels
            )
            
            all_pred_indices.append(pred_indices + offset)
            all_gt_indices.append(gt_indices)
            
            offset += n_predictions

        # time2 = time.time()
        # a = gt_sizes.cpu()

        pred_indices, gt_indices = torch.cat(all_pred_indices), torch.cat(all_gt_indices)
        # pred_indices, gt_indices = self.matcher._get_targets(
        #     all_centers, all_sizes, all_cls, all_objness,
        #     gt_centers, gt_sizes, gt_labels
        # )

        matched_centers = all_centers[pred_indices]
        matched_sizes = all_sizes[pred_indices]
        matched_cls = all_cls[pred_indices]
        # matched_objness = all_objness[pred_indices]
        matched_gt_centers = gt_centers[gt_indices]
        matched_gt_sizes = gt_sizes[gt_indices]
        matched_gt_labels = gt_labels[gt_indices]



        # #########################debug
        # # if self.visualize_3d_bbox:
        # # gt_boxes = data_samples.gt_instances_3d.bboxes_3d
        # pred_boxes = torch.cat(
        # scene_path = input_meta['img_path'][0].split('/')[-2]

        # gt_boxes = torch.cat(
        #     (matched_gt_centers, matched_gt_sizes), dim=1)
        # # max_giou, max_gt_box_idx, max_pred_box_idx = self.find_max_iou_from_center_size_boxes(bboxes_after_nms, gt_boxes)
        # # if isinstance(max_pred_box_idx, torch.Tensor):
        #     # max_pred_box_idx = max_pred_box_idx.item()
        # write_bbox(pred_boxes.detach().cpu().numpy(), self.visualize_path+'/'+'%s_match_pred_boxes.ply' % scene_path)
        
        # write_bbox(gt_boxes.detach().cpu().numpy(), self.visualize_path+'/'+'%s_match_gt_boxes.ply' % scene_path)


        # # write_bbox(gt_boxes[max_gt_box_idx_tmp].unsqueeze(0).cpu().numpy(), self.visualize_path+'/'+'%s_max_iou_gt_boxes.ply' % scene_path)

        # # write_bbox(bboxes_after_nms.cpu().numpy(), self.visualize_path+'/'+'%s_pred_boxes.ply' % scene_path)
        # # write_bbox(bboxes_after_nms[max_pred_box_idx].unsqueeze(0).cpu().numpy(), self.visualize_path+'/'+'%s_max_iou_%f_boxes.ply' % (scene_path, max_giou))
        # if input_points is not None:
        #     write_ply_rgb(input_points.detach().cpu().numpy(), self.visualize_path+'/'+'%s_gt_points.ply' % scene_path)

        # #########################debug




        # time3 = time.time()
        # a = gt_sizes.cpu()

        center_loss = F.l1_loss(matched_centers, matched_gt_centers) * self.loss_weights['center_loss']
        
        size_loss = F.l1_loss(matched_sizes, matched_gt_sizes) * self.loss_weights['size_loss']
        

        # time4 = time.time()
        # a = gt_sizes.cpu()
        cls_target = torch.ones((all_centers.shape[0]), device=all_centers.device) * self.n_classes
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
                data_samples=batch_data_samples[i]))
        return results

    def _predict_by_feat_single(self, center_preds, size_preds, cls_preds,
                                input_meta: dict, input_points, data_samples) -> InstanceData:
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

            bboxes = self._center_size_pred_to_bbox(centers, sizes)
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
            gt_boxes = data_samples.gt_instances_3d.bboxes_3d
            gt_boxes = torch.cat(
                (gt_boxes.gravity_center, gt_boxes.tensor[:, 3:6]), dim=1)
            scene_path = input_meta['img_path'][0].split('/')[-2]

            max_giou, max_gt_box_idx, max_pred_box_idx = self.find_max_iou_from_center_size_boxes(bboxes_after_nms, gt_boxes)
            if isinstance(max_pred_box_idx, torch.Tensor):
                max_pred_box_idx = max_pred_box_idx.item()
            write_bbox(gt_boxes.cpu().numpy(), self.visualize_path+'/'+'%s_gt_boxes.ply' % scene_path)

            max_gt_box_idx_tmp = max_gt_box_idx[:, max_pred_box_idx]
            if isinstance(max_gt_box_idx_tmp, torch.Tensor):
                max_gt_box_idx_tmp = max_gt_box_idx_tmp.item()

            write_bbox(gt_boxes[max_gt_box_idx_tmp].unsqueeze(0).cpu().numpy(), self.visualize_path+'/'+'%s_max_iou_gt_boxes.ply' % scene_path)

            write_bbox(bboxes_after_nms.cpu().numpy(), self.visualize_path+'/'+'%s_pred_boxes.ply' % scene_path)
            write_bbox(bboxes_after_nms[max_pred_box_idx].unsqueeze(0).cpu().numpy(), self.visualize_path+'/'+'%s_max_iou_%f_boxes.ply' % (scene_path, max_giou))
            if input_points is not None:
                write_ply_rgb(input_points.cpu().numpy(), self.visualize_path+'/'+'%s_gt_points.ply' % scene_path)

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
        # pred_corners = get_3d_box_batch_depth_tensor(
        #     all_sizes.unsqueeze(0), 
        #     torch.zeros(1, all_centers.size(0), device=all_centers.device),
        #     all_centers.unsqueeze(0)
        # )  # (1, Total_Pred, 8, 3)

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

        cost_class = -all_cls.sigmoid()[:, gt_labels]  # (Total_Pred, M)
        cost_center = torch.cdist(all_centers, gt_centers, p=1)  # (Total_Pred, M)
        cost_objness = -all_objness.sigmoid()       # (Total_Pred, M)
        # giou = generalized_box3d_iou(pred_corners, gt_corners, torch.tensor([gt_centers.size(0)]))[0]  # (Total_Pred, M)
        cost_giou =  -giou

        total_cost = (
            self.cost_weights['cls'] * cost_class +
            self.cost_weights['center'] * cost_center +
            self.cost_weights['obj_ness'] * cost_objness +
            self.cost_weights['giou'] * cost_giou
        )

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
        
        pred_tp_bbox = self._center_size_pred_to_bbox(all_centers, all_sizes)
        gt_tp_bbox = self._center_size_pred_to_bbox(gt_centers, gt_sizes)

        with torch.no_grad():
            giou = axis_aligned_bbox_overlaps_3d(pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0), mode='giou')
            assert giou.shape[0] == 1
            giou = giou.squeeze(0)  # (Total_Pred, M)

        cost_class = -all_cls.sigmoid()[:, gt_labels]
        cost_center = torch.cdist(all_centers, gt_centers, p=1)
        cost_objness = -all_objness.sigmoid()
        cost_giou = -giou

        total_cost = (
            self.cost_weights['cls'] * cost_class +
            self.cost_weights['center'] * cost_center +
            self.cost_weights['obj_ness'] * cost_objness +
            self.cost_weights['giou'] * cost_giou
        )

        pred_indices, gt_indices = linear_sum_assignment(total_cost.cpu().numpy())
        pred_indices = torch.from_numpy(pred_indices).long().to(all_centers.device)
        gt_indices = torch.from_numpy(gt_indices).long().to(all_centers.device)

        used_pred_mask = torch.zeros(giou.size(0), dtype=torch.bool, device=giou.device)
        used_pred_mask[pred_indices] = True

        iou_mask = giou > self.iou_threshold[0]

        dynamic_preds = []
        dynamic_gts = []

        max_iou_per_gt = giou.max(dim=0).values
        sorted_gt_indices = torch.argsort(max_iou_per_gt)

        for gt_idx in sorted_gt_indices:
            candidate_mask = iou_mask[:, gt_idx] & ~used_pred_mask
            candidate_preds = torch.nonzero(candidate_mask, as_tuple=True)[0]
            
            if candidate_preds.numel() == 0:
                continue

            giou_values = giou[candidate_preds, gt_idx]
            
            if self.matcher_max_dynamic_samples < len(giou_values):
                _, topk_indices = torch.topk(giou_values, k=self.matcher_max_dynamic_samples)
                selected_preds = candidate_preds[topk_indices]
            else:
                selected_preds = candidate_preds

            dynamic_preds.append(selected_preds)
            dynamic_gts.append(torch.full_like(selected_preds, gt_idx))

            used_pred_mask[selected_preds] = True

        if dynamic_preds:
            dynamic_preds = torch.cat(dynamic_preds)
            dynamic_gts = torch.cat(dynamic_gts)
        else:
            dynamic_preds = torch.empty(0, dtype=torch.long, device=giou.device)
            dynamic_gts = torch.empty(0, dtype=torch.long, device=giou.device)

        combined_preds = torch.cat([pred_indices, dynamic_preds])
        combined_gts = torch.cat([gt_indices, dynamic_gts])

        return combined_preds, combined_gts

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0]/2.0, centers[:, 1] - sizes[:, 1]/2.0,
            centers[:, 2] - sizes[:, 2]/2.0, centers[:, 0] + sizes[:, 0]/2.0,
            centers[:, 1] + sizes[:, 1]/2.0, centers[:, 2] + sizes[:, 2]/2.0
        ], -1)
    
















