# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from .nerf_utils.nerf_mlp import VanillaNeRF
from .nerf_utils.projection import Projector
from .nerf_utils.render_ray import render_rays

from .nerf_utils.save_rendered_img import save_rendered_img, save_rendered_img_Image, save_rendered_depth, save_predict_ray_depth, save_pcd, save_src_depth, MultiViewMixin, save_pcd_WorldSpace
from gs_src.model.encoder.epipolar.depth_predictor_monocular import DepthPredictorMonocular
from gs_src.geometry.projection import sample_image_grid
from gs_src.model.encoder.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, GaussianAdapter_3DVox
from gs_src.model.decoder import get_decoder
from gs_src.model.types import Gaussians

from jaxtyping import Float
from torch import Tensor
from einops import rearrange, repeat
import os
import matplotlib.pyplot as plt
import pickle

from .mvs_models.module import homo_warping
from .mvs_models.mvsnet import CostRegNet_3DGS
from .mvs_models.homography import *

from torch_scatter import scatter_max, scatter_add, scatter_min
from scipy.special import erf
from scipy.stats import norm
import math

from mmengine.logging import MessageHub


def knn(x, ref, k, maskself=False):
    '''
    x: (b,c,num_src)
    ref: (b, c, num_ref)
    k: top-k neigbour for each src
    assume x and ref are the same here!!
    '''
    
    inner = -2 * torch.matmul(x.transpose(2, 1), ref) #(B,num_src,num_ref)
    xx = torch.sum(x ** 2, dim=1, keepdim=True) #(B,1,num_src)
    yy = torch.sum(ref ** 2, dim=1, keepdim=True) #(B,1,num_ref)
    
    pairwise_distance = -yy - inner - xx.transpose(2, 1) #(B, num_src, num_ref)
    
    # mask out self
    if maskself:
        assert x.shape == ref.shape
        mask = torch.arange(xx.shape[2]) # (num_src)
        pairwise_distance[:, mask, mask] = -100000
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B,N,k)
    return idx


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, maskself=False, angular_dist_method='dist',
                         scene_center=(0, 0, 0)):
        '''
        Args:
            tar_pose: target pose [num_tgt, 4, 4]. c2w
            ref_poses: reference poses [num_src, 4, 4]. C2W. we pick neighbour views from this pool
            num_select: the number of nearest views to select
        Returns: the selected indices # (num_tgt, k)
        '''
        num_cams = len(ref_poses)
        num_select = min(num_select, num_cams-1)
        # batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0) # (N, 3, 3)

        if angular_dist_method == 'matrix':
            dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
        elif angular_dist_method == 'vector':
            tar_cam_locs = batched_tar_pose[:, :3, 3]
            ref_cam_locs = ref_poses[:, :3, 3]
            scene_center = np.array(scene_center)[None, ...]
            tar_vectors = tar_cam_locs - scene_center
            ref_vectors = ref_cam_locs - scene_center
            dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
        elif angular_dist_method == 'dist':
            tar_cam_locs = tar_pose[:, :3, 3].unsqueeze(0).transpose(2, 1) # (1, 3, N)
            ref_cam_locs = ref_poses[:, :3, 3].unsqueeze(0).transpose(2, 1) # (1, 3, N)
            
            neigh_ids = knn(tar_cam_locs, ref_cam_locs, k=num_select, maskself=maskself)[0] # (num_src, k)
        else:
            raise Exception('unknown angular distance calculation method!')

        # if tar_id >= 0:
        #     assert tar_id < num_cams
        #     dists[tar_id] = 1e3  # make sure not to select the target id itself

        # sorted_ids = np.argsort(dists)
        # selected_ids = sorted_ids[:num_select]
        # # print(angular_dists[selected_ids] * 180 / np.pi)
        return neigh_ids
        
def compute_std(cost):
    '''
    cost: (num_src, num_rays, num_depth). compute std along num_depth
    return:
        std: (num_src, num_rays)
    '''
    num_depth = cost.shape[-1]
    mean = torch.mean(cost, dim=-1, keepdim=True)
    var = torch.sum((cost - mean) ** 2, dim=-1) / num_depth # (num_src, num_rays)
    std = torch.sqrt(var + 1e-8)
    return std





@MODELS.register_module()
class MVSDet(Base3DDetector):

    def __init__(
            self,
            backbone: ConfigType,
            neck: ConfigType,
            neck_3d: ConfigType,
            bbox_head: ConfigType,
            prior_generator: ConfigType,
            n_voxels: List,
            voxel_size: List,
            head_2d: ConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptConfigType = None,
            #  pretrained,
            aabb: Tuple = None,
            near_far_range: List = None,
            N_samples: int = 64,
            N_rand: int = 2048,
            depth_supervise: bool = False,
            use_nerf_mask: bool = True,
            nerf_sample_view: int = 3,
            nerf_mode: str = 'volume',
            squeeze_scale: int = 4,
            rgb_supervision: bool = True,
            nerf_density: bool = False,
            render_testing: bool = False,
            gs_cfg=None,
            vis_dir=None,
            visualize_bbox = False,
            topk=3,
            alpha_thres=None,
            sigma_w=0.5):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone) # resnet
        self.neck = MODELS.build(neck) # FPN
        self.neck_3d = MODELS.build(neck_3d) # IndoorImVoxelNeck
        bbox_head.update(train_cfg=train_cfg) # NerfDetHead
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.head_2d = MODELS.build(head_2d) if head_2d is not None else None
        self.n_voxels = n_voxels # 40, 40, 16
        self.prior_generator = TASK_UTILS.build(prior_generator) # AlignedAnchor3DRangeGenerator
        self.voxel_size = voxel_size # [.16, .16, .2]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.aabb = aabb # ([-2.7, -2.7, -0.78], [3.7, 3.7, 1.78])
        self.near_far_range = near_far_range # [0.2, 5.0]
        self.N_samples = N_samples # 64
        self.N_rand = N_rand # 2048
        self.depth_supervise = depth_supervise
        self.projector = Projector() # seems: lift 2d features into 3d spaces.
        self.squeeze_scale = squeeze_scale # 4
        self.use_nerf_mask = use_nerf_mask # false
        self.rgb_supervision = rgb_supervision # true
        nerf_feature_dim = neck['out_channels'] // squeeze_scale
        # self.nerf_mlp = VanillaNeRF(
        #     net_depth=4,  # The depth of the MLP
        #     net_width=256,  # The width of the MLP
        #     skip_layer=3,  # The layer to add skip layers to.
        #     feature_dim=nerf_feature_dim + 6,  # + RGB original imgs
        #     net_depth_condition=1,  # The depth of the second part of MLP
        #     net_width_condition=128)
        self.nerf_mode = nerf_mode # 'image'
        self.nerf_density = nerf_density # true
        self.nerf_sample_view = nerf_sample_view # 20
        self.render_testing = render_testing # bool

        self.render_testing = render_testing

        # ----------------- depth estimation and gs --------------------
        # self.depth_predictor = DepthPredictorMonocular(
        #     gs_cfg.d_feature,
        #     gs_cfg.num_monocular_samples,
        #     gs_cfg.num_surfaces,
        #     gs_cfg.use_transmittance,
        # )
        self.gs_cfg = gs_cfg
        self.use_rgb_gaussian = gs_cfg.use_rgb_gaussian # whether use rgb as gaussian input, true
        
        self.gaussian_adapter = GaussianAdapter(gs_cfg.gaussian_adapter_cfg)
        if self.use_rgb_gaussian:
            gaussian_in_dim = self.gs_cfg.d_feature + 1 + 3 # feat + depth + rgb, d_feature: 256
            print('------------------- use rgb as gaussian input -------------------')
        else:
            gaussian_in_dim = self.gs_cfg.d_feature + 1
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                gaussian_in_dim,
                self.gs_cfg.num_surfaces * (2 + self.gaussian_adapter.d_in), # num_surfaces: 1, 
            ),
        )
        self.decoder = get_decoder(gs_cfg.decoder, gs_cfg.dataset) # decoder: splatting_cuda, dataset: background_color = [0.0, 0.0, 0.0]
        
        self.vis_dir = vis_dir
        
        # ----------------------- mvsnet -------------------------------
        self.depth_interval = (self.near_far_range[1] - self.near_far_range[0]) / self.gs_cfg.num_monocular_samples # num_monocular_samples: 12
        self.depth_values = np.arange(self.near_far_range[0], self.near_far_range[1], self.depth_interval,
                                         dtype=np.float32) # (n_depth,)
        assert len(self.depth_values) == self.gs_cfg.num_monocular_samples
        print("depth planes are: {}".format(self.depth_values))
        
        self.cost_regularization = CostRegNet_3DGS() # 3d unet
        
        self.topk = topk # pick top-k depth in detection.

        self.alpha_thres = alpha_thres
        self.sigma_w = sigma_w
        
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"], # (1, num_view, h*w, num_surface, num_gs)
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.gs_cfg.opacity_mapping
        # x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial) # final and initial are 0..
        x = 0
        exponent = 2**x

        # Map the probability density to an opacity.
        opacity =  0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
        
        return opacity

    def collect_proj(self, w2c, intr, neighbor_ids):
        ''' 
        w2c: (num_src, 4, 4) cuda
        intr: (4, 4). cuda or (num_src, 4, 4)
        neighbour_ids: (num_src, k)
        return:
            [(b, 4, 4), (b, 4, 4) ...] len=k. b=num_src. each (4, 4) is K[R|t], world2img
        '''
        if len(intr.shape) == 2:
            intr = intr.unsqueeze(0).repeat(w2c.shape[0], 1, 1) # (4, 4) -> (num_src, 4, 4)
        proj = torch.matmul(intr, w2c) # (num_src, 4, 4)
        num_src = neighbor_ids.shape[0]
        num_nei = neighbor_ids.shape[1]
        nei_projs = proj[neighbor_ids.view(-1)].view(num_src, num_nei, *proj.shape[1:]) # (num_src, k, 4, 4)
        nei_projs = torch.unbind(nei_projs, dim=1) 
        return proj, nei_projs
    
    def sample_depth_prob(self, prob_volume, off_pred, topk=3):
        '''
        sample depth from prob_volume.
        prob_volume: (num_src, d, h, w). probability.
        off_pred: (num_src, d, h, w). offset (0~1)
        topk: 
        return:
            est_depth: (num_src, k, h, w)
            est_density: (num_src, k, h, w)
        '''
        est_densities, est_idx = prob_volume.topk(k=topk, dim=1)
        
        est_depth = est_idx * self.depth_interval + torch.tensor(self.near_far_range[0], device=prob_volume.device) # (num_src, k, h, w)
        # add offset
        off_d = torch.gather(off_pred, 1,  est_idx) * torch.tensor(self.depth_interval, device=prob_volume.device) # (num_src, k, h, w)
        est_depth += off_d
        
        return est_depth, est_densities
    

    def sample_all_depth_prob(self, prob_volume, off_pred, topk=None):
        '''
        sample depth from prob_volume (without topk selection)
        prob_volume: (num_src, d, h, w). probability.
        off_pred: (num_src, d, h, w). offset (0~1)
        return:
            est_depth: (num_src, d, h, w)
            est_density: (num_src, d, h, w)
        '''
        d_dim = prob_volume.size(1)
        device = prob_volume.device
        
        est_idx = torch.arange(d_dim, device=device).view(1, -1, 1, 1).expand_as(prob_volume)
        
        est_densities = prob_volume
        
        base_depth = est_idx * self.depth_interval + self.near_far_range[0]
        
        off_d = off_pred * self.depth_interval
        est_depth = base_depth + off_d
        
        return est_depth, est_densities

    def compute_depth_coding(self, est_depth, est_densities):
        '''
        compute a weighted sum of depth as the estimated depth
        est_depth: (num_src, num_gs, h, w)
        est_densities: (num_src, num_gs, h, w)
        return:
            depth_coding: (num_src, 1, h, w)
        '''
        # normalize probility
        est_densities = est_densities / est_densities.sum(dim=1, keepdim=True)  # (num_src, num_gs. h, w)
        depth_coding = torch.sum(est_depth * est_densities, dim=1, keepdim=True) # (num_src, 1, h, w)
        return depth_coding
    
    def compute_avg_depth(self, prob_volume, off_pred):
        '''
        sample depth from prob_volume.
        prob_volume: (num_src, d, h, w). probability.
        off_pred: (num_src, d, h, w). offset (0~1)
        depth_values: (num_src, d)
        return:
            est_depth: (num_src, k, h, w)
            est_density: (num_src, k, h, w)
        '''
        est_densities, est_idx = prob_volume.topk(k=self.gs_cfg.num_monocular_samples, dim=1) # (num_src, d, h, w) select all depth candidate
        
        est_depth = est_idx * self.depth_interval + torch.tensor(self.near_far_range[0], device=prob_volume.device) # (num_src, d, h, w)
        # add offset
        off_d = torch.gather(off_pred, 1,  est_idx) * torch.tensor(self.depth_interval, device=prob_volume.device) # (num_src, d, h, w)
        est_depth += off_d
        
        final_pred = torch.sum(est_depth * est_densities, dim=1) # (num_src, h, w)
        
        return final_pred
    
    def process_rgb_raw(self, orig_rgb, ratio, height, width, src_id):
        '''
        orig_rgb: (n_src, 3, h, w)
        resize it to (height, width)
        return:
            new_rgb: (1, num_nei, h*w, 3)
        '''
        assert ratio == 4 # (240, 320) -> (60,80)
        num_nei = len(src_id)
        new_rgb = F.interpolate(
                orig_rgb[src_id], scale_factor=1./ratio,
                mode='bilinear') # (n_nei, 3, height, width)
        new_rgb = new_rgb[:,:, :height, :width]
        new_rgb = new_rgb.view(*new_rgb.shape[:2], -1).transpose(2,1).unsqueeze(0) # (n_nei, 3, h*w) -> (1, n_nei, h*w, 3)
        return new_rgb
        
        
    def extract_feat(self,
                     batch_inputs_dict: dict,
                     batch_data_samples: SampleList,
                     mode,
                     depth=None, # list. [bs=1, bs=2, ...] each sample (num_src, h, w)
                     ray_batch=None,
                     visualization_dump=None,
                     save_dir=None):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instances` of `gt_panoptic_seg` or `gt_sem_seg`

        Returns:
            Tuple:
            - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
            - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
            - torch.Tensor: 2D features if needed.
            - dict: The nerf rendered information including the
                'output_coarse', 'gt_rgb' and 'gt_depth' keys.
        """
        img = batch_inputs_dict['imgs'] # (bs, 40, 3, 240, 320)
        img = img.float()
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_size = img.shape[0]

        if len(img.shape) > 4:
            img = img.reshape([-1] + list(img.shape)[2:])
            x = self.backbone(img)
            x = self.neck(x)[0]
            x = x.reshape([batch_size, -1] + list(x.shape[1:])) # (bs, 40, f, 60, 80)
        else:
            x = self.backbone(img)
            x = self.neck(x)[0] # x.shape = ([1, 40, 256, 60, 80])

        # if depth is not None:
        #     depth_bs = depth.shape[0]
        #     assert depth_bs == batch_size
        #     depth = batch_inputs_dict['depth']
        #     depth = depth.reshape([-1] + list(depth.shape)[2:])

        features_2d = self.head_2d.forward(x[-1], batch_img_metas) \
            if self.head_2d is not None else None

        stride = img.shape[-1] / x.shape[-1]  
        assert stride == 4
        stride = int(stride)

        # ----------- estimate depth for each pixel_feat. 'img_shape' is the shape before padding (only resize).
    
        # -------- generate feature volume --------
        volumes, valids = [], []
        rgb_preds = []
        count = 0
        est_depth_crop = []
        num_src = x.shape[1]
        weight_gap_list = []
        src_rmse_list = []
        for feature, img_meta in zip(x, batch_img_metas):
            angles = features_2d[
                0] if features_2d is not None and mode == 'test' else None
            projection = self._compute_projection(img_meta, stride, 
                                                  angles).to(x.device) # project lidar (3d) to img (2d) 
            points = get_points(
                n_voxels=torch.tensor(self.n_voxels),
                voxel_size=torch.tensor(self.voxel_size),
                origin=torch.tensor(img_meta['lidar2img']['origin'])).to(
                    x.device)

            # for each scene, we re-compute its src feat size using their own img_meta!!!
            height = img_meta['img_shape'][0] // stride # feature-level src size
            width = img_meta['img_shape'][1] // stride
            
            src_w2c = torch.tensor(np.array(img_meta['lidar2img']['extrinsic']), device=x.device) # stack to tensor (num_src, 4, 4)
            src_intrinsic = torch.tensor(np.array(img_meta['lidar2img']['intrinsic']), device=x.device) # (4, 4). correspond to original img resolution. or (num_src, 4, 4)
            # whether intrinsic is list: intrin_list_flag
            intrin_list_flag = isinstance(img_meta['lidar2img']['intrinsic'], list)
            ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
            src_feat_intrinsic = src_intrinsic.clone()
            if not intrin_list_flag: # (4,4)
                src_feat_intrinsic[:2] /= ratio # (4,4). correspond to src_feat intrisinc
            else: # (num_src, 4, 4)
                src_feat_intrinsic[:,:2] /= ratio
            
            # --------------- estimate depth ----------
            # 1. get neighbour frames
            k=min(2, num_src-1) # incase num_src is very few....
            src_c2w = src_w2c.inverse() # c2w (num_src, 4, 4) 
            neighbor_ids =  get_nearest_pose_ids(src_c2w, src_c2w, k, maskself=True) # (num_src, k)
            
            # 2. plane sweep
            # useful_feature = feature[:, :, :height, :width]
            num_depth = self.gs_cfg.num_monocular_samples # number of depth planes = 12
            ref_volume = feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) # (num_src, c, num_depth, h, w). treat num_src as bs for parallel
            volume_sum = ref_volume # (b, c, num_depth, h, w)
            volume_sq_sum = ref_volume ** 2
            del ref_volume
            
            nei_features = feature[neighbor_ids.view(-1)].view(num_src, k, *feature.shape[1:]) # (num_src, k, c, h, w)
            # print(nei_features.shape)=([40, 2, 256, 60, 80])
            
            nei_features = torch.unbind(nei_features, dim=1) # [(bs, c, h, w), (bs, c, h, w), ...] len=k
            ref_proj, nei_projs = self.collect_proj(src_w2c, src_feat_intrinsic, neighbor_ids) # (b, 4, 4). [(bs, 4, 4), (bs, 4, 4), ...]. intrinsic should change !!!
            
            depth_values = torch.tensor(self.depth_values, device=x.device).unsqueeze(0).repeat(num_src, 1)
            # we follow definition in mvsnet. neigbour is src in mvsnet.
            # 3. cost volume
            for nei_fea, nei_proj in zip(nei_features, nei_projs):
                # warpped features
                warped_volume = homo_warping(nei_fea, nei_proj, ref_proj, depth_values)
                if self.training:
                    # print("---------------------------- train -----------------------------")
                    volume_sum = volume_sum + warped_volume # (b, c, num_depth, h, w)
                    volume_sq_sum = volume_sq_sum + warped_volume ** 2
                else:
                    # print("--------------------------------- not train ------------------------")
                    # TODO: this is only a temporal solution to save memory, better way?
                    volume_sum += warped_volume
                    volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                del warped_volume
            # aggregate multiple feature volumes by variance
            volume_variance = volume_sq_sum.div_(k+1).sub_(volume_sum.div_(k+1).pow_(2)) # eqn 2 in mvsnet. should include src view itself. (bs, c, num_depth, h, w)

            # 4. cost volume regularization
            cost_reg, off_pred = torch.unbind(self.cost_regularization(volume_variance), dim=1) # (num_src, 1, d, h, w), (num_src, 1, d, h, w)
            cost_reg = cost_reg.squeeze(1)
            prob_volume = F.softmax(cost_reg, dim=1) # (num_src, d, h, w). depth probability
            # print(prob_volume[0,:,10,0])
            
            off_pred = torch.sigmoid(off_pred.squeeze(1)) # (num_src, d, h, w), an offset for the depth
            
            # 5. this depth is z depth!!
            est_depth, est_densities = self.sample_depth_prob(prob_volume, off_pred, topk=self.topk)

            ############idea1#############
            est_depth = est_depth[:, :, :height, :width]
            est_densities = est_densities[:, :, :height, :width]
            # depth_coding = self.compute_depth_coding(est_depth, est_densities) # (num_src, 1, h, w) weighted mean of topk depth..
            depth_coding = self.compute_avg_depth(prob_volume, off_pred)[:,:height, :width].unsqueeze(1)
            
            est_depth_crop.append(depth_coding)

            est_depth = est_depth.view(*est_depth.shape[:2], -1).transpose(2, 1).unsqueeze(2) # (num_src, h*w, num_surface, num_gs)


            # 6. convert depth to gaussian input
            if not intrin_list_flag: # for scannet
                cur_depth_scale = self.compute_depth_scale(height, width, x.device, 
                                    img_meta, stride, num_src)[0] # (num_src, height*width, 1)
            else:
                cur_depth_scale = self.compute_depth_scale_MultiIntrin(height, width, x.device, 
                                    img_meta, stride, num_src)[0] # (num_src, height*width, 1)
            
            est_ray_depth = est_depth / (cur_depth_scale.unsqueeze(-1).repeat(1,1,1,est_depth.shape[-1]) + 1e-8) # (num_src, h*w, num_surface, num_gs)
            est_densities = est_densities.view(*est_densities.shape[:2], -1).transpose(2, 1).unsqueeze(2) # (num_src, h*w, num_surface, num_gs)
            
            # feat volume

            volume, valid, weight_gap, src_rmse = backproject_Weigh(feature[:, :, :height, :width], points, # a scene's feat: (num_src, d, 59, 80)
            ############idea1#############
            # volume, valid, weight_gap, src_rmse = backproject_Weigh_differentiable_propagation(feature[:, :, :height, :width], points,
                                        projection, 
                                        est_depth, # (num_view, h*w, num_surface, num_gs)
                                        self.voxel_size,
                                        est_densities,
                                        gt_depth=depth[count] if depth != None else None,
                                        save_dir=save_dir,
                                        img_meta=img_meta,
                                        depth_mean=depth_coding.squeeze(1),
                                        # alpha_thres=self.alpha_thres, sigma_w=self.sigma_w,
                                        ) # volume: (40,256,40,40,16) valid:(40,1,40,40,16)
            weight_gap_list.append(weight_gap)
            src_rmse_list.append(src_rmse)
            
            volume_sum = volume.sum(dim=0) # volume_sum = ([256, 40, 40, 16])
            # cov_valid = valid.clone().detach()
            valid = valid.sum(dim=0)
            volume_mean = volume_sum / (valid + 1e-8)
            volume_mean[:, valid[0] == 0] = .0
            
            
            # ----- novel view rendering -----
            if ray_batch is not None: # rendering novel view
                # choose all src_views. should not set the number directly because train and test are different!!
                # chosen_src = num_src
                # src_id = torch.randint(0, feature.shape[0], (chosen_src,))
                # src_id = torch.arange(chosen_src)
                
                tgt_extrinsic = np.array(ray_batch["c2w"]) # (len, bs, 4, 4)
                tgt_extrinsic = torch.tensor(tgt_extrinsic, device=feature.device).transpose(1,0) # (bs, n_tgt, 4, 4)
                tgt_extrinsic = tgt_extrinsic[count].unsqueeze(0) # (1, n_tgt, 4, 4)
                num_tgt = tgt_extrinsic.shape[1]
                
                
                # for every tgt view, we find its top-3 nearest src views.
                render_src_id = get_nearest_pose_ids(tgt_extrinsic[0], src_c2w, num_select=3, maskself=False) # (num_tgt, 3)
                # check if confict
                src_id = torch.unique(render_src_id.view(-1))
                chosen_src = len(src_id)
                # print(src_id, chosen_src)
                
                # Convert the features and depths into Gaussians.
                xy_ray, _ = sample_image_grid((height, width), volume.device) # coordinates in 0-1. (x,y,2)
                xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy") # (x*y, 1, 2)
                
                # get context_intrinsic: (original_intrinsic / scale) / (h, w)
                context_intrinsic = src_feat_intrinsic.clone()
                if not intrin_list_flag: # scannet
                    # normalize intrinisc! so aligns with normalized coordinates(0~1)
                    context_intrinsic[0] /= width # width
                    context_intrinsic[1] /= height # height 
                    context_intrinsic = context_intrinsic.unsqueeze(0).unsqueeze(0).repeat(1,chosen_src,1,1)[:,:,:3,:3] # (1, num_img, 3, 3)
                else: # (num_src, 4, 4)
                    # normalize intrinisc! so aligns with normalized coordinates(0~1)
                    context_intrinsic[:,0] /= width # width
                    context_intrinsic[:,1] /= height # height 
                    context_intrinsic = context_intrinsic[src_id, :3, :3].unsqueeze(0) # (1, num_img, 3, 3)
                # reshape extrinsic. only select some src extrinsic. # TODO nvs requires c2w extrinsic, check!!
                # context_w2c = list(map(torch.tensor, img_meta['lidar2img']['extrinsic'])) # extrinsic: list, len=40.
                # context_w2c = torch.stack(context_w2c, dim=0).unsqueeze(0).to(feature.device) # (1, n_img, 4, 4)
                context_c2w = src_c2w.unsqueeze(0) # (1, n_img, 4, 4)
                context_extrinsic = context_c2w[:, src_id, :, :] # (1, num_src, 4, 4)
                
                # reshape feature (num_img, f, h, w) to gs_feature
                gs_feature = feature[src_id, :, :height, :width].view(chosen_src, feature.shape[1], -1).transpose(1,2).unsqueeze(0) # (1, num_img, h*w, f)
                # concat feature with depth
                depth_coding = depth_coding.view(*depth_coding.shape[:2], -1).unsqueeze(0).transpose(3,2) # (1, num_src, h*w, 1)
                # add rgb feat
                gs_feature = torch.cat([gs_feature, depth_coding[:,src_id]], dim=-1) # (1, num_src, h*w, f+1)
                # add rgb as input
                if self.use_rgb_gaussian:
                    rgb_raw = self.process_rgb_raw(ray_batch['denorm_images'][count], stride, height, width, src_id) # (1, num_nei, h*w, 3)
                    gs_feature = torch.cat([gs_feature, rgb_raw], dim=-1)
                
                gaussians = rearrange(
                    self.to_gaussians(gs_feature),
                    "... (srf c) -> ... srf c",
                    srf=self.gs_cfg.num_surfaces,
                ) # (bs, view, h*w, num_surface, 84)
                offset_xy = gaussians[..., :2].sigmoid() # why offset? (bs, num_img, h*w, num_surface, 2)
                pixel_size = 1 / torch.tensor((width, height), dtype=torch.float32, device=feature.device)
                xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
                # print(xy_ray.shape)
                # gpp = self.gs_cfg.gaussians_per_pixel
                opacity = torch.max(prob_volume, dim=1)[0] # (num_src, h, w)
                opacity = opacity[src_id, :height, :width].view(chosen_src, -1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (n_view, h*w) -> (1, num_view, h*w, num_surface, num_gs)
                ray_depth_coding = depth_coding / (cur_depth_scale.unsqueeze(0) + 1e-8) # gaussian_adapter needs ray depth!! (1, num_src, h*w, 1)
                
                # generate gaussian
                gaussians = self.gaussian_adapter.forward(
                    rearrange(context_extrinsic, "b v i j -> b v () () () i j"),
                    rearrange(context_intrinsic, "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    # est_ray_depth[src_id].unsqueeze(0), # (bs, num_view, h*w, num_surface, num_gs)
                    ray_depth_coding[:, src_id].unsqueeze(-2), # (bs, num_view, h*w, num_surface, num_gs)
                    # self.map_pdf_to_opacity(est_densities[src_id].unsqueeze(0)), # (1, num_view, h*w, num_surface, num_gs)
                    opacity,
                    rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
                    (height, width),
                )
                
                # Dump visualizations if needed.
                if visualization_dump is not None:
                    visualization_dump[count] = {}
                    visualization_dump[count]["depth"] = rearrange(
                        est_ray_depth[src_id].unsqueeze(0), "b v (h w) srf s -> b v h w srf s", h=height, w=width
                    )
                    visualization_dump[count]["scales"] = rearrange(
                        gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
                    )
                    visualization_dump[count]["rotations"] = rearrange(
                        gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
                    )
                    visualization_dump[count]["src_h"] = height
                    visualization_dump[count]["src_w"] = width
                    visualization_dump[count]["src_id"] = src_id
                
                opacity_multiplier = 1
                gaussians = Gaussians(
                        rearrange(
                        gaussians.means,
                        "b v r srf spp xyz -> b (v r srf spp) xyz",
                        ),
                        rearrange(
                        gaussians.covariances,
                        "b v r srf spp i j -> b (v r srf spp) i j",
                        ),
                        rearrange(
                        gaussians.harmonics,
                        "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                        ),
                        rearrange(
                        opacity_multiplier * gaussians.opacities,
                        "b v r srf spp -> b (v r srf spp)",
                        ),
                        )
                
                if visualization_dump != None:
                    visualization_dump[count]['gaussians'] = gaussians
                
                # ------------- decode to image. ray_batch is the info for a batch?
                # extrinsinc
                # tgt_extrinsic = np.array(ray_batch["c2w"]) # (len, bs, 4, 4)
                # tgt_extrinsic = torch.tensor(tgt_extrinsic, device=feature.device).transpose(1,0) # (bs, n_tgt, 4, 4)
                # tgt_extrinsic = tgt_extrinsic[count].unsqueeze(0) # (1, n_tgt, 4, 4)
                # num_tgt = tgt_extrinsic.shape[1]
                
                # intrinsic
                if not intrin_list_flag: # (4, 4) scannet
                    tgt_intrinsic = torch.tensor(np.array(ray_batch['intrinsic']), device=feature.device)[count] # corresponds to resized tgt image (bs,4,4)[count] = (4, 4)
                    # normalize
                    tgt_shape = ray_batch['nerf_sizes'][0][count] # eg. (239, 320, 3)
                    tgt_intrinsic[0] /= tgt_shape[1] # /width
                    tgt_intrinsic[1] /= tgt_shape[0] # /height
                    tgt_intrinsic = tgt_intrinsic[:3,:3].unsqueeze(0).unsqueeze(0).repeat(1, num_tgt, 1, 1) # (bs, n_view, 3, 3)
                else:
                    tgt_intrinsic = torch.tensor(np.array(ray_batch['intrinsic']), device=feature.device).transpose(1,0)[count] # corresponds to resized tgt image (bs,n_tgt,4,4)[count] = (n_tgt,4, 4)
                    # normalize
                    tgt_shape = ray_batch['nerf_sizes'][0][count] # eg. (239, 320, 3)
                    tgt_intrinsic[:,0] /= tgt_shape[1] # /width
                    tgt_intrinsic[:,1] /= tgt_shape[0] # /height
                    tgt_intrinsic = tgt_intrinsic[:, :3,:3].unsqueeze(0) # (bs, n_view, 3, 3)
                
                # near and far using depth range
                near = torch.tensor(self.near_far_range[0]).unsqueeze(0).repeat(1,num_tgt).to(feature.device) # (bs, n_tgt)
                far  = torch.tensor(self.near_far_range[1]).unsqueeze(0).repeat(1,num_tgt).to(feature.device) # (bs, n_tgt)
                # decode
                # print(img_meta['img_shape'])
                scene = img_meta['img_path'][0].split('/')[-2]
                output = self.decoder.forward(
                gaussians,
                tgt_extrinsic, # (bs, n_view, 4, 4)
                tgt_intrinsic, # (bs, n_view, 3, 3)
                near, # (bs, n_view)
                far, # (bs_n_view)
                (tgt_shape[0], tgt_shape[1]), # should corresponds to gt target image shape! which only been resized!! No padding!!
                depth_mode=None,
                scene=scene
                )
                
                rgb_preds.append(output) # torch.Size([1, 2, 3, 120, 160])
            

                
            volume = volume_mean
            volume[:, valid[0] == 0] = .0

            volumes.append(volume)
            valids.append(valid)
            
            # # --------------- rebuttal add photometric loss. depth_coding: (1,40,4720,1)
            # warped_img, mask, chose_ref_imgs = self.warp_img(ray_batch['denorm_images'][count], neighbor_ids, src_w2c, src_feat_intrinsic, tmp_depth_coding)
            # warped_img_list.append(warped_img)
            # mask_list.append(mask)
            # chose_ref_img_list.append(chose_ref_imgs)
            # # --------------------------- end rebuttal -------------------------------
            
            count += 1
        x = torch.stack(volumes)
        x = self.neck_3d(x)

        return x, torch.stack(valids).float(), features_2d, rgb_preds, est_depth_crop, weight_gap_list, src_rmse_list
    
    def warp_img(self, ref_imgs, neighbor_ids, ref_w2c, ref_feat_intrinsic, ref_depth):
        '''
        ref_imgs: (n_src, 3, img_h, img_w). denorm imgs (0-1) (240, 320)
        neighbour_ids: (n_src, k)
        ref_w2c: (n_src, 4, 4)
        ref_feat_intrinsic: (4,4) img = 4*feat
        ref_depth: (num_src, 1, h, w) (60,80)
        '''
        # warped_img_list = []
        # mask_list = []
        # reprojection_losses = []
        
        h = ref_depth.shape[2]
        w = ref_depth.shape[3]
        # ref_img: resize to depth size. also correspond to intrinsic
        total_ref = ref_imgs.shape[0]
        ref_imgs = F.interpolate(ref_imgs, scale_factor=0.25, mode='bilinear') # resize to feat-level img. so correspond to feat_intrinsic
        ref_imgs = ref_imgs[:,:,:h, :w]
        ref_imgs = ref_imgs.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        # random select n ref views to save computation
        n_select = min(10, total_ref)
        ref_id = torch.randperm(total_ref)[:n_select] 
        chose_ref_imgs = ref_imgs[ref_id] # (n_select, h, w, c)
        
        # src_imgs:
        n_nei = neighbor_ids.shape[1]
        src_imgs = ref_imgs[neighbor_ids.view(-1)].view(total_ref, n_nei, *ref_imgs.shape[1:]) # (n_ref, n_nei, h, w, 3)
        src_imgs = src_imgs[ref_id, 0] # (n_select, 3, h, w)
        # print(src_imgs.shape)
        
        # ref_cam: (B, 2,3,4)
        full_cam = torch.stack([ref_w2c, ref_feat_intrinsic.unsqueeze(0).expand(total_ref, -1, -1)], dim=1) # (n_ref, 2, 4, 4)
        full_cam = full_cam[:,:,:3, :] # (n_ref, 2, 3, 4)
        
        ref_cam = full_cam[ref_id] # (n_select, 2, 3, 4)
        
        # view cam: (B, 2, 3, 4). neighbours view 
        view_cam = full_cam[neighbor_ids.view(-1)].view(total_ref, n_nei, *full_cam.shape[1:])
        view_cam = view_cam[ref_id, 0] # (n_select, 2, 3, 4)
        
        # warp view_img to the ref_img using the dmap of the ref_img
        warped_img, mask = inverse_warping(src_imgs, ref_cam, view_cam, ref_depth[ref_id]) # (b, h, w, c)
        # warped_img_list.append(warped_img)
        # mask_list.append(mask)
        
        # reconstr_loss = compute_reconstr_loss(warped_img, chose_ref_imgs, mask, simple=False) # whether simple !!!
        # valid_mask = 1 - mask  # replace all 0 values with INF
        # reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

    

        # # top-k operates along the last dimension, so swap the axes accordingly
        # reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        # top_vals = torch.neg(top_vals)
        # # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        # top_mask = top_mask.float()
        # top_vals = torch.mul(top_vals, top_mask)
        # # print('top_vals: {}'.format(top_vals.shape))

        # reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        # return reconstr_loss
        
        return warped_img, mask, chose_ref_imgs
        
        
    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj: `DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """ 
        ray_batchs = {}
        batch_images = []
        batch_depths = []
        if 'images' in batch_data_samples[0].gt_nerf_images:
            for data_samples in batch_data_samples:
                image = data_samples.gt_nerf_images['images']
                batch_images.append(image)
        batch_images = torch.stack(batch_images)

        if 'depths' in batch_data_samples[0].gt_nerf_depths:
            for data_samples in batch_data_samples:
                depth = data_samples.gt_nerf_depths['depths']
                batch_depths.append(depth)
        batch_depths = torch.stack(batch_depths)

        # print(batch_inputs_dict.keys())
        if 'raydirs' in batch_inputs_dict.keys():
            ray_batchs['ray_o'] = batch_inputs_dict['lightpos']
            ray_batchs['ray_d'] = batch_inputs_dict['raydirs']
            ray_batchs['gt_rgb'] = batch_images
            ray_batchs['gt_depth'] = batch_depths
            ray_batchs['nerf_sizes'] = batch_inputs_dict['nerf_sizes']
            ray_batchs['denorm_images'] = batch_inputs_dict['denorm_images'] # (bs, n_src, 3, h, w) cuda  
            ray_batchs['c2w'] = batch_inputs_dict['c2w'] # a list. len=10, corresponds to 10 tgt view under bs=1
            ray_batchs['intrinsic'] = batch_inputs_dict['intrinsic'] # a list. len=bs, corresponds to 10 tgt view under bs=1
            # print(ray_batchs['gt_rgb'].shape)
            
            x, valids, features_2d, rgb_preds, est_depth_crop, _, _ = self.extract_feat( # return x, torch.stack(valids).float(), features_2d, rgb_preds, est_depth_crop, weight_gap_list, src_rmse_list
                batch_inputs_dict,
                batch_data_samples,
                'train',
                depth=None,
                ray_batch=ray_batchs)
        else:
            x, valids, features_2d, rgb_preds, est_depth_crop = self.extract_feat(
                batch_inputs_dict, batch_data_samples, 'train')
        x += (valids, )
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)

        # if self.head_2d is not None:
        #     losses.update(
        #         self.head_2d.loss(*features_2d, batch_data_samples)
        #     )
        if len(ray_batchs) != 0 and self.rgb_supervision:
            losses.update(self.nvs_loss_func(rgb_preds, ray_batchs))
        
        # 
        if self.depth_supervise:
            depth = batch_inputs_dict['depth']
            losses.update(self.depth_loss_func_new(est_depth_crop, depth))
        
        # # rebuttal photometric loss
        # losses.update(self.photometric_loss(warped_img_list, mask_list, chose_ref_img_list))
        
        
        
        return losses

    def photometric_loss(self, warped_img_list, mask_list, chose_ref_img_list):
        n = len(warped_img_list)
        reprojection_losses = []
        for i in range(n):
            warped_img = warped_img_list[i]
            mask = mask_list[i]
            chose_ref_img = chose_ref_img_list[i]
            
            reconstr_loss = compute_reconstr_loss(warped_img, chose_ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)
            
        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        # print('top_vals: {}'.format(top_vals.shape))

        reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        # self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss 
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return dict(loss_pc=reconstr_loss)

        
        
    def nvs_loss_func(self, rgb_pred, ray_batchs):
        loss = 0
        for i, ret in enumerate(rgb_pred): # for each scene
            rgb = ret.color[0] # (n_tgt, c, h, w). # TODO: why rendering create a bs dim?
            gt  = ray_batchs['gt_rgb'][i] # (n_tgt, c, h, w)
            assert gt.max() <= 1 and gt.min() >=0
            assert rgb.shape == gt.shape
            if self.use_nerf_mask: # false
                loss += torch.sum(
                    masks.unsqueeze(-1)*(rgb - gt)**2)/(masks.sum() + 1e-6)
            else:
                loss += torch.mean((rgb - gt)**2)
        return dict(loss_nvs=loss)


    def depth_loss_func_new(self, est_depth, gt_depth):
        '''
        est_depth: list. len=bs. each sample: (num_src, 1, height, width). eg (59, 80)
        gt_depth: list. len=bs. each sample: (num_src, h, w) eg. (239, 320)
        '''
        bs = len(est_depth)
        loss = 0.
        for i in range(bs):
            # 1. resize depth
            cur_est_depth = est_depth[i].squeeze(1) # (num_src, height, width)
            height = cur_est_depth.shape[1]
            width = cur_est_depth.shape[2]
            
            cur_gt_depth = F.interpolate(
            gt_depth[i].unsqueeze(1), size=(height, width),
            mode='bilinear').squeeze(1) # (num_src, height, width)
            # print(cur_est_depth.shape, cur_gt_depth.shape)
            # loss
            #mask out 0
            tmp_mask = cur_gt_depth>0
            loss += torch.mean(torch.abs(cur_est_depth[tmp_mask] - cur_gt_depth[tmp_mask]))
        
        return dict(loss_depth=loss)
    
    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`NeRFDet3DDataSample`]: Detection results of the
            input images. Each NeRFDet3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C = 6.
        """
        ray_batchs = {}
        batch_images = []
        batch_depths = []
        if 'images' in batch_data_samples[0].gt_nerf_images:
            for data_samples in batch_data_samples:
                image = data_samples.gt_nerf_images['images']
                batch_images.append(image)
        batch_images = torch.stack(batch_images)

        if 'depths' in batch_data_samples[0].gt_nerf_depths:
            for data_samples in batch_data_samples:
                depth = data_samples.gt_nerf_depths['depths']
                batch_depths.append(depth)
        batch_depths = torch.stack(batch_depths)

        if 'raydirs' in batch_inputs_dict.keys():
            ray_batchs['ray_o'] = batch_inputs_dict['lightpos']
            ray_batchs['ray_d'] = batch_inputs_dict['raydirs']
            ray_batchs['gt_rgb'] = batch_images
            ray_batchs['gt_depth'] = batch_depths
            ray_batchs['nerf_sizes'] = batch_inputs_dict['nerf_sizes']
            ray_batchs['denorm_images'] = batch_inputs_dict['denorm_images']
            ray_batchs['c2w'] = batch_inputs_dict['c2w']
            ray_batchs['intrinsic'] = batch_inputs_dict['intrinsic']
            
            
            if 'depth' in batch_inputs_dict:
                src_gt_depth = batch_inputs_dict['depth']
            else:
                src_gt_depth = None
            
            if self.vis_dir != None: # only during test time, vis_dir work
                src_depth_save_dir = os.path.join(self.vis_dir, 'src_depth')
                os.makedirs(src_depth_save_dir, exist_ok=True)
                visualization_dump = {} # save gaussian output for visualization
            else: 
                src_depth_save_dir = None
                visualization_dump = None
                
            x, valids, features_2d, rgb_preds, est_depth_crop, weight_gap, src_rmse_list = self.extract_feat(
                batch_inputs_dict,
                batch_data_samples,
                'test',
                depth=src_gt_depth,
                ray_batch=ray_batchs,
                visualization_dump=visualization_dump,
                save_dir=src_depth_save_dir)
        else:
            x, valids, features_2d, rgb_preds, _, weight_gap = self.extract_feat(
                batch_inputs_dict, batch_data_samples, 'test')
        x += (valids, )
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        
        # save bbox to pickle
        # for i in range(len(batch_data_samples)):
        #     scene = batch_data_samples[i].metainfo['img_path'][0].split('/')[-2]
        #     save_path = os.path.join('bbox_pred_scannet', scene + '_pred_bbox.pickle')
        #     pred_bbox = results_list[i].bboxes_3d.tensor.clone() # (n, 7)
        #     pred_bbox[:,:3] = results_list[i].bboxes_3d.gravity_center.clone()
        #     pred_bbox = pred_bbox.cpu().numpy() 
        #     pred_score = results_list[i].scores_3d.clone().cpu().numpy()
        #     pred_label = results_list[i].labels_3d.clone().cpu().numpy()
        #     pred = {'boxes': pred_bbox, 'scores': pred_score, "labels": pred_label}
        #     with open(save_path, 'wb') as f:
        #         pickle.dump(pred, f)
        
        # evaluate nvs. only support bs =1.
        # rgb_preds[0].color.shape: (1, n_tgt, 3, 239, 320)
        # batch_images.shape: (bs, n_tgt, 3, 239, 320)
        
        if self.rgb_supervision:
            save_dir = self.vis_dir # only during test, save_dir is not None
            for i in range(len(rgb_preds)):
                psnr, ssim, rmse = save_rendered_img_Image(rgb_preds[i].color[0], batch_images[i], batch_data_samples[i].metainfo, save_dir=save_dir)
                # print(psnr, ssim, rmse)
                
                if rgb_preds[i].depth != None: # overwirte rmse, otherwise is 0
                    rmse = save_rendered_depth(rgb_preds[i].depth[0], batch_depths[i], batch_data_samples[i].metainfo, save_dir=save_dir)

                # if save_dir != None:
                #     cur_vis_dump = visualization_dump[i]
                #     device = batch_depths[i].device
                #     # print(device)
                #     save_pcd(cur_vis_dump, batch_data_samples[i].metainfo, save_dir, device)

                # # save input data to visualize epipolar lines
                # input_to_save = {}
                # input_to_save['batch_inputs_dict'] = batch_inputs_dict
                # input_to_save['batch_data_samples'] = batch_data_samples
                # torch.save(input_to_save, 'input_to_save.pt')

                predictions[i].psnr = psnr
                predictions[i].ssim = ssim
                predictions[i].rmse = rmse
        
        # # evalaute predict_ray_depth compare with gt_src_depth:
        # for i in range(len(est_depth_crop)):
        #     # cur ray_depth pred:
        #     cur_pred = est_depth_crop[i].squeeze(-1) # (num_src, h, w, 1)
        #     cur_gt = batch_inputs_dict['depth'][i] # (num_src, h, w)
        #     # reshape gt to low resolution
        #     cur_gt = F.interpolate(cur_gt.unsqueeze(1), size=(cur_pred.shape[1], cur_pred.shape[2]),
        #             mode='bilinear').squeeze(1) # (num_src, h, w)
            
        #     src_rmse = save_predict_ray_depth(cur_pred, cur_gt, batch_data_samples[i].metainfo, save_dir=save_dir)
        #     predictions[i].src_rmse = src_rmse
        
        for i in range(len(weight_gap)):
            # weight gap
            predictions[i].weight_gap = weight_gap[i] 
            # depth rmse
            predictions[i].src_rmse = src_rmse_list[i]
            
            
        return predictions

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`NeRFDet3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward
        """
        ray_batchs = {}
        batch_images = []
        batch_depths = []
        if 'images' in batch_data_samples[0].gt_nerf_images:
            for data_samples in batch_data_samples:
                image = data_samples.gt_nerf_images['images']
                batch_images.append(image)
        batch_images = torch.stack(batch_images)

        if 'depths' in batch_data_samples[0].gt_nerf_depths:
            for data_samples in batch_data_samples:
                depth = data_samples.gt_nerf_depths['depths']
                batch_depths.append(depth)
        batch_depths = torch.stack(batch_depths)
        if 'raydirs' in batch_inputs_dict.keys():
            ray_batchs['ray_o'] = batch_inputs_dict['lightpos']
            ray_batchs['ray_d'] = batch_inputs_dict['raydirs']
            ray_batchs['gt_rgb'] = batch_images
            ray_batchs['gt_depth'] = batch_depths
            ray_batchs['nerf_sizes'] = batch_inputs_dict['nerf_sizes']
            ray_batchs['denorm_images'] = batch_inputs_dict['denorm_images']
            ray_batchs['c2w'] = batch_inputs_dict['c2w']
            ray_batchs['intrinsic'] = batch_inputs_dict['intrinsic']
            
            x, valids, features_2d, rgb_preds = self.extract_feat(
                batch_inputs_dict,
                batch_data_samples,
                'train',
                depth=None,
                ray_batch=ray_batchs)
        else:
            x, valids, features_2d, rgb_preds = self.extract_feat(
                batch_inputs_dict, batch_data_samples, 'train')
        x += (valids, )
        results = self.bbox_head.forward(x)
        return results

    def aug_test(self, batch_inputs_dict, batch_data_samples):
        pass

    def show_results(self, *args, **kwargs):
        pass

    @staticmethod
    def _compute_projection(img_meta, stride, angles):
        projection = []
        if isinstance(img_meta['lidar2img']['intrinsic'], list):
            intrinsic = torch.tensor(np.array(img_meta['lidar2img']['intrinsic']))
            if angles is not None:
                extrinsics = []
                for angle in angles:
                    extrinsics.append(get_extrinsics(angle).to(intrinsic.device))
            else:
                extrinsics = torch.tensor(np.array(img_meta['lidar2img']['extrinsic'])) # (n_src, 4, 4)
            for i in range(len(extrinsics)):
                cur_intrinsic = intrinsic[i][:3,:3]
                ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
                # print(img_meta['ori_shape'],img_meta['img_shape'], stride, ratio)
                cur_intrinsic[:2] /= ratio
                extrinsic = extrinsics[i]
                projection.append(cur_intrinsic @ extrinsic[:3])
            
        else:
            intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
            ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
            intrinsic[:2] /= ratio
            # use predict pitch and roll for SUNRGBDTotal test
            if angles is not None:
                extrinsics = []
                for angle in angles:
                    extrinsics.append(get_extrinsics(angle).to(intrinsic.device))
            else:
                extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
            for extrinsic in extrinsics:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def compute_depth_scale(self, height, width, device, img_meta, stride, num_src):
        '''
        device:
        img_meta:
        stride: feat downsampling scale =4
        
        return: 
            depth_scale: (bs, num_view, h*w, 1). bs=1 !!!
        '''
        # # for each scene, we re-compute its src feat size using their own img_meta!!!
        # height = img_meta['img_shape'][0] // stride # feature-level src size
        # width = img_meta['img_shape'][1] // stride
        # 1. uv
        indices = [torch.arange(height, device=device), torch.arange(width, device=device)]
        # stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)
        uv = torch.stack(torch.meshgrid(*indices, indexing='ij'), dim=0) # (2, h, w)
        uv = torch.flip(uv, dims=[0]).float() # xy. (2, h, w)
        uv = uv.reshape(2, -1).transpose(1, 0).unsqueeze(0) # (1, h*w, 2)
        # 2. intrinsic
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic']) # (4, 4)
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio # corresponds to feature-level src view
        intrinsic = intrinsic.unsqueeze(0) # (1,4,4)
        # 3. 
        ray_dirs_tmp, _ = get_camera_params(uv, torch.eye(4).to(device)[None], intrinsic) # (1, h*w, 3)
        depth_scale = ray_dirs_tmp[0, :, 2:] # (h*w, 1)
        # repeat for num_src
        depth_scale = depth_scale.unsqueeze(0).unsqueeze(0).repeat(1, num_src, 1, 1) # (1, num_src, h*w, 1)
        
        return depth_scale

    def compute_depth_scale_MultiIntrin(self, height, width, device, img_meta, stride, num_src):
        ''' intrinsic is list!
        device:
        img_meta:
        stride: feat downsampling scale =4
        
        return: 
            depth_scale: (bs, num_view, h*w, 1). bs=1 !!!
        '''
        num_src = len(img_meta['lidar2img']['intrinsic'])
        # 1. uv
        indices = [torch.arange(height, device=device), torch.arange(width, device=device)]
        # stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)
        uv = torch.stack(torch.meshgrid(*indices, indexing='ij'), dim=0) # (2, h, w)
        uv = torch.flip(uv, dims=[0]).float() # xy. (2, h, w)
        uv = uv.reshape(2, -1).transpose(1, 0).unsqueeze(0) # (1, h*w, 2)
        uv = uv.repeat(num_src, 1, 1) # (num_src, h*w, 2)
        # 2. intrinsic
        intrinsic = torch.tensor(np.array(img_meta['lidar2img']['intrinsic'])) # (num_src, 4, 4)
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        # print(img_meta['ori_shape'], img_meta['img_shape'], stride, ratio)
        intrinsic[:,:2] /= ratio # corresponds to feature-level src view
        # 3. 
        fake_extrinsic = torch.eye(4, device=device).unsqueeze(0).repeat(num_src, 1, 1) # (num_src, 4, 4)
        ray_dirs_tmp, _ = get_camera_params(uv, fake_extrinsic, intrinsic) # (num_src, h*w, 3)
        depth_scale = ray_dirs_tmp[:, :, 2:] # (num_src, h*w, 1)
        # 
        depth_scale = depth_scale.unsqueeze(0) # (1, num_src, h*w, 1)
        
        return depth_scale
    
    def RayDepth_to_Depth(self, new_depth, depth_scale, height, width):
        '''
        for a scene, convert its est_depth (ray depth) to real depth
        new_depth: (num_view, height*width, num_surface, num_gs) for a scene
        depth_scale: (num_src, height*width, 1). height and width corresponds to src feat size!
        x: img feature for a batch. last two dims corresponds to (60, 80)
        height: src feat h
        width: src feat w
        return:
            real_depth: (num_view, h*w, num_surface, num_gs)
        '''
        num_src = new_depth.shape[0]
        num_surface = new_depth.shape[-2]
        gpp = new_depth.shape[-1]
        num_d = num_surface * gpp # num_surface * gpp
        depth_scale = depth_scale.repeat(1,1, num_d) # (num_src, height*width, num_depth)
        
        # 1. only get est_depth corresponds to src_feat
        # real_depth = est_depth.view(num_src, *x.shape[-2:], *est_depth.shape[-2:])  # (num_view, 60, 80, num_surface, num_gs)
        # real_depth = real_depth[:, :height, :width, :, :] # (num_src, height, width, num_surface, gpp)
        # real_depth = real_depth.view(num_src, height*width, *est_depth.shape[-2:]) # (num_src, height*width, num_surface, gpp)
        # print("input to RayDepth_to_Depth: {}".format(new_depth.shape))
        real_depth = new_depth.view(*new_depth.shape[:2], -1) * depth_scale # check!! # (num_src, height*width, num_depth)
        # print("real_depth: {}".format(real_depth.shape))
        
        real_depth = real_depth.view(num_src, height*width, num_surface, gpp)
        
        return real_depth
    
    def process_est_depth(self, est_depth, x, height, width):
        ''' crop depth to correspond to src_feat, remove pad area!
        est_depth: depth prediction from full img feature. (num_view, 60*80, num_surface, num_gs)
        x: img feature for a batch. last two dims corresponds to (60, 80)
        height: src feat h
        width: src feat w
        return:
            new_depth: (num_view, height*width, num_surface, num_gs)
        '''
        num_src = est_depth.shape[0]
        num_surface = est_depth.shape[-2]
        gpp = est_depth.shape[-1]
        
        new_depth = est_depth.view(num_src, *x.shape[-2:], num_surface, gpp)  # (num_view, 60, 80, num_surface, num_gs)
        new_depth = new_depth[:, :height, :width, :, :] # (num_src, height, width, num_surface, gpp)
        new_depth = new_depth.view(num_src, height*width, num_surface, gpp) # (num_src, height*width, num_surface, gpp)
        # print(new_depth.shape)
        
        return new_depth
        



def get_camera_params(uv, pose, intrinsics): 
    ''' process for bs=1
    uv: pixel coord. (1, num_ray, 2)
    pose: (1, 4, 4)
    intrinsic: (1, 4, 4)
    '''
    cam_loc = pose[:, :3, 3] # (b, 3)
    p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1) # (bs, num_ray)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics) # (bs, num_ray, 4). homo coord

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    # import pdb; pdb.set_trace();
    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2) # (bs, num_ray, 3)

    return ray_dirs, cam_loc

def lift(x, y, z, intrinsics):
    # parse intrinsics. project image coord to cam coord, with Z=1. suitable for bs > 1
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1] # [0,]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


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



def backproject_GTDepth(features, points, projection, depth, voxel_size):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1) # (n_src, 3, n_voxels)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points) # (n_img, 3, n_voxels)

    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long() # (n_img, n_voxels)
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) # (n_img, n_voxels)
    # below is using depth to sample feature
    
    # ------------ using gt_depth for debug --------------
    with torch.no_grad():
        if depth is not None: # (num_view, h, w)
            depth = F.interpolate(
                depth.unsqueeze(1), size=(height, width),
                mode='bilinear').squeeze(1)
            gt_valid = valid.clone() # (n_src, n_voxs)
            for i in range(n_images):
                z_mask = gt_valid.clone()
                z_mask[i, valid[i]] = \
                    (z[i, valid[i]] > depth[i, y[i, valid[i]], x[i, valid[i]]] - voxel_size[-1]) & \
                    (z[i, valid[i]] < depth[i, y[i, valid[i]], x[i, valid[i]]] + voxel_size[-1]) # noqa
                gt_valid = gt_valid & z_mask # update i-th valid # (n_src, n_voxs)
        else:
            gt_valid = None
    # ----------- end debug ---------
    
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device) # (n_src, c, n_voxels)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels,
                         n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)

    yx_map = torch.stack([y,x], dim=1) # (n_src, 2, n_vox)
    return volume, valid, yx_map, gt_valid


def backproject_Weigh(features, points, projection, depth, voxel_size, prob, 
                      gt_depth=None, save_dir=None, img_meta=None, depth_mean=None, alpha_thres=None):  
    '''the feature volume is multiply with density
    features: (n_img, f, h,w)
    projection: (n_img, 3, 4)
    depth: estimated depth. (n_img, h*w, num_surface, gs_per_pixel). estimated depth. num_surface*gpp = number of depth val for each pixel.
    prob: estimated prob. (num_src, h*w, num_surface, num_gs)
    gt_depth: (num_src, 240, 320) 
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    n_vox = n_x_voxels * n_y_voxels * n_z_voxels
    points = points.view(1, 3, -1).expand(n_images, 3, -1) # (n_img, 3, x*y*z)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points) # (n_img, 3, x*y*z)

    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long() # (n_img, n_voxels) img coord
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long() # (n_img, n_voxels)
    z = points_2d_3[:, 2] # (n_img, n_voxels)
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) # (n_img, n_voxels)
    original_valid = valid.clone()
    ##### below is using depth to sample feature ########
    depth = depth.view(n_images, height, width, -1) # (n_img, h, w, depth_vals)
    prob_norm = prob.clone().view(n_images, height, width, -1) # (n_img, h, w, depth_vals)
    prob_norm /= prob_norm.sum(dim=-1, keepdim=True) # (n_img, h, w, depth_values)
    
    if depth is not None:
        # prob_volume = torch.zeros((n_images, 1, n_vox), device=depth.device) # (n_images, 1, n_vox)
        prob_volume = []
        for i in range(n_images):
            per_img_prob = []
            for j in range(depth.shape[-1]):
                cur_depth = depth[:,:,:,j] # get depth proposal (n_img, h, w)
                # cur_mask = z.clone() > 0 # (n_img, xyz)
                cur_mask = valid.clone()
                cur_mask[i, valid[i]] = (z[i, valid[i]] > cur_depth[i, y[i, valid[i]], x[i, valid[i]]] - voxel_size[-1]) & \
                    (z[i, valid[i]] < cur_depth[i, y[i, valid[i]], x[i, valid[i]]] + voxel_size[-1])
                # record prob
                cur_prob = torch.zeros((1, n_vox), device=depth.device)
                cur_prob[:, cur_mask[i]] = prob_norm[i, y[i, cur_mask[i]], x[i, cur_mask[i]], j] # (1, n_vox)
                per_img_prob.append(cur_prob)
                
                
                if j == 0:
                    final_mask = cur_mask
                else:
                    final_mask = final_mask | cur_mask
            
            # finalize prob for each voxel
            per_img_prob = torch.cat(per_img_prob, dim=0) # (n_depth, n_vox)
            per_img_prob = torch.max(per_img_prob, dim=0, keepdim=True)[0] # (1, n_vox)
            # put in the prob_volume
            prob_volume.append(per_img_prob)        
                
            z_mask = final_mask
            valid = valid & z_mask # new valid
            
            
        prob_volume = torch.stack(prob_volume, dim=0) # (n_src, 1, n_vox)
        # print(prob_volume.shape)
    ######################################################
    
    # ------------- debug --------------------
    with torch.no_grad():
        if gt_depth is not None: # (num_view, h, w)
            gt_depth = F.interpolate(
                gt_depth.unsqueeze(1), size=(height, width),
                mode='bilinear').squeeze(1)
            gap_all = []
            # -------- check depth_map accuracy rmsed --------------
            # pred_depth = torch.sum(depth * prob_norm, dim=-1) # (n_src, h, w)
            pred_depth = depth_mean # depth expectation
            tmp_mask = gt_depth > 0
            rmse = torch.mean((pred_depth[tmp_mask] - gt_depth[tmp_mask])**2)
            # print(rmse)
            # save depth map
            if save_dir != None:
                num_src = pred_depth.shape[0]
                tmp_idx = torch.randint(0, num_src, (3,))
                scene = img_meta['img_path'][0].split('/')[-2] # scene0568_00
                save_src_depth(gt_depth, pred_depth,
                            save_dir, scene, img_meta['img_path'], tmp_idx)

            # pred_depth = depth_mean
                    
            #         eval_min_depth=0.5
            #         eval_max_depth=10
            #         valid_mask = (single_gt > eval_min_depth) & (single_gt < eval_max_depth)
                    
            #         if len(valid_gt) == 0:
            #         rmse = torch.sqrt(torch.mean((valid_pred - valid_gt)**2))
                    
            ##################################
    # ----------------- end debug -------------------
    
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device) # (n_img, f, x*y*z)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
        volume[i] *= prob_volume[i]
        
        # -------------- debug -----------
        with torch.no_grad():
            if torch.sum(valid[i]) < 1:
                continue
            if gt_depth is not None: # (num_view, h, w)
                # ------ for new_valid, check its weight accuracy. idealy, weight should be the same as gt_mask.
                gt_valid = original_valid[i].clone() # (n_vox)
                # check those original_valid fits gt depth or not
                gt_valid[original_valid[i]] = (z[i, original_valid[i]] > gt_depth[i, y[i, original_valid[i]], x[i, original_valid[i]]] - voxel_size[-1]) & \
                        (z[i, original_valid[i]] < gt_depth[i, y[i, original_valid[i]], x[i, original_valid[i]]] + voxel_size[-1]) 
                
                gap_i = torch.mean((gt_valid[original_valid[i]].float() - prob_volume[i,0, original_valid[i]])**2)
                gap_all.append(gap_i) # debug to check every value!
                
                # ------------- check binary valid accuracy ----------.
                orig_gap = torch.mean((gt_valid.float() - original_valid[i].float())**2)
                new_gap = torch.mean((gt_valid.float() - valid[i].float())**2)
                n_reduce_vox = torch.sum(original_valid[i]) - torch.sum(valid[i])
                print("orig_gap - new_gap: {:.5f}, reduce {} voxels, weight_gap vs gt: {:.5f}".format(orig_gap - new_gap, n_reduce_vox, gap_i))
        # ----------------- end debug -----------
    
    if gt_depth is not None:
        gap_all = sum(gap_all) / len(gap_all)
        # print(gap_all)
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)

    if gt_depth == None:
        gap_all = torch.tensor(1.)
        rmse = torch.tensor(1.)
    return volume, valid, gap_all, rmse


def backproject_Weigh_differentiable_propagation(features, points, projection, depth, voxel_size, prob, 
                      gt_depth=None, save_dir=None, img_meta=None, depth_mean=None, alpha_thres=None, sigma_w=0.5):  
    '''
    Modified version with differentiable depth-feature propagation
    features: (n_img, f, h,w)
    projection: (n_img, 3, 4)
    depth: estimated depth. (n_img, h*w, num_surface, gs_per_pixel)
    prob: estimated prob. (num_src, h*w, num_surface, num_gs)
    gt_depth: (num_src, 240, 320) 
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    n_vox = n_x_voxels * n_y_voxels * n_z_voxels
    
    # Project 3D points to 2D coordinates
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)

    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]

    # Initial valid mask (basic frustum check)
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)
    # original_valid = valid.clone()

    pixel_indices = y * width + x


    # ================== Differentiable Propagation Core ==================
    # Reshape depth and probability
    depth = depth.view(n_images, height, width, -1)  # (n_img, h, w, n_depth)
    n_depth = depth.shape[-1]
    prob_norm = prob.clone().view(n_images, height, width, -1)
    prob_norm = prob_norm / (prob_norm.sum(dim=-1, keepdim=True) + 1e-8)

    # Compute geometric weights using Gaussian kernel
    sigma = voxel_size[-1] * sigma_w  # Adaptive bandwidth
    
    # Flatten spatial dimensions
    depth_flat = depth.view(n_images, height, width, -1)  # (n_img, h, w, n_depth)
    depth_flat = depth_flat.view(n_images, -1, depth_flat.shape[-1])           # (n_img, h*w, n_depth)

    pixel_indices_safe = torch.where(valid, pixel_indices, 0)
    depth_mapped = torch.gather(depth_flat, dim=1, index=pixel_indices_safe.unsqueeze(-1).expand(-1, -1, n_depth))
    depth_mapped = depth_mapped * valid.unsqueeze(-1)

    prob_flat = prob_norm.view(n_images, -1, prob_norm.shape[-1])  # (n_img, h*w, n_depth)
    prob_flat = torch.gather(prob_flat, dim=1, index=pixel_indices_safe.unsqueeze(-1).expand(-1, -1, n_depth))
    prob_flat = prob_flat * valid.unsqueeze(-1)
    # Compute depth difference matrix
    z_expanded = z.unsqueeze(-1)  # (n_img, n_voxels, 1)
    depth_diff = torch.abs(z_expanded - depth_mapped)  # (n_img, n_voxels, n_depth)

    
    # Compute continuous weights
    weights = torch.exp(-depth_diff / (2 * sigma**2))  # (n_img, n_voxels, n_depth)
    
    # Fuse probability and geometric weights
    alpha = (weights * prob_flat).sum(dim=-1)  # (n_img, n_voxels)
    
    # Dynamic valid mask (preserve gradients)
    valid = valid & (alpha > alpha_thres)

    # ================== Differentiable Feature Aggregation ==================
    volume = torch.zeros((n_images, n_channels, n_vox), device=features.device)
    
    for i in range(n_images):
        # Get valid indices for current view
        valid_mask = valid[i]
        y_valid = y[i, valid_mask]
        x_valid = x[i, valid_mask]
        
        if y_valid.numel() == 0:
            continue
            
        # Get corresponding features (C, K)
        feats = features[i, :, y_valid, x_valid]  # (C, K)
        
        # Get weights for valid positions (K, n_depth)
        valid_weights = weights[i, valid_mask]  # (K, n_depth)
        
        # Weighted feature aggregation (C, K, n_depth) -> (C, K)
        weighted_feats = torch.einsum('ck,kd->ckd', feats, valid_weights)  # (C, K, n_depth)
        weighted_feats = weighted_feats.mean(dim=-1)  # (C, K)
        
        # Assign to volume
        volume[i, :, valid_mask] = weighted_feats

    # ================== Post-processing ==================
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)

    # ------------- debug --------------------
    with torch.no_grad():
        if gt_depth is not None: # (num_view, h, w)
            gt_depth = F.interpolate(
                gt_depth.unsqueeze(1), size=(height, width),
                mode='bilinear').squeeze(1)
            gap_all = []
            # -------- check depth_map accuracy rmse --------------
            # pred_depth = torch.sum(depth * prob_norm, dim=-1) # (n_src, h, w)
            pred_depth = depth_mean # depth expectation
            tmp_mask = gt_depth > 0
            rmse = torch.mean((pred_depth[tmp_mask] - gt_depth[tmp_mask])**2)
            # print(rmse)
            # save depth map
            if save_dir != None:
                num_src = pred_depth.shape[0]
                tmp_idx = torch.randint(0, num_src, (3,))
                scene = img_meta['img_path'][0].split('/')[-2] # scene0568_00
                save_src_depth(gt_depth, pred_depth,
                            save_dir, scene, img_meta['img_path'], tmp_idx)
    
    # ----------------- end debug -------------------

    if gt_depth == None:
        gap_all = torch.tensor(1.)
        rmse = torch.tensor(1.)
        
    return volume, valid.float(), gap_all, rmse

# for SUNRGBDTotal test
def get_extrinsics(angles):
    yaw = angles.new_zeros(())
    pitch, roll = angles
    r = angles.new_zeros((3, 3))
    r[0, 0] = torch.cos(yaw) * torch.cos(pitch)
    r[0, 1] = torch.sin(yaw) * torch.sin(roll) - torch.cos(yaw) * torch.cos(
        roll) * torch.sin(pitch)
    r[0, 2] = torch.cos(roll) * torch.sin(yaw) + torch.cos(yaw) * torch.sin(
        pitch) * torch.sin(roll)
    r[1, 0] = torch.sin(pitch)
    r[1, 1] = torch.cos(pitch) * torch.cos(roll)
    r[1, 2] = -torch.cos(pitch) * torch.sin(roll)
    r[2, 0] = -torch.cos(pitch) * torch.sin(yaw)
    r[2, 1] = torch.cos(yaw) * torch.sin(roll) + torch.cos(roll) * torch.sin(
        yaw) * torch.sin(pitch)
    r[2, 2] = torch.cos(yaw) * torch.cos(roll) - torch.sin(yaw) * torch.sin(
        pitch) * torch.sin(roll)

    # follow Total3DUnderstanding
    t = angles.new_tensor([[0., 0., 1.], [0., -1., 0.], [-1., 0., 0.]])
    r = t @ r.T
    # follow DepthInstance3DBoxes
    r = r[:, [2, 0, 1]]
    r[2] *= -1
    extrinsic = angles.new_zeros((4, 4))
    extrinsic[:3, :3] = r
    extrinsic[3, 3] = 1.
    return extrinsic


