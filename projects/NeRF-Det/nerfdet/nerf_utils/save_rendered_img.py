# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from PIL import Image
import matplotlib.pyplot as plt
from einops import einsum, rearrange, repeat
from gs_src.geometry.projection import homogenize_points, project
from gs_src.model.ply_export import export_ply, export_ply_WorldSpace
import torch.nn.functional as F
import pickle
import skimage.io

def compute_psnr_from_mse(mse):
    return -10.0 * torch.log(mse) / np.log(10.0)


def compute_psnr(pred, target, mask=None):
    """Compute psnr value (we assume the maximum pixel value is 1)."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    mse = ((pred - target)**2).mean()
    return compute_psnr_from_mse(mse).cpu().numpy()


def compute_ssim(pred, target, mask=None):
    """Computes Masked SSIM following the neuralbody paper."""
    assert pred.shape == target.shape and pred.shape[-1] == 3
    if mask is not None:
        x, y, w, h = cv2.boundingRect(mask.cpu().numpy().astype(np.uint8))
        pred = pred[y:y + h, x:x + w]
        target = target[y:y + h, x:x + w]
    # try:
    #     ssim = structural_similarity(
    #         pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1)
    # except ValueError:
    #     ssim = structural_similarity(
    #         pred.cpu().numpy(), target.cpu().numpy(), multichannel=True)
    ssim = structural_similarity(
            pred.cpu().numpy(), target.cpu().numpy(), channel_axis=-1, data_range=1) # image value 0-1
    return ssim


# def save_rendered_img(img_meta, rendered_results):
#     filename = img_meta[0]['filename']
#     scenes = filename.split('/')[-2]

#     for ret in rendered_results:
#         depth = ret['outputs_coarse']['depth']
#         rgb = ret['outputs_coarse']['rgb']
#         gt = ret['gt_rgb']
#         gt_depth = ret['gt_depth']

#     # save images
#     psnr_total = 0
#     ssim_total = 0
#     rsme = 0
#     for v in range(gt.shape[0]):
#         rsme += ((depth[v] - gt_depth[v])**2).cpu().numpy()
#         depth_ = ((depth[v] - depth[v].min()) /
#                   (depth[v].max() - depth[v].min() + 1e-8)).repeat(1, 1, 3)
#         img_to_save = torch.cat([rgb[v], gt[v], depth_], dim=1)
#         image_path = os.path.join('nerf_vs_rebuttal', scenes)
#         if not os.path.exists(image_path):
#             os.makedirs(image_path)
#         save_dir = os.path.join(image_path, 'view_' + str(v) + '.png')

#         font = cv2.FONT_HERSHEY_SIMPLEX
#         org = (50, 50)
#         fontScale = 1
#         color = (255, 0, 0)
#         thickness = 2
#         image = np.uint8(img_to_save.cpu().numpy() * 255.0)
#         psnr = compute_psnr(rgb[v], gt[v], mask=None)
#         psnr_total += psnr
#         ssim = compute_ssim(rgb[v], gt[v], mask=None)
#         ssim_total += ssim
#         image = cv2.putText(
#             image, 'PSNR: ' + '%.2f' % compute_psnr(rgb[v], gt[v], mask=None),
#             org, font, fontScale, color, thickness, cv2.LINE_AA)

#         cv2.imwrite(save_dir, image)

#     return psnr_total / gt.shape[0], ssim_total / gt.shape[0], rsme / gt.shape[
#         0]


def save_rendered_imgNoDepth(img_meta, rendered_results):
    '''
    return avg ssim and psnr for current scene. average over number of target views.
    '''
    filename = img_meta[0]["filename"]
    scenes = filename.split('/')[-2]
    metrics = dict()

    for ret in rendered_results:
        # depth = ret['outputs_coarse']['depth']
        rgb = ret['outputs_coarse']['rgb']
        gt  = ret['gt_rgb']
        # gt_depth = ret['gt_depth']

    # psnr = compute_psnr(rgb, gt, mask=None)
    # # save images
    psnr_total = 0
    ssim_total = 0
    rsme = 0
    for v in range(gt.shape[0]):
        # rsme += ((depth[v] - gt_depth[v])**2).cpu().numpy()
        # depth_ = ((depth[v]-depth[v].min()) / (depth[v].max() - depth[v].min()+1e-8)).repeat(1, 1, 3)
        img_to_save = torch.cat([rgb[v], gt[v]], dim=1)
        image_path = os.path.join('nerf_vs_rebuttal', scenes)
        print(image_path)
        os.makedirs(image_path, exist_ok=True)
        save_dir = os.path.join(image_path, 'view_'+str(v)+'.png')

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        image = np.uint8(img_to_save.cpu().numpy() * 255.0)
        psnr = compute_psnr(rgb[v], gt[v], mask=None)
        psnr_total += psnr
        ssim = compute_ssim(rgb[v], gt[v], mask=None)
        ssim_total += ssim
        image = cv2.putText(image, 'PSNR: ' + "%.2f" % compute_psnr(rgb[v], gt[v], mask=None),
            org, font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.imwrite(save_dir, image)

    return psnr_total/gt.shape[0], ssim_total/gt.shape[0], 0


def save_rendered_img(pred, gt, img_meta, save_dir=None):
    '''
    pred: (n_tgt, 3, 239, 320) (0-1). check!
    gt: (n_tgt, 3, 239, 320). (0-1)
    '''
    psnr_total = 0
    ssim_total = 0
    rsme = 0
    scene = img_meta['img_path'][0].split('/')[-2] # scene0568_00
    num_tgt = pred.shape[0]
    pred = pred.permute(0,2,3,1) # (num_gtg, h, w, 3)
    gt = gt.permute(0,2,3,1)
    for v in range(num_tgt):
        # rsme += ((depth[v] - gt_depth[v])**2).cpu().numpy()
        # depth_ = ((depth[v]-depth[v].min()) / (depth[v].max() - depth[v].min()+1e-8)).repeat(1, 1, 3)
        if save_dir != None:
            img_to_save = torch.cat([pred[v], gt[v]], dim=1) # (h, 2w, 3)
            image_path = os.path.join(save_dir, scene)
            os.makedirs(image_path, exist_ok=True)
            save_dir = os.path.join(image_path, 'view_'+str(v)+'.png')
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1
            image = np.uint8(img_to_save.cpu().numpy() * 255.0) # image here is bgr!!!
        
        psnr = compute_psnr(pred[v], gt[v], mask=None) # float32, cuda. (239, 320, 3)
        psnr_total += psnr
        
        ssim = compute_ssim(pred[v], gt[v], mask=None) # check dimension order
        ssim_total += ssim
        if save_dir != None:
            image = cv2.putText(image, 'PSNR: ' + "%.2f" % psnr + "SSIM: " + "%.2f" % ssim,
                org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite(save_dir, image)

    return psnr_total/gt.shape[0], ssim_total/gt.shape[0], 0



def save_rendered_img_Image(pred, gt, img_meta, save_dir=None):
    '''
    pred: (n_tgt, 3, 239, 320) (0-1). check!. rgb
    gt: (n_tgt, 3, 239, 320). (0-1)
    '''
    psnr_total = 0
    ssim_total = 0
    rsme = 0
    scene = img_meta['img_path'][0].split('/')[-2] # scene0568_00
    num_tgt = pred.shape[0]
    pred = pred.permute(0,2,3,1) # (num_gtg, h, w, 3)
    gt = gt.permute(0,2,3,1)
    for v in range(num_tgt):
        # rsme += ((depth[v] - gt_depth[v])**2).cpu().numpy()
        # depth_ = ((depth[v]-depth[v].min()) / (depth[v].max() - depth[v].min()+1e-8)).repeat(1, 1, 3)
        if save_dir != None:
            img_to_save = torch.cat([pred[v], gt[v]], dim=1) # (h, 2w, 3)
            image_path = os.path.join(save_dir, scene)
            os.makedirs(image_path, exist_ok=True)
            save_dir = os.path.join(image_path, 'view_'+str(v)+'.png')
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # org = (50, 50)
            # fontScale = 0.5
            # color = (255, 0, 0)
            # thickness = 1
            image = np.uint8(img_to_save.cpu().numpy() * 255.0) # image here is bgr!!!
            Image.fromarray(image).save(save_dir)
        
        psnr = compute_psnr(pred[v], gt[v], mask=None) # float32, cuda. (239, 320, 3)
        psnr_total += psnr
        
        ssim = compute_ssim(pred[v], gt[v], mask=None) # check dimension order
        ssim_total += ssim

    return psnr_total/gt.shape[0], ssim_total/gt.shape[0], 0


def save_rendered_depth(pred, gt, img_meta, save_dir=None):
    '''
    pred: (n_tgt, 239, 320). 
    gt: (n_tgt, 239, 320).
    '''
    # print(pred.shape, gt.shape)
    print(pred.max(), pred.min(), gt.max(), gt.min())
    rmse = 0
    scene = img_meta['img_path'][0].split('/')[-2] # scene0568_00
    num_tgt = pred.shape[0]
    assert pred.shape == gt.shape
    for v in range(num_tgt):
        cur_rmse = ((pred[v] - gt[v])**2).cpu().numpy()
        cur_rmse = np.mean(cur_rmse)
        rmse += cur_rmse
        # depth_ = ((depth[v]-depth[v].min()) / (depth[v].max() - depth[v].min()+1e-8)).repeat(1, 1, 3)
        if save_dir != None:
            depth_to_save = torch.cat([pred[v], gt[v]], dim=1) # (h, 2w)
            plt.imsave(os.path.join(save_dir, scene, 'depth_{}_rmse={}.png'.format(v, cur_rmse)), depth_to_save.cpu(), cmap='viridis')

    return rmse/num_tgt

def save_predict_ray_depth(pred, gt, img_meta, save_dir=None):
    ''' evaluate src depth prediction
    pred: (n_tgt, h, w). 
    gt: (n_tgt, h, w).
    '''
    assert pred.shape == gt.shape
    
    rmse = 0
    scene = img_meta['img_path'][0].split('/')[-2] # scene0568_00
    num_tgt = pred.shape[0]
    for v in range(num_tgt):
        cur_rmse = ((pred[v] - gt[v])**2).cpu().numpy()
        cur_rmse = np.mean(cur_rmse)
        rmse += cur_rmse
        # depth_ = ((depth[v]-depth[v].min()) / (depth[v].max() - depth[v].min()+1e-8)).repeat(1, 1, 3)
        if save_dir != None and v % 10 ==0 :
            depth_to_save = torch.cat([pred[v], gt[v]], dim=1) # (h, 2w)
            os.makedirs(os.path.join(save_dir, scene, 'src'), exist_ok=True)
            plt.imsave(os.path.join(save_dir, scene, 'src', 'depth_{}_rmse={}.png'.format(v, cur_rmse)), depth_to_save.cpu(), cmap='viridis')

    return rmse/num_tgt


# save pcd
def save_pcd(visualization_dump, img_meta, save_dir=None, device=None):
    '''
    transform gaussians to camera views.
    '''
    gaussians = visualization_dump['gaussians']
    scene = img_meta['img_path'][0].split('/')[-2] # scene0568_00
    scene_dir = os.path.join(save_dir, scene)
    
    h = visualization_dump['src_h']
    w = visualization_dump['src_w']
    # num_src = visualization_dump['num_src']
    src_id = visualization_dump['src_id']
    num_src = len(src_id)

    # Transform means into camera space.
    means = rearrange(
        gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=num_src, h=h, w=w
    ) # (59, 80, 3, 100, 3)
    means = homogenize_points(means)
    w2c = list(map(torch.tensor, img_meta['lidar2img']['extrinsic'])) # extrinsic: list, len=40.
    w2c = torch.stack(w2c, dim=0).to(device)[src_id] # (n_src, 4, 4)
    means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3] # (59, 80, 3, 100, 3)

    # Create a mask to filter the Gaussians. First, throw away Gaussians at the
    # borders, since they're generally of lower quality.
    mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
    GAUSSIAN_TRIM = 8
    far = 3
    mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1 # (h, w, spp,num_src)

    # Then, drop Gaussians that are really far away.
    mask = mask & (means[..., 2] < far)
    
    # # mask out low opacity
    # opacity = rearrange(
    #     gaussians.opacities, "() (v h w spp) -> h w spp v", v=num_src, h=h, w=w
    # ) # (59, 80, 3, n_src)
    # mask = mask & (opacity > 0.1)

    def trim(element):
        print(element.shape)
        element = rearrange(
            element, "() (v h w spp) ... -> h w spp v ...", v=num_src, h=h, w=w
        )
        # print(element.shape, mask.shape)
        return element[mask][None]
    
    
    export_ply(
            torch.tensor(img_meta['lidar2img']["extrinsic"][0], device=device).inverse(), # c2w of first view (4,4)
            trim(gaussians.means)[0],
            trim(visualization_dump["scales"])[0],
            trim(visualization_dump["rotations"])[0],
            trim(gaussians.harmonics)[0],
            trim(gaussians.opacities)[0],
            os.path.join(scene_dir, "gaussians.ply"),
    )
    
def save_pcd_WorldSpace(visualization_dump, img_meta, save_dir=None, device=None):
    '''
    transform gaussians to camera views.
    '''
    gaussians = visualization_dump['gaussians']
    scene = img_meta['img_path'][0].split('/')[-2] # scene0568_00
    scene_dir = os.path.join(save_dir, scene)
    
    '''h = visualization_dump['src_h']
    w = visualization_dump['src_w']
    # num_src = visualization_dump['num_src']
    src_id = visualization_dump['src_id']
    num_src = len(src_id)

    # Transform means into camera space.
    means = rearrange(
        gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=num_src, h=h, w=w
    ) # (59, 80, 3, 100, 3)
    means = homogenize_points(means)
    w2c = list(map(torch.tensor, img_meta['lidar2img']['extrinsic'])) # extrinsic: list, len=40.
    w2c = torch.stack(w2c, dim=0).to(device)[src_id] # (n_src, 4, 4)
    means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3] # (59, 80, 3, 100, 3)

    # Create a mask to filter the Gaussians. First, throw away Gaussians at the
    # borders, since they're generally of lower quality.
    mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
    GAUSSIAN_TRIM = 8
    far = 8
    mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1 # (h, w, spp,num_src)

    # Then, drop Gaussians that are really far away.
    mask = mask & (means[..., 2] < far)
    
    # # mask out low opacity
    # opacity = rearrange(
    #     gaussians.opacities, "() (v h w spp) -> h w spp v", v=num_src, h=h, w=w
    # ) # (59, 80, 3, n_src)
    # mask = mask & (opacity > 0.1)

    def trim(element):
        print(element.shape)
        element = rearrange(
            element, "() (v h w spp) ... -> h w spp v ...", v=num_src, h=h, w=w
        )
        # print(element.shape, mask.shape)
        return element[mask][None]'''
    
    
    export_ply_WorldSpace(
            torch.tensor(img_meta['lidar2img']["extrinsic"][0], device=device).inverse(), # c2w of first view (4,4)
            gaussians.means[0],
            visualization_dump["scales"][0],
            visualization_dump["rotations"][0],
            gaussians.harmonics[0],
            gaussians.opacities[0],
            os.path.join(scene_dir, "gaussians.ply"),
    )


def save_src_SingleDepth(gt_depth, pred_mu, pred_sigma, save_dir, scene):
    '''
    gt_depth: (orig_h, orig_w)
    pred_mu: (h, w)
    pred_sigma: (h, w)
    '''
    os.makedirs(os.path.join(save_dir, scene), exist_ok=True)
    
    if gt_depth.shape != pred_mu.shape:
        gt_depth = F.interpolate(
            gt_depth.unsqueeze(0).unsqueeze(0), size=pred_mu.shape,
            mode='bilinear').squeeze(0).squeeze(0)
    # depth
    rmse = torch.mean((pred_mu - gt_depth)**2)
    depth_to_save = torch.cat([pred_mu, gt_depth], dim=1) # (h, 2w)
    plt.imsave(os.path.join(save_dir, scene, 'depth_rmse={}.png'.format(rmse)), depth_to_save.cpu(), cmap='viridis')
    # sigma
    min_sigma = pred_sigma.min()
    max_sigma = pred_sigma.max()
    avg_sigma = torch.mean(pred_sigma)
    plt.imsave(os.path.join(save_dir, scene, 'sigma_min={}_max={}_avg={}.png'.format(min_sigma, max_sigma, avg_sigma)), pred_sigma.cpu(), cmap='viridis')
    
def save_src_GaussianDepth(gt_depth, pred_mu, pred_sigma, save_dir, scene, src_img_path, tmp_idx):
    '''
    gt_depth: (num_src, orig_h, orig_w)
    pred_mu: (num_src, h, w)
    pred_sigma: (num_src, h, w)
    tmp_idx: chosen ids
    '''
    os.makedirs(os.path.join(save_dir, scene), exist_ok=True)
    
    if gt_depth.shape != pred_mu.shape:
        gt_depth = F.interpolate(
            gt_depth.unsqueeze(1), size=(pred_mu.shape[1], pred_mu.shape[2]),
            mode='bilinear').squeeze(1)
    for id in tmp_idx:
        # depth
        rmse = torch.mean((pred_mu[id] - gt_depth[id])**2)
        depth_to_save = torch.cat([pred_mu[id], gt_depth[id]], dim=1) # (h, 2w)
        plt.imsave(os.path.join(save_dir, scene, '{}_depth_rmse={}.png'.format(id,rmse)), depth_to_save.cpu(), cmap='viridis')
        # sigma
        min_sigma = pred_sigma[id].min()
        max_sigma = pred_sigma[id].max()
        avg_sigma = torch.mean(pred_sigma[id])
        plt.imsave(os.path.join(save_dir, scene, '{}_sigma_min={}_max={}_avg={}.png'.format(id, min_sigma, max_sigma, avg_sigma)), pred_sigma[id].cpu(), cmap='viridis')
        # src img
        img_path = src_img_path[id]
        img = cv2.imread(img_path)
        img_ = cv2.resize(img, (pred_mu.shape[2], pred_mu.shape[1]))
        cv2.imwrite(os.path.join(save_dir, scene, '{}_rgb.png'.format(id)), img_)
        # save into pickel for gaussian visualization
        save_dict = {}
        save_dict['mu'] = pred_mu[id].cpu().numpy()
        save_dict['sigma'] = pred_sigma[id].cpu().numpy()
        save_dict['gt_depth'] = gt_depth[id].cpu().numpy()
        save_dict['rgb'] = img_ # use opencv to read it!
        with open(os.path.join(save_dir, scene, '{}_data.pickle'.format(id)), 'wb') as f:
            pickle.dump(save_dict, f)



def save_src_depth(gt_depth, pred_depth, save_dir, scene, src_img_path, tmp_idx):
    '''
    gt_depth: (num_src, orig_h, orig_w)
    pred_mu: (num_src, h, w)
    tmp_idx: chosen ids 
    '''
    os.makedirs(os.path.join(save_dir, scene), exist_ok=True)
    
    if gt_depth.shape != pred_depth.shape:
        gt_depth = F.interpolate(
            gt_depth.unsqueeze(1), size=(pred_depth.shape[1], pred_depth.shape[2]),
            mode='bilinear').squeeze(1)
    for id in tmp_idx:
        # depth
        rmse = torch.mean((pred_depth[id] - gt_depth[id])**2)
        depth_to_save = torch.cat([pred_depth[id], gt_depth[id]], dim=1) # (h, 2w)
        plt.imsave(os.path.join(save_dir, scene, '{}_depth_rmse={}.png'.format(id,rmse)), depth_to_save.cpu(), cmap='viridis')
        # src img
        img_path = src_img_path[id]
        img = cv2.imread(img_path)
        img_ = cv2.resize(img, (pred_depth.shape[2], pred_depth.shape[1]))
        cv2.imwrite(os.path.join(save_dir, scene, '{}_rgb.png'.format(id)), img_)
        # # save into pickel for gaussian visualization
        # save_dict = {}
        # save_dict['mu'] = pred_depth[id].cpu().numpy()
        # save_dict['gt_depth'] = gt_depth[id].cpu().numpy()
        # save_dict['rgb'] = img_ # use opencv to read it!
        # with open(os.path.join(save_dir, scene, '{}_data.pickle'.format(id)), 'wb') as f:
        #     pickle.dump(save_dict, f)

            
scannet_nyu_id = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
scannet_class_names = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'garbagebin'
] # its order correspond to idx


class MultiViewMixin:
    colors = np.multiply([
        plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
    ], 255).astype(np.uint8).tolist()

    @staticmethod
    def draw_corners(img, corners, color, projection, label):
        corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1)
        corners_2d_3 = corners_3d_4 @ projection.T
        z_mask = corners_2d_3[:, 2] > 0
        corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:]
        corners_2d = corners_2d.astype(np.int_)
        for i, j in [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]:
            if z_mask[i] and z_mask[j]:
                img = cv2.line(
                    img=img,
                    pt1=tuple(corners_2d[i]),
                    pt2=tuple(corners_2d[j]),
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
        # add pred label:
        label_name = scannet_class_names[label]
        cv2.putText(img, label_name, tuple(corners_2d[0]), cv2.FONT_HERSHEY_COMPLEX, 1.0, color)

    def show(self, predictions, batch_data_samples, out_dir):
        for i in range(len(predictions)):
            # for i_th scene
            img_meta = batch_data_samples[i].metainfo
            result = predictions[i].pred_instances_3d
            
            for j in range(len(img_meta['img_path'])): # all images in the scene
                assert len(img_meta['img_path']) == len(img_meta['lidar2img']['extrinsic'])
                img = skimage.io.imread(img_meta['img_path'][j]) # original size image
                scene = img_meta['img_path'][0].split('/')[-2]
                extrinsic = img_meta['lidar2img']['extrinsic'][j] # (4, 4) w2c numpy
                intrinsic = img_meta['lidar2img']['intrinsic'][:3, :3] # (3,3) correspond to original size image
                
                
                
                projection = intrinsic @ extrinsic[:3] # world2img
                
                if not len(result['scores_3d']):
                    continue
                corners = result['bboxes_3d'].corners.cpu().numpy() # (n_box, 8, 3)
                scores = result['scores_3d'].cpu().numpy() # (n_box,)
                labels = result['labels_3d'].cpu().numpy() # (n_box, )
                for corner, score, label in zip(corners, scores, labels):
                    self.draw_corners(img, corner, self.colors[label], projection, label)
                # out_file_name = os.path.split(['img_info'][j]['filename'])[-1][:-4]
                os.makedirs(os.path.join(out_dir,'det', scene), exist_ok=True)
                skimage.io.imsave(os.path.join(out_dir,'det', scene, '{}.png'.format(j)), img)
            