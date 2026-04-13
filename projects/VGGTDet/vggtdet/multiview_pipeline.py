# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform, Compose
from PIL import Image

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import BasePoints, get_points_type

def get_dtu_raydir(pixelcoords, intrinsic, rot, dir_norm=None):
    # rot is c2w
    # pixelcoords: H x W x 2
    x = (pixelcoords[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (pixelcoords[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    # dirs = np.sum(dirs[...,None,:] * rot[:,:], axis=-1) # h*w*1*3   x   3*3
    dirs = dirs @ rot[:, :].T  #
    if dir_norm:
        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)

    return dirs


def read_pose_matrix(file_path):
    """
    Reads a 4x4 pose matrix from a text file.

    Args:
        file_path (str): The path to the text file containing the pose matrix.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the pose matrix.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        matrix = [list(map(float, line.strip().split())) for line in lines]
        
        pose_matrix = np.array(matrix)
        
        if pose_matrix.shape != (4, 4):
            raise ValueError("The input file does not contain a valid 4x4 pose matrix.")
        
        return pose_matrix
    
    except Exception as e:
        print(f"Error reading pose matrix: {e}")
        return None

@TRANSFORMS.register_module()
class ProjectPCtoFirstFrameAndNorm(BaseTransform):
    """
    Project unaligned point clouds to the axis of the first-frame-axis
    """

    def __init__(self, coord_type):
        self.coord_type = coord_type

    def transform(self, results: dict) -> dict:

        the_first_img_path = results['img_path'][0]
        # extrinsic_matrix = results['lidar2cam']
        pose_file_path = the_first_img_path[:-3]+'txt'
        pose_matrix = read_pose_matrix(pose_file_path) # the matrix of lidar2cam
        extrinsic_matrix = np.linalg.inv(pose_matrix)
        points = results['points']

        points_pos = points[:, :3]
        points_pos_homo = np.hstack((points_pos, np.ones((points_pos.shape[0], 1))))

        points_in_first_axis_homo = np.dot(extrinsic_matrix, points_pos_homo.T).T  # (N x 4)
        points_in_first_axis = points_in_first_axis_homo[:, :3] 

        avg_distance = np.mean(np.linalg.norm(points_in_first_axis, axis=1))

        points_in_first_axis_normalized = points_in_first_axis #/ avg_distance
        pc_cam_with_rgb = np.hstack((points_in_first_axis_normalized, points[:, 3:]))

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(pc_cam_with_rgb, points_dim=points.points_dim, attribute_dims=points.attribute_dims)
        results['pose_matrix'] = pose_matrix
        results['avg_distance'] = avg_distance

        return results

@TRANSFORMS.register_module()
class ProjectPCtoFirstFrameAndNormArkit(BaseTransform):
    """
    Project unaligned point clouds to the axis of the first-frame-axis
    """

    def __init__(self, coord_type):
        self.coord_type = coord_type

    def transform(self, results: dict) -> dict:

        the_first_img_path = results['img_path'][0]
        # extrinsic_matrix = results['lidar2cam']
        pose_file_path = the_first_img_path.replace('_color.png', '_pose.npy')
        pose_matrix = np.load(pose_file_path) #read_pose_matrix(pose_file_path) # the matrix of lidar2cam
        extrinsic_matrix = np.linalg.inv(pose_matrix)
        points = results['points']

        points_pos = points[:, :3]
        points_pos_homo = np.hstack((points_pos, np.ones((points_pos.shape[0], 1))))

        points_in_first_axis_homo = np.dot(extrinsic_matrix, points_pos_homo.T).T  # (N x 4)
        points_in_first_axis = points_in_first_axis_homo[:, :3] 

        avg_distance = np.mean(np.linalg.norm(points_in_first_axis, axis=1))

        points_in_first_axis_normalized = points_in_first_axis #/ avg_distance
        pc_cam_with_rgb = np.hstack((points_in_first_axis_normalized, points[:, 3:]))

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(pc_cam_with_rgb, points_dim=points.points_dim, attribute_dims=points.attribute_dims)
        results['pose_matrix'] = pose_matrix
        results['avg_distance'] = avg_distance

        return results

# @TRANSFORMS.register_module()
# class ProjectPCtoFirstFrameAndNormKeepGTPoint(BaseTransform):
#     """
#     Project unaligned point clouds to the axis of the first-frame-axis
#     """

#     def __init__(self, coord_type):
#         self.coord_type = coord_type
#         self.rotation_axis = 2

#     def transform(self, results: dict) -> dict:

#         the_first_img_path = results['img_path'][0]
#         # extrinsic_matrix = results['lidar2cam']
#         pose_file_path = the_first_img_path[:-3]+'txt'
#         pose_matrix = read_pose_matrix(pose_file_path) # the matrix of lidar2cam
#         extrinsic_matrix = np.linalg.inv(pose_matrix)
#         points = results['points']
#         results['original_gt_points'] = np.copy(points)[:, :3]

#         points_pos = points[:, :3]
#         points_pos_homo = np.hstack((points_pos, np.ones((points_pos.shape[0], 1))))

#         points_in_first_axis_homo = np.dot(extrinsic_matrix, points_pos_homo.T).T  # (N x 4)
#         points_in_first_axis = points_in_first_axis_homo[:, :3] 

#         avg_distance = np.mean(np.linalg.norm(points_in_first_axis, axis=1))

#         points_in_first_axis_normalized = points_in_first_axis #/ avg_distance
#         pc_cam_with_rgb = np.hstack((results['original_gt_points'], points[:, 3:]))
#         results['points'] = pc_cam_with_rgb
#         self.align_point(results)
#         points_class = get_points_type(self.coord_type)
#         results['points'] = points_class(results['points'], points_dim=points.points_dim, attribute_dims=points.attribute_dims)
#         results['pose_matrix'] = pose_matrix
#         results['avg_distance'] = avg_distance

#         return results


#     def _trans_points(self, results: dict, trans_factor: np.ndarray) -> None:
#         """Private function to translate points.

#         Args:
#             input_dict (dict): Result dict from loading pipeline.
#             trans_factor (np.ndarray): Translation vector to be applied.

#         Returns:
#             dict: Results after translation, 'points' is updated in the dict.
#         """
#         results['points'].translate(trans_factor)

#     def _rot_points(self, results: dict, rot_mat: np.ndarray) -> None:
#         """Private function to rotate bounding boxes and points.

#         Args:
#             input_dict (dict): Result dict from loading pipeline.
#             rot_mat (np.ndarray): Rotation matrix to be applied.

#         Returns:
#             dict: Results after rotation, 'points' is updated in the dict.
#         """
#         # input should be rot_mat_T so I transpose it here
#         results['points'].rotate(rot_mat.T)

#     def _check_rot_mat(self, rot_mat: np.ndarray) -> None:
#         """Check if rotation matrix is valid for self.rotation_axis.

#         Args:
#             rot_mat (np.ndarray): Rotation matrix to be applied.
#         """
#         is_valid = np.allclose(np.linalg.det(rot_mat), 1.0)
#         valid_array = np.zeros(3)
#         valid_array[self.rotation_axis] = 1.0
#         is_valid &= (rot_mat[self.rotation_axis, :] == valid_array).all()
#         is_valid &= (rot_mat[:, self.rotation_axis] == valid_array).all()
#         assert is_valid, f'invalid rotation matrix {rot_mat}'

#     def align_point(self, results: dict) -> dict:
#         """Call function to shuffle points.

#         Args:
#             input_dict (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Results after global alignment, 'points' and keys in
#             input_dict['bbox3d_fields'] are updated in the result dict.
#         """
#         assert 'axis_align_matrix' in results, \
#             'axis_align_matrix is not provided in GlobalAlignment'

#         axis_align_matrix = results['axis_align_matrix']
#         assert axis_align_matrix.shape == (4, 4), \
#             f'invalid shape {axis_align_matrix.shape} for axis_align_matrix'
#         rot_mat = axis_align_matrix[:3, :3]
#         trans_vec = axis_align_matrix[:3, -1]

#         self._check_rot_mat(rot_mat)
#         self._rot_points(results, rot_mat)
#         self._trans_points(results, trans_vec)

#         return results

@TRANSFORMS.register_module()
class NormBoxes(BaseTransform):
    """
    Project unaligned point clouds to the axis of the first-frame-axis
    """

    def transform(self, results: dict) -> dict:
        gt_bboxes_3d = results['gt_bboxes_3d'] 
        gt_bboxes_3d_center = gt_bboxes_3d.gravity_center
        gt_bboxes_3d_size = gt_bboxes_3d.tensor[:, 3:6]
        norm_scale = results['avg_distance']
        
        gt_bboxes_3d_center_normed = gt_bboxes_3d_center / norm_scale
        gt_bboxes_3d_size_normed = gt_bboxes_3d_size / norm_scale

        # gt_bboxes_3d.gravity_center = gt_bboxes_3d_center_normed
        # gt_bboxes_3d.tensor[:, 3:6] = gt_bboxes_3d_size_normed

        gt_bboxes_concatenated = np.concatenate([gt_bboxes_3d_center_normed, gt_bboxes_3d_size_normed], axis=1)

        gt_bboxes_normed = results['box_type_3d'](
            gt_bboxes_concatenated, box_dim=6, with_yaw=False, origin=(.5, .5, .5))
        results['gt_bboxes_3d'] = gt_bboxes_normed

        return results

@TRANSFORMS.register_module()
class MultiViewPipeline(BaseTransform):
    """MultiViewPipeline used in nerfdet.

    Required Keys:

    - depth_info
    - img_prefix
    - img_info
    - lidar2img
    - c2w
    - cammrotc2w
    - lightpos
    - ray_info

    Modified Keys:

    - lidar2img

    Added Keys:

    - img
    - denorm_images
    - depth
    - c2w
    - camrotc2w
    - lightpos
    - pixels
    - raydirs
    - gt_images
    - gt_depths
    - nerf_sizes
    - depth_range

    Args:
        transforms (list[dict]): The transform pipeline
            used to process the imgs.
        n_images (int): The number of sampled views.
        mean (array): The mean values used in normalization.
        std (array): The variance values used in normalization.
        margin (int): The margin value. Defaults to 10.
        depth_range (array): The range of the depth.
            Defaults to [0.5, 5.5].
        loading (str): The mode of loading. Defaults to 'random'.
        nerf_target_views (int): The number of novel views.
        sample_freq (int): The frequency of sampling.
    """

    def __init__(self,
                 transforms: dict,
                 n_images: int,
                 mean: tuple = [123.675, 116.28, 103.53],
                 std: tuple = [58.395, 57.12, 57.375],
                 margin: int = 10,
                 depth_range: tuple = [0.5, 5.5],
                 loading: str = 'random',
                 nerf_target_views: int = 0,
                 sample_freq: int = 3):
        self.transforms = Compose(transforms)
        self.depth_transforms = Compose(transforms[1])
        self.n_images = n_images
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.margin = margin
        self.depth_range = depth_range
        self.loading = loading
        self.sample_freq = sample_freq
        self.nerf_target_views = nerf_target_views

    def transform(self, results: dict) -> dict:
        """Nerfdet transform function.

        Args:
            results (dict): Result dict from loading pipeline

        Returns:
            dict: The result dict containing the processed results.
            Updated key and value are described below.

                - img (list): The loaded origin image.
                - denorm_images (list): The denormalized image.
                - depth (list): The origin depth image.
                - c2w (list): The c2w matrixes.
                - camrotc2w (list): The rotation matrixes.
                - lightpos (list): The transform parameters of the camera.
                - pixels (list): Some pixel information.
                - raydirs (list): The ray-directions.
                - gt_images (list): The groundtruth images.
                - gt_depths (list): The groundtruth depth images.
                - nerf_sizes (array): The size of the groundtruth images.
                - depth_range (array): The range of the depth.

        Here we give a detailed explanation of some keys mentioned above.
        Let P_c be the coordinate of camera, P_w be the coordinate of world.
        There is such a conversion relationship: P_c = R @ P_w + T.
        The 'camrotc2w' mentioned above corresponds to the R matrix here.
        The 'lightpos' corresponds to the T matrix here. And if you put
        R and T together, you can get the camera extrinsics matrix. It
        corresponds to the 'c2w' mentioned above.
        """
        imgs = []
        depths = []
        extrinsics = []
        c2ws = []
        camrotc2ws = []
        lightposes = []
        pixels = []
        raydirs = []
        gt_images = []
        gt_depths = []
        denorm_imgs_list = []
        nerf_sizes = []

        if self.loading == 'random':
            ids = np.arange(len(results['img_info']))
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
            if self.nerf_target_views != 0:
                target_id = np.random.choice(
                    ids, self.nerf_target_views, replace=False)
                ids = np.setdiff1d(ids, target_id)
                ids = ids.tolist()
                target_id = target_id.tolist()

        elif self.loading == 'gap':
            # min_gap = 3
            # max_gap = 6
            # src_gap = np.random.choice(max_gap-min_gap+1, 1) + min_gap
            # randomly choose scr 1
            ids = np.arange(len(results['img_info']))
            src_1 = np.random.randint(0, len(ids)//2 - self.nerf_target_views//2 - 1, (1,))[0] # choose one from first half of images 
            src_3 = np.random.randint(len(ids)//2, len(ids)- self.nerf_target_views//2 - 1, (1,))[0]
            
            src_used_id = [src_1, src_1+self.nerf_target_views//2+1, src_3, src_3+self.nerf_target_views//2+1]
            target_id = []
            for k in range(self.nerf_target_views//2):
                target_id = target_id + [src_1+1+k, src_3+1+k]
            used_id = src_used_id + target_id
            # print(src_used_id, target_id, len(ids))
            # rest src
            replace = True if self.n_images > len(ids) else False
            rest_src = np.random.choice(np.setdiff1d(ids, np.array(used_id)), self.n_images-len(used_id), replace=replace)
            ids = rest_src.tolist() + src_used_id
            assert max(ids) < len(results['img_info'])
            # print(target_id)
            # print(ids)
            
        
        else:
            ids = np.arange(len(results['img_info']))
            begin_id = 0
            ids = np.arange(begin_id,
                            begin_id + self.n_images * self.sample_freq,
                            self.sample_freq)
            if self.nerf_target_views != 0:
                target_id = ids

        ratio = 0
        size = (240, 320)
        src_img_paths = []
        for i in ids:
            _results = dict()
            _results['img_path'] = results['img_info'][i]['filename']
            src_img_paths.append(results['img_info'][i]['filename'])
            _results = self.transforms(_results) # load and resize.
            imgs.append(_results['img']) # after resize, image is (239, 320, 3)            
            # normalize
            for key in _results.get('img_fields', ['img']):
                _results[key] = mmcv.imnormalize(_results[key], self.mean,
                                                 self.std, True) # to_rgb=True
            _results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=True)
            # pad
            for key in _results.get('img_fields', ['img']):
                padded_img = mmcv.impad(_results[key], shape=size, pad_val=0)
                _results[key] = padded_img
            _results['pad_shape'] = padded_img.shape # (240, 320, 3)
            _results['pad_fixed_size'] = size # (240, 320)
            ori_shape = _results['ori_shape'] # (968, 1296)
            aft_shape = _results['img_shape'] # (239, 320)
            ratio = ori_shape[0] / aft_shape[0]
            # prepare the depth information
            if 'depth_info' in results.keys():
                if '.npy' in results['depth_info'][i]['filename']:
                    _results['depth'] = np.load(
                        results['depth_info'][i]['filename'])
                else:
                    _results['depth'] = np.asarray((Image.open(
                        results['depth_info'][i]['filename']))) / 1000
                    _results['depth'] = mmcv.imresize(
                        _results['depth'], (aft_shape[1], aft_shape[0]))
                depths.append(_results['depth'])

            denorm_img = mmcv.imdenormalize(
                _results['img'], self.mean, self.std, to_bgr=True).astype(
                    np.uint8) / 255.0 # bgr. 
            denorm_imgs_list.append(denorm_img)
            height, width = padded_img.shape[:2]
            extrinsics.append(results['lidar2img']['extrinsic'][i])

        # prepare the nerf information
        if 'ray_info' in results.keys():
            intrinsics_nerf = results['lidar2img']['intrinsic'].copy()
            intrinsics_nerf[:2] = intrinsics_nerf[:2] / ratio # this is intrinsic for downsampling x4
            assert self.nerf_target_views > 0
            for i in target_id:
                c2ws.append(results['c2w'][i])
                camrotc2ws.append(results['camrotc2w'][i])
                lightposes.append(results['lightpos'][i])
                px, py = np.meshgrid(
                    np.arange(self.margin,
                              width - self.margin).astype(np.float32),
                    np.arange(self.margin,
                              height - self.margin).astype(np.float32))
                pixelcoords = np.stack((px, py),
                                       axis=-1).astype(np.float32)  # H x W x 2
                pixels.append(pixelcoords)
                raydir = get_dtu_raydir(pixelcoords, intrinsics_nerf,
                                        results['camrotc2w'][i])
                raydirs.append(np.reshape(raydir.astype(np.float32), (-1, 3)))
                # read target images
                temp_results = dict()
                temp_results['img_path'] = results['img_info'][i]['filename']

                temp_results = self.transforms(temp_results) # load and resize. tmp_results_['img']: (239, 320, 3)
                # normalize
                for key in temp_results.get('img_fields', ['img']):
                    temp_results[key] = mmcv.imnormalize(
                        temp_results[key], self.mean, self.std, True) # to_rgb = True
                temp_results['img_norm_cfg'] = dict(
                    mean=self.mean, std=self.std, to_rgb=True)
                # # pad
                # for key in temp_results.get('img_fields', ['img']):
                #     padded_img = mmcv.impad(
                #         temp_results[key], shape=size, pad_val=0)
                #     temp_results[key] = padded_img
                # temp_results['pad_shape'] = padded_img.shape
                # temp_results['pad_fixed_size'] = size
                # denormalize target_images.
                denorm_imgs = mmcv.imdenormalize(
                    temp_results['img'], self.mean, self.std,
                    to_bgr=False).astype(np.uint8) # fix bug: remove temp_results_. (240, 320, 3). set rgb
                gt_rgb_shape = denorm_imgs.shape

                gt_image = denorm_imgs # pad should not be taget image!
                nerf_sizes.append(np.array(gt_image.shape))
                # gt_image = np.reshape(gt_image, (-1, 3))
                gt_image = gt_image.transpose(2,0,1)
                gt_images.append(gt_image / 255.0)
                
                
                if 'depth_info' in results.keys():
                    if '.npy' in results['depth_info'][i]['filename']:
                        _results['depth'] = np.load(
                            results['depth_info'][i]['filename'])
                    else:
                        depth_image = Image.open(
                            results['depth_info'][i]['filename'])
                        _results['depth'] = np.asarray(depth_image) / 1000
                        _results['depth'] = mmcv.imresize(
                            _results['depth'],
                            (gt_rgb_shape[1], gt_rgb_shape[0])) # (w, h)

                    _results['depth'] = _results['depth']
                    gt_depth = _results['depth']
                    gt_depths.append(gt_depth)

        for key in _results.keys():
            if key not in ['img', 'img_info', 'img_path']:
                results[key] = _results[key]
        results['img'] = imgs # bug here.. imgs
        results['img_path'] = src_img_paths # manually add in img_path

        if 'ray_info' in results.keys():
            results['c2w'] = c2ws # only tgt view c2w!!
            results['intrinsic'] = intrinsics_nerf
            results['camrotc2w'] = camrotc2ws
            results['lightpos'] = lightposes
            results['pixels'] = pixels
            results['raydirs'] = raydirs
            results['gt_images'] = gt_images # bgr image, 255 normalized
            results['gt_depths'] = gt_depths
            results['nerf_sizes'] = nerf_sizes
            results['denorm_images'] = denorm_imgs_list
            results['depth_range'] = np.array([self.depth_range])

        if len(depths) != 0:
            results['depth'] = depths
        results['lidar2img']['extrinsic'] = extrinsics # w2c src view.
        return results


@TRANSFORMS.register_module()
class MultiViewPipeline_Tgt(BaseTransform):
    """MultiViewPipeline used in nerfdet.

    Required Keys:

    - depth_info
    - img_prefix
    - img_info
    - lidar2img
    - c2w
    - cammrotc2w
    - lightpos
    - ray_info

    Modified Keys:

    - lidar2img

    Added Keys:

    - img
    - denorm_images
    - depth
    - c2w
    - camrotc2w
    - lightpos
    - pixels
    - raydirs
    - gt_images
    - gt_depths
    - nerf_sizes
    - depth_range

    Args:
        transforms (list[dict]): The transform pipeline
            used to process the imgs.
        n_images (int): The number of sampled views.
        mean (array): The mean values used in normalization.
        std (array): The variance values used in normalization.
        margin (int): The margin value. Defaults to 10.
        depth_range (array): The range of the depth.
            Defaults to [0.5, 5.5].
        loading (str): The mode of loading. Defaults to 'random'.
        nerf_target_views (int): The number of novel views.
        sample_freq (int): The frequency of sampling.
    """

    def __init__(self,
                 transforms: dict,
                 n_images: int,
                 mean: tuple = [123.675, 116.28, 103.53],
                 std: tuple = [58.395, 57.12, 57.375],
                 margin: int = 10,
                 depth_range: tuple = [0.5, 5.5],
                 loading: str = 'random',
                 nerf_target_views: int = 0,
                 sample_freq: int = 3,
                 tgt_transforms=None):
        self.transforms = Compose(transforms)
        self.depth_transforms = Compose(transforms[1])
        self.n_images = n_images
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.margin = margin
        self.depth_range = depth_range
        self.loading = loading
        self.sample_freq = sample_freq
        self.nerf_target_views = nerf_target_views
        self.tgt_transforms = Compose(tgt_transforms)

    def transform(self, results: dict) -> dict:
        """Nerfdet transform function.

        Args:
            results (dict): Result dict from loading pipeline

        Returns:
            dict: The result dict containing the processed results.
            Updated key and value are described below.

                - img (list): The loaded origin image.
                - denorm_images (list): The denormalized image.
                - depth (list): The origin depth image.
                - c2w (list): The c2w matrixes.
                - camrotc2w (list): The rotation matrixes.
                - lightpos (list): The transform parameters of the camera.
                - pixels (list): Some pixel information.
                - raydirs (list): The ray-directions.
                - gt_images (list): The groundtruth images.
                - gt_depths (list): The groundtruth depth images.
                - nerf_sizes (array): The size of the groundtruth images.
                - depth_range (array): The range of the depth.

        Here we give a detailed explanation of some keys mentioned above.
        Let P_c be the coordinate of camera, P_w be the coordinate of world.
        There is such a conversion relationship: P_c = R @ P_w + T.
        The 'camrotc2w' mentioned above corresponds to the R matrix here.
        The 'lightpos' corresponds to the T matrix here. And if you put
        R and T together, you can get the camera extrinsics matrix. It
        corresponds to the 'c2w' mentioned above.
        """
        imgs = []
        depths = []
        extrinsics = []
        c2ws = []
        camrotc2ws = []
        lightposes = []
        pixels = []
        raydirs = []
        gt_images = []
        gt_depths = []
        denorm_imgs_list = []
        nerf_sizes = []

        if self.loading == 'random':
            ids = np.arange(len(results['img_info']))
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
            if self.nerf_target_views != 0:
                target_id = np.random.choice(
                    ids, self.nerf_target_views, replace=False)
                ids = np.setdiff1d(ids, target_id)
                ids = ids.tolist()
                target_id = target_id.tolist()

        elif self.loading == 'gap':
            # min_gap = 3
            # max_gap = 6
            # src_gap = np.random.choice(max_gap-min_gap+1, 1) + min_gap
            # randomly choose scr 1
            ids = np.arange(len(results['img_info']))
            src_1 = np.random.randint(0, len(ids)//2 - self.nerf_target_views//2 - 1, (1,))[0] # choose one from first half of images 
            src_3 = np.random.randint(len(ids)//2, len(ids)- self.nerf_target_views//2 - 1, (1,))[0]
            
            src_used_id = [src_1, src_1+self.nerf_target_views//2+1, src_3, src_3+self.nerf_target_views//2+1]
            target_id = []
            for k in range(self.nerf_target_views//2):
                target_id = target_id + [src_1+1+k, src_3+1+k]
            used_id = src_used_id + target_id
            # rest src
            replace = True if self.n_images > len(ids) else False
            rest_src = np.random.choice(np.setdiff1d(ids, np.array(used_id)), self.n_images-len(used_id), replace=replace)
            ids = rest_src.tolist() + src_used_id
            assert max(ids) < len(results['img_info'])
            # print(target_id)
            # print(ids)
            
        
        else:
            ids = np.arange(len(results['img_info']))
            begin_id = 0
            ids = np.arange(begin_id,
                            begin_id + self.n_images * self.sample_freq,
                            self.sample_freq)
            if self.nerf_target_views != 0:
                target_id = ids

        ratio = 0
        size = (240, 320)
        src_img_paths = []
        for i in ids:
            _results = dict()
            _results['img_path'] = results['img_info'][i]['filename']
            src_img_paths.append(results['img_info'][i]['filename'])
            _results = self.transforms(_results) # load and resize.
            imgs.append(_results['img']) # after resize, image is (239, 320, 3)            
            # normalize
            for key in _results.get('img_fields', ['img']):
                _results[key] = mmcv.imnormalize(_results[key], self.mean,
                                                 self.std, True) # to_rgb=True
            _results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=True)
            # pad
            for key in _results.get('img_fields', ['img']):
                padded_img = mmcv.impad(_results[key], shape=size, pad_val=0)
                _results[key] = padded_img
            _results['pad_shape'] = padded_img.shape # (240, 320, 3)
            _results['pad_fixed_size'] = size # (240, 320)
            ori_shape = _results['ori_shape'] # (968, 1296)
            aft_shape = _results['img_shape'] # (239, 320)
            ratio = ori_shape[0] / aft_shape[0]
            # prepare the depth information
            if 'depth_info' in results.keys():
                if '.npy' in results['depth_info'][i]['filename']:
                    _results['depth'] = np.load(
                        results['depth_info'][i]['filename'])
                else:
                    _results['depth'] = np.asarray((Image.open(
                        results['depth_info'][i]['filename']))) / 1000
                    _results['depth'] = mmcv.imresize(
                        _results['depth'], (aft_shape[1], aft_shape[0]))
                depths.append(_results['depth'])

            denorm_img = mmcv.imdenormalize(
                _results['img'], self.mean, self.std, to_bgr=False).astype(
                    np.uint8) / 255.0 # rgb!! 
            denorm_imgs_list.append(denorm_img)
            height, width = padded_img.shape[:2]
            extrinsics.append(results['lidar2img']['extrinsic'][i])

        # prepare the nerf information
        if 'ray_info' in results.keys():
            intrinsics_nerf = results['lidar2img']['intrinsic'].copy()
            tgt_shape = (120,160)
            tgt_ratio = ori_shape[0] / tgt_shape[0]
            intrinsics_nerf[:2] = intrinsics_nerf[:2] / tgt_ratio # this is intrinsic for downsampling x4
            assert self.nerf_target_views > 0
            for i in target_id:
                c2ws.append(results['c2w'][i])
                camrotc2ws.append(results['camrotc2w'][i])
                lightposes.append(results['lightpos'][i])
                px, py = np.meshgrid(
                    np.arange(self.margin,
                              width - self.margin).astype(np.float32),
                    np.arange(self.margin,
                              height - self.margin).astype(np.float32))
                pixelcoords = np.stack((px, py),
                                       axis=-1).astype(np.float32)  # H x W x 2
                pixels.append(pixelcoords)
                raydir = get_dtu_raydir(pixelcoords, intrinsics_nerf,
                                        results['camrotc2w'][i])
                raydirs.append(np.reshape(raydir.astype(np.float32), (-1, 3)))
                # read target images
                temp_results = dict()
                temp_results['img_path'] = results['img_info'][i]['filename']

                temp_results = self.tgt_transforms(temp_results) # load and resize. tmp_results_['img']: (239, 320, 3)
                # # normalize
                # for key in temp_results.get('img_fields', ['img']):
                #     temp_results[key] = mmcv.imnormalize(
                #         temp_results[key], self.mean, self.std, True) # to_rgb = True
                # temp_results['img_norm_cfg'] = dict(
                #     mean=self.mean, std=self.std, to_rgb=True)
                # # # pad
                # # for key in temp_results.get('img_fields', ['img']):
                # #     padded_img = mmcv.impad(
                # #         temp_results[key], shape=size, pad_val=0)
                # #     temp_results[key] = padded_img
                # # temp_results['pad_shape'] = padded_img.shape
                # # temp_results['pad_fixed_size'] = size
                # # denormalize target_images.
                # denorm_imgs = mmcv.imdenormalize(
                #     temp_results['img'], self.mean, self.std,
                #     to_bgr=False).astype(np.uint8) # fix bug: remove temp_results_. (240, 320, 3). set rgb
                
                denorm_imgs = temp_results['img'][:,:,::-1] # to rgb
                gt_rgb_shape = denorm_imgs.shape

                gt_image = denorm_imgs # pad should not be taget image!
                nerf_sizes.append(np.array(gt_image.shape))
                # gt_image = np.reshape(gt_image, (-1, 3))
                gt_image = gt_image.transpose(2,0,1)
                gt_images.append(gt_image / 255.0)
                
                
                if 'depth_info' in results.keys():
                    if '.npy' in results['depth_info'][i]['filename']:
                        _results['depth'] = np.load(
                            results['depth_info'][i]['filename'])
                    else:
                        depth_image = Image.open(
                            results['depth_info'][i]['filename'])
                        _results['depth'] = np.asarray(depth_image) / 1000
                        _results['depth'] = mmcv.imresize(
                            _results['depth'],
                            (gt_rgb_shape[1], gt_rgb_shape[0])) # (w, h)

                    _results['depth'] = _results['depth']
                    gt_depth = _results['depth']
                    gt_depths.append(gt_depth)

        for key in _results.keys():
            if key not in ['img', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs # bug here.. imgs
        results['img_path'] = src_img_paths # manually add in img_path

        if 'ray_info' in results.keys():
            results['c2w'] = c2ws # only tgt view c2w!!
            results['intrinsic'] = intrinsics_nerf
            results['camrotc2w'] = camrotc2ws
            results['lightpos'] = lightposes
            results['pixels'] = pixels
            results['raydirs'] = raydirs
            results['gt_images'] = gt_images # bgr image, 255 normalized
            results['gt_depths'] = gt_depths
            results['nerf_sizes'] = nerf_sizes
            results['denorm_images'] = denorm_imgs_list
            results['depth_range'] = np.array([self.depth_range])

        if len(depths) != 0:
            results['depth'] = depths
        results['lidar2img']['extrinsic'] = extrinsics # w2c src view.
        return results


@TRANSFORMS.register_module()
class MultiViewPipeline_ARKit(BaseTransform):
    def __init__(self,
                 transforms: dict,
                 n_images: int,
                 mean: tuple = [123.675, 116.28, 103.53],
                 std: tuple = [58.395, 57.12, 57.375],
                 margin: int = 10,
                 depth_range: tuple = [0.5, 5.5],
                 loading: str = 'random',
                 nerf_target_views: int = 0,
                 sample_freq: int = 3,
                 tgt_transforms=None):
        self.transforms = Compose(transforms)
        self.depth_transforms = Compose(transforms[1])
        self.n_images = n_images
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.margin = margin
        self.depth_range = depth_range
        self.loading = loading
        self.sample_freq = sample_freq
        self.nerf_target_views = nerf_target_views
        self.tgt_transforms = Compose(tgt_transforms)

    def transform(self, results: dict) -> dict: 
        imgs = []
        depths = []
        extrinsics = []
        intrinsics = []
        c2ws = []
        camrotc2ws = []
        lightposes = []
        pixels = []
        raydirs = []
        gt_images = []
        gt_depths = []
        denorm_imgs_list = []
        nerf_sizes = []

        if self.loading == 'random':
            ids = np.arange(len(results['img_info']))
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
            if self.nerf_target_views != 0:
                target_id = np.random.choice(
                    ids, self.nerf_target_views, replace=False)
                ids = np.setdiff1d(ids, target_id) # here will remove duplicate values in ids. so causing len(ids) < n_images
                ids = ids.tolist()
                target_id = target_id.tolist()

        elif self.loading == 'gap':
            # min_gap = 3
            # max_gap = 6
            # src_gap = np.random.choice(max_gap-min_gap+1, 1) + min_gap
            # randomly choose scr 1
            ids = np.arange(len(results['img_info']))
            src_1 = np.random.randint(0, len(ids)//2 - self.nerf_target_views//2 - 1, (1,))[0] # choose one from first half of images 
            src_3 = np.random.randint(len(ids)//2, len(ids)- self.nerf_target_views//2 - 1, (1,))[0]
            
            src_used_id = [src_1, src_1+self.nerf_target_views//2+1, src_3, src_3+self.nerf_target_views//2+1]
            target_id = []
            for k in range(self.nerf_target_views//2):
                target_id = target_id + [src_1+1+k, src_3+1+k]
            used_id = src_used_id + target_id
            # rest src
            replace = True if self.n_images > len(ids) else False
            rest_src = np.random.choice(np.setdiff1d(ids, np.array(used_id)), self.n_images-len(used_id), replace=replace)
            ids = rest_src.tolist() + src_used_id
            assert max(ids) < len(results['img_info'])
            # print(target_id)
            # print(ids)
            
        else:
            ids = np.arange(len(results['img_info']))
            begin_id = 0
            ids = np.arange(begin_id,
                            begin_id + self.n_images * self.sample_freq,
                            self.sample_freq)
            if self.nerf_target_views != 0:
                target_id = ids

        ratio = 0
        size = (240, 320)
        tgt_shape = (120,160) # novel view size
        src_img_paths = []
        for i in ids:
            _results = dict()
            _results['img_path'] = results['img_info'][i]['filename']
            src_img_paths.append(results['img_info'][i]['filename'])
            _results = self.transforms(_results) # load and resize.
            imgs.append(_results['img']) # after resize, image is (239, 320, 3)            
            # normalize
            for key in _results.get('img_fields', ['img']):
                _results[key] = mmcv.imnormalize(_results[key], self.mean,
                                                 self.std, True) # to_rgb=True
            _results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=True)
            # pad
            for key in _results.get('img_fields', ['img']):
                padded_img = mmcv.impad(_results[key], shape=size, pad_val=0)
                _results[key] = padded_img
            _results['pad_shape'] = padded_img.shape # (240, 320, 3)
            _results['pad_fixed_size'] = size # (240, 320)
            ori_shape = _results['ori_shape'] # (968, 1296)
            aft_shape = _results['img_shape'] # (239, 320)
            ratio = ori_shape[0] / aft_shape[0]
            # prepare the depth information
            if 'depth_info' in results.keys():
                if '.npy' in results['depth_info'][i]['filename']:
                    _results['depth'] = np.load(
                        results['depth_info'][i]['filename'])
                else:
                    _results['depth'] = np.asarray((Image.open(
                        results['depth_info'][i]['filename']))) / 1000
                    _results['depth'] = mmcv.imresize(
                        _results['depth'], (aft_shape[1], aft_shape[0]))
                depths.append(_results['depth'])

            denorm_img = mmcv.imdenormalize(
                _results['img'], self.mean, self.std, to_bgr=False).astype(
                    np.uint8) / 255.0 # rgb!! 
            denorm_imgs_list.append(denorm_img)
            height, width = padded_img.shape[:2]
            extrinsics.append(results['lidar2img']['extrinsic'][i])
            intrinsics.append(results['lidar2img']['intrinsic'][i])

        # prepare the nerf information
        if 'ray_info' in results.keys():
            intrinsics_nerf_list = []
            assert self.nerf_target_views > 0
            for i in target_id:
                c2ws.append(results['c2w'][i])
                camrotc2ws.append(results['camrotc2w'][i])
                lightposes.append(results['lightpos'][i])
                # intrinsic
                intrinsics_nerf = results['lidar2img']['intrinsic'][i].copy()
                # tgt_shape = (120,160)
                tgt_ratio = ori_shape[0] / tgt_shape[0]
                intrinsics_nerf[:2] = intrinsics_nerf[:2] / tgt_ratio
                intrinsics_nerf_list.append(intrinsics_nerf)
                # rays
                px, py = np.meshgrid(
                    np.arange(self.margin,
                              width - self.margin).astype(np.float32),
                    np.arange(self.margin,
                              height - self.margin).astype(np.float32))
                pixelcoords = np.stack((px, py),
                                       axis=-1).astype(np.float32)  # H x W x 2
                pixels.append(pixelcoords)
                raydir = get_dtu_raydir(pixelcoords, intrinsics_nerf,
                                        results['camrotc2w'][i])
                raydirs.append(np.reshape(raydir.astype(np.float32), (-1, 3)))
                # read target images
                temp_results = dict()
                temp_results['img_path'] = results['img_info'][i]['filename']

                temp_results = self.tgt_transforms(temp_results) # load and resize. tmp_results_['img']: (239, 320, 3)
                # # normalize
                # for key in temp_results.get('img_fields', ['img']):
                #     temp_results[key] = mmcv.imnormalize(
                #         temp_results[key], self.mean, self.std, True) # to_rgb = True
                # temp_results['img_norm_cfg'] = dict(
                #     mean=self.mean, std=self.std, to_rgb=True)
                # # # pad
                # # for key in temp_results.get('img_fields', ['img']):
                # #     padded_img = mmcv.impad(
                # #         temp_results[key], shape=size, pad_val=0)
                # #     temp_results[key] = padded_img
                # # temp_results['pad_shape'] = padded_img.shape
                # # temp_results['pad_fixed_size'] = size
                # # denormalize target_images.
                # denorm_imgs = mmcv.imdenormalize(
                #     temp_results['img'], self.mean, self.std,
                #     to_bgr=False).astype(np.uint8) # fix bug: remove temp_results_. (240, 320, 3). set rgb
                
                denorm_imgs = temp_results['img'][:,:,::-1] # to rgb
                gt_rgb_shape = denorm_imgs.shape

                gt_image = denorm_imgs # pad should not be taget image!
                nerf_sizes.append(np.array(gt_image.shape))
                # gt_image = np.reshape(gt_image, (-1, 3))
                gt_image = gt_image.transpose(2,0,1)
                gt_images.append(gt_image / 255.0)
                
                
                if 'depth_info' in results.keys():
                    if '.npy' in results['depth_info'][i]['filename']:
                        _results['depth'] = np.load(
                            results['depth_info'][i]['filename'])
                    else:
                        depth_image = Image.open(
                            results['depth_info'][i]['filename'])
                        _results['depth'] = np.asarray(depth_image) / 1000
                        _results['depth'] = mmcv.imresize(
                            _results['depth'],
                            (gt_rgb_shape[1], gt_rgb_shape[0])) # (w, h)

                    _results['depth'] = _results['depth']
                    gt_depth = _results['depth']
                    gt_depths.append(gt_depth)

        for key in _results.keys():
            if key not in ['img', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs # bug here.. imgs
        results['img_path'] = src_img_paths # manually add in img_path

        if 'ray_info' in results.keys():
            results['c2w'] = c2ws # only tgt view c2w!!
            results['intrinsic'] = intrinsics_nerf_list
            results['camrotc2w'] = camrotc2ws
            results['lightpos'] = lightposes
            results['pixels'] = pixels
            results['raydirs'] = raydirs
            results['gt_images'] = gt_images # bgr image, 255 normalized
            results['gt_depths'] = gt_depths
            results['nerf_sizes'] = nerf_sizes
            results['denorm_images'] = denorm_imgs_list
            results['depth_range'] = np.array([self.depth_range])

        if len(depths) != 0:
            results['depth'] = depths
        results['lidar2img']['extrinsic'] = extrinsics # w2c src view.
        results['lidar2img']['intrinsic'] = intrinsics
        return results



@TRANSFORMS.register_module()
class RandomShiftOrigin(BaseTransform):

    def __init__(self, std):
        self.std = std

    def transform(self, results):
        shift = np.random.normal(.0, self.std, 3)
        results['lidar2img']['origin'] += shift
        return results
