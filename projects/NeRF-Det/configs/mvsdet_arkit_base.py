_base_ = ['../../../configs/_base_/default_runtime.py']

# this script is used to train nerfdet.
custom_imports = dict(imports=['projects.NeRF-Det.nerfdet'])
prior_generator = dict(
    type='AlignedAnchor3DRangeGenerator',
    ranges=[[-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]],
    rotations=[.0])

 # use ImVoxelHead written in mmdet3d and original nerfdet
model = dict(
    type='MVSDet',
    data_preprocessor=dict(
        type='NeRFDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=10), # change here to 1
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    neck_3d=dict(
        type='IndoorImVoxelNeck',
        in_channels=256,
        out_channels=128,
        n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='ImVoxelHead_ARKit',
        n_classes=17,
        n_levels=3,
        n_channels=128,
        n_reg_outs=7,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        prior_generator=prior_generator),
    prior_generator=prior_generator,
    voxel_size=[.16, .16, .2],
    n_voxels=[40, 40, 16],
    aabb=([-2.7, -2.7, -0.78], [3.7, 3.7, 1.78]),
    near_far_range=[0.2, 8.0],
    N_samples=64,
    N_rand=2048,
    nerf_mode='image',
    depth_supervise=False,
    use_nerf_mask=True,
    nerf_sample_view=20,
    squeeze_scale=4,
    nerf_density=True,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.25, score_thr=.01)
    )



dataset_type = 'MultiViewARKitDataset'
data_root = '/mnt/6202BA0A02B9E369/arkit/' # local

class_names = [
    "cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
            "sink", "washer", "toilet", "bathtub", "oven", # 5..10
            "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
            "tv_monitor", "sofa"
]
metainfo = dict(CLASSES=class_names)
file_client_args = dict(backend='disk')

use_depth = False

input_modality = dict(
    use_camera=True,
    use_depth=use_depth,
    use_lidar=False,
    use_neuralrecon_depth=False,
    use_ray=True)
backend_args = None

train_collect_keys = [
    'img', 'gt_bboxes_3d', 'gt_labels_3d', 'lightpos', 'nerf_sizes', 'raydirs',
    'gt_images', 'gt_depths', 'denorm_images'
]

test_collect_keys = [
    'img',
    'lightpos',
    'nerf_sizes',
    'raydirs',
    'gt_images',
    'gt_depths',
    'denorm_images',
]

# whether depth input
if use_depth == True:
    train_collect_keys.append('depth')
    test_collect_keys.append('depth')


train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline_ARKit',
        n_images=50,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(320, 240), keep_ratio=True),
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5], # for what purpose?
        loading='random',
        nerf_target_views=10),
    dict(type='RandomShiftOrigin', std=(.7, .7, .0)),
    dict(type='PackNeRFDetInputs', keys=train_collect_keys)
]

test_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline_ARKit',
        n_images=101,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(320, 240), keep_ratio=True),
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5],
        loading='random',
        nerf_target_views=1),
    dict(type='PackNeRFDetInputs', keys=test_collect_keys)
]



train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2, # arkit only repeat twice
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file= data_root + 'arkit_infos_train_new_ReverseYaw.pkl', # TODO to be changed
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Depth',
            metainfo=metainfo)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= data_root + 'arkit_infos_val_new_ReverseYaw.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        filter_empty_gt=True,
        box_type_3d='Depth',
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(type='IndoorMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
test_cfg = dict()
val_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]


# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best=['mAP_0.25'], rule="greater", interval=1, max_keep_ckpts=1),
    )


vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used