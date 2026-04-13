_base_ = ['./mvsdet_arkit_base.py']

prior_generator = dict(
    type='AlignedAnchor3DRangeGenerator',
    ranges=[[-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]],
    rotations=[.0])

custom_imports = dict(
    imports=['projects.NeRF-Det.nerfdet.mvsdet', 'mmdet3d.evaluation.metrics.Indoor_NVS'],
    allow_failed_imports=False)

 # use ImVoxelHead written in mmdet3d and original nerfdet
model = dict(
    type='MVSDet',
    near_far_range=[0.5, 5.5], # TODO change
    rgb_supervision = True, # TODO whether NVS
    depth_supervise=False, # add depth loss in the training.
    use_nerf_mask=False,
    gs_cfg = dict(
        use_rgb_gaussian=True,
        d_feature=256,
        num_monocular_samples=12, # TODO how many depth planes
        num_surfaces=1,
        use_transmittance=False,
        gaussians_per_pixel=3, # TODO how many gpp.
        gaussian_adapter_cfg = dict(
            gaussian_scale_min=0.5,
            gaussian_scale_max=15.0, # tune it!
            sh_degree=4
        ),
        opacity_mapping = dict(
            initial=0.0,
            final=0.0,
            warm_up=1
        ),
        decoder = dict(
            name="splatting_cuda"
        ),
        dataset = dict(
            background_color = [0.0, 0.0, 0.0]
        )
    ),
    vis_dir = None,
    visualize_bbox = False,
    topk=3 # for detection.
    )

dataset_type = 'MultiViewARKitDataset'
# data_root = '/mnt/6202BA0A02B9E369/arkit/' # loacl
data_root = '/ssd/ytxu/nerf/data/arkit/' # server
# data_root = '/scratch/users/nus/e0546245/nerf/data/scannet/' # nscc

class_names = [
    "cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
            "sink", "washer", "toilet", "bathtub", "oven", # 5..10
            "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
            "tv_monitor", "sofa"
]
metainfo = dict(CLASSES=class_names)
file_client_args = dict(backend='disk')

use_depth = False
input_modality = dict(use_depth=use_depth) # use_depth will load depth map during multi-view pipeline
backend_args = None

train_collect_keys = [
    'img', 'gt_bboxes_3d', 'gt_labels_3d', 'lightpos', 'nerf_sizes', 'raydirs',
    'gt_images', 'gt_depths', 'denorm_images',
    'c2w', 'intrinsic'
    ] # here add depth will load depth as input

test_collect_keys = [
    'img',
    'lightpos',
    'nerf_sizes',
    'raydirs',
    'gt_images',
    'gt_depths',
    'denorm_images',
    'c2w', 'intrinsic'
]

if use_depth == True:
    train_collect_keys.append('depth')
    test_collect_keys.append('depth')

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline_ARKit',
        n_images=42,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(320, 240), keep_ratio=True), # original size
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5], # for what purpose?
        loading='gap',
        nerf_target_views=2,
        tgt_transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(160, 120), keep_ratio=True),
        ]),
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
        nerf_target_views=1,
        tgt_transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(160, 120), keep_ratio=True),
        ]
        ),
    dict(type='PackNeRFDetInputs', keys=test_collect_keys)
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file= data_root + 'arkit_infos_train_new_ReverseYaw.pkl',
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

val_evaluator = [dict(type='IndoorMetric'), dict(type='NVSMetric'), dict(type='MVSMetric')]
# val_evaluator = [dict(type='IndoorMetric'), dict(type='MVSMetric')]
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best=['mAP_0.25'], rule="greater", interval=1, max_keep_ckpts=1),
    # checkpoint=dict(type='CheckpointHook', save_best='ssim', rule="greater"),
    # checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1)
    )

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')