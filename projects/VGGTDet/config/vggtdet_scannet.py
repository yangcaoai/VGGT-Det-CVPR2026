_base_ = ['./vggt_res50_2x_low_res_depth.py']


resume = True

custom_imports = dict(
    imports=['projects.VGGTDet.vggtdet', 'mmdet3d.evaluation.metrics.Indoor_NVS'],
    allow_failed_imports=False)

prior_generator = dict(
    type='AlignedAnchor3DRangeGenerator',
    ranges=[[-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]],
    rotations=[.0])

_token_dim_ = 512
_decoder_layer_num = 8
model = dict(
    type='VGGTDet',
    near_far_range=[0.2, 5.0],
    rgb_supervision = True, # TODO for nvs
    depth_supervise=False, # add depth loss in the training.
    use_nerf_mask=False,
    gs_cfg = dict(
        use_rgb_gaussian=True,
        d_feature=256,
        num_monocular_samples=12, # TODO how many depth planes
        num_surfaces=1,
        use_transmittance=False,
        gaussians_per_pixel=3, 
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
        ),
    ),
    vis_dir = None,
    visualize_bbox = False,
    topk=3, # for detection.
    decoder_cfg = dict( # the same with 3detr
        dec_dim=_token_dim_,
        dec_nhead=4,
        dec_ffn_dim=_token_dim_,
        dec_dropout=0.1,
        dec_nlayers=_decoder_layer_num
    ),
    bbox_head=dict(
        type='VGGTDetHead',
        # bbox_loss=dict(type='AxisAlignedIoULoss', loss_weight=1.0),
        n_classes=18,
        n_levels=_decoder_layer_num,
        n_channels=_token_dim_,
        n_reg_outs=6,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        prior_generator=prior_generator,
        mlp_dropout=0.3,
        matcher_cost_weights=dict(
            cls=1.0, 
            center=0.0, 
            obj_ness=0.0, 
            giou=2.0
        ),
        loss_weights=dict(
            center_loss=5.0,
            size_loss=1.0,
            cls_loss=1.0,
            objness_loss=1.0,
            iou_loss=1.0,
            not_objness_loss = 0.25
        ),
        learn_center_diff=True,
        if_v2_head=True,
        if_project_frist_frame_back=True,
        visualize_path='vis_dir/',
        matcher='one2more',
        matcher_iou_thres=0.1,
        matcher_max_dynamic_samples=5
        ),
    num_queries=256,
    token_dim=_token_dim_,
    test_only_last_layer=True,
    if_learnable_query=False,
    if_use_gt_query=False,
    if_mix_precision=True,
    use_multi_layers=True,
    if_simpler_project=True,
    if_use_pred_pc_query=True,
    if_use_atten_sample=False,
    atten_sample_ratio=10,
    if_use_atten_fps=True,
    lambda_dist=0.8,
    if_task_query=True
    )

dataset_type = 'MultiViewScanNetDataset'
# Configure the data_root path to your dataset location
# data_root = '/path/to/your/scannet/data/'
data_root = './data/scannet/'


class_names = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'garbagebin'
]
metainfo = dict(CLASSES=class_names)
file_client_args = dict(backend='disk')

use_depth = False
input_modality = dict(use_depth=use_depth) # use_depth will load depth map during multi-view pipeline
backend_args = None

train_collect_keys = [
    'img', 'gt_bboxes_3d', 'gt_labels_3d', 'lightpos', 'nerf_sizes', 'raydirs',
    'gt_images',  'denorm_images',
    'c2w', 'intrinsic', 'points', 'pose_matrix', 'axis_align_matrix', 'avg_distance'
    ] # here add depth will load depth as input

test_collect_keys = [
    'img',
    'lightpos',
    'nerf_sizes',
    'raydirs',
    'gt_images',
    'denorm_images',
    'c2w', 'intrinsic', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'pose_matrix', 'axis_align_matrix', 'avg_distance'
]

# if use_depth == True:
#     train_collect_keys.append('depth')
#     test_collect_keys.append('depth')


n_points = 100000

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args,
        data_root=data_root),
    dict(type='LoadAnnotations3D'),
    # dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='PointSample', num_points=n_points),
    dict(
        type='MultiViewPipeline_Tgt',
        n_images=42,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5], # for what purpose?
        loading='gap',
        nerf_target_views=2,
        tgt_transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ]
        ),
    dict(type='RandomShiftOrigin', std=(.7, .7, .0)),
    dict(type='ProjectPCtoFirstFrameAndNorm', coord_type='DEPTH'),
    # dict(type='NormBoxes'),
    dict(type='PackNeRFDetInputs', keys=train_collect_keys)
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args,
        data_root=data_root),
    dict(type='LoadAnnotations3D'),
    # dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='PointSample', num_points=n_points),
    dict(
        type='MultiViewPipeline_Tgt',
        n_images=81,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5],
        loading='random',
        nerf_target_views=1,
        tgt_transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ]
        ),
    dict(type='ProjectPCtoFirstFrameAndNorm', coord_type='DEPTH'),
    # dict(type='NormBoxes'),
    dict(type='PackNeRFDetInputs', keys=test_collect_keys)
]

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file= 'scannet_infos_train_pts.pkl', #'scannet_infos_train_pts_10scenes.pkl', #'scannet_infos_train_pts.pkl',
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Depth',
            metainfo=metainfo)))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file= 'scannet_infos_val_pts.pkl', #'scannet_infos_train_pts_10scenes.pkl', #'scannet_infos_val_pts.pkl', # 
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        filter_empty_gt=True,
        box_type_3d='Depth',
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = [dict(type='IndoorMetric')]
# val_evaluator = [dict(type='IndoorMetric'), dict(type='MVSMetric')]
test_evaluator = val_evaluator

_warm_epoch=0
_max_epoch=400
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_max_epoch, val_interval=2)
test_cfg = dict()
val_cfg = dict()

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
#     paramwise_cfg=dict(
#         custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
#     clip_grad=dict(max_norm=35., norm_type=2))


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2.5e-4,
        weight_decay=1e-4
    ),
    clip_grad=dict(max_norm=35., norm_type=2)
)


param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=1,  # 0.002
    #     end_factor=1e-6 / 5e-4,
    #     by_epoch=True,
    #     begin=0,
    # ),
    dict(
        type='CosineAnnealingLR',
        T_max=_max_epoch-1,  # max_epochs - 1
        eta_min=1e-6,
        by_epoch=True,
        begin=_warm_epoch,
        end=_max_epoch
    )
]




default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best=['mAP_0.25'], rule="greater", interval=2, max_keep_ckpts=1000),
    logger=dict(type='LoggerHook', interval=10)
    # checkpoint=dict(type='CheckpointHook', save_best='ssim', rule="greater"),
    # checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1)
    )



vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs={
            'project': 'vggt_det',
            'group': 'baseline',
            'entity':'3dv_team', 
            'name': '4layer_scannet_axis_no_norm_predpc_c2lr_400e_atten_fps_lmdis_08_one2more_matching_task_query_again',
            'notes': 'debug'
         })]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
