ann_file_test = 'data/normal_vs_3critical/normal_vs_3critical_test.txt'
ann_file_train = 'data/normal_vs_3critical/normal_vs_3critical_train.txt'
ann_file_val = 'data/normal_vs_3critical/normal_vs_3critical_val.txt'
auto_scale_lr = dict(base_batch_size=256, enable=True)
base_lr = 1e-05
data_root = 'data/normal_vs_3critical/train'
data_root_test = 'data/normal_vs_3critical/test'
data_root_val = 'data/normal_vs_3critical/val'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=20,
        save_best='auto',
        save_last=True,
        save_optimizer=True,
        type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw_confusion_matrix=True, type='VisualizationHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = 'work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics700-rgb_transfer/epoch_38.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        backbone_drop_path_rate=0.0,
        clip_pretrained=True,
        double_lmhra=True,
        drop_path_rate=0.0,
        dw_reduction=1.5,
        heads=12,
        input_resolution=224,
        layers=12,
        mlp_dropout=[
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        mlp_factor=4.0,
        n_dim=768,
        n_head=12,
        n_layers=4,
        no_lmhra=True,
        patch_size=16,
        pretrained='ViT-B/16',
        return_list=[
            8,
            9,
            10,
            11,
        ],
        t_size=8,
        temporal_downsample=False,
        type='UniFormerV2',
        width=768),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=768,
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                1.0,
                0.3,
            ],
            loss_weight=1.0,
            type='CrossEntropyLoss'),
        num_classes=4,
        type='UniFormerHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            114.75,
            114.75,
            114.75,
        ],
        std=[
            57.375,
            57.375,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
num_frames = 8
optim_wrapper = dict(
    clip_grad=dict(max_norm=20, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=1e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0))
param_scheduler = [
    dict(
        T_max=12,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=12,
        eta_min_ratio=0.1,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/normal_vs_3critical/normal_vs_3critical_test.txt',
        data_prefix=dict(video='data/normal_vs_3critical/test'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=8, num_clips=4, test_mode=True, type='UniformSample'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='ThreeCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(metrics=[
    dict(type='AccMetric'),
    dict(num_classes=4, type='ConfusionMatrix'),
])
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, num_clips=4, test_mode=True, type='UniformSample'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=50, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/normal_vs_3critical/normal_vs_3critical_train.txt',
        data_prefix=dict(video='data/normal_vs_3critical/train'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(clip_len=8, num_clips=1, type='UniformSample'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                magnitude=7,
                num_layers=4,
                op='RandAugment',
                type='PytorchVideoWrapper'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, num_clips=1, type='UniformSample'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        magnitude=7,
        num_layers=4,
        op='RandAugment',
        type='PytorchVideoWrapper'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/normal_vs_3critical/normal_vs_3critical_val.txt',
        data_prefix=dict(video='data/normal_vs_3critical/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=8, num_clips=1, test_mode=True, type='UniformSample'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metrics=[
    dict(type='AccMetric'),
    dict(num_classes=4, type='ConfusionMatrix'),
])
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=8, num_clips=1, test_mode=True, type='UniformSample'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics700-rgb_transfer'
