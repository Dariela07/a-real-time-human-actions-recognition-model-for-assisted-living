ann_file_test = 'data/normal_vs_3critical/normal_vs_3critical_test.txt'
ann_file_train = 'data/normal_vs_3critical/normal_vs_3critical_train.txt'
ann_file_val = 'data/normal_vs_3critical/normal_vs_3critical_val.txt'
data_root = 'data/normal_vs_3critical/train'
data_root_test = 'data/normal_vs_3critical/test'
data_root_val = 'data/normal_vs_3critical/val'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(interval=3, save_best='auto', type='CheckpointHook'),
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
load_from = 'work_dirs/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb-transfer_jointST/last_checkpoint'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        attention_type='joint_space_time',
        dropout_ratio=0.0,
        embed_dims=768,
        img_size=224,
        in_channels=3,
        norm_cfg=dict(eps=1e-06, type='LN'),
        num_frames=8,
        patch_size=16,
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',
        transformer_layers=None,
        type='TimeSformer'),
    cls_head=dict(
        average_clips='prob',
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
        type='TimeSformerHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(
        lr=0.0001,
        momentum=0.9,
        nesterov=True,
        type='SGD',
        weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict({
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=50,
        begin=5,
        by_epoch=True,
        end=50,
        eta_min=0,
        type='CosineAnnealingLR'),
]
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
                clip_len=8,
                frame_interval=32,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
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
    dict(
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
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
    max_epochs=60, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='data/normal_vs_3critical/normal_vs_3critical_train.txt',
        data_prefix=dict(video='data/normal_vs_3critical/train'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=8,
                frame_interval=32,
                num_clips=1,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale_range=(
                256,
                320,
            ), type='RandomRescale'),
            dict(size=224, type='RandomCrop'),
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
    dict(clip_len=8, frame_interval=32, num_clips=1, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale_range=(
        256,
        320,
    ), type='RandomRescale'),
    dict(size=224, type='RandomCrop'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='data/normal_vs_3critical/normal_vs_3critical_val.txt',
        data_prefix=dict(video='data/normal_vs_3critical/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=8,
                frame_interval=32,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
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
    dict(
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
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
work_dir = 'work_dirs/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb-transfer_jointST/test_results_last'
