_base_ = [
    '../../_base_/models/slowfast_r50.py', '../../_base_/default_runtime.py'
]

## Modify Head
model = dict(
    cls_head=dict(
        num_classes=4,  # Set to 4 for your task
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[1.0, 1.0, 1.0, 0.172])  # Address class imbalance
    )
)

## Dataset settings
dataset_type = 'VideoDataset'

data_root = 'data/four_classes_8k_train_test/train'
data_root_val = 'data/four_classes_8k_train_test/val'
ann_file_train = 'data/four_classes_8k_train_test/four_classes_8k_train.txt'
ann_file_val = 'data/four_classes_8k_train_test/four_classes_8k_val.txt'
ann_file_test = 'data/four_classes_8k_train_test/four_classes_8k_val.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

## Evaluation Metrics
# val_evaluator = dict(
#     type='MetricEvaluator',
#     metrics=[
#         dict(type='Accuracy'),
#         dict(type='ConfusionMatrix', num_classes=4),  # Save confusion matrix
#         dict(type='F1Score', average='macro')  # F1 score for imbalanced datasets
#     ]
# )
# test_evaluator = val_evaluator

val_evaluator = dict(
    metrics=[
        dict(type='AccMetric'),  # Accuracy metric
        dict(type='ConfusionMatrix', num_classes=4),  # Confusion matrix for 4 classes
        # dict(type='PrecisionRecallF1Metric', average='macro')  # Macro precision/recall/F1
    ]
)

test_evaluator = val_evaluator



## Modify Training Schedule
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=2  # Fewer epochs for fine-tuning
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

## Optimizer with Lower Learning Rate for Fine-Tuning
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=1e-4),  # Smaller LR for fine-tuning
    clip_grad=dict(max_norm=40, norm_type=2)
)

## Parameter Scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,  # Gradual warm-up
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=20,  # Match with the number of epochs
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=20
    )
]

## Hooks
default_hooks = dict(
    checkpoint=dict(interval=4, max_keep_ckpts=3),  # Save checkpoints periodically
    logger=dict(interval=100),
    visualization=dict(type='VisualizationHook', draw_confusion_matrix=True)  # Draw confusion matrix
)
