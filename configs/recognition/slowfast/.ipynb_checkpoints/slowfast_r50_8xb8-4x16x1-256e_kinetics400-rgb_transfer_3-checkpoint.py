### 
'''
(“This transfer learning version (3) improves based on the experience of the previous transfer learning version (2). The previous version had significant fluctuations, recall and precision were likely affected by imbalanced datasets and slow training speed.

- Update the class_weight parameter to better emphasize underrepresented classes (modify model, set "class_weight=[5.0, 5.0, 5.0, 1.0]")

- Make changes to reduce training fluctuations: Slow down the warm-up phase to reduce fluctuations and ensure a smoother start (modify param_scheduler). Increase the batch size to reduce gradient noise and fluctuations, considering GPU memory constraints (requires large computational power). Modify "train_dataloader". 

- Make changes to improve Recall: Ensure each batch contains a proportional number of samples from all classes. Modify "train_dataloader", and "val/test_dataloader". Augment Critical Classes: Add augmentations for minority classes to increase data diversity (need more experimenting).

- Make changes to Improve Training Speed: Reduce validation frequency to speed up training: modify train_cfg; Set train_dataloader to add prefetch_factor=2 for faster loading

'''



_base_ = [
    '../../_base_/models/slowfast_r50.py', '../../_base_/default_runtime.py'
]

## Modify Head
model = dict(
    cls_head=dict(
        num_classes=4,  # Set to 4 for your task
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[4.0, 4.0, 4.0, 1.0])  # Address class imbalance  948 948 948 3200, emphasize small samples
    )
)

## Dataset settings
dataset_type = 'VideoDataset'

data_root = 'data/normal_vs_3critical/train'
data_root_val = 'data/normal_vs_3critical/val'
data_root_test = 'data/normal_vs_3critical/test'
ann_file_train = 'data/normal_vs_3critical/normal_vs_3critical_train.txt'
ann_file_val = 'data/normal_vs_3critical/normal_vs_3critical_val.txt'
ann_file_test = 'data/normal_vs_3critical/normal_vs_3critical_test.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    
    # dict(type='RandomResizedCrop'), #G
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Resize', scale=(128, 128), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),

    # Version 3 new components
    # dict(type='ColorJitter', brightness=0.2, contrast=0.2),  # Add jittering for diversity

    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
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
        clip_len=16,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]


##3 based on instruction
train_dataloader = dict(
    batch_size=32,  #2 batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),  #2 type='DefaultSampler'
    
    prefetch_factor=2,  #3 Prefetch batches for faster loading
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True), #2 sampler=dict(type='DefaultSampler', shuffle=False),
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
    sampler=dict(type='DefaultSampler', shuffle=True),  # sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_test),
        pipeline=test_pipeline,
        test_mode=True))

## Evaluation Metrics

val_evaluator = dict(
    metrics=[
        dict(type='AccMetric'),  # Accuracy metric
        dict(type='ConfusionMatrix', num_classes=4),  # Confusion matrix for 4 classes
        # dict(type='PrecisionRecallF1Metric', average='macro')  # Macro precision/recall/F1
    ]
)

test_evaluator = val_evaluator



## Modify Training Schedule

# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=2  # Fewer epochs for fine-tuning
# )
##3 based on introduction
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

## Parameter Scheduler Version 3
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,  # Slower warm-up #2 start_factor=0.1,
        
        by_epoch=True,
        begin=0,
        end=5,  # Gradual warm-up
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=20,  # Match with the number of epochs
        eta_min=1e-6,  # Lower minimum learning rate for stability #2 eta_min=0   
        by_epoch=True,
        begin=5,  #2 begin=0,
        end=20
    )
]

## Hooks
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=20),  # Save checkpoints periodically
    logger=dict(interval=200),
    visualization=dict(type='VisualizationHook', draw_confusion_matrix=True)  # Draw confusion matrix
)

