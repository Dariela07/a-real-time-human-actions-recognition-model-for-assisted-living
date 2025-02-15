_base_ = ['slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb2.py']


model = dict(
    backbone=dict(slow_pathway=dict(depth=101), fast_pathway=dict(depth=101)))



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
        T_max=60,  # Match with the number of epochs
        eta_min=0,
        by_epoch=True,
        begin=6,   #begin for CosineAnnealingLR to match the end of the LinearLR scheduler. end for CosineAnnealingLR to the total number of epochs
        end=60
    )
]

## Hooks
default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=3),  # Save checkpoints periodically
    logger=dict(interval=100),
    visualization=dict(type='VisualizationHook', draw_confusion_matrix=True)  # Draw confusion matrix
)

