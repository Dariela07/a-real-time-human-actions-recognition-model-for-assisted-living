_base_ = ['slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py']

_base_ = ['mmaction2/configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_transfer_2_NorVsCrit_epoch50.py']

model = dict(
    backbone=dict(slow_pathway=dict(depth=101), fast_pathway=dict(depth=101)))





