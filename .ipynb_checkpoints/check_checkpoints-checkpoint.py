# import torch
# checkpoint = torch.load('work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics700-rgb_transfer/epoch_38.pth')
# print(checkpoint['meta']['epoch'])

# path = 'work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics700-rgb_transfer/epoch_38.pth'
path = 'work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics700-rgb_transfer/first run checkpoints/best_acc_top1_epoch_38.pth'
import torch

# Load the checkpoint file
checkpoint = torch.load(path)

# Print the saved epoch
print(f"Checkpoint saved at epoch: {checkpoint['meta']['epoch']}")

# Ensure optimizer and scheduler states are present
if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
    print("Optimizer and scheduler states are correctly saved.")
else:
    print("Optimizer or scheduler states are missing!")
