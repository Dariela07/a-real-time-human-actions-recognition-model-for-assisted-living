from mmaction.apis import inference_recognizer, init_recognizer
import pandas as pd
import logging
# logging.getLogger().setLevel(logging.ERROR)


from mmengine.logging import MMLogger
MMLogger.get_instance("mmengine").setLevel(logging.ERROR)



# I3D
config_path = 'configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_transfer_epoch60.py'

checkpoint_path = 'work_dirs/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_transfer_epoch60/best_acc_top1_epoch_56.pth'

img_path = 'data/normal_vs_3critical/val/S001C001P001R001A009.mp4'

model = init_recognizer(config_path, checkpoint_path, device="cuda:0")   
result = inference_recognizer(model, img_path)

print(result)
pred_label = result.pred_label.item()

classes = ['Falling', 'Staggering', 'Chest Pain', "Normal"]

print("Predicted Class:", classes[pred_label])
