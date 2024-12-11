from mmaction.apis import inference_recognizer, init_recognizer
import pandas as pd
import logging
# logging.getLogger().setLevel(logging.ERROR)


from mmengine.logging import MMLogger
MMLogger.get_instance("mmengine").setLevel(logging.ERROR)



## I3D
# config_path = 'configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_transfer_epoch60.py'

# checkpoint_path = 'work_dirs/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_transfer_epoch60/best_acc_top1_epoch_56.pth'

# img_path = 'data/normal_vs_3critical/val/S001C001P001R001A009.mp4'


## UniFormerV2
# config_path = 'configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics700-rgb_transfer.py'
# checkpoint_path = 'work_dirs/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics700-rgb_transfer/resume_50_checkpoints/best_acc_top1_epoch_10.pth'

## TimeSformer
# config_path = 'configs/recognition/timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb-transfer_jointST.py'
# checkpoint_path = 'work_dirs/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb-transfer_jointST/second run checkpoints 50 epochs/epoch_50.pth'

# SlowFast
config_path = 'configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_transfer_2_NorVsCrit_epoch50.py'
checkpoint_path = 'work_dirs/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_transfer_2_NorVsCrit_epoch50/best_acc_top1_epoch_50.pth' 

test_path = 'data/normal_vs_3critical_test.csv'
val_path = 'data/normal_vs_3critical_val.csv'

df_test = pd.read_csv(test_path)
# df_val = pd.read_csv(val_path)

ls_test = []
# i = 0
for _, row in df_test.iterrows():
# for _, row in df_val.iterrows():
    ls_row = list(row)
    img_path = row.values[0]    
    model = init_recognizer(config_path, checkpoint_path, device="cuda:0")   
    result = inference_recognizer(model, img_path)
    pred_label = result.pred_label.item()
    ls_row.append(pred_label)
    # i+=1
    # print(i)
    ls_test.append(ls_row)

df_with_prediction = pd.DataFrame(ls_test, columns = ['File_path', 'Label', 'Prediction'])

print(df_with_prediction.head())

# new_path = "Inference_comparison_i3d.csv"
# new_path = "i3d_val_comparison.csv"
# new_path = "UniformerV2_test_comparison.csv"
new_path = "SlowFast_test_comparison.csv"

df_with_prediction.to_csv(new_path)

# print(result)
# pred_label = result.pred_label.item()
# print("Predicted Label:", pred_label)
