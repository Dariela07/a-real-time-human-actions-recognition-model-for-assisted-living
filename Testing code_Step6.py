from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)


# Step1: Build a Pipeline
import mmcv
import decord
import numpy as np
from mmcv.transforms import TRANSFORMS, BaseTransform, to_tensor
from mmaction.structures import ActionDataSample


@TRANSFORMS.register_module()
class VideoInit(BaseTransform):
    def transform(self, results):
        container = decord.VideoReader(results['filename'])
        results['total_frames'] = len(container)
        results['video_reader'] = container
        return results


@TRANSFORMS.register_module()
class VideoSample(BaseTransform):
    def __init__(self, clip_len, num_clips, test_mode=False):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def transform(self, results):
        total_frames = results['total_frames']
        interval = total_frames // self.clip_len

        if self.test_mode:
            # Make the sampling during testing deterministic
            np.random.seed(42)

        inds_of_all_clips = []
        for i in range(self.num_clips):
            bids = np.arange(self.clip_len) * interval
            offset = np.random.randint(interval, size=bids.shape)
            inds = bids + offset
            inds_of_all_clips.append(inds)

        results['frame_inds'] = np.concatenate(inds_of_all_clips)
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips
        return results


@TRANSFORMS.register_module()
class VideoDecode(BaseTransform):
    def transform(self, results):
        frame_inds = results['frame_inds']
        container = results['video_reader']

        imgs = container.get_batch(frame_inds).asnumpy()
        imgs = list(imgs)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoResize(BaseTransform):
    def __init__(self, r_size):
        self.r_size = (np.inf, r_size)

    def transform(self, results):
        img_h, img_w = results['img_shape']
        new_w, new_h = mmcv.rescale_size((img_w, img_h), self.r_size)

        imgs = [mmcv.imresize(img, (new_w, new_h))
                for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoCrop(BaseTransform):
    def __init__(self, c_size):
        self.c_size = c_size

    def transform(self, results):
        img_h, img_w = results['img_shape']
        center_x, center_y = img_w // 2, img_h // 2
        x1, x2 = center_x - self.c_size // 2, center_x + self.c_size // 2
        y1, y2 = center_y - self.c_size // 2, center_y + self.c_size // 2
        imgs = [img[y1:y2, x1:x2] for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoFormat(BaseTransform):
    def transform(self, results):
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        imgs = results['imgs']

        # [num_clips*clip_len, H, W, C]
        imgs = np.array(imgs)
        # [num_clips, clip_len, H, W, C]
        imgs = imgs.reshape((num_clips, clip_len) + imgs.shape[1:])
        # [num_clips, C, clip_len, H, W]
        imgs = imgs.transpose(0, 4, 1, 2, 3)

        results['imgs'] = imgs
        return results


@TRANSFORMS.register_module()
class VideoPack(BaseTransform):
    def __init__(self, meta_keys=('img_shape', 'num_clips', 'clip_len')):
        self.meta_keys = meta_keys

    def transform(self, results):
        packed_results = dict()
        inputs = to_tensor(results['imgs'])
        data_sample = ActionDataSample()
        data_sample.set_gt_label(results['label'])
        metainfo = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(metainfo)
        packed_results['inputs'] = inputs
        packed_results['data_samples'] = data_sample
        return packed_results


# Step2: Build a Dataset and DataLoader
import os.path as osp
from mmengine.fileio import list_from_file
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS


@DATASETS.register_module()
class DatasetZelda(BaseDataset):
    def __init__(self, ann_file, pipeline, data_root, data_prefix=dict(video=''),
                 test_mode=False, modality='RGB', **kwargs):
        self.modality = modality
        super(DatasetZelda, self).__init__(ann_file=ann_file, pipeline=pipeline, data_root=data_root,
                                           data_prefix=data_prefix, test_mode=test_mode,
                                           **kwargs)

    def load_data_list(self):
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            filename, label = line_split
            label = int(label)
            filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label))
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        return data_info


from mmaction.registry import DATASETS

train_pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

val_pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=5, test_mode=True),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

train_dataset_cfg = dict(
    type='DatasetZelda',
    ann_file='kinetics_tiny_train_video.txt',
    pipeline=train_pipeline_cfg,
    data_root='data/kinetics400_tiny/',
    data_prefix=dict(video='train'))

val_dataset_cfg = dict(
    type='DatasetZelda',
    ann_file='kinetics_tiny_val_video.txt',
    pipeline=val_pipeline_cfg,
    data_root='data/kinetics400_tiny/',
    data_prefix=dict(video='val'))

train_dataset = DATASETS.build(train_dataset_cfg)

packed_results = train_dataset[0]

inputs = packed_results['inputs']
data_sample = packed_results['data_samples']

print('shape of the inputs: ', inputs.shape)

# Get metainfo of the inputs
print('image_shape: ', data_sample.img_shape)
print('num_clips: ', data_sample.num_clips)
print('clip_len: ', data_sample.clip_len)

# Get label of the inputs
print('label: ', data_sample.gt_label)

from mmengine.runner import Runner

BATCH_SIZE = 2

train_dataloader_cfg = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset_cfg)

val_dataloader_cfg = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset_cfg)

train_data_loader = Runner.build_dataloader(dataloader=train_dataloader_cfg)
val_data_loader = Runner.build_dataloader(dataloader=val_dataloader_cfg)

batched_packed_results = next(iter(train_data_loader))

batched_inputs = batched_packed_results['inputs']
batched_data_sample = batched_packed_results['data_samples']

assert len(batched_inputs) == BATCH_SIZE
assert len(batched_data_sample) == BATCH_SIZE


# Step3: Build a Recognizer

import torch
from mmengine.model import BaseDataPreprocessor, stack_batch
from mmaction.registry import MODELS


@MODELS.register_module()
class DataPreprocessorZelda(BaseDataPreprocessor):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer(
            'mean',
            torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1),
            False)
        self.register_buffer(
            'std',
            torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1),
            False)

    def forward(self, data, training=False):
        data = self.cast_data(data)
        inputs = data['inputs']
        batch_inputs = stack_batch(inputs)  # Batching
        batch_inputs = (batch_inputs - self.mean) / self.std  # Normalization
        data['inputs'] = batch_inputs
        return data



# Here is the usage of data_preprocessor: feed the batched_packed_results obtained from the Step2: Build a Dataset and DataLoader into the data_preprocessor for batching and normalization.
from mmaction.registry import MODELS

data_preprocessor_cfg = dict(
    type='DataPreprocessorZelda',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375])

data_preprocessor = MODELS.build(data_preprocessor_cfg)

preprocessed_inputs = data_preprocessor(batched_packed_results)
print(preprocessed_inputs['inputs'].shape)



# The implementations of backbone, cls_head and recognizer are as follows:
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule, Sequential
from mmengine.structures import LabelData
from mmaction.registry import MODELS


@MODELS.register_module()
class BackBoneZelda(BaseModule):
    def __init__(self, init_cfg=None):
        if init_cfg is None:
            init_cfg = [dict(type='Kaiming', layer='Conv3d', mode='fan_out', nonlinearity="relu"),
                        dict(type='Constant', layer='BatchNorm3d', val=1, bias=0)]

        super(BackBoneZelda, self).__init__(init_cfg=init_cfg)

        self.conv1 = Sequential(nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                                          stride=(1, 2, 2), padding=(1, 3, 3)),
                                nn.BatchNorm3d(64), nn.ReLU())
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                    padding=(0, 1, 1))

        self.conv = Sequential(nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
                               nn.BatchNorm3d(128), nn.ReLU())

    def forward(self, imgs):
        # imgs: [batch_size*num_views, 3, T, H, W]
        # features: [batch_size*num_views, 128, T/2, H//8, W//8]
        features = self.conv(self.maxpool(self.conv1(imgs)))
        return features


@MODELS.register_module()
class ClsHeadZelda(BaseModule):
    def __init__(self, num_classes, in_channels, dropout=0.5, average_clips='prob', init_cfg=None):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(ClsHeadZelda, self).__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.average_clips = average_clips

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        N, C, T, H, W = x.shape
        x = self.pool(x)
        x = x.view(N, C)
        assert x.shape[1] == self.in_channels

        if self.dropout is not None:
            x = self.dropout(x)

        cls_scores = self.fc(x)
        return cls_scores

    def loss(self, feats, data_samples):
        cls_scores = self(feats)
        labels = torch.stack([x.gt_label for x in data_samples])
        labels = labels.squeeze()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        loss_cls = self.loss_fn(cls_scores, labels)
        return dict(loss_cls=loss_cls)

    def predict(self, feats, data_samples):
        cls_scores = self(feats)
        num_views = cls_scores.shape[0] // len(data_samples)
        # assert num_views == data_samples[0].num_clips
        cls_scores = self.average_clip(cls_scores, num_views)

        for ds, sc in zip(data_samples, cls_scores):
            pred = LabelData(item=sc)         
            # pred = LabelData(item=torch.tensor(sc))
            ds.pred_scores = pred
            print(ds.pred_scores)
        return data_samples
    
        # for ds, sc in zip(data_samples, cls_scores):
        #     pred = LabelData(item=sc)
        #     # pred = LabelData(item=torch.tensor(sc))
        #     ds.pred_scores = torch.tensor(pred)
        return data_samples


    def average_clip(self, cls_scores, num_views):
          if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

          total_views = cls_scores.shape[0]
          cls_scores = cls_scores.view(total_views // num_views, num_views, -1)

          if self.average_clips is None:
              return cls_scores
          elif self.average_clips == 'prob':
              cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
          elif self.average_clips == 'score':
              cls_scores = cls_scores.mean(dim=1)

          return cls_scores


@MODELS.register_module()
class RecognizerZelda(BaseModel):
    def __init__(self, backbone, cls_head, data_preprocessor):
        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)

    def extract_feat(self, inputs):
        inputs = inputs.view((-1, ) + inputs.shape[2:])
        return self.backbone(inputs)

    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        loss = self.cls_head.loss(feats, data_samples)
        return loss

    def predict(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        predictions = self.cls_head.predict(feats, data_samples)
        return predictions

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'tensor':
            return self.extract_feat(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode: {mode}')


# The init_cfg is used for model weight initialization. For more information on model weight initialization, please refer to MMEngine Tutorial. The usage of the above modules is as follows:

import torch
import copy
from mmaction.registry import MODELS

model_cfg = dict(
    type='RecognizerZelda',
    backbone=dict(type='BackBoneZelda'),
    cls_head=dict(
        type='ClsHeadZelda',
        num_classes=2,
        in_channels=128,
        average_clips='prob'),
    data_preprocessor = dict(
        type='DataPreprocessorZelda',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

model = MODELS.build(model_cfg)

# Train
model.train()
model.init_weights()
data_batch_train = copy.deepcopy(batched_packed_results)
data = model.data_preprocessor(data_batch_train, training=True)
loss = model(**data, mode='loss')
print('loss dict: ', loss)

# Test
with torch.no_grad():
    model.eval()
    data_batch_test = copy.deepcopy(batched_packed_results)
    data = model.data_preprocessor(data_batch_test, training=False)
    predictions = model(**data, mode='predict')
print('Label of Sample[0]', predictions[0].gt_label)
print('Scores of Sample[0]', predictions[0].pred_scores)


# Step4: Build a Evaluation Metric
import copy
from collections import OrderedDict
from mmengine.evaluator import BaseMetric
from mmaction.evaluation import top_k_accuracy
from mmaction.registry import METRICS


@METRICS.register_module()
class AccuracyMetric(BaseMetric):
    def __init__(self, topk=(1, 5), collect_device='cpu', prefix='acc'):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.topk = topk

    def process(self, data_batch, data_samples):
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            # scores = data_sample['pred_score'].cpu().numpy()
            scores = data_sample['pred_scores']['item'].cpu().numpy()
            # print(data_sample)
            label = data_sample['gt_label'].item()
            result['scores'] = scores
            result['label'] = label
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        eval_results = OrderedDict()
        labels = [res['label'] for res in results]
        scores = [res['scores'] for res in results]
        topk_acc = top_k_accuracy(scores, labels, self.topk)
        for k, acc in zip(self.topk, topk_acc):
            eval_results[f'topk{k}'] = acc
        return eval_results


from mmaction.registry import METRICS

metric_cfg = dict(type='AccuracyMetric', topk=(1, 5))

metric = METRICS.build(metric_cfg)

data_samples = [d.to_dict() for d in predictions]

# print("Batched_packed_results:", batched_packed_results)

# print("Data Samples:", data_samples)

metric.process(batched_packed_results, data_samples)
acc = metric.compute_metrics(metric.results)
print(acc)


# # Step 5: Train and Test with Native PyTorch
# import torch.optim as optim
# from mmengine import track_iter_progress


# device = 'cuda' # or 'cpu'
# max_epochs = 10

# optimizer = optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(max_epochs):
#     model.train()
#     losses = []
#     for data_batch in track_iter_progress(train_data_loader):
#         data = model.data_preprocessor(data_batch, training=True)
#         loss_dict = model(**data, mode='loss')
#         loss = loss_dict['loss_cls']

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         losses.append(loss.item())

#     print(f'Epoch[{epoch}]: loss ', sum(losses) / len(train_data_loader))

#     with torch.no_grad():
#         model.eval()
#         for data_batch in track_iter_progress(val_data_loader):
#             data = model.data_preprocessor(data_batch, training=False)
#             predictions = model(**data, mode='predict')
#             data_samples = [d.to_dict() for d in predictions]
#             metric.process(data_batch, data_samples)

#         acc = metric.acc = metric.compute_metrics(metric.results)
#         for name, topk in acc.items():
#             print(f'{name}: ', topk)


# Step6: Train and Test with MMEngine (Recommende

from mmengine.runner import Runner

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.01))

runner = Runner(model=model_cfg, work_dir='./work_dirs/guide',
                train_dataloader=train_dataloader_cfg,
                train_cfg=train_cfg,
                val_dataloader=val_dataloader_cfg,
                val_cfg=val_cfg,
                optim_wrapper=optim_wrapper,
                val_evaluator=[metric_cfg],
                default_scope='mmaction')
runner.train()
