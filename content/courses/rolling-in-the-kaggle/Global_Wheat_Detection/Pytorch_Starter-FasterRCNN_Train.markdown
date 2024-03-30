---
date: "2024-03-30"
title: Pytorch_Starter-FasterRCNN_Train
author: JDW
type: book
weight: 20
output: md_document
---





<center>

 **Original Notebook** : <https://www.kaggle.com/code/pestipeti/pytorch-starter-fasterrcnn-train>

</center>


&nbsp; In this notebookm I enabled the GPU and the Internet access (needed for the pre-trained weights). We can not use Internet during inference, so I'll create another notebook for commiting. Stay tuned!

&nbsp; You can find the [inference notebook here](https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-inference)

 - FasterRCNN from torchvision
 - Use Resnet50 backbone
 - Albumentation enabled (simple filp for now)



```python
import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt


DIR_INPUT = os.path.join(os.getcwd(), "data")
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST  = f'{DIR_INPUT}/test'
```


```python
train_df = pd.read_csv(f"{DIR_INPUT}/train.csv")
train_df.shape
#> (147793, 5)
```


```python
train_df.info
#> <bound method DataFrame.info of          image_id  width  height                         bbox     source
#> 0       b6ab77fd7   1024    1024   [834.0, 222.0, 56.0, 36.0]    usask_1
#> 1       b6ab77fd7   1024    1024  [226.0, 548.0, 130.0, 58.0]    usask_1
#> 2       b6ab77fd7   1024    1024  [377.0, 504.0, 74.0, 160.0]    usask_1
#> 3       b6ab77fd7   1024    1024  [834.0, 95.0, 109.0, 107.0]    usask_1
#> 4       b6ab77fd7   1024    1024  [26.0, 144.0, 124.0, 117.0]    usask_1
#> ...           ...    ...     ...                          ...        ...
#> 147788  5e0747034   1024    1024    [64.0, 619.0, 84.0, 95.0]  arvalis_2
#> 147789  5e0747034   1024    1024  [292.0, 549.0, 107.0, 82.0]  arvalis_2
#> 147790  5e0747034   1024    1024  [134.0, 228.0, 141.0, 71.0]  arvalis_2
#> 147791  5e0747034   1024    1024   [430.0, 13.0, 184.0, 79.0]  arvalis_2
#> 147792  5e0747034   1024    1024   [875.0, 740.0, 94.0, 61.0]  arvalis_2
#>
#> [147793 rows x 5 columns]>
```


```python
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float64)
train_df['y'] = train_df['y'].astype(np.float64)
train_df['w'] = train_df['w'].astype(np.float64)
train_df['h'] = train_df['h'].astype(np.float64)
```


```python
image_ids = train_df['image_id'].unique()
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]
```


```python
valid_df = train_df[train_df['image_id'].isin(valid_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]
```


```python
valid_df.shape, train_df.shape
#> ((25006, 8), (122787, 8))
```



```python
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms = None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f"{self.image_dir}/{image_id}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype = torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0], ), dtype = torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0], ), dtype = torch.int64)

        target = {}
        target['boxes']    = boxes
        target['labels']   = labels
        # target['masks']  = None
        target['image_id'] = torch.tensor([index])
        target['area']     = area
        target['iscrowd']  = iscrowd

        if self.transforms:
            sample = {
                'image' : image,
                'bboxes' : target['boxes'],
                'labels' : labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

```



```python
# Albumentations
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p = 1.0)
    ], bbox_params = {'format' : 'pascal_voc', 'label_fields' : ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p = 1.0)
    ], bbox_params = {'format' : 'pascal_voc', 'label_fields' : ['labels']})
```


# Create the model


```python
# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
#> /home/rstudio/.local/share/r-miniconda/envs/r-reticulate/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#>   warnings.warn(
#> /home/rstudio/.local/share/r-miniconda/envs/r-reticulate/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1`. You can also use `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT` to get the most up-to-date weights.
#>   warnings.warn(msg)
```


```python
num_classes = 2 # 1 class(wheat) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```


```python
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
```


```python
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())

# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size = 16,
    shuffle = False,
    num_workers = 4,
    collate_fn = collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 8,
    shuffle = False,
    num_workers = 4,
    collate_fn = collate_fn
)
```


```python
device = torch.device('cuda')
```


# Sample


```python
images, targets, image_ids = next(iter(train_data_loader))
images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
```


```python
boxes = targets[2]['boxes'].cpu().numpy().astype(np.int32)
sample = images[2].permute(1, 2, 0).cpu().numpy()
```


```python
fig, ax = plt.subplots(1, 1, figsize = (16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0),
                  3)

ax.set_axis_off()
ax.imshow(sample)
plt.show()
```

<img src="/courses/rolling-in-the-kaggle/Global_Wheat_Detection/Pytorch_Starter-FasterRCNN_Train_files/figure-html/unnamed-chunk-19-1.png" width="1536" />


# Train


```python
model.to(device)
#> FasterRCNN(
#>   (transform): GeneralizedRCNNTransform(
#>       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#>       Resize(min_size=(800,), max_size=1333, mode='bilinear')
#>   )
#>   (backbone): BackboneWithFPN(
#>     (body): IntermediateLayerGetter(
#>       (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#>       (bn1): FrozenBatchNorm2d(64, eps=0.0)
#>       (relu): ReLU(inplace=True)
#>       (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#>       (layer1): Sequential(
#>         (0): Bottleneck(
#>           (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(256, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>           (downsample): Sequential(
#>             (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>             (1): FrozenBatchNorm2d(256, eps=0.0)
#>           )
#>         )
#>         (1): Bottleneck(
#>           (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(256, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (2): Bottleneck(
#>           (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(256, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>       )
#>       (layer2): Sequential(
#>         (0): Bottleneck(
#>           (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>           (downsample): Sequential(
#>             (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#>             (1): FrozenBatchNorm2d(512, eps=0.0)
#>           )
#>         )
#>         (1): Bottleneck(
#>           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (2): Bottleneck(
#>           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (3): Bottleneck(
#>           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>       )
#>       (layer3): Sequential(
#>         (0): Bottleneck(
#>           (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>           (downsample): Sequential(
#>             (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#>             (1): FrozenBatchNorm2d(1024, eps=0.0)
#>           )
#>         )
#>         (1): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (2): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (3): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (4): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (5): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>       )
#>       (layer4): Sequential(
#>         (0): Bottleneck(
#>           (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(2048, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>           (downsample): Sequential(
#>             (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#>             (1): FrozenBatchNorm2d(2048, eps=0.0)
#>           )
#>         )
#>         (1): Bottleneck(
#>           (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(2048, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (2): Bottleneck(
#>           (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(2048, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>       )
#>     )
#>     (fpn): FeaturePyramidNetwork(
#>       (inner_blocks): ModuleList(
#>         (0): Conv2dNormActivation(
#>           (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#>         )
#>         (1): Conv2dNormActivation(
#>           (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
#>         )
#>         (2): Conv2dNormActivation(
#>           (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
#>         )
#>         (3): Conv2dNormActivation(
#>           (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
#>         )
#>       )
#>       (layer_blocks): ModuleList(
#>         (0-3): 4 x Conv2dNormActivation(
#>           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#>         )
#>       )
#>       (extra_blocks): LastLevelMaxPool()
#>     )
#>   )
#>   (rpn): RegionProposalNetwork(
#>     (anchor_generator): AnchorGenerator()
#>     (head): RPNHead(
#>       (conv): Sequential(
#>         (0): Conv2dNormActivation(
#>           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#>           (1): ReLU(inplace=True)
#>         )
#>       )
#>       (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
#>       (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
#>     )
#>   )
#>   (roi_heads): RoIHeads(
#>     (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
#>     (box_head): TwoMLPHead(
#>       (fc6): Linear(in_features=12544, out_features=1024, bias=True)
#>       (fc7): Linear(in_features=1024, out_features=1024, bias=True)
#>     )
#>     (box_predictor): FastRCNNPredictor(
#>       (cls_score): Linear(in_features=1024, out_features=2, bias=True)
#>       (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
#>     )
#>   )
#> )
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)
# lr_scheduler = None

num_epochs = 2
```


```python
loss_hist = Averager()
itr = 1

for epoch in range(num_epochs):
    loss_hist.reset()

    for images, targets, image_ids in train_data_loader:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value:.4f}")

        itr += 1

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value:.4f}")
#> Iteration #50 loss: 1.0978
#> Iteration #100 loss: 0.8494
#> Iteration #150 loss: 0.8136
#> Epoch #0 loss: 1.0524
#> Iteration #200 loss: 0.9033
#> Iteration #250 loss: 0.8955
#> Iteration #300 loss: 0.7119
#> Epoch #1 loss: 0.8972
```



```python
images, targets, image_ids = next(iter(valid_data_loader))

images = list(img.to(device) for img in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
sample = images[1].permute(1,2,0).cpu().numpy()
```


```python
model.eval()
#> FasterRCNN(
#>   (transform): GeneralizedRCNNTransform(
#>       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#>       Resize(min_size=(800,), max_size=1333, mode='bilinear')
#>   )
#>   (backbone): BackboneWithFPN(
#>     (body): IntermediateLayerGetter(
#>       (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#>       (bn1): FrozenBatchNorm2d(64, eps=0.0)
#>       (relu): ReLU(inplace=True)
#>       (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#>       (layer1): Sequential(
#>         (0): Bottleneck(
#>           (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(256, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>           (downsample): Sequential(
#>             (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>             (1): FrozenBatchNorm2d(256, eps=0.0)
#>           )
#>         )
#>         (1): Bottleneck(
#>           (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(256, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (2): Bottleneck(
#>           (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(64, eps=0.0)
#>           (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(256, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>       )
#>       (layer2): Sequential(
#>         (0): Bottleneck(
#>           (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>           (downsample): Sequential(
#>             (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#>             (1): FrozenBatchNorm2d(512, eps=0.0)
#>           )
#>         )
#>         (1): Bottleneck(
#>           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (2): Bottleneck(
#>           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (3): Bottleneck(
#>           (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(128, eps=0.0)
#>           (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(512, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>       )
#>       (layer3): Sequential(
#>         (0): Bottleneck(
#>           (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>           (downsample): Sequential(
#>             (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#>             (1): FrozenBatchNorm2d(1024, eps=0.0)
#>           )
#>         )
#>         (1): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (2): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (3): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (4): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (5): Bottleneck(
#>           (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(256, eps=0.0)
#>           (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(1024, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>       )
#>       (layer4): Sequential(
#>         (0): Bottleneck(
#>           (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(2048, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>           (downsample): Sequential(
#>             (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#>             (1): FrozenBatchNorm2d(2048, eps=0.0)
#>           )
#>         )
#>         (1): Bottleneck(
#>           (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(2048, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>         (2): Bottleneck(
#>           (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn1): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#>           (bn2): FrozenBatchNorm2d(512, eps=0.0)
#>           (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#>           (bn3): FrozenBatchNorm2d(2048, eps=0.0)
#>           (relu): ReLU(inplace=True)
#>         )
#>       )
#>     )
#>     (fpn): FeaturePyramidNetwork(
#>       (inner_blocks): ModuleList(
#>         (0): Conv2dNormActivation(
#>           (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
#>         )
#>         (1): Conv2dNormActivation(
#>           (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
#>         )
#>         (2): Conv2dNormActivation(
#>           (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
#>         )
#>         (3): Conv2dNormActivation(
#>           (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
#>         )
#>       )
#>       (layer_blocks): ModuleList(
#>         (0-3): 4 x Conv2dNormActivation(
#>           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#>         )
#>       )
#>       (extra_blocks): LastLevelMaxPool()
#>     )
#>   )
#>   (rpn): RegionProposalNetwork(
#>     (anchor_generator): AnchorGenerator()
#>     (head): RPNHead(
#>       (conv): Sequential(
#>         (0): Conv2dNormActivation(
#>           (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#>           (1): ReLU(inplace=True)
#>         )
#>       )
#>       (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
#>       (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
#>     )
#>   )
#>   (roi_heads): RoIHeads(
#>     (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
#>     (box_head): TwoMLPHead(
#>       (fc6): Linear(in_features=12544, out_features=1024, bias=True)
#>       (fc7): Linear(in_features=1024, out_features=1024, bias=True)
#>     )
#>     (box_predictor): FastRCNNPredictor(
#>       (cls_score): Linear(in_features=1024, out_features=2, bias=True)
#>       (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
#>     )
#>   )
#> )
cpu_device = torch.device("cpu")

outputs = model(images)
outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

fig, ax = plt.subplots(1, 1, figsize = (16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0),
                  3)

ax.set_axis_off()
ax.imshow(sample)
plt.show()
```

<img src="/courses/rolling-in-the-kaggle/Global_Wheat_Detection/Pytorch_Starter-FasterRCNN_Train_files/figure-html/unnamed-chunk-23-3.png" width="1536" />


```python
# torch.save(model.state_dict(), "./models/fasterrcnn_resnet50_fpn.pth")
```





















































