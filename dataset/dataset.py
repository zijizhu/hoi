import os
import json
import torch
import random
import argparse
import numpy as np
import pickle as pkl
from PIL import Image
from tqdm import tqdm
from torchvision import ops
from torch.utils.data import Dataset, DataLoader

from detr.models import build_model
from detr.datasets import transforms as T

from meter import DetectionAPMeter
from association import BoxAssociation
from relocate import relocate_to_device


class HicoDetDataset(Dataset):
    def __init__(self, dataset_dir, split, transforms=None, nms_threshold=0.7) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transforms = transforms
        self.nms_threshold = nms_threshold
        if self.split == 'train':
            ann_filename = 'trainval_hico.json'
        else:
            ann_filename = 'test_hico.json'
        with open(os.path.join(self.dataset_dir, ann_filename), 'r') as fp:
            self.annotation = json.load(fp=fp)
        with open('id2coco_classes.json', 'r') as fp:
            id2coco_classes = json.load(fp=fp)
        # Map class classes to contiguous indices [1, 90] to [0, 79]
        label_indices = sorted([int(i) for i in id2coco_classes.keys()])
        self.label_map = {i: j for i, j in zip(label_indices, range(80))}

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann = self.annotation[idx]
        img_fn = ann['file_name']
        box_list = [box_class['bbox'] for box_class in ann['annotations']]
        label_list = [box_class['category_id'] for box_class in ann['annotations']]
        # Map label ids to contiguous integers (coco dataset has some None classes)
        label_list_mapped = [self.label_map[idx] for idx in label_list]
        image = Image.open(os.path.join(self.dataset_dir, 'images', f'{self.split}2015', img_fn)).convert('RGB')

        boxes = torch.tensor(box_list).float()
        labels = torch.tensor(label_list_mapped).int()
        # The annotation files has duplicated boxes that need to be removed via nms
        nms_keep = ops.batched_nms(boxes=boxes,
                                   scores=torch.ones(len(boxes)),
                                   idxs=labels,
                                   iou_threshold=self.nms_threshold)
        
        boxes, labels = boxes[nms_keep], labels[nms_keep]

        target = dict(boxes=boxes, labels=labels)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return img_fn, image, target