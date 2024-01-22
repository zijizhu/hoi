import os
import json
import torch
from PIL import Image
from torchvision import ops
from torch.utils.data import Dataset


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


def collate_fn(batch):
    batch_image_paths = []; batch_images = []; batch_targets = []
    for path, img, target in batch:
        batch_image_paths.append(path)
        batch_images.append(img)
        batch_targets.append(target)
    return batch_image_paths, batch_images, batch_targets
