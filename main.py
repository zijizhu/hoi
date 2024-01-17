import os
import json
import torch
import random
import numpy as np
import torchvision
import pickle as pkl
from PIL import Image
from tqdm import tqdm
from pprint import pprint
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
        self.nms_threshold = nms_threshold
        self.transforms = transforms
        with open(os.path.join(self.dataset_dir, 'anno_cleaned.json'), 'r') as fp:
            self.annotation = json.load(fp=fp)
        # Remove instance if all hois in instance is invisible
        self.annotation[f'bbox_{self.split}'] = [anno
                           for anno
                           in self.annotation[f'bbox_{self.split}'].copy()
                           if not all(hoi['invis'] == 1 for hoi in anno['hoi'])]
        with open(os.path.join(self.dataset_dir, 'coco_class_indices.json'), 'r') as fp:
            self.label_index_map = json.load(fp=fp)

    def __len__(self):
        return len(self.annotation[f'bbox_{self.split}'])

    def __getitem__(self, idx):
        bbox_anno = self.annotation[f'bbox_{self.split}'][idx]
        img_path = os.path.join(self.dataset_dir, 'images', f'{self.split}2015', bbox_anno['filename'])
        image = Image.open(img_path).convert('RGB')
        # Remove invisible hois
        all_hois = [hoi for hoi in bbox_anno['hoi'] if hoi['invis'] == 0]
        # Collect bounding boxes
        all_bboxes = []
        all_labels = []
        for hoi in all_hois:
            for bbox in hoi['bboxhuman']:
                [x1, x2, y1, y2] = list(bbox.values())
                all_bboxes.append([x1, y1, x2, y2])
                all_labels.append('person')
            for bbox in hoi['bboxobject']:
                [x1, x2, y1, y2] = list(bbox.values())
                all_bboxes.append([x1, y1, x2, y2])
                hoi_idx = hoi['id'] - 1     # Indices start from 1 in the original .mat annotation
                label = self.annotation['list_action'][hoi_idx]['nname']
                all_labels.append(label)

        bbox = torch.tensor(all_bboxes).float()
        all_label_idxs = torch.tensor(
            [self.label_index_map[label] for label in all_labels])

        keep_idxs = torchvision.ops.batched_nms(bbox,
                                                torch.ones(len(bbox)),
                                                all_label_idxs,
                                                self.nms_threshold)
        target = dict(boxes=bbox[keep_idxs], labels=all_label_idxs[keep_idxs])
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return img_path, image, target

def eval(model: torch.nn.Module, dataloader, postprocessors, threshold=0.7, device='cpu'):
    model.eval()
    model.to(device=device)

    associate = BoxAssociation(min_iou=0.5)
    meter = DetectionAPMeter(80, algorithm='INT', nproc=10)
    num_gt = torch.zeros(80)

    model.to(device)

    if dataloader.batch_size != 1:
        raise ValueError(f"The batch size shoud be 1, not {dataloader.batch_size}")
    for image_path, image, target in tqdm(dataloader):
        image = relocate_to_device(image, device=device)
        output = model(image)
        output = relocate_to_device(output, device='cpu')
        
        scores, labels, boxes = postprocessors(output, target[0]['size'].unsqueeze(0))[0].values()
        keep = torch.nonzero(scores >= threshold).squeeze(1)

        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        gt_boxes = target[0]['boxes']
        gt_boxes = ops.box_convert(gt_boxes, 'cxcywh', 'xyxy')
        h, w = target[0]['size']
        scale_fct = torch.stack([w, h, w, h])
        gt_boxes *= scale_fct
        gt_labels = target[0]['labels']

        for c in gt_labels:
            num_gt[c] += 1

        # For each unique class in the predicted labels:
        # Check if the class exists in the ground truth labels.
        # If it exists:
        # Get the indices of GT and prediction bounding boxes of this class.
        # Check if each prediciton bbox has enough iou with some GT bouding box.
        # And create a vector of binary indicator.
        # If it does not:
        # The binary indicator vector will be all zero.

        # Associate detections with ground truth
        binary_labels = torch.zeros(len(labels))
        unique_cls = labels.unique()

        for c in unique_cls:
            det_idx = torch.nonzero(labels == c).squeeze(1)
            gt_idx = torch.nonzero(gt_labels == c).squeeze(1)

            if len(gt_idx) == 0:
                continue

            associate_results = associate(gt_boxes[gt_idx].view(-1, 4),
                                          boxes[det_idx].view(-1, 4),
                                          scores[det_idx].view(-1))
            
            binary_labels[det_idx] = associate_results

        meter.append(scores, labels, binary_labels)

    meter.num_gt = num_gt.tolist()
    return meter.eval(), meter.max_rec


def collate_fn(batch):
    batch_image_paths = []; batch_images = []; batch_targets = []
    for path, img, target in batch:
        batch_image_paths.append(path)
        batch_images.append(img)
        batch_targets.append(target)
    return batch_image_paths, batch_images, batch_targets


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Hardcoded configs
    class Args(object):
        def __init__(self):
            self.split = 'train'
            self.device = 'cpu'
            self.output_dir = 'out'
            self.print_interval = 1000
            self.num_workers = 9
            self.pretrained_weights_url = 'checkpoints/detr-r50-e632da11.pth'

            ### Others (Not found yet) ###
            self.lr = 1e-5
            self.batch_size = 2
            self.weight_decay = 1e-4
            self.epochs = 300
            self.lr_drop = 200
            self.clip_max_norm = 0.1
            ##############################

            ### Backbone ####
            # Positional Encoding
            self.hidden_dim = 256
            self.position_embedding = 'sine' # Type of positional embedding to use on top of the image features
            # Other backbone args
            self.lr_backbone = 1e-6
            self.backbone = 'resnet50'       # Name of the convolutional backbone to use
            self.dilation = False     # If true, we replace stride with dilation in the last convolutional block (DC5)
            #################

            ### Transformer ####
            self.hidden_dim = 256            # Size of the embeddings (dimension of the transformer)
            self.dropout = 0.1               # Dropout applied in the transformer
            self.nheads = 8                  # Number of attention heads inside the transformer's attentions
            self.dim_feedforward = 2048      # Intermediate size of the feedforward layers in the transformer blocks
            self.enc_layers = 6              # Number of encoding layers in the transformer
            self.dec_layers = 6              # Number of decoding layers in the transformer
            self.pre_norm = False            # action='store_true'
            #################

            ### DETR ###
            self.num_queries = 100           # Number of query slots
            self.aux_loss = True             # Auxiliary decoding losses (loss at each layer)
            ############

            ### Matcher ###
            self.set_cost_class = 1          # Class coefficient in the matching cost
            self.set_cost_bbox = 5           # L1 box coefficient in the matching cost
            self.set_cost_giou = 2           # giou box coefficient in the matching cost
            ###############
            
            ### Weight dict ###
            self.bbox_loss_coef = 5
            self.giou_loss_coef = 2
            ####################

            ### Criterion ###
            self.eos_coef = 0.1              # Relative classification weight of the no-object class
            #################

    args = Args()
    
    detr, criterion, postprocessors = build_model(args)

    print(f"Load pre-trained model from {args.pretrained_weights_url}")
    detr.load_state_dict(torch.load(args.pretrained_weights_url)['model'])

    # Swap the class prediction head to one that has 80 classes
    # i.e. No nodes for N/A classes
    class_embed = torch.nn.Linear(256, 81, bias=True)
    w, b = detr.class_embed.state_dict().values()
    keep = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90, 91
    ]
    class_embed.load_state_dict(dict(weight=w[keep], bias=b[keep]))
    detr.class_embed = class_embed

    # Prepare dataset transforms
    # Note: those custom transform function will add 'size' field to label dict.
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if args.split == 'train':
        transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333)
                ])
            ),
            normalize,
        ])
    elif args.split == 'test':
        transforms = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    
    transforms = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    # dataset = HicoDetDataset('hico_20160224_det', split=args.split, transforms=transforms)
    dataset = HicoDetDataset('hico_20160224_det', split=args.split, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=args.num_workers)

    ap, rec = eval(detr, dataloader, postprocessors['bbox'], threshold=0.7, device=args.device)
    print(f"The mAP is {ap.mean().item():.4f}, the mRec is {rec.mean().item():.4f}")
