import torch
import random
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from torchvision import ops
from torch.utils.data import DataLoader

from detr.models import build_model
from detr.datasets import transforms as T

from meter import DetectionAPMeter
from association import BoxAssociation
from utils.relocate import relocate_to_device
from dataset.hicodet import HicoDetDataset, collate_fn


def eval(model: torch.nn.Module, dataloader, postprocessors, threshold=0.7, device='cpu'):
    model.eval()
    model.to(device=device)

    associate = BoxAssociation(min_iou=0.5)
    meter = DetectionAPMeter(80, algorithm='INT', nproc=10)
    num_gt = torch.zeros(80)

    model.to(device)

    save_data = []

    if dataloader.batch_size != 1:
        raise ValueError(f"The batch size shoud be 1, not {dataloader.batch_size}")
    for image_path, image, target in tqdm(dataloader):
        image = relocate_to_device(image, device=device)
        output = model(image)
        output = relocate_to_device(output, device='cpu')

        scores, labels, boxes = postprocessors(output, target[0]['size'].unsqueeze(0))[0].values()
        keep = torch.nonzero(scores >= threshold).squeeze(1)

        postprocessed_output = dict(raw=output, scores=scores, labels=labels, boxes=boxes)

        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        # Convert normalized GT boxes to full scale
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

        kept_output = dict(scores=scores, labels=labels, boxes=boxes, binary_labels=binary_labels)

        meter.append(scores, labels, binary_labels)

        save_data.append(dict(postprocessed=postprocessed_output, kept=kept_output))

    meter.num_gt = num_gt.tolist()

    with open('save_data.pkl', 'wb') as fp:
        pkl.dump(save_data, file=fp)

    return meter.eval(), meter.max_rec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--num-workers', default=9, type=int)
    parser.add_argument('--print-interval', default=1000, type=int)
    parser.add_argument('--data-dir', default='hico_20160224_det', type=str)
    parser.add_argument('--resume', default='', type=str, help='Resume from a checkpoint')
    parser.add_argument('--pretrained', default='', type=str, help='Start from a pre-trained model')

    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--output-dir', default='checkpoints')

    ##### Training #####
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr-drop', default=200, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    ##### Backbone #####
    parser.add_argument('--lr-backbone', default=1e-6, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position-embedding', default='sine', type=str, choices=['sine', 'learned'],
                        help="Type of positional embedding to use on top of the image features")

    ##### Transformer #####
    parser.add_argument('--enc-layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec-layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim-feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden-dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num-queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre-norm', action='store_true')

    ##### Loss & Matcher #####

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set-cost-class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set-cost-bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set-cost-giou', default=2, type=float,
                        help="Giou box coefficient in the matching cost")
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    args = parser.parse_args()
    print(args)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Build model
    detr, criterion, postprocessors = build_model(args)

    if args.pretrained:
        print(f'Load pre-trained model from {args.pretrained}')
        detr.load_state_dict(torch.load(args.pretrained)['model'])

    # Swap the class prediction head to one that has 80 classes
    # i.e. No output for N/A classes
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

    if args.resume:
        print(f'Load checkpoint from {args.resume}')
        detr.load_state_dict(torch.load(args.resume)['model_state_dict'])

    # Prepare dataset transforms
    # DETR's custom transform functions will add 'size' field to label dict.
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if args.eval:
        transforms = T.Compose([T.RandomResize([800], max_size=1333), normalize])
    else:
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

    split = 'test' if args.eval else 'train'
    dataset = HicoDetDataset('hico_20160224_det', split=split, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=args.num_workers)

    ap, rec = eval(detr, dataloader, postprocessors['bbox'], threshold=0.1, device=args.device)
    print(f"The mAP is {ap.mean().item():.4f}, the mRec is {rec.mean().item():.4f}")
