import os
import json
import torch
import numpy as np
import pickle as pkl


def auc(precision: torch.Tensor, recall: torch.Tensor):
    ap = torch.tensor(0, dtype=torch.float64)
    max_rec = recall[-1]
    for i in range(recall.numel()):
        if recall[i] >= max_rec:
            break
        if recall[i] - recall[i - 1] == 0:
            continue
        if i == 0:
            ap += precision[i] * recall[i]
        else:
            ap += 0.5 * (precision[i] + precision[i - 1]) * (recall[i] - recall[i - 1])
    return ap


class AveragePrecision:
    def __init__(self) -> None:
        pass





with open('save_data.pkl', 'rb') as fp:
    save_data = pkl.load(file=fp)


all_boxes = []
all_labels = []
all_scores = []
all_tp_masks = []

for detection in save_data:
    num_dets = len(detection['scores'])
    for i in range(num_dets):
        all_boxes.append(detection['boxes'][i])
        all_labels.append(detection['labels'][i])
        all_scores.append(detection['scores'][i])
        all_tp_masks.append(detection['binary_labels'][i])

all_boxes = torch.stack(all_boxes)
all_labels = torch.tensor(all_labels)
all_scores = torch.tensor(all_scores, dtype=torch.float64)
all_tp_masks = torch.tensor(all_tp_masks)

unique_lables = all_labels.unique()


with open('hico_20160224_det/test_hico.json') as fp:
    train_val = json.load(fp=fp)

label_map = {i.item(): j for i, j in zip(unique_lables, list(range(len(unique_lables))))}
label_map_vec = np.vectorize(label_map.get)

num_gt = torch.zeros(80, dtype=int)
for ann in train_val:
    box_anns = ann['annotations']
    for box in box_anns:
        gt_label = box['category_id']
        gt_label_mapped = label_map[gt_label]
        num_gt[gt_label_mapped] += 1


ap_list = []
precs, recalls = [], []

for i, label in enumerate(unique_lables):
    label_mask = all_labels == label
    
    boxes = all_boxes[label_mask]
    scores = all_scores[label_mask]
    tp_masks = all_tp_masks[label_mask]

    sort_idx = scores.argsort(descending=True)

    boxes_sorted = boxes[sort_idx]
    scores_sorted = scores[sort_idx]
    tp_masks_sorted = tp_masks[sort_idx]

    total_tp = tp_masks_sorted.sum()
    fp = 1 - tp_masks_sorted

    # precision = tp_masks_sorted.cumsum(0) / (tp_masks_sorted.cumsum(0) + fp.cumsum(0))
    precision = (tp_masks_sorted.cumsum(0).to(torch.float64) / torch.ones(len(tp_masks_sorted)).cumsum(0).to(torch.float64)).to(torch.float64)
    recall = (tp_masks_sorted.cumsum(0).to(torch.float64) / num_gt[i]).to(torch.float64)
    # print(precision.dtype)
    precs.append(precision)
    recalls.append(recall)

    ap = auc(precision, recall)
    ap_list.append(ap)

aps = torch.tensor(ap_list)
print(aps)
aps.mean()




