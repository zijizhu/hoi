import os
import json
import numpy as np
import scipy.io as IO
from pprint import pprint
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

# Clean and unify bbox


def clean_bbox_anns(bbox_anns):
    '''
    Given the parsed original bbox_annotation
    Make the data types consistent
    '''
    bbox_cleaned = []
    for instance in bbox_anns:
        instance_cleaned = {}
        all_hoi_cleaned = []
        if not isinstance(instance['hoi'], list):
            instance['hoi'] = [instance['hoi']]
        for hoi in instance['hoi']:
            hoi_cleaned = hoi.copy()
            hoi_cleaned['connection'] = hoi['connection'].tolist()
            
            # There are 3 cases for hoi['bboxhuman'] and hoi['bboxobject']:
            # 1. list: There are multiple human-object pairt for one interaction
            # 2. dict: One human-object pair for the interation
            # 3. An empty np.ndarray: empty (invisible) interaction
            # If it's the 2nd or 3rd case, they are turned into lists
            if isinstance(hoi_cleaned['bboxhuman'], dict) or isinstance(hoi_cleaned['bboxobject'], dict):
                if isinstance(hoi_cleaned['bboxhuman'], dict):
                    hoi_cleaned['bboxhuman'] = [hoi_cleaned['bboxhuman']]
                if isinstance(hoi_cleaned['bboxobject'], dict):
                    hoi_cleaned['bboxobject'] = [hoi_cleaned['bboxobject']]
            elif not (isinstance(hoi_cleaned['bboxhuman'], list) or isinstance(hoi_cleaned['bboxobject'], list)):
                hoi_cleaned['bboxhuman'] = []
                hoi_cleaned['bboxobject'] = []

            all_hoi_cleaned.append(hoi_cleaned)
        instance_cleaned = instance.copy()
        instance_cleaned['hoi'] = all_hoi_cleaned

        bbox_cleaned.append(instance_cleaned)
    return bbox_cleaned


if __name__ == '__main__':
    bbox_mat = IO.loadmat(
        'hico_20160224_det/anno_bbox.mat',
        simplify_cells=True)
    label_mat = IO.loadmat('hico_20160224_det/anno.mat', simplify_cells=True)

    bbox_train = bbox_mat['bbox_train']
    bbox_test = bbox_mat['bbox_test']
    list_action = bbox_mat['list_action']

    # Clean bounding box annotations for each instance
    bbox_train_cleaned = clean_bbox_anns(bbox_train)
    bbox_test_cleaned = clean_bbox_anns(bbox_test)
    list_action_cleaned = [{'nname': action['nname'],
                            'vname': action['vname']} for action in list_action]

    # Generate action indices for each instance
    anno_train = label_mat['anno_train'].T
    anno_test = label_mat['anno_test'].T

    train_hoi_indices = []
    test_hoi_indices = []

    for i in range(anno_train.shape[0]):
        hoi_indices = np.argwhere(anno_train[i] == 1).flatten().tolist()
        train_hoi_indices.append(hoi_indices)

    for i in range(anno_test.shape[0]):
        hoi_indices = np.argwhere(anno_test[i] == 1).flatten().tolist()
        test_hoi_indices.append(hoi_indices)

    anno_cleaned = {'bbox_train': bbox_train_cleaned,
                    'bbox_test': bbox_test_cleaned,
                    'hoi_train': train_hoi_indices,
                    'hoi_test': test_hoi_indices,
                    'list_action': list_action_cleaned,
                    'num_test_instance_invis_only': 485,
                    'num_train_instance_invis_only': 112}

    with open('hico_20160224_det/anno_cleaned.json', 'w') as fp:
        json.dump(obj=anno_cleaned, fp=fp)
