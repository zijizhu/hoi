import json


if __name__ == '__main__':
    contiguous = True
    with open('coco/instances_train2017.json', 'r') as fp:
        coco = json.load(fp=fp)

    coco_class_indices = {}

    if contiguous:
        for idx, category in enumerate(coco['categories']):
            coco_class_indices.update({category['name'].replace(' ', '_'): idx})
        
        with open('hico_20160224_det/coco_class_indices_contiguous.json', 'w') as fp:
            json.dump(obj=coco_class_indices, fp=fp)
    else:
        for category in coco['categories']:
            coco_class_indices.update(
                {category['name'].replace(' ', '_'): category['id']})

        with open('hico_20160224_det/coco_class_indices.json', 'w') as fp:
            json.dump(obj=coco_class_indices, fp=fp)
