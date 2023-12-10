import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ExtractCOCOClasses",
        description="Extract a mapping of COCO dataset classes from the annotation file.",
    )

    parser.add_argument("-i", "--input-path", default="coco/instances_train2017.json")
    parser.add_argument("-o", "--output-path", default="hico_20160224_det/coco_class_indices.json")
    parser.add_argument("-c", "--contiguous", action="store_true")
    args = parser.parse_args()

    with open(args.input_path, 'r') as fp:
        coco = json.load(fp=fp)

    coco_class_indices = {}

    if args.contiguous:
        for idx, category in enumerate(coco['categories']):
            coco_class_indices.update({category['name'].replace(' ', '_'): idx})
    else:
        for category in coco['categories']:
            coco_class_indices.update(
                {category['name'].replace(' ', '_'): category['id']})

    with open(args.output_path, 'w') as fp:
        json.dump(obj=coco_class_indices, fp=fp)
