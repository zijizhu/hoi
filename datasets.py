import os
import json
import torch
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple

from torchvision.ops.boxes import batched_nms


class ImageDataset(Dataset):
    """
    Base class for image dataset

    Arguments:
        root(str): Root directory where images are downloaded to
        transform(callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        self._root = root
        self._transform = transform
        self._target_transform = target_transform
        if transforms is None:
            self._transforms = StandardTransform(transform, target_transform)
        elif transform is not None or target_transform is not None:
            print("WARNING: Argument transforms is given, transform/target_transform are ignored.")
            self._transforms = transforms
        else:
            self._transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError
    
    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tRoot path: {}\n'.format(self._root)
        return reprstr

    def load_image(self, path: str) -> Image: 
        """Load an image as PIL.Image"""
        return Image.open(path).convert('RGB')


class StandardTransform:
    """https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py"""

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, inputs: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


class DataSubset(Dataset):
    """
    A subset of data with access to all attributes of original dataset

    Arguments:
        dataset(Dataset): Original dataset
        pool(List[int]): The pool of indices for the subset
    """
    def __init__(self, dataset: Dataset, pool: List[int]) -> None:
        self.dataset = dataset
        self.pool = pool
    def __len__(self) -> int:
        return len(self.pool)
    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.pool[idx]]
    def __getattr__(self, key: str) -> Any:
        if hasattr(self.dataset, key):
            return getattr(self.dataset, key)
        else:
            raise AttributeError("Given dataset has no attribute \'{}\'".format(key))


class HICODetSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]
    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]
    @property
    def anno_interaction(self) -> List[int]:
        """Override: Number of annotated box pairs for each interaction class"""
        num_anno = [0 for _ in range(self.num_interation_cls)]
        intra_idx = [self._idx[i] for i in self.pool]
        for idx in intra_idx:
            for hoi in self._anno[idx]['hoi']:
                num_anno[hoi] += 1
        return num_anno
    @property
    def anno_object(self) -> List[int]:
        """Override: Number of annotated box pairs for each object class"""
        num_anno = [0 for _ in range(self.num_object_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[1]] += anno_interaction[corr[0]]
        return num_anno
    @property
    def anno_action(self) -> List[int]:
        """Override: Number of annotated box pairs for each action class"""
        num_anno = [0 for _ in range(self.num_action_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[2]] += anno_interaction[corr[0]]
        return num_anno


class HICODet(ImageDataset):
    """
    Arguments:
        root(str): Root directory where images are downloaded to
        anno_file(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """
    def __init__(self, root: str, anno_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        super(HICODet, self).__init__(root, transform, target_transform, transforms)
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = 80
        self.num_interation_cls = 600
        self.num_action_cls = 117
        self._anno_file = anno_file

        # Load annotations
        self._load_annotation_and_metadata(anno)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._idx)

    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image
        
        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "boxes_h": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
        """
        intra_idx = self._idx[i]
        img, anno =  self._transforms(
            self.load_image(os.path.join(self._root, self._filenames[intra_idx])), 
            self._anno[intra_idx]
            )
        return self._filenames[intra_idx], img, anno

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def class_corr(self) -> List[Tuple[int, int, int]]:
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()

    @property
    def object_n_verb_to_interaction(self) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([self.num_object_cls, self.num_action_cls], None)
        for i, j, k in self._class_corr:
            lut[j, k] = i
        return lut.tolist()

    @property
    def object_to_interaction(self) -> List[list]:
        """
        The interaction classes that involve each object type
        
        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

    @property
    def object_to_verb(self) -> List[list]:
        """
        The valid verbs for each object type

        Returns:
            list[list]
        """
        obj_to_verb = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_verb[corr[1]].append(corr[2])
        return obj_to_verb

    @property
    def anno_interaction(self) -> List[int]:
        """
        Number of annotated box pairs for each interaction class

        Returns:
            list[600]
        """
        return self._num_anno.copy()

    @property
    def anno_object(self) -> List[int]:
        """
        Number of annotated box pairs for each object class

        Returns:
            list[80]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """
        Number of annotated box pairs for each action class

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def objects(self) -> List[str]:
        """
        Object names 

        Returns:
            list[str]
        """
        return self._objects.copy()

    @property
    def verbs(self) -> List[str]:
        """
        Verb (action) names

        Returns:
            list[str]
        """
        return self._verbs.copy()

    @property
    def interactions(self) -> List[str]:
        """
        Combination of verbs and objects

        Returns:
            list[str]
        """
        return [self._verbs[j] + ' ' + self.objects[i] 
            for _, i, j in self._class_corr]

    def split(self, ratio: float) -> Tuple[HICODetSubset, HICODetSubset]:
        """
        Split the dataset according to given ratio

        Arguments:
            ratio(float): The percentage of training set between 0 and 1
        Returns:
            train(Dataset)
            val(Dataset)
        """
        perm = np.random.permutation(len(self._idx))
        n = int(len(perm) * ratio)
        return HICODetSubset(self, perm[:n]), HICODetSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[self._idx[idx]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_sizes[self._idx[idx]]

    def _load_annotation_and_metadata(self, f: dict) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """
        idx = list(range(len(f['filenames'])))
        for empty_idx in f['empty']:
            idx.remove(empty_idx)

        num_anno = [0 for _ in range(self.num_interation_cls)]
        for anno in f['annotation']:
            for hoi in anno['hoi']:
                num_anno[hoi] += 1

        self._idx = idx
        self._num_anno = num_anno

        self._anno = f['annotation']
        self._filenames = f['filenames']
        self._image_sizes = f['size']
        self._class_corr = f['correspondence']
        self._empty_idx = f['empty']
        self._objects = f['objects']
        self._verbs = f['verbs']


class HICODetObject(Dataset):
    def __init__(self, dataset, transforms, nms_thresh=0.7):
        self.dataset = dataset
        self.transforms = transforms
        self.nms_thresh = nms_thresh
        self.conversion = [
             4, 47, 24, 46, 34, 35, 21, 59, 13,  1, 14,  8, 73, 39, 45, 50,  5,
            55,  2, 51, 15, 67, 56, 74, 57, 19, 41, 60, 16, 54, 20, 10, 42, 29,
            23, 78, 26, 17, 52, 66, 33, 43, 63, 68,  3, 64, 49, 69, 12,  0, 53,
            58, 72, 65, 48, 76, 18, 71, 36, 30, 31, 44, 32, 11, 28, 37, 77, 38,
            27, 70, 61, 79,  9,  6,  7, 62, 25, 75, 40, 22
        ]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        fn, image, target = self.dataset[idx]
        boxes = torch.cat([
            target['boxes_h'],
            target['boxes_o']
        ])
        # Convert ground truth boxes to zero-based index and the
        # representation from pixel indices to coordinates
        boxes[:, :2] -= 1
        labels = torch.cat([
            49 * torch.ones_like(target['object']),
            target['object']
        ])
        # Remove overlapping ground truth boxes
        keep = batched_nms(
            boxes, torch.ones(len(boxes)),
            labels, iou_threshold=self.nms_thresh
        )
        boxes = boxes[keep]
        labels = labels[keep]
        # Convert HICODet object indices to COCO indices
        converted_labels = torch.as_tensor([self.conversion[i.item()] for i in labels])
        # Apply transform
        image, target = self.transforms(image, dict(boxes=boxes, labels=converted_labels))
        return fn, image, target


def _to_list_of_tensor(x, dtype=None, device=None):
    return [torch.as_tensor(item, dtype=dtype, device=device) for item in x]

def _to_tuple_of_tensor(x, dtype=None, device=None):
    return tuple(torch.as_tensor(item, dtype=dtype, device=device) for item in x)

def _to_dict_of_tensor(x, dtype=None, device=None):
    return dict([(k, torch.as_tensor(v, dtype=dtype, device=device)) for k, v in x.items()])


def to_tensor(x, input_format='tensor', dtype=None, device=None):
    """Convert input data to tensor based on its format"""
    if input_format == 'tensor':
        return torch.as_tensor(x, dtype=dtype, device=device)
    elif input_format == 'pil':
        return torchvision.transforms.functional.to_tensor(x).to(
            dtype=dtype, device=device)
    elif input_format == 'list':
        return _to_list_of_tensor(x, dtype=dtype, device=device)
    elif input_format == 'tuple':
        return _to_tuple_of_tensor(x, dtype=dtype, device=device)
    elif input_format == 'dict':
        return _to_dict_of_tensor(x, dtype=dtype, device=device)
    else:
        raise ValueError("Unsupported format {}".format(input_format))


class ToTensor:
    """Convert to tensor"""
    def __init__(self, input_format='tensor', dtype=None, device=None):
        self.input_format = input_format
        self.dtype = dtype
        self.device = device
    def __call__(self, x):
        return to_tensor(x, 
            input_format=self.input_format,
            dtype=self.dtype,
            device=self.device
        )
    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'input_format={}'.format(repr(self.input_format))
        reprstr += ', dtype='
        reprstr += repr(self.dtype)
        reprstr += ', device='
        reprstr += repr(self.device)
        reprstr += ')'
        return reprstr
