"""
Meters for the purpose of statistics tracking

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import multiprocessing

import torch
from torch import Tensor
from collections import deque
from typing import Optional, Iterable, Any, List, Union, Tuple

__all__ = [
    'Meter', 'NumericalMeter', 'HandyTimer',
    'AveragePrecisionMeter', 'DetectionAPMeter'
]

def div(numerator: Tensor, denom: Union[Tensor, int, float]) -> Tensor:
    """Handle division by zero"""
    if type(denom) in [int, float]:
        if denom == 0:
            return torch.zeros_like(numerator)
        else:
            return numerator / denom
    elif type(denom) is Tensor:
        zero_idx = torch.nonzero(denom == 0).squeeze(1)
        denom[zero_idx] += 1e-8
        return numerator / denom
    else:
        raise TypeError("Unsupported data type ", type(denom))


def compute_per_class_ap_as_auc(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
    """
    Arguments: 
        tuple_(Tuple[Tensor, Tensor]): precision and recall
    Returns:
        ap(Tensor[1])
    """
    prec, rec = tuple_
    ap = 0
    max_rec = rec[-1]
    for idx in range(prec.numel()):
        # Stop when maximum recall is reached
        if rec[idx] >= max_rec:
            break
        d_x = rec[idx] - rec[idx - 1]
        # Skip when negative example is registered
        if d_x == 0:
            continue
        ap +=  prec[idx] * rec[idx] if idx == 0 \
            else 0.5 * (prec[idx] + prec[idx - 1]) * d_x
    return ap


def compute_per_class_ap_with_interpolation(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
    """
    Arguments:
        tuple_(Tuple[Tensor, Tensor]): precision and recall
    Returns:
        ap(Tensor[1])
    """
    prec, rec = tuple_
    ap = 0
    max_rec = rec[-1]
    for idx in range(prec.numel()):
        # Stop when maximum recall is reached
        if rec[idx] >= max_rec:
            break
        d_x = rec[idx] - rec[idx - 1]
        # Skip when negative example is registered
        if d_x == 0:
            continue
        # Compute interpolated precision
        max_ = prec[idx:].max()
        ap +=  max_ * rec[idx] if idx == 0 \
            else 0.5 * (max_ + torch.max(prec[idx - 1], max_)) * d_x
    return ap


def compute_per_class_ap_with_11_point_interpolation(tuple_: Tuple[Tensor, Tensor]) -> Tensor:
    """
    Arguments:
        tuple_(Tuple[Tensor, Tensor]): precision and recall
    Returns:
        ap(Tensor[1])
    """
    prec, rec = tuple_
    dtype = rec.dtype
    ap = 0
    for t in torch.linspace(0, 1, 11, dtype=dtype):
        inds = torch.nonzero(rec >= t).squeeze()
        if inds.numel():
            ap += (prec[inds].max() / 11)
    return ap


class DetectionAPMeter:
    """
    A variant of AP meter, where network outputs are assumed to be class-specific.
    Different classes could potentially have different number of samples.

    Required Arguments:
        num_cls(int): Number of target classes
    Optional Arguemnts:
        num_gt(iterable): Number of ground truth instances for each class. When left
            as None, all positives are assumed to have been included in the collected
            results. As a result, full recall is guaranteed when the lowest scoring
            example is accounted for.
        algorithm(str, optional): A choice between '11P' and 'AUC'
            '11P': 11-point interpolation algorithm prior to voc2010
            'INT': Interpolation algorithm with all points used in voc2010
            'AUC': Precisely as the area under precision-recall curve
        nproc(int, optional): The number of processes used to compute mAP. Default: 20
        precision(int, optional): Precision used for float-point operations. Choose
            amongst 64, 32 and 16. Default is 64
        output(list[tensor], optinoal): A collection of output scores for K classes
        labels(list[tensor], optinoal): Binary labels

    Usage:

    (1) Evalute AP using provided output scores and labels

        >>> # Given output(list[tensor]) and labels(list[tensor])
        >>> meter = pocket.utils.DetectionAPMeter(num_cls, output=output, labels=labels)
        >>> ap = meter.eval(); map_ = ap.mean()

    (2) Collect results on the fly and evaluate AP

        >>> meter = pocket.utils.DetectionAPMeter(num_cls)
        >>> # Get class-specific predictions. The following is an example
        >>> # Assume output(tensor[N, K]) and target(tensor[N]) is given
        >>> pred = output.argmax(1)
        >>> scores = output.max(1)
        >>> meter.append(scores, pred, pred==target)
        >>> ap = meter.eval(); map_ = ap.mean()
        >>> # If you are to start new evaluation and want to reset the meter
        >>> meter.reset()

    """
    def __init__(self, num_cls: int, num_gt: Optional[Tensor] = None,
            algorithm: str = 'AUC', nproc: int = 20,
            precision: int = 64,
            output: Optional[List[Tensor]] = None,
            labels: Optional[List[Tensor]] = None) -> None:
        if num_gt is not None and len(num_gt) != num_cls:
            raise AssertionError("Provided ground truth instances"
                "do not have the same number of classes as specified")

        self.num_cls = num_cls
        self.num_gt = num_gt if num_gt is not None else \
            [None for _ in range(num_cls)]
        self.algorithm = algorithm
        self._nproc = nproc
        self._dtype = eval('torch.float' + str(precision))

        is_none = (output is None, labels is None)
        if is_none == (True, True):
            self._output = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
            self._labels = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
        elif is_none == (False, False):
            assert len(output) == len(labels), \
                "The given output does not have the same number of classes as labels"
            assert len(output) == num_cls, \
                "The number of classes in the given output does not match the argument"
            # self._output = torch.as_tensor(output, 
            #     input_format='list', dtype=self._dtype, device='cpu')
            self._output = [torch.as_tensor(item, dtype=self._dtype, device='cpu') for item in output]
            # self._labels = torch.as_tensor(labels,
            #     input_format='list', dtype=self._dtype, device='cpu')
            self._labels = [torch.as_tensor(item, dtype=self._dtype, device='cpu') for item in labels]
        else:
            raise AssertionError("Output and labels should both be given or None")

        self._output_temp = [[] for _ in range(num_cls)]
        self._labels_temp = [[] for _ in range(num_cls)]
    
    @classmethod
    def compute_ap(cls, output: List[Tensor], labels: List[Tensor],
            num_gt: Iterable, nproc: int, algorithm: str = 'AUC') -> Tuple[Tensor, Tensor]:
        """
        Compute average precision under the detection setting. Only scores of the 
        predicted classes are retained for each sample. As a result, different classes
        could have different number of predictions.

        Arguments:
            output(list[Tensor])
            labels(list[Tensor])
            num_gt(iterable): Number of ground truth instances for each class
            nproc(int, optional): The number of processes used to compute mAP
            algorithm(str): AP evaluation algorithm
        Returns:
            ap(Tensor[K])
            max_rec(Tensor[K])
        """
        ap = torch.zeros(len(output), dtype=output[0].dtype)
        max_rec = torch.zeros_like(ap)

        if algorithm == 'INT':
            algorithm_handle = compute_per_class_ap_with_interpolation
        elif algorithm == '11P':
            algorithm_handle = compute_per_class_ap_with_11_point_interpolation
        elif algorithm == 'AUC':
            algorithm_handle = compute_per_class_ap_as_auc
        else:
            raise ValueError("Unknown algorithm option {}.".format(algorithm))

        # Avoid multiprocessing when the number of processes is fewer than two
        precs, recalls = [], []
        if nproc < 2:
            for idx in range(len(output)):
                ap[idx], max_rec[idx], p, r = cls.compute_ap_for_each((
                    idx, list(num_gt)[idx],
                    output[idx], labels[idx],
                    algorithm_handle
                ))
                precs.append(p)
                recalls.append(r)
            return ap, max_rec, precs, recalls

        with multiprocessing.get_context('spawn').Pool(nproc) as pool:
            for idx, results in enumerate(pool.map(
                func=cls.compute_ap_for_each,
                iterable=[(idx, ngt, out, gt, algorithm_handle) 
                    for idx, (ngt, out, gt) in enumerate(zip(num_gt, output, labels))]
            )):
                ap[idx], max_rec[idx] = results

        return ap, max_rec, precs, recalls

    @classmethod
    def compute_ap_for_each(cls, tuple_):
        idx, num_gt, output, labels, algorithm = tuple_
        # Sanity check
        if num_gt is not None and labels.sum() > num_gt:
            raise AssertionError("Class {}: ".format(idx)+
                "Number of true positives larger than that of ground truth")
        if len(output) and len(labels):
            prec, rec = cls.compute_pr_for_each(output, labels, num_gt)
            return algorithm((prec, rec)), rec[-1], prec, rec
        else:
            print("WARNING: Collected results are empty. "
                "Return zero AP for class {}.".format(idx))
            return 0, 0

    @staticmethod
    def compute_pr_for_each(output: Tensor, labels: Tensor,
            num_gt: Optional[Union[int, float]] = None) -> Tuple[Tensor, Tensor]:
        """
        Arguments:
            output(Tensor[N])
            labels(Tensor[N]): Binary labels for each sample
            num_gt(int or float): Number of ground truth instances
        Returns:
            prec(Tensor[N])
            rec(Tensor[N])
        """
        order = output.argsort(descending=True)

        tp = labels[order]
        fp = 1 - tp
        tp = tp.cumsum(0)
        fp = fp.cumsum(0)

        prec = tp / (tp + fp)
        rec = div(tp, labels.sum().item()) if num_gt is None \
            else div(tp, num_gt)

        return prec, rec

    def append(self, output: Tensor, prediction: Tensor, labels: Tensor) -> None:
        """
        Add new results to the meter

        Arguments:
            output(tensor[N]): Output scores for each sample
            prediction(tensor[N]): Predicted classes 0~(K-1)
            labels(tensor[N]): Binary labels for the predicted classes
        """
        if isinstance(output, torch.Tensor) and \
                isinstance(prediction, torch.Tensor) and \
                isinstance(labels, torch.Tensor):
            prediction = prediction.long()
            unique_cls = prediction.unique()
            for cls_idx in unique_cls:
                sample_idx = torch.nonzero(prediction == cls_idx).squeeze(1)
                ######## Debug ###########
                # print('len(self._output_temp):', len(self._output_temp))
                # print('cls_idx.item():', cls_idx.item())
                # print('len(output):', len(output))
                # print('sample_idx:', sample_idx)
                self._output_temp[cls_idx.item()] += output[sample_idx].tolist()
                self._labels_temp[cls_idx.item()] += labels[sample_idx].tolist()
        else:
            raise TypeError("Arguments should be torch.Tensor")

    def reset(self, keep_old: bool = False) -> None:
        """
        Clear saved statistics

        Arguments:
            keep_tracked(bool): If True, clear only the newly collected statistics
                since last evaluation
        """
        num_cls = len(self._output_temp)
        if not keep_old:
            self._output = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
            self._labels = [torch.tensor([], dtype=self._dtype) for _ in range(num_cls)]
        self._output_temp = [[] for _ in range(num_cls)]
        self._labels_temp = [[] for _ in range(num_cls)]

    def eval(self) -> Tensor:
        """
        Evaluate the average precision based on collected statistics

        Returns:
            torch.Tensor[K]: Average precisions for K classes
        """
        self._output = [torch.cat([out1, torch.as_tensor(out2, dtype=self._dtype)]) for out1, out2 in zip(self._output, self._output_temp)]
        self._labels = [torch.cat([tar1, torch.as_tensor(tar2, dtype=self._dtype)]) for tar1, tar2 in zip(self._labels, self._labels_temp)]
        self.reset(keep_old=True)

        self.ap, self.max_rec, precs, recs = self.compute_ap(self._output,
                                                self._labels,
                                                self.num_gt,
                                                nproc=self._nproc,
                                                algorithm=self.algorithm)

        return self.ap, precs, recs
