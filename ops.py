import torch
import torchvision.ops.boxes as box_ops

def box_iou(boxes_1: torch.Tensor, boxes_2: torch.Tensor, encoding: str = 'coord') -> torch.Tensor:
    """
    Compute intersection over union between boxes

    Parameters:
    -----------
    boxes_1: torch.Tensor
        (N, 4) Bounding boxes formatted as [[x1, y1, x2, y2],...]
    boxes_2: torch.Tensor
        (M, 4) Bounding boxes formatted as [[x1, y1, x2, y2],...]
    encoding: str
        A string that indicates what the boxes encode
            'coord': Coordinates of the two corners
            'pixel': Pixel indices of the two corners

    Returns:
    --------
    torch.Tensor
        Intersection over union of size (N, M)

    """
    if encoding == 'coord':
        return box_ops.box_iou(boxes_1, boxes_2)
    elif encoding == 'pixel':
        w1 = (boxes_1[:, 2] - boxes_1[:, 0] + 1).clamp(min=0)
        h1 = (boxes_1[:, 3] - boxes_1[:, 1] + 1).clamp(min=0)
        s1 = w1 * h1
        w2 = (boxes_2[:, 2] - boxes_2[:, 0] + 1).clamp(min=0)
        h2 = (boxes_2[:, 3] - boxes_2[:, 1] + 1).clamp(min=0)
        s2 = w2 * h2

        n1 = len(boxes_1); n2 = len(boxes_2)
        i, j = torch.meshgrid(
            torch.arange(n1),
            torch.arange(n2)
        )
        i = i.flatten(); j = j.flatten()
        
        x1, y1 = torch.max(boxes_1[i, :2], boxes_2[j, :2]).unbind(1)
        x2, y2 = torch.min(boxes_1[i, 2:], boxes_2[j, 2:]).unbind(1)
        w_intr = (x2 - x1 + 1).clamp(min=0)
        h_intr = (y2 - y1 + 1).clamp(min=0)
        s_intr = w_intr * h_intr

        iou = s_intr / (s1[i] + s2[j] - s_intr)
        return iou.reshape(n1, n2)
    else:
        raise ValueError("The encoding type should be either \"coord\" or \"pixel\"")