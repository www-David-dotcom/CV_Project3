from __future__ import annotations
import torch

def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    # note x,y,w,h are all tensors here
    x, y, w, h = boxes.unbind(dim=-1)
    half_w = w / 2
    half_h = h / 2
    # torch stack means stack the tensors according to the final dimension, i.e. [x1, y1, x2, y2]
    return torch.stack((x - half_w, y - half_h, x + half_w, y + half_h), dim=-1)

def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2
    return torch.stack((x, y, w, h), dim=-1)

def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    heights = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return widths * heights

def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    calculate intersection over union over ALL pairs of boxes between boxes1 and boxes2
    boxes1: tensor of shape [N, 4]
    boxes2: tensor of shape [M, 4]
    """
    # numel() counts the total elements number in the tensor
    # if any of the detected boxes set were empty, return a zero tensor
    # with the same type as boxes1, with shape [N, M]
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    
    # :2 means x1, y1(top and left), while 2: means x2, y2(bottom and right)
    # by broadcasting ,these tensors [N, None,  2] and [None, M, 2] becomes tensors of the same shape
    # [N, M, 2], then maximum or minimum function does a per-element compare
    # the final tensor is of shape [N, M, 2]
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, : , :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    # wh(last dimension has dim 2 -> (w, h)) is the width and height of the intersection
    # as width and height must be nonnegative, we clamp to min 0
    wh = (bottom_right - top_left).clamp(min=0)
    # claculate the area, the output tensor is of shape [N, M]
    intersection = wh[..., 0] * wh[..., 1]

    area1 = box_area_xyxy(boxes1)
    area2 = box_area_xyxy(boxes2)
    #inclusion exclusion
    union = area1[:, None] + area2[None, :] - intersection
    # as the denominator can't be 0, we clamp to a nonzero value
    # the output tensor is of size [N, M]
    return intersection / union.clamp(min=1e-8)

def clip_boxes_xyxy(boxes: torch.Tensor, min_value: float = 0.0, max_value: float = 1.0) -> torch.Tensor:
    return boxes.clamp(min=min_value, max=max_value)