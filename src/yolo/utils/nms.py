from __future__ import annotations
import torch
from yolo.utils.boxes import box_iou_xyxy

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    nms: only keeping the largest-score boxes among a set of boxes with much intersection (iou above threshold) 
    """
    # detect if it's empty
    if boxes.numel() == 0: return torch.empty((0,), dtype=torch.long, device=boxes.device)

    # argsort means the returned tensor is the INDEX of the numbers according to the order
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1: break

        #unsqueeze(0) means append a dimension at front. We need this to turn a (4) tensor to a (1, 4) tensor to go into IOU function
        #squeeze(0) means squeeze out that dimension
        ious = box_iou_xyxy(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        # delete all the remaining boxes that have much intersection with this selected box
        order= order[1:][ious <= iou_threshold]

    return torch.stack(keep).long()
