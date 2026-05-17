from __future__ import annocations
import torch
from yolo.utils.boxes import box_iou_xyxy

def interpolated_average_precision(precisions: torch.Tensor, recalls: torch.Tensor) -> float:
    """
    calculate the ap using interpolation method.

    The theory is, if we gradually let down the threshold, there'll definitely be more(no less) matched ones(TP), so Recall will definitely nondecrease
    but precision may be floating. Here if the precision drops when threshold
    """
    # add 0.0 and 0.0 on top and bottom
    mprec = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    # add 0.0 and 1.0 on top and bottom (high threshold -> low threshold)
    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])

    # from the last precision to the first, backward iteration
    # -2 because we added a 0.0 after the last precision in the last step
    for index in range(mprec.numel() - 2, -1, -1):
        # let the Precision be decreasing (i.e. if one precision is lower than the precision of a higher recall, then align to that precision). 
        # This is because, if 0.2 recall corresponds to 0.5 precision, but 0.3 recall corresponds to 0.7 precision,
        # then we can definitely lower the threshold to 0.3 recall to get a 0.67 precision if we want to get the best precision under 0.2 recall
        # so we let the precision be nonincreasing.
        mprec[index] = torch.maximum(mprec[index], mprec[index + 1])

    # torch.where means finding the indices where the boolean value inside it happens
    # this is finding the changing points of x-axis(mrec) to calculate the length of each small rectangles
    changing_points = torch.where(mrec[1:] != mrec[:-1])[0]
    # integrate by summing areas of little rectangles
    ap = torch.sum((mrec[changing_points + 1] - mrec[changing_points]) * mprec[changing_points + 1])
    return float(ap.item())

def precision_recall_ap(
        pred_boxes: list[torch.Tensor],
        pred_scores: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        iou_threshold: float,
) -> dict[str, float | list[float]]:
    """
    TP(True Positive): prediction matches the target(IOU > threhold)
    FP: prediction fails to match the target
    Precision = TP / (TP + FP)
    Recall = TP / (target_num)
    AP: the area under the precision-recall curve
    """
    # the tuple is (score, is_tp, is_fp)
    records: list[tuple[float, int, int]] = []
    # sum up the total targets number in all the images
    total_targets = sum(int(boxes.shape[0]) for boxes in target_boxes)

    for image_index, boxes in enumerate(pred_boxes):
        scores = pred_scores[image_index]
        targets = target_boxes[image_index]
        # create a boolean tensor to mark if every target is successfully matched
        matched = torch.zeros((targets.shape[0],), dtype=torch.bool, device=targets.device)

        if boxes.numel() == 0: continue

        # sort from high score to low score
        order = scores.argsort(descending=True)
        for pred_index in order:
            score = float(scores[pred_index].item())
            if targets.numel() == 0: # if there's no targets in this picture, mark all the predictions as FP
                records.append((score, 0, 1))
                continue

            # calculate iou of the current prediction and all the targets
            # as current prediciton has just one number, we need to add a (1, ) dimension to make it a tensor
            ious = box_iou_xyxy(boxes[pred_index].unsqueeze(0), targets).squeeze(0)
            best_iou, best_target = ious.max(dim=0)
            # if the best iou is larger than threshold and the target has not been matched yet
            # then mark the target as matched, and add a TP
            if best_iou >= iou_threshold and not matched[best_target]:
                matched[best_target] = True
                records.append((score, 1, 0))
            else:
                # else this prediction is FP
                records.append((score, 0, 1))

        if total_targets == 0:
            return {"precision": 0.0, "recall": 0.0, "ap": 0.0, "precisions": [], "recalls": []}
        
        records.sort(key=lambda item: item[0], reverse=True)
        tp = torch.tensor([item[1] for item in records], dtype=torch.float32)
        fp = torch.tensor([item[2] for item in records], dtype=torch.float32)

        if tp.numel() == 0:
            return {"precision": 0.0, "recall": 0.0, "ap": 0.0, "precisions": [], "recalls": []}
        
        # cumsum menas add according to some dimension to get a lower-dimension tensor
        # add the boolean variables to get the total TP and FP num
        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)
        recalls = cum_tp / max(total_targets, 1)
        precisions = cum_tp / (cum_tp + cum_fp).clamp(min=1e-8)
        ap = interpolated_average_precision(precisions, recalls)
        
        return{
            "precision": float(precisions[-1].item()),
            "recall": float(recalls[-1].item()),
            "ap": float(ap),
            "precisions": [float(x) for x in precisions.tolist()],
            "recalls": [float(x) for x in recalls.tolist()],
        }