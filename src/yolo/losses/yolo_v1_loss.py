from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from yolo.utils.boxes import box_iou_xyxy, xywh_to_xyxy

class YOLOv1Loss(nn.Module):
    def __init__(
            self,
            grid_size: int = 7,
            boxes_per_cell: int = 2,
            num_classes: int = 1,
            lambda_coord: float = 5.0,
            lambda_noobj: float = 0.5
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def _cell_to_image_xywh(
            self,
            xy: torch.Tensor, wh: torch.Tensor,
            cell_x: torch.Tensor, cell_y: torch.Tensor
    ) -> torch.Tensor:
        """
        turn local x, y to global x, y
        and return global (x, y, w, h) in the whole image
        """
        s = self.grid_size
        # stack to get a (:, 2) tensor, then unsqeeze behind to add a dim
        grid = torch.stack((cell_x, cell_y), dim=-1).unsqueeze(1)
        centers = (xy + grid) / s
        # concat (center_x, center_y) and (wh) to get (x, y, w, h) in global coordinates
        return torch.cat((centers, wh), dim=-1)
    
    def _best_box_indices(
            self, 
            pred_xy: torch.Tensor,
            pred_wh: torch.Tensor,
            target_xy: torch.Tensor,
            target_wh: torch.Tensor,
            obj_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_indices = obj_indices[:, 0]
        cell_y = obj_indices[:, 1].float()
        cell_x = obj_indices[:, 2].float()

        pred_xy_obj = pred_xy[batch_indices, obj_indices[:, 1], obj_indices[:, 2]]
        pred_wh_obj = pred_wh[batch_indices, obj_indices[:, 1], obj_indices[:, 2]]
        target_xy_obj = target_xy[batch_indices, obj_indices[:, 1], obj_indices[:, 2]]
        target_wh_obj = target_wh[batch_indices, obj_indices[:, 1], obj_indices[:, 2]]
        
        pred_global = self._cell_to_image_xywh(pred_xy_obj, pred_wh_obj, cell_x, cell_y)
        target_global = self._cell_to_image_xywh(target_xy_obj, target_wh_obj, cell_x, cell_y)
        best_indices = []
        # if _global is (S, S, b, 5)
        # then _per_cell (b, 5)
        for pred_per_cell, target_per_cell in zip(pred_global, target_global, strict=True):
            ious = box_iou_xyxy(
                xywh_to_xyxy(pred_per_cell),
                # as there's only one target per cell, only the first row is valid
                # we repeat it horizontally for self.boxes_per_cell times to align with the pred_per_cell
                xywh_to_xyxy(target_per_cell[:1]).repeat(self.boxes_per_cell, 1),
            ).diag()
            best_indices.append(torch.argmax(ious))
        return torch.stack(best_indices) # a (*,) 1-dim tensor of the best predictor box indices in each cell.

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # split the last 5b values into (b, 5) (two dimensions)
        pred_boxes = predictions[..., : self.boxes_per_cell * 5].view(
            *predictions.shape[:3], self.boxes_per_cell, 5 #x, y, w, h, conf
        )
        target_boxes = targets[..., : self.boxes_per_cell * 5].view(
            *predictions.shape[:3], self.boxes_per_cell, 5 # x, w, w, h, 1.0
        )
        pred_classes = predictions[..., self.boxes_per_cell * 5:]
        target_classes = targets[..., self.boxes_per_cell * 5:]

        pred_xy = torch.sigmoid(pred_boxes[..., :2]) # use sigmoid to normalize it to (0, 1)
        pred_wh = pred_boxes[..., 2:4].clamp(min=0)
        pred_conf = torch.sigmoid(pred_boxes[..., 4])
        target_xy = target_boxes[...,:2]
        target_wh = target_boxes[..., 2:4]
        target_conf = target_boxes[..., 4]

        obj_mask = target_conf[..., 0] > 0
        noobj_mask = ~obj_mask

        responsible = torch.zeros_like(target_conf, dtype=torch.bool)
        if obj_mask.any():
            # the indices can be seen as a list of tuples, whose element is
            # (batch_index, grid_y, grid_x, box_index)
            # as_tuple=False means we treat it as a torch tensor
            obj_indices = obj_mask.nonzero(as_tuple=False)
            # select the best candidate for a real target box
            best_box_indices = self._best_box_indices(pred_xy, pred_wh, target_xy, target_wh, obj_indices)
            responsible[obj_indices[:, 0], obj_indices[:, 1], obj_indices[:, 2], best_box_indices] = True

        coord_loss = F.mse_loss(pred_xy[responsible], target_xy[responsible], reduction="sum") + \
                    F.mse_loss(torch.sqrt(pred_wh[responsible].clamp(min=1e-6)), torch.sqrt(target_wh[responsible].clamp(min=1e-6)), reduction="sum")
        object_loss = F.mse_loss(pred_conf[responsible], target_conf[responsible], reduction="sum")
        noobject_loss = F.mse_loss(pred_conf[noobj_mask], target_conf[noobj_mask], reduction="sum")
        class_loss = F.mse_loss(pred_classes[obj_mask], target_classes[obj_mask], reduction="sum")

        batch_size = predictions.shape[0]
        return (self.lambda_coord * coord_loss + object_loss + self.lambda_noobj * noobject_loss + class_loss) / batch_size