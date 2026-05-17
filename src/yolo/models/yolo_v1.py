from __future__ import annotations
import torch
from torch import nn
from yolo.models.backbone import TinyYOLOBackbone
from yolo.models.head import YOLOv1HEAD

class YOLOv1(nn.Module):
    def __init__(
            self,
            grid_size: int = 7,
            boxes_per_cell: int = 2,
            num_classes: int = 1,
            dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.backbone = TinyYOLOBackbone()
        self.head = YOLOv1HEAD(grid_size, boxes_per_cell, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.head(self.backbone(x))