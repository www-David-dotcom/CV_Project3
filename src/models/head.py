from __future__ import annotations
import torch
from torch import nn


class YOLOv1HEAD(nn.Module):
    def __init__(
            self,
            grid_size: int,
            boxes_per_cell: int,
            num_classes: int,
            dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        output_channels = boxes_per_cell * 5 + num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, output_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out.permute(0, 2, 3, 1)
