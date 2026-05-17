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
        output_dim = (grid_size ** 2) * (boxes_per_cell * 5 + num_classes)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
            nn.Flatten(),
            nn.Linear(1024 * (grid_size ** 2), 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        output_channels = self.boxes_per_cell * 5 + self.num_classes
        out = self.head(x)
        # .view means reshape the torch tensor to the below dimensions
        return out.view(batch_size, self.grid_size, self.grid_size, output_channels)