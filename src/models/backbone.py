from __future__ import annotations
import torch 
from torch import nn

def conv_block(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )

class TinyYOLOBackbone(nn.Module):
    def __init__(
            self,
            image_size: int = 448,
            grid_size: int = 14,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        output_stride = image_size // grid_size
        if image_size % grid_size != 0 or output_stride not in {16, 32}:
            raise ValueError(
                "TinyYOLOBackbone supports grid sizes that require output stride "
                f"16 or 32, got image_size={image_size}, grid_size={grid_size}."
            )

        layers: list[nn.Module] = [
            conv_block(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_block(32, 64, 3),
            nn.MaxPool2d(2, 2),
            conv_block(64, 128, 3),
            conv_block(128, 64, 1),
            conv_block(64, 128, 3),
            nn.MaxPool2d(2, 2),
            conv_block(128, 256, 3),
            conv_block(256, 128, 1),
            conv_block(128, 256, 3),
            nn.MaxPool2d(2, 2),
            conv_block(256, 512, 3),
            conv_block(512, 256, 1),
            conv_block(256, 512, 3),
            conv_block(512, 256, 1),
            conv_block(256, 512, 3),
        ]
        if output_stride == 32:
            layers.append(nn.MaxPool2d(2, 2))
        layers.extend([
            conv_block(512, 1024, 3),
            conv_block(1024, 512, 1),
            conv_block(512, 1024, 3),
            conv_block(1024, 512, 1),
            conv_block(512, 1024, 3),
        ])
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.features = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.features(x)
