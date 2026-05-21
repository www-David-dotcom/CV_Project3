from __future__ import annotations
from dataclasses import dataclass
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F

@dataclass # this decorator will generate the __init__ method automatically
class YOLOTransform:
    image_size: int
    train: bool = True # turn on image augmentation when training
    hflip_prob: float = 0.5
    affine_prob: float = 0.5
    color_jitter_prob: float = 0.8
    grayscale_prob: float = 0.1
    blur_prob: float = 0.1
    cutout_prob: float = 0.2

    def __call__(self, image: Image.Image, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        image = F.resize(image, [self.image_size, self.image_size])

        if self.train:
            image, boxes = self._augment(image, boxes)

        image_tensor = F.to_tensor(image)
        if self.train and torch.rand(()) < self.cutout_prob:
            image_tensor = self._random_cutout(image_tensor)
        # the mean and std is the pretrained mean and std_dev of imagenet
        image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image_tensor, boxes

    def _augment(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        if torch.rand(()) < self.hflip_prob:
            image = F.hflip(image)
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, 1] = 1.0 - boxes[:, 1]

        if torch.rand(()) < self.affine_prob:
            image, boxes = self._random_affine(image, boxes)

        if torch.rand(()) < self.color_jitter_prob:
            brightness = float(torch.empty(()).uniform_(0.75, 1.25).item())
            contrast = float(torch.empty(()).uniform_(0.75, 1.25).item())
            saturation = float(torch.empty(()).uniform_(0.75, 1.25).item())
            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)
            image = F.adjust_saturation(image, saturation)

        if torch.rand(()) < self.grayscale_prob:
            image = F.rgb_to_grayscale(image, num_output_channels=3)

        if torch.rand(()) < self.blur_prob:
            radius = float(torch.empty(()).uniform_(0.1, 1.2).item())
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image, boxes

    def _random_affine(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        scale = float(torch.empty(()).uniform_(0.85, 1.15).item())
        max_translate = int(round(self.image_size * 0.08))
        tx = int(torch.randint(-max_translate, max_translate + 1, ()).item())
        ty = int(torch.randint(-max_translate, max_translate + 1, ()).item())

        image = F.affine(
            image,
            angle=0.0,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            fill=[114, 114, 114],
        )
        if boxes.numel() == 0:
            return image, boxes

        boxes = boxes.clone()
        boxes[:, 1] = (boxes[:, 1] - 0.5) * scale + 0.5 + tx / self.image_size
        boxes[:, 2] = (boxes[:, 2] - 0.5) * scale + 0.5 + ty / self.image_size
        boxes[:, 3] = boxes[:, 3] * scale
        boxes[:, 4] = boxes[:, 4] * scale
        boxes = self._clip_boxes(boxes)
        return image, boxes

    @staticmethod
    def _clip_boxes(boxes: torch.Tensor) -> torch.Tensor:
        x1 = (boxes[:, 1] - boxes[:, 3] / 2).clamp(0.0, 1.0)
        y1 = (boxes[:, 2] - boxes[:, 4] / 2).clamp(0.0, 1.0)
        x2 = (boxes[:, 1] + boxes[:, 3] / 2).clamp(0.0, 1.0)
        y2 = (boxes[:, 2] + boxes[:, 4] / 2).clamp(0.0, 1.0)

        widths = x2 - x1
        heights = y2 - y1
        keep = (widths > 1e-4) & (heights > 1e-4)
        if not keep.any():
            return boxes.new_zeros((0, 5))

        clipped = boxes[keep].clone()
        clipped[:, 1] = (x1[keep] + x2[keep]) / 2
        clipped[:, 2] = (y1[keep] + y2[keep]) / 2
        clipped[:, 3] = widths[keep]
        clipped[:, 4] = heights[keep]
        return clipped

    @staticmethod
    def _random_cutout(image_tensor: torch.Tensor) -> torch.Tensor:
        _, height, width = image_tensor.shape
        erase_w = int(width * float(torch.empty(()).uniform_(0.05, 0.18).item()))
        erase_h = int(height * float(torch.empty(()).uniform_(0.05, 0.18).item()))
        if erase_w <= 0 or erase_h <= 0:
            return image_tensor

        x1 = int(torch.randint(0, max(width - erase_w + 1, 1), ()).item())
        y1 = int(torch.randint(0, max(height - erase_h + 1, 1), ()).item())
        image_tensor = image_tensor.clone()
        image_tensor[:, y1:y1 + erase_h, x1:x1 + erase_w] = 0.45
        return image_tensor
