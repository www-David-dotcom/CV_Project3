from __future__ import annotations
from dataclasses import dataclass
import torch
from PIL import Image
from torchvision.transforms import functional as F

@dataclass # this decorator will generate the __init__ method automatically
class YOLOTransform:
    image_size: int
    train: bool = True # turn on image augmentation when training

    def __call__(self, image: Image.Image, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        image = F.resize(image, [self.image_size, self.image_size])

        if self.train and torch.rand(()) < 0.5:
            # hflip with probability 0.5
            image = F.hflip(image)
            if boxes.numel() > 0:
                boxes = boxes.clone()
                # the [:, 1] is the x-center of a box (xywh)
                # as box is horizontal flipped, you should turn x to 1 - x
                boxes[:, 1] = 1.0 - boxes[:, 1]

        image_tensor = F.to_tensor(image)
        # the mean and std is the pretrained mean and std_dev of imagenet
        image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image_tensor, boxes