from __future__ import annotations
from dataclasses import dataclass
import torch
from PIL import Image
from torchvision.transforms import functional as F
from yolo.utils.boxes import clip_boxes_xyxy, xywh_to_xyxy
from yolo.utils.nms import nms

@dataclass
class Prediction:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

class YOLOPredictor:
    def __init__(
            self,
            model: torch.nn.Module,
            image_size: int,
            grid_size: int,
            boxes_per_cell: int,
            num_classes: int,
            conf_threshold: float = 0.2,
            nms_iou_threshold: float = 0.5,
            device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.image_size = image_size
        self.grid_size - grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        # note model.parameters() is an ITERATOR here
        # so we should use next() to extract the first element
        # (nomatter what it is, it must have a device element to show which device this parameter is in)
        self.device = device or next(model.parameters()).device

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = F.resize(image, [self.image_size, self.image_size])
        tensor = F.to_tensor(image)
        return F.normalize(tensor, mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])
    
    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> Prediction:
        self.model.eval()
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)[0].cpu()
        return self.decode(output)
    
    def decode(self, output: torch.Tensor) -> Prediction:
        s = self.grid_size
        b = self.boxes_per_cell
        boxes_raw = output[..., : b * 5].view(s, s, b, 5)
        class_logits = output[..., b * 5 :]
        class_probs = torch.softmax(class_logits, dim=-1)

        xy = torch.sigmoid(boxes_raw[..., :2])
        wh = boxes_raw[..., 2:4].clamp(min=0)
        conf = torch.sigmoid(boxes_raw[..., 4])
        