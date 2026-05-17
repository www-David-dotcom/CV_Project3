# YOLO Training Code Plan

This file is a handoff document for implementing a compact YOLOv1-style face detector yourself. It keeps the actual repository files untouched except for this guide.

The design targets:

- PyTorch implementation from scratch
- Single-class face detection
- `uv` for environment and command execution
- YOLOv1-style grid prediction, loss, decode, NMS, training, evaluation, and plotting

## 1. Target File Structure

```text
YOLO-Training/
├── .python-version
├── pyproject.toml
├── uv.lock
├── README.md
├── PLANCODE.md
├── configs/
│   ├── dataset.yaml
│   └── yolo_v1_face.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── README.md
├── src/
│   └── yolo/
│       ├── __init__.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── converters.py
│       │   ├── face_dataset.py
│       │   └── transforms.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   └── trainer.py
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predictor.py
│       ├── losses/
│       │   ├── __init__.py
│       │   └── yolo_v1_loss.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── backbone.py
│       │   ├── head.py
│       │   └── yolo_v1.py
│       └── utils/
│           ├── __init__.py
│           ├── boxes.py
│           ├── checkpoint.py
│           ├── config.py
│           ├── metrics.py
│           ├── nms.py
│           └── visualization.py
├── scripts/
│   ├── evaluate.py
│   ├── infer.py
│   ├── plot_curves.py
│   ├── prepare_widerface.py
│   └── train.py
├── tests/
│   ├── test_boxes.py
│   ├── test_dataset.py
│   └── test_loss.py
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── predictions/
└── report/
    ├── report.md
    └── figures/
```

## 9. Inference Code

### `src/yolo/inference/predictor.py`

```python
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
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device or next(model.parameters()).device

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

        xy = torch.sigmoid(boxes_raw[..., 0:2])
        wh = boxes_raw[..., 2:4].clamp(min=0)
        conf = torch.sigmoid(boxes_raw[..., 4])

        y_indices, x_indices = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
        grid = torch.stack((x_indices, y_indices), dim=-1).float().unsqueeze(2)
        centers = (xy + grid) / s
        boxes_xyxy = clip_boxes_xyxy(xywh_to_xyxy(torch.cat((centers, wh), dim=-1)))

        scores_per_class = conf.unsqueeze(-1) * class_probs.unsqueeze(2)
        scores, labels = scores_per_class.max(dim=-1)

        boxes_flat = boxes_xyxy.reshape(-1, 4)
        scores_flat = scores.reshape(-1)
        labels_flat = labels.reshape(-1)
        keep_conf = scores_flat >= self.conf_threshold

        boxes_flat = boxes_flat[keep_conf]
        scores_flat = scores_flat[keep_conf]
        labels_flat = labels_flat[keep_conf]

        keep_all = []
        for class_id in labels_flat.unique():
            class_mask = labels_flat == class_id
            class_indices = torch.where(class_mask)[0]
            keep = nms(boxes_flat[class_mask], scores_flat[class_mask], self.nms_iou_threshold)
            keep_all.append(class_indices[keep])

        if keep_all:
            keep_indices = torch.cat(keep_all)
        else:
            keep_indices = torch.empty((0,), dtype=torch.long)

        return Prediction(
            boxes=boxes_flat[keep_indices],
            scores=scores_flat[keep_indices],
            labels=labels_flat[keep_indices],
        )

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = F.resize(image, [self.image_size, self.image_size])
        tensor = F.to_tensor(image)
        return F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

## 10. Training and Evaluation Engine

### `src/yolo/engine/trainer.py`

```python
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo.engine.evaluator import evaluate_model
from yolo.utils.checkpoint import save_checkpoint


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, targets, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.shape[0]

    return total_loss / len(loader.dataset)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_dir: str | Path,
    log_path: str | Path,
    eval_config: dict,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_map = -1.0

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "map_50", "map_70", "map_90"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            metrics = evaluate_model(model, val_loader, device, eval_config)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "map_50": metrics.get("map_50", 0.0),
                "map_70": metrics.get("map_70", 0.0),
                "map_90": metrics.get("map_90", 0.0),
            }
            writer.writerow(row)
            file.flush()

            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row)
            if row["map_50"] > best_map:
                best_map = row["map_50"]
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row)

            print(
                f"epoch={epoch} loss={train_loss:.4f} "
                f"mAP@0.5={row['map_50']:.4f} mAP@0.7={row['map_70']:.4f} mAP@0.9={row['map_90']:.4f}"
            )
```

### `src/yolo/engine/evaluator.py`

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolo.inference.predictor import YOLOPredictor
from yolo.utils.boxes import xywh_to_xyxy
from yolo.utils.metrics import precision_recall_ap


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_config: dict,
) -> dict[str, float]:
    dataset = loader.dataset
    predictor = YOLOPredictor(
        model=model,
        image_size=dataset.transform.image_size,
        grid_size=dataset.grid_size,
        boxes_per_cell=dataset.boxes_per_cell,
        num_classes=dataset.num_classes,
        conf_threshold=eval_config.get("conf_threshold", 0.2),
        nms_iou_threshold=eval_config.get("nms_iou_threshold", 0.5),
        device=device,
    )

    pred_boxes: list[torch.Tensor] = []
    pred_scores: list[torch.Tensor] = []
    target_boxes: list[torch.Tensor] = []

    model.eval()
    for images, _, raw_targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        outputs = model(images).cpu()

        for output, raw in zip(outputs, raw_targets, strict=True):
            prediction = predictor.decode(output)
            pred_boxes.append(prediction.boxes)
            pred_scores.append(prediction.scores)
            if raw.numel() == 0:
                target_boxes.append(torch.zeros((0, 4), dtype=torch.float32))
            else:
                target_boxes.append(xywh_to_xyxy(raw[:, 1:5]))

    result: dict[str, float] = {}
    thresholds = eval_config.get("map_iou_thresholds", [0.5, 0.7, 0.9])
    for threshold in thresholds:
        pr = precision_recall_ap(pred_boxes, pred_scores, target_boxes, float(threshold))
        key = str(int(round(float(threshold) * 100)))
        result[f"map_{key}"] = float(pr["ap"])

    coco_thresholds = [round(0.5 + 0.05 * index, 2) for index in range(10)]
    coco_aps = [
        float(precision_recall_ap(pred_boxes, pred_scores, target_boxes, threshold)["ap"])
        for threshold in coco_thresholds
    ]
    result["map_50_95"] = sum(coco_aps) / len(coco_aps)
    return result
```

## 11. Scripts

### `scripts/train.py`

```python
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from yolo.datasets.face_dataset import FaceDetectionDataset, detection_collate
from yolo.engine.trainer import train_model
from yolo.losses import YOLOv1Loss
from yolo.models import YOLOv1
from yolo.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))

    dataset_config = config["dataset"]
    model_config = config["model"]
    train_config = config["train"]

    train_dataset = FaceDetectionDataset(
        image_dir=dataset_config["train_images"],
        label_dir=dataset_config["train_labels"],
        image_size=model_config["image_size"],
        grid_size=model_config["grid_size"],
        boxes_per_cell=model_config["boxes_per_cell"],
        num_classes=dataset_config["num_classes"],
        train=True,
    )
    val_dataset = FaceDetectionDataset(
        image_dir=dataset_config["val_images"],
        label_dir=dataset_config["val_labels"],
        image_size=model_config["image_size"],
        grid_size=model_config["grid_size"],
        boxes_per_cell=model_config["boxes_per_cell"],
        num_classes=dataset_config["num_classes"],
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config["num_workers"],
        collate_fn=detection_collate,
    )

    model = YOLOv1(
        grid_size=model_config["grid_size"],
        boxes_per_cell=model_config["boxes_per_cell"],
        num_classes=dataset_config["num_classes"],
        dropout=model_config["dropout"],
    ).to(device)
    loss_fn = YOLOv1Loss(
        grid_size=model_config["grid_size"],
        boxes_per_cell=model_config["boxes_per_cell"],
        num_classes=dataset_config["num_classes"],
        lambda_coord=config["loss"]["lambda_coord"],
        lambda_noobj=config["loss"]["lambda_noobj"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=train_config["epochs"],
        checkpoint_dir=train_config["checkpoint_dir"],
        log_path=train_config["log_path"],
        eval_config=config["eval"],
    )


if __name__ == "__main__":
    main()
```

### `scripts/evaluate.py`

```python
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from yolo.datasets.face_dataset import FaceDetectionDataset, detection_collate
from yolo.engine.evaluator import evaluate_model
from yolo.models import YOLOv1
from yolo.utils.checkpoint import load_checkpoint
from yolo.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_config = config["dataset"]
    model_config = config["model"]

    val_dataset = FaceDetectionDataset(
        image_dir=dataset_config["val_images"],
        label_dir=dataset_config["val_labels"],
        image_size=model_config["image_size"],
        grid_size=model_config["grid_size"],
        boxes_per_cell=model_config["boxes_per_cell"],
        num_classes=dataset_config["num_classes"],
        train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=detection_collate,
    )

    model = YOLOv1(
        grid_size=model_config["grid_size"],
        boxes_per_cell=model_config["boxes_per_cell"],
        num_classes=dataset_config["num_classes"],
        dropout=model_config["dropout"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device)
    metrics = evaluate_model(model, val_loader, device, config["eval"])
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
```

### `scripts/infer.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

from yolo.inference import YOLOPredictor
from yolo.models import YOLOv1
from yolo.utils.checkpoint import load_checkpoint
from yolo.utils.config import load_config
from yolo.utils.visualization import draw_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/predictions/infer.jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    dataset_config = config["dataset"]

    model = YOLOv1(
        grid_size=model_config["grid_size"],
        boxes_per_cell=model_config["boxes_per_cell"],
        num_classes=dataset_config["num_classes"],
        dropout=model_config["dropout"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device)

    predictor = YOLOPredictor(
        model=model,
        image_size=model_config["image_size"],
        grid_size=model_config["grid_size"],
        boxes_per_cell=model_config["boxes_per_cell"],
        num_classes=dataset_config["num_classes"],
        conf_threshold=config["eval"]["conf_threshold"],
        nms_iou_threshold=config["eval"]["nms_iou_threshold"],
        device=device,
    )

    image = Image.open(args.image)
    prediction = predictor.predict_image(image)
    image_bgr = cv2.imread(args.image)
    output = draw_boxes(image_bgr, prediction.boxes, prediction.scores)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
```

### `scripts/plot_curves.py`

```python
from __future__ import annotations

import argparse

from yolo.utils.visualization import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="outputs/logs/train_history.csv")
    parser.add_argument("--output", default="outputs/figures/training_curve.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_training_history(args.history, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
```

### `scripts/prepare_widerface.py`

This parser assumes the official WIDER FACE annotation format:

```text
image/path.jpg
number_of_faces
x y width height blur expression illumination invalid occlusion pose
...
```

```python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from yolo.datasets.converters import write_yolo_labels, xyxy_pixels_to_yolo_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--output-images", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--skip-invalid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_root = Path(args.images_root)
    output_images = Path(args.output_images)
    output_labels = Path(args.output_labels)
    lines = Path(args.annotation_file).read_text(encoding="utf-8").splitlines()

    index = 0
    converted = 0
    while index < len(lines):
        rel_image = lines[index].strip()
        index += 1
        if not rel_image:
            continue

        face_count = int(lines[index].strip())
        index += 1
        image_path = images_root / rel_image
        target_image_path = output_images / rel_image
        label_path = output_labels / Path(rel_image).with_suffix(".txt")

        with Image.open(image_path) as image:
            width, height = image.size

        label_lines: list[str] = []
        for _ in range(face_count):
            values = [int(value) for value in lines[index].split()]
            index += 1
            x, y, w, h = values[:4]
            invalid = values[7] if len(values) > 7 else 0
            if args.skip_invalid and invalid:
                continue
            if w <= 0 or h <= 0:
                continue
            label_lines.append(
                xyxy_pixels_to_yolo_line(
                    class_id=0,
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    image_width=width,
                    image_height=height,
                )
            )

        target_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image_path)
        write_yolo_labels(label_path, label_lines)
        converted += 1

    print(f"converted {converted} images")


if __name__ == "__main__":
    main()
```

## 12. Tests

### `tests/test_boxes.py`

```python
import torch

from yolo.utils.boxes import box_iou_xyxy, xywh_to_xyxy, xyxy_to_xywh
from yolo.utils.nms import nms


def test_xywh_xyxy_round_trip():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    converted = xywh_to_xyxy(boxes)
    restored = xyxy_to_xywh(converted)
    assert torch.allclose(restored, boxes)


def test_iou_identical_boxes_is_one():
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    iou = box_iou_xyxy(boxes, boxes)
    assert torch.allclose(iou, torch.tensor([[1.0]]))


def test_nms_keeps_highest_scoring_overlap():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.05, 0.05, 0.95, 0.95],
            [2.0, 2.0, 3.0, 3.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]
```

### `tests/test_dataset.py`

```python
from pathlib import Path

from PIL import Image

from yolo.datasets.face_dataset import FaceDetectionDataset


def test_dataset_encodes_one_face(tmp_path: Path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (64, 64), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    dataset = FaceDetectionDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=448,
        grid_size=7,
        boxes_per_cell=2,
        num_classes=1,
        train=False,
    )
    image, target, raw_boxes = dataset[0]

    assert image.shape == (3, 448, 448)
    assert raw_boxes.shape == (1, 5)
    assert target.shape == (7, 7, 11)
    assert target[3, 3, 4].item() == 1.0
    assert target[3, 3, 10].item() == 1.0
```

### `tests/test_loss.py`

```python
import torch

from yolo.losses import YOLOv1Loss


def test_yolo_loss_returns_scalar():
    loss_fn = YOLOv1Loss(grid_size=7, boxes_per_cell=2, num_classes=1)
    predictions = torch.randn(2, 7, 7, 11)
    targets = torch.zeros(2, 7, 7, 11)
    targets[:, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.2, 0.2, 1.0])
    targets[:, 3, 3, 5:10] = torch.tensor([0.5, 0.5, 0.2, 0.2, 1.0])
    targets[:, 3, 3, 10] = 1.0

    loss = loss_fn(predictions, targets)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
```

## 13. Implementation Order

1. Create directories.
2. Add `.python-version`, configs, package `__init__.py` files.
3. Implement `utils/boxes.py` and `utils/nms.py`.
4. Add and run `tests/test_boxes.py`.
5. Implement dataset files.
6. Add and run `tests/test_dataset.py`.
7. Implement model files.
8. Implement loss.
9. Add and run `tests/test_loss.py`.
10. Implement predictor, evaluator, trainer, and scripts.
11. Convert a small dataset subset.
12. Run a smoke training pass with 2 to 5 images.
13. Train on the selected dataset split.
14. Generate loss curve, mAP metrics, PR curve data, and inference examples for the report.

## 14. Suggested Verification Commands

```bash
uv sync
uv run pytest
uv run python scripts/train.py --config configs/yolo_v1_face.yaml
uv run python scripts/evaluate.py --config configs/yolo_v1_face.yaml --checkpoint outputs/checkpoints/best.pt
uv run python scripts/plot_curves.py
```

## 15. Notes Before Implementation

- This is intentionally a small YOLOv1-style detector, not a reproduction of YOLOv5/v8/v11.
- The model is likely to train slowly on CPU; use a small subset first.
- If memory is tight, reduce `batch_size` to `2` or `4`.
- If the loss is unstable, lower `learning_rate` to `0.00005`.
- If your face dataset has many tiny faces, YOLOv1's `7x7` grid will struggle. You can later try `grid_size: 14`, but the fully connected head becomes much larger.
