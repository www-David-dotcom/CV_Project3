from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference import YOLOPredictor
from models import YOLOv1
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.visualization import draw_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/predictions/infer.jpg")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device id, e.g. --gpu 0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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
