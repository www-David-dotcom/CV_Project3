from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.face_dataset import FaceDetectionDataset, detection_collate
from engine.evaluator import evaluate_model
from models import YOLOv1
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device id, e.g. --gpu 0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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
