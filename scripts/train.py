from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.face_dataset import FaceDetectionDataset, detection_collate
from engine.trainer import train_model
from losses import YOLOv1Loss
from models import YOLOv1
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device id, e.g. --gpu 0")
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
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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
        image_size=model_config["image_size"],
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

    scheduler_config = train_config.get("lr_scheduler", {})
    scheduler = None
    if scheduler_config.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config["epochs"],
            eta_min=scheduler_config.get("eta_min", 0),
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
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
