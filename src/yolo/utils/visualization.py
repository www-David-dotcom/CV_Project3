from __future__ import annotations
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import torch

def draw_boxes(
        image_bgr,
        boxes_xyxy: torch.Tensor,
        scores: torch.Tensor | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
):
    output = image_bgr.copy()
    height, width = output.shape[:2]

    for index, box in enumerate(boxes_xyxy):
        # these x and y are proportions (in [0, 1])
        x1, y1, x2, y2 = box.tolist()
        # upper-left of the ractangle
        p1 = (int(x1 * width, int(y1 * height)))
        # bottom-right of hte rectangle
        p2 = (int(x2 * width), int(y2 * height))
        cv2.rectangle(output, p1, p2, color, 2)
        if scores is not None:
            label = f"{float(scores[index]):.2f}"
            cv2.putText(output, label, p1, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)
    return output

def plot_training_history(csv_path: str | Path, output_path: str | Path) -> None:
    import csv

    epochs: list[int] = []
    losses: list[float] = []
    val_maps: list[float] = [] #map is mAP

    with Path(csv_path).open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            val_maps.append(float(row["map_59"]))
    
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, label="train loss")
    plt.plot(epochs, val_maps, label="mAP@0.5")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

