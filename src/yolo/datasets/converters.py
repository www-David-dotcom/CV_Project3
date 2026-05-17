from __future__ import annotations
from pathlib import Path

def xyxy_pixels_to_yolo_line(
        class_id: int,
        x1: float, y1: float, x2: float, y2: float,
        image_width: int, image_height: int,
) -> str:
    """
    turn one index into a line in the txt file:
    class_id, x_center, y_center, width, height
    """
    x_center = ((x1 + x2) / 2) / image_width
    y_center = ((y1 + y2) / 2) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    values = [class_id, x_center, y_center, width, height]
    return " ".join(str(round(value, 6)) for value in values)

def write_yolo_labels(path: str | Path, lines: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # remember to append \n at the last line
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")