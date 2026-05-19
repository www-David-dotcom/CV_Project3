"""
Assuming the original data is the WIDERFACE format:
image/path.jpg
number_of_faces
x y width height blur expression illumination invalid occlusion pose
...

Convert it to txt files with:
class_id x y w h 
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.converters import write_yolo_labels, xyxy_pixels_to_yolo_line


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
        if face_count == 0:
            # WIDERFACE inserts a dummy all-zero bbox line after a 0 face count
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
