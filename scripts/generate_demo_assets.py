from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "demo"
WIDTH, HEIGHT = 640, 360
COLORS = {0: (40, 70, 230), 1: (20, 170, 240), 2: (220, 120, 60)}
NAMES = ["person", "forklift", "car"]


def draw_scene(index: int, total: int, noise: bool = True):
    rng = random.Random(index * 104729 + total)
    frame = np.full((HEIGHT, WIDTH, 3), (24, 28, 35), dtype=np.uint8)
    cv2.rectangle(frame, (0, 275), (WIDTH, HEIGHT), (42, 46, 52), -1)
    cv2.line(frame, (0, 275), (WIDTH, 275), (90, 96, 105), 2)
    cv2.rectangle(frame, (360, 80), (610, 330), (33, 38, 47), 2)
    cv2.putText(frame, "RESTRICTED", (430, 105), cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 90, 105), 1, cv2.LINE_AA)
    phase = index / max(total - 1, 1)
    boxes = []
    px = int(55 + phase * 470)
    py = int(205 + 18 * math.sin(phase * math.pi * 3))
    boxes.append((0, px, py, 28, 68))
    fx = int(570 - phase * 360)
    fy = int(245 + 8 * math.cos(phase * math.pi * 2))
    boxes.append((1, fx, fy, 78, 50))
    cx = int(80 + ((phase * 1.35) % 1.0) * 500)
    boxes.append((2, cx, 294, 82, 38))
    for class_id, x, y, w, h in boxes:
        x = max(0, min(WIDTH - w - 1, x))
        y = max(0, min(HEIGHT - h - 1, y))
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS[class_id], -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 245, 245), 1)
    if noise:
        for _ in range(20):
            x = rng.randrange(WIDTH); y = rng.randrange(HEIGHT)
            cv2.circle(frame, (x, y), rng.randrange(1, 4), (rng.randrange(35, 80),) * 3, -1)
    return frame, boxes


def yolo_line(class_id, x, y, w, h):
    cx = (x + w / 2) / WIDTH; cy = (y + h / 2) / HEIGHT
    return f"{class_id} {cx:.6f} {cy:.6f} {w / WIDTH:.6f} {h / HEIGHT:.6f}"


def generate_dataset():
    for split, count in (("train", 72), ("val", 18)):
        image_dir = DATA / "images" / split; label_dir = DATA / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            multiplier = 17 if split == "train" else 7
            idx = (i * multiplier) % count
            frame, boxes = draw_scene(idx, count)
            image_path = image_dir / f"scene_{i:04d}.jpg"
            label_path = label_dir / f"scene_{i:04d}.txt"
            cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            label_path.write_text("\n".join(yolo_line(*box) for box in boxes) + "\n", encoding="utf-8")
    config = {"path": ".", "train": "images/train", "val": "images/val", "names": {i: name for i, name in enumerate(NAMES)}}
    (DATA / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def generate_video():
    DATA.mkdir(parents=True, exist_ok=True)
    output = DATA / "demo_scene.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 24, (WIDTH, HEIGHT))
    for i in range(288):
        frame, _ = draw_scene(i, 288, noise=False)
        writer.write(frame)
    writer.release()


def main():
    generate_dataset(); generate_video(); print(f"Generated assets in {DATA}")


if __name__ == "__main__":
    main()
