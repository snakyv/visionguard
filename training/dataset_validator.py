from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import yaml


def validate(dataset_yaml: str) -> tuple[list[str], Counter[int]]:
    config_path = Path(dataset_yaml).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = Path(data.get("path", config_path.parent))
    if not base.is_absolute():
        base = (config_path.parent / base).resolve()
    errors: list[str] = []
    counts: Counter[int] = Counter()
    for split in ("train", "val"):
        image_dir = base / data[split]
        label_dir = base / "labels" / split
        for image_path in sorted(image_dir.glob("*")):
            image = cv2.imread(str(image_path))
            if image is None:
                errors.append(f"Unreadable image: {image_path}")
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                errors.append(f"Missing label: {label_path}")
                continue
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Invalid label columns: {label_path}:{line_number}")
                    continue
                class_id = int(parts[0])
                values = [float(v) for v in parts[1:]]
                if any(value < 0 or value > 1 for value in values):
                    errors.append(f"Out-of-range label: {label_path}:{line_number}")
                counts[class_id] += 1
    return errors, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/demo/dataset.yaml")
    args = parser.parse_args()
    errors, counts = validate(args.data)
    print(f"Objects by class: {dict(counts)}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("Dataset validation passed")


if __name__ == "__main__":
    main()
