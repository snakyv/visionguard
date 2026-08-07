from __future__ import annotations

import argparse
import json

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="datasets/demo/dataset.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metrics = YOLO(args.model).val(data=args.data, device=args.device)
    payload = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    precision = payload["precision"]
    recall = payload["recall"]
    payload["f1"] = 2 * precision * recall / max(precision + recall, 1e-9)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
