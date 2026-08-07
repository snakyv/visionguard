from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--quantize", default=None)
    args = parser.parse_args()
    quantize = int(args.quantize) if args.quantize in {"8", "16", "32"} else args.quantize
    YOLO(args.model).export(format="onnx", imgsz=args.imgsz, dynamic=True, quantize=quantize)


if __name__ == "__main__":
    main()
