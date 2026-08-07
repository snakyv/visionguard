from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--quantize", default=None)
    parser.add_argument("--data", default="datasets/demo/dataset.yaml")
    args = parser.parse_args()
    quantize = int(args.quantize) if args.quantize in {"8", "16", "32"} else args.quantize
    kwargs = {"format": "openvino", "imgsz": args.imgsz, "dynamic": True, "quantize": quantize}
    if quantize == 8:
        kwargs["data"] = args.data
    YOLO(args.model).export(**kwargs)


if __name__ == "__main__":
    main()
