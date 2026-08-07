# Models

Model weights are intentionally excluded from Git. Ultralytics downloads configured pretrained weights on first use.

Recommended workflow:

1. Fine-tune a YOLO model with `training/train.py`.
2. Copy the best checkpoint to this directory.
3. Export ONNX with `optimization/export_onnx.py`.
4. Export OpenVINO FP16/INT8 with `optimization/export_openvino.py`.
5. Store benchmark JSON results under `benchmarks/`.
