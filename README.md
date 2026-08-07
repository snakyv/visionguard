<div align="center">

# VisionGuard

### Real-Time Multi-Camera Video Analytics Platform

Production-oriented computer vision pipeline for object detection, multi-object tracking, event analytics and optimized edge inference.

**Python 3.12 · YOLO · PyTorch · OpenCV · ByteTrack · ONNX · OpenVINO · NNCF · FastAPI · Docker**

</div>

---

## Overview

VisionGuard is a portfolio-grade video analytics platform designed around the same problems found in production surveillance and industrial vision systems: continuous video ingestion, object detection, persistent identities, geometric analytics, event generation, model evaluation and deployment optimization.

The repository includes a fully offline demo mode with a bundled synthetic YOLO-format dataset and generated video, so the complete pipeline can be tested immediately after cloning. The production mode swaps in Ultralytics YOLO and ByteTrack without changing the surrounding analytics architecture.

### Core pipeline

```text
Video / RTSP / Webcam
        │
        ▼
   Frame Decoder
        │
        ▼
 Object Detection ───────────── Synthetic smoke-test backend
        │                       YOLO production backend
        ▼
 Multi-Object Tracking ──────── Centroid smoke-test tracker
        │                       ByteTrack production tracker
        ▼
 Event Analytics
   ├─ Zone enter / exit
   ├─ Dwell time
   ├─ Line crossing
   └─ Person–forklift near-miss
        │
        ▼
 FastAPI + MJPEG Dashboard + Metrics
        │
        ▼
 ONNX / OpenVINO FP16 / INT8 deployment workflow
```

## Why this project is different

VisionGuard is intentionally more than a `YOLO.predict()` demo. Detection and tracking are only one layer. The project also includes dataset validation, reproducible training entry points, event logic, multi-camera runtime primitives, live API endpoints, inference benchmarking, OpenVINO export and quantization workflows.

## Features

- Pluggable detection backends: synthetic reproducible backend and Ultralytics YOLO.
- Pluggable trackers: deterministic centroid tracker and current Roboflow `trackers` ByteTrack implementation.
- YOLO-format custom dataset bundled for smoke tests.
- Dataset integrity validator.
- Zone enter, zone exit and dwell events.
- Line crossing analytics.
- Person–forklift near-miss heuristic based on track distance and motion trend.
- Track trajectories and annotated video rendering.
- FastAPI REST API and MJPEG live preview.
- Multi-camera worker runtime.
- PyTorch/YOLO → ONNX export.
- YOLO → OpenVINO FP32/FP16/INT8 export.
- NNCF-backed INT8 workflow through current Ultralytics/OpenVINO export APIs.
- Latency, throughput and CPU-time benchmark utility.
- Docker deployment.
- GitHub Actions CI for Python 3.12 and 3.13.

## Quick start in PyCharm

### 1. Open the project

Open the repository root as a project in PyCharm and select a **Python 3.12** interpreter.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

For the complete CV stack:

```powershell
pip install -r requirements.txt
```

For the fast offline demo only:

```powershell
pip install -r requirements-demo.txt
```

### 4. Run the tests

```powershell
pytest
```

### 5. Run the offline video analytics demo

```powershell
python -m visionguard demo --source datasets/demo/demo_scene.mp4
```

Output:

```text
outputs/demo_annotated.mp4
```

### 6. Launch the API and dashboard

```powershell
python -m visionguard serve
```

Open:

```text
http://127.0.0.1:8000
```

Press **Start demo**. The dashboard will show the annotated live feed, runtime metrics and generated analytics events.

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Bundled dataset

The repository ships with a small generated dataset under `datasets/demo/`:

```text
datasets/demo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── dataset.yaml
└── demo_scene.mp4
```

Classes:

| ID | Class |
|---:|---|
| 0 | person |
| 1 | forklift |
| 2 | car |

Validate the dataset:

```powershell
python training/dataset_validator.py --data datasets/demo/dataset.yaml
```

The bundled data is deliberately synthetic. It exists to make testing reproducible, not to claim real-world accuracy. For a serious portfolio experiment, add a legally usable real dataset, review annotations manually and record error analysis and metrics.

## Training a custom YOLO model

Install the full dependency set, then run:

```powershell
python training/train.py ^
  --model yolo26n.pt ^
  --data datasets/demo/dataset.yaml ^
  --epochs 50 ^
  --device cpu
```

On an NVIDIA environment, use an appropriate CUDA device instead of `cpu`.

Evaluate a trained checkpoint:

```powershell
python training/evaluate.py ^
  --model runs/train/visionguard/weights/best.pt ^
  --data datasets/demo/dataset.yaml
```

The evaluator reports precision, recall, F1, mAP50 and mAP50-95.

## Production YOLO + ByteTrack mode

Edit `configs/cameras.yaml`:

```yaml
cameras:
  - id: warehouse_01
    source: rtsp://user:password@camera/stream
    backend: yolo
    tracker: bytetrack
    model: models/best.pt
    confidence: 0.35
    device: cpu
    enabled: true
```

The runtime will use the YOLO detector and ByteTrack while keeping the same analytics and API layers. A pretrained COCO model does not provide a `forklift` class, so use a custom fine-tuned checkpoint when forklift analytics is required.

## Analytics configuration

`configs/analytics.yaml` contains zones, lines and near-miss thresholds:

```yaml
zones:
  - name: restricted_zone
    points:
      - [360, 80]
      - [610, 80]
      - [610, 330]
      - [360, 330]
    classes: [person]
    dwell_seconds: 2.0
```

Rules are intentionally independent from the detection backend.

## Export to ONNX

```powershell
python optimization/export_onnx.py ^
  --model runs/train/visionguard/weights/best.pt
```

FP16 export:

```powershell
python optimization/export_onnx.py ^
  --model runs/train/visionguard/weights/best.pt ^
  --quantize 16
```

## Export to OpenVINO

FP32:

```powershell
python optimization/export_openvino.py ^
  --model runs/train/visionguard/weights/best.pt
```

FP16:

```powershell
python optimization/export_openvino.py ^
  --model runs/train/visionguard/weights/best.pt ^
  --quantize 16
```

INT8 post-training quantization:

```powershell
python optimization/export_openvino.py ^
  --model runs/train/visionguard/weights/best.pt ^
  --quantize 8 ^
  --data datasets/demo/dataset.yaml
```

Current Ultralytics OpenVINO export uses NNCF for INT8 post-training quantization with representative calibration data.

## Benchmarking

Compare PyTorch, ONNX and OpenVINO artifacts with the same source video:

```powershell
python optimization/benchmark.py ^
  --model path/to/model ^
  --source datasets/demo/demo_scene.mp4 ^
  --frames 200 ^
  --output benchmarks/result.json
```

Record at least:

- average latency;
- p95 latency;
- throughput FPS;
- precision format;
- mAP50 and mAP50-95;
- CPU/GPU device;
- input resolution;
- number of concurrent streams.

Do not publish invented performance numbers. Benchmark results belong to a specific model, input size and hardware configuration.

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness and version |
| GET | `/api/v1/cameras` | Camera states and metrics |
| POST | `/api/v1/cameras/{id}/start` | Start worker |
| POST | `/api/v1/cameras/{id}/stop` | Stop worker |
| GET | `/api/v1/cameras/{id}/mjpeg` | Annotated MJPEG stream |
| GET | `/api/v1/events` | Recent analytics events |
| GET | `/api/v1/metrics` | Runtime metrics |
| GET | `/api/v1/models` | Local model registry |
| WS | `/ws/events` | Live analytics event stream |

## Repository structure

```text
VisionGuard/
├── src/visionguard/
│   ├── analytics/
│   ├── detection/
│   ├── tracking/
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── demo.py
│   ├── domain.py
│   ├── factory.py
│   ├── pipeline.py
│   ├── rendering.py
│   └── runtime.py
├── training/
├── optimization/
├── datasets/
├── configs/
├── models/
├── benchmarks/
├── tests/
├── web/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Verification

The core repository has automated tests plus an end-to-end offline smoke pipeline. The exact build-time verification record and the external-package testing boundary are documented in [`TESTING.md`](TESTING.md).

Current dependency baseline targets Python 3.12 with FastAPI 0.140.x, Ultralytics 8.4.x, Supervision 0.29.x, Roboflow Trackers 2.5.x and OpenVINO 2026.2.1.

## Docker

```bash
docker compose up --build
```

The dashboard will be exposed on port `8000`.

## Recommended portfolio extension

The codebase is ready for the next production-oriented steps:

1. Replace the synthetic dataset with a documented real-world dataset.
2. Fine-tune YOLO and keep experiment metadata.
3. Produce a false-positive / false-negative analysis.
4. Export the best checkpoint to OpenVINO FP16 and INT8.
5. Benchmark PyTorch vs ONNX vs OpenVINO on the same Intel hardware.
6. Add real RTSP sources and test multiple concurrent streams.
7. Calibrate pixel distances to world coordinates for physically meaningful near-miss TTC calculations.
8. Add persistent event storage and authentication if deploying beyond a portfolio environment.

## Engineering notes

- Python requirement: `>=3.12,<3.15`.
- OpenVINO is pinned to `2026.2.1` in the optional CV dependency set.
- YOLO weights are not committed and are downloaded or produced by training.
- RTSP credentials must be stored outside Git.
- The demo backend is deterministic and intended for CI/smoke testing.
- The near-miss algorithm is a pixel-space portfolio heuristic, not a certified safety system.

## License

VisionGuard source code is MIT licensed. See `NOTICE.md` for third-party dependency licensing, especially Ultralytics.
