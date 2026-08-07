from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2

from visionguard.config import CameraDefinition, ROOT
from visionguard.factory import build_pipeline


def run_demo(source: str, output: str | None = None, max_frames: int | None = None) -> dict:
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    output_path = Path(output) if output else ROOT / "outputs" / "demo_annotated.mp4"
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    camera = CameraDefinition("demo", str(source_path), "synthetic", "centroid")
    pipeline = build_pipeline(camera)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames = 0
    events = 0
    inference_total = 0.0
    started = perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = pipeline.process(frame)
        writer.write(result.frame)
        frames += 1
        events += len(result.events)
        inference_total += result.inference_ms
        if max_frames and frames >= max_frames:
            break
    capture.release()
    writer.release()
    elapsed = perf_counter() - started
    return {
        "frames": frames,
        "events": events,
        "elapsed_seconds": round(elapsed, 3),
        "processing_fps": round(frames / max(elapsed, 1e-6), 2),
        "avg_detection_ms": round(inference_total / max(frames, 1), 3),
        "output": str(output_path),
    }
