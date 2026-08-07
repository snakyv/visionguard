from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import psutil
from ultralytics import YOLO


def benchmark(model_path: str, source: str, frames: int, device: str | None) -> dict:
    model = YOLO(model_path)
    capture = cv2.VideoCapture(source)
    timings: list[float] = []
    processed = 0
    process = psutil.Process()
    cpu_before = process.cpu_times()
    started = perf_counter()
    while processed < frames:
        ok, frame = capture.read()
        if not ok:
            break
        t0 = perf_counter()
        model.predict(frame, device=device, verbose=False)
        timings.append((perf_counter() - t0) * 1000)
        processed += 1
    elapsed = perf_counter() - started
    cpu_after = process.cpu_times()
    capture.release()
    return {
        "model": model_path,
        "frames": processed,
        "throughput_fps": round(processed / max(elapsed, 1e-6), 2),
        "avg_latency_ms": round(sum(timings) / max(len(timings), 1), 2),
        "p95_latency_ms": round(sorted(timings)[int(len(timings) * 0.95) - 1], 2) if timings else 0,
        "cpu_time_seconds": round((cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="datasets/demo/demo_scene.mp4")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = benchmark(args.model, args.source, args.frames, args.device)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
