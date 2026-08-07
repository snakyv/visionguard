from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from visionguard.config import CameraDefinition
from visionguard.domain import AnalyticsEvent
from visionguard.factory import build_pipeline


@dataclass(slots=True)
class CameraMetrics:
    frames: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    events: int = 0


class CameraWorker:
    def __init__(self, camera: CameraDefinition, on_event) -> None:
        self.camera = camera
        self.pipeline = None
        self.on_event = on_event
        self.metrics = CameraMetrics()
        self._latest_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.last_error = None
        if self.pipeline is None:
            self.pipeline = build_pipeline(self.camera)
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.camera.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def _run(self) -> None:
        if self.pipeline is None:
            self.pipeline = build_pipeline(self.camera)
        capture = cv2.VideoCapture(self.camera.source)
        if not capture.isOpened():
            self.last_error = f"Cannot open source: {self.camera.source}"
            return
        started = perf_counter()
        while not self._stop.is_set():
            ok, frame = capture.read()
            if not ok:
                if isinstance(self.camera.source, str) and not self.camera.source.startswith("rtsp://"):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            result = self.pipeline.process(frame)
            encoded_ok, encoded = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if encoded_ok:
                with self._lock:
                    self._latest_jpeg = encoded.tobytes()
            self.metrics.frames += 1
            self.metrics.fps = self.metrics.frames / max(perf_counter() - started, 1e-6)
            self.metrics.inference_ms = result.inference_ms
            self.metrics.events += len(result.events)
            for event in result.events:
                self.on_event(event)
        capture.release()


class Runtime:
    def __init__(self, cameras: list[CameraDefinition]) -> None:
        self.events: deque[AnalyticsEvent] = deque(maxlen=1000)
        self._event_lock = threading.Lock()
        self.event_sequence = 0
        self.workers = {camera.id: CameraWorker(camera, self.publish) for camera in cameras}

    def publish(self, event: AnalyticsEvent) -> None:
        with self._event_lock:
            self.events.appendleft(event)
            self.event_sequence += 1

    def event_snapshot(self) -> tuple[int, list[AnalyticsEvent]]:
        with self._event_lock:
            return self.event_sequence, list(self.events)

    def start_enabled(self) -> None:
        for worker in self.workers.values():
            if worker.camera.enabled:
                worker.start()

    def stop_all(self) -> None:
        for worker in self.workers.values():
            worker.stop()
