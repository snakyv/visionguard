from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from visionguard.analytics.engine import AnalyticsEngine
from visionguard.detection.base import Detector
from visionguard.domain import AnalyticsEvent, Track
from visionguard.rendering import render_frame
from visionguard.tracking.base import Tracker


@dataclass(slots=True)
class PipelineResult:
    frame: np.ndarray
    tracks: list[Track]
    events: list[AnalyticsEvent]
    inference_ms: float
    fps: float


class VideoAnalyticsPipeline:
    def __init__(self, detector: Detector, tracker: Tracker, analytics: AnalyticsEngine) -> None:
        self.detector = detector
        self.tracker = tracker
        self.analytics = analytics
        self._fps = 0.0

    def process(self, frame: np.ndarray) -> PipelineResult:
        started = perf_counter()
        detections = self.detector.detect(frame)
        inference_ms = (perf_counter() - started) * 1000.0
        tracks = self.tracker.update(detections)
        events = self.analytics.process(tracks)
        elapsed = max(perf_counter() - started, 1e-6)
        instant = 1.0 / elapsed
        self._fps = instant if self._fps == 0 else self._fps * 0.9 + instant * 0.1
        rendered = render_frame(frame, tracks, self.analytics.zones, self.analytics.lines, self._fps)
        return PipelineResult(rendered, tracks, events, inference_ms, self._fps)
