from __future__ import annotations

from visionguard.analytics.engine import AnalyticsEngine
from visionguard.config import CameraDefinition, load_yaml
from visionguard.detection.synthetic import SyntheticDetector
from visionguard.detection.yolo import YoloDetector
from visionguard.domain import LineRule, ZoneRule
from visionguard.pipeline import VideoAnalyticsPipeline
from visionguard.tracking.bytetrack import ByteTrackTracker
from visionguard.tracking.centroid import CentroidTracker


def load_rules(path: str = "configs/analytics.yaml") -> tuple[list[ZoneRule], list[LineRule], float]:
    data = load_yaml(path)
    zones = [
        ZoneRule(
            name=item["name"],
            points=[tuple(point) for point in item["points"]],
            classes=set(item.get("classes", [])),
            dwell_seconds=float(item.get("dwell_seconds", 0)),
        )
        for item in data.get("zones", [])
    ]
    lines = [
        LineRule(
            name=item["name"],
            start=tuple(item["start"]),
            end=tuple(item["end"]),
            classes=set(item.get("classes", [])),
        )
        for item in data.get("lines", [])
    ]
    near_miss_distance = float(data.get("near_miss", {}).get("distance_px", 70))
    return zones, lines, near_miss_distance


def build_pipeline(camera: CameraDefinition, analytics_path: str = "configs/analytics.yaml") -> VideoAnalyticsPipeline:
    zones, lines, near_miss_distance = load_rules(analytics_path)
    if camera.backend == "yolo":
        detector = YoloDetector(model_path=camera.model, confidence=camera.confidence, device=camera.device)
    elif camera.backend == "synthetic":
        detector = SyntheticDetector()
    else:
        raise ValueError(f"Unknown detection backend: {camera.backend}")

    if camera.tracker == "bytetrack":
        tracker = ByteTrackTracker()
    elif camera.tracker == "centroid":
        tracker = CentroidTracker()
    else:
        raise ValueError(f"Unknown tracker: {camera.tracker}")

    analytics = AnalyticsEngine(camera.id, zones, lines, near_miss_distance=near_miss_distance)
    return VideoAnalyticsPipeline(detector, tracker, analytics)
